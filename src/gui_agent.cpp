// SPDX-License-Identifier: Apache-2.0
#include "gui_agent.h"

#include <fstream>
#include <mutex>
#include <thread>

#include "core/text.h"
#include "interaction.h"
#include "telemetry.h"

namespace slm {

struct AgentController::Impl {
  AgentRuntime* rt = nullptr;
  InteractionHub* hub = nullptr;
  Telemetry* tel = nullptr;

  mutable std::mutex m;
  AgentSnapshot snap;
  std::atomic<bool> busy{false};
  std::atomic<bool> cancel{false};
  std::thread ask_th;

  IndexJob index;
  std::thread index_th;
  std::atomic<bool> index_cancel{false};

  TrainJob train;
  std::thread train_th;
  std::atomic<bool> train_cancel{false};

  void join_all() {
    cancel.store(true);
    index_cancel.store(true);
    train_cancel.store(true);
    if (ask_th.joinable()) ask_th.join();
    if (index_th.joinable()) index_th.join();
    if (train_th.joinable()) train_th.join();
  }
};

AgentController::AgentController() : p_(new Impl) {}
AgentController::~AgentController() { p_->join_all(); }

void AgentController::attach(AgentRuntime* rt, InteractionHub* hub, Telemetry* tel) {
  p_->rt = rt;
  p_->hub = hub;
  p_->tel = tel;
  std::lock_guard<std::mutex> g(p_->m);
  if (rt) p_->snap.status = rt->status_lines();
}

bool AgentController::attached() const { return p_->rt != nullptr; }
AgentRuntime* AgentController::runtime() const { return p_->rt; }
bool AgentController::busy() const { return p_->busy.load(); }

AgentSnapshot AgentController::snapshot() const {
  std::lock_guard<std::mutex> g(p_->m);
  AgentSnapshot s = p_->snap;
  s.busy = p_->busy.load();
  if (p_->rt) s.pending_approvals = p_->rt->gate().pending_count();
  return s;
}

void AgentController::clear_history() {
  std::lock_guard<std::mutex> g(p_->m);
  p_->snap.history.clear();
}

void AgentController::cancel() {
  p_->cancel.store(true);
  // A pending approval would otherwise keep the worker parked until it times out.
  if (p_->rt) p_->rt->gate().deny_all();
}

void AgentController::ask(const std::string& question, AskMode mode, int fast_mult,
                          int strong_mult, int voices, bool use_tools,
                          bool use_codebase, int max_tokens) {
  if (!p_->rt || question.empty() || p_->busy.load()) return;
  if (p_->ask_th.joinable()) p_->ask_th.join();
  p_->cancel.store(false);
  p_->busy.store(true);
  {
    std::lock_guard<std::mutex> g(p_->m);
    p_->snap.question = question;
    p_->snap.partial.clear();
    p_->snap.progress = 0.0;
    p_->snap.live = DebateTranscript();
  }
  AskRequest req;
  req.question = question;
  req.mode = mode;
  req.fast_multiplier = fast_mult;
  req.strong_multiplier = strong_mult;
  req.voices = voices;
  req.use_tools = use_tools;
  req.use_codebase = use_codebase;
  req.max_tokens = max_tokens;

  p_->ask_th = std::thread([this, req] {
    AskObserver obs;
    obs.on_text = [this](const std::string& piece) {
      std::lock_guard<std::mutex> g(p_->m);
      p_->snap.partial += piece;
    };
    obs.on_debate = [this](const DebateTranscript& tr) {
      std::lock_guard<std::mutex> g(p_->m);
      p_->snap.live = tr;
      p_->snap.progress = tr.progress;
    };
    obs.on_tool = [this](const ToolTrace& t) {
      std::lock_guard<std::mutex> g(p_->m);
      p_->snap.partial += "\n[tool " + t.tool + ": " +
                          (t.denied ? "denied" : (t.ok ? "ok" : "failed")) + "]\n";
    };
    const AskResult res = p_->rt->ask(req, &p_->cancel, obs);
    AgentTurn turn;
    turn.question = req.question;
    turn.answer = res.answer;
    turn.mode = req.mode;
    turn.debate = res.debate;
    turn.tools = res.tools;
    turn.context_used = res.context_used;
    turn.seconds = res.seconds;
    turn.prompt_tokens = res.prompt_tokens;
    turn.gen_tokens = res.gen_tokens;
    turn.reused_tokens = res.reused_tokens;
    turn.was_debate = res.was_debate;
    turn.error = res.error;
    {
      std::lock_guard<std::mutex> g(p_->m);
      p_->snap.history.push_back(std::move(turn));
      while (p_->snap.history.size() > 40) p_->snap.history.erase(p_->snap.history.begin());
      p_->snap.partial.clear();
      p_->snap.question.clear();
      p_->snap.progress = res.was_debate ? 1.0 : 0.0;
      p_->snap.status = p_->rt->status_lines();
    }
    if (p_->tel)
      p_->tel->log(res.error.empty() ? "info" : "warn", "agent",
                   std::string("answered in ") + ask_mode_name(req.mode),
                   {{"seconds", std::to_string(res.seconds)},
                    {"gen_tokens", std::to_string(res.gen_tokens)},
                    {"reused", std::to_string(res.reused_tokens)},
                    {"error", res.error}});
    p_->busy.store(false);
  });
}

// ------------------------------------------------------------------- codebase
void AgentController::start_index(const std::string& root) {
  if (!p_->rt) return;
  {
    std::lock_guard<std::mutex> g(p_->m);
    if (p_->index.running) return;
    p_->index = IndexJob();
    p_->index.running = true;
    p_->index.root = root;
  }
  if (p_->index_th.joinable()) p_->index_th.join();
  p_->index_cancel.store(false);
  p_->index_th = std::thread([this, root] {
    const double t0 = Telemetry::now();
    std::string err;
    const bool ok = p_->rt->index_codebase(
        root,
        [this](int64_t done, int64_t total, const std::string& path) {
          std::lock_guard<std::mutex> g(p_->m);
          p_->index.files_done = done;
          p_->index.files_total = total;
          p_->index.current = path;
        },
        &p_->index_cancel, &err);
    std::lock_guard<std::mutex> g(p_->m);
    p_->index.running = false;
    p_->index.done = ok;
    p_->index.error = ok ? std::string() : err;
    p_->index.seconds = Telemetry::now() - t0;
    p_->snap.status = p_->rt->status_lines();
  });
}

IndexJob AgentController::index_job() const {
  std::lock_guard<std::mutex> g(p_->m);
  return p_->index;
}

// -------------------------------------------------------------------- dataset
void AgentController::teach_dataset(const std::string& path, int64_t max_lines,
                                    int64_t chunk_chars) {
  if (!p_->hub || path.empty()) return;
  {
    std::lock_guard<std::mutex> g(p_->m);
    if (p_->train.running) return;
    p_->train = TrainJob();
    p_->train.running = true;
    p_->train.path = path;
  }
  if (p_->train_th.joinable()) p_->train_th.join();
  p_->train_cancel.store(false);
  p_->train_th = std::thread([this, path, max_lines, chunk_chars] {
    const double t0 = Telemetry::now();
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      std::lock_guard<std::mutex> g(p_->m);
      p_->train.running = false;
      p_->train.error = "cannot open " + path;
      p_->train.seconds = Telemetry::now() - t0;
      return;
    }
    // Paragraph-sized samples: the continual thread tokenises whatever it is
    // given, and one line of a corpus is usually too short to carry a gradient
    // worth taking, while a whole file is longer than the context window.
    std::string block;
    int64_t lines = 0, queued = 0, bytes = 0;
    std::string line;
    auto flush = [&] {
      if (block.empty()) return;
      p_->hub->push_text(utf8_sanitize(block));
      ++queued;
      bytes += static_cast<int64_t>(block.size());
      block.clear();
      std::lock_guard<std::mutex> g(p_->m);
      p_->train.lines = lines;
      p_->train.queued = queued;
      p_->train.bytes = bytes;
    };
    while (std::getline(f, line)) {
      if (p_->train_cancel.load()) break;
      ++lines;
      if (max_lines > 0 && lines > max_lines) break;
      if (line.empty() && !block.empty()) {
        flush();
        continue;
      }
      block += line;
      block.push_back('\n');
      if (static_cast<int64_t>(block.size()) >= chunk_chars) flush();
      // Back-pressure: the hub keeps at most a few thousand items, so filling it
      // faster than the trainer drains it would just discard the tail.
      while (p_->hub->pending_texts() > 2000 && !p_->train_cancel.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    flush();
    if (p_->tel)
      p_->tel->log("info", "train",
                   "queued a dataset for continual learning",
                   {{"file", path},
                    {"samples", std::to_string(queued)},
                    {"bytes", std::to_string(bytes)}});
    std::lock_guard<std::mutex> g(p_->m);
    p_->train.running = false;
    p_->train.done = true;
    p_->train.seconds = Telemetry::now() - t0;
  });
}

TrainJob AgentController::train_job() const {
  std::lock_guard<std::mutex> g(p_->m);
  return p_->train;
}

void AgentController::stop_jobs() {
  p_->index_cancel.store(true);
  p_->train_cancel.store(true);
}

}  // namespace slm
