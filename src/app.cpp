// SPDX-License-Identifier: Apache-2.0
//
// Wiring of the live system:
//
//        InteractionHub                       Coordinator (single writer)
//   user ---> texts ---> [ContinualTrainer] ------\
//                        [SelfGenTrainer]  --------+--> merge -> gate -> publish
//        ratings ------> [FeedbackTrainer] -------/                |
//                                                                  v
//                        ChatEngine  <---- published snapshot (RCU) |
//                             |                                     |
//                          Telemetry <------ audit log, losses, stats
//                             |
//                          Dashboard (ImGui or terminal)
#include <algorithm>

#include "core/text.h"
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <thread>

#include "app.h"
#include "agent/runtime.h"
#include "gui_agent.h"
#include "spt_assets.h"
#include "chat.h"
#include "coordinator.h"
#include "core/dataset.h"
#include "core/serialize.h"
#include "gui.h"
#include "interaction.h"
#include "memory.h"
#include "telemetry.h"
#include "tokenizer.h"
#include "train_continual.h"
#include "train_feedback.h"
#include "train_selfgen.h"

namespace slm {
namespace {

std::vector<int32_t> span_tokens(const TokenStore& t, int64_t a, int64_t b) {
  std::vector<int32_t> out;
  for (int64_t i = a; i < b; ++i) out.push_back(t.at(static_cast<size_t>(i)));
  return out;
}

TrainerConfig trainer_defaults(const Config& c, const char* prefix, float lr,
                               int64_t local_steps, double interval, int replay,
                               int last_k) {
  TrainerConfig t;
  const std::string p = prefix;
  t.lr = static_cast<float>(c.get_num(p + ".lr", lr));
  t.weight_decay = static_cast<float>(c.get_num(p + ".weight_decay", 0.0));
  t.grad_clip = static_cast<float>(c.get_num(p + ".grad_clip", 0.5));
  t.local_steps = c.get_int(p + ".local_steps", local_steps);
  t.batch = c.get_int(p + ".batch", 2);
  t.ctx = c.get_int(p + ".ctx", 128);
  t.min_interval_s = c.get_num(p + ".min_interval_s", interval);
  t.replay_percent = static_cast<int>(c.get_int(p + ".replay_percent", replay));
  const int64_t k = c.get_int(p + ".last_k_blocks", last_k);
  t.freeze = FreezePolicy::last_k(static_cast<int>(k),
                                  c.get_bool(p + ".train_head", true));
  t.freeze.train_final_ln = c.get_bool(p + ".train_final_ln", true);
  t.freeze.train_embeddings = c.get_bool(p + ".train_embeddings", false);
  if (c.get_bool(p + ".all_blocks", false)) t.freeze.last_k_blocks = -1;
  return t;
}

// A tiny automatic user: it asks questions taken from the corpus (so the
// expected answer is known) and rates the reply by token overlap.  Used for
// soak tests and for the --autopilot demo, never required by the system.
class Autopilot {
 public:
  Autopilot(const Tokenizer* tok, const MixtureDataset* ds, ChatEngine* chat,
            InteractionHub* hub, Telemetry* tel, uint64_t seed, double period)
      : tok_(tok), chat_(chat), hub_(hub), tel_(tel), rng_(seed), period_(period) {
    if (!ds) return;
    // Mine question/answer pairs from *every* source, so the synthetic user
    // rates Persian, English and Python in turn instead of biasing one language.
    for (int si = 0; si < ds->num_sources(); ++si) {
      const TokenStore& t = ds->data(si).tokens();
      const int64_t limit = ds->data(si).train_tokens();
      size_t added = 0;
      for (int64_t i = 0; i < limit && added < 256; ++i) {
        if (t.at(static_cast<size_t>(i)) != Tokenizer::kUser) continue;
        int64_t a = i + 1;
        while (a < limit && t.at(static_cast<size_t>(a)) != Tokenizer::kAssistant && a - i < 48) ++a;
        if (a >= limit || t.at(static_cast<size_t>(a)) != Tokenizer::kAssistant) continue;
        int64_t e = a + 1;
        while (e < limit && t.at(static_cast<size_t>(e)) != Tokenizer::kEot && e - a < 96) ++e;
        if (e >= limit) break;
        QA qa;
        qa.lang = ds->info(si).lang;
        qa.question = tok_->decode(span_tokens(t, i + 1, a));
        qa.answer = tok_->decode(span_tokens(t, a + 1, e));
        if (!qa.question.empty() && !qa.answer.empty()) {
          qa_.push_back(std::move(qa));
          ++added;
        }
        i = e;
      }
      lang_offsets_.push_back(qa_.size());
    }
  }

  bool available() const { return !qa_.empty(); }

  void start() {
    if (!available() || run_.exchange(true)) return;
    th_ = std::thread([this] { loop(); });
  }
  void stop() {
    if (!run_.exchange(false)) return;
    if (th_.joinable()) th_.join();
  }
  ~Autopilot() { stop(); }

 private:
  struct QA {
    std::string question, answer;
    Lang lang = Lang::kUnknown;
  };

  static float overlap_score(const std::string& want, const std::string& got) {
    // crude bag-of-words F1 -> 1..5 stars
    auto split = [](const std::string& s) {
      std::vector<std::string> w;
      std::string cur;
      for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)))
          cur += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        else if (!cur.empty()) {
          w.push_back(cur);
          cur.clear();
        }
      }
      if (!cur.empty()) w.push_back(cur);
      return w;
    };
    const std::vector<std::string> a = split(want), b = split(got);
    if (a.empty() || b.empty()) return 1.0f;
    size_t hit = 0;
    for (const std::string& w : a)
      if (std::find(b.begin(), b.end(), w) != b.end()) ++hit;
    const float recall = static_cast<float>(hit) / static_cast<float>(a.size());
    return 1.0f + 4.0f * std::min(1.0f, recall);
  }

  void loop() {
    size_t asked = 0;
    while (run_.load()) {
      if (tel_ && tel_->stopped()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        continue;
      }
      // Ask a fresh question most of the time, but deliberately repeat an
      // earlier one now and then: two different answers to the *same* prompt
      // with different ratings is what turns the feedback thread into DPO.
      // Walk the languages in turn (so the feedback stream stays balanced),
      // and now and then repeat an earlier question to create a DPO pair.
      size_t pick = 0;
      if (lang_offsets_.empty()) {
        pick = static_cast<size_t>(rng_.below(qa_.size()));
      } else {
        const size_t g = lang_cursor_++ % lang_offsets_.size();
        const size_t begin = g ? lang_offsets_[g - 1] : 0;
        const size_t end = lang_offsets_[g];
        pick = (end > begin) ? begin + static_cast<size_t>(rng_.below(end - begin))
                             : static_cast<size_t>(rng_.below(qa_.size()));
      }
      if (!recent_.empty() && rng_.uniform() < 0.45f)
        pick = recent_[static_cast<size_t>(rng_.below(recent_.size()))];
      recent_.push_back(pick);
      if (recent_.size() > 8) recent_.erase(recent_.begin());
      const QA& qa = qa_[pick];
      const size_t before = chat_->history().size();
      chat_->ask(qa.question);
      // wait for the answer
      for (int i = 0; i < 600 && run_.load(); ++i) {
        if (chat_->history().size() > before) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      std::vector<ChatTurn> h = chat_->history();
      if (h.size() > before) {
        const float score = overlap_score(qa.answer, h.back().response);
        chat_->rate(h.size() - 1, score);
        if (tel_)
          tel_->log("info", "autopilot", "asked and rated",
                    {{"lang", lang_code(qa.lang)},
                     {"q", utf8_truncate(qa.question, 60)},
                     {"expected", utf8_truncate(qa.answer, 60)},
                     {"got", utf8_truncate(h.back().response, 60)},
                     {"score", std::to_string(score)}});
      }
      ++asked;
      const double wait = period_;
      for (double t = 0; t < wait && run_.load(); t += 0.1)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  const Tokenizer* tok_;
  ChatEngine* chat_;
  InteractionHub* hub_;
  Telemetry* tel_;
  Rng rng_;
  double period_;
  std::vector<QA> qa_;
  std::vector<size_t> lang_offsets_;
  std::vector<size_t> recent_;
  size_t lang_cursor_ = 0;
  std::thread th_;
  std::atomic<bool> run_{false};
};

}  // namespace

int run_self_training(const AppOptions& opt) {
  backend_init(opt.threads, opt.cuda);

  // Zero-argument operation: when the caller did not name a model, find the one
  // SPT model (or create a starter one).  Failing with "cannot load tokenizer
  // 'run/tok.slmtok'" was the single most common way this program refused to
  // start, and no user should have to know that path.
  AppOptions o = opt;
  if (o.ckpt.empty() || o.tokenizer.empty()) {
    SptAssets a;
    std::string err;
    if (!resolve_spt(&a, true, &err,
                     [](const std::string& m) { std::printf("  %s\n", m.c_str()); })) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
    if (o.ckpt.empty()) o.ckpt = a.ckpt;
    if (o.tokenizer.empty()) o.tokenizer = a.tokenizer;
    if (o.workdir.empty() || o.workdir == "runs") o.workdir = a.workdir;
    if (!a.note.empty()) std::printf("  %s\n", a.note.c_str());
  }
  const AppOptions& opt_ref = o;

  Tokenizer tok;
  if (!tok.load(opt_ref.tokenizer)) {
    std::fprintf(stderr, "cannot load tokenizer '%s'\n", opt_ref.tokenizer.c_str());
    return 1;
  }
  ParamStore ps;
  CheckpointMeta meta;
  if (!load_checkpoint(opt_ref.ckpt, &ps, &meta)) {
    std::fprintf(stderr, "cannot load checkpoint '%s'\n", opt_ref.ckpt.c_str());
    return 1;
  }
  GPTConfig mcfg = GPTConfig::from_config(meta.extra);
  mcfg.vocab_size = tok.vocab_size();
  mcfg.grad_checkpointing = opt_ref.cfg.get_bool("model.grad_checkpointing", true);

  MixtureDataset ds;
  if (opt_ref.data.empty()) {
    std::fprintf(stderr,
                 "warning: no corpus given (--data F or --mix fa=F:0.4,en=F:0.3,py=F:0.3):\n"
                 "         replay, per-language hold-out gating and the autopilot are disabled\n");
  }
  if (!opt_ref.data.empty()) {
    std::string err;
    if (!ds.add_spec(opt_ref.data, tok, &err))
      std::fprintf(stderr,
                   "warning: corpus not loaded (%s); replay and hold-out gating are disabled\n",
                   err.c_str());
    else
      std::printf("corpus mixture:\n%s", ds.describe().c_str());
  }

  std::error_code ec;
  std::filesystem::create_directories(opt_ref.workdir, ec);

  Telemetry tel;
  tel.open_audit(opt_ref.workdir + "/audit.jsonl");
  // Off unless asked for, either in the config or with --train.
  tel.set_self_training_enabled(opt_ref.cfg.get_bool("train.enabled", false));
  tel.log("info", "app", "session start",
          {{"backend", backend_name()},
           {"model", mcfg.describe()},
           {"checkpoint", opt_ref.ckpt},
           {"corpus_tokens", std::to_string(ds.total_tokens())},
           {"sources", std::to_string(ds.num_sources())}});

  // ------------------------------------------------- trainer configurations
  const Config& c = opt_ref.cfg;
  TrainerConfig cl = trainer_defaults(c, "continual", 2e-5f, 2, 6.0, 50, 2);
  TrainerConfig sg = trainer_defaults(c, "selfgen", 1e-5f, 1, 12.0, 60, 2);
  TrainerConfig fb = trainer_defaults(c, "feedback", 3e-5f, 2, 8.0, 40, 2);
  cl.note = "continual";
  sg.note = "selfgen";
  fb.note = "feedback";

  // The merge space is the union of every trainer's trainable parameters.
  std::vector<std::string> merge_space;
  {
    GPT probe(mcfg);
    std::vector<std::string> all = probe.param_names();
    std::vector<std::string> uni;
    for (const TrainerConfig* t : {&cl, &sg, &fb}) {
      probe.set_freeze_policy(t->freeze);
      for (const std::string& n : probe.trainable_names())
        if (std::find(uni.begin(), uni.end(), n) == uni.end()) uni.push_back(n);
    }
    // keep the canonical model order
    for (const std::string& n : all)
      if (std::find(uni.begin(), uni.end(), n) != uni.end()) merge_space.push_back(n);
  }

  CoordinatorConfig ccfg = CoordinatorConfig::from_config(c);
  auto initial = std::make_shared<ParamStore>(ps);
  Coordinator coord(mcfg, initial, merge_space, ds.empty() ? nullptr : &ds, &tel, ccfg,
                    opt_ref.workdir);

  InteractionHub hub;
  GenOptions go;
  go.max_new_tokens = static_cast<int>(c.get_int("chat.max_new_tokens", 64));
  go.temperature = static_cast<float>(c.get_num("chat.temperature", 0.85));
  go.top_k = static_cast<int>(c.get_int("chat.top_k", 40));
  go.top_p = static_cast<float>(c.get_num("chat.top_p", 0.95));
  ChatEngine chat(mcfg, &coord, &tel, &hub, &tok, go);
  MemoryStore memory;
  {
    const std::string mem_path =
        c.get_str("memory.file", opt_ref.workdir + "/memory.jsonl");
    if (memory.open(mem_path)) {
      chat.set_memory(&memory, static_cast<int>(c.get_int("memory.top_k", 3)));
      tel.log("info", "memory", "long term memory ready",
              {{"file", mem_path}, {"entries", std::to_string(memory.size())}});
    }
  }

  ContinualTrainer t_cl(&coord, &tel, &hub, mcfg, cl, ContinualConfig::from_config(c),
                        &tok, ds.empty() ? nullptr : &ds, opt_ref.seed + 11);
  SelfGenTrainer t_sg(&coord, &tel, mcfg, sg, SelfGenConfig::from_config(c), &tok,
                      ds.empty() ? nullptr : &ds, opt_ref.seed + 22);
  FeedbackTrainer t_fb(&coord, &tel, &hub, mcfg, fb, FeedbackConfig::from_config(c),
                       &tok, ds.empty() ? nullptr : &ds, opt_ref.seed + 33);

  // ------------------------------------------------------- the agent runtime
  // The live SPT backend reads the coordinator's published weights, so the model
  // the agent answers with is the model the three threads are improving.  The
  // second model is optional: without --gguf there is one model and the debate
  // modes fall back to self-debate.
  AgentRuntime agent;
  AgentController actrl;
  bool agent_ok = false;
  {
    AgentRuntimeOptions ao;
    ao.spt_live = [&coord](uint64_t* v) { return coord.snapshot(v); };
    ao.spt_cfg = mcfg;
    ao.tok = &tok;
    ao.tokenizer = opt_ref.tokenizer;
    ao.workspace = opt_ref.workspace;
    ao.workdir = opt_ref.workdir;
    ao.index_cache = opt_ref.workdir + "/codebase.idx";
    ao.gguf = opt_ref.gguf.empty() ? find_gguf() : opt_ref.gguf;
    if (!ao.gguf.empty() && opt_ref.gguf.empty())
      std::printf("  ready-made model: %s\n", ao.gguf.c_str());
    ao.gguf_ctx = static_cast<int>(c.get_int("gguf.ctx", 4096));
    ao.gguf_threads = static_cast<int>(c.get_int("gguf.threads", 0));
    ao.gguf_gpu_layers = static_cast<int>(c.get_int("gguf.gpu_layers", 0));
    ao.gguf_kv_q8 = c.get_bool("gguf.kv_q8", false);
    ao.tel = &tel;
    std::string aerr;
    agent_ok = agent.init(ao, &aerr);
    if (!agent_ok)
      tel.log("warn", "agent", "agent runtime unavailable: " + aerr);
    else {
      actrl.attach(&agent, &hub, &tel);
      tel.log("info", "agent", "agent ready",
              {{"backends", std::to_string(agent.backends().size())},
               {"gguf", opt_ref.gguf.empty() ? "none" : opt_ref.gguf},
               {"workspace", opt_ref.workspace}});
    }
  }

  std::atomic<bool> quit{false};
  DashboardContext ctx;
  ctx.tel = &tel;
  ctx.coord = &coord;
  ctx.hub = &hub;
  ctx.chat = &chat;
  ctx.tok = &tok;
  ctx.trainers = {&t_cl, &t_sg, &t_fb};
  ctx.opt = &opt_ref;
  ctx.mcfg = &mcfg;
  ctx.quit = &quit;
  ctx.memory = &memory;
  ctx.agent = agent_ok ? &agent : nullptr;
  ctx.actrl = agent_ok ? &actrl : nullptr;
  ctx.corpus = &ds;

  coord.start();
  chat.start();
  t_cl.start();
  t_sg.start();
  t_fb.start();

  Autopilot autopilot(&tok, ds.empty() ? nullptr : &ds, &chat, &hub, &tel, opt_ref.seed + 44,
                      c.get_num("autopilot.period_s", 6.0));
  if (opt_ref.autopilot) {
    if (autopilot.available()) {
      tel.log("info", "autopilot", "enabled: synthesising user turns and ratings");
      autopilot.start();
    } else {
      tel.log("warn", "autopilot", "no <|user|>/<|assistant|> pairs in the corpus");
    }
  }

  int rc = 0;
  if (opt_ref.headless)
    rc = run_terminal_dashboard(ctx);
  else if (gui_available())
    rc = run_imgui_dashboard(ctx);
  else {
    std::printf("this binary was built without the ImGui dashboard, using the terminal one\n");
    rc = run_terminal_dashboard(ctx);
  }

  autopilot.stop();
  t_cl.stop();
  t_sg.stop();
  t_fb.stop();
  chat.stop();
  coord.stop();

  // final state
  ParamStorePtr final_ps = coord.snapshot();
  if (final_ps) {
    CheckpointMeta m;
    m.step = coord.stats().rounds;
    mcfg.write_to(m.extra);
    m.extra.set("tokenizer", opt_ref.tokenizer);
    m.extra.set("coord.holdout_loss", std::to_string(coord.baseline().loss));
    m.extra.set("origin", "self-training session");
    const std::string out = opt_ref.workdir + "/final.slm";
    if (save_checkpoint(out, *final_ps, m, Dtype::F16))
      std::printf("\nsaved %s (holdout loss %.4f after %lld coordinator rounds)\n",
                  out.c_str(), coord.baseline().loss,
                  static_cast<long long>(coord.stats().rounds));
  }
  tel.log("info", "app", "session end");
  return rc;
}

}  // namespace slm
