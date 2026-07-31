// SPDX-License-Identifier: Apache-2.0
#include "agent/runtime.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>

#include "core/text.h"
#include "telemetry.h"
#include "tokenizer.h"

namespace slm {
namespace {

std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::string lower(const std::string& s) {
  std::string t = s;
  for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return t;
}

// Does this question look like it is about the indexed code?  Cheap heuristics
// beat asking the model, because asking costs a full generation and a 30 M model
// answers that meta-question badly.
bool looks_like_code_question(const std::string& q) {
  static const char* kHints[] = {
      "function", "class",   "file",     "code",    "implement", "defined",
      "where is", "method",  "struct",   "header",  "compile",   "build",
      "repo",     "project", "module",   "bug",     "test",      "api",
      "تابع",     "کلاس",    "فایل",     "کد",      "پروژه",     "ماژول",
      "کجاست",    "پیاده",   "باگ",      "تست"};
  const std::string l = lower(q);
  for (const char* h : kHints)
    if (l.find(h) != std::string::npos) return true;
  // A bare identifier-looking token is almost always a symbol lookup.
  if (q.find('_') != std::string::npos || q.find("::") != std::string::npos) return true;
  return false;
}

std::string args_of(const ToolCall& c) {
  std::string s;
  for (const auto& kv : c.args) {
    if (!s.empty()) s += " ";
    s += kv.first + "=" + utf8_truncate(kv.second, 60);
  }
  return s;
}

}  // namespace

const char* ask_mode_name(AskMode m) {
  switch (m) {
    case AskMode::kFast: return "fast (SPT)";
    case AskMode::kStrong: return "strong (GGUF)";
    case AskMode::kDebate: return "debate (both)";
    case AskMode::kSelfDebate: return "self-debate";
  }
  return "?";
}

struct AgentRuntime::Impl {
  AgentRuntimeOptions opt;
  BackendRegistry reg;
  ToolRegistry tools;
  ApprovalGate gate;
  ToolPolicy policy;
  CodebaseIndex index;
  std::unique_ptr<DebateEngine> debate;
  std::shared_ptr<ModelBackendSPT> spt;
  std::shared_ptr<ModelBackendGGUF> strong;
  std::string spt_id = "spt";
  std::string strong_id = "olmo";

  ToolContext make_ctx(std::atomic<bool>* cancel) {
    ToolContext c;
    c.workspace = opt.workspace;
    c.workdir = opt.workdir;
    c.tel = opt.tel;
    c.codebase = &index;
    c.cancel = cancel;
    return c;
  }

  // The reason-act loop.  Returns the final prose.
  std::string tool_loop(IModelBackend* be, const std::string& system,
                        const std::string& question, const AskRequest& req,
                        std::atomic<bool>* cancel, const AskObserver& obs,
                        AskResult* out) {
    std::string convo = question;
    std::string last;
    int nudges = 0;
    for (int step = 0; step <= req.max_tool_steps; ++step) {
      if (cancel && cancel->load()) {
        out->cancelled = true;
        break;
      }
      GenRequest q;
      q.slot = -1;
      q.sp = SamplingParams::critical(req.seed + static_cast<uint64_t>(step));
      q.sp.max_new_tokens = req.max_tokens;
      q.sp.stop = {"<|user|>"};
      std::vector<std::pair<std::string, std::string>> turns;
      turns.emplace_back(convo, std::string());
      q.prompt = be->format_chat(system, turns, true);

      GenResponse res;
      {
        std::lock_guard<std::mutex> hw(be->lock());
        res = be->generate(q, obs.on_text ? [&](const GenChunk& c) {
          obs.on_text(c.text);
          return true;
        } : ChunkFn(), cancel);
      }
      out->prompt_tokens += res.prompt_tokens;
      out->gen_tokens += res.gen_tokens;
      out->reused_tokens += res.reused_tokens;
      if (!res.error.empty()) {
        out->error = res.error;
        break;
      }
      last = trim(res.text);

      const std::vector<ToolCall> calls = ToolRegistry::parse(last);
      if (calls.empty()) {
        if (req.use_tools && nudges == 0 && step < req.max_tool_steps &&
            ToolRegistry::looks_like_broken_call(last)) {
          // One corrective attempt: small models get the syntax wrong far more
          // often than they get the intent wrong, and throwing the intent away
          // is the expensive mistake.
          ++nudges;
          convo += "\n" + last +
                   "\nThat tool call was malformed. Use exactly this form:\n"
                   "[[tool:NAME]]\nkey: value\n[[/tool]]";
          continue;
        }
        break;
      }
      if (!req.use_tools) break;

      std::string feedback;
      for (const ToolCall& c : calls) {
        if (cancel && cancel->load()) break;
        ToolContext tc = make_ctx(cancel);
        const ToolResult r = tools.invoke(c, tc, &gate, policy);
        ToolTrace tr;
        tr.tool = c.name;
        tr.args = args_of(c);
        tr.ok = r.ok;
        tr.denied = r.denied;
        tr.seconds = r.seconds;
        tr.output = utf8_truncate(r.ok ? r.output : r.error, 400);
        out->tools.push_back(tr);
        if (obs.on_tool) obs.on_tool(tr);
        feedback += "[[result:" + c.name + "]]\n" +
                    (r.ok ? r.output : ("error: " + r.error)) + "\n";
      }
      convo += "\n" + last + "\n" + feedback +
               "\nUse the results above to answer. Do not call any more tools "
               "unless something is still missing.";
    }
    return last;
  }
};

AgentRuntime::AgentRuntime() : p_(new Impl) {}
AgentRuntime::~AgentRuntime() = default;

bool AgentRuntime::init(const AgentRuntimeOptions& o, std::string* err) {
  p_->opt = o;
  std::error_code ec;
  std::filesystem::create_directories(o.workdir, ec);

  // ------------------------------------------------------------------ SPT
  SptBackendOptions so;
  so.id = p_->spt_id;
  so.display_name = "SPT";
  so.tok = o.tok;
  so.tokenizer_path = o.tokenizer;
  so.tel = o.tel;
  if (o.spt_live) {
    so.source = SptBackendOptions::kLive;
    so.weights = o.spt_live;
    so.cfg = o.spt_cfg;
    so.display_name = "SPT (live)";
  } else if (!o.spt_quant.empty()) {
    so.source = SptBackendOptions::kQuant;
    so.path = o.spt_quant;
    so.display_name = "SPT (int4)";
  } else if (!o.spt_ckpt.empty()) {
    so.source = SptBackendOptions::kCheckpoint;
    so.path = o.spt_ckpt;
  } else {
    if (err) *err = "no SPT model given (checkpoint, .slmq or live weights)";
    return false;
  }
  p_->spt = std::make_shared<ModelBackendSPT>(so);
  std::string e;
  if (!p_->spt->load(&e)) {
    if (err) *err = "SPT: " + e;
    return false;
  }
  p_->reg.add(p_->spt);

  // ------------------------------------------------------- the strong model
  if (!o.gguf.empty()) {
    GgufBackendOptions go;
    go.path = o.gguf;
    go.id = p_->strong_id;
    go.n_ctx = o.gguf_ctx;
    go.n_threads = o.gguf_threads;
    go.n_gpu_layers = o.gguf_gpu_layers;
    go.kv_q8 = o.gguf_kv_q8;
    go.base_model = o.gguf_base;
    go.tel = o.tel;
    // Named from the file so the GUI shows something meaningful before loading.
    go.display_name = std::filesystem::path(o.gguf).stem().string();
    if (go.display_name.empty()) go.display_name = "GGUF";
    // n_seq_max must cover every debate participant that shares this context.
    go.n_seq_max = 4;
    p_->strong = std::make_shared<ModelBackendGGUF>(go);
    p_->reg.add(p_->strong);
    // Loading 4.5 GB is slow, so it is deferred until a mode needs it - except
    // when the caller clearly wants it up front.
  }

  // ------------------------------------------------------------------ tools
  register_builtin_tools(&p_->tools, o.enable_web, o.enable_shell, o.enable_codebase);
  p_->debate = std::make_unique<DebateEngine>(&p_->reg, o.tel);

  if (o.enable_codebase && !o.index_cache.empty()) {
    std::string ie;
    if (p_->index.load(o.index_cache, &ie) && o.tel)
      o.tel->log("info", "agent", "codebase index loaded",
                 {{"file", o.index_cache},
                  {"chunks", std::to_string(p_->index.stats().chunks)}});
  }
  if (o.enable_codebase && o.index_workspace && p_->index.empty()) {
    std::string ie;
    index_codebase(o.workspace, nullptr, nullptr, &ie);
  }
  return true;
}

const AgentRuntimeOptions& AgentRuntime::options() const { return p_->opt; }
BackendRegistry& AgentRuntime::backends() { return p_->reg; }
ToolRegistry& AgentRuntime::tools() { return p_->tools; }
ApprovalGate& AgentRuntime::gate() { return p_->gate; }
ToolPolicy& AgentRuntime::policy() { return p_->policy; }
CodebaseIndex& AgentRuntime::codebase() { return p_->index; }
DebateEngine& AgentRuntime::debate() { return *p_->debate; }
std::string AgentRuntime::spt_id() const { return p_->spt_id; }
std::string AgentRuntime::strong_id() const { return p_->strong_id; }

bool AgentRuntime::mode_available(AskMode m) const {
  switch (m) {
    case AskMode::kFast:
    case AskMode::kSelfDebate:
      return p_->spt && p_->spt->loaded();
    case AskMode::kStrong:
    case AskMode::kDebate:
      return p_->strong != nullptr;
  }
  return false;
}

bool AgentRuntime::load_strong(std::string* err) {
  if (!p_->strong) {
    if (err) *err = "no GGUF model configured (--gguf F.gguf)";
    return false;
  }
  if (p_->strong->loaded()) return true;
  return p_->strong->load(err);
}

void AgentRuntime::unload_strong() {
  if (p_->strong) p_->strong->unload();
}

bool AgentRuntime::index_codebase(
    const std::string& root,
    const std::function<void(int64_t, int64_t, const std::string&)>& progress,
    std::atomic<bool>* cancel, std::string* err) {
  ScanOptions so;
  so.root = root.empty() ? p_->opt.workspace : root;
  if (!p_->index.scan(so, progress, cancel, err)) return false;
  if (!p_->opt.index_cache.empty()) {
    std::string se;
    p_->index.save(p_->opt.index_cache, &se);
  }
  if (p_->opt.tel) {
    const IndexStats st = p_->index.stats();
    p_->opt.tel->log("info", "agent", "codebase indexed",
                     {{"root", so.root},
                      {"files", std::to_string(st.files)},
                      {"chunks", std::to_string(st.chunks)},
                      {"seconds", std::to_string(st.scan_seconds + st.embed_seconds)}});
  }
  return true;
}

std::vector<std::string> AgentRuntime::status_lines() const {
  std::vector<std::string> out;
  for (const BackendPtr& b : p_->reg.all()) {
    const BackendStatus s = b->status();
    std::ostringstream os;
    os << b->display_name() << " [" << b->runtime() << "] ";
    if (!s.loaded) {
      os << "not loaded";
      if (!s.error.empty()) os << " (" << s.error << ")";
    } else {
      os << (s.params / 1e6) << "M params, " << (s.weight_bytes / (1024.0 * 1024.0))
         << " MiB";
      if (s.kv_bytes) os << " + " << (s.kv_bytes / (1024.0 * 1024.0)) << " MiB KV";
      if (s.last_decode_tps > 0.0) os << ", " << s.last_decode_tps << " tok/s";
      if (s.cache_hit_tokens) os << ", " << s.cache_hit_tokens << " cached tokens";
    }
    out.push_back(os.str());
  }
  if (!p_->index.empty()) {
    const IndexStats st = p_->index.stats();
    std::ostringstream os;
    os << "codebase " << st.root << ": " << st.files << " files, " << st.chunks
       << " chunks, " << st.embedder << ", " << (st.memory_bytes / (1024.0 * 1024.0))
       << " MiB";
    out.push_back(os.str());
  }
  return out;
}

AskResult AgentRuntime::ask(const AskRequest& req, std::atomic<bool>* cancel,
                            const AskObserver& obs) {
  AskResult out;
  const double t0 = Telemetry::now();

  // ------------------------------------------------ automatic retrieval
  std::string context;
  if (req.use_codebase && !p_->index.empty() && looks_like_code_question(req.question)) {
    // A very broad question is better served by the structural digest than by the
    // top few chunks, which would be an arbitrary sample of the repository.
    const std::string l = lower(req.question);
    const bool overview = l.find("what does") != std::string::npos ||
                          l.find("overview") != std::string::npos ||
                          l.find("چیکار") != std::string::npos ||
                          l.find("چه کاری") != std::string::npos;
    context = overview ? p_->index.overview(2000)
                       : p_->index.context_block(req.question, 2400, 8);
    out.context_used = context;
  }

  const std::string system =
      req.system_prompt.empty()
          ? std::string(
                "You are a precise assistant. Answer in the language of the "
                "question. If you are unsure, say so.")
          : req.system_prompt;

  // ------------------------------------------------------------- debates
  if (req.mode == AskMode::kDebate || req.mode == AskMode::kSelfDebate) {
    out.was_debate = true;
    if (req.mode == AskMode::kDebate) {
      std::string le;
      if (!load_strong(&le)) {
        // Degrade rather than refuse: a debate needs two voices, and one model
        // can supply them.
        out.answer.clear();
        AskRequest alt = req;
        alt.mode = AskMode::kSelfDebate;
        AskResult r = ask(alt, cancel, obs);
        r.debate.decision_log =
            "strong model unavailable (" + le + "); fell back to self-debate\n" +
            r.debate.decision_log;
        return r;
      }
    }
    DebateConfig cfg =
        req.mode == AskMode::kDebate
            ? DebateConfig::two_model(p_->spt_id, req.fast_multiplier, p_->strong_id,
                                      req.strong_multiplier)
            : DebateConfig::self_debate(p_->spt_id, req.voices, req.fast_multiplier);
    cfg.system_prompt = system;
    cfg.context = context;
    cfg.max_answer_tokens = req.max_tokens;
    cfg.seed = req.seed ? req.seed : 1234;
    out.debate = p_->debate->run(req.question, cfg, cancel, obs.on_debate);
    out.answer = out.debate.final_answer;
    out.error = out.debate.error;
    out.cancelled = out.debate.cancelled;
    for (const DebateTranscript::Usage& u : out.debate.usage) {
      out.prompt_tokens += u.prompt_tokens;
      out.gen_tokens += u.gen_tokens;
      out.reused_tokens += u.reused_tokens;
    }
    out.seconds = Telemetry::now() - t0;
    return out;
  }

  // ------------------------------------------------------ single model
  IModelBackend* be = nullptr;
  if (req.mode == AskMode::kStrong) {
    std::string le;
    if (!load_strong(&le)) {
      out.error = le;
      out.seconds = Telemetry::now() - t0;
      return out;
    }
    be = p_->strong.get();
  } else {
    be = p_->spt.get();
  }
  if (!be || !be->loaded()) {
    out.error = "backend not available";
    out.seconds = Telemetry::now() - t0;
    return out;
  }

  std::string sys = system;
  if (req.use_tools) {
    const std::string cat = p_->tools.catalogue(true);
    if (!cat.empty())
      sys += "\n\nTools you may call, one per reply, using exactly this form:\n"
             "[[tool:NAME]]\nkey: value\n[[/tool]]\n" +
             cat;
  }
  std::string question = req.question;
  if (!context.empty())
    question = "Reference material:\n" + context + "\nQuestion: " + req.question;

  out.answer = p_->tool_loop(be, sys, question, req, cancel, obs, &out);
  out.seconds = Telemetry::now() - t0;
  return out;
}

}  // namespace slm
