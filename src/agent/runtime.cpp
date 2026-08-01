// SPDX-License-Identifier: Apache-2.0
#include "agent/runtime.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>

#include "agent/http.h"
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

// Substring matching on short words is wrong here: "latest" contains "test", so
// "what is the latest release of CMake?" was being treated as a question about
// the code and answered from the repository index.  ASCII needles therefore have
// to land on word boundaries.  Persian needles are matched as substrings, because
// Persian is agglutinative enough that requiring boundaries loses real matches
// ("فایلها", "کدنویسی"), and its words are long enough that accidental hits
// are not a practical problem.
bool contains_word(const std::string& hay, const std::string& needle) {
  bool ascii = true;
  for (char c : needle)
    if (static_cast<unsigned char>(c) > 0x7F) ascii = false;
  if (!ascii) return hay.find(needle) != std::string::npos;
  auto wordish = [](char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return std::isalnum(u) != 0 || c == '_' || u > 0x7F;
  };
  size_t at = 0;
  while ((at = hay.find(needle, at)) != std::string::npos) {
    const bool left = at == 0 || !wordish(hay[at - 1]);
    const size_t end = at + needle.size();
    const bool right = end >= hay.size() || !wordish(hay[end]);
    if (left && right) return true;
    at += 1;
  }
  return false;
}

// Does this question look like it is about the indexed code?  Cheap heuristics
// beat asking the model, because asking costs a full generation and a 30 M model
// answers that meta-question badly.
bool looks_like_code_question(const std::string& q) {
  static const char* kHints[] = {
      "function", "class",   "file",     "files",   "code",      "defined",
      "where is", "method",  "struct",   "header",  "compile",   "build",
      "repo",     "project", "module",   "bug",     "test",      "tests",
      "api",      "implement", "implemented", "implementation",
      "تابع",     "کلاس",    "فایل",     "کد",      "پروژه",     "ماژول",
      "کجاست",    "پیاده",   "باگ",      "تست"};
  const std::string l = lower(q);
  for (const char* h : kHints)
    if (contains_word(l, h)) return true;
  // A bare identifier-looking token is almost always a symbol lookup.
  if (q.find('_') != std::string::npos || q.find("::") != std::string::npos) return true;
  return false;
}

// GGUF files are named after the upload, not after the model: the file we ship
// is "allenai_Olmo-3-7B-Instruct-Q4_K_M.gguf", which is both too long for a
// sidebar and not what anyone calls it.  Recover the short name.
std::string friendly_gguf_name(const std::string& path) {
  const std::string stem = std::filesystem::path(path).stem().string();
  if (stem.empty()) return "GGUF";
  const std::string l = lower(stem);
  if (l.find("olmo") != std::string::npos) return "OLMo";
  // Anything else: drop the uploader prefix and the quantisation suffix, which
  // is where almost all of the length lives.
  std::string s = stem;
  const size_t us = s.find('_');
  if (us != std::string::npos && us + 1 < s.size() && s.find('-') > us) s = s.substr(us + 1);
  for (const char* q : {"-Q4", "-Q5", "-Q6", "-Q8", "-q4", "-q5", "-q6", "-q8", "-F16"}) {
    const size_t at = s.find(q);
    if (at != std::string::npos) s = s.substr(0, at);
  }
  return s.empty() ? stem : s;
}

std::string args_of(const ToolCall& c) {
  std::string s;
  for (const auto& kv : c.args) {
    if (!s.empty()) s += " ";
    s += kv.first + "=" + utf8_truncate(kv.second, 60);
  }
  return s;
}


bool has_any(const std::string& hay, const std::vector<std::string>& needles) {
  for (const std::string& n : needles)
    if (contains_word(hay, n)) return true;
  return false;
}

// Every URL in a search result, best first.
//
// Which page gets read matters more than which results get listed: only one page
// fits in the context, so picking the wrong one wastes the whole retrieval.  The
// ranking is deliberately crude but it encodes the two things that actually
// predict a useful fetch - whether the host is the subject's own site, and
// whether the host is one that blocks or buries the answer.
std::vector<std::string> rank_urls(const std::string& search_output,
                                   const std::string& query) {
  std::vector<std::string> urls;
  size_t at = 0;
  while ((at = search_output.find("http", at)) != std::string::npos) {
    size_t e = at;
    while (e < search_output.size() &&
           !std::isspace(static_cast<unsigned char>(search_output[e])))
      ++e;
    while (e > at && (search_output[e - 1] == '.' || search_output[e - 1] == ',' ||
                      search_output[e - 1] == ')'))
      --e;
    const std::string u = search_output.substr(at, e - at);
    if (u.size() > 12 && std::find(urls.begin(), urls.end(), u) == urls.end())
      urls.push_back(u);
    at = e;
  }

  // Words from the question, long enough to be distinctive.
  std::vector<std::string> words;
  {
    std::string cur;
    for (char c : lower(query)) {
      if (std::isalnum(static_cast<unsigned char>(c))) {
        cur += c;
      } else {
        if (cur.size() >= 4) words.push_back(cur);
        cur.clear();
      }
    }
    if (cur.size() >= 4) words.push_back(cur);
  }
  static const char* kNoisy[] = {"stackoverflow.", "reddit.",   "quora.",
                                 "pinterest.",     "facebook.", "twitter.",
                                 "x.com",          "medium.",   "linkedin.",
                                 "youtube.",       "releasealert.", "chocolatey."};
  auto score = [&](const std::string& u) {
    const std::string host = lower(HttpClient::host_of(u));
    int s = 0;
    for (const char* n : kNoisy)
      if (host.find(n) != std::string::npos) s -= 6;
    for (const std::string& w : words)
      if (host.find(w) != std::string::npos) s += 5;   // the subject's own site
    if (host.find("wikipedia.") != std::string::npos) s += 2;
    if (u.find("/docs") != std::string::npos || u.find("/download") != std::string::npos ||
        u.find("/releases") != std::string::npos)
      s += 2;
    return s;
  };
  std::stable_sort(urls.begin(), urls.end(),
                   [&](const std::string& a, const std::string& b) {
                     return score(a) > score(b);
                   });
  return urls;
}

// A bare URL anywhere in the text.
std::string first_url(const std::string& q) {
  static const char* kSchemes[] = {"https://", "http://"};
  for (const char* sc : kSchemes) {
    const size_t at = q.find(sc);
    if (at == std::string::npos) continue;
    size_t end = at;
    while (end < q.size() && !std::isspace(static_cast<unsigned char>(q[end])) &&
           q[end] != '"' && q[end] != '\'' && q[end] != ',' && q[end] != ')')
      ++end;
    // Trailing sentence punctuation is not part of the URL.
    while (end > at && (q[end - 1] == '.' || q[end - 1] == '?' || q[end - 1] == '!'))
      --end;
    return q.substr(at, end - at);
  }
  return {};
}

}  // namespace

ToolPlan plan_tools(const std::string& q, bool have_index, bool web_ok) {
  ToolPlan p;
  const std::string l = lower(q);

  p.url = first_url(q);
  if (!p.url.empty() && web_ok) {
    p.fetch = true;
    p.reason = "the question contains a link";
  }

  // Explicit instruction, in either language.
  static const std::vector<std::string> kAskedForWeb = {
      "search the web", "look it up", "google", "on the web", "search online",
      "جست‌وجو", "جستجو", "سرچ", "در وب", "اینترنت", "گوگل"};
  // Things the weights cannot know: anything current, versioned or priced.
  static const std::vector<std::string> kNeedsFresh = {
      "latest", "newest", "current version", "release", "today", "this year",
      "2024", "2025", "2026", "price", "news", "who is the", "when did",
      "how much does",
      "آخرین", "جدیدترین", "نسخه", "امروز", "اخبار", "قیمت", "چند سال", "الان"};
  if (web_ok && !p.fetch) {
    if (has_any(l, kAskedForWeb)) {
      p.search = true;
      p.reason = "you asked for a web search";
    } else if (has_any(l, kNeedsFresh)) {
      p.search = true;
      p.reason = "the answer depends on current information";
    }
  }

  if (have_index && looks_like_code_question(q)) {
    p.codebase = true;
    if (p.reason.empty()) p.reason = "the question is about the indexed code";
  }
  return p;
}

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
      // Tool output grows the conversation every round, and handing a prompt
      // longer than the context to llama.cpp returns "llama_decode failed (1)".
      // Trim from the *front* of the accumulated tool results: the question and
      // the latest results matter more than the first round's output.
      const int64_t limit = be->context_limit();
      const int64_t room = limit - req.max_tokens - 64;
      if (room > 128 && be->count_tokens(convo) > room) {
        // Binary search on bytes: count_tokens is not free, and being within a
        // few tokens of the budget is enough.
        size_t lo = 0, hi = convo.size();
        while (lo + 256 < hi) {
          const size_t mid = (lo + hi) / 2;
          if (be->count_tokens(convo.substr(convo.size() - mid)) > room) hi = mid;
          else lo = mid;
        }
        convo = "[earlier tool output trimmed to fit the context]\n" +
                utf8_sanitize(convo.substr(convo.size() - lo));
      }
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
    go.display_name = friendly_gguf_name(o.gguf);
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

  // ------------------------------------------------ deterministic retrieval
  auto status = [&](const std::string& m) {
    if (obs.on_status) obs.on_status(m);
  };
  // How much reference material this answer can actually carry.  A fixed budget
  // cannot work for both models: 2400 characters is a third of what OLMo can
  // hold and about twice SPT's entire 512-token window, so injecting it into SPT
  // left no room for the question itself.  Derive it from whichever backend is
  // going to answer, at a conservative 3 characters per token.
  const int64_t answer_ctx =
      (req.mode == AskMode::kStrong || req.mode == AskMode::kDebate) && p_->strong
          ? p_->opt.gguf_ctx
          : (p_->spt ? p_->spt->context_limit() : 512);
  const size_t ctx_chars = static_cast<size_t>(
      std::max<int64_t>(400, (answer_ctx - req.max_tokens - 128) * 3));

  std::string context;
  if (req.auto_tools) {
    const bool web_ok = req.use_tools && p_->opt.enable_web &&
                        p_->tools.enabled("web_search") && HttpClient::available();
    const ToolPlan plan =
        plan_tools(req.question, req.use_codebase && !p_->index.empty(), web_ok);

    // ---- the codebase, and which files it chose
    if (plan.codebase) {
      const std::string l = lower(req.question);
      const bool overview = l.find("what does") != std::string::npos ||
                            l.find("overview") != std::string::npos ||
                            l.find("چیکار") != std::string::npos ||
                            l.find("چه کاری") != std::string::npos;
      status("searching the indexed code");
      ToolTrace tr;
      tr.tool = "codebase";
      tr.args = utf8_truncate(req.question, 60);
      const double t1 = Telemetry::now();
      if (overview) {
        context += p_->index.overview(std::min<size_t>(ctx_chars, 2400));
        tr.detail = "read the repository overview";
      } else {
        const std::vector<SearchHit> hits = p_->index.search(req.question, 8);
        for (const SearchHit& h : hits) {
          const CodeChunk* c = p_->index.chunk(h.chunk);
          if (!c) continue;
          if (std::find(tr.sources.begin(), tr.sources.end(), c->path) ==
              tr.sources.end()) {
            tr.sources.push_back(c->path);
            status("reading " + c->header());
          }
        }
        context += p_->index.context_block(req.question, ctx_chars, 8);
        tr.detail = "read " + std::to_string(tr.sources.size()) + " file(s): ";
        for (size_t i = 0; i < tr.sources.size() && i < 6; ++i)
          tr.detail += (i ? ", " : "") + tr.sources[i];
      }
      tr.ok = !context.empty();
      tr.seconds = Telemetry::now() - t1;
      tr.output = utf8_truncate(context, 400);
      out.tools.push_back(tr);
      for (const std::string& sc : tr.sources) out.sources.push_back(sc);
      if (obs.on_tool) obs.on_tool(tr);
    }

    // ---- the web
    auto run_tool = [&](const std::string& name,
                        const std::map<std::string, std::string>& args,
                        const std::string& doing) {
      status(doing);
      ToolCall c;
      c.name = name;
      c.args = args;
      ToolContext tc = p_->make_ctx(cancel);
      tc.output_budget = static_cast<int>(std::min<size_t>(ctx_chars, 3000));
      const ToolResult r = p_->tools.invoke(c, tc, &p_->gate, p_->policy);
      ToolTrace tr;
      tr.tool = name;
      tr.args = args.empty() ? std::string() : args.begin()->second;
      tr.ok = r.ok;
      tr.denied = r.denied;
      tr.seconds = r.seconds;
      tr.output = utf8_truncate(r.ok ? r.output : r.error, 600);
      tr.detail = doing;
      if (r.ok) {
        // Pull the URLs out of the result so the answer can cite them and the UI
        // can show which sites were actually consulted.
        size_t at = 0;
        while ((at = r.output.find("http", at)) != std::string::npos) {
          size_t e = at;
          while (e < r.output.size() &&
                 !std::isspace(static_cast<unsigned char>(r.output[e])))
            ++e;
          const std::string u = r.output.substr(at, e - at);
          if (u.size() > 12 &&
              std::find(tr.sources.begin(), tr.sources.end(), u) == tr.sources.end())
            tr.sources.push_back(u);
          at = e;
          if (tr.sources.size() >= 6) break;
        }
      }
      out.tools.push_back(tr);
      for (const std::string& sc : tr.sources) out.sources.push_back(sc);
      if (obs.on_tool) obs.on_tool(tr);
      return r;
    };

    if (plan.fetch && !plan.url.empty()) {
      const ToolResult r = run_tool("web_fetch",
                                    {{"url", plan.url}, {"query", req.question}},
                                    "reading " + HttpClient::host_of(plan.url));
      if (r.ok) context += "\n" + r.output;
    } else if (plan.search) {
      const ToolResult r = run_tool("web_search", {{"query", req.question}, {"k", "5"}},
                                    "searching the web: " +
                                        utf8_truncate(req.question, 70));
      if (r.ok) {
        context += "\n" + r.output;
        // Snippets are rarely enough to answer with, so read a page too.  Sites
        // that refuse robots (403) are common enough that stopping at the first
        // failure loses the answer outright - so try the next candidate instead,
        // best-ranked first.
        for (const std::string& u : rank_urls(r.output, req.question)) {
          if (cancel && cancel->load()) break;
          const ToolResult f =
              run_tool("web_fetch", {{"url", u}, {"query", req.question}},
                       "reading " + HttpClient::host_of(u));
          if (f.ok && f.output.size() > 200) {
            context += "\n" + f.output;
            break;
          }
        }
      }
    }
    if (!plan.reason.empty() && !context.empty())
      out.thinking += "why tools ran: " + plan.reason + "\n";
  }
  if (!context.empty()) {
    out.context_used = context;
    out.thinking += "retrieved " + std::to_string(context.size()) +
                    " characters of reference material\n";
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
  // Only advertise the tool syntax when it can still help: a model that already
  // has the retrieved context does not need to ask for it, and a model shown the
  // catalogue tends to imitate it in its answer (OLMo echoing
  // "[[web_search:...]]" back at us).  And never to a model too small to emit it.
  const bool advertise =
      req.use_tools && context.empty() && be->context_limit() >= 2048;
  if (advertise) {
    const std::string cat = p_->tools.catalogue(true);
    if (!cat.empty())
      sys += "\n\nTools you may call, one per reply, using exactly this form:\n"
             "[[tool:NAME]]\nkey: value\n[[/tool]]\n" +
             cat;
  }
  std::string question = req.question;
  if (!context.empty())
    question = "Reference material:\n" + context + "\nQuestion: " + req.question;

  status(req.mode == AskMode::kStrong ? "thinking (OLMo)" : "thinking (SPT)");
  AskRequest rq = req;
  rq.use_tools = advertise;   // the loop only parses calls if we invited them
  out.answer = p_->tool_loop(be, sys, question, rq, cancel, obs, &out);
  out.seconds = Telemetry::now() - t0;
  return out;
}

}  // namespace slm
