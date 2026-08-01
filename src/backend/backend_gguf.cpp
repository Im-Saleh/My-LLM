// SPDX-License-Identifier: Apache-2.0
#include "backend/backend_gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

#include "telemetry.h"

#ifdef SLM_WITH_LLAMA
#include "llama.h"
#endif

namespace slm {

#ifndef SLM_WITH_LLAMA
// ---------------------------------------------------------------- stub build
// Kept deliberately complete rather than #ifdef-ing the class away: the GUI
// lists the backend, shows why it is unavailable, and the rest of the system
// (debate, agent, registry) compiles and runs unchanged.
struct ModelBackendGGUF::Impl {
  GgufBackendOptions opt;
  BackendStatus st;
};

ModelBackendGGUF::ModelBackendGGUF(const GgufBackendOptions& o) : p_(new Impl) {
  p_->opt = o;
  p_->st.detail = "built without llama.cpp";
}
ModelBackendGGUF::~ModelBackendGGUF() = default;
bool ModelBackendGGUF::compiled_in() { return false; }
std::string ModelBackendGGUF::llama_version() { return "not built"; }
std::string ModelBackendGGUF::id() const { return p_->opt.id; }
std::string ModelBackendGGUF::display_name() const { return p_->opt.display_name; }
std::string ModelBackendGGUF::runtime() const { return "llama.cpp (not built)"; }
BackendCaps ModelBackendGGUF::caps() const { return BackendCaps(); }
BackendStatus ModelBackendGGUF::status() const { return p_->st; }
bool ModelBackendGGUF::load(std::string* err) {
  const std::string m =
      "this binary was built without llama.cpp - rebuild with "
      "./build.sh --llama to use GGUF models";
  if (err) *err = m;
  p_->st.error = m;
  return false;
}
void ModelBackendGGUF::unload() {}
bool ModelBackendGGUF::loaded() const { return false; }
int64_t ModelBackendGGUF::context_limit() const { return 0; }
std::vector<int32_t> ModelBackendGGUF::tokenize(const std::string&) const { return {}; }
std::string ModelBackendGGUF::detokenize(const std::vector<int32_t>&) const { return {}; }
std::string ModelBackendGGUF::format_chat(
    const std::string&, const std::vector<std::pair<std::string, std::string>>&,
    bool) const {
  return {};
}
int ModelBackendGGUF::open_session() { return -1; }
void ModelBackendGGUF::close_session(int) {}
void ModelBackendGGUF::reset_session(int) {}
int64_t ModelBackendGGUF::session_tokens(int) const { return 0; }
GenResponse ModelBackendGGUF::generate(const GenRequest& req, const ChunkFn&,
                                       std::atomic<bool>*) {
  GenResponse r;
  r.tag = req.tag;
  r.error = "built without llama.cpp";
  return r;
}
std::vector<GenResponse> ModelBackendGGUF::generate_many(
    const std::vector<GenRequest>& reqs, const ChunkFn& f, std::atomic<bool>* c) {
  return IModelBackend::generate_many(reqs, f, c);
}
bool ModelBackendGGUF::score(const std::string&, const std::string&, double*,
                             std::string* err) {
  if (err) *err = "built without llama.cpp";
  return false;
}
bool ModelBackendGGUF::embed(const std::string&, std::vector<float>*,
                             std::string* err) {
  if (err) *err = "built without llama.cpp";
  return false;
}
int ModelBackendGGUF::embedding_dim() const { return 0; }

#else  // ================================================== real llama.cpp build

namespace {

bool g_backend_ready = false;

void ensure_backend() {
  if (g_backend_ready) return;
  llama_backend_init();
  // llama.cpp is chatty on stderr by default; the dashboard owns the terminal.
  llama_log_set([](ggml_log_level lvl, const char* text, void* /*ud*/) {
    if (lvl >= GGML_LOG_LEVEL_ERROR && text) std::fputs(text, stderr);
  }, nullptr);
  g_backend_ready = true;
}

// A stop string can straddle two tokens, so it is matched on decoded text.
bool ends_with_any(const std::string& text, const std::vector<std::string>& stops,
                   size_t* cut) {
  for (const std::string& s : stops) {
    if (s.empty()) continue;
    const size_t p = text.rfind(s);
    if (p != std::string::npos && p + s.size() == text.size()) {
      *cut = p;
      return true;
    }
  }
  return false;
}

size_t common_prefix(const std::vector<llama_token>& a,
                     const std::vector<llama_token>& b) {
  const size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

}  // namespace

struct ModelBackendGGUF::Impl {
  GgufBackendOptions opt;
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  const llama_vocab* vocab = nullptr;
  BackendStatus st;
  std::string chat_template;
  int n_embd = 0;

  struct Session {
    int seq = -1;                     // llama sequence id
    std::vector<llama_token> tokens;  // exactly what this sequence's cache holds
  };
  std::map<int, Session> sessions;
  std::vector<bool> seq_used;
  int next_slot = 1;

  ~Impl() { close(); }

  void close() {
    sessions.clear();
    seq_used.clear();
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    ctx = nullptr;
    model = nullptr;
    vocab = nullptr;
  }

  int claim_seq() {
    for (size_t i = 0; i < seq_used.size(); ++i)
      if (!seq_used[i]) {
        seq_used[i] = true;
        return static_cast<int>(i);
      }
    return -1;
  }

  std::vector<llama_token> tokenize(const std::string& text, bool add_special) const {
    if (!vocab) return {};
    int need = -llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                               nullptr, 0, add_special, true);
    if (need <= 0) return {};
    std::vector<llama_token> out(static_cast<size_t>(need));
    const int got = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                   out.data(), need, add_special, true);
    if (got < 0) return {};
    out.resize(static_cast<size_t>(got));
    return out;
  }

  std::string piece(llama_token t) const {
    char buf[256];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
    if (n < 0) {
      std::vector<char> big(static_cast<size_t>(-n));
      const int m = llama_token_to_piece(vocab, t, big.data(),
                                        static_cast<int32_t>(big.size()), 0, false);
      return m > 0 ? std::string(big.data(), static_cast<size_t>(m)) : std::string();
    }
    return std::string(buf, static_cast<size_t>(n));
  }

  // Decodes `tokens` into sequence `seq` starting at position `pos`.  Only the
  // last token of the batch is asked for logits, which is all generation needs.
  bool decode_range(int seq, const std::vector<llama_token>& tokens, int pos,
                    std::atomic<bool>* cancel, std::string* err) {
    const int nb = std::max(1, opt.n_batch);
    llama_batch batch = llama_batch_init(nb, 0, 1);
    bool ok = true;
    for (size_t off = 0; off < tokens.size() && ok; off += static_cast<size_t>(nb)) {
      if (cancel && cancel->load()) {
        if (err) *err = "cancelled";
        ok = false;
        break;
      }
      const size_t chunk = std::min<size_t>(static_cast<size_t>(nb), tokens.size() - off);
      batch.n_tokens = static_cast<int32_t>(chunk);
      for (size_t i = 0; i < chunk; ++i) {
        batch.token[i] = tokens[off + i];
        batch.pos[i] = pos + static_cast<int>(off + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = 0;
      }
      const bool last_chunk = off + chunk >= tokens.size();
      if (last_chunk) batch.logits[chunk - 1] = 1;
      const int rc = llama_decode(ctx, batch);
      if (rc != 0) {
        if (err)
          *err = "llama_decode failed (" + std::to_string(rc) +
                 "); context may be too small";
        ok = false;
      }
    }
    llama_batch_free(batch);
    return ok;
  }

  llama_sampler* make_sampler(const SamplingParams& sp) const {
    llama_sampler_chain_params cp = llama_sampler_chain_default_params();
    cp.no_perf = true;
    llama_sampler* chain = llama_sampler_chain_init(cp);
    // Order matters: penalties act on raw logits, then truncation narrows the
    // candidate set, then temperature reshapes what is left, then we draw.
    if (sp.repetition_penalty > 1.0f || sp.presence_penalty > 0.0f)
      llama_sampler_chain_add(
          chain, llama_sampler_init_penalties(128, sp.repetition_penalty, 0.0f,
                                              sp.presence_penalty));
    if (sp.top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
    if (sp.top_p > 0.0f && sp.top_p < 1.0f)
      llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, 1));
    if (sp.min_p > 0.0f)
      llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(std::max(0.0f, sp.temperature)));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(
                                       sp.seed ? static_cast<uint32_t>(sp.seed)
                                               : LLAMA_DEFAULT_SEED));
    return chain;
  }
};

ModelBackendGGUF::ModelBackendGGUF(const GgufBackendOptions& o) : p_(new Impl) {
  p_->opt = o;
}
ModelBackendGGUF::~ModelBackendGGUF() = default;

bool ModelBackendGGUF::compiled_in() { return true; }
std::string ModelBackendGGUF::llama_version() {
  // llama.cpp does not expose a version string in its public header; the build
  // system pins the tag, so report that plus what the runtime says about itself.
  return std::string("llama.cpp b6833, ") + llama_print_system_info();
}

std::string ModelBackendGGUF::id() const { return p_->opt.id; }
std::string ModelBackendGGUF::display_name() const { return p_->opt.display_name; }
std::string ModelBackendGGUF::runtime() const {
  std::string r = "llama.cpp";
  if (p_->opt.n_gpu_layers > 0) r += " (" + std::to_string(p_->opt.n_gpu_layers) + " layers on GPU)";
  else r += " (CPU)";
  return r;
}

BackendCaps ModelBackendGGUF::caps() const {
  BackendCaps c;
  c.trainable = false;   // LoRA fine-tuning is out of process, not here
  c.attention_capture = false;
  c.logprobs = true;
  c.batched = true;      // several sequences share one decode loop
  c.gpu = p_->opt.n_gpu_layers > 0;
  c.max_parallel = p_->opt.n_seq_max;
  return c;
}

BackendStatus ModelBackendGGUF::status() const {
  BackendStatus s = p_->st;
  s.loaded = p_->model != nullptr;
  if (s.loaded) {
    s.params = static_cast<int64_t>(llama_model_n_params(p_->model));
    s.weight_bytes = static_cast<size_t>(llama_model_size(p_->model));
    size_t tok = 0;
    for (const auto& kv : p_->sessions) tok += kv.second.tokens.size();
    // KV bytes are not exposed per sequence; report the honest estimate from the
    // context geometry instead of pretending to know.
    s.kv_bytes = tok * static_cast<size_t>(llama_model_n_embd(p_->model)) * 2 * 2;
  }
  return s;
}

bool ModelBackendGGUF::load(std::string* err) {
  ensure_backend();
  p_->close();
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    p_->st.error = m;
    return false;
  };
  if (p_->opt.path.empty()) return fail("no GGUF path configured");

  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = p_->opt.n_gpu_layers;
  mp.use_mmap = p_->opt.use_mmap;   // this is what keeps a 4.5 GB model out of RSS
  mp.use_mlock = p_->opt.use_mlock;
  p_->model = llama_model_load_from_file(p_->opt.path.c_str(), mp);
  if (!p_->model) return fail("llama.cpp could not load " + p_->opt.path);
  p_->vocab = llama_model_get_vocab(p_->model);
  p_->n_embd = llama_model_n_embd(p_->model);

  const int hw = static_cast<int>(std::thread::hardware_concurrency());
  const int threads = p_->opt.n_threads > 0 ? p_->opt.n_threads : std::max(1, hw);
  llama_context_params cp = llama_context_default_params();
  const int trained = llama_model_n_ctx_train(p_->model);
  cp.n_ctx = static_cast<uint32_t>(std::min(p_->opt.n_ctx, trained > 0 ? trained : p_->opt.n_ctx));
  cp.n_batch = static_cast<uint32_t>(p_->opt.n_batch);
  cp.n_ubatch = static_cast<uint32_t>(std::min(p_->opt.n_batch, 512));
  cp.n_seq_max = static_cast<uint32_t>(std::max(1, p_->opt.n_seq_max));
  // Without this, llama.cpp splits the KV cache into n_seq_max equal slices, so
  // n_ctx=4096 with n_seq_max=4 gives every sequence only 1024 cells - while
  // llama_n_ctx() still reports 4096.  Any prompt over 1024 tokens then dies
  // with "llama_decode failed (1)", which is exactly what a retrieval-augmented
  // question produces.  A unified pool lets one sequence use the whole context
  // when it is the only one running, which is the common case, and still lets
  // four debate voices share it when they are short.
  cp.kv_unified = true;
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  cp.no_perf = true;
  cp.flash_attn_type = p_->opt.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                          : LLAMA_FLASH_ATTN_TYPE_DISABLED;
  if (p_->opt.kv_q8) {
    // Quantising the KV cache is the cheapest way to buy context on a CPU box:
    // q8_0 halves it for a perplexity change that is lost in the noise.
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
  }
  if (p_->opt.embeddings) {
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
  }
  p_->ctx = llama_init_from_model(p_->model, cp);
  if (!p_->ctx) {
    p_->close();
    return fail("llama.cpp could not create a context (try a smaller --ctx)");
  }
  p_->seq_used.assign(static_cast<size_t>(cp.n_seq_max), false);

  const char* tmpl = llama_model_chat_template(p_->model, nullptr);
  p_->chat_template = tmpl ? tmpl : "";

  char desc[256] = {0};
  llama_model_desc(p_->model, desc, sizeof(desc));
  p_->st.detail = std::string(desc) + ", ctx " + std::to_string(cp.n_ctx) +
                  ", " + std::to_string(threads) + " threads" +
                  (p_->chat_template.empty() ? ", no chat template" : "") +
                  (p_->opt.kv_q8 ? ", q8 KV" : "");
  p_->st.error.clear();
  if (p_->opt.tel)
    p_->opt.tel->log("info", "backend", "loaded " + p_->opt.display_name,
                     {{"path", p_->opt.path},
                      {"detail", p_->st.detail},
                      {"params", std::to_string(llama_model_n_params(p_->model))}});
  return true;
}

void ModelBackendGGUF::unload() { p_->close(); }
bool ModelBackendGGUF::loaded() const { return p_->model != nullptr; }

int64_t ModelBackendGGUF::context_limit() const {
  return p_->ctx ? static_cast<int64_t>(llama_n_ctx(p_->ctx)) : 0;
}

std::vector<int32_t> ModelBackendGGUF::tokenize(const std::string& text) const {
  const std::vector<llama_token> t = p_->tokenize(text, false);
  return std::vector<int32_t>(t.begin(), t.end());
}

std::string ModelBackendGGUF::detokenize(const std::vector<int32_t>& ids) const {
  std::string out;
  for (int32_t t : ids) out += p_->piece(static_cast<llama_token>(t));
  return out;
}

std::string ModelBackendGGUF::format_chat(
    const std::string& system,
    const std::vector<std::pair<std::string, std::string>>& turns,
    bool add_generation_prompt) const {
  // A base model has no template and no notion of roles; concatenating is the
  // only honest thing to do, and the debate engine is told (base_model) not to
  // give such a participant a judging role.
  if (p_->chat_template.empty() || p_->opt.base_model) {
    std::string out;
    if (!system.empty()) out += system + "\n\n";
    for (const auto& t : turns) {
      out += t.first;
      if (!t.second.empty()) out += "\n" + t.second + "\n\n";
    }
    return out;
  }
  std::vector<llama_chat_message> msgs;
  std::vector<std::string> keep;  // owns the strings the messages point into
  keep.reserve(turns.size() * 2 + 1);
  if (!system.empty()) {
    keep.push_back(system);
    msgs.push_back({"system", keep.back().c_str()});
  }
  for (const auto& t : turns) {
    keep.push_back(t.first);
    msgs.push_back({"user", keep.back().c_str()});
    if (!t.second.empty()) {
      keep.push_back(t.second);
      msgs.push_back({"assistant", keep.back().c_str()});
    }
  }
  std::vector<char> buf(4096);
  int n = llama_chat_apply_template(p_->chat_template.c_str(), msgs.data(), msgs.size(),
                                    add_generation_prompt, buf.data(),
                                    static_cast<int32_t>(buf.size()));
  if (n > static_cast<int>(buf.size())) {
    buf.resize(static_cast<size_t>(n) + 1);
    n = llama_chat_apply_template(p_->chat_template.c_str(), msgs.data(), msgs.size(),
                                  add_generation_prompt, buf.data(),
                                  static_cast<int32_t>(buf.size()));
  }
  if (n <= 0) return system + "\n" + (turns.empty() ? "" : turns.front().first);
  return std::string(buf.data(), static_cast<size_t>(n));
}

int ModelBackendGGUF::open_session() {
  const int seq = p_->claim_seq();
  if (seq < 0) return -1;
  const int slot = p_->next_slot++;
  Impl::Session s;
  s.seq = seq;
  p_->sessions[slot] = s;
  return slot;
}

void ModelBackendGGUF::close_session(int slot) {
  auto it = p_->sessions.find(slot);
  if (it == p_->sessions.end()) return;
  if (p_->ctx) llama_memory_seq_rm(llama_get_memory(p_->ctx), it->second.seq, -1, -1);
  if (it->second.seq >= 0 && static_cast<size_t>(it->second.seq) < p_->seq_used.size())
    p_->seq_used[static_cast<size_t>(it->second.seq)] = false;
  p_->sessions.erase(it);
}

void ModelBackendGGUF::reset_session(int slot) {
  auto it = p_->sessions.find(slot);
  if (it == p_->sessions.end()) return;
  if (p_->ctx) llama_memory_seq_rm(llama_get_memory(p_->ctx), it->second.seq, -1, -1);
  it->second.tokens.clear();
}

int64_t ModelBackendGGUF::session_tokens(int slot) const {
  auto it = p_->sessions.find(slot);
  return it == p_->sessions.end() ? 0 : static_cast<int64_t>(it->second.tokens.size());
}

GenResponse ModelBackendGGUF::generate(const GenRequest& req, const ChunkFn& on_chunk,
                                       std::atomic<bool>* cancel) {
  GenResponse r;
  r.tag = req.tag;
  const double t0 = Telemetry::now();
  if (!p_->ctx) {
    r.error = p_->st.error.empty() ? "backend not loaded" : p_->st.error;
    return r;
  }

  // A slot-less request borrows a scratch sequence and wipes it afterwards.
  int seq = 0;
  Impl::Session* s = nullptr;
  bool scratch = false;
  auto it = p_->sessions.find(req.slot);
  if (it != p_->sessions.end()) {
    s = &it->second;
    seq = s->seq;
  } else {
    seq = p_->claim_seq();
    if (seq < 0) seq = 0;
    scratch = true;
    llama_memory_seq_rm(llama_get_memory(p_->ctx), seq, -1, -1);
  }

  std::vector<llama_token> ids = p_->tokenize(req.prompt, true);
  const int64_t ctx = context_limit();
  const int64_t want = std::max(1, req.sp.max_new_tokens);
  const int64_t room = std::max<int64_t>(4, ctx - want - 4);
  if (static_cast<int64_t>(ids.size()) > room)
    ids.erase(ids.begin(), ids.end() - room);  // keep the instruction, drop the head
  r.prompt_tokens = static_cast<int>(ids.size());

  size_t reuse = 0;
  if (s) {
    reuse = common_prefix(s->tokens, ids);
    if (reuse >= ids.size()) reuse = ids.size() - 1;
    // Everything after the shared prefix is stale; drop exactly that.
    llama_memory_seq_rm(llama_get_memory(p_->ctx), seq, static_cast<llama_pos>(reuse), -1);
  } else {
    llama_memory_seq_rm(llama_get_memory(p_->ctx), seq, -1, -1);
  }
  r.reused_tokens = static_cast<int>(reuse);
  p_->st.cache_hit_tokens += static_cast<int64_t>(reuse);

  std::vector<llama_token> feed(ids.begin() + static_cast<long>(reuse), ids.end());
  const double tp0 = Telemetry::now();
  std::string derr;
  if (!p_->decode_range(seq, feed, static_cast<int>(reuse), cancel, &derr)) {
    r.error = derr;
    if (scratch && seq >= 0 && static_cast<size_t>(seq) < p_->seq_used.size())
      p_->seq_used[static_cast<size_t>(seq)] = false;
    return r;
  }
  const double prefill = Telemetry::now() - tp0;
  if (s) s->tokens = ids;

  llama_sampler* smpl = p_->make_sampler(req.sp);
  std::string text;
  int pos = static_cast<int>(ids.size());
  double lp_sum = 0.0;
  const int64_t cap = std::min<int64_t>(want, ctx - static_cast<int64_t>(ids.size()) - 1);
  for (int64_t step = 0; step < cap; ++step) {
    if (cancel && cancel->load()) {
      r.cancelled = true;
      break;
    }
    const llama_token tok = llama_sampler_sample(smpl, p_->ctx, -1);
    llama_sampler_accept(smpl, tok);
    if (llama_vocab_is_eog(p_->vocab, tok)) break;

    // Probability of what was actually chosen, for the fluency signal.
    {
      const float* lg = llama_get_logits_ith(p_->ctx, -1);
      const int nv = llama_vocab_n_tokens(p_->vocab);
      if (lg && nv > 0) {
        float mx = lg[0];
        for (int i = 1; i < nv; ++i) mx = std::max(mx, lg[i]);
        double sum = 0.0;
        for (int i = 0; i < nv; ++i) sum += std::exp(static_cast<double>(lg[i]) - mx);
        lp_sum += static_cast<double>(lg[tok]) - mx - std::log(sum);
      }
    }
    const std::string piece = p_->piece(tok);
    text += piece;
    ++r.gen_tokens;
    if (on_chunk) {
      GenChunk c;
      c.text = piece;
      c.token = tok;
      c.index = req.tag;
      if (!on_chunk(c)) {
        r.cancelled = true;
        break;
      }
    }
    size_t cut = 0;
    if (!req.sp.stop.empty() && ends_with_any(text, req.sp.stop, &cut)) {
      text.resize(cut);
      break;
    }
    if (s) s->tokens.push_back(tok);
    const std::vector<llama_token> one{tok};
    if (!p_->decode_range(seq, one, pos, cancel, &derr)) {
      r.error = derr;
      break;
    }
    ++pos;
  }
  llama_sampler_free(smpl);
  if (r.gen_tokens >= cap) r.truncated = true;
  r.text = text;
  r.mean_logprob = r.gen_tokens ? lp_sum / r.gen_tokens : 0.0;
  r.seconds = Telemetry::now() - t0;

  if (scratch) {
    llama_memory_seq_rm(llama_get_memory(p_->ctx), seq, -1, -1);
    if (seq >= 0 && static_cast<size_t>(seq) < p_->seq_used.size())
      p_->seq_used[static_cast<size_t>(seq)] = false;
  }
  p_->st.calls += 1;
  p_->st.total_prompt_tokens += r.prompt_tokens;
  p_->st.total_gen_tokens += r.gen_tokens;
  p_->st.busy_seconds += r.seconds;
  p_->st.last_latency_s = r.seconds;
  const int fresh = r.prompt_tokens - r.reused_tokens;
  p_->st.last_prompt_tps = prefill > 0 ? fresh / prefill : 0.0;
  const double dec = r.seconds - prefill;
  p_->st.last_decode_tps = dec > 0 ? r.gen_tokens / dec : 0.0;
  return r;
}

std::vector<GenResponse> ModelBackendGGUF::generate_many(
    const std::vector<GenRequest>& reqs, const ChunkFn& on_chunk,
    std::atomic<bool>* cancel) {
  // True batched decoding: N sequences advance one token per llama_decode, so N
  // answers cost roughly the wall time of one.  On a memory-bound CPU decode the
  // weights are read once for the whole batch, which is exactly where the win
  // comes from - this is the "batch the debate rounds into one call" path.
  if (!p_->ctx || reqs.size() <= 1)
    return IModelBackend::generate_many(reqs, on_chunk, cancel);

  const size_t n = reqs.size();
  const int64_t ctx = context_limit();
  struct Run {
    int seq = 0;
    bool scratch = false;
    bool done = false;
    int pos = 0;
    std::vector<llama_token> ids;
    llama_sampler* smpl = nullptr;
    GenResponse out;
    Impl::Session* sess = nullptr;
    double lp_sum = 0.0;
    int64_t cap = 0;
    int logit_row = -1;
  };
  std::vector<Run> runs(n);
  std::vector<int> claimed;

  for (size_t i = 0; i < n; ++i) {
    Run& R = runs[i];
    R.out.tag = reqs[i].tag;
    auto it = p_->sessions.find(reqs[i].slot);
    if (it != p_->sessions.end()) {
      R.sess = &it->second;
      R.seq = it->second.seq;
    } else {
      R.seq = p_->claim_seq();
      if (R.seq < 0) {  // out of sequences: fall back to serial for the rest
        for (int c : claimed) p_->seq_used[static_cast<size_t>(c)] = false;
        return IModelBackend::generate_many(reqs, on_chunk, cancel);
      }
      R.scratch = true;
      claimed.push_back(R.seq);
    }
  }

  const double t0 = Telemetry::now();
  // ---- prefill each sequence (serial, but each one reuses its own prefix)
  for (size_t i = 0; i < n; ++i) {
    Run& R = runs[i];
    R.ids = p_->tokenize(reqs[i].prompt, true);
    const int64_t want = std::max(1, reqs[i].sp.max_new_tokens);
    const int64_t room = std::max<int64_t>(4, ctx - want - 4);
    if (static_cast<int64_t>(R.ids.size()) > room)
      R.ids.erase(R.ids.begin(), R.ids.end() - room);
    R.out.prompt_tokens = static_cast<int>(R.ids.size());
    size_t reuse = 0;
    if (R.sess) {
      reuse = common_prefix(R.sess->tokens, R.ids);
      if (reuse >= R.ids.size()) reuse = R.ids.size() - 1;
      llama_memory_seq_rm(llama_get_memory(p_->ctx), R.seq,
                          static_cast<llama_pos>(reuse), -1);
    } else {
      llama_memory_seq_rm(llama_get_memory(p_->ctx), R.seq, -1, -1);
    }
    R.out.reused_tokens = static_cast<int>(reuse);
    std::vector<llama_token> feed(R.ids.begin() + static_cast<long>(reuse), R.ids.end());
    std::string derr;
    if (!p_->decode_range(R.seq, feed, static_cast<int>(reuse), cancel, &derr)) {
      R.out.error = derr;
      R.done = true;
      continue;
    }
    if (R.sess) R.sess->tokens = R.ids;
    R.pos = static_cast<int>(R.ids.size());
    R.smpl = p_->make_sampler(reqs[i].sp);
    R.cap = std::min<int64_t>(want, ctx - static_cast<int64_t>(R.ids.size()) - 1);
    // The first sampled token comes from the prefill's last logit row.
    R.logit_row = -1;
  }

  // ---- joint decode loop
  llama_batch batch = llama_batch_init(static_cast<int32_t>(n), 0, 1);
  int64_t step = 0;
  int64_t max_cap = 0;
  for (const Run& R : runs) max_cap = std::max(max_cap, R.cap);
  while (step < max_cap) {
    if (cancel && cancel->load()) {
      for (Run& R : runs)
        if (!R.done) R.out.cancelled = true;
      break;
    }
    // Sample one token for every live sequence from the logits it just produced.
    batch.n_tokens = 0;
    for (size_t i = 0; i < n; ++i) {
      Run& R = runs[i];
      if (R.done || R.out.gen_tokens >= R.cap) {
        if (!R.done && R.out.gen_tokens >= R.cap) {
          R.out.truncated = true;
          R.done = true;
        }
        continue;
      }
      const llama_token tok = llama_sampler_sample(R.smpl, p_->ctx, R.logit_row);
      llama_sampler_accept(R.smpl, tok);
      if (llama_vocab_is_eog(p_->vocab, tok)) {
        R.done = true;
        continue;
      }
      {
        const float* lg = llama_get_logits_ith(p_->ctx, R.logit_row);
        const int nv = llama_vocab_n_tokens(p_->vocab);
        if (lg && nv > 0) {
          float mx = lg[0];
          for (int k = 1; k < nv; ++k) mx = std::max(mx, lg[k]);
          double sum = 0.0;
          for (int k = 0; k < nv; ++k) sum += std::exp(static_cast<double>(lg[k]) - mx);
          R.lp_sum += static_cast<double>(lg[tok]) - mx - std::log(sum);
        }
      }
      const std::string piece = p_->piece(tok);
      R.out.text += piece;
      ++R.out.gen_tokens;
      if (on_chunk) {
        GenChunk c;
        c.text = piece;
        c.token = tok;
        c.index = reqs[i].tag;
        if (!on_chunk(c)) {
          R.out.cancelled = true;
          R.done = true;
          continue;
        }
      }
      size_t cut = 0;
      if (!reqs[i].sp.stop.empty() && ends_with_any(R.out.text, reqs[i].sp.stop, &cut)) {
        R.out.text.resize(cut);
        R.done = true;
        continue;
      }
      if (R.sess) R.sess->tokens.push_back(tok);
      const int row = batch.n_tokens;
      batch.token[row] = tok;
      batch.pos[row] = R.pos++;
      batch.n_seq_id[row] = 1;
      batch.seq_id[row][0] = R.seq;
      batch.logits[row] = 1;
      R.logit_row = row;  // where this sequence's next logits will live
      ++batch.n_tokens;
    }
    if (batch.n_tokens == 0) break;
    const int rc = llama_decode(p_->ctx, batch);
    if (rc != 0) {
      for (Run& R : runs)
        if (!R.done) {
          R.out.error = "llama_decode failed (" + std::to_string(rc) + ")";
          R.done = true;
        }
      break;
    }
    ++step;
  }
  llama_batch_free(batch);

  const double total = Telemetry::now() - t0;
  std::vector<GenResponse> out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    Run& R = runs[i];
    if (R.smpl) llama_sampler_free(R.smpl);
    if (R.scratch) {
      llama_memory_seq_rm(llama_get_memory(p_->ctx), R.seq, -1, -1);
      if (static_cast<size_t>(R.seq) < p_->seq_used.size())
        p_->seq_used[static_cast<size_t>(R.seq)] = false;
    }
    R.out.mean_logprob = R.out.gen_tokens ? R.lp_sum / R.out.gen_tokens : 0.0;
    // Wall time is shared by construction; report each run's share of it so the
    // numbers still add up to something meaningful.
    R.out.seconds = total / static_cast<double>(n);
    p_->st.calls += 1;
    p_->st.total_prompt_tokens += R.out.prompt_tokens;
    p_->st.total_gen_tokens += R.out.gen_tokens;
    out.push_back(std::move(R.out));
  }
  p_->st.busy_seconds += total;
  p_->st.last_latency_s = total;
  int gen = 0;
  for (const GenResponse& g : out) gen += g.gen_tokens;
  p_->st.last_decode_tps = total > 0 ? gen / total : 0.0;
  return out;
}

bool ModelBackendGGUF::score(const std::string& context, const std::string& text,
                             double* mean_logprob, std::string* err) {
  if (!p_->ctx) {
    if (err) *err = "backend not loaded";
    return false;
  }
  const std::vector<llama_token> ctx_ids = p_->tokenize(context, true);
  const std::vector<llama_token> txt_ids = p_->tokenize(text, false);
  if (txt_ids.empty()) {
    if (mean_logprob) *mean_logprob = 0.0;
    return true;
  }
  std::vector<llama_token> all = ctx_ids;
  all.insert(all.end(), txt_ids.begin(), txt_ids.end());
  size_t first = ctx_ids.size();
  const int64_t limit = context_limit();
  if (static_cast<int64_t>(all.size()) > limit) {
    const size_t drop = std::min<size_t>(first, all.size() - static_cast<size_t>(limit));
    all.erase(all.begin(), all.begin() + static_cast<long>(drop));
    first -= drop;
  }
  if (all.size() < 2) {
    if (mean_logprob) *mean_logprob = 0.0;
    return true;
  }

  const int seq = p_->claim_seq() >= 0 ? p_->claim_seq() : 0;
  llama_memory_seq_rm(llama_get_memory(p_->ctx), seq, -1, -1);
  // Every position needs logits here, so the batch asks for all of them.
  const int nb = std::max(1, p_->opt.n_batch);
  llama_batch batch = llama_batch_init(nb, 0, 1);
  double total = 0.0;
  int64_t n = 0;
  bool ok = true;
  const int nv = llama_vocab_n_tokens(p_->vocab);
  for (size_t off = 0; off + 1 < all.size() && ok; off += static_cast<size_t>(nb)) {
    const size_t chunk = std::min<size_t>(static_cast<size_t>(nb), all.size() - 1 - off);
    batch.n_tokens = static_cast<int32_t>(chunk);
    for (size_t i = 0; i < chunk; ++i) {
      batch.token[i] = all[off + i];
      batch.pos[i] = static_cast<llama_pos>(off + i);
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = seq;
      batch.logits[i] = 1;
    }
    if (llama_decode(p_->ctx, batch) != 0) {
      ok = false;
      break;
    }
    for (size_t i = 0; i < chunk; ++i) {
      const size_t next = off + i + 1;
      if (next < first) continue;  // only the scored text counts
      const float* lg = llama_get_logits_ith(p_->ctx, static_cast<int32_t>(i));
      if (!lg) continue;
      float mx = lg[0];
      for (int k = 1; k < nv; ++k) mx = std::max(mx, lg[k]);
      double sum = 0.0;
      for (int k = 0; k < nv; ++k) sum += std::exp(static_cast<double>(lg[k]) - mx);
      total += static_cast<double>(lg[all[next]]) - mx - std::log(sum);
      ++n;
    }
  }
  llama_batch_free(batch);
  llama_memory_seq_rm(llama_get_memory(p_->ctx), seq, -1, -1);
  if (static_cast<size_t>(seq) < p_->seq_used.size())
    p_->seq_used[static_cast<size_t>(seq)] = false;
  if (!ok) {
    if (err) *err = "llama_decode failed while scoring";
    return false;
  }
  if (mean_logprob) *mean_logprob = n ? total / static_cast<double>(n) : 0.0;
  return true;
}

bool ModelBackendGGUF::embed(const std::string& text, std::vector<float>* out,
                             std::string* err) {
  if (!p_->ctx || !p_->opt.embeddings) {
    if (err) *err = "this backend was not opened with embeddings=true";
    return false;
  }
  std::vector<llama_token> ids = p_->tokenize(text, true);
  if (ids.empty()) {
    if (err) *err = "empty text";
    return false;
  }
  const int64_t limit = context_limit();
  if (static_cast<int64_t>(ids.size()) > limit) ids.resize(static_cast<size_t>(limit));
  llama_memory_seq_rm(llama_get_memory(p_->ctx), 0, -1, -1);
  llama_batch batch = llama_batch_init(static_cast<int32_t>(ids.size()), 0, 1);
  batch.n_tokens = static_cast<int32_t>(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    batch.token[i] = ids[i];
    batch.pos[i] = static_cast<llama_pos>(i);
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = 1;  // pooling needs every position marked as an output
  }
  const int rc = llama_decode(p_->ctx, batch);
  llama_batch_free(batch);
  if (rc != 0) {
    if (err) *err = "llama_decode failed while embedding";
    return false;
  }
  const float* e = llama_get_embeddings_seq(p_->ctx, 0);
  if (!e) {
    if (err) *err = "no embedding produced (is this an embedding model?)";
    return false;
  }
  out->assign(e, e + p_->n_embd);
  double ss = 0.0;
  for (float v : *out) ss += static_cast<double>(v) * v;
  if (ss > 0.0) {
    const float inv = static_cast<float>(1.0 / std::sqrt(ss));
    for (float& v : *out) v *= inv;
  }
  return true;
}

int ModelBackendGGUF::embedding_dim() const { return p_->n_embd; }

#endif  // SLM_WITH_LLAMA

}  // namespace slm
