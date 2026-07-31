// SPDX-License-Identifier: Apache-2.0
#include "backend/backend_spt.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/serialize.h"
#include "telemetry.h"
#include "tokenizer.h"

namespace slm {
namespace {

// How many tokens the shared prefix of two token sequences has.  This is the one
// number that decides whether a debate round costs a prefill or nothing.
size_t common_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

// Does `text` end with any stop string?  Checked on the decoded suffix, because a
// stop string can straddle two tokens.
bool hit_stop(const std::string& text, const std::vector<std::string>& stops,
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

}  // namespace

ModelBackendSPT::ModelBackendSPT(const SptBackendOptions& o) : opt_(o), cfg_(o.cfg) {}
ModelBackendSPT::~ModelBackendSPT() = default;

std::string ModelBackendSPT::runtime() const {
  switch (opt_.source) {
    case SptBackendOptions::kLive: return std::string("native/live+") + backend_name();
    case SptBackendOptions::kQuant: return "native/int4-mmap";
    default: break;
  }
  return std::string("native/") + backend_name();
}

BackendCaps ModelBackendSPT::caps() const {
  BackendCaps c;
  c.trainable = opt_.source == SptBackendOptions::kLive;
  c.attention_capture = opt_.source != SptBackendOptions::kQuant;
  c.logprobs = true;
  c.batched = false;  // one model, one thread: drafts are serialised
  c.gpu = backend_on_gpu();
  c.max_parallel = opt_.max_sessions;
  return c;
}

BackendStatus ModelBackendSPT::status() const {
  BackendStatus s = st_;
  s.loaded = loaded_;
  if (loaded_) {
    s.params = param_count();
    if (qmodel_) {
      s.weight_bytes = static_cast<size_t>(qmodel_->file_bytes());
      s.detail = qmodel_->describe();
    } else if (model_) {
      s.weight_bytes = static_cast<size_t>(s.params) * sizeof(float);
      s.detail = cfg_.describe();
      if (opt_.source == SptBackendOptions::kLive)
        s.detail += "  [live, weights v" + std::to_string(weight_version_) + "]";
    }
    size_t kv = 0;
    for (const auto& kvp : sessions_) {
      kv += kvp.second.kv.bytes();
      kv += kvp.second.qs.cache_bytes();
    }
    s.kv_bytes = kv;
  }
  return s;
}

int64_t ModelBackendSPT::param_count() const {
  if (qmodel_) return qmodel_->param_count();
  if (model_) return model_->num_params();
  return cfg_.param_count();
}

bool ModelBackendSPT::load(std::string* err) {
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    st_.error = m;
    loaded_ = false;
    return false;
  };
  // The tokenizer: supplied, or loaded from a path, or read from the checkpoint.
  tok_ = opt_.tok;
  if (!tok_ && !opt_.tokenizer_path.empty()) {
    owned_tok_ = std::make_unique<Tokenizer>();
    if (!owned_tok_->load(opt_.tokenizer_path))
      return fail("cannot load tokenizer " + opt_.tokenizer_path);
    tok_ = owned_tok_.get();
  }

  if (opt_.source == SptBackendOptions::kQuant) {
    qmodel_ = std::make_unique<QModel>();
    std::string e;
    if (!qmodel_->open(opt_.path, &e)) return fail(e);
    cfg_ = qmodel_->config();
  } else if (opt_.source == SptBackendOptions::kCheckpoint) {
    ParamStore ps;
    CheckpointMeta meta;
    if (!load_checkpoint(opt_.path, &ps, &meta))
      return fail("cannot load checkpoint " + opt_.path);
    cfg_ = GPTConfig::from_config(meta.extra);
    // Progressive growth means the checkpoint, not the config, knows the depth.
    int64_t layers = 0;
    for (const std::string& n : ps.names) {
      if (n.rfind("h.", 0) != 0) continue;
      const size_t dot = n.find('.', 2);
      layers = std::max<int64_t>(layers, std::stoll(n.substr(2, dot - 2)) + 1);
    }
    if (layers > 0) cfg_.n_layer = layers;
    if (!tok_ && !meta.extra.get_str("tokenizer", "").empty()) {
      owned_tok_ = std::make_unique<Tokenizer>();
      if (owned_tok_->load(meta.extra.get_str("tokenizer", "")))
        tok_ = owned_tok_.get();
      else
        owned_tok_.reset();
    }
    if (!tok_) return fail("no tokenizer for " + opt_.path);
    cfg_.vocab_size = tok_->vocab_size();
    cfg_.grad_checkpointing = false;
    model_ = std::make_unique<GPT>(cfg_);
    model_->load_params(ps);
  } else {  // kLive
    if (!opt_.weights) return fail("live SPT backend needs a weight source");
    if (!tok_) return fail("live SPT backend needs a tokenizer");
    cfg_.vocab_size = tok_->vocab_size();
    cfg_.grad_checkpointing = false;
    model_ = std::make_unique<GPT>(cfg_);
    uint64_t v = 0;
    ParamStorePtr ps = opt_.weights(&v);
    if (!ps) return fail("no published weights yet");
    model_->load_params(*ps);
    weight_version_ = v;
  }
  st_.error.clear();
  loaded_ = true;
  return true;
}

void ModelBackendSPT::unload() {
  sessions_.clear();
  model_.reset();
  qmodel_.reset();
  owned_tok_.reset();
  tok_ = nullptr;
  loaded_ = false;
}

void ModelBackendSPT::resync_weights() {
  if (opt_.source != SptBackendOptions::kLive || !opt_.weights || !model_) return;
  uint64_t v = 0;
  ParamStorePtr ps = opt_.weights(&v);
  if (!ps || v == weight_version_) return;
  try {
    model_->load_params(*ps);
    weight_version_ = v;
    // Every cached KV state was produced by the old weights, so it is stale.
    for (auto& kvp : sessions_) {
      kvp.second.tokens.clear();
      kvp.second.primed = false;
    }
  } catch (const std::exception& e) {
    if (opt_.tel)
      opt_.tel->log("warn", "backend",
                    std::string("SPT weight resync failed: ") + e.what());
  }
}

int64_t ModelBackendSPT::context_limit() const { return cfg_.block_size; }

std::vector<int32_t> ModelBackendSPT::tokenize(const std::string& text) const {
  if (!tok_) return {};
  return tok_->encode(text);
}

std::string ModelBackendSPT::detokenize(const std::vector<int32_t>& ids) const {
  if (!tok_) return {};
  return tok_->decode(ids);
}

std::string ModelBackendSPT::format_chat(
    const std::string& system,
    const std::vector<std::pair<std::string, std::string>>& turns,
    bool add_generation_prompt) const {
  // The control tokens the corpus was built with; Tokenizer::encode maps these
  // literal markers to ids 1..3 rather than to their bytes.
  std::string out;
  if (!system.empty()) out += "<|system|>" + system;
  for (const auto& t : turns) {
    out += "<|user|>" + t.first;
    if (!t.second.empty()) out += "<|assistant|>" + t.second;
  }
  if (add_generation_prompt) out += "<|assistant|>";
  return out;
}

int ModelBackendSPT::open_session() {
  const int slot = next_slot_++;
  sessions_[slot] = Session();
  return slot;
}

void ModelBackendSPT::close_session(int slot) { sessions_.erase(slot); }

void ModelBackendSPT::reset_session(int slot) {
  auto it = sessions_.find(slot);
  if (it == sessions_.end()) return;
  it->second.tokens.clear();
  it->second.primed = false;
}

int64_t ModelBackendSPT::session_tokens(int slot) const {
  auto it = sessions_.find(slot);
  return it == sessions_.end() ? 0 : static_cast<int64_t>(it->second.tokens.size());
}

ModelBackendSPT::Session* ModelBackendSPT::session(int slot) {
  auto it = sessions_.find(slot);
  return it == sessions_.end() ? nullptr : &it->second;
}

bool ModelBackendSPT::run_prefix(Session& s, const std::vector<int32_t>& ids,
                                 int64_t pos, std::vector<float>* logits,
                                 std::string* err) {
  if (ids.empty()) {
    if (err) *err = "nothing to run";
    return false;
  }
  const int64_t V = cfg_.vocab_size;
  if (qmodel_) {
    s.qs.pos = pos;
    qmodel_->forward(&s.qs, ids, logits);
    return true;
  }
  NoGradGuard ng;
  ForwardOptions fo;
  fo.cache = &s.kv;
  fo.pos_offset = pos;
  s.kv.T = pos;
  Tensor out = model_->forward(ids, 1, static_cast<int64_t>(ids.size()), fo);
  const int64_t T = out.size(1);
  logits->assign(static_cast<size_t>(V), 0.0f);
  std::memcpy(logits->data(), out.host_ptr() + (T - 1) * V,
              sizeof(float) * static_cast<size_t>(V));
  return true;
}

int32_t ModelBackendSPT::sample(const std::vector<float>& logits,
                                const SamplingParams& sp,
                                const std::vector<int32_t>& history, Rng& rng,
                                float* logprob) const {
  const int32_t V = static_cast<int32_t>(
      std::min<size_t>(logits.size(), static_cast<size_t>(cfg_.vocab_size)));
  std::vector<float> row(logits.begin(), logits.begin() + V);

  if (sp.repetition_penalty > 1.0f && !history.empty()) {
    const size_t look = std::min<size_t>(history.size(), 128);
    for (size_t i = history.size() - look; i < history.size(); ++i) {
      const int32_t t = history[i];
      if (t < 0 || t >= V) continue;
      float& r = row[static_cast<size_t>(t)];
      r = (r > 0.0f) ? r / sp.repetition_penalty : r * sp.repetition_penalty;
    }
  }
  if (sp.presence_penalty > 0.0f) {
    for (const int32_t t : history)
      if (t >= 0 && t < V) row[static_cast<size_t>(t)] -= sp.presence_penalty;
  }
  const float temp = std::max(1e-4f, sp.temperature);
  for (float& v : row) v /= temp;

  std::vector<std::pair<float, int32_t>> cand;
  cand.reserve(static_cast<size_t>(V));
  for (int32_t i = 0; i < V; ++i) cand.emplace_back(row[static_cast<size_t>(i)], i);
  const int k = (sp.top_k > 0) ? std::min<int>(sp.top_k, V) : V;
  std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                    [](const std::pair<float, int32_t>& a,
                       const std::pair<float, int32_t>& b) { return a.first > b.first; });
  cand.resize(static_cast<size_t>(k));

  const float mx = cand[0].first;
  double sum = 0.0;
  for (auto& c : cand) {
    c.first = std::exp(c.first - mx);
    sum += c.first;
  }
  for (auto& c : cand) c.first = static_cast<float>(c.first / sum);

  if (sp.min_p > 0.0f && cand.size() > 1) {
    const float floor_p = sp.min_p * cand[0].first;
    size_t keep = 1;
    while (keep < cand.size() && cand[keep].first >= floor_p) ++keep;
    if (keep < cand.size()) {
      cand.resize(keep);
      double s2 = 0.0;
      for (const auto& c : cand) s2 += c.first;
      for (auto& c : cand) c.first = static_cast<float>(c.first / s2);
    }
  }
  if (sp.top_p > 0.0f && sp.top_p < 1.0f) {
    double acc = 0.0;
    size_t keep = 0;
    for (; keep < cand.size(); ++keep) {
      acc += cand[keep].first;
      if (acc >= sp.top_p) {
        ++keep;
        break;
      }
    }
    cand.resize(std::max<size_t>(1, keep));
    double s2 = 0.0;
    for (const auto& c : cand) s2 += c.first;
    for (auto& c : cand) c.first = static_cast<float>(c.first / s2);
  }

  const float u = rng.uniform();
  double acc = 0.0;
  int32_t chosen = cand.back().second;
  float p = cand.back().first;
  for (const auto& c : cand) {
    acc += c.first;
    if (u <= acc) {
      chosen = c.second;
      p = c.first;
      break;
    }
  }
  if (logprob) *logprob = std::log(std::max(1e-12f, p));
  return chosen;
}

GenResponse ModelBackendSPT::generate(const GenRequest& req, const ChunkFn& on_chunk,
                                      std::atomic<bool>* cancel) {
  GenResponse r;
  r.tag = req.tag;
  const double t0 = Telemetry::now();
  if (!loaded_) {
    r.error = "backend not loaded";
    return r;
  }
  resync_weights();

  std::vector<int32_t> ids = tokenize(req.prompt);
  if (ids.empty()) ids.push_back(Tokenizer::kEot);

  const int64_t ctx = context_limit();
  const int64_t want_new = std::max(1, req.sp.max_new_tokens);
  // Keep the tail of the prompt: the instruction and the peer answers matter
  // more than the opening of a long system block.
  const int64_t room = std::max<int64_t>(1, ctx - want_new - 1);
  if (static_cast<int64_t>(ids.size()) > room)
    ids.erase(ids.begin(), ids.end() - room);
  r.prompt_tokens = static_cast<int>(ids.size());

  // -------------------------------------------------- session and cache reuse
  Session local;
  Session* s = req.slot >= 0 ? session(req.slot) : nullptr;
  if (!s) s = &local;
  const int64_t Tmax = ctx;
  const bool need_alloc =
      !s->primed || (qmodel_ ? s->qs.max_ctx < Tmax : s->kv.Tmax < Tmax);
  if (need_alloc) {
    if (qmodel_) qmodel_->reset(&s->qs, Tmax);
    else s->kv.reset(1, cfg_.kv_heads(), cfg_.head_dim(), Tmax, cfg_.n_layer);
    s->tokens.clear();
    s->primed = true;
  }
  size_t reuse = common_prefix(s->tokens, ids);
  // The last cached token must be re-run to obtain its logits, so never reuse
  // the whole prompt.
  if (reuse >= ids.size()) reuse = ids.size() - 1;
  s->tokens.resize(reuse);
  r.reused_tokens = static_cast<int>(reuse);
  st_.cache_hit_tokens += static_cast<int64_t>(reuse);

  std::vector<int32_t> feed(ids.begin() + static_cast<long>(reuse), ids.end());
  std::vector<float> logits;
  const double tp0 = Telemetry::now();
  std::string err;
  if (!run_prefix(*s, feed, static_cast<int64_t>(reuse), &logits, &err)) {
    r.error = err;
    return r;
  }
  s->tokens = ids;
  const double prefill = Telemetry::now() - tp0;

  // ------------------------------------------------------------------ decode
  Rng rng(req.sp.seed ? req.sp.seed : 0x5eed1234u);
  std::string text;
  std::vector<int32_t> emitted;
  double lp_sum = 0.0;
  const int64_t hard_cap = std::min<int64_t>(want_new, ctx - static_cast<int64_t>(ids.size()));
  for (int64_t step = 0; step < hard_cap; ++step) {
    if (cancel && cancel->load()) {
      r.cancelled = true;
      break;
    }
    float lp = 0.0f;
    const int32_t tok = sample(logits, req.sp, s->tokens, rng, &lp);
    lp_sum += lp;
    emitted.push_back(tok);
    if (req.sp.stop_on_eot && tok == Tokenizer::kEot) break;

    const std::string piece = detokenize({tok});
    text += piece;
    if (on_chunk) {
      GenChunk c;
      c.text = piece;
      c.token = tok;
      c.logprob = lp;
      c.index = req.tag;
      if (!on_chunk(c)) {
        r.cancelled = true;
        break;
      }
    }
    size_t cut = 0;
    if (!req.sp.stop.empty() && hit_stop(text, req.sp.stop, &cut)) {
      text.resize(cut);
      break;
    }
    s->tokens.push_back(tok);
    std::vector<float> next;
    if (!run_prefix(*s, {tok}, static_cast<int64_t>(s->tokens.size()) - 1, &next, &err)) {
      r.error = err;
      break;
    }
    logits.swap(next);
    if (static_cast<int64_t>(s->tokens.size()) + 1 >= ctx) {
      r.truncated = true;
      break;
    }
  }
  r.gen_tokens = static_cast<int>(emitted.size());
  if (r.gen_tokens >= hard_cap) r.truncated = true;
  r.text = text;
  r.mean_logprob = r.gen_tokens ? lp_sum / r.gen_tokens : 0.0;
  r.seconds = Telemetry::now() - t0;

  st_.calls += 1;
  st_.total_prompt_tokens += r.prompt_tokens;
  st_.total_gen_tokens += r.gen_tokens;
  st_.busy_seconds += r.seconds;
  st_.last_latency_s = r.seconds;
  const int fresh = r.prompt_tokens - r.reused_tokens;
  st_.last_prompt_tps = prefill > 0 ? fresh / prefill : 0.0;
  const double decode_s = r.seconds - prefill;
  st_.last_decode_tps = decode_s > 0 ? r.gen_tokens / decode_s : 0.0;
  return r;
}

bool ModelBackendSPT::score(const std::string& context, const std::string& text,
                            double* mean_logprob, std::string* err) {
  if (!loaded_) {
    if (err) *err = "backend not loaded";
    return false;
  }
  resync_weights();
  const std::vector<int32_t> ctx_ids = tokenize(context);
  const std::vector<int32_t> txt_ids = tokenize(text);
  if (txt_ids.empty()) {
    if (mean_logprob) *mean_logprob = 0.0;
    return true;
  }
  std::vector<int32_t> all = ctx_ids;
  all.insert(all.end(), txt_ids.begin(), txt_ids.end());
  // Keep the tail: the scored text must survive, the context may be clipped.
  int64_t drop = static_cast<int64_t>(all.size()) - context_limit();
  size_t first_scored = ctx_ids.size();
  if (drop > 0) {
    if (drop > static_cast<int64_t>(first_scored)) drop = static_cast<int64_t>(first_scored);
    all.erase(all.begin(), all.begin() + drop);
    first_scored -= static_cast<size_t>(drop);
  }
  if (all.size() < 2) {
    if (mean_logprob) *mean_logprob = 0.0;
    return true;
  }

  double total = 0.0;
  int64_t n = 0;
  if (qmodel_) {
    QGenState st;
    qmodel_->reset(&st, static_cast<int64_t>(all.size()) + 1);
    std::vector<float> logits;
    for (size_t i = 0; i + 1 < all.size(); ++i) {
      qmodel_->forward_token(&st, all[i], &logits);
      if (i + 1 < first_scored) continue;  // only score the text part
      const int32_t tgt = all[i + 1];
      float mx = logits[0];
      for (float v : logits) mx = std::max(mx, v);
      double sum = 0.0;
      for (float v : logits) sum += std::exp(static_cast<double>(v) - mx);
      total += static_cast<double>(logits[static_cast<size_t>(tgt)]) - mx - std::log(sum);
      ++n;
    }
  } else {
    NoGradGuard ng;
    const int64_t T = static_cast<int64_t>(all.size()) - 1;
    std::vector<int32_t> inp(all.begin(), all.end() - 1);
    Tensor out = model_->forward(inp, 1, T, ForwardOptions());
    const int64_t V = cfg_.vocab_size;
    const float* lg = out.host_ptr();
    for (int64_t i = 0; i < T; ++i) {
      if (static_cast<size_t>(i) + 1 < first_scored) continue;
      const float* row = lg + i * V;
      float mx = row[0];
      for (int64_t j = 1; j < V; ++j) mx = std::max(mx, row[j]);
      double sum = 0.0;
      for (int64_t j = 0; j < V; ++j) sum += std::exp(static_cast<double>(row[j]) - mx);
      const int32_t tgt = all[static_cast<size_t>(i) + 1];
      total += static_cast<double>(row[tgt]) - mx - std::log(sum);
      ++n;
    }
  }
  if (mean_logprob) *mean_logprob = n ? total / static_cast<double>(n) : 0.0;
  return true;
}

}  // namespace slm
