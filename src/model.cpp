// SPDX-License-Identifier: Apache-2.0
#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "core/config.h"
#include "core/rng.h"
#include "tokenizer.h"

namespace slm {
namespace {

uint64_t name_seed(const std::string& n, uint64_t base) {
  uint64_t h = 1469598103934665603ULL ^ base;
  for (char c : n) {
    h ^= static_cast<unsigned char>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

struct BlockWeights {
  Tensor ln1g, ln1b, qkvw, qkvb, pw, pb;
  Tensor ln2g, ln2b, fcw, fcb, mw, mb;
};

void cache_append(std::vector<float>& buf, const Tensor& t, int64_t B, int64_t H,
                  int64_t Tmax, int64_t D, int64_t t0) {
  const float* src = t.host_ptr();
  const int64_t T = t.size(2);
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h) {
      float* dst = buf.data() + (((b * H + h) * Tmax) + t0) * D;
      std::memcpy(dst, src + ((b * H + h) * T) * D,
                  sizeof(float) * static_cast<size_t>(T * D));
    }
}

Tensor cache_read(const std::vector<float>& buf, int64_t B, int64_t H, int64_t Tmax,
                  int64_t D, int64_t Tk) {
  Tensor out = Tensor::zeros({B, H, Tk, D});
  float* dst = out.mutable_host_ptr();
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h)
      std::memcpy(dst + ((b * H + h) * Tk) * D,
                  buf.data() + ((b * H + h) * Tmax) * D,
                  sizeof(float) * static_cast<size_t>(Tk * D));
  return out;
}

// One transformer block.  Everything it needs is passed in explicitly so the
// function can be safely re-executed by the gradient-checkpointing machinery.
Tensor block_forward(const BlockWeights& w, const Tensor& x, int64_t B, int64_t Tq,
                     int64_t H, int64_t D, float eps, KVCache* cache, int64_t layer,
                     AttentionCapture* capture) {
  const int64_t C = H * D;
  Tensor h = x.layernorm(w.ln1g, w.ln1b, eps);
  Tensor qkv = linear(h, w.qkvw, &w.qkvb);  // [B,Tq,3C]
  Tensor q = qkv.slice(2, 0, C).reshape({B, Tq, H, D}).transpose(1, 2);
  Tensor k = qkv.slice(2, C, 2 * C).reshape({B, Tq, H, D}).transpose(1, 2);
  Tensor v = qkv.slice(2, 2 * C, 3 * C).reshape({B, Tq, H, D}).transpose(1, 2);

  int64_t Tk = Tq;
  if (cache) {
    const int64_t t0 = cache->T;
    Tk = t0 + Tq;
    cache_append(cache->k[static_cast<size_t>(layer)], k, B, H, cache->Tmax, D, t0);
    cache_append(cache->v[static_cast<size_t>(layer)], v, B, H, cache->Tmax, D, t0);
    k = cache_read(cache->k[static_cast<size_t>(layer)], B, H, cache->Tmax, D, Tk);
    v = cache_read(cache->v[static_cast<size_t>(layer)], B, H, cache->Tmax, D, Tk);
  }

  Tensor att = q.matmul(k.transpose(2, 3))
                   .scale(1.0f / std::sqrt(static_cast<float>(D)))
                   .causal_mask()
                   .softmax_last();  // [B,H,Tq,Tk]
  if (capture) {
    capture->n_head = H;
    capture->Tq = Tq;
    capture->Tk = Tk;
    if (static_cast<int64_t>(capture->layers.size()) <= layer)
      capture->layers.resize(static_cast<size_t>(layer) + 1);
    std::vector<float>& dst = capture->layers[static_cast<size_t>(layer)];
    dst.assign(static_cast<size_t>(H * Tq * Tk), 0.0f);
    std::memcpy(dst.data(), att.host_ptr(),
                sizeof(float) * static_cast<size_t>(H * Tq * Tk));  // batch 0
  }
  Tensor y = att.matmul(v).transpose(1, 2).reshape({B, Tq, C});
  Tensor xa = x.add(linear(y, w.pw, &w.pb));
  Tensor h2 = xa.layernorm(w.ln2g, w.ln2b, eps);
  Tensor m = linear(linear(h2, w.fcw, &w.fcb).gelu(), w.mw, &w.mb);
  return xa.add(m);
}

}  // namespace

// ============================================================== GPTConfig
int64_t GPTConfig::param_count() const {
  const int64_t C = n_embd;
  int64_t n = static_cast<int64_t>(vocab_size) * C;  // token embedding
  n += block_size * C;                               // positional embedding
  const int64_t per_block = 2 * (2 * C)              // 2 layernorms
                            + (C * 3 * C + 3 * C)    // qkv
                            + (C * C + C)            // attn proj
                            + (C * 4 * C + 4 * C)    // mlp fc
                            + (4 * C * C + C);       // mlp proj
  n += n_layer * per_block;
  n += 2 * C;  // final ln
  if (!tie_weights) n += C * static_cast<int64_t>(vocab_size);
  return n;
}

std::string GPTConfig::describe() const {
  std::ostringstream os;
  os << "GPT(layers=" << n_layer << ", heads=" << n_head << ", dim=" << n_embd
     << ", ctx=" << block_size << ", vocab=" << vocab_size
     << ", tied=" << (tie_weights ? "yes" : "no")
     << ", ckpt=" << (grad_checkpointing ? "on" : "off") << ") ~"
     << (param_count() / 1000000.0) << "M params";
  return os.str();
}

void GPTConfig::validate() const {
  if (n_embd % n_head != 0)
    throw std::runtime_error("model.n_embd must be divisible by model.n_head");
  if (n_layer <= 0 || n_head <= 0 || n_embd <= 0 || block_size <= 0)
    throw std::runtime_error("model dimensions must be positive");
  if (vocab_size < Tokenizer::kBaseVocab)
    throw std::runtime_error("model.vocab_size must be >= " +
                             std::to_string(Tokenizer::kBaseVocab));
}

GPTConfig GPTConfig::from_config(const Config& c) {
  GPTConfig g;
  g.vocab_size = static_cast<int32_t>(c.get_int("model.vocab_size", g.vocab_size));
  g.n_layer = c.get_int("model.n_layer", g.n_layer);
  g.n_head = c.get_int("model.n_head", g.n_head);
  g.n_embd = c.get_int("model.n_embd", g.n_embd);
  g.block_size = c.get_int("model.block_size", g.block_size);
  g.tie_weights = c.get_bool("model.tie_weights", g.tie_weights);
  g.grad_checkpointing = c.get_bool("model.grad_checkpointing", g.grad_checkpointing);
  g.init_std = static_cast<float>(c.get_num("model.init_std", g.init_std));
  g.ln_eps = static_cast<float>(c.get_num("model.ln_eps", g.ln_eps));
  return g;
}

void GPTConfig::write_to(Config& c) const {
  c.set("model.vocab_size", std::to_string(vocab_size));
  c.set("model.n_layer", std::to_string(n_layer));
  c.set("model.n_head", std::to_string(n_head));
  c.set("model.n_embd", std::to_string(n_embd));
  c.set("model.block_size", std::to_string(block_size));
  c.set("model.tie_weights", tie_weights ? "true" : "false");
  c.set("model.grad_checkpointing", grad_checkpointing ? "true" : "false");
  c.set("model.init_std", std::to_string(init_std));
  c.set("model.ln_eps", std::to_string(ln_eps));
}

std::string FreezePolicy::describe() const {
  std::ostringstream os;
  if (layernorms_only) return "layernorms only";
  if (last_k_blocks < 0)
    os << "all blocks";
  else
    os << "last " << last_k_blocks << " blocks";
  os << (train_embeddings ? " +emb" : "") << (train_head ? " +head" : "")
     << (train_final_ln ? " +ln_f" : "");
  return os.str();
}

void KVCache::reset(int64_t B_, int64_t H_, int64_t D_, int64_t Tmax_, int64_t n_layer) {
  B = B_;
  H = H_;
  D = D_;
  Tmax = Tmax_;
  T = 0;
  k.assign(static_cast<size_t>(n_layer),
           std::vector<float>(static_cast<size_t>(B * H * Tmax * D), 0.0f));
  v = k;
}

// =================================================================== GPT
GPT::GPT(const GPTConfig& cfg) : cfg_(cfg) {
  cfg_.validate();
  const int64_t C = cfg_.n_embd;
  register_param("tok_emb", {cfg_.vocab_size, C});
  register_param("pos_emb", {cfg_.block_size, C});
  for (int64_t l = 0; l < cfg_.n_layer; ++l) {
    const std::string p = block_prefix(l);
    register_param(p + "ln1.g", {C});
    register_param(p + "ln1.b", {C});
    register_param(p + "attn.qkv.w", {C, 3 * C});
    register_param(p + "attn.qkv.b", {3 * C});
    register_param(p + "attn.proj.w", {C, C});
    register_param(p + "attn.proj.b", {C});
    register_param(p + "ln2.g", {C});
    register_param(p + "ln2.b", {C});
    register_param(p + "mlp.fc.w", {C, 4 * C});
    register_param(p + "mlp.fc.b", {4 * C});
    register_param(p + "mlp.proj.w", {4 * C, C});
    register_param(p + "mlp.proj.b", {C});
  }
  register_param("ln_f.g", {C});
  register_param("ln_f.b", {C});
  if (!cfg_.tie_weights) register_param("lm_head.w", {C, cfg_.vocab_size});
  recompute_trainable();
}

std::string GPT::block_prefix(int64_t l) const {
  return "h." + std::to_string(l) + ".";
}

void GPT::register_param(const std::string& name, const Shape& s) {
  params_.emplace(name, Tensor::zeros(s, true));
  order_.push_back(name);
}

Tensor* GPT::param(const std::string& name) {
  auto it = params_.find(name);
  return it == params_.end() ? nullptr : &it->second;
}
const Tensor* GPT::param(const std::string& name) const {
  auto it = params_.find(name);
  return it == params_.end() ? nullptr : &it->second;
}

int64_t GPT::num_params() const {
  int64_t n = 0;
  for (const std::string& nm : order_) n += params_.at(nm).numel();
  return n;
}

int64_t GPT::num_trainable() const {
  int64_t n = 0;
  for (const std::string& nm : trainable_) n += params_.at(nm).numel();
  return n;
}

void GPT::init_weights(uint64_t seed) {
  const float resid_std =
      cfg_.init_std / std::sqrt(2.0f * static_cast<float>(cfg_.n_layer));
  for (const std::string& n : order_) {
    Tensor& t = params_.at(n);
    const uint64_t s = name_seed(n, seed);
    const bool is_ln_gain = n.size() > 2 && n.compare(n.size() - 2, 2, ".g") == 0 &&
                            (n.find("ln") != std::string::npos);
    const bool is_bias = n.size() > 2 && n.compare(n.size() - 2, 2, ".b") == 0;
    if (is_ln_gain) {
      t = Tensor::full(t.shape(), 1.0f, true);
    } else if (is_bias) {
      t = Tensor::zeros(t.shape(), true);
    } else if (n.find("proj.w") != std::string::npos) {
      t = Tensor::randn(t.shape(), resid_std, s, true);
    } else if (n == "pos_emb") {
      t = Tensor::randn(t.shape(), cfg_.init_std * 0.5f, s, true);
    } else {
      t = Tensor::randn(t.shape(), cfg_.init_std, s, true);
    }
  }
  recompute_trainable();
}

void GPT::set_freeze_policy(const FreezePolicy& p) {
  policy_ = p;
  recompute_trainable();
}

void GPT::recompute_trainable() {
  trainable_.clear();
  for (const std::string& n : order_) {
    bool ok = true;
    if (n == "tok_emb" || n == "pos_emb") {
      ok = policy_.train_embeddings;
    } else if (n == "lm_head.w") {
      ok = policy_.train_head;
    } else if (n.rfind("ln_f.", 0) == 0) {
      ok = policy_.train_final_ln;
    } else if (n.rfind("h.", 0) == 0) {
      const size_t dot = n.find('.', 2);
      const int64_t l = std::stoll(n.substr(2, dot - 2));
      ok = (policy_.last_k_blocks < 0) ||
           (l >= cfg_.n_layer - policy_.last_k_blocks);
      if (ok && policy_.layernorms_only) ok = n.find(".ln") != std::string::npos;
    }
    Tensor& t = params_.at(n);
    t.set_requires_grad(ok);
    if (ok) trainable_.push_back(n);
  }
}

std::vector<Tensor*> GPT::trainable_params() {
  std::vector<Tensor*> out;
  out.reserve(trainable_.size());
  for (const std::string& n : trainable_) out.push_back(&params_.at(n));
  return out;
}

void GPT::zero_grad() {
  for (const std::string& n : order_) params_.at(n).zero_grad();
}

ParamStorePtr GPT::snapshot() const {
  auto s = std::make_shared<ParamStore>();
  std::vector<float> tmp;
  for (const std::string& n : order_) {
    params_.at(n).copy_to_host(tmp);
    s->add(n, params_.at(n).shape(), tmp);
  }
  return s;
}

void GPT::load_params(const ParamStore& s) {
  for (const std::string& n : order_) {
    const int i = s.find(n);
    if (i < 0) throw std::runtime_error("load_params: missing parameter " + n);
    Tensor& t = params_.at(n);
    const std::vector<float>& d = s.data[static_cast<size_t>(i)];
    if (static_cast<int64_t>(d.size()) != t.numel())
      throw std::runtime_error("load_params: size mismatch for " + n);
    t.copy_from_host(d.data(), static_cast<int64_t>(d.size()));
  }
}

// ------------------------------------------------------------------ forward
Tensor GPT::forward(const std::vector<int32_t>& ids, int64_t B, int64_t T,
                    const ForwardOptions& opt) {
  const int64_t H = cfg_.n_head, D = cfg_.head_dim();
  if (static_cast<int64_t>(ids.size()) != B * T)
    throw std::runtime_error("forward: ids size != B*T");
  if (opt.pos_offset + T > cfg_.block_size)
    throw std::runtime_error("forward: sequence longer than block_size");

  const bool use_ckpt = opt.checkpointing && grad_enabled() && !opt.cache && !opt.capture;

  Tensor x = embedding(params_.at("tok_emb"), ids, B, T);
  Tensor pos = params_.at("pos_emb").slice(0, opt.pos_offset, opt.pos_offset + T);
  x = x.add_bias(pos);

  if (opt.capture) {
    opt.capture->clear();
    opt.capture->n_layer = cfg_.n_layer;
  }
  for (int64_t l = 0; l < cfg_.n_layer; ++l) {
    const std::string p = block_prefix(l);
    BlockWeights w{params_.at(p + "ln1.g"),      params_.at(p + "ln1.b"),
                   params_.at(p + "attn.qkv.w"), params_.at(p + "attn.qkv.b"),
                   params_.at(p + "attn.proj.w"), params_.at(p + "attn.proj.b"),
                   params_.at(p + "ln2.g"),      params_.at(p + "ln2.b"),
                   params_.at(p + "mlp.fc.w"),   params_.at(p + "mlp.fc.b"),
                   params_.at(p + "mlp.proj.w"), params_.at(p + "mlp.proj.b")};
    if (use_ckpt) {
      const float eps = cfg_.ln_eps;
      // Captured *by value*: the body is replayed during the backward pass.
      auto body = [w, B, T, H, D, eps](const Tensor& inp) {
        return block_forward(w, inp, B, T, H, D, eps, nullptr, 0, nullptr);
      };
      x = checkpoint(body, x);
    } else {
      x = block_forward(w, x, B, T, H, D, cfg_.ln_eps, opt.cache, l, opt.capture);
    }
  }
  x = x.layernorm(params_.at("ln_f.g"), params_.at("ln_f.b"), cfg_.ln_eps);
  Tensor logits = cfg_.tie_weights
                      ? linear(x, params_.at("tok_emb").transpose(0, 1), nullptr)
                      : linear(x, params_.at("lm_head.w"), nullptr);
  if (opt.cache) opt.cache->T += T;
  return logits;
}

Tensor GPT::loss(const std::vector<int32_t>& ids, const std::vector<int32_t>& targets,
                 int64_t B, int64_t T, float* loss_out, int64_t* ntok_out,
                 bool checkpointing_default) {
  ForwardOptions opt;
  opt.checkpointing = cfg_.grad_checkpointing && checkpointing_default;
  Tensor logits = forward(ids, B, T, opt);
  Tensor flat = logits.reshape({B * T, cfg_.vocab_size});
  return cross_entropy(flat, targets, -100, loss_out, ntok_out);
}

float GPT::eval_loss(const std::vector<int32_t>& ids,
                     const std::vector<int32_t>& targets, int64_t B, int64_t T) {
  NoGradGuard ng;
  Tensor logits = forward(ids, B, T, ForwardOptions());
  Tensor flat = logits.reshape({B * T, cfg_.vocab_size});
  float l = 0.0f;
  cross_entropy(flat, targets, -100, &l, nullptr);
  return l;
}

float GPT::sequence_logprob(const std::vector<int32_t>& ids, int64_t* ntok) {
  NoGradGuard ng;
  if (ids.size() < 2) {
    if (ntok) *ntok = 0;
    return 0.0f;
  }
  const int64_t T = std::min<int64_t>(static_cast<int64_t>(ids.size()) - 1,
                                      cfg_.block_size);
  std::vector<int32_t> inp(ids.begin(), ids.begin() + T);
  std::vector<int32_t> tgt(ids.begin() + 1, ids.begin() + T + 1);
  Tensor logits = forward(inp, 1, T, ForwardOptions());
  Tensor lp = seq_logprob(logits, tgt, -100);
  if (ntok) *ntok = T;
  return lp.host_ptr()[0];
}

// ----------------------------------------------------------------- sampling
std::vector<int32_t> GPT::generate(const std::vector<int32_t>& prompt,
                                   const GenOptions& opt,
                                   const std::function<bool(const GenStep&)>& on_step,
                                   AttentionCapture* capture) {
  NoGradGuard ng;
  const int64_t H = cfg_.n_head, D = cfg_.head_dim();
  const int32_t V = cfg_.vocab_size;
  std::vector<int32_t> out;
  if (prompt.empty()) return out;

  const int64_t max_ctx = cfg_.block_size;
  const int64_t keep = std::max<int64_t>(
      1, std::min<int64_t>(static_cast<int64_t>(prompt.size()),
                           max_ctx - std::max(1, opt.max_new_tokens)));
  std::vector<int32_t> ctx(prompt.end() - keep, prompt.end());

  KVCache cache;
  cache.reset(1, H, D, max_ctx, cfg_.n_layer);

  Rng rng(opt.seed ? opt.seed : 0x5eed1234u);
  std::vector<int32_t> history = ctx;
  std::vector<float> row(static_cast<size_t>(V));
  std::vector<std::pair<float, int32_t>> cand;

  ForwardOptions fo;
  fo.cache = &cache;
  fo.capture = capture;
  Tensor logits = forward(ctx, 1, static_cast<int64_t>(ctx.size()), fo);

  for (int step = 0; step < opt.max_new_tokens; ++step) {
    // last position of the current logits
    const int64_t Tq = logits.size(1);
    const float* lp = logits.host_ptr() + (Tq - 1) * V;
    std::copy(lp, lp + V, row.begin());

    if (opt.repetition_penalty > 1.0f) {
      const size_t look = std::min<size_t>(history.size(), 128);
      for (size_t i = history.size() - look; i < history.size(); ++i) {
        const int32_t t = history[i];
        if (t < 0 || t >= V) continue;
        row[static_cast<size_t>(t)] /= (row[static_cast<size_t>(t)] > 0.0f)
                                           ? opt.repetition_penalty
                                           : 1.0f / opt.repetition_penalty;
      }
    }
    const float temp = std::max(1e-4f, opt.temperature);
    for (float& v : row) v /= temp;

    cand.clear();
    cand.reserve(static_cast<size_t>(V));
    for (int32_t i = 0; i < V; ++i) cand.emplace_back(row[static_cast<size_t>(i)], i);
    const int k = (opt.top_k > 0) ? std::min<int>(opt.top_k, V) : V;
    std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                      [](const std::pair<float, int32_t>& a,
                         const std::pair<float, int32_t>& b) { return a.first > b.first; });
    cand.resize(static_cast<size_t>(k));

    float mx = cand[0].first;
    double sum = 0.0;
    for (auto& c : cand) {
      c.first = std::exp(c.first - mx);
      sum += c.first;
    }
    for (auto& c : cand) c.first = static_cast<float>(c.first / sum);

    // nucleus filtering
    if (opt.top_p > 0.0f && opt.top_p < 1.0f) {
      double acc = 0.0;
      size_t keep_n = 0;
      for (; keep_n < cand.size(); ++keep_n) {
        acc += cand[keep_n].first;
        if (acc >= opt.top_p) {
          ++keep_n;
          break;
        }
      }
      cand.resize(std::max<size_t>(1, keep_n));
      double s2 = 0.0;
      for (const auto& c : cand) s2 += c.first;
      for (auto& c : cand) c.first = static_cast<float>(c.first / s2);
    }

    // sample
    const float u = rng.uniform();
    double acc = 0.0;
    int32_t chosen = cand.back().second;
    float chosen_p = cand.back().first;
    for (const auto& c : cand) {
      acc += c.first;
      if (u <= acc) {
        chosen = c.second;
        chosen_p = c.first;
        break;
      }
    }

    GenStep gs;
    gs.token = chosen;
    gs.logprob = std::log(std::max(1e-12f, chosen_p));
    const size_t ntop = std::min<size_t>(cand.size(), 12);
    for (size_t i = 0; i < ntop; ++i) gs.top.emplace_back(cand[i].second, cand[i].first);

    out.push_back(chosen);
    history.push_back(chosen);
    if (on_step && !on_step(gs)) break;
    if (opt.stop_on_eot && chosen == 0 /*<|endoftext|>*/) break;
    if (cache.T + 1 > max_ctx) break;

    const std::vector<int32_t> one{chosen};
    ForwardOptions fo2;
    fo2.cache = &cache;
    fo2.pos_offset = cache.T;
    logits = forward(one, 1, 1, fo2);
  }
  return out;
}

}  // namespace slm
