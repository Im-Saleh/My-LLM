// SPDX-License-Identifier: Apache-2.0
#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
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

// Everything one block needs.  Undefined tensors mean "this architecture does
// not have that parameter" (no biases with RMSNorm/SwiGLU, for example).
struct BlockWeights {
  Tensor qn, kn;                // optional QK-norm gains (per head dim)
  Tensor n1g, n1b;              // attention norm
  Tensor qkvw, qkvb;            // fused q,k,v projection
  Tensor pw, pb;                // attention output projection
  Tensor n2g, n2b;              // ffn norm
  Tensor fcw, fcb;              // GELU path: up projection
  Tensor gatew;                 // SwiGLU path: gate projection
  Tensor mw, mb;                // down projection
};

struct BlockShape {
  int64_t B = 0, Tq = 0;
  int64_t H = 0, Hkv = 0, Dh = 0;
  int64_t pos_offset = 0;
  float eps = 1e-5f;
  float rope_theta = 10000.0f;
  NormKind norm = NormKind::kRMSNorm;
  PosKind pos = PosKind::kRoPE;
  FFNKind ffn = FFNKind::kSwiGLU;
  bool qk_norm = true;
};

Tensor apply_norm(const Tensor& x, const Tensor& g, const Tensor& b, NormKind kind,
                  float eps) {
  return kind == NormKind::kRMSNorm ? x.rmsnorm(g, eps) : x.layernorm(g, b, eps);
}

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
      std::memcpy(dst + ((b * H + h) * Tk) * D, buf.data() + ((b * H + h) * Tmax) * D,
                  sizeof(float) * static_cast<size_t>(Tk * D));
  return out;
}

// One transformer block.  Everything it needs is passed explicitly so the
// function can be replayed by the gradient-checkpointing machinery.
Tensor block_forward(const BlockWeights& w, const Tensor& x, const BlockShape& s,
                     KVCache* cache, int64_t layer, AttentionCapture* capture) {
  const int64_t B = s.B, Tq = s.Tq, H = s.H, Hkv = s.Hkv, Dh = s.Dh;
  const int64_t C = H * Dh;
  const int64_t KV = Hkv * Dh;

  Tensor h = apply_norm(x, w.n1g, w.n1b, s.norm, s.eps);
  Tensor qkv = linear(h, w.qkvw, w.qkvb.defined() ? &w.qkvb : nullptr);  // [B,Tq,C+2KV]
  Tensor q = qkv.slice(2, 0, C).reshape({B, Tq, H, Dh}).transpose(1, 2);
  Tensor k = qkv.slice(2, C, C + KV).reshape({B, Tq, Hkv, Dh}).transpose(1, 2);
  Tensor v = qkv.slice(2, C + KV, C + 2 * KV).reshape({B, Tq, Hkv, Dh}).transpose(1, 2);

  if (s.qk_norm && w.qn.defined()) {
    q = q.rmsnorm(w.qn, s.eps);
    k = k.rmsnorm(w.kn, s.eps);
  }
  if (s.pos == PosKind::kRoPE) {
    q = rope(q, s.pos_offset, s.rope_theta);
    k = rope(k, s.pos_offset, s.rope_theta);
  }

  int64_t Tk = Tq;
  if (cache) {
    const int64_t t0 = cache->T;
    Tk = t0 + Tq;
    cache_append(cache->k[static_cast<size_t>(layer)], k, B, Hkv, cache->Tmax, Dh, t0);
    cache_append(cache->v[static_cast<size_t>(layer)], v, B, Hkv, cache->Tmax, Dh, t0);
    k = cache_read(cache->k[static_cast<size_t>(layer)], B, Hkv, cache->Tmax, Dh, Tk);
    v = cache_read(cache->v[static_cast<size_t>(layer)], B, Hkv, cache->Tmax, Dh, Tk);
  }
  if (Hkv != H) {  // grouped-query attention
    k = repeat_kv(k, H / Hkv);
    v = repeat_kv(v, H / Hkv);
  }

  Tensor att = q.matmul(k.transpose(2, 3))
                   .scale(1.0f / std::sqrt(static_cast<float>(Dh)))
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
  Tensor xa = x.add(linear(y, w.pw, w.pb.defined() ? &w.pb : nullptr));

  Tensor h2 = apply_norm(xa, w.n2g, w.n2b, s.norm, s.eps);
  Tensor m;
  if (s.ffn == FFNKind::kSwiGLU) {
    Tensor gate = linear(h2, w.gatew, nullptr).silu();
    Tensor up = linear(h2, w.fcw, nullptr);
    m = linear(gate.mul(up), w.mw, nullptr);
  } else {
    Tensor up = linear(h2, w.fcw, w.fcb.defined() ? &w.fcb : nullptr).gelu();
    m = linear(up, w.mw, w.mb.defined() ? &w.mb : nullptr);
  }
  return xa.add(m);
}

}  // namespace

// ============================================================== GPTConfig
int64_t GPTConfig::hidden_dim() const {
  if (ffn_hidden > 0) return ffn_hidden;
  if (ffn == FFNKind::kSwiGLU) {
    // 8/3 C keeps the parameter count of the two-matrix GELU MLP, rounded up to
    // a multiple of 64 for cache friendly GEMMs.
    const int64_t h = (n_embd * 8) / 3;
    return ((h + 63) / 64) * 64;
  }
  return 4 * n_embd;
}

int64_t GPTConfig::param_count() const {
  const int64_t C = n_embd;
  const int64_t Dh = head_dim();
  const int64_t KV = kv_heads() * Dh;
  const int64_t Hf = hidden_dim();
  int64_t n = static_cast<int64_t>(vocab_size) * C;  // token embedding
  if (pos == PosKind::kLearned) n += block_size * C;
  const int64_t norm_params = (norm == NormKind::kRMSNorm) ? C : 2 * C;
  int64_t per_block = 2 * norm_params;
  if (qk_norm) per_block += 2 * Dh;                 // q/k norm gains
  per_block += C * (C + 2 * KV);                    // qkv
  per_block += C * C;                               // attention out
  if (linear_bias) per_block += (C + 2 * KV) + C;
  if (ffn == FFNKind::kSwiGLU) {
    per_block += 3 * C * Hf;                        // gate + up + down
  } else {
    per_block += 2 * C * Hf;                        // up + down
    if (linear_bias) per_block += Hf + C;
  }
  n += n_layer * per_block;
  n += norm_params;  // final norm
  if (!tie_weights) n += C * static_cast<int64_t>(vocab_size);
  return n;
}

std::string GPTConfig::arch_summary() const {
  std::ostringstream os;
  os << (qk_norm ? "qknorm+" : "")
     << (norm == NormKind::kRMSNorm ? "rmsnorm" : "layernorm") << "+"
     << (pos == PosKind::kRoPE ? "rope" : "learned-pos") << "+"
     << (ffn == FFNKind::kSwiGLU ? "swiglu" : "gelu-mlp");
  if (kv_heads() != n_head) os << "+gqa" << n_head << ":" << kv_heads();
  return os.str();
}

std::string GPTConfig::describe() const {
  std::ostringstream os;
  os << "GPT(layers=" << n_layer << ", heads=" << n_head;
  if (kv_heads() != n_head) os << "/kv" << kv_heads();
  os << ", dim=" << n_embd << ", ffn=" << hidden_dim() << ", ctx=" << block_size
     << ", vocab=" << vocab_size << ", " << arch_summary()
     << (tie_weights ? ", tied" : "") << (grad_checkpointing ? ", ckpt" : "") << ") ~"
     << (param_count() / 1000000.0) << "M params";
  return os.str();
}

void GPTConfig::validate() const {
  if (n_embd % n_head != 0)
    throw std::runtime_error("model.n_embd must be divisible by model.n_head");
  if (n_layer <= 0 || n_head <= 0 || n_embd <= 0 || block_size <= 0)
    throw std::runtime_error("model dimensions must be positive");
  if (n_kv_head > 0 && n_head % n_kv_head != 0)
    throw std::runtime_error("model.n_head must be divisible by model.n_kv_head");
  if (pos == PosKind::kRoPE && head_dim() % 2 != 0)
    throw std::runtime_error("RoPE needs an even head dimension");
  if (vocab_size < Tokenizer::kBaseVocab)
    throw std::runtime_error("model.vocab_size must be >= " +
                             std::to_string(Tokenizer::kBaseVocab));
}

GPTConfig GPTConfig::from_config(const Config& c) {
  GPTConfig g;
  // Checkpoints and config files written before the architecture upgrade have
  // model.* keys but no model.arch_version: those describe the legacy
  // (GPT-2 style) stack, so the defaults flip for them.
  if (c.has("model.n_layer") && !c.has("model.arch_version")) {
    g.norm = NormKind::kLayerNorm;
    g.pos = PosKind::kLearned;
    g.ffn = FFNKind::kGeluMLP;
    g.linear_bias = true;
    g.qk_norm = false;
  }
  g.vocab_size = static_cast<int32_t>(c.get_int("model.vocab_size", g.vocab_size));
  g.n_layer = c.get_int("model.n_layer", g.n_layer);
  g.n_head = c.get_int("model.n_head", g.n_head);
  g.n_kv_head = c.get_int("model.n_kv_head", g.n_kv_head);
  g.n_embd = c.get_int("model.n_embd", g.n_embd);
  g.block_size = c.get_int("model.block_size", g.block_size);
  g.ffn_hidden = c.get_int("model.ffn_hidden", g.ffn_hidden);
  g.tie_weights = c.get_bool("model.tie_weights", g.tie_weights);
  g.grad_checkpointing = c.get_bool("model.grad_checkpointing", g.grad_checkpointing);
  g.init_std = static_cast<float>(c.get_num("model.init_std", g.init_std));
  g.ln_eps = static_cast<float>(c.get_num("model.ln_eps", g.ln_eps));
  g.rope_theta = static_cast<float>(c.get_num("model.rope_theta", g.rope_theta));
  g.qk_norm = c.get_bool("model.qk_norm", g.qk_norm);
  const std::string norm = c.get_str("model.norm", g.norm == NormKind::kRMSNorm ? "rms" : "layer");
  g.norm = (norm == "layer" || norm == "layernorm") ? NormKind::kLayerNorm : NormKind::kRMSNorm;
  const std::string pos = c.get_str("model.pos", g.pos == PosKind::kRoPE ? "rope" : "learned");
  g.pos = (pos == "learned" || pos == "absolute") ? PosKind::kLearned : PosKind::kRoPE;
  const std::string ffn = c.get_str("model.ffn", g.ffn == FFNKind::kSwiGLU ? "swiglu" : "gelu");
  g.ffn = (ffn == "gelu" || ffn == "mlp") ? FFNKind::kGeluMLP : FFNKind::kSwiGLU;
  g.linear_bias = c.get_bool("model.linear_bias", g.ffn == FFNKind::kGeluMLP && g.norm == NormKind::kLayerNorm);
  return g;
}

void GPTConfig::write_to(Config& c) const {
  c.set("model.arch_version", "2");
  c.set("model.vocab_size", std::to_string(vocab_size));
  c.set("model.n_layer", std::to_string(n_layer));
  c.set("model.n_head", std::to_string(n_head));
  c.set("model.n_kv_head", std::to_string(n_kv_head));
  c.set("model.n_embd", std::to_string(n_embd));
  c.set("model.block_size", std::to_string(block_size));
  c.set("model.ffn_hidden", std::to_string(ffn_hidden));
  c.set("model.tie_weights", tie_weights ? "true" : "false");
  c.set("model.grad_checkpointing", grad_checkpointing ? "true" : "false");
  c.set("model.init_std", std::to_string(init_std));
  c.set("model.ln_eps", std::to_string(ln_eps));
  c.set("model.norm", norm == NormKind::kRMSNorm ? "rms" : "layer");
  c.set("model.pos", pos == PosKind::kRoPE ? "rope" : "learned");
  c.set("model.ffn", ffn == FFNKind::kSwiGLU ? "swiglu" : "gelu");
  c.set("model.linear_bias", linear_bias ? "true" : "false");
  c.set("model.rope_theta", std::to_string(rope_theta));
  c.set("model.qk_norm", qk_norm ? "true" : "false");
}

std::string FreezePolicy::describe() const {
  std::ostringstream os;
  if (layernorms_only) return "norms only";
  if (last_k_blocks < 0)
    os << "all blocks";
  else
    os << "last " << last_k_blocks << " blocks";
  os << (train_embeddings ? " +emb" : "") << (train_head ? " +head" : "")
     << (train_final_ln ? " +norm_f" : "");
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

size_t KVCache::bytes() const {
  return 2 * k.size() * static_cast<size_t>(B * H * Tmax * D) * sizeof(float);
}

// =================================================================== GPT
GPT::GPT(const GPTConfig& cfg) : cfg_(cfg) {
  cfg_.validate();
  const int64_t C = cfg_.n_embd;
  const int64_t KV = cfg_.kv_heads() * cfg_.head_dim();
  const int64_t Hf = cfg_.hidden_dim();
  const bool rms = cfg_.norm == NormKind::kRMSNorm;

  register_param("tok_emb", {cfg_.vocab_size, C});
  if (cfg_.pos == PosKind::kLearned) register_param("pos_emb", {cfg_.block_size, C});
  for (int64_t l = 0; l < cfg_.n_layer; ++l) {
    const std::string p = block_prefix(l);
    register_param(p + "n1.g", {C});
    if (!rms) register_param(p + "n1.b", {C});
    register_param(p + "attn.qkv.w", {C, C + 2 * KV});
    if (cfg_.linear_bias) register_param(p + "attn.qkv.b", {C + 2 * KV});
    if (cfg_.qk_norm) {
      register_param(p + "attn.qnorm.g", {cfg_.head_dim()});
      register_param(p + "attn.knorm.g", {cfg_.head_dim()});
    }
    register_param(p + "attn.proj.w", {C, C});
    if (cfg_.linear_bias) register_param(p + "attn.proj.b", {C});
    register_param(p + "n2.g", {C});
    if (!rms) register_param(p + "n2.b", {C});
    if (cfg_.ffn == FFNKind::kSwiGLU) {
      register_param(p + "mlp.gate.w", {C, Hf});
      register_param(p + "mlp.up.w", {C, Hf});
      register_param(p + "mlp.down.w", {Hf, C});
    } else {
      register_param(p + "mlp.up.w", {C, Hf});
      if (cfg_.linear_bias) register_param(p + "mlp.up.b", {Hf});
      register_param(p + "mlp.down.w", {Hf, C});
      if (cfg_.linear_bias) register_param(p + "mlp.down.b", {C});
    }
  }
  register_param("norm_f.g", {C});
  if (!rms) register_param("norm_f.b", {C});
  if (!cfg_.tie_weights) register_param("lm_head.w", {C, cfg_.vocab_size});
  recompute_trainable();
}

std::string GPT::block_prefix(int64_t l) const { return "h." + std::to_string(l) + "."; }

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
    const bool is_norm_gain = n.size() > 2 && n.compare(n.size() - 2, 2, ".g") == 0;
    const bool is_bias = n.size() > 2 && n.compare(n.size() - 2, 2, ".b") == 0;
    if (is_norm_gain) {
      t = Tensor::full(t.shape(), 1.0f, true);
    } else if (is_bias) {
      t = Tensor::zeros(t.shape(), true);
    } else if (n.find("proj.w") != std::string::npos ||
               n.find("down.w") != std::string::npos) {
      t = Tensor::randn(t.shape(), resid_std, s, true);  // residual scaling
    } else if (n == "pos_emb") {
      t = Tensor::randn(t.shape(), cfg_.init_std * 0.5f, s, true);
    } else {
      t = Tensor::randn(t.shape(), cfg_.init_std, s, true);
    }
  }
  recompute_trainable();
}

GPT::GrowthEvent GPT::grow_depth(int64_t new_blocks, int64_t step) {
  GrowthEvent ev;
  ev.step = step;
  ev.layers_before = cfg_.n_layer;
  ev.params_before = num_params();
  const bool rms = cfg_.norm == NormKind::kRMSNorm;
  for (int64_t k = 0; k < new_blocks; ++k) {
    const std::string src = block_prefix(cfg_.n_layer - 1);
    const std::string dst = block_prefix(cfg_.n_layer);
    std::vector<std::string> suffixes = {"n1.g", "attn.qkv.w", "attn.proj.w", "n2.g"};
    if (!rms) {
      suffixes.push_back("n1.b");
      suffixes.push_back("n2.b");
    }
    if (cfg_.qk_norm) {
      suffixes.push_back("attn.qnorm.g");
      suffixes.push_back("attn.knorm.g");
    }
    if (cfg_.linear_bias) {
      suffixes.push_back("attn.qkv.b");
      suffixes.push_back("attn.proj.b");
    }
    if (cfg_.ffn == FFNKind::kSwiGLU) {
      suffixes.push_back("mlp.gate.w");
      suffixes.push_back("mlp.up.w");
      suffixes.push_back("mlp.down.w");
    } else {
      suffixes.push_back("mlp.up.w");
      suffixes.push_back("mlp.down.w");
      if (cfg_.linear_bias) {
        suffixes.push_back("mlp.up.b");
        suffixes.push_back("mlp.down.b");
      }
    }
    for (const std::string& suf : suffixes) {
      const Tensor& from = params_.at(src + suf);
      std::vector<float> data;
      from.copy_to_host(data);
      // The two residual output projections start at zero so that the fresh
      // block computes exactly identity: growth does not disturb the loss.
      if (suf == "attn.proj.w" || suf == "mlp.down.w" || suf == "attn.proj.b" ||
          suf == "mlp.down.b")
        std::fill(data.begin(), data.end(), 0.0f);
      params_.emplace(dst + suf, Tensor::from_host(from.shape(), data.data(), true));
      order_.push_back(dst + suf);
    }
    ++cfg_.n_layer;
  }
  recompute_trainable();
  ev.layers_after = cfg_.n_layer;
  ev.params_after = num_params();
  growth_.push_back(ev);
  return ev;
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
    } else if (n.rfind("norm_f.", 0) == 0) {
      ok = policy_.train_final_ln;
    } else if (n.rfind("h.", 0) == 0) {
      const size_t dot = n.find('.', 2);
      const int64_t l = std::stoll(n.substr(2, dot - 2));
      ok = (policy_.last_k_blocks < 0) || (l >= cfg_.n_layer - policy_.last_k_blocks);
      if (ok && policy_.layernorms_only)
        ok = n.find(".n1.") != std::string::npos || n.find(".n2.") != std::string::npos;
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
  // Guard against silently dropping weights: a checkpoint that carries more
  // parameters than this model has (for instance after progressive growth) is a
  // configuration mistake, not something to ignore.
  if (s.names.size() > order_.size()) {
    size_t missing = 0;
    for (const std::string& n : s.names)
      if (params_.find(n) == params_.end()) ++missing;
    if (missing)
      std::fprintf(stderr,
                   "warning: checkpoint has %zu parameters this model does not "
                   "(architecture mismatch: %zu vs %zu tensors)\n",
                   missing, s.names.size(), order_.size());
  }
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
  if (static_cast<int64_t>(ids.size()) != B * T)
    throw std::runtime_error("forward: ids size != B*T");
  if (opt.pos_offset + T > cfg_.block_size && cfg_.pos == PosKind::kLearned)
    throw std::runtime_error("forward: sequence longer than block_size");

  const bool use_ckpt = opt.checkpointing && grad_enabled() && !opt.cache && !opt.capture;
  const bool rms = cfg_.norm == NormKind::kRMSNorm;

  BlockShape shape;
  shape.B = B;
  shape.Tq = T;
  shape.H = cfg_.n_head;
  shape.Hkv = cfg_.kv_heads();
  shape.Dh = cfg_.head_dim();
  shape.pos_offset = opt.pos_offset;
  shape.eps = cfg_.ln_eps;
  shape.rope_theta = cfg_.rope_theta;
  shape.norm = cfg_.norm;
  shape.pos = cfg_.pos;
  shape.ffn = cfg_.ffn;
  shape.qk_norm = cfg_.qk_norm;

  Tensor x = embedding(params_.at("tok_emb"), ids, B, T);
  if (cfg_.pos == PosKind::kLearned) {
    Tensor pos = params_.at("pos_emb").slice(0, opt.pos_offset, opt.pos_offset + T);
    x = x.add_bias(pos);
  }

  if (opt.capture) {
    opt.capture->clear();
    opt.capture->n_layer = cfg_.n_layer;
  }
  for (int64_t l = 0; l < cfg_.n_layer; ++l) {
    const std::string p = block_prefix(l);
    BlockWeights w;
    w.n1g = params_.at(p + "n1.g");
    if (!rms) w.n1b = params_.at(p + "n1.b");
    w.qkvw = params_.at(p + "attn.qkv.w");
    if (cfg_.linear_bias) w.qkvb = params_.at(p + "attn.qkv.b");
    if (cfg_.qk_norm) {
      w.qn = params_.at(p + "attn.qnorm.g");
      w.kn = params_.at(p + "attn.knorm.g");
    }
    w.pw = params_.at(p + "attn.proj.w");
    if (cfg_.linear_bias) w.pb = params_.at(p + "attn.proj.b");
    w.n2g = params_.at(p + "n2.g");
    if (!rms) w.n2b = params_.at(p + "n2.b");
    w.fcw = params_.at(p + "mlp.up.w");
    if (cfg_.ffn == FFNKind::kSwiGLU) {
      w.gatew = params_.at(p + "mlp.gate.w");
    } else if (cfg_.linear_bias) {
      w.fcb = params_.at(p + "mlp.up.b");
    }
    w.mw = params_.at(p + "mlp.down.w");
    if (cfg_.ffn == FFNKind::kGeluMLP && cfg_.linear_bias) w.mb = params_.at(p + "mlp.down.b");

    if (use_ckpt) {
      // Captured *by value*: the body is replayed during the backward pass.
      auto body = [w, shape](const Tensor& inp) {
        return block_forward(w, inp, shape, nullptr, 0, nullptr);
      };
      x = checkpoint(body, x);
    } else {
      x = block_forward(w, x, shape, opt.cache, l, opt.capture);
    }
  }
  x = apply_norm(x, params_.at("norm_f.g"),
                 rms ? Tensor() : params_.at("norm_f.b"), cfg_.norm, cfg_.ln_eps);
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

float GPT::eval_loss(const std::vector<int32_t>& ids, const std::vector<int32_t>& targets,
                     int64_t B, int64_t T) {
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
  const int64_t T =
      std::min<int64_t>(static_cast<int64_t>(ids.size()) - 1, cfg_.block_size);
  std::vector<int32_t> inp(ids.begin(), ids.begin() + T);
  std::vector<int32_t> tgt(ids.begin() + 1, ids.begin() + T + 1);
  Tensor logits = forward(inp, 1, T, ForwardOptions());
  Tensor lp = seq_logprob(logits, tgt, -100);
  if (ntok) *ntok = T;
  return lp.host_ptr()[0];
}

// ----------------------------------------------------------------- sampling
std::vector<int32_t> GPT::generate(const std::vector<int32_t>& prompt, const GenOptions& opt,
                                   const std::function<bool(const GenStep&)>& on_step,
                                   AttentionCapture* capture) {
  NoGradGuard ng;
  const int64_t Hkv = cfg_.kv_heads(), Dh = cfg_.head_dim();
  const int32_t V = cfg_.vocab_size;
  std::vector<int32_t> out;
  if (prompt.empty()) return out;

  const int64_t max_ctx = cfg_.block_size;
  const int64_t keep = std::max<int64_t>(
      1, std::min<int64_t>(static_cast<int64_t>(prompt.size()),
                           max_ctx - std::max(1, opt.max_new_tokens)));
  std::vector<int32_t> ctx(prompt.end() - keep, prompt.end());

  KVCache cache;
  cache.reset(1, Hkv, Dh, max_ctx, cfg_.n_layer);

  Rng rng(opt.seed ? opt.seed : 0x5eed1234u);
  std::vector<int32_t> history = ctx;
  std::vector<float> row(static_cast<size_t>(V));
  std::vector<std::pair<float, int32_t>> cand;

  ForwardOptions fo;
  fo.cache = &cache;
  fo.capture = capture;
  Tensor logits = forward(ctx, 1, static_cast<int64_t>(ctx.size()), fo);

  for (int step = 0; step < opt.max_new_tokens; ++step) {
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

    const float mx = cand[0].first;
    double sum = 0.0;
    for (auto& c : cand) {
      c.first = std::exp(c.first - mx);
      sum += c.first;
    }
    for (auto& c : cand) c.first = static_cast<float>(c.first / sum);

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
