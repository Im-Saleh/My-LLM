// SPDX-License-Identifier: Apache-2.0
//
// Decoder-only transformer, written against the backend-agnostic tensor facade.
//
// Two architecture families are supported by the same code, selected in the
// config file:
//
//   legacy  (GPT-2 / nanoGPT)         modern  (Llama / Qwen / DeepSeek style)
//   ------------------------------    ---------------------------------------
//   learned positional embeddings     RoPE (rotary), extrapolates, no table
//   LayerNorm (gain + bias)           RMSNorm (gain only, cheaper, stabler)
//   MLP 4x with GELU                  SwiGLU (gate * up, 8/3x hidden)
//   multi-head attention              grouped-query attention (n_kv_head)
//   biases on every linear            no biases
//
// The modern stack is the default because it is what every current model of
// this shape uses: better quality per parameter, a much smaller KV cache (GQA)
// and no context-length ceiling baked into a weight matrix.
//
// Extras the self-training system depends on:
//   * per-parameter freezing (partial fine-tuning),
//   * optional gradient checkpointing per block,
//   * a KV cache so interactive generation is O(1) per token,
//   * attention probability capture for the GUI heat map.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/params.h"
#include "core/tensor.h"

namespace slm {

class Config;

enum class NormKind : uint8_t { kLayerNorm = 0, kRMSNorm = 1 };
enum class PosKind : uint8_t { kLearned = 0, kRoPE = 1 };
enum class FFNKind : uint8_t { kGeluMLP = 0, kSwiGLU = 1 };

struct GPTConfig {
  int32_t vocab_size = 4096;
  int64_t n_layer = 6;
  int64_t n_head = 6;
  int64_t n_kv_head = 0;  // 0 or == n_head -> MHA, otherwise GQA
  int64_t n_embd = 384;
  int64_t block_size = 256;
  int64_t ffn_hidden = 0;  // 0 -> automatic (4C for GELU, ~8/3 C for SwiGLU)
  bool tie_weights = true;
  bool grad_checkpointing = false;
  float init_std = 0.02f;
  float ln_eps = 1e-5f;
  NormKind norm = NormKind::kRMSNorm;
  PosKind pos = PosKind::kRoPE;
  FFNKind ffn = FFNKind::kSwiGLU;
  bool linear_bias = false;  // modern stacks drop biases
  // RMSNorm on q and k before RoPE.  OLMo 2 introduced this to stop attention
  // logits from growing without bound; it is nearly free and removes the most
  // common cause of loss spikes in long runs.
  bool qk_norm = true;
  float rope_theta = 10000.0f;

  int64_t param_count() const;
  int64_t head_dim() const { return n_embd / n_head; }
  int64_t kv_heads() const { return n_kv_head > 0 ? n_kv_head : n_head; }
  int64_t kv_repeat() const { return n_head / kv_heads(); }
  int64_t hidden_dim() const;
  std::string describe() const;
  std::string arch_summary() const;
  static GPTConfig from_config(const Config& c);
  void write_to(Config& c) const;
  void validate() const;
};

// Which parameters a given trainer is allowed to move.
struct FreezePolicy {
  int last_k_blocks = -1;  // -1 == every block
  bool train_embeddings = true;
  bool train_head = true;
  bool train_final_ln = true;
  bool layernorms_only = false;
  static FreezePolicy all() { return FreezePolicy{}; }
  static FreezePolicy last_k(int k, bool head = true) {
    FreezePolicy p;
    p.last_k_blocks = k;
    p.train_embeddings = false;
    p.train_head = head;
    return p;
  }
  std::string describe() const;
};

// Attention probabilities of the most recent forward pass, [layer][head*Tq*Tk].
struct AttentionCapture {
  int64_t n_layer = 0, n_head = 0, Tq = 0, Tk = 0;
  std::vector<std::vector<float>> layers;
  void clear() { layers.clear(); }
  const float* at(int64_t layer, int64_t head) const {
    return layers[static_cast<size_t>(layer)].data() + head * Tq * Tk;
  }
};

// Incremental decoding state (sized for the *kv* heads, which is what makes
// GQA cheap at inference time).
struct KVCache {
  int64_t B = 0, H = 0, D = 0, Tmax = 0, T = 0;
  std::vector<std::vector<float>> k, v;  // per layer, [B, H, Tmax, D]
  void reset(int64_t B, int64_t H, int64_t D, int64_t Tmax, int64_t n_layer);
  void clear() { T = 0; }
  size_t bytes() const;
};

struct ForwardOptions {
  int64_t pos_offset = 0;
  KVCache* cache = nullptr;
  AttentionCapture* capture = nullptr;
  bool checkpointing = false;
};

struct GenOptions {
  int max_new_tokens = 64;
  float temperature = 0.9f;
  int top_k = 40;
  float top_p = 0.95f;
  float repetition_penalty = 1.08f;
  uint64_t seed = 0;
  bool stop_on_eot = true;
};

struct GenStep {
  int32_t token = 0;
  float logprob = 0.0f;
  std::vector<std::pair<int32_t, float>> top;
};

class GPT {
 public:
  explicit GPT(const GPTConfig& cfg);

  const GPTConfig& config() const { return cfg_; }
  GPTConfig& mutable_config() { return cfg_; }

  void init_weights(uint64_t seed);

  // ---------------------------------------------------------------- growth
  // Progressive depth growth (function-preserving "stacking"): the top block is
  // duplicated and its two residual output projections are zeroed, so the new
  // block starts as an exact identity and the loss does not jump.  Parameters
  // therefore *increase while the model trains*, which is both a real training
  // speed-up (early steps run on a small model) and visible in the dashboard.
  //   Gong et al., "Efficient Training of BERT by Progressively Stacking"
  //   Chen et al., "Net2Net"
  struct GrowthEvent {
    int64_t step = 0;
    int64_t layers_before = 0, layers_after = 0;
    int64_t params_before = 0, params_after = 0;
  };
  GrowthEvent grow_depth(int64_t new_blocks, int64_t step);
  const std::vector<GrowthEvent>& growth_events() const { return growth_; }

  const std::vector<std::string>& param_names() const { return order_; }
  Tensor* param(const std::string& name);
  const Tensor* param(const std::string& name) const;
  int64_t num_params() const;

  void set_freeze_policy(const FreezePolicy& p);
  const FreezePolicy& freeze_policy() const { return policy_; }
  std::vector<Tensor*> trainable_params();
  const std::vector<std::string>& trainable_names() const { return trainable_; }
  int64_t num_trainable() const;
  void zero_grad();

  ParamStorePtr snapshot() const;
  void load_params(const ParamStore& s);

  Tensor forward(const std::vector<int32_t>& ids, int64_t B, int64_t T,
                 const ForwardOptions& opt = ForwardOptions());

  Tensor loss(const std::vector<int32_t>& ids, const std::vector<int32_t>& targets,
              int64_t B, int64_t T, float* loss_out = nullptr,
              int64_t* ntok_out = nullptr, bool checkpointing_default = true);

  float eval_loss(const std::vector<int32_t>& ids, const std::vector<int32_t>& targets,
                  int64_t B, int64_t T);

  std::vector<int32_t> generate(const std::vector<int32_t>& prompt, const GenOptions& opt,
                                const std::function<bool(const GenStep&)>& on_step = nullptr,
                                AttentionCapture* capture = nullptr);

  float sequence_logprob(const std::vector<int32_t>& ids, int64_t* ntok = nullptr);

 private:
  void register_param(const std::string& name, const Shape& s);
  void recompute_trainable();
  std::string block_prefix(int64_t l) const;

  GPTConfig cfg_;
  FreezePolicy policy_;
  std::unordered_map<std::string, Tensor> params_;
  std::vector<std::string> order_;
  std::vector<std::string> trainable_;
  std::vector<GrowthEvent> growth_;
};

}  // namespace slm
