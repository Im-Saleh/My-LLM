// SPDX-License-Identifier: Apache-2.0
//
// Decoder-only transformer (GPT-2 / nanoGPT style), written against the
// backend-agnostic tensor facade.
//
//   token emb + learned pos emb
//   N x [ pre-LN -> causal MHSA -> residual ; pre-LN -> MLP(4x, GELU) -> residual ]
//   final LN -> lm head (optionally tied to the token embedding)
//
// Extras that the self-training system depends on:
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

struct GPTConfig {
  int32_t vocab_size = 4096;
  int64_t n_layer = 6;
  int64_t n_head = 6;
  int64_t n_embd = 384;
  int64_t block_size = 256;
  bool tie_weights = true;
  bool grad_checkpointing = false;
  float init_std = 0.02f;
  float ln_eps = 1e-5f;

  int64_t param_count() const;
  int64_t head_dim() const { return n_embd / n_head; }
  std::string describe() const;
  static GPTConfig from_config(const Config& c);
  void write_to(Config& c) const;
  void validate() const;
};

// Which parameters a given trainer is allowed to move.
struct FreezePolicy {
  // -1 == all layers trainable, otherwise only the last `last_k_blocks`.
  int last_k_blocks = -1;
  bool train_embeddings = true;
  bool train_head = true;
  bool train_final_ln = true;
  bool layernorms_only = false;  // strongest restriction: only LN gains/biases
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

// Incremental decoding state.
struct KVCache {
  int64_t B = 0, H = 0, D = 0, Tmax = 0, T = 0;
  std::vector<std::vector<float>> k, v;  // per layer, [B, H, Tmax, D]
  void reset(int64_t B, int64_t H, int64_t D, int64_t Tmax, int64_t n_layer);
  void clear() { T = 0; }
};

// Options for a single forward pass.
struct ForwardOptions {
  int64_t pos_offset = 0;
  KVCache* cache = nullptr;
  AttentionCapture* capture = nullptr;
  bool checkpointing = false;
};

// Sampling configuration.
struct GenOptions {
  int max_new_tokens = 64;
  float temperature = 0.9f;
  int top_k = 40;
  float top_p = 0.95f;
  float repetition_penalty = 1.08f;
  uint64_t seed = 0;
  bool stop_on_eot = true;
};

// One decoding step, streamed to the caller (GUI / CLI).
struct GenStep {
  int32_t token = 0;
  float logprob = 0.0f;
  // Highest probability candidates of this step (for the GUI panel).
  std::vector<std::pair<int32_t, float>> top;
};

class GPT {
 public:
  explicit GPT(const GPTConfig& cfg);

  const GPTConfig& config() const { return cfg_; }
  GPTConfig& mutable_config() { return cfg_; }

  void init_weights(uint64_t seed);

  // ---------------------------------------------------------------- params
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

  // --------------------------------------------------------------- forward
  // ids: B*T token ids -> logits [B,T,V]
  Tensor forward(const std::vector<int32_t>& ids, int64_t B, int64_t T,
                 const ForwardOptions& opt = ForwardOptions());

  // Cross-entropy over `targets` (use -100 to ignore a position).
  Tensor loss(const std::vector<int32_t>& ids, const std::vector<int32_t>& targets,
              int64_t B, int64_t T, float* loss_out = nullptr,
              int64_t* ntok_out = nullptr, bool checkpointing_default = true);

  // Mean cross-entropy without building a graph (validation / gating).
  float eval_loss(const std::vector<int32_t>& ids, const std::vector<int32_t>& targets,
                  int64_t B, int64_t T);

  // --------------------------------------------------------------- sampling
  // Returns the generated continuation (prompt excluded).  `on_step` may
  // return false to abort generation (used by Emergency Stop).
  std::vector<int32_t> generate(const std::vector<int32_t>& prompt,
                                const GenOptions& opt,
                                const std::function<bool(const GenStep&)>& on_step = nullptr,
                                AttentionCapture* capture = nullptr);

  // Sum of log p(target) for a full sequence, no graph (used for quality
  // filtering and reward computation).
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
};

}  // namespace slm
