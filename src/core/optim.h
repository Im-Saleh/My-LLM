// SPDX-License-Identifier: Apache-2.0
// AdamW + learning-rate schedules + global gradient clipping.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/tensor.h"

namespace slm {

struct AdamWConfig {
  float lr = 3e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.95f;
  float eps = 1e-8f;
  float weight_decay = 0.1f;
  float grad_clip = 1.0f;  // <= 0 disables
};

class AdamW {
 public:
  explicit AdamW(const AdamWConfig& cfg = {}) : cfg_(cfg) {}

  AdamWConfig& config() { return cfg_; }
  const AdamWConfig& config() const { return cfg_; }

  // `names` is used to decide which tensors get weight decay (2-D weights do,
  // biases / layernorm gains / embeddings do not).
  void set_params(std::vector<Tensor*> params, const std::vector<std::string>& names);
  void reset_state();

  // Applies clipping then one AdamW update. Returns the pre-clip gradient norm.
  // When `skip_above > 0` and the gradient norm exceeds it, the update is
  // *skipped* entirely (a loss spike caused by one bad batch cannot damage the
  // weights) and `*skipped` is set.
  float step(float lr_override = -1.0f, float skip_above = 0.0f,
             bool* skipped = nullptr);

  int64_t num_steps() const { return t_; }
  size_t state_bytes() const;

 private:
  AdamWConfig cfg_;
  std::vector<Tensor*> params_;
  std::vector<Tensor> m_, v_;
  std::vector<float> decay_;
  int64_t t_ = 0;
};

// Linear warmup + cosine decay down to `min_ratio * base_lr`.
float lr_schedule(int64_t step, float base_lr, int64_t warmup, int64_t total,
                  float min_ratio = 0.1f);

// Warmup-Stable-Decay: warmup, then a long constant plateau, then a short decay
// to ~0 over the last `decay_frac` of the run.  This is what makes "continue
// training later" practical: any checkpoint on the plateau is a valid starting
// point, unlike a cosine schedule which bakes the total step count in.
float lr_schedule_wsd(int64_t step, float base_lr, int64_t warmup, int64_t total,
                      float decay_frac = 0.2f, float min_ratio = 0.02f);

}  // namespace slm
