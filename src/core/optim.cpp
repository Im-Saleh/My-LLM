// SPDX-License-Identifier: Apache-2.0
#include "core/optim.h"

#include <algorithm>
#include <cmath>

namespace slm {

void AdamW::set_params(std::vector<Tensor*> params,
                       const std::vector<std::string>& names) {
  params_ = std::move(params);
  m_.clear();
  v_.clear();
  decay_.clear();
  m_.reserve(params_.size());
  v_.reserve(params_.size());
  decay_.reserve(params_.size());
  for (size_t i = 0; i < params_.size(); ++i) {
    m_.push_back(Tensor::zeros(params_[i]->shape()));
    v_.push_back(Tensor::zeros(params_[i]->shape()));
    const bool is_matrix = params_[i]->dim() >= 2;
    const std::string n = i < names.size() ? names[i] : std::string();
    const bool is_embedding = n == "tok_emb" || n == "pos_emb";
    decay_.push_back((is_matrix && !is_embedding) ? 1.0f : 0.0f);
  }
  t_ = 0;
}

void AdamW::reset_state() {
  for (size_t i = 0; i < m_.size(); ++i) {
    m_[i] = Tensor::zeros(m_[i].shape());
    v_[i] = Tensor::zeros(v_[i].shape());
  }
  t_ = 0;
}

size_t AdamW::state_bytes() const {
  size_t n = 0;
  for (const Tensor& t : m_) n += static_cast<size_t>(t.numel()) * sizeof(float) * 2;
  return n;
}

float AdamW::step(float lr_override, float skip_above, bool* skipped) {
  const float norm = grads_global_norm(params_);
  if (skipped) *skipped = false;
  if (skip_above > 0.0f && norm > skip_above) {
    if (skipped) *skipped = true;
    return norm;  // one bad batch never reaches the weights
  }
  if (cfg_.grad_clip > 0.0f && norm > cfg_.grad_clip && norm > 0.0f)
    grads_scale(params_, cfg_.grad_clip / norm);
  ++t_;
  const float lr = lr_override >= 0.0f ? lr_override : cfg_.lr;
  for (size_t i = 0; i < params_.size(); ++i) {
    if (!params_[i]->has_grad()) continue;
    adamw_apply(*params_[i], m_[i], v_[i], lr, cfg_.beta1, cfg_.beta2, cfg_.eps,
                cfg_.weight_decay * decay_[i], t_);
  }
  return norm;
}

float lr_schedule_wsd(int64_t step, float base_lr, int64_t warmup, int64_t total,
                      float decay_frac, float min_ratio) {
  if (step < warmup && warmup > 0)
    return base_lr * static_cast<float>(step + 1) / static_cast<float>(warmup);
  const int64_t decay_steps =
      std::max<int64_t>(1, static_cast<int64_t>(static_cast<double>(total) * decay_frac));
  const int64_t decay_start = std::max<int64_t>(warmup, total - decay_steps);
  if (step < decay_start) return base_lr;
  const float p = static_cast<float>(step - decay_start) / static_cast<float>(decay_steps);
  const float t = std::min(1.0f, std::max(0.0f, p));
  // 1-sqrt decay: spends more time at a useful LR than a linear ramp
  return base_lr * std::max(min_ratio, 1.0f - std::sqrt(t));
}

float lr_schedule(int64_t step, float base_lr, int64_t warmup, int64_t total,
                  float min_ratio) {
  if (step < warmup && warmup > 0)
    return base_lr * static_cast<float>(step + 1) / static_cast<float>(warmup);
  if (total <= warmup) return base_lr;
  const float p = static_cast<float>(step - warmup) / static_cast<float>(total - warmup);
  const float t = std::min(1.0f, std::max(0.0f, p));
  const float cos_out = 0.5f * (1.0f + std::cos(3.14159265358979f * t));
  return base_lr * (min_ratio + (1.0f - min_ratio) * cos_out);
}

}  // namespace slm
