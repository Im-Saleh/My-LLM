// SPDX-License-Identifier: Apache-2.0
//
// Backend-agnostic tensor / autograd facade.
//
// Two implementations exist and are selected at compile time:
//   * tensor_native.cpp  -> zero-dependency reverse-mode autodiff on CPU
//                           (default, SLM_BACKEND_NATIVE)
//   * tensor_torch.cpp   -> libtorch (PyTorch C++ API), CPU or CUDA
//                           (-DSLM_WITH_LIBTORCH=ON, SLM_BACKEND_TORCH)
//
// The whole model / training / coordinator stack is written against *this*
// header only, so both backends share one single model implementation.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace slm {

using Shape = std::vector<int64_t>;

int64_t numel_of(const Shape& s);
std::string shape_str(const Shape& s);

namespace detail {
struct TensorImpl;  // backend private
}

class Tensor {
 public:
  Tensor();
  ~Tensor();
  Tensor(const Tensor&);
  Tensor(Tensor&&) noexcept;
  Tensor& operator=(const Tensor&);
  Tensor& operator=(Tensor&&) noexcept;

  explicit Tensor(std::shared_ptr<detail::TensorImpl> impl);

  bool defined() const { return static_cast<bool>(impl_); }

  // ---------------------------------------------------------------- factories
  static Tensor zeros(const Shape& s, bool requires_grad = false);
  static Tensor full(const Shape& s, float value, bool requires_grad = false);
  static Tensor from_host(const Shape& s, const float* data,
                          bool requires_grad = false);
  static Tensor randn(const Shape& s, float stddev, uint64_t seed,
                      bool requires_grad = false);
  static Tensor scalar(float v, bool requires_grad = false);

  // ------------------------------------------------------------------- shape
  Shape shape() const;
  int dim() const;
  int64_t size(int d) const;
  int64_t numel() const;

  bool requires_grad() const;
  void set_requires_grad(bool v);

  // ------------------------------------------------- host (CPU) data access
  // For the torch/CUDA backend these synchronise to host memory; parameters
  // always live on the compute device and are only copied for checkpointing,
  // the coordinator and the GUI.
  void copy_to_host(std::vector<float>& out) const;
  void copy_from_host(const float* src, int64_t n);
  void copy_grad_to_host(std::vector<float>& out) const;  // zeros if no grad
  void add_to_host_data(const float* delta, int64_t n, float alpha);
  bool has_grad() const;
  void zero_grad();
  float item() const;

  // Direct pointer access. Only valid for CPU tensors (always true for the
  // native backend); returns nullptr otherwise.
  const float* host_ptr() const;
  float* mutable_host_ptr();

  // -------------------------------------------------------------- autograd
  Tensor reshape(const Shape& s) const;
  Tensor transpose(int d0, int d1) const;
  Tensor slice(int d, int64_t start, int64_t end) const;
  Tensor detach() const;
  Tensor clone_leaf(bool requires_grad) const;  // fresh leaf with same values

  Tensor scale(float f) const;
  Tensor add(const Tensor& o) const;       // identical shapes
  Tensor mul(const Tensor& o) const;       // identical shapes, elementwise
  Tensor add_bias(const Tensor& b) const;  // [..., C] + [C]
  Tensor matmul(const Tensor& o) const;    // batched over leading dims
  Tensor gelu() const;
  Tensor softmax_last() const;
  Tensor layernorm(const Tensor& gain, const Tensor& bias, float eps) const;
  Tensor causal_mask() const;  // [..., T, T]: j > i -> -inf
  Tensor sum_all() const;
  Tensor mean_all() const;

  Tensor operator+(const Tensor& o) const { return add(o); }

  // Seeded with 1.0; tensor must be a scalar.
  void backward();

  const std::shared_ptr<detail::TensorImpl>& impl() const { return impl_; }

 private:
  std::shared_ptr<detail::TensorImpl> impl_;
};

// -------------------------------------------------------------------- ops
// weight: [vocab, C]; ids: B*T token ids -> [B, T, C]
Tensor embedding(const Tensor& weight, const std::vector<int32_t>& ids,
                 int64_t B, int64_t T);

// x: [..., in]; w: [in, out]; bias: [out] or nullptr -> [..., out]
Tensor linear(const Tensor& x, const Tensor& w, const Tensor* bias);

// logits: [N, V]; targets: N ids (ignore_index skips a position).
// Returns a scalar tensor; *loss_out (optional) receives the value and
// *ntok_out the number of scored tokens.
Tensor cross_entropy(const Tensor& logits, const std::vector<int32_t>& targets,
                     int32_t ignore_index, float* loss_out, int64_t* ntok_out);

// logits: [B, T, V]; targets: B*T ids -> [B]: sum of log p(target) per row.
Tensor seq_logprob(const Tensor& logits, const std::vector<int32_t>& targets,
                   int32_t ignore_index);

Tensor logsigmoid(const Tensor& x);

// Gradient checkpointing: fn is run without a graph during forward and
// recomputed with a graph during backward. Parameter gradients used inside fn
// accumulate correctly.
Tensor checkpoint(const std::function<Tensor(const Tensor&)>& fn,
                  const Tensor& x);

// ------------------------------------------------- raw (no-autograd) helpers
// Used by the optimiser and the coordinator; they never touch the graph.
void adamw_apply(Tensor& p, Tensor& m, Tensor& v, float lr, float beta1,
                 float beta2, float eps, float weight_decay, int64_t step);
float grads_global_norm(const std::vector<Tensor*>& params);
void grads_scale(const std::vector<Tensor*>& params, float s);

// ------------------------------------------------------------------ context
class NoGradGuard {
 public:
  NoGradGuard();
  ~NoGradGuard();
  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  bool prev_;
};
bool grad_enabled();

// ----------------------------------------------------------------- backend
const char* backend_name();
void backend_init(int threads, bool prefer_cuda);
bool backend_on_gpu();
// Bytes currently held by live tensor storages (native backend only, else 0).
size_t backend_allocated_bytes();

}  // namespace slm
