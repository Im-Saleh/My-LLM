// SPDX-License-Identifier: Apache-2.0
//
// libtorch (PyTorch C++ API) implementation of the tensor facade.
//
// Enabled with -DSLM_WITH_LIBTORCH=ON, in which case *the same* model,
// coordinator and trainer sources run on ATen kernels (oneDNN/MKL on CPU, cuDNN
// on CUDA) instead of the bundled engine.  Nothing above this file changes.
//
// Notes
//  * Parameters live on the compute device; the host accessors synchronise
//    explicitly.  That is why the coordinator only ever talks to ParamStore.
//  * Gradient checkpointing is implemented with a custom autograd Function plus
//    a small closure registry, because unlike Python there is no
//    torch.utils.checkpoint in the C++ API.  The registry is emptied by the
//    thread that finishes a top-level backward pass.
#include <torch/torch.h>

#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "core/tensor.h"

namespace slm {
namespace {

[[noreturn]] void fail(const std::string& m) {
  throw std::runtime_error("slm/tensor(torch): " + m);
}
#define SLM_CHECK(cond, msg) \
  do {                       \
    if (!(cond)) fail(msg);  \
  } while (0)

torch::Device g_device(torch::kCPU);
int g_threads = 0;

torch::TensorOptions opts() {
  return torch::TensorOptions().dtype(torch::kFloat32).device(g_device);
}

std::vector<int64_t> to_sizes(const Shape& s) { return s; }

}  // namespace

namespace detail {
struct TensorImpl {
  torch::Tensor t;
};
}  // namespace detail

using detail::TensorImpl;
using ImplPtr = std::shared_ptr<TensorImpl>;

namespace {

ImplPtr wrap(torch::Tensor t) {
  auto p = std::make_shared<TensorImpl>();
  p->t = std::move(t);
  return p;
}

const torch::Tensor& raw(const Tensor& x) {
  SLM_CHECK(x.defined(), "operation on an undefined tensor");
  return x.impl()->t;
}

// ------------------------------------------------------- checkpoint registry
struct CkptEntry {
  std::function<Tensor(const Tensor&)> fn;
  std::thread::id owner;
};
std::mutex g_ckpt_m;
std::unordered_map<int64_t, CkptEntry> g_ckpt;
int64_t g_ckpt_next = 1;

int64_t ckpt_register(const std::function<Tensor(const Tensor&)>& fn) {
  std::lock_guard<std::mutex> g(g_ckpt_m);
  const int64_t id = g_ckpt_next++;
  g_ckpt[id] = CkptEntry{fn, std::this_thread::get_id()};
  return id;
}
std::function<Tensor(const Tensor&)> ckpt_get(int64_t id) {
  std::lock_guard<std::mutex> g(g_ckpt_m);
  auto it = g_ckpt.find(id);
  if (it == g_ckpt.end()) fail("checkpoint closure expired");
  return it->second.fn;
}
void ckpt_release_own() {
  const std::thread::id me = std::this_thread::get_id();
  std::lock_guard<std::mutex> g(g_ckpt_m);
  for (auto it = g_ckpt.begin(); it != g_ckpt.end();)
    it = (it->second.owner == me) ? g_ckpt.erase(it) : std::next(it);
}

class CheckpointFn : public torch::autograd::Function<CheckpointFn> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x,
                               int64_t id) {
    ctx->saved_data["id"] = id;
    ctx->save_for_backward({x});
    torch::NoGradGuard ng;
    Tensor out = ckpt_get(id)(Tensor(wrap(x.detach())));
    return raw(out).detach();
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list gout) {
    const int64_t id = ctx->saved_data["id"].toInt();
    torch::Tensor x = ctx->get_saved_variables()[0].detach().set_requires_grad(true);
    torch::Tensor y;
    {
      torch::AutoGradMode enable(true);
      y = raw(ckpt_get(id)(Tensor(wrap(x))));
    }
    // A reentrant backward: gradients also accumulate into the parameters used
    // inside the block, which is exactly what checkpointing needs.
    torch::autograd::backward({y}, {gout[0]});
    return {x.grad(), torch::Tensor()};
  }
};

}  // namespace

// =========================================================================
Tensor::Tensor() = default;
Tensor::~Tensor() = default;
Tensor::Tensor(const Tensor&) = default;
Tensor::Tensor(Tensor&&) noexcept = default;
Tensor& Tensor::operator=(const Tensor&) = default;
Tensor& Tensor::operator=(Tensor&&) noexcept = default;
Tensor::Tensor(std::shared_ptr<detail::TensorImpl> impl) : impl_(std::move(impl)) {}

Tensor Tensor::zeros(const Shape& s, bool requires_grad) {
  return Tensor(wrap(torch::zeros(to_sizes(s), opts()).set_requires_grad(requires_grad)));
}
Tensor Tensor::full(const Shape& s, float v, bool requires_grad) {
  return Tensor(wrap(torch::full(to_sizes(s), v, opts()).set_requires_grad(requires_grad)));
}
Tensor Tensor::from_host(const Shape& s, const float* data, bool requires_grad) {
  torch::Tensor t = torch::zeros(to_sizes(s), opts());
  if (data) {
    torch::Tensor host = torch::from_blob(const_cast<float*>(data), to_sizes(s),
                                          torch::TensorOptions().dtype(torch::kFloat32));
    t.copy_(host);
  }
  return Tensor(wrap(t.set_requires_grad(requires_grad)));
}
Tensor Tensor::randn(const Shape& s, float stddev, uint64_t seed, bool requires_grad) {
  torch::Generator gen = torch::make_generator<torch::CPUGeneratorImpl>();
  gen.set_current_seed(seed);
  torch::Tensor host = torch::randn(to_sizes(s), gen,
                                    torch::TensorOptions().dtype(torch::kFloat32)) * stddev;
  return Tensor(wrap(host.to(g_device).set_requires_grad(requires_grad)));
}
Tensor Tensor::scalar(float v, bool requires_grad) { return full({}, v, requires_grad); }

Shape Tensor::shape() const { return impl_ ? impl_->t.sizes().vec() : Shape{}; }
int Tensor::dim() const { return impl_ ? static_cast<int>(impl_->t.dim()) : 0; }
int64_t Tensor::size(int d) const { return raw(*this).size(d); }
int64_t Tensor::numel() const { return impl_ ? impl_->t.numel() : 0; }
bool Tensor::requires_grad() const { return impl_ && impl_->t.requires_grad(); }
void Tensor::set_requires_grad(bool v) {
  SLM_CHECK(impl_ != nullptr, "set_requires_grad on undefined tensor");
  impl_->t.set_requires_grad(v);
  if (!v && impl_->t.grad().defined()) impl_->t.mutable_grad().reset();
}

void Tensor::copy_to_host(std::vector<float>& out) const {
  const torch::Tensor c = raw(*this).detach().to(torch::kCPU).contiguous();
  out.assign(c.data_ptr<float>(), c.data_ptr<float>() + c.numel());
}
void Tensor::copy_from_host(const float* src, int64_t n) {
  SLM_CHECK(n == numel(), "copy_from_host size mismatch");
  torch::NoGradGuard ng;
  torch::Tensor host = torch::from_blob(const_cast<float*>(src), {n},
                                        torch::TensorOptions().dtype(torch::kFloat32));
  impl_->t.detach().view({n}).copy_(host);
}
void Tensor::copy_grad_to_host(std::vector<float>& out) const {
  const torch::Tensor g = raw(*this).grad();
  if (!g.defined()) {
    out.assign(static_cast<size_t>(numel()), 0.0f);
    return;
  }
  const torch::Tensor c = g.detach().to(torch::kCPU).contiguous();
  out.assign(c.data_ptr<float>(), c.data_ptr<float>() + c.numel());
}
void Tensor::add_to_host_data(const float* delta, int64_t n, float alpha) {
  SLM_CHECK(n == numel(), "add_to_host_data size mismatch");
  torch::NoGradGuard ng;
  torch::Tensor host = torch::from_blob(const_cast<float*>(delta), {n},
                                        torch::TensorOptions().dtype(torch::kFloat32));
  impl_->t.detach().view({n}).add_(host.to(g_device), alpha);
}
bool Tensor::has_grad() const { return impl_ && impl_->t.grad().defined(); }
void Tensor::zero_grad() {
  if (impl_ && impl_->t.grad().defined()) impl_->t.mutable_grad().zero_();
}
float Tensor::item() const { return raw(*this).item<float>(); }
const float* Tensor::host_ptr() const {
  if (!impl_ || !impl_->t.device().is_cpu() || !impl_->t.is_contiguous()) return nullptr;
  return impl_->t.data_ptr<float>();
}
float* Tensor::mutable_host_ptr() {
  if (!impl_ || !impl_->t.device().is_cpu() || !impl_->t.is_contiguous()) return nullptr;
  return impl_->t.data_ptr<float>();
}

// ------------------------------------------------------------------ shape ops
Tensor Tensor::reshape(const Shape& s) const {
  return Tensor(wrap(raw(*this).reshape(to_sizes(s))));
}
Tensor Tensor::transpose(int d0, int d1) const {
  return Tensor(wrap(raw(*this).transpose(d0, d1).contiguous()));
}
Tensor Tensor::slice(int d, int64_t start, int64_t end) const {
  return Tensor(wrap(raw(*this).slice(d, start, end).contiguous()));
}
Tensor Tensor::detach() const { return Tensor(wrap(raw(*this).detach())); }
Tensor Tensor::clone_leaf(bool requires_grad) const {
  return Tensor(wrap(raw(*this).detach().clone().set_requires_grad(requires_grad)));
}

// ---------------------------------------------------------------- elementwise
Tensor Tensor::scale(float f) const { return Tensor(wrap(raw(*this) * f)); }
Tensor Tensor::add(const Tensor& o) const {
  SLM_CHECK(shape() == o.shape(), "add: shape mismatch");
  return Tensor(wrap(raw(*this) + raw(o)));
}
Tensor Tensor::mul(const Tensor& o) const {
  SLM_CHECK(shape() == o.shape(), "mul: shape mismatch");
  return Tensor(wrap(raw(*this) * raw(o)));
}
Tensor Tensor::add_bias(const Tensor& b) const {
  return Tensor(wrap(raw(*this) + raw(b)));  // broadcasting over the leading dims
}
Tensor Tensor::gelu() const { return Tensor(wrap(torch::gelu(raw(*this), "tanh"))); }
Tensor Tensor::softmax_last() const {
  return Tensor(wrap(torch::softmax(raw(*this), -1)));
}
Tensor Tensor::layernorm(const Tensor& gain, const Tensor& bias, float eps) const {
  const int64_t C = size(-1);
  return Tensor(wrap(torch::layer_norm(raw(*this), {C}, raw(gain), raw(bias), eps)));
}
Tensor Tensor::causal_mask() const {
  const torch::Tensor& t = raw(*this);
  const int64_t T1 = t.size(-2), T2 = t.size(-1);
  const torch::Tensor keep =
      torch::ones({T1, T2}, torch::TensorOptions().dtype(torch::kBool).device(t.device()))
          .tril(T2 - T1);
  return Tensor(wrap(t.masked_fill(keep.logical_not(), -1.0e9f)));
}
Tensor Tensor::sum_all() const { return Tensor(wrap(raw(*this).sum())); }
Tensor Tensor::mean_all() const { return Tensor(wrap(raw(*this).mean())); }
Tensor Tensor::matmul(const Tensor& o) const {
  return Tensor(wrap(torch::matmul(raw(*this), raw(o))));
}
void Tensor::backward() {
  raw(*this).backward();
  ckpt_release_own();
}

// ---------------------------------------------------------------------- ops
Tensor embedding(const Tensor& weight, const std::vector<int32_t>& ids, int64_t B,
                 int64_t T) {
  SLM_CHECK(static_cast<int64_t>(ids.size()) == B * T, "embedding: ids size");
  torch::Tensor idx =
      torch::from_blob(const_cast<int32_t*>(ids.data()), {B, T},
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kLong)
          .to(raw(weight).device());
  return Tensor(wrap(torch::embedding(raw(weight), idx)));
}

Tensor linear(const Tensor& x, const Tensor& w, const Tensor* bias) {
  torch::Tensor y = torch::matmul(raw(x), raw(w));  // w is [in, out]
  if (bias) y = y + raw(*bias);
  return Tensor(wrap(y));
}

Tensor cross_entropy(const Tensor& logits, const std::vector<int32_t>& targets,
                     int32_t ignore_index, float* loss_out, int64_t* ntok_out) {
  const torch::Tensor& l = raw(logits);
  const int64_t V = l.size(-1);
  const int64_t N = l.numel() / V;
  SLM_CHECK(static_cast<int64_t>(targets.size()) == N, "cross_entropy: target count");
  torch::Tensor tg = torch::from_blob(const_cast<int32_t*>(targets.data()), {N},
                                      torch::TensorOptions().dtype(torch::kInt32))
                         .to(torch::kLong)
                         .to(l.device());
  int64_t ntok = 0;
  for (int32_t t : targets)
    if (t != ignore_index) ++ntok;
  if (ntok_out) *ntok_out = ntok;
  torch::Tensor loss = torch::nn::functional::cross_entropy(
      l.reshape({N, V}), tg,
      torch::nn::functional::CrossEntropyFuncOptions().ignore_index(ignore_index).reduction(
          torch::kMean));
  if (ntok == 0) loss = loss.nan_to_num(0.0);
  if (loss_out) *loss_out = loss.item<float>();
  return Tensor(wrap(loss));
}

Tensor seq_logprob(const Tensor& logits, const std::vector<int32_t>& targets,
                   int32_t ignore_index) {
  const torch::Tensor& l = raw(logits);
  SLM_CHECK(l.dim() == 3, "seq_logprob: logits must be [B,T,V]");
  const int64_t B = l.size(0), T = l.size(1);
  SLM_CHECK(static_cast<int64_t>(targets.size()) == B * T, "seq_logprob: target count");
  torch::Tensor tg = torch::from_blob(const_cast<int32_t*>(targets.data()), {B, T},
                                      torch::TensorOptions().dtype(torch::kInt32))
                         .to(torch::kLong)
                         .to(l.device());
  torch::Tensor mask = (tg != ignore_index);
  torch::Tensor safe = torch::where(mask, tg, torch::zeros_like(tg));
  torch::Tensor lp = torch::log_softmax(l, -1);
  torch::Tensor picked = lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1);  // [B,T]
  return Tensor(wrap((picked * mask.to(picked.dtype())).sum(1)));
}

Tensor logsigmoid(const Tensor& x) {
  return Tensor(wrap(torch::log_sigmoid(raw(x))));
}

Tensor checkpoint(const std::function<Tensor(const Tensor&)>& fn, const Tensor& x) {
  if (!torch::GradMode::is_enabled()) return fn(x);
  const int64_t id = ckpt_register(fn);
  return Tensor(wrap(CheckpointFn::apply(raw(x), id)));
}

// -------------------------------------------------------------- raw helpers
void adamw_apply(Tensor& p, Tensor& m, Tensor& v, float lr, float beta1, float beta2,
                 float eps, float weight_decay, int64_t step) {
  torch::NoGradGuard ng;
  torch::Tensor pd = raw(p).detach();
  torch::Tensor md = raw(m).detach();
  torch::Tensor vd = raw(v).detach();
  torch::Tensor g = raw(p).grad();
  if (!g.defined()) return;
  md.mul_(beta1).add_(g.detach(), 1.0f - beta1);
  vd.mul_(beta2).addcmul_(g.detach(), g.detach(), 1.0f - beta2);
  const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
  const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
  torch::Tensor upd = (md / bc1) / ((vd / bc2).sqrt() + eps);
  if (weight_decay != 0.0f) upd = upd + pd * weight_decay;
  pd.add_(upd, -lr);
}

float grads_global_norm(const std::vector<Tensor*>& params) {
  torch::NoGradGuard ng;
  double acc = 0.0;
  for (Tensor* t : params) {
    if (!t || !t->has_grad()) continue;
    acc += raw(*t).grad().detach().pow(2).sum().item<double>();
  }
  return static_cast<float>(std::sqrt(acc));
}

void grads_scale(const std::vector<Tensor*>& params, float s) {
  torch::NoGradGuard ng;
  for (Tensor* t : params)
    if (t && t->has_grad()) raw(*t).grad().detach().mul_(s);
}

// ----------------------------------------------------------------- context
namespace {
thread_local std::vector<std::unique_ptr<torch::NoGradGuard>> g_guards;
}
NoGradGuard::NoGradGuard() : prev_(torch::GradMode::is_enabled()) {
  torch::GradMode::set_enabled(false);
}
NoGradGuard::~NoGradGuard() { torch::GradMode::set_enabled(prev_); }
bool grad_enabled() { return torch::GradMode::is_enabled(); }

const char* backend_name() {
  static std::string n;
  if (n.empty()) {
    n = std::string("libtorch ") + TORCH_VERSION + (g_device.is_cuda() ? "/cuda" : "/cpu") +
        " threads=" + std::to_string(at::get_num_threads());
  }
  return n.c_str();
}

void backend_init(int threads, bool prefer_cuda) {
  g_threads = threads;
  if (threads > 0) {
    at::set_num_threads(threads);
    at::set_num_interop_threads(1);
  }
  if (prefer_cuda && torch::cuda::is_available())
    g_device = torch::Device(torch::kCUDA, 0);
  else
    g_device = torch::Device(torch::kCPU);
}

bool backend_on_gpu() { return g_device.is_cuda(); }
size_t backend_allocated_bytes() { return 0; }

}  // namespace slm
