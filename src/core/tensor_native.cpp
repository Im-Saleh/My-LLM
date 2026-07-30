// SPDX-License-Identifier: Apache-2.0
//
// Native (zero dependency) reverse-mode autodiff engine.
//
// Design notes
// ------------
//  * Every tensor is contiguous row-major.  `reshape` shares storage, every
//    other shape op materialises a copy; this keeps all kernels trivially
//    correct at the price of a few memcpys (the model is GEMM bound anyway).
//  * The graph is built out of `TensorImpl` nodes holding a closure that pushes
//    gradient from the node into its parents.  `run_backward` performs a
//    reverse post-order traversal (a valid topological order for a DAG) and is
//    re-entrant, which is what makes gradient checkpointing possible.
//  * Gradients of leaves (parameters) live in `TensorImpl::grad` and are
//    accumulated, exactly like PyTorch.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/gemm.h"
#include "core/rng.h"
#include "core/tensor.h"

namespace slm {
namespace {

std::atomic<size_t> g_alloc_bytes{0};
thread_local bool t_grad_enabled = true;

[[noreturn]] void fail(const std::string& m) {
  throw std::runtime_error("slm/tensor: " + m);
}
#define SLM_CHECK(cond, msg)  \
  do {                        \
    if (!(cond)) fail(msg);   \
  } while (0)

}  // namespace

namespace detail {

struct Storage {
  std::vector<float> v;
  explicit Storage(int64_t n) : v(static_cast<size_t>(n), 0.0f) {
    g_alloc_bytes.fetch_add(sizeof(float) * v.size(), std::memory_order_relaxed);
  }
  ~Storage() {
    g_alloc_bytes.fetch_sub(sizeof(float) * v.size(), std::memory_order_relaxed);
  }
};

struct TensorImpl {
  Shape shape;
  std::shared_ptr<Storage> st;
  int64_t n = 0;
  bool requires_grad = false;
  bool is_leaf = true;
  const char* op = "leaf";
  std::vector<float> grad;
  std::vector<std::shared_ptr<TensorImpl>> parents;
  std::function<void(TensorImpl&)> bwd;

  float* data() { return st->v.data(); }
  const float* data() const { return st->v.data(); }
  float* g() {
    if (grad.size() != static_cast<size_t>(n)) grad.assign(static_cast<size_t>(n), 0.0f);
    return grad.data();
  }
  const float* gconst() const {
    return grad.size() == static_cast<size_t>(n) ? grad.data() : nullptr;
  }
};

}  // namespace detail

using detail::TensorImpl;
using ImplPtr = std::shared_ptr<TensorImpl>;

namespace {

ImplPtr make_impl(const Shape& s) {
  auto p = std::make_shared<TensorImpl>();
  p->shape = s;
  p->n = numel_of(s);
  SLM_CHECK(p->n >= 0, "negative numel");
  p->st = std::make_shared<detail::Storage>(p->n);
  return p;
}

// Shares storage with `src` (used by reshape only).
ImplPtr make_view(const ImplPtr& src, const Shape& s) {
  auto p = std::make_shared<TensorImpl>();
  p->shape = s;
  p->n = numel_of(s);
  p->st = src->st;
  return p;
}

bool tracking(const std::initializer_list<const ImplPtr*>& in) {
  if (!t_grad_enabled) return false;
  for (const ImplPtr* p : in)
    if (*p && (*p)->requires_grad) return true;
  return false;
}

void link(const ImplPtr& out, std::vector<ImplPtr> parents, const char* op,
          std::function<void(TensorImpl&)> bwd) {
  out->requires_grad = true;
  out->is_leaf = false;
  out->op = op;
  out->parents = std::move(parents);
  out->bwd = std::move(bwd);
}

std::vector<int64_t> contiguous_strides(const Shape& s) {
  std::vector<int64_t> st(s.size(), 1);
  for (int i = static_cast<int>(s.size()) - 2; i >= 0; --i) st[i] = st[i + 1] * s[i + 1];
  return st;
}

// Reverse post-order backward pass; `seed` is added into root's gradient.
void run_backward(const ImplPtr& root, const float* seed, int64_t seed_n) {
  SLM_CHECK(root != nullptr, "backward on undefined tensor");
  SLM_CHECK(root->requires_grad, "backward on tensor that does not require grad");
  SLM_CHECK(seed_n == root->n, "backward seed size mismatch");

  std::vector<TensorImpl*> order;
  std::unordered_set<TensorImpl*> seen;
  struct Frame {
    TensorImpl* node;
    size_t idx;
  };
  std::vector<Frame> stack;
  stack.push_back({root.get(), 0});
  seen.insert(root.get());
  while (!stack.empty()) {
    Frame& f = stack.back();
    if (f.idx < f.node->parents.size()) {
      TensorImpl* p = f.node->parents[f.idx++].get();
      if (p && p->requires_grad && !seen.count(p)) {
        seen.insert(p);
        stack.push_back({p, 0});
      }
    } else {
      order.push_back(f.node);
      stack.pop_back();
    }
  }
  float* rg = root->g();
  for (int64_t i = 0; i < seed_n; ++i) rg[i] += seed[i];
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    TensorImpl* node = *it;
    if (node->bwd && node->gconst()) node->bwd(*node);
  }
}

}  // namespace

// =========================================================================
// Tensor boilerplate
// =========================================================================
Tensor::Tensor() = default;
Tensor::~Tensor() = default;
Tensor::Tensor(const Tensor&) = default;
Tensor::Tensor(Tensor&&) noexcept = default;
Tensor& Tensor::operator=(const Tensor&) = default;
Tensor& Tensor::operator=(Tensor&&) noexcept = default;
Tensor::Tensor(std::shared_ptr<detail::TensorImpl> impl) : impl_(std::move(impl)) {}

Tensor Tensor::zeros(const Shape& s, bool requires_grad) {
  auto p = make_impl(s);
  p->requires_grad = requires_grad && t_grad_enabled;
  return Tensor(p);
}

Tensor Tensor::full(const Shape& s, float value, bool requires_grad) {
  auto p = make_impl(s);
  std::fill(p->st->v.begin(), p->st->v.end(), value);
  p->requires_grad = requires_grad && t_grad_enabled;
  return Tensor(p);
}

Tensor Tensor::from_host(const Shape& s, const float* data, bool requires_grad) {
  auto p = make_impl(s);
  if (data && p->n) std::memcpy(p->data(), data, sizeof(float) * static_cast<size_t>(p->n));
  p->requires_grad = requires_grad && t_grad_enabled;
  return Tensor(p);
}

Tensor Tensor::randn(const Shape& s, float stddev, uint64_t seed, bool requires_grad) {
  auto p = make_impl(s);
  Rng rng(seed);
  float* d = p->data();
  for (int64_t i = 0; i < p->n; ++i) d[i] = rng.normal() * stddev;
  p->requires_grad = requires_grad && t_grad_enabled;
  return Tensor(p);
}

Tensor Tensor::scalar(float v, bool requires_grad) {
  return Tensor::full({}, v, requires_grad);
}

Shape Tensor::shape() const { return impl_ ? impl_->shape : Shape{}; }
int Tensor::dim() const { return impl_ ? static_cast<int>(impl_->shape.size()) : 0; }
int64_t Tensor::size(int d) const {
  SLM_CHECK(impl_ != nullptr, "size() on undefined tensor");
  const int nd = static_cast<int>(impl_->shape.size());
  if (d < 0) d += nd;
  SLM_CHECK(d >= 0 && d < nd, "size(): dim out of range");
  return impl_->shape[d];
}
int64_t Tensor::numel() const { return impl_ ? impl_->n : 0; }
bool Tensor::requires_grad() const { return impl_ && impl_->requires_grad; }
void Tensor::set_requires_grad(bool v) {
  SLM_CHECK(impl_ != nullptr, "set_requires_grad on undefined tensor");
  SLM_CHECK(impl_->is_leaf, "only leaf tensors can change requires_grad");
  impl_->requires_grad = v;
  if (!v) impl_->grad.clear();
}

void Tensor::copy_to_host(std::vector<float>& out) const {
  SLM_CHECK(impl_ != nullptr, "copy_to_host on undefined tensor");
  out.assign(impl_->data(), impl_->data() + impl_->n);
}
void Tensor::copy_from_host(const float* src, int64_t n) {
  SLM_CHECK(impl_ != nullptr, "copy_from_host on undefined tensor");
  SLM_CHECK(n == impl_->n, "copy_from_host size mismatch");
  if (n) std::memcpy(impl_->data(), src, sizeof(float) * static_cast<size_t>(n));
}
void Tensor::copy_grad_to_host(std::vector<float>& out) const {
  SLM_CHECK(impl_ != nullptr, "copy_grad_to_host on undefined tensor");
  const float* g = impl_->gconst();
  if (!g) {
    out.assign(static_cast<size_t>(impl_->n), 0.0f);
    return;
  }
  out.assign(g, g + impl_->n);
}
void Tensor::add_to_host_data(const float* delta, int64_t n, float alpha) {
  SLM_CHECK(impl_ != nullptr, "add_to_host_data on undefined tensor");
  SLM_CHECK(n == impl_->n, "add_to_host_data size mismatch");
  float* d = impl_->data();
  for (int64_t i = 0; i < n; ++i) d[i] += alpha * delta[i];
}
bool Tensor::has_grad() const { return impl_ && impl_->gconst() != nullptr; }
void Tensor::zero_grad() {
  if (impl_) std::fill(impl_->grad.begin(), impl_->grad.end(), 0.0f);
}
float Tensor::item() const {
  SLM_CHECK(impl_ && impl_->n == 1, "item() requires a 1-element tensor");
  return impl_->data()[0];
}
const float* Tensor::host_ptr() const { return impl_ ? impl_->data() : nullptr; }
float* Tensor::mutable_host_ptr() { return impl_ ? impl_->data() : nullptr; }

// =========================================================================
// Shape ops
// =========================================================================
Tensor Tensor::reshape(const Shape& s) const {
  SLM_CHECK(impl_ != nullptr, "reshape on undefined tensor");
  SLM_CHECK(numel_of(s) == impl_->n,
            "reshape " + shape_str(impl_->shape) + " -> " + shape_str(s));
  auto out = make_view(impl_, s);
  if (tracking({&impl_})) {
    ImplPtr in = impl_;
    link(out, {in}, "reshape", [in](TensorImpl& o) {
      float* gi = in->g();
      const float* go = o.gconst();
      for (int64_t i = 0; i < o.n; ++i) gi[i] += go[i];
    });
  }
  return Tensor(out);
}

Tensor Tensor::transpose(int d0, int d1) const {
  SLM_CHECK(impl_ != nullptr, "transpose on undefined tensor");
  const int nd = static_cast<int>(impl_->shape.size());
  if (d0 < 0) d0 += nd;
  if (d1 < 0) d1 += nd;
  SLM_CHECK(d0 >= 0 && d0 < nd && d1 >= 0 && d1 < nd, "transpose: dim out of range");
  Shape os = impl_->shape;
  std::swap(os[d0], os[d1]);
  auto out = make_impl(os);
  const auto istr_in = contiguous_strides(impl_->shape);
  // Input stride associated with each *output* dimension.
  auto mapped = std::make_shared<std::vector<int64_t>>(nd);
  for (int k = 0; k < nd; ++k) {
    const int kk = (k == d0) ? d1 : (k == d1 ? d0 : k);
    (*mapped)[static_cast<size_t>(k)] = istr_in[static_cast<size_t>(kk)];
  }
  auto oshape = std::make_shared<Shape>(os);
  const int64_t n = out->n;
  // If the innermost output dimension is also innermost in the input, whole
  // runs are contiguous in both and can be memcpy'd.
  const int64_t run = (*mapped)[static_cast<size_t>(nd - 1)] == 1 ? os[nd - 1] : 1;

  // Odometer walk: no divisions, one add per dimension carry.
  auto walk = [mapped, oshape, nd, n, run](const float* src, float* dst, bool accumulate) {
    std::vector<int64_t> coord(static_cast<size_t>(nd), 0);
    int64_t in_off = 0;
    for (int64_t idx = 0; idx < n; idx += run) {
      if (run > 1) {
        if (accumulate) {
          float* d = dst + in_off;
          const float* s2 = src + idx;
          for (int64_t r = 0; r < run; ++r) d[r] += s2[r];
        } else {
          std::memcpy(dst + idx, src + in_off, sizeof(float) * static_cast<size_t>(run));
        }
      } else if (accumulate) {
        dst[in_off] += src[idx];
      } else {
        dst[idx] = src[in_off];
      }
      for (int k = nd - 1; k >= 0; --k) {
        const int64_t stepc = (k == nd - 1) ? run : 1;
        coord[static_cast<size_t>(k)] += stepc;
        in_off += (*mapped)[static_cast<size_t>(k)] * stepc;
        if (coord[static_cast<size_t>(k)] < (*oshape)[static_cast<size_t>(k)]) break;
        in_off -= (*mapped)[static_cast<size_t>(k)] * coord[static_cast<size_t>(k)];
        coord[static_cast<size_t>(k)] = 0;
      }
    }
  };
  walk(impl_->data(), out->data(), false);
  if (tracking({&impl_})) {
    ImplPtr in = impl_;
    link(out, {in}, "transpose", [in, walk](TensorImpl& o) {
      walk(o.gconst(), in->g(), true);
    });
  }
  return Tensor(out);
}

Tensor Tensor::slice(int d, int64_t start, int64_t end) const {
  SLM_CHECK(impl_ != nullptr, "slice on undefined tensor");
  const int nd = static_cast<int>(impl_->shape.size());
  if (d < 0) d += nd;
  SLM_CHECK(d >= 0 && d < nd, "slice: dim out of range");
  const int64_t D = impl_->shape[d];
  if (end < 0) end += D;
  SLM_CHECK(start >= 0 && end <= D && start < end, "slice: bad range");
  Shape os = impl_->shape;
  os[d] = end - start;
  auto out = make_impl(os);

  int64_t outer = 1, inner = 1;
  for (int i = 0; i < d; ++i) outer *= impl_->shape[i];
  for (int i = d + 1; i < nd; ++i) inner *= impl_->shape[i];
  const int64_t len = end - start;
  const int64_t chunk = len * inner;

  const float* src = impl_->data();
  float* dst = out->data();
  for (int64_t o = 0; o < outer; ++o)
    std::memcpy(dst + o * chunk, src + (o * D + start) * inner,
                sizeof(float) * static_cast<size_t>(chunk));

  if (tracking({&impl_})) {
    ImplPtr in = impl_;
    link(out, {in}, "slice", [in, outer, D, inner, start, chunk, len](TensorImpl& o) {
      const float* go = o.gconst();
      float* gi = in->g();
      for (int64_t b = 0; b < outer; ++b) {
        float* dstp = gi + (b * D + start) * inner;
        const float* srcp = go + b * chunk;
        for (int64_t i = 0; i < len * inner; ++i) dstp[i] += srcp[i];
      }
    });
  }
  return Tensor(out);
}

Tensor Tensor::detach() const {
  SLM_CHECK(impl_ != nullptr, "detach on undefined tensor");
  auto out = make_view(impl_, impl_->shape);
  return Tensor(out);
}

Tensor Tensor::clone_leaf(bool requires_grad) const {
  SLM_CHECK(impl_ != nullptr, "clone_leaf on undefined tensor");
  return Tensor::from_host(impl_->shape, impl_->data(), requires_grad);
}

// =========================================================================
// Elementwise ops
// =========================================================================
Tensor Tensor::scale(float f) const {
  SLM_CHECK(impl_ != nullptr, "scale on undefined tensor");
  auto out = make_impl(impl_->shape);
  const float* a = impl_->data();
  float* y = out->data();
  for (int64_t i = 0; i < out->n; ++i) y[i] = a[i] * f;
  if (tracking({&impl_})) {
    ImplPtr in = impl_;
    link(out, {in}, "scale", [in, f](TensorImpl& o) {
      float* gi = in->g();
      const float* go = o.gconst();
      for (int64_t i = 0; i < o.n; ++i) gi[i] += f * go[i];
    });
  }
  return Tensor(out);
}

Tensor Tensor::add(const Tensor& other) const {
  SLM_CHECK(impl_ && other.impl_, "add on undefined tensor");
  SLM_CHECK(impl_->shape == other.impl_->shape,
            "add: shape mismatch " + shape_str(impl_->shape) + " vs " +
                shape_str(other.impl_->shape));
  auto out = make_impl(impl_->shape);
  const float* a = impl_->data();
  const float* b = other.impl_->data();
  float* y = out->data();
  for (int64_t i = 0; i < out->n; ++i) y[i] = a[i] + b[i];
  if (tracking({&impl_, &other.impl_})) {
    ImplPtr x = impl_, z = other.impl_;
    link(out, {x, z}, "add", [x, z](TensorImpl& o) {
      const float* go = o.gconst();
      if (x->requires_grad) {
        float* g = x->g();
        for (int64_t i = 0; i < o.n; ++i) g[i] += go[i];
      }
      if (z->requires_grad) {
        float* g = z->g();
        for (int64_t i = 0; i < o.n; ++i) g[i] += go[i];
      }
    });
  }
  return Tensor(out);
}

Tensor Tensor::mul(const Tensor& other) const {
  SLM_CHECK(impl_ && other.impl_, "mul on undefined tensor");
  SLM_CHECK(impl_->shape == other.impl_->shape, "mul: shape mismatch");
  auto out = make_impl(impl_->shape);
  const float* a = impl_->data();
  const float* b = other.impl_->data();
  float* y = out->data();
  for (int64_t i = 0; i < out->n; ++i) y[i] = a[i] * b[i];
  if (tracking({&impl_, &other.impl_})) {
    ImplPtr x = impl_, z = other.impl_;
    link(out, {x, z}, "mul", [x, z](TensorImpl& o) {
      const float* go = o.gconst();
      if (x->requires_grad) {
        float* g = x->g();
        const float* bz = z->data();
        for (int64_t i = 0; i < o.n; ++i) g[i] += go[i] * bz[i];
      }
      if (z->requires_grad) {
        float* g = z->g();
        const float* ax = x->data();
        for (int64_t i = 0; i < o.n; ++i) g[i] += go[i] * ax[i];
      }
    });
  }
  return Tensor(out);
}

// x: [..., trailing...]  b: trailing dims (broadcast over the leading dims)
Tensor Tensor::add_bias(const Tensor& b) const {
  SLM_CHECK(impl_ && b.impl_, "add_bias on undefined tensor");
  const int64_t m = b.impl_->n;
  SLM_CHECK(m > 0 && impl_->n % m == 0, "add_bias: not broadcastable");
  const int nd = static_cast<int>(impl_->shape.size());
  const int bd = static_cast<int>(b.impl_->shape.size());
  SLM_CHECK(bd <= nd, "add_bias: too many dims");
  for (int i = 0; i < bd; ++i)
    SLM_CHECK(b.impl_->shape[bd - 1 - i] == impl_->shape[nd - 1 - i],
              "add_bias: trailing dims must match");
  auto out = make_impl(impl_->shape);
  const int64_t rows = impl_->n / m;
  const float* x = impl_->data();
  const float* bb = b.impl_->data();
  float* y = out->data();
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t j = 0; j < m; ++j) y[r * m + j] = x[r * m + j] + bb[j];
  if (tracking({&impl_, &b.impl_})) {
    ImplPtr xi = impl_, bi = b.impl_;
    link(out, {xi, bi}, "add_bias", [xi, bi, rows, m](TensorImpl& o) {
      const float* go = o.gconst();
      if (xi->requires_grad) {
        float* g = xi->g();
        for (int64_t i = 0; i < o.n; ++i) g[i] += go[i];
      }
      if (bi->requires_grad) {
        float* g = bi->g();
        for (int64_t r = 0; r < rows; ++r)
          for (int64_t j = 0; j < m; ++j) g[j] += go[r * m + j];
      }
    });
  }
  return Tensor(out);
}

Tensor Tensor::gelu() const {
  SLM_CHECK(impl_ != nullptr, "gelu on undefined tensor");
  auto out = make_impl(impl_->shape);
  const float* x = impl_->data();
  float* y = out->data();
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  const int64_t gn = out->n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && gn > 4096)
#endif
  for (int64_t i = 0; i < gn; ++i) {
    const float v = x[i];
    const float u = k * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.0f + std::tanh(u));
  }
  if (tracking({&impl_})) {
    ImplPtr in = impl_;
    link(out, {in}, "gelu", [in, k](TensorImpl& o) {
      const float* x2 = in->data();
      const float* go = o.gconst();
      float* g = in->g();
      const int64_t gn2 = o.n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && gn2 > 4096)
#endif
      for (int64_t i = 0; i < gn2; ++i) {
        const float v = x2[i];
        const float u = k * (v + 0.044715f * v * v * v);
        const float th = std::tanh(u);
        const float dudx = k * (1.0f + 3.0f * 0.044715f * v * v);
        const float d = 0.5f * (1.0f + th) + 0.5f * v * (1.0f - th * th) * dudx;
        g[i] += go[i] * d;
      }
    });
  }
  return Tensor(out);
}

Tensor Tensor::softmax_last() const {
  SLM_CHECK(impl_ && !impl_->shape.empty(), "softmax_last needs >=1 dim");
  const int64_t C = impl_->shape.back();
  const int64_t rows = impl_->n / C;
  auto out = make_impl(impl_->shape);
  const float* x = impl_->data();
  float* y = out->data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && rows > 8)
#endif
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * C;
    float* yr = y + r * C;
    float mx = xr[0];
    for (int64_t j = 1; j < C; ++j) mx = std::max(mx, xr[j]);
    float s = 0.0f;
    for (int64_t j = 0; j < C; ++j) {
      yr[j] = std::exp(xr[j] - mx);
      s += yr[j];
    }
    const float inv = 1.0f / s;
    for (int64_t j = 0; j < C; ++j) yr[j] *= inv;
  }
  if (tracking({&impl_})) {
    ImplPtr in = impl_;
    // Capture the *storage* (not the impl) to keep the softmax output around
    // for the backward pass without creating a reference cycle.
    std::shared_ptr<detail::Storage> ost = out->st;
    link(out, {in}, "softmax", [in, ost, rows, C](TensorImpl& o) {
      const float* y2 = ost->v.data();
      const float* go = o.gconst();
      float* g = in->g();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && rows > 8)
#endif
      for (int64_t r = 0; r < rows; ++r) {
        const float* yr = y2 + r * C;
        const float* gr = go + r * C;
        float dot = 0.0f;
        for (int64_t j = 0; j < C; ++j) dot += gr[j] * yr[j];
        float* gi = g + r * C;
        for (int64_t j = 0; j < C; ++j) gi[j] += yr[j] * (gr[j] - dot);
      }
    });
  }
  return Tensor(out);
}

Tensor Tensor::layernorm(const Tensor& gain, const Tensor& bias, float eps) const {
  SLM_CHECK(impl_ && gain.impl_ && bias.impl_, "layernorm on undefined tensor");
  const int64_t C = impl_->shape.back();
  SLM_CHECK(gain.impl_->n == C && bias.impl_->n == C, "layernorm: gain/bias size");
  const int64_t rows = impl_->n / C;
  auto out = make_impl(impl_->shape);
  auto stats = std::make_shared<std::vector<float>>(2 * rows);  // mean, rstd
  const float* x = impl_->data();
  const float* gn = gain.impl_->data();
  const float* bs = bias.impl_->data();
  float* y = out->data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && rows > 8)
#endif
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * C;
    float m = 0.0f;
    for (int64_t j = 0; j < C; ++j) m += xr[j];
    m /= static_cast<float>(C);
    float v = 0.0f;
    for (int64_t j = 0; j < C; ++j) {
      const float d = xr[j] - m;
      v += d * d;
    }
    v /= static_cast<float>(C);
    const float rstd = 1.0f / std::sqrt(v + eps);
    (*stats)[2 * r] = m;
    (*stats)[2 * r + 1] = rstd;
    float* yr = y + r * C;
    for (int64_t j = 0; j < C; ++j) yr[j] = (xr[j] - m) * rstd * gn[j] + bs[j];
  }
  if (tracking({&impl_, &gain.impl_, &bias.impl_})) {
    ImplPtr xi = impl_, gi = gain.impl_, bi = bias.impl_;
    link(out, {xi, gi, bi}, "layernorm", [xi, gi, bi, stats, rows, C](TensorImpl& o) {
      const float* go = o.gconst();
      const float* x2 = xi->data();
      const float* gn2 = gi->data();
      float* gx = xi->requires_grad ? xi->g() : nullptr;
      float* gg = gi->requires_grad ? gi->g() : nullptr;
      float* gb = bi->requires_grad ? bi->g() : nullptr;
      const float invC = 1.0f / static_cast<float>(C);
      for (int64_t r = 0; r < rows; ++r) {
        const float m = (*stats)[2 * r];
        const float rstd = (*stats)[2 * r + 1];
        const float* xr = x2 + r * C;
        const float* gr = go + r * C;
        float sum_dxhat = 0.0f, sum_dxhat_xhat = 0.0f;
        for (int64_t j = 0; j < C; ++j) {
          const float xhat = (xr[j] - m) * rstd;
          const float dxhat = gr[j] * gn2[j];
          sum_dxhat += dxhat;
          sum_dxhat_xhat += dxhat * xhat;
          if (gg) gg[j] += gr[j] * xhat;
          if (gb) gb[j] += gr[j];
        }
        if (gx) {
          float* gxr = gx + r * C;
          for (int64_t j = 0; j < C; ++j) {
            const float xhat = (xr[j] - m) * rstd;
            const float dxhat = gr[j] * gn2[j];
            gxr[j] += rstd * (dxhat - invC * sum_dxhat - xhat * invC * sum_dxhat_xhat);
          }
        }
      }
    });
  }
  return Tensor(out);
}

Tensor Tensor::causal_mask() const {
  SLM_CHECK(impl_ && impl_->shape.size() >= 2, "causal_mask needs >=2 dims");
  const int nd = static_cast<int>(impl_->shape.size());
  const int64_t T2 = impl_->shape[nd - 1];
  const int64_t T1 = impl_->shape[nd - 2];
  const int64_t rows = impl_->n / (T1 * T2);
  auto out = make_impl(impl_->shape);
  const float* x = impl_->data();
  float* y = out->data();
  const float neg = -1.0e9f;
  // Rows are aligned to the *end* of the key axis so that KV-cache style
  // partial queries (T1 < T2) still mask the future correctly.
  const int64_t off = T2 - T1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && rows > 4)
#endif
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t i = 0; i < T1; ++i)
      for (int64_t j = 0; j < T2; ++j) {
        const int64_t idx = (r * T1 + i) * T2 + j;
        y[idx] = (j > i + off) ? neg : x[idx];
      }
  if (tracking({&impl_})) {
    ImplPtr in = impl_;
    link(out, {in}, "causal_mask", [in, rows, T1, T2, off](TensorImpl& o) {
      const float* go = o.gconst();
      float* g = in->g();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && rows > 4)
#endif
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t i = 0; i < T1; ++i)
          for (int64_t j = 0; j < T2; ++j) {
            const int64_t idx = (r * T1 + i) * T2 + j;
            if (j <= i + off) g[idx] += go[idx];
          }
    });
  }
  return Tensor(out);
}

Tensor Tensor::sum_all() const {
  SLM_CHECK(impl_ != nullptr, "sum_all on undefined tensor");
  auto out = make_impl({});
  double s = 0.0;
  const float* x = impl_->data();
  for (int64_t i = 0; i < impl_->n; ++i) s += x[i];
  out->data()[0] = static_cast<float>(s);
  if (tracking({&impl_})) {
    ImplPtr in = impl_;
    link(out, {in}, "sum", [in](TensorImpl& o) {
      const float gv = o.gconst()[0];
      float* g = in->g();
      for (int64_t i = 0; i < in->n; ++i) g[i] += gv;
    });
  }
  return Tensor(out);
}

Tensor Tensor::mean_all() const {
  SLM_CHECK(impl_ && impl_->n > 0, "mean_all on empty tensor");
  return sum_all().scale(1.0f / static_cast<float>(impl_->n));
}

// =========================================================================
// matmul / linear
// =========================================================================
Tensor Tensor::matmul(const Tensor& other) const {
  SLM_CHECK(impl_ && other.impl_, "matmul on undefined tensor");
  const Shape& as = impl_->shape;
  const Shape& bs = other.impl_->shape;
  SLM_CHECK(as.size() >= 2 && as.size() == bs.size(), "matmul: rank mismatch");
  const int nd = static_cast<int>(as.size());
  const int64_t M = as[nd - 2], K = as[nd - 1];
  const int64_t N = bs[nd - 1];
  SLM_CHECK(bs[nd - 2] == K, "matmul: inner dim mismatch " + shape_str(as) + " x " + shape_str(bs));
  int64_t batch = 1;
  for (int i = 0; i < nd - 2; ++i) {
    SLM_CHECK(as[i] == bs[i], "matmul: batch dim mismatch");
    batch *= as[i];
  }
  Shape os = as;
  os[nd - 1] = N;
  auto out = make_impl(os);
  const float* A = impl_->data();
  const float* B = other.impl_->data();
  float* C = out->data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (batch > 1)
#endif
  for (int64_t b = 0; b < batch; ++b)
    sgemm(false, false, M, N, K, 1.0f, A + b * M * K, K, B + b * K * N, N, 0.0f,
          C + b * M * N, N);

  if (tracking({&impl_, &other.impl_})) {
    ImplPtr xi = impl_, yi = other.impl_;
    link(out, {xi, yi}, "matmul", [xi, yi, batch, M, N, K](TensorImpl& o) {
      const float* go = o.gconst();
      const float* A2 = xi->data();
      const float* B2 = yi->data();
      float* gA = xi->requires_grad ? xi->g() : nullptr;
      float* gB = yi->requires_grad ? yi->g() : nullptr;
      if (gA) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (batch > 1)
#endif
        for (int64_t b = 0; b < batch; ++b)  // gA = go @ B^T
          sgemm(false, true, M, K, N, 1.0f, go + b * M * N, N, B2 + b * K * N, N,
                1.0f, gA + b * M * K, K);
      }
      if (gB) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (batch > 1)
#endif
        for (int64_t b = 0; b < batch; ++b)  // gB = A^T @ go
          sgemm(true, false, K, N, M, 1.0f, A2 + b * M * K, K, go + b * M * N, N,
                1.0f, gB + b * K * N, N);
      }
    });
  }
  return Tensor(out);
}

Tensor linear(const Tensor& x, const Tensor& w, const Tensor* bias) {
  const ImplPtr xi = x.impl();
  const ImplPtr wi = w.impl();
  SLM_CHECK(xi && wi, "linear on undefined tensor");
  SLM_CHECK(wi->shape.size() == 2, "linear: weight must be 2-D [in,out]");
  const int64_t I = wi->shape[0], O = wi->shape[1];
  SLM_CHECK(!xi->shape.empty() && xi->shape.back() == I, "linear: input dim mismatch");
  const int64_t N = xi->n / I;
  Shape os = xi->shape;
  os.back() = O;
  auto out = make_impl(os);
  sgemm(false, false, N, O, I, 1.0f, xi->data(), I, wi->data(), O, 0.0f,
        out->data(), O);
  if (bias) {
    const ImplPtr bi = bias->impl();
    SLM_CHECK(bi && bi->n == O, "linear: bias size");
    const float* b = bi->data();
    float* y = out->data();
    for (int64_t n = 0; n < N; ++n)
      for (int64_t o = 0; o < O; ++o) y[n * O + o] += b[o];
  }
  ImplPtr bi = bias ? bias->impl() : nullptr;
  const bool track = t_grad_enabled && ((xi && xi->requires_grad) ||
                                        (wi && wi->requires_grad) ||
                                        (bi && bi->requires_grad));
  if (track) {
    std::vector<ImplPtr> parents{xi, wi};
    if (bi) parents.push_back(bi);
    link(out, parents, "linear", [xi, wi, bi, N, I, O](TensorImpl& o) {
      const float* go = o.gconst();
      if (xi->requires_grad)  // gx = go @ w^T
        sgemm(false, true, N, I, O, 1.0f, go, O, wi->data(), O, 1.0f, xi->g(), I);
      if (wi->requires_grad)  // gw = x^T @ go
        sgemm(true, false, I, O, N, 1.0f, xi->data(), I, go, O, 1.0f, wi->g(), O);
      if (bi && bi->requires_grad) {
        float* gb = bi->g();
        for (int64_t n = 0; n < N; ++n)
          for (int64_t k = 0; k < O; ++k) gb[k] += go[n * O + k];
      }
    });
  }
  return Tensor(out);
}

// =========================================================================
// Indexing / loss ops
// =========================================================================
Tensor embedding(const Tensor& weight, const std::vector<int32_t>& ids, int64_t B,
                 int64_t T) {
  const ImplPtr wi = weight.impl();
  SLM_CHECK(wi && wi->shape.size() == 2, "embedding: weight must be [V,C]");
  const int64_t V = wi->shape[0], C = wi->shape[1];
  SLM_CHECK(static_cast<int64_t>(ids.size()) == B * T, "embedding: ids size");
  auto out = make_impl({B, T, C});
  const float* w = wi->data();
  float* y = out->data();
  for (int64_t i = 0; i < B * T; ++i) {
    const int32_t id = ids[static_cast<size_t>(i)];
    SLM_CHECK(id >= 0 && id < V, "embedding: id out of range");
    std::memcpy(y + i * C, w + static_cast<int64_t>(id) * C,
                sizeof(float) * static_cast<size_t>(C));
  }
  if (t_grad_enabled && wi->requires_grad) {
    auto idsp = std::make_shared<std::vector<int32_t>>(ids);
    link(out, {wi}, "embedding", [wi, idsp, C](TensorImpl& o) {
      const float* go = o.gconst();
      float* g = wi->g();
      const int64_t n = static_cast<int64_t>(idsp->size());
      for (int64_t i = 0; i < n; ++i) {
        float* dst = g + static_cast<int64_t>((*idsp)[static_cast<size_t>(i)]) * C;
        const float* src = go + i * C;
        for (int64_t j = 0; j < C; ++j) dst[j] += src[j];
      }
    });
  }
  return Tensor(out);
}

Tensor cross_entropy(const Tensor& logits, const std::vector<int32_t>& targets,
                     int32_t ignore_index, float* loss_out, int64_t* ntok_out) {
  const ImplPtr li = logits.impl();
  SLM_CHECK(li && !li->shape.empty(), "cross_entropy: bad logits");
  const int64_t V = li->shape.back();
  const int64_t N = li->n / V;
  SLM_CHECK(static_cast<int64_t>(targets.size()) == N, "cross_entropy: target count");
  auto lse = std::make_shared<std::vector<float>>(static_cast<size_t>(N), 0.0f);
  const float* x = li->data();
  double total = 0.0;
  int64_t ntok = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : total, ntok) if (!omp_in_parallel() && N > 32)
#endif
  for (int64_t r = 0; r < N; ++r) {
    const int32_t t = targets[static_cast<size_t>(r)];
    const float* xr = x + r * V;
    float mx = xr[0];
    for (int64_t j = 1; j < V; ++j) mx = std::max(mx, xr[j]);
    float s = 0.0f;
    for (int64_t j = 0; j < V; ++j) s += std::exp(xr[j] - mx);
    (*lse)[static_cast<size_t>(r)] = mx + std::log(s);
    if (t == ignore_index || t < 0 || t >= V) continue;
    total += static_cast<double>((*lse)[static_cast<size_t>(r)]) - xr[t];
    ++ntok;
  }
  const float loss = ntok ? static_cast<float>(total / static_cast<double>(ntok)) : 0.0f;
  if (loss_out) *loss_out = loss;
  if (ntok_out) *ntok_out = ntok;
  auto out = make_impl({});
  out->data()[0] = loss;
  if (t_grad_enabled && li->requires_grad && ntok > 0) {
    auto tg = std::make_shared<std::vector<int32_t>>(targets);
    link(out, {li}, "cross_entropy", [li, tg, lse, N, V, ntok, ignore_index](TensorImpl& o) {
      const float gout = o.gconst()[0] / static_cast<float>(ntok);
      const float* x2 = li->data();
      float* g = li->g();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 16)
#endif
      for (int64_t r = 0; r < N; ++r) {
        const int32_t t = (*tg)[static_cast<size_t>(r)];
        if (t == ignore_index) continue;
        const float* xr = x2 + r * V;
        float* gr = g + r * V;
        const float l = (*lse)[static_cast<size_t>(r)];
        for (int64_t j = 0; j < V; ++j) gr[j] += gout * std::exp(xr[j] - l);
        gr[t] -= gout;
      }
    });
  }
  return Tensor(out);
}

Tensor seq_logprob(const Tensor& logits, const std::vector<int32_t>& targets,
                   int32_t ignore_index) {
  const ImplPtr li = logits.impl();
  SLM_CHECK(li && li->shape.size() == 3, "seq_logprob: logits must be [B,T,V]");
  const int64_t B = li->shape[0], T = li->shape[1], V = li->shape[2];
  SLM_CHECK(static_cast<int64_t>(targets.size()) == B * T, "seq_logprob: target count");
  auto lse = std::make_shared<std::vector<float>>(static_cast<size_t>(B * T), 0.0f);
  auto out = make_impl({B});
  const float* x = li->data();
  float* y = out->data();
  for (int64_t b = 0; b < B; ++b) {
    double acc = 0.0;
    for (int64_t t = 0; t < T; ++t) {
      const int64_t r = b * T + t;
      const float* xr = x + r * V;
      float mx = xr[0];
      for (int64_t j = 1; j < V; ++j) mx = std::max(mx, xr[j]);
      float s = 0.0f;
      for (int64_t j = 0; j < V; ++j) s += std::exp(xr[j] - mx);
      const float l = mx + std::log(s);
      (*lse)[static_cast<size_t>(r)] = l;
      const int32_t tid = targets[static_cast<size_t>(r)];
      if (tid == ignore_index) continue;
      SLM_CHECK(tid >= 0 && tid < V, "seq_logprob: target out of range");
      acc += static_cast<double>(xr[tid]) - l;
    }
    y[b] = static_cast<float>(acc);
  }
  if (t_grad_enabled && li->requires_grad) {
    auto tg = std::make_shared<std::vector<int32_t>>(targets);
    link(out, {li}, "seq_logprob", [li, tg, lse, B, T, V, ignore_index](TensorImpl& o) {
      const float* go = o.gconst();
      const float* x2 = li->data();
      float* g = li->g();
      for (int64_t b = 0; b < B; ++b) {
        const float gb = go[b];
        if (gb == 0.0f) continue;
        for (int64_t t = 0; t < T; ++t) {
          const int64_t r = b * T + t;
          const int32_t tid = (*tg)[static_cast<size_t>(r)];
          if (tid == ignore_index) continue;
          const float* xr = x2 + r * V;
          float* gr = g + r * V;
          const float l = (*lse)[static_cast<size_t>(r)];
          for (int64_t j = 0; j < V; ++j) gr[j] -= gb * std::exp(xr[j] - l);
          gr[tid] += gb;
        }
      }
    });
  }
  return Tensor(out);
}

Tensor logsigmoid(const Tensor& x) {
  const ImplPtr xi = x.impl();
  SLM_CHECK(xi != nullptr, "logsigmoid on undefined tensor");
  auto out = make_impl(xi->shape);
  const float* a = xi->data();
  float* y = out->data();
  for (int64_t i = 0; i < out->n; ++i) {
    const float v = a[i];
    // log(sigmoid(v)) = -log1p(exp(-|v|)) + min(v,0)
    y[i] = -std::log1p(std::exp(-std::fabs(v))) + std::min(v, 0.0f);
  }
  if (t_grad_enabled && xi->requires_grad) {
    link(out, {xi}, "logsigmoid", [xi](TensorImpl& o) {
      const float* a2 = xi->data();
      const float* go = o.gconst();
      float* g = xi->g();
      for (int64_t i = 0; i < o.n; ++i) {
        const float v = a2[i];
        const float sig_neg = 1.0f / (1.0f + std::exp(v));  // sigmoid(-v)
        g[i] += go[i] * sig_neg;
      }
    });
  }
  return Tensor(out);
}


Tensor Tensor::silu() const {
  SLM_CHECK(impl_ != nullptr, "silu on undefined tensor");
  auto out = make_impl(impl_->shape);
  const float* x = impl_->data();
  float* y = out->data();
  const int64_t n = out->n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && n > 4096)
#endif
  for (int64_t i = 0; i < n; ++i) y[i] = x[i] / (1.0f + std::exp(-x[i]));
  if (tracking({&impl_})) {
    ImplPtr in = impl_;
    link(out, {in}, "silu", [in](TensorImpl& o) {
      const float* x2 = in->data();
      const float* go = o.gconst();
      float* g = in->g();
      const int64_t m = o.n;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && m > 4096)
#endif
      for (int64_t i = 0; i < m; ++i) {
        const float sig = 1.0f / (1.0f + std::exp(-x2[i]));
        g[i] += go[i] * sig * (1.0f + x2[i] * (1.0f - sig));
      }
    });
  }
  return Tensor(out);
}

Tensor Tensor::rmsnorm(const Tensor& gain, float eps) const {
  SLM_CHECK(impl_ && gain.impl_, "rmsnorm on undefined tensor");
  const int64_t C = impl_->shape.back();
  SLM_CHECK(gain.impl_->n == C, "rmsnorm: gain size");
  const int64_t rows = impl_->n / C;
  auto out = make_impl(impl_->shape);
  auto rstd = std::make_shared<std::vector<float>>(static_cast<size_t>(rows));
  const float* x = impl_->data();
  const float* gn = gain.impl_->data();
  float* y = out->data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && rows > 8)
#endif
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * C;
    double ss = 0.0;
    for (int64_t j = 0; j < C; ++j) ss += static_cast<double>(xr[j]) * xr[j];
    const float rs = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(C)) + eps);
    (*rstd)[static_cast<size_t>(r)] = rs;
    float* yr = y + r * C;
    for (int64_t j = 0; j < C; ++j) yr[j] = xr[j] * rs * gn[j];
  }
  if (tracking({&impl_, &gain.impl_})) {
    ImplPtr xi = impl_, gi = gain.impl_;
    link(out, {xi, gi}, "rmsnorm", [xi, gi, rstd, rows, C](TensorImpl& o) {
      const float* go = o.gconst();
      const float* x2 = xi->data();
      const float* gn2 = gi->data();
      float* gx = xi->requires_grad ? xi->g() : nullptr;
      float* gg = gi->requires_grad ? gi->g() : nullptr;
      const float invC = 1.0f / static_cast<float>(C);
      for (int64_t r = 0; r < rows; ++r) {
        const float rs = (*rstd)[static_cast<size_t>(r)];
        const float* xr = x2 + r * C;
        const float* gr = go + r * C;
        double dot = 0.0;  // sum_i dy_i * g_i * x_i
        for (int64_t j = 0; j < C; ++j) dot += static_cast<double>(gr[j]) * gn2[j] * xr[j];
        if (gg)
          for (int64_t j = 0; j < C; ++j) gg[j] += gr[j] * xr[j] * rs;
        if (gx) {
          float* gxr = gx + r * C;
          const float k = static_cast<float>(dot) * rs * rs * rs * invC;
          for (int64_t j = 0; j < C; ++j) gxr[j] += rs * gn2[j] * gr[j] - xr[j] * k;
        }
      }
    });
  }
  return Tensor(out);
}

Tensor rope(const Tensor& x, int64_t pos_offset, float theta) {
  const ImplPtr xi = x.impl();
  SLM_CHECK(xi && xi->shape.size() == 4, "rope: expected [B,H,T,D]");
  const int64_t B = xi->shape[0], H = xi->shape[1], T = xi->shape[2], D = xi->shape[3];
  SLM_CHECK(D % 2 == 0, "rope: head dim must be even");
  auto out = make_impl(xi->shape);
  const int64_t half = D / 2;
  // cos/sin table for this call (T x half), shared with the backward pass
  auto tab = std::make_shared<std::vector<float>>(static_cast<size_t>(2 * T * half));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t i = 0; i < half; ++i) {
      const float freq =
          1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(D));
      const float ang = static_cast<float>(pos_offset + t) * freq;
      (*tab)[static_cast<size_t>((t * half + i) * 2 + 0)] = std::cos(ang);
      (*tab)[static_cast<size_t>((t * half + i) * 2 + 1)] = std::sin(ang);
    }
  const float* src = xi->data();
  float* dst = out->data();
  const int64_t rows = B * H;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && rows > 4)
#endif
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t t = 0; t < T; ++t) {
      const float* s = src + (r * T + t) * D;
      float* d = dst + (r * T + t) * D;
      for (int64_t i = 0; i < half; ++i) {
        const float c = (*tab)[static_cast<size_t>((t * half + i) * 2 + 0)];
        const float sn = (*tab)[static_cast<size_t>((t * half + i) * 2 + 1)];
        const float a = s[2 * i], b = s[2 * i + 1];
        d[2 * i] = a * c - b * sn;
        d[2 * i + 1] = a * sn + b * c;
      }
    }
  if (t_grad_enabled && xi->requires_grad) {
    link(out, {xi}, "rope", [xi, tab, rows, T, D, half](TensorImpl& o) {
      const float* go = o.gconst();
      float* g = xi->g();
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t t = 0; t < T; ++t) {
          const float* gr = go + (r * T + t) * D;
          float* gi = g + (r * T + t) * D;
          for (int64_t i = 0; i < half; ++i) {
            const float c = (*tab)[static_cast<size_t>((t * half + i) * 2 + 0)];
            const float sn = (*tab)[static_cast<size_t>((t * half + i) * 2 + 1)];
            const float ga = gr[2 * i], gb = gr[2 * i + 1];
            gi[2 * i] += ga * c + gb * sn;       // rotation by -angle
            gi[2 * i + 1] += -ga * sn + gb * c;
          }
        }
    });
  }
  return Tensor(out);
}

Tensor repeat_kv(const Tensor& x, int64_t repeat) {
  const ImplPtr xi = x.impl();
  SLM_CHECK(xi && xi->shape.size() == 4, "repeat_kv: expected [B,Hkv,T,D]");
  if (repeat == 1) return x;
  SLM_CHECK(repeat > 1, "repeat_kv: repeat must be >= 1");
  const int64_t B = xi->shape[0], Hkv = xi->shape[1], T = xi->shape[2], D = xi->shape[3];
  auto out = make_impl({B, Hkv * repeat, T, D});
  const int64_t plane = T * D;
  const float* src = xi->data();
  float* dst = out->data();
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < Hkv; ++h)
      for (int64_t k = 0; k < repeat; ++k)
        std::memcpy(dst + ((b * Hkv * repeat) + h * repeat + k) * plane,
                    src + (b * Hkv + h) * plane,
                    sizeof(float) * static_cast<size_t>(plane));
  if (t_grad_enabled && xi->requires_grad) {
    link(out, {xi}, "repeat_kv", [xi, B, Hkv, repeat, plane](TensorImpl& o) {
      const float* go = o.gconst();
      float* g = xi->g();
      for (int64_t b = 0; b < B; ++b)
        for (int64_t h = 0; h < Hkv; ++h) {
          float* gi = g + (b * Hkv + h) * plane;
          for (int64_t k = 0; k < repeat; ++k) {
            const float* gr = go + ((b * Hkv * repeat) + h * repeat + k) * plane;
            for (int64_t i = 0; i < plane; ++i) gi[i] += gr[i];
          }
        }
    });
  }
  return Tensor(out);
}

Tensor z_loss(const Tensor& logits) {
  const ImplPtr li = logits.impl();
  SLM_CHECK(li && !li->shape.empty(), "z_loss: bad logits");
  const int64_t V = li->shape.back();
  const int64_t N = li->n / V;
  auto lse = std::make_shared<std::vector<float>>(static_cast<size_t>(N), 0.0f);
  const float* x = li->data();
  double acc = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : acc) if (!omp_in_parallel() && N > 32)
#endif
  for (int64_t r = 0; r < N; ++r) {
    const float* xr = x + r * V;
    float mx = xr[0];
    for (int64_t j = 1; j < V; ++j) mx = std::max(mx, xr[j]);
    float s = 0.0f;
    for (int64_t j = 0; j < V; ++j) s += std::exp(xr[j] - mx);
    const float l = mx + std::log(s);
    (*lse)[static_cast<size_t>(r)] = l;
    acc += static_cast<double>(l) * l;
  }
  auto out = make_impl({});
  out->data()[0] = static_cast<float>(acc / static_cast<double>(std::max<int64_t>(1, N)));
  if (t_grad_enabled && li->requires_grad) {
    link(out, {li}, "z_loss", [li, lse, N, V](TensorImpl& o) {
      const float g = o.gconst()[0] * 2.0f / static_cast<float>(std::max<int64_t>(1, N));
      const float* x2 = li->data();
      float* gr = li->g();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && N > 32)
#endif
      for (int64_t r = 0; r < N; ++r) {
        const float l = (*lse)[static_cast<size_t>(r)];
        const float* xr = x2 + r * V;
        float* g2 = gr + r * V;
        for (int64_t j = 0; j < V; ++j) g2[j] += g * l * std::exp(xr[j] - l);
      }
    });
  }
  return Tensor(out);
}

// =========================================================================
// Gradient checkpointing
// =========================================================================
Tensor checkpoint(const std::function<Tensor(const Tensor&)>& fn, const Tensor& x) {
  if (!t_grad_enabled) return fn(x);
  Tensor y;
  {
    NoGradGuard ng;
    y = fn(x);
  }
  const ImplPtr yi = y.impl();
  auto out = make_impl(yi->shape);
  std::memcpy(out->data(), yi->data(), sizeof(float) * static_cast<size_t>(yi->n));
  const ImplPtr xi = x.impl();
  auto fnp = std::make_shared<std::function<Tensor(const Tensor&)>>(fn);
  link(out, {xi}, "checkpoint", [xi, fnp](TensorImpl& o) {
    // Recompute the block *with* a graph, then push the incoming gradient
    // through it.  Parameters used inside accumulate into their own buffers.
    Tensor xl = Tensor::from_host(xi->shape, xi->data(), true);
    Tensor y2 = (*fnp)(xl);
    SLM_CHECK(y2.numel() == o.n, "checkpoint: recompute shape mismatch");
    if (!y2.requires_grad()) return;
    run_backward(y2.impl(), o.gconst(), o.n);
    if (xi->requires_grad && xl.has_grad()) {
      std::vector<float> gx;
      xl.copy_grad_to_host(gx);
      float* g = xi->g();
      for (int64_t i = 0; i < xi->n; ++i) g[i] += gx[static_cast<size_t>(i)];
    }
  });
  return Tensor(out);
}

void Tensor::backward() {
  SLM_CHECK(impl_ && impl_->n == 1, "backward() requires a scalar tensor");
  const float one = 1.0f;
  run_backward(impl_, &one, 1);
}

// =========================================================================
// Raw helpers (no autograd)
// =========================================================================
void adamw_apply(Tensor& p, Tensor& m, Tensor& v, float lr, float beta1,
                 float beta2, float eps, float weight_decay, int64_t step) {
  const int64_t n = p.numel();
  SLM_CHECK(m.numel() == n && v.numel() == n, "adamw: state size mismatch");
  float* pd = p.mutable_host_ptr();
  float* md = m.mutable_host_ptr();
  float* vd = v.mutable_host_ptr();
  std::vector<float> gh;
  p.copy_grad_to_host(gh);
  const float* g = gh.data();
  const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
  const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
  for (int64_t i = 0; i < n; ++i) {
    md[i] = beta1 * md[i] + (1.0f - beta1) * g[i];
    vd[i] = beta2 * vd[i] + (1.0f - beta2) * g[i] * g[i];
    const float mh = md[i] / bc1;
    const float vh = vd[i] / bc2;
    pd[i] -= lr * (mh / (std::sqrt(vh) + eps) + weight_decay * pd[i]);
  }
}

float grads_global_norm(const std::vector<Tensor*>& params) {
  double acc = 0.0;
  std::vector<float> g;
  for (Tensor* t : params) {
    if (!t || !t->has_grad()) continue;
    t->copy_grad_to_host(g);
    for (float v : g) acc += static_cast<double>(v) * static_cast<double>(v);
  }
  return static_cast<float>(std::sqrt(acc));
}

void grads_scale(const std::vector<Tensor*>& params, float s) {
  for (Tensor* t : params) {
    if (!t || !t->has_grad()) continue;
    const ImplPtr i = t->impl();
    float* g = i->g();
    for (int64_t k = 0; k < i->n; ++k) g[k] *= s;
  }
}

// =========================================================================
// Context / backend info
// =========================================================================
NoGradGuard::NoGradGuard() : prev_(t_grad_enabled) { t_grad_enabled = false; }
NoGradGuard::~NoGradGuard() { t_grad_enabled = prev_; }
bool grad_enabled() { return t_grad_enabled; }

const char* backend_name() {
  static std::string name = std::string("native(") + gemm_backend_name() + ")";
  return name.c_str();
}
void backend_init(int threads, bool prefer_cuda) {
  (void)prefer_cuda;
  gemm_set_num_threads(threads);
}
bool backend_on_gpu() { return false; }
int backend_threads() { return gemm_num_threads(); }
size_t backend_allocated_bytes() {
  return g_alloc_bytes.load(std::memory_order_relaxed);
}

}  // namespace slm
