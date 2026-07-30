// SPDX-License-Identifier: Apache-2.0
//
// Numerical correctness suite for the tensor engine:
//   1. GEMM (all transpose combinations, odd sizes) against a naive reference.
//   2. Central-difference gradient checks for every autograd op.
//   3. Gradient checkpointing must produce bit-comparable gradients.
//
// Run with:  slm_gradcheck            (exit code 0 == everything passed)
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#ifdef SLM_BACKEND_NATIVE
#include "core/gemm.h"
#endif
#include "core/rng.h"
#include "core/tensor.h"

using namespace slm;

namespace {

int g_failures = 0;
int g_checks = 0;

void report(const std::string& name, bool ok, const std::string& info = "") {
  ++g_checks;
  if (!ok) ++g_failures;
  std::printf("[%s] %-34s %s\n", ok ? " OK " : "FAIL", name.c_str(), info.c_str());
}

// ------------------------------------------------------------------ GEMM test
#ifdef SLM_BACKEND_NATIVE
void test_gemm() {
  Rng rng(7);
  struct Case {
    int64_t M, N, K;
  };
  const std::vector<Case> cases = {{1, 1, 1},   {3, 5, 7},    {8, 16, 32},
                                   {17, 33, 5}, {64, 64, 64}, {4, 129, 63}};
  double worst = 0.0;
  for (const Case& c : cases) {
    for (int ta = 0; ta < 2; ++ta) {
      for (int tb = 0; tb < 2; ++tb) {
        std::vector<float> A(static_cast<size_t>(c.M * c.K)),
            B(static_cast<size_t>(c.K * c.N)), C1(static_cast<size_t>(c.M * c.N)),
            C2(static_cast<size_t>(c.M * c.N));
        for (float& v : A) v = rng.normal();
        for (float& v : B) v = rng.normal();
        for (size_t i = 0; i < C1.size(); ++i) C1[i] = C2[i] = rng.normal();
        const int64_t lda = ta ? c.M : c.K;
        const int64_t ldb = tb ? c.K : c.N;
        sgemm(ta, tb, c.M, c.N, c.K, 1.3f, A.data(), lda, B.data(), ldb, 0.7f,
              C1.data(), c.N);
        sgemm_ref(ta, tb, c.M, c.N, c.K, 1.3f, A.data(), lda, B.data(), ldb, 0.7f,
                  C2.data(), c.N);
        for (size_t i = 0; i < C1.size(); ++i) {
          const double d = std::fabs(C1[i] - C2[i]) / (1.0 + std::fabs(C2[i]));
          worst = std::max(worst, d);
        }
      }
    }
  }
  report("gemm vs reference", worst < 1e-5,
         "max rel err " + std::to_string(worst));
}
#else
void test_gemm() {
  std::printf("[skip] gemm test (backend provides its own BLAS)\n");
}
#endif

// -------------------------------------------------------------- grad checking
using LossFn = std::function<Tensor(std::vector<Tensor>&)>;

void check_grad(const std::string& name, std::vector<Tensor> leaves, LossFn fn,
                float h = 6e-3f, float tol = 2e-2f, int max_probes = 12) {
  for (Tensor& t : leaves) t.set_requires_grad(true);
  Tensor loss = fn(leaves);
  loss.backward();

  std::vector<std::vector<float>> analytic(leaves.size());
  for (size_t i = 0; i < leaves.size(); ++i) leaves[i].copy_grad_to_host(analytic[i]);

  Rng rng(1234 + static_cast<uint64_t>(name.size()));
  double worst = 0.0;
  std::string where;
  for (size_t li = 0; li < leaves.size(); ++li) {
    const int64_t n = leaves[li].numel();
    const int probes = static_cast<int>(std::min<int64_t>(n, max_probes));
    for (int p = 0; p < probes; ++p) {
      const int64_t idx = (probes == n) ? p : static_cast<int64_t>(rng.below(static_cast<uint64_t>(n)));
      float* d = leaves[li].mutable_host_ptr();
      const float orig = d[idx];
      double lp = 0.0, lm = 0.0;
      {
        NoGradGuard ng;
        d[idx] = orig + h;
        lp = fn(leaves).item();
        d[idx] = orig - h;
        lm = fn(leaves).item();
      }
      d[idx] = orig;
      const double num = (lp - lm) / (2.0 * static_cast<double>(h));
      const double ana = analytic[li][static_cast<size_t>(idx)];
      const double denom = 1.0 + std::max(std::fabs(num), std::fabs(ana));
      const double rel = std::fabs(num - ana) / denom;
      if (rel > worst) {
        worst = rel;
        where = "leaf " + std::to_string(li) + " idx " + std::to_string(idx) +
                " num " + std::to_string(num) + " ana " + std::to_string(ana);
      }
    }
  }
  report(name, worst < tol, "max rel err " + std::to_string(worst) + "  " + where);
}

Tensor rnd(const Shape& s, uint64_t seed, float sd = 1.0f) {
  return Tensor::randn(s, sd, seed, false);
}

// A constant projection so that every test reduces to a scalar.
Tensor to_scalar(const Tensor& x, uint64_t seed) {
  Tensor w = rnd(x.shape(), seed, 1.0f);
  return x.mul(w).sum_all();
}

void test_ops() {
  check_grad("scale", {rnd({3, 4}, 1)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].scale(-2.5f), 11);
  });
  check_grad("add", {rnd({2, 3}, 2), rnd({2, 3}, 3)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].add(v[1]), 12);
  });
  check_grad("mul", {rnd({2, 3}, 4), rnd({2, 3}, 5)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].mul(v[1]), 13);
  });
  check_grad("reshape", {rnd({2, 6}, 6)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].reshape({3, 4}), 14);
  });
  check_grad("transpose 2d", {rnd({3, 5}, 7)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].transpose(0, 1), 15);
  });
  check_grad("transpose 4d(1,2)", {rnd({2, 3, 4, 5}, 8)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].transpose(1, 2), 16);
  });
  check_grad("transpose 4d(2,3)", {rnd({2, 3, 4, 5}, 9)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].transpose(2, 3), 17);
  });
  check_grad("slice", {rnd({2, 3, 9}, 10)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].slice(2, 3, 6), 18);
  });
  check_grad("add_bias [C]", {rnd({4, 5}, 11), rnd({5}, 12)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].add_bias(v[1]), 19);
  });
  check_grad("add_bias [T,C] bcast", {rnd({3, 4, 5}, 13), rnd({4, 5}, 14)},
             [](std::vector<Tensor>& v) { return to_scalar(v[0].add_bias(v[1]), 20); });
  check_grad("gelu", {rnd({4, 6}, 15)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].gelu(), 21);
  });
  check_grad("softmax_last", {rnd({3, 7}, 16)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].softmax_last(), 22);
  });
  check_grad("layernorm", {rnd({4, 8}, 17), rnd({8}, 18), rnd({8}, 19)},
             [](std::vector<Tensor>& v) {
               return to_scalar(v[0].layernorm(v[1], v[2], 1e-5f), 23);
             });
  check_grad("causal_mask+softmax", {rnd({2, 2, 5, 5}, 20)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].causal_mask().softmax_last(), 24);
  });
  check_grad("sum_all", {rnd({3, 3}, 21)},
             [](std::vector<Tensor>& v) { return v[0].sum_all(); });
  check_grad("mean_all", {rnd({3, 3}, 22)},
             [](std::vector<Tensor>& v) { return v[0].mean_all(); });
  check_grad("matmul 2d", {rnd({4, 5}, 23), rnd({5, 3}, 24)}, [](std::vector<Tensor>& v) {
    return to_scalar(v[0].matmul(v[1]), 25);
  });
  check_grad("matmul batched 4d", {rnd({2, 3, 4, 6}, 25), rnd({2, 3, 6, 5}, 26)},
             [](std::vector<Tensor>& v) { return to_scalar(v[0].matmul(v[1]), 26); });
  check_grad("linear+bias", {rnd({3, 4, 6}, 27), rnd({6, 5}, 28), rnd({5}, 29)},
             [](std::vector<Tensor>& v) {
               return to_scalar(linear(v[0], v[1], &v[2]), 27);
             });
  check_grad("embedding", {rnd({7, 4}, 30)}, [](std::vector<Tensor>& v) {
    const std::vector<int32_t> ids = {0, 3, 6, 3, 1, 5};
    return to_scalar(embedding(v[0], ids, 2, 3), 28);
  });
  check_grad("cross_entropy", {rnd({6, 9}, 31)}, [](std::vector<Tensor>& v) {
    const std::vector<int32_t> tg = {1, 4, 8, -100, 0, 3};
    return cross_entropy(v[0], tg, -100, nullptr, nullptr);
  });
  check_grad("seq_logprob", {rnd({2, 3, 7}, 32)}, [](std::vector<Tensor>& v) {
    const std::vector<int32_t> tg = {1, 3, 6, 0, -100, 5};
    return to_scalar(seq_logprob(v[0], tg, -100), 29);
  });
  check_grad("logsigmoid (DPO)", {rnd({5}, 33)}, [](std::vector<Tensor>& v) {
    return to_scalar(logsigmoid(v[0]), 30);
  });

  // A miniature transformer block, both plain and checkpointed, must agree.
  const int64_t B = 2, T = 4, C = 8, H = 2, D = C / H;
  auto block = [&](std::vector<Tensor>& v, bool ckpt) {
    Tensor x = v[0], wq = v[1], wo = v[2], g = v[3], b = v[4];
    // NOTE: everything used inside a checkpointed body must be captured *by
    // value* -- the body is re-invoked during the backward pass.
    auto body = [=](const Tensor& inp) {
      Tensor h1 = inp.layernorm(g, b, 1e-5f);
      Tensor q = linear(h1, wq, nullptr).reshape({B, T, H, D}).transpose(1, 2);
      Tensor att = q.matmul(q.transpose(2, 3))
                       .scale(1.0f / std::sqrt(static_cast<float>(D)))
                       .causal_mask()
                       .softmax_last();
      Tensor y = att.matmul(q).transpose(1, 2).reshape({B, T, C});
      return inp.add(linear(y, wo, nullptr).gelu());
    };
    Tensor out = ckpt ? checkpoint(body, x) : body(x);
    return to_scalar(out, 31);
  };
  check_grad("mini transformer block", {rnd({B, T, C}, 40), rnd({C, C}, 41, 0.3f),
                                        rnd({C, C}, 42, 0.3f), Tensor::full({C}, 1.0f),
                                        Tensor::zeros({C})},
             [&](std::vector<Tensor>& v) { return block(v, false); }, 6e-3f, 3e-2f);

  // Checkpointed gradients must match the plain ones closely.
  {
    std::vector<Tensor> a = {rnd({B, T, C}, 40), rnd({C, C}, 41, 0.3f),
                             rnd({C, C}, 42, 0.3f), Tensor::full({C}, 1.0f),
                             Tensor::zeros({C})};
    std::vector<Tensor> b2;
    for (const Tensor& t : a) b2.push_back(t.clone_leaf(true));
    for (Tensor& t : a) t.set_requires_grad(true);
    block(a, false).backward();
    block(b2, true).backward();
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      std::vector<float> ga, gb;
      a[i].copy_grad_to_host(ga);
      b2[i].copy_grad_to_host(gb);
      for (size_t k = 0; k < ga.size(); ++k)
        worst = std::max<double>(worst, std::fabs(ga[k] - gb[k]) /
                                            (1e-4 + std::fabs(ga[k])));
    }
    report("checkpoint == plain grads", worst < 1e-4,
           "max rel err " + std::to_string(worst));
  }
}

}  // namespace

int main() {
  backend_init(0, false);
  std::printf("backend: %s\n\n", backend_name());
  test_gemm();
  test_ops();
  std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
  if (g_failures) std::printf("FAILURES: %d\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
