// SPDX-License-Identifier: Apache-2.0
//
// Tests for the quantised inference path:
//   1. encode/decode round-trip error stays inside the theoretical bound,
//   2. the AVX2 integer kernels agree with the scalar reference exactly enough,
//   3. qmatmul agrees with an f64 reference GEMM,
//   4. a packed .slmq model reproduces the f32 model's logits.
//
// The last one is the test that actually matters: it is an end-to-end check that
// the mmap container, the transposed weight layout, RoPE/GQA/SwiGLU and the
// integer kernels all agree with the training-time implementation.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/quant.h"
#include "core/rng.h"
#include "core/serialize.h"
#include "model.h"
#include "qmodel.h"

using namespace slm;

namespace {

int g_pass = 0, g_fail = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
  if (ok) {
    ++g_pass;
    std::printf("  ok   %s%s%s\n", what.c_str(), detail.empty() ? "" : "  ",
                detail.c_str());
  } else {
    ++g_fail;
    std::printf("  FAIL %s%s%s\n", what.c_str(), detail.empty() ? "" : "  ",
                detail.c_str());
  }
}

std::vector<float> gaussian(int64_t n, uint64_t seed, float stddev = 1.0f) {
  std::mt19937_64 g(seed);
  std::normal_distribution<float> d(0.0f, stddev);
  std::vector<float> v(static_cast<size_t>(n));
  for (float& x : v) x = d(g);
  return v;
}

double rel_err(double a, double b) {
  const double s = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
  return std::fabs(a - b) / s;
}

// ------------------------------------------------------------------ round trip
void test_roundtrip() {
  std::printf("[1] encode / decode round trip\n");
  const int64_t K = 4096;
  std::vector<float> x = gaussian(K, 7, 0.02f);
  double amax = 0.0;
  for (float v : x) amax = std::max<double>(amax, std::fabs(v));

  const double r4 = quantize_row_rmse(QType::Q4, x.data(), K);
  const double r8 = quantize_row_rmse(QType::Q8, x.data(), K);
  const double r16 = quantize_row_rmse(QType::F16, x.data(), K);
  std::printf("       amax %.4g   rmse q4 %.4g  q8 %.4g  f16 %.4g\n", amax, r4, r8, r16);

  // Uniform quantisation with step d has rmse d/sqrt(12); with group-wise amax
  // the step is at most amax/7, so rmse must be well under amax/8.
  check(r4 < amax / 8.0, "q4 rmse below amax/8");
  check(r8 < amax / 100.0, "q8 rmse below amax/100");
  check(r16 < amax / 1000.0, "f16 rmse below amax/1000");
  check(r8 < r4 && r16 < r8, "error ordering f16 < q8 < q4");

  // An all-zero group must survive (scale 0 -> no NaN).
  std::vector<float> z(static_cast<size_t>(kQBlock), 0.0f);
  check(quantize_row_rmse(QType::Q4, z.data(), kQBlock) == 0.0, "zero group is exact");

  // A single huge outlier must not destroy the rest of its group's neighbours in
  // *other* groups (that is the whole point of group-wise scales).
  std::vector<float> out = x;
  out[0] = 1000.0f;
  std::vector<uint8_t> buf(static_cast<size_t>(qrow_bytes(QType::Q4, K)));
  std::vector<float> back(static_cast<size_t>(K));
  quantize_row(QType::Q4, out.data(), K, buf.data());
  dequantize_row(QType::Q4, buf.data(), K, back.data());
  double se_far = 0.0;
  for (int64_t i = kQBlock; i < K; ++i) {
    const double e = out[static_cast<size_t>(i)] - back[static_cast<size_t>(i)];
    se_far += e * e;
  }
  const double rmse_far = std::sqrt(se_far / static_cast<double>(K - kQBlock));
  check(rmse_far < amax / 8.0, "outlier stays inside its group",
        "rmse elsewhere " + std::to_string(rmse_far));
}

// ------------------------------------------------------------------- kernels
void test_kernels() {
  std::printf("[2] integer dot kernels vs f64 reference (%s)\n", quant_backend_name());
  for (int64_t K : {64, 128, 384, 1024, 4096}) {
    std::vector<float> w = gaussian(K, 100 + K, 0.05f);
    std::vector<float> a = gaussian(K, 200 + K, 1.0f);

    double ref = 0.0;  // reference uses the *dequantised* weights: the kernel is
                       // only responsible for the integer arithmetic, not for
                       // the quantisation error, which test [1] covers.
    std::vector<uint8_t> wq(static_cast<size_t>(qrow_bytes(QType::Q4, K)));
    std::vector<float> wdq(static_cast<size_t>(K));
    quantize_row(QType::Q4, w.data(), K, wq.data());
    dequantize_row(QType::Q4, wq.data(), K, wdq.data());

    QAct act;
    act.set(a.data(), 1, K);
    std::vector<float> adq(static_cast<size_t>(K));
    for (int64_t g = 0; g < K / kQBlock; ++g)
      for (int64_t j = 0; j < kQBlock; ++j)
        adq[static_cast<size_t>(g * kQBlock + j)] =
            static_cast<float>(act.q[static_cast<size_t>(g * kQBlock + j)]) *
            act.scale[static_cast<size_t>(g)];
    for (int64_t i = 0; i < K; ++i)
      ref += static_cast<double>(wdq[static_cast<size_t>(i)]) *
             adq[static_cast<size_t>(i)];

    const float got = qdot_q4(wq.data(), act.row(0), act.row_scale(0), act.row_sum(0), K);
    check(rel_err(got, ref) < 1e-5, "q4 dot K=" + std::to_string(K),
          "got " + std::to_string(got) + " ref " + std::to_string(ref));

    // q8
    std::vector<uint8_t> w8(static_cast<size_t>(qrow_bytes(QType::Q8, K)));
    std::vector<float> w8dq(static_cast<size_t>(K));
    quantize_row(QType::Q8, w.data(), K, w8.data());
    dequantize_row(QType::Q8, w8.data(), K, w8dq.data());
    double ref8 = 0.0;
    for (int64_t i = 0; i < K; ++i)
      ref8 += static_cast<double>(w8dq[static_cast<size_t>(i)]) *
              adq[static_cast<size_t>(i)];
    const float got8 = qdot_q8(w8.data(), act.row(0), act.row_scale(0), K);
    check(rel_err(got8, ref8) < 1e-5, "q8 dot K=" + std::to_string(K),
          "got " + std::to_string(got8) + " ref " + std::to_string(ref8));

    // f16 / f32 rows
    std::vector<uint8_t> w16(static_cast<size_t>(qrow_bytes(QType::F16, K)));
    quantize_row(QType::F16, w.data(), K, w16.data());
    std::vector<float> w16dq(static_cast<size_t>(K));
    dequantize_row(QType::F16, w16.data(), K, w16dq.data());
    double ref16 = 0.0;
    for (int64_t i = 0; i < K; ++i)
      ref16 += static_cast<double>(w16dq[static_cast<size_t>(i)]) *
               static_cast<double>(a[static_cast<size_t>(i)]);
    const float got16 = qdot_f16(w16.data(), a.data(), K);
    check(rel_err(got16, ref16) < 1e-4, "f16 dot K=" + std::to_string(K));
  }
}

// -------------------------------------------------------------------- qmatmul
void test_qmatmul() {
  std::printf("[3] qmatmul vs f64 reference\n");
  const int64_t M = 5, N = 37, K = 256;
  std::vector<float> A = gaussian(M * K, 11, 1.0f);
  std::vector<float> W = gaussian(N * K, 12, 0.05f);  // [N, K]
  std::vector<float> bias = gaussian(N, 13, 0.1f);

  const int64_t stride = qrow_bytes(QType::Q4, K);
  std::vector<uint8_t> Wq(static_cast<size_t>(N * stride));
  std::vector<float> Wdq(static_cast<size_t>(N * K));
  for (int64_t n = 0; n < N; ++n) {
    quantize_row(QType::Q4, W.data() + n * K, K, Wq.data() + n * stride);
    dequantize_row(QType::Q4, Wq.data() + n * stride, K, Wdq.data() + n * K);
  }

  QAct act;
  std::vector<float> C(static_cast<size_t>(M * N), 0.0f);
  qlinear(QType::Q4, Wq.data(), N, K, A.data(), M, bias.data(), C.data(), N, &act);

  // The only error left here is int8 *activation* quantisation (the weights are
  // dequantised into the reference), so the right metric is the norm of the
  // residual relative to the norm of the result, not per-element error: a single
  // output that happens to land near zero after 256 cancelling terms has a large
  // relative error for entirely uninteresting reasons.
  double sse = 0.0, ssr = 0.0, worst_abs = 0.0;
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) {
      double ref = bias[static_cast<size_t>(n)];
      for (int64_t k = 0; k < K; ++k)
        ref += static_cast<double>(Wdq[static_cast<size_t>(n * K + k)]) *
               static_cast<double>(A[static_cast<size_t>(m * K + k)]);
      const double e = C[static_cast<size_t>(m * N + n)] - ref;
      sse += e * e;
      ssr += ref * ref;
      worst_abs = std::max(worst_abs, std::fabs(e));
    }
  const double rel = std::sqrt(sse / ssr);
  std::printf("       relative residual %.4f  max abs %.5f\n", rel, worst_abs);
  check(rel < 0.05, "qmatmul within the int8 activation noise floor",
        "rel " + std::to_string(rel));

  // f32 rows must be *exact* to float precision.
  const int64_t s32 = qrow_bytes(QType::F32, K);
  std::vector<uint8_t> W32(static_cast<size_t>(N * s32));
  for (int64_t n = 0; n < N; ++n)
    quantize_row(QType::F32, W.data() + n * K, K, W32.data() + n * s32);
  std::vector<float> C32(static_cast<size_t>(M * N), 0.0f);
  qlinear(QType::F32, W32.data(), N, K, A.data(), M, nullptr, C32.data(), N, &act);
  double worst32 = 0.0;
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) {
      double ref = 0.0;
      for (int64_t k = 0; k < K; ++k)
        ref += static_cast<double>(W[static_cast<size_t>(n * K + k)]) *
               static_cast<double>(A[static_cast<size_t>(m * K + k)]);
      worst32 = std::max(worst32, std::fabs(C32[static_cast<size_t>(m * N + n)] - ref));
    }
  check(worst32 < 1e-4, "f32 rows are exact", "max abs err " + std::to_string(worst32));
}

// ------------------------------------------------------- end to end vs f32 GPT
void test_end_to_end() {
  std::printf("[4] packed .slmq model vs the f32 model\n");
  GPTConfig cfg;
  cfg.vocab_size = 512;
  cfg.n_layer = 3;
  cfg.n_head = 4;
  cfg.n_kv_head = 2;  // exercise GQA
  cfg.n_embd = 128;
  cfg.block_size = 64;
  cfg.tie_weights = true;
  GPT model(cfg);
  model.init_weights(4242);
  ParamStorePtr ps = model.snapshot();

  const std::string path = "/tmp/slm_qtest.slmq";
  for (QType t : {QType::F32, QType::Q8, QType::Q4}) {
    QPackOptions po;
    po.type = t;
    std::string err;
    if (!qpack_from_params(*ps, cfg, po, path, &err)) {
      check(false, std::string("pack ") + qtype_name(t), err);
      continue;
    }
    QModel qm;
    if (!qm.open(path, &err)) {
      check(false, std::string("open ") + qtype_name(t), err);
      continue;
    }
    check(qm.config().n_layer == cfg.n_layer && qm.config().n_embd == cfg.n_embd,
          std::string("config survives the round trip (") + qtype_name(t) + ")");

    std::vector<int32_t> ids;
    Rng rng(99);
    for (int i = 0; i < 24; ++i)
      ids.push_back(static_cast<int32_t>(rng.uniform() * (cfg.vocab_size - 1)));

    // f32 reference logits for the last position
    Tensor logits = model.forward(ids, 1, static_cast<int64_t>(ids.size()));
    const float* ref = logits.host_ptr() + (static_cast<int64_t>(ids.size()) - 1) *
                                               cfg.vocab_size;

    std::vector<float> got;
    qm.logits_for(ids, &got);

    // Compare as distributions: cosine similarity of the logit vectors and the
    // agreement of the arg max.  Logit *offsets* do not matter for sampling.
    double dot = 0.0, na = 0.0, nb = 0.0, maxdiff = 0.0;
    double mean_r = 0.0, mean_g = 0.0;
    for (int32_t i = 0; i < cfg.vocab_size; ++i) {
      mean_r += ref[i];
      mean_g += got[static_cast<size_t>(i)];
    }
    mean_r /= cfg.vocab_size;
    mean_g /= cfg.vocab_size;
    for (int32_t i = 0; i < cfg.vocab_size; ++i) {
      const double x = ref[i] - mean_r, y = got[static_cast<size_t>(i)] - mean_g;
      dot += x * y;
      na += x * x;
      nb += y * y;
      maxdiff = std::max(maxdiff, std::fabs(x - y));
    }
    const double cos = dot / std::sqrt(std::max(1e-12, na * nb));
    int32_t am_r = 0, am_g = 0;
    for (int32_t i = 1; i < cfg.vocab_size; ++i) {
      if (ref[i] > ref[am_r]) am_r = i;
      if (got[static_cast<size_t>(i)] > got[static_cast<size_t>(am_g)]) am_g = i;
    }
    std::printf("       %-3s cos %.6f  max|dlogit| %.4f  argmax %d vs %d\n",
                qtype_name(t), cos, maxdiff, am_r, am_g);
    const double need = (t == QType::F32) ? 0.9999 : (t == QType::Q8 ? 0.999 : 0.97);
    check(cos > need, std::string("logits agree (") + qtype_name(t) + ")",
          "cos " + std::to_string(cos));
    if (t == QType::F32)
      check(am_r == am_g, "f32 argmax identical");

    // The KV-cache path (one token at a time) must match the batch path.
    QGenState st;
    qm.reset(&st, static_cast<int64_t>(ids.size()) + 4);
    std::vector<float> step;
    for (size_t i = 0; i < ids.size(); ++i)
      qm.forward_token(&st, ids[i], &step);
    double d2 = 0.0;
    for (int32_t i = 0; i < cfg.vocab_size; ++i)
      d2 = std::max<double>(d2, std::fabs(step[static_cast<size_t>(i)] -
                                          got[static_cast<size_t>(i)]));
    check(d2 < 2e-2, std::string("kv-cache path matches batch path (") +
                         qtype_name(t) + ")",
          "max diff " + std::to_string(d2));
  }
  std::remove(path.c_str());
}

}  // namespace

int main() {
  std::printf("quantised inference tests (backend %s)\n\n", quant_backend_name());
  test_roundtrip();
  test_kernels();
  test_qmatmul();
  test_end_to_end();
  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
