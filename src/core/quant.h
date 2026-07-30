// SPDX-License-Identifier: Apache-2.0
//
// Weight-only quantisation with integer dot products (the "PicoLM trick").
//
// Why this file exists
// -------------------
// A transformer spends >95% of its inference time in `y = W x`, and for a
// single decoded token that product is completely *memory bound*: every weight
// is touched exactly once, so tokens/second is (memory bandwidth) / (bytes per
// weight).  Storing weights as int4 instead of float32 therefore makes
// generation ~8x faster *and* makes a model 8x smaller:
//
//     2.0 B parameters, f32  ->  8.0 GB   (does not fit anywhere)
//     2.0 B parameters, f16  ->  4.0 GB   (does not fit in 2 GB of VRAM)
//     2.0 B parameters, q8   ->  2.1 GB   (borderline)
//     2.0 B parameters, q4   ->  1.06 GB  (fits, with room for the KV cache)
//
// Layout
// ------
// Weights are quantised in groups of 64 consecutive values along the *reduction*
// (K) axis, each group with its own scale.  Group-wise scaling is what makes
// 4 bits usable at all: a single scale for a whole 4096-long row is destroyed by
// one outlier, while 64-value groups keep the relative error near 1%.
//
//   Q4 row of K values:  [K/2 bytes of packed nibbles][K/64 halfs of scale]
//                        4.25 bits per weight
//   Q8 row of K values:  [K int8][K/64 halfs of scale]
//                        8.25 bits per weight
//
// Inside one group the nibbles are interleaved llama.cpp-style: byte j holds
// value j in its low nibble and value j+32 in its high nibble, so a single
// 32-byte AVX2 load yields all 64 weights of the group.
//
// Arithmetic
// ----------
// Activations are quantised on the fly to int8 (also per 64-value group, one
// pass, negligible cost) and the dot product runs entirely in integers:
//
//   sum_j w_j a_j  =  d_w * d_a * ( sum_j (wu_j * aq_j) - 8 * sum_j aq_j )
//
// where wu = w + 8 is the *unsigned* nibble.  The bracket is computed with
// `vpmaddubsw` + `vpmaddwd`, i.e. 64 multiply-accumulates per two instructions,
// and the -8 correction only needs the per-group sum of the activations, which
// the activation quantiser produces for free.
//
// Everything degrades gracefully: without AVX2 the scalar path produces bitwise
// identical results (integer arithmetic), just slower.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace slm {

// Values per quantisation group.  64 is the sweet spot: small enough that a
// group has one dominant magnitude, large enough that the scale overhead is
// 0.25 bits/weight and that one group is exactly one AVX2 register pair.
constexpr int64_t kQBlock = 64;

enum class QType : uint8_t { F32 = 0, F16 = 1, Q8 = 2, Q4 = 3 };

const char* qtype_name(QType t);
bool parse_qtype(const std::string& s, QType* out);
// Bytes needed for one row of K values (K must be a multiple of kQBlock for the
// quantised types).
int64_t qrow_bytes(QType t, int64_t K);
double qtype_bits(QType t);
bool qtype_is_quantised(QType t);

// ------------------------------------------------------------------ encoding
// Encode / decode a single row of K values.  `dst` must have qrow_bytes() room.
// For Q4 the scale of each group is chosen by a small search that minimises the
// squared error instead of just using amax/8; it costs nothing at pack time and
// removes roughly 15% of the quantisation error.
void quantize_row(QType t, const float* src, int64_t K, void* dst);
void dequantize_row(QType t, const void* src, int64_t K, float* dst);

// Root-mean-square error of a round trip, for diagnostics / tests.
double quantize_row_rmse(QType t, const float* src, int64_t K);

// ---------------------------------------------------------------- activations
// int8-quantised activation block: M rows of K values, one scale and one sum per
// 64-value group.  Reused across calls so the buffers are allocated once.
struct QAct {
  int64_t M = 0, K = 0, G = 0;
  std::vector<int8_t> q;      // M * K
  std::vector<float> scale;   // M * G
  std::vector<int32_t> sum;   // M * G  (sum of the int8 codes of the group)

  void set(const float* a, int64_t M, int64_t K);
  const int8_t* row(int64_t m) const { return q.data() + m * K; }
  const float* row_scale(int64_t m) const { return scale.data() + m * G; }
  const int32_t* row_sum(int64_t m) const { return sum.data() + m * G; }
};

// ------------------------------------------------------------------- kernels
// Dot product of one quantised weight row with one quantised activation row.
float qdot_q4(const uint8_t* wrow, const int8_t* aq, const float* as,
              const int32_t* asum, int64_t K);
float qdot_q8(const uint8_t* wrow, const int8_t* aq, const float* as, int64_t K);
float qdot_f16(const uint8_t* wrow, const float* a, int64_t K);
float qdot_f32(const uint8_t* wrow, const float* a, int64_t K);

// C[M, N] = A[M, K] * W[N, K]^T (+ bias[N]).  `W` is a packed matrix of N rows,
// each qrow_bytes(t, K) long; `a` must already hold the quantised A (or, for the
// float types, `af` is used directly).  Threaded over N.
void qmatmul(QType t, const uint8_t* W, int64_t N, int64_t K, const QAct& a,
             const float* af, const float* bias, float* C, int64_t ldc);

// Convenience wrapper: quantises A into a scratch block and calls qmatmul.
void qlinear(QType t, const uint8_t* W, int64_t N, int64_t K, const float* A,
             int64_t M, const float* bias, float* C, int64_t ldc, QAct* scratch);

// Runtime kernel report, e.g. "avx2" or "scalar".
const char* quant_backend_name();

}  // namespace slm
