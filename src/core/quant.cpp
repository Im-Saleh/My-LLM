// SPDX-License-Identifier: Apache-2.0
#include "core/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "core/serialize.h"  // float_to_half / half_to_float

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define SLM_X86 1
#include <immintrin.h>
#endif

namespace slm {
namespace {

bool have_avx2() {
#if SLM_X86
  static const bool v = __builtin_cpu_supports("avx2");
  return v;
#else
  return false;
#endif
}

bool have_f16c() {
#if SLM_X86
  static const bool v = __builtin_cpu_supports("f16c");
  return v;
#else
  return false;
#endif
}

// Inline half -> float.  serialize.cpp has the fully general version, but this
// one is in the hot loop (one call per 64 multiply-accumulates), so it has to be
// inlinable and branch-light.  Scales are never inf/nan.
inline float h2f(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t mant = h & 0x3ffu;
  if (exp == 0) {
    if (mant == 0) {
      float f;
      std::memcpy(&f, &sign, 4);
      return f;
    }
    const float f = static_cast<float>(mant) * 5.9604645e-8f;  // 2^-24
    return sign ? -f : f;
  }
  const uint32_t bits = sign | ((exp + 112u) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

inline void check_groups(int64_t K) {
  if (K % kQBlock != 0)
    throw std::runtime_error("quant: reduction dimension " + std::to_string(K) +
                             " is not a multiple of " + std::to_string(kQBlock));
}

// ---------------------------------------------------------------- Q4 encoding
// Scale search: q = clamp(round(x/d), -8, 7) with d = amax / L.  L = 8 is the
// obvious choice but clips the largest positive value; slightly smaller and
// slightly larger steps often win.  Twelve candidates, evaluated on 64 values,
// is nothing at pack time.
constexpr float kQ4Cand[] = {8.0f,  7.75f, 7.5f, 7.25f, 7.0f,  6.75f,
                             6.5f,  6.25f, 6.0f, 8.25f, 8.5f,  8.75f};

float q4_best_scale(const float* x, int64_t n) {
  float amax = 0.0f;
  for (int64_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
  if (amax == 0.0f) return 0.0f;
  float best_d = amax / 8.0f;
  double best_err = 1e300;
  for (float L : kQ4Cand) {
    const float d = amax / L;
    const float inv = 1.0f / d;
    double err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      float q = std::rint(x[i] * inv);
      q = std::max(-8.0f, std::min(7.0f, q));
      const double e = static_cast<double>(x[i]) - static_cast<double>(d) * q;
      err += e * e;
    }
    if (err < best_err) {
      best_err = err;
      best_d = d;
    }
  }
  return best_d;
}

// ------------------------------------------------------------------- hsum
#if SLM_X86
__attribute__((target("avx2"))) inline int32_t hsum_i32(__m256i v) {
  const __m128i lo = _mm256_castsi256_si128(v);
  const __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(s);
}

__attribute__((target("avx2"))) inline float hsum_ps(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}
#endif

// ---------------------------------------------------------------- Q4 kernels
// A Q4 row is [K/2 nibble bytes][K/64 f16 scales].  The integer part of one
// group is  sum_j wu_j * aq_j  with wu in [0,15]; the true weight is wu-8, hence
// the  -8 * sum(aq)  correction, which is why QAct carries the group sums.
float qdot_q4_scalar(const uint8_t* wrow, const int8_t* aq, const float* as,
                     const int32_t* asum, int64_t K) {
  const int64_t G = K / kQBlock;
  const uint16_t* ws = reinterpret_cast<const uint16_t*>(wrow + K / 2);
  float acc = 0.0f;
  for (int64_t g = 0; g < G; ++g) {
    const uint8_t* nib = wrow + g * (kQBlock / 2);
    const int8_t* a = aq + g * kQBlock;
    int32_t dot = 0;
    for (int j = 0; j < kQBlock / 2; ++j) {
      const uint8_t b = nib[j];
      dot += static_cast<int32_t>(b & 0x0F) * a[j];
      dot += static_cast<int32_t>(b >> 4) * a[j + kQBlock / 2];
    }
    dot -= 8 * asum[g];
    acc += static_cast<float>(dot) * as[g] * h2f(ws[g]);
  }
  return acc;
}

#if SLM_X86
// One group: 64 weights x 64 activations in two vpmaddubsw + one vpmaddwd.
__attribute__((target("avx2"))) inline int32_t q4_group_dot(const uint8_t* nib,
                                                           const int8_t* a) {
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i wb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(nib));
  const __m256i lo = _mm256_and_si256(wb, m4);
  const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(wb, 4), m4);
  const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
  const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32));
  // u8 x i8 -> i16 pairs; |15 * 127 * 2| = 3810, no saturation
  const __m256i p =
      _mm256_add_epi16(_mm256_maddubs_epi16(lo, a0), _mm256_maddubs_epi16(hi, a1));
  return hsum_i32(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}

__attribute__((target("avx2,fma,f16c"))) float qdot_q4_avx2(const uint8_t* wrow,
                                                            const int8_t* aq,
                                                            const float* as,
                                                            const int32_t* asum,
                                                            int64_t K) {
  const int64_t G = K / kQBlock;
  const uint16_t* ws = reinterpret_cast<const uint16_t*>(wrow + K / 2);
  __m256 accv = _mm256_setzero_ps();
  alignas(32) int32_t part[8];
  int64_t g = 0;
  // Eight groups at a time so the f16 scale decode and the float scaling are
  // vectorised as well (this is the path a multi-thousand-wide model takes).
  for (; g + 8 <= G; g += 8) {
    for (int u = 0; u < 8; ++u)
      part[u] = q4_group_dot(wrow + (g + u) * (kQBlock / 2), aq + (g + u) * kQBlock);
    const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(asum + g));
    const __m256i d = _mm256_sub_epi32(
        _mm256_load_si256(reinterpret_cast<const __m256i*>(part)),
        _mm256_slli_epi32(s, 3));
    const __m256 sw =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ws + g)));
    accv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d),
                           _mm256_mul_ps(_mm256_loadu_ps(as + g), sw), accv);
  }
  float acc = hsum_ps(accv);
  for (; g < G; ++g) {
    const int32_t dot =
        q4_group_dot(wrow + g * (kQBlock / 2), aq + g * kQBlock) - 8 * asum[g];
    acc += static_cast<float>(dot) * as[g] * h2f(ws[g]);
  }
  return acc;
}
#endif

// ---------------------------------------------------------------- Q8 kernels
float qdot_q8_scalar(const uint8_t* wrow, const int8_t* aq, const float* as,
                     int64_t K) {
  const int64_t G = K / kQBlock;
  const int8_t* w = reinterpret_cast<const int8_t*>(wrow);
  const uint16_t* ws = reinterpret_cast<const uint16_t*>(wrow + K);
  float acc = 0.0f;
  for (int64_t g = 0; g < G; ++g) {
    int32_t dot = 0;
    const int8_t* wp = w + g * kQBlock;
    const int8_t* a = aq + g * kQBlock;
    for (int j = 0; j < kQBlock; ++j) dot += static_cast<int32_t>(wp[j]) * a[j];
    acc += static_cast<float>(dot) * as[g] * h2f(ws[g]);
  }
  return acc;
}

#if SLM_X86
__attribute__((target("avx2"))) inline int32_t q8_group_dot(const int8_t* wp,
                                                           const int8_t* a) {
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i s32 = _mm256_setzero_si256();
  for (int j = 0; j < kQBlock; j += 32) {
    const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + j));
    const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
    // vpmaddubsw needs an unsigned left operand: |w| * (a * sign(w)) == w * a
    s32 = _mm256_add_epi32(
        s32, _mm256_madd_epi16(
                 _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(av, wv)),
                 ones));
  }
  return hsum_i32(s32);
}

__attribute__((target("avx2,fma,f16c"))) float qdot_q8_avx2(const uint8_t* wrow,
                                                            const int8_t* aq,
                                                            const float* as,
                                                            int64_t K) {
  const int64_t G = K / kQBlock;
  const int8_t* w = reinterpret_cast<const int8_t*>(wrow);
  const uint16_t* ws = reinterpret_cast<const uint16_t*>(wrow + K);
  __m256 accv = _mm256_setzero_ps();
  alignas(32) int32_t part[8];
  int64_t g = 0;
  for (; g + 8 <= G; g += 8) {
    for (int u = 0; u < 8; ++u)
      part[u] = q8_group_dot(w + (g + u) * kQBlock, aq + (g + u) * kQBlock);
    const __m256 sw =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ws + g)));
    accv = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(_mm256_load_si256(reinterpret_cast<const __m256i*>(part))),
        _mm256_mul_ps(_mm256_loadu_ps(as + g), sw), accv);
  }
  float acc = hsum_ps(accv);
  for (; g < G; ++g)
    acc += static_cast<float>(q8_group_dot(w + g * kQBlock, aq + g * kQBlock)) *
           as[g] * h2f(ws[g]);
  return acc;
}
#endif

// --------------------------------------------------------------- float rows
float qdot_f32_scalar(const uint8_t* wrow, const float* a, int64_t K) {
  const float* w = reinterpret_cast<const float*>(wrow);
  double acc = 0.0;
  for (int64_t i = 0; i < K; ++i) acc += static_cast<double>(w[i]) * a[i];
  return static_cast<float>(acc);
}

float qdot_f16_scalar(const uint8_t* wrow, const float* a, int64_t K) {
  const uint16_t* w = reinterpret_cast<const uint16_t*>(wrow);
  double acc = 0.0;
  for (int64_t i = 0; i < K; ++i) acc += static_cast<double>(h2f(w[i])) * a[i];
  return static_cast<float>(acc);
}

#if SLM_X86
__attribute__((target("avx2,fma"))) float qdot_f32_avx2(const uint8_t* wrow,
                                                        const float* a, int64_t K) {
  const float* w = reinterpret_cast<const float*>(wrow);
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= K; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(a + i), acc);
  float s = hsum_ps(acc);
  for (; i < K; ++i) s += w[i] * a[i];
  return s;
}

__attribute__((target("avx2,fma,f16c"))) float qdot_f16_avx2(const uint8_t* wrow,
                                                             const float* a,
                                                             int64_t K) {
  const uint16_t* w = reinterpret_cast<const uint16_t*>(wrow);
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= K; i += 8) {
    const __m256 wv =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
    acc = _mm256_fmadd_ps(wv, _mm256_loadu_ps(a + i), acc);
  }
  float s = hsum_ps(acc);
  for (; i < K; ++i) s += h2f(w[i]) * a[i];
  return s;
}
#endif

}  // namespace

// =========================================================== type descriptors
const char* qtype_name(QType t) {
  switch (t) {
    case QType::F32: return "f32";
    case QType::F16: return "f16";
    case QType::Q8: return "q8";
    case QType::Q4: return "q4";
  }
  return "?";
}

bool parse_qtype(const std::string& s, QType* out) {
  if (s == "f32" || s == "fp32" || s == "float") *out = QType::F32;
  else if (s == "f16" || s == "fp16" || s == "half") *out = QType::F16;
  else if (s == "q8" || s == "int8" || s == "i8" || s == "8") *out = QType::Q8;
  else if (s == "q4" || s == "int4" || s == "i4" || s == "4") *out = QType::Q4;
  else return false;
  return true;
}

int64_t qrow_bytes(QType t, int64_t K) {
  switch (t) {
    case QType::F32: return K * 4;
    case QType::F16: return K * 2;
    case QType::Q8: check_groups(K); return K + (K / kQBlock) * 2;
    case QType::Q4: check_groups(K); return K / 2 + (K / kQBlock) * 2;
  }
  return K * 4;
}

double qtype_bits(QType t) {
  switch (t) {
    case QType::F32: return 32.0;
    case QType::F16: return 16.0;
    case QType::Q8: return 8.0 + 16.0 / kQBlock;
    case QType::Q4: return 4.0 + 16.0 / kQBlock;
  }
  return 32.0;
}

bool qtype_is_quantised(QType t) { return t == QType::Q4 || t == QType::Q8; }

// =================================================================== encoding
void quantize_row(QType t, const float* src, int64_t K, void* dst) {
  switch (t) {
    case QType::F32:
      std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(K));
      return;
    case QType::F16: {
      uint16_t* h = static_cast<uint16_t*>(dst);
      for (int64_t i = 0; i < K; ++i) h[i] = float_to_half(src[i]);
      return;
    }
    case QType::Q8: {
      check_groups(K);
      int8_t* q = static_cast<int8_t*>(dst);
      uint16_t* sc = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(dst) + K);
      for (int64_t g = 0; g < K / kQBlock; ++g) {
        const float* x = src + g * kQBlock;
        float amax = 0.0f;
        for (int64_t j = 0; j < kQBlock; ++j) amax = std::max(amax, std::fabs(x[j]));
        const float d = amax / 127.0f;
        // The scale is stored as f16, so quantise against the *stored* value:
        // otherwise the decoder uses a slightly different scale than the encoder
        // assumed and the error is larger than it needs to be.
        sc[g] = float_to_half(d);
        const float dq = half_to_float(sc[g]);
        const float inv = dq > 0.0f ? 1.0f / dq : 0.0f;
        for (int64_t j = 0; j < kQBlock; ++j) {
          float v = std::rint(x[j] * inv);
          v = std::max(-127.0f, std::min(127.0f, v));
          q[g * kQBlock + j] = static_cast<int8_t>(v);
        }
      }
      return;
    }
    case QType::Q4: {
      check_groups(K);
      uint8_t* nib = static_cast<uint8_t*>(dst);
      uint16_t* sc = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(dst) + K / 2);
      for (int64_t g = 0; g < K / kQBlock; ++g) {
        const float* x = src + g * kQBlock;
        const float d = q4_best_scale(x, kQBlock);
        sc[g] = float_to_half(d);
        const float dq = half_to_float(sc[g]);
        const float inv = dq > 0.0f ? 1.0f / dq : 0.0f;
        uint8_t* out = nib + g * (kQBlock / 2);
        for (int64_t j = 0; j < kQBlock / 2; ++j) {
          float a = std::rint(x[j] * inv);
          float b = std::rint(x[j + kQBlock / 2] * inv);
          a = std::max(-8.0f, std::min(7.0f, a));
          b = std::max(-8.0f, std::min(7.0f, b));
          const uint8_t ua = static_cast<uint8_t>(static_cast<int>(a) + 8);
          const uint8_t ub = static_cast<uint8_t>(static_cast<int>(b) + 8);
          out[j] = static_cast<uint8_t>(ua | (ub << 4));
        }
      }
      return;
    }
  }
}

void dequantize_row(QType t, const void* src, int64_t K, float* dst) {
  switch (t) {
    case QType::F32:
      std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(K));
      return;
    case QType::F16: {
      const uint16_t* h = static_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < K; ++i) dst[i] = half_to_float(h[i]);
      return;
    }
    case QType::Q8: {
      const int8_t* q = static_cast<const int8_t*>(src);
      const uint16_t* sc =
          reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(src) + K);
      for (int64_t g = 0; g < K / kQBlock; ++g) {
        const float d = half_to_float(sc[g]);
        for (int64_t j = 0; j < kQBlock; ++j)
          dst[g * kQBlock + j] = static_cast<float>(q[g * kQBlock + j]) * d;
      }
      return;
    }
    case QType::Q4: {
      const uint8_t* nib = static_cast<const uint8_t*>(src);
      const uint16_t* sc =
          reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(src) + K / 2);
      for (int64_t g = 0; g < K / kQBlock; ++g) {
        const float d = half_to_float(sc[g]);
        const uint8_t* in = nib + g * (kQBlock / 2);
        float* o = dst + g * kQBlock;
        for (int64_t j = 0; j < kQBlock / 2; ++j) {
          o[j] = static_cast<float>(static_cast<int>(in[j] & 0x0F) - 8) * d;
          o[j + kQBlock / 2] =
              static_cast<float>(static_cast<int>(in[j] >> 4) - 8) * d;
        }
      }
      return;
    }
  }
}

double quantize_row_rmse(QType t, const float* src, int64_t K) {
  std::vector<uint8_t> buf(static_cast<size_t>(qrow_bytes(t, K)));
  std::vector<float> back(static_cast<size_t>(K));
  quantize_row(t, src, K, buf.data());
  dequantize_row(t, buf.data(), K, back.data());
  double se = 0.0;
  for (int64_t i = 0; i < K; ++i) {
    const double e = static_cast<double>(src[i]) - back[static_cast<size_t>(i)];
    se += e * e;
  }
  return std::sqrt(se / static_cast<double>(K));
}

// =============================================================== activations
void QAct::set(const float* a, int64_t M_, int64_t K_) {
  check_groups(K_);
  M = M_;
  K = K_;
  G = K / kQBlock;
  q.resize(static_cast<size_t>(M * K));
  scale.resize(static_cast<size_t>(M * G));
  sum.resize(static_cast<size_t>(M * G));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M > 4)
#endif
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t g = 0; g < G; ++g) {
      const float* x = a + m * K + g * kQBlock;
      float amax = 0.0f;
      for (int64_t j = 0; j < kQBlock; ++j) amax = std::max(amax, std::fabs(x[j]));
      const float d = amax / 127.0f;
      const float inv = d > 0.0f ? 1.0f / d : 0.0f;
      int32_t s = 0;
      int8_t* out = q.data() + m * K + g * kQBlock;
      for (int64_t j = 0; j < kQBlock; ++j) {
        float v = std::rint(x[j] * inv);
        v = std::max(-127.0f, std::min(127.0f, v));
        const int8_t c = static_cast<int8_t>(v);
        out[j] = c;
        s += c;
      }
      scale[static_cast<size_t>(m * G + g)] = d;
      sum[static_cast<size_t>(m * G + g)] = s;
    }
  }
}

// =================================================================== kernels
float qdot_q4(const uint8_t* wrow, const int8_t* aq, const float* as,
              const int32_t* asum, int64_t K) {
#if SLM_X86
  // Every CPU that has AVX2 also has F16C (both arrived with Haswell), so this
  // is the path on any machine from 2013 onwards.
  if (have_avx2() && have_f16c()) return qdot_q4_avx2(wrow, aq, as, asum, K);
#endif
  return qdot_q4_scalar(wrow, aq, as, asum, K);
}

float qdot_q8(const uint8_t* wrow, const int8_t* aq, const float* as, int64_t K) {
#if SLM_X86
  if (have_avx2() && have_f16c()) return qdot_q8_avx2(wrow, aq, as, K);
#endif
  return qdot_q8_scalar(wrow, aq, as, K);
}

float qdot_f32(const uint8_t* wrow, const float* a, int64_t K) {
#if SLM_X86
  if (have_avx2()) return qdot_f32_avx2(wrow, a, K);
#endif
  return qdot_f32_scalar(wrow, a, K);
}

float qdot_f16(const uint8_t* wrow, const float* a, int64_t K) {
#if SLM_X86
  if (have_avx2() && have_f16c()) return qdot_f16_avx2(wrow, a, K);
#endif
  return qdot_f16_scalar(wrow, a, K);
}

void qmatmul(QType t, const uint8_t* W, int64_t N, int64_t K, const QAct& a,
             const float* af, const float* bias, float* C, int64_t ldc) {
  const int64_t stride = qrow_bytes(t, K);
  const int64_t M = qtype_is_quantised(t) ? a.M : (af ? a.M : 0);
  // Loop order: n outside so each weight row is streamed from memory exactly
  // once (it is the large operand); the activation rows stay in cache.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t n = 0; n < N; ++n) {
    const uint8_t* wrow = W + n * stride;
    const float b = bias ? bias[n] : 0.0f;
    switch (t) {
      case QType::Q4:
        for (int64_t m = 0; m < M; ++m)
          C[m * ldc + n] =
              b + qdot_q4(wrow, a.row(m), a.row_scale(m), a.row_sum(m), K);
        break;
      case QType::Q8:
        for (int64_t m = 0; m < M; ++m)
          C[m * ldc + n] = b + qdot_q8(wrow, a.row(m), a.row_scale(m), K);
        break;
      case QType::F16:
        for (int64_t m = 0; m < M; ++m)
          C[m * ldc + n] = b + qdot_f16(wrow, af + m * K, K);
        break;
      case QType::F32:
        for (int64_t m = 0; m < M; ++m)
          C[m * ldc + n] = b + qdot_f32(wrow, af + m * K, K);
        break;
    }
  }
}

void qlinear(QType t, const uint8_t* W, int64_t N, int64_t K, const float* A,
             int64_t M, const float* bias, float* C, int64_t ldc, QAct* scratch) {
  QAct local;
  QAct& act = scratch ? *scratch : local;
  if (qtype_is_quantised(t)) {
    act.set(A, M, K);
  } else {
    act.M = M;
    act.K = K;
  }
  qmatmul(t, W, N, K, act, A, bias, C, ldc);
}

const char* quant_backend_name() {
#if SLM_X86
  if (have_avx2()) return have_f16c() ? "avx2+f16c" : "avx2";
#endif
  return "scalar";
}

}  // namespace slm
