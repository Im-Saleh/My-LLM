// SPDX-License-Identifier: Apache-2.0
#include "core/gemm.h"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define SLM_X86 1
#include <immintrin.h>
#endif

namespace slm {
namespace {

int g_threads = 0;

bool have_avx2() {
#if SLM_X86
  static const bool v =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return v;
#else
  return false;
#endif
}

// ---------------------------------------------------------------- scalar path
void kernel_scalar(int64_t mr, int64_t nb, int64_t kb, float alpha,
                   const float* A, int64_t lda, const float* B, int64_t ldb,
                   float* C, int64_t ldc) {
  for (int64_t i = 0; i < mr; ++i) {
    float* c = C + i * ldc;
    const float* a = A + i * lda;
    for (int64_t k = 0; k < kb; ++k) {
      const float av = alpha * a[k];
      if (av == 0.0f) continue;
      const float* b = B + k * ldb;
      for (int64_t j = 0; j < nb; ++j) c[j] += av * b[j];
    }
  }
}

#if SLM_X86
// 4x16 micro kernel: C[0..3][0..nb) += alpha * A[0..3][0..kb) * B[0..kb)[0..nb)
__attribute__((target("avx2,fma"))) void kernel_avx2_m4(
    int64_t nb, int64_t kb, float alpha, const float* A, int64_t lda,
    const float* B, int64_t ldb, float* C, int64_t ldc) {
  const __m256 va = _mm256_set1_ps(alpha);
  int64_t j = 0;
  for (; j + 16 <= nb; j += 16) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 d0 = _mm256_setzero_ps(), d1 = _mm256_setzero_ps();
    for (int64_t k = 0; k < kb; ++k) {
      const float* bp = B + k * ldb + j;
      const __m256 v0 = _mm256_loadu_ps(bp);
      const __m256 v1 = _mm256_loadu_ps(bp + 8);
      const __m256 s0 = _mm256_set1_ps(A[0 * lda + k]);
      const __m256 s1 = _mm256_set1_ps(A[1 * lda + k]);
      const __m256 s2 = _mm256_set1_ps(A[2 * lda + k]);
      const __m256 s3 = _mm256_set1_ps(A[3 * lda + k]);
      a0 = _mm256_fmadd_ps(s0, v0, a0);
      a1 = _mm256_fmadd_ps(s0, v1, a1);
      b0 = _mm256_fmadd_ps(s1, v0, b0);
      b1 = _mm256_fmadd_ps(s1, v1, b1);
      c0 = _mm256_fmadd_ps(s2, v0, c0);
      c1 = _mm256_fmadd_ps(s2, v1, c1);
      d0 = _mm256_fmadd_ps(s3, v0, d0);
      d1 = _mm256_fmadd_ps(s3, v1, d1);
    }
    float* p0 = C + 0 * ldc + j;
    float* p1 = C + 1 * ldc + j;
    float* p2 = C + 2 * ldc + j;
    float* p3 = C + 3 * ldc + j;
    _mm256_storeu_ps(p0, _mm256_fmadd_ps(va, a0, _mm256_loadu_ps(p0)));
    _mm256_storeu_ps(p0 + 8, _mm256_fmadd_ps(va, a1, _mm256_loadu_ps(p0 + 8)));
    _mm256_storeu_ps(p1, _mm256_fmadd_ps(va, b0, _mm256_loadu_ps(p1)));
    _mm256_storeu_ps(p1 + 8, _mm256_fmadd_ps(va, b1, _mm256_loadu_ps(p1 + 8)));
    _mm256_storeu_ps(p2, _mm256_fmadd_ps(va, c0, _mm256_loadu_ps(p2)));
    _mm256_storeu_ps(p2 + 8, _mm256_fmadd_ps(va, c1, _mm256_loadu_ps(p2 + 8)));
    _mm256_storeu_ps(p3, _mm256_fmadd_ps(va, d0, _mm256_loadu_ps(p3)));
    _mm256_storeu_ps(p3 + 8, _mm256_fmadd_ps(va, d1, _mm256_loadu_ps(p3 + 8)));
  }
  if (j < nb) kernel_scalar(4, nb - j, kb, alpha, A, lda, B + j, ldb, C + j, ldc);
}
#endif

// Row-major NN GEMM with N/K blocking. C must already be scaled by beta.
void sgemm_nn_acc(int64_t M, int64_t N, int64_t K, float alpha, const float* A,
                  int64_t lda, const float* B, int64_t ldb, float* C,
                  int64_t ldc) {
  const int64_t NB = 192;
  const int64_t KB = 256;
  const bool avx = have_avx2();
  const int64_t m_tiles = M / 4;

  for (int64_t jj = 0; jj < N; jj += NB) {
    const int64_t nb = std::min<int64_t>(NB, N - jj);
    for (int64_t kk = 0; kk < K; kk += KB) {
      const int64_t kb = std::min<int64_t>(KB, K - kk);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && m_tiles > 1)
#endif
      for (int64_t t = 0; t < m_tiles; ++t) {
        const int64_t i = t * 4;
        const float* a = A + i * lda + kk;
        const float* b = B + kk * ldb + jj;
        float* c = C + i * ldc + jj;
#if SLM_X86
        if (avx)
          kernel_avx2_m4(nb, kb, alpha, a, lda, b, ldb, c, ldc);
        else
#endif
          kernel_scalar(4, nb, kb, alpha, a, lda, b, ldb, c, ldc);
      }
      const int64_t rest = M - m_tiles * 4;
      if (rest > 0) {
        const int64_t i = m_tiles * 4;
        kernel_scalar(rest, nb, kb, alpha, A + i * lda + kk, lda,
                      B + kk * ldb + jj, ldb, C + i * ldc + jj, ldc);
      }
    }
  }
  (void)avx;
}

void transpose_into(const float* src, int64_t rows, int64_t cols, int64_t ld,
                    std::vector<float>& dst) {
  dst.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  // src is [rows, cols] with stride ld -> dst is [cols, rows]
  const int64_t BS = 32;
  for (int64_t i0 = 0; i0 < rows; i0 += BS) {
    const int64_t i1 = std::min(rows, i0 + BS);
    for (int64_t j0 = 0; j0 < cols; j0 += BS) {
      const int64_t j1 = std::min(cols, j0 + BS);
      for (int64_t i = i0; i < i1; ++i)
        for (int64_t j = j0; j < j1; ++j)
          dst[static_cast<size_t>(j) * rows + i] = src[i * ld + j];
    }
  }
}

}  // namespace

void gemm_set_num_threads(int n) {
  g_threads = n;
#ifdef _OPENMP
  if (n > 0) omp_set_num_threads(n);
#endif
}

int gemm_num_threads() {
#ifdef _OPENMP
  return g_threads > 0 ? g_threads : omp_get_max_threads();
#else
  return 1;
#endif
}

const char* gemm_backend_name() {
#if SLM_X86
  if (have_avx2()) {
#ifdef _OPENMP
    return "avx2+fma/openmp";
#else
    return "avx2+fma";
#endif
  }
#endif
#ifdef _OPENMP
  return "scalar/openmp";
#else
  return "scalar";
#endif
}

void sgemm_ref(bool ta, bool tb, int64_t M, int64_t N, int64_t K, float alpha,
               const float* A, int64_t lda, const float* B, int64_t ldb,
               float beta, float* C, int64_t ldc) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      double s = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        const float av = ta ? A[k * lda + i] : A[i * lda + k];
        const float bv = tb ? B[j * ldb + k] : B[k * ldb + j];
        s += static_cast<double>(av) * static_cast<double>(bv);
      }
      float* c = &C[i * ldc + j];
      *c = (beta == 0.0f ? 0.0f : beta * *c) + alpha * static_cast<float>(s);
    }
  }
}

void sgemm(bool ta, bool tb, int64_t M, int64_t N, int64_t K, float alpha,
           const float* A, int64_t lda, const float* B, int64_t ldb, float beta,
           float* C, int64_t ldc) {
  if (M <= 0 || N <= 0) return;
  // Apply beta first so the kernels can pure-accumulate.
  if (beta == 0.0f) {
    for (int64_t i = 0; i < M; ++i) std::memset(C + i * ldc, 0, sizeof(float) * static_cast<size_t>(N));
  } else if (beta != 1.0f) {
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j) C[i * ldc + j] *= beta;
  }
  if (K <= 0 || alpha == 0.0f) return;

  static thread_local std::vector<float> bufA, bufB;
  const float* Ap = A;
  int64_t ldap = lda;
  const float* Bp = B;
  int64_t ldbp = ldb;
  if (ta) {  // stored [K,M] -> want [M,K]
    transpose_into(A, K, M, lda, bufA);
    Ap = bufA.data();
    ldap = K;
  }
  if (tb) {  // stored [N,K] -> want [K,N]
    transpose_into(B, N, K, ldb, bufB);
    Bp = bufB.data();
    ldbp = N;
  }
  sgemm_nn_acc(M, N, K, alpha, Ap, ldap, Bp, ldbp, C, ldc);
}

}  // namespace slm
