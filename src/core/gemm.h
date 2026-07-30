// SPDX-License-Identifier: Apache-2.0
// Small, dependency-free SGEMM used by the native tensor backend.
#pragma once
#include <cstdint>

namespace slm {

// C = alpha * op(A) * op(B) + beta * C   (all row-major)
//   op(A) is [M,K], op(B) is [K,N], C is [M,N]
// ta/tb indicate that the *stored* matrix is transposed relative to op():
//   ta == false -> A stored [M,K] with row stride lda
//   ta == true  -> A stored [K,M] with row stride lda
void sgemm(bool ta, bool tb, int64_t M, int64_t N, int64_t K, float alpha,
           const float* A, int64_t lda, const float* B, int64_t ldb, float beta,
           float* C, int64_t ldc);

// Reference implementation (slow, used by tests).
void sgemm_ref(bool ta, bool tb, int64_t M, int64_t N, int64_t K, float alpha,
               const float* A, int64_t lda, const float* B, int64_t ldb,
               float beta, float* C, int64_t ldc);

// Runtime CPU feature report, e.g. "avx2+fma" or "scalar".
const char* gemm_backend_name();

// Global switch: number of OpenMP threads used by GEMM (0 = library default).
void gemm_set_num_threads(int n);
int gemm_num_threads();

}  // namespace slm
