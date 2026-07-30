// SPDX-License-Identifier: Apache-2.0
//
// Checkpoint container (.slm).
//
//   magic "SLMCKPT" | u8 version | u8 default dtype
//   u64 step | u64 tokens_seen
//   u32 meta_len | meta text ("key = value" lines, includes the model config)
//   u32 nparams
//   nparams x { u16 name_len | name | u8 ndim | i64 dims[] | u8 dtype | u64 bytes | payload }
//
// Supported payload encodings:
//   F32  raw float32
//   F16  IEEE half (2x smaller)
//   Q8   group-wise symmetric int8, group = 64 values, one f32 scale per group
//        (~3.9x smaller; used for shipping / cold storage of weights)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/params.h"

namespace slm {

enum class Dtype : uint8_t { F32 = 0, F16 = 1, Q8 = 2 };

const char* dtype_name(Dtype d);
bool parse_dtype(const std::string& s, Dtype* out);

struct CheckpointMeta {
  int64_t step = 0;
  int64_t tokens_seen = 0;
  Config extra;  // model.* keys, tokenizer path, notes, ...
};

// 1-D tensors (biases, layernorms) are always stored as F32 because they are
// tiny and quantising them hurts quality noticeably.
bool save_checkpoint(const std::string& path, const ParamStore& ps,
                     const CheckpointMeta& meta, Dtype dtype);
bool load_checkpoint(const std::string& path, ParamStore* ps, CheckpointMeta* meta);
// Reads only the header (no weights).
bool peek_checkpoint(const std::string& path, CheckpointMeta* meta, Dtype* dtype);

// ---------------------------------------------------------- numeric encodings
uint16_t float_to_half(float f);
float half_to_float(uint16_t h);

constexpr int kQuantGroup = 64;
void quantize_q8(const float* src, int64_t n, std::vector<int8_t>* q,
                 std::vector<float>* scales);
void dequantize_q8(const int8_t* q, const float* scales, int64_t n, float* dst);

// Convenience: bytes needed to store `n` values in the given dtype.
int64_t encoded_bytes(int64_t n, Dtype d);

}  // namespace slm
