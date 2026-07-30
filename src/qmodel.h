// SPDX-License-Identifier: Apache-2.0
//
// QModel - inference-only transformer over memory-mapped quantised weights.
//
// This is the deployment path.  The training path (model.h + the tensor facade)
// keeps everything in float32 with an autograd graph, which is what you want
// while learning and exactly what you do not want while serving:
//
//                        training GPT              QModel
//   weights              f32 in the heap           int4/int8 in a mmap'd file
//   resident memory      params * 4 bytes          only the pages touched
//   process start        read + decode whole file  mmap, ~0 ms
//   two processes        2x the memory             share the same pages
//   matmul               f32 GEMM                  integer dot products
//
// The container is a single file (.slmq) whose payload is 64-byte aligned so it
// can be mapped straight into the address space and used in place - no parsing,
// no copying, no dequantisation pass.  A 2 B parameter model is a 1.06 GB file
// and starts instantly; the kernel pages weights in on demand and evicts them
// under pressure, which is what makes "bigger than RAM" models merely slow
// instead of impossible.
//
//   .slmq layout
//   ------------
//     "SLMQ001\0"                       8 bytes
//     u32 header_bytes                  text header ("key = value", the config)
//     u32 tensor_count
//     u64 payload_offset                start of the aligned payload region
//     header text
//     tensor_count x directory entries:
//        u16 name_len | name | u8 qtype | u8 ndim | i64 dims[ndim]
//        u64 offset (absolute) | u64 bytes
//     zero padding
//     payloads, each aligned to 64 bytes
//
// Two-dimensional weights are stored **transposed** relative to the training
// layout: the training code computes `x @ W` with W as [in, out], while the
// integer kernels need each output's input vector to be contiguous, so the file
// holds [out, in].  The packer does the transpose once.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/params.h"
#include "core/quant.h"
#include "model.h"

namespace slm {

// ----------------------------------------------------------------- packing
struct QPackOptions {
  QType type = QType::Q4;         // the big matrices (attention + MLP)
  QType embed_type = QType::Q8;   // token embedding / lm head: read one row at a
                                  // time for embedding lookup, so quantisation
                                  // error there hits the input directly
  bool progress = false;
  std::string note;
};

// Pack an in-memory parameter set (or a checkpoint on disk) into a .slmq file.
bool qpack_from_params(const ParamStore& ps, const GPTConfig& cfg,
                       const QPackOptions& o, const std::string& path,
                       std::string* err);
bool qpack_from_checkpoint(const std::string& ckpt, const QPackOptions& o,
                           const std::string& out, std::string* err);

// Write a randomly initialised model of an arbitrary size *directly* in
// quantised form, one row at a time, so peak memory stays at a few megabytes.
// This is how a 2 B parameter file gets produced on a machine that could never
// hold the f32 version: it is a real, runnable model file (untrained, so it
// predicts noise - it exists to measure the engine, not the quality).
bool qpack_synthesise(const GPTConfig& cfg, const QPackOptions& o,
                      const std::string& path, uint64_t seed, std::string* err);

// Bytes a .slmq file will occupy for a given config, without writing it.
int64_t qpack_estimate_bytes(const GPTConfig& cfg, const QPackOptions& o);

// ------------------------------------------------------------------- runtime
// One mapped tensor.
struct QTensorView {
  QType type = QType::F32;
  int64_t rows = 0, cols = 0;  // [N, K] for 2-D; [n] for 1-D (rows == 1)
  int64_t row_bytes = 0;
  const uint8_t* data = nullptr;
  bool defined() const { return data != nullptr; }
  const float* f32() const { return reinterpret_cast<const float*>(data); }
};

// KV cache + scratch buffers for one generation stream.  The cache is f16, which
// halves its footprint and costs nothing measurable in quality.
struct QGenState {
  int64_t max_ctx = 0, pos = 0;
  std::vector<std::vector<uint16_t>> k, v;  // per layer, [Hkv, max_ctx, Dh]
  std::vector<float> x, h, qkv, att, ffn, scores, logits;
  QAct act;
  std::vector<int32_t> history;
  size_t cache_bytes() const;
};

class QModel {
 public:
  QModel();
  ~QModel();
  QModel(const QModel&) = delete;
  QModel& operator=(const QModel&) = delete;

  bool open(const std::string& path, std::string* err);
  void close();
  bool is_open() const { return base_ != nullptr; }

  const GPTConfig& config() const { return cfg_; }
  const Config& meta() const { return meta_; }
  const std::string& path() const { return path_; }
  int64_t file_bytes() const { return size_; }
  int64_t param_count() const { return params_; }
  double bits_per_weight() const;
  QType body_type() const { return body_type_; }
  // Resident set of the mapping (pages actually faulted in), in bytes.
  size_t resident_bytes() const;
  std::string describe() const;

  // Hint the kernel to read the whole file now (faster first token, more RSS).
  void prefetch();
  // Ask the kernel to forget the file's clean pages, so the next run really
  // starts cold.  Only useful for honest benchmarking.
  void drop_page_cache();

  // ------------------------------------------------------------- inference
  void reset(QGenState* st, int64_t max_ctx = 0) const;
  // Feed `ids` starting at st->pos; returns the logits of the last token.
  void forward(QGenState* st, const std::vector<int32_t>& ids,
               std::vector<float>* logits) const;
  void forward_token(QGenState* st, int32_t id, std::vector<float>* logits) const;

  // Convenience: fresh state, feed the whole prompt, return the last logits.
  void logits_for(const std::vector<int32_t>& ids, std::vector<float>* logits) const;

  std::vector<int32_t> generate(const std::vector<int32_t>& prompt,
                                const GenOptions& opt,
                                const std::function<bool(const GenStep&)>& on_step =
                                    nullptr) const;

  // Mean cross-entropy (nats/token) of `ids` under the model, teacher forced.
  // `chunk` = tokens per fresh context window (0 -> min(block_size, 512)).
  double eval_nats(const std::vector<int32_t>& ids, int64_t* ntok,
                   int64_t chunk = 0) const;

 private:
  const QTensorView* find(const std::string& name) const;
  const QTensorView& need(const std::string& name) const;
  void block(QGenState* st, int64_t layer, int64_t T, int64_t pos0) const;
  void head_logits(QGenState* st, const float* x, int64_t T, bool last_only,
                   std::vector<float>* out) const;

  std::string path_;
  const uint8_t* base_ = nullptr;
  int64_t size_ = 0;
  int fd_ = -1;
  bool owns_heap_ = false;  // fell back to a heap copy (no mmap available)
  GPTConfig cfg_;
  Config meta_;
  int64_t params_ = 0;
  int64_t quant_bytes_ = 0;
  QType body_type_ = QType::Q4;
  std::unordered_map<std::string, QTensorView> tensors_;
  std::vector<std::string> order_;
};

}  // namespace slm
