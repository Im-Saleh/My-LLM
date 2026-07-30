// SPDX-License-Identifier: Apache-2.0
//
// Token corpus with a deterministic hold-out split.
//
// The hold-out is the tail of the corpus and is *never* touched by any of the
// three self-training threads: the coordinator uses it as the single source of
// truth when deciding whether an update is allowed to land.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/rng.h"

namespace slm {

class Tokenizer;

struct Batch {
  std::vector<int32_t> ids;      // B*T
  std::vector<int32_t> targets;  // B*T (-100 == ignore)
  int64_t B = 0, T = 0;
};

class TokenDataset {
 public:
  bool load_text_file(const std::string& path, const Tokenizer& tok,
                      std::string* err = nullptr);
  void set_tokens(std::vector<int32_t> tokens, float holdout_frac = 0.05f);
  void append_text(const std::string& text, const Tokenizer& tok);

  size_t num_tokens() const { return tokens_.size(); }
  int64_t train_tokens() const { return train_end_; }
  int64_t holdout_tokens() const { return static_cast<int64_t>(tokens_.size()) - train_end_; }
  bool empty() const { return tokens_.size() < 8; }
  const std::vector<int32_t>& tokens() const { return tokens_; }

  // Random window batch from the training region.
  Batch sample_batch(int64_t B, int64_t T, Rng& rng) const;
  // Deterministic batches from the hold-out region (same every call).
  std::vector<Batch> holdout_batches(int64_t B, int64_t T, int count) const;

  // Turn a list of token sequences into padded batches (targets = next token,
  // padding positions are ignored).  Used by the continual / feedback threads.
  static std::vector<Batch> batches_from_sequences(
      const std::vector<std::vector<int32_t>>& seqs, int64_t B, int64_t T);

 private:
  std::vector<int32_t> tokens_;
  int64_t train_end_ = 0;
  float holdout_frac_ = 0.05f;
};

}  // namespace slm
