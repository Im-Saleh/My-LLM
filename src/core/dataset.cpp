// SPDX-License-Identifier: Apache-2.0
#include "core/dataset.h"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "tokenizer.h"

namespace slm {

bool TokenDataset::load_text_file(const std::string& path, const Tokenizer& tok,
                                  std::string* err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (err) *err = "cannot open " + path;
    return false;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  if (text.empty()) {
    if (err) *err = "empty corpus " + path;
    return false;
  }
  set_tokens(tok.encode(text), holdout_frac_);
  if (tokens_.size() < 32) {
    if (err) *err = "corpus too small after tokenisation";
    return false;
  }
  return true;
}

void TokenDataset::set_tokens(std::vector<int32_t> tokens, float holdout_frac) {
  tokens_ = std::move(tokens);
  holdout_frac_ = holdout_frac;
  const int64_t n = static_cast<int64_t>(tokens_.size());
  const int64_t hold = std::min<int64_t>(
      std::max<int64_t>(0, static_cast<int64_t>(static_cast<double>(n) * holdout_frac)),
      std::max<int64_t>(0, n / 2));
  train_end_ = n - hold;
}

void TokenDataset::append_text(const std::string& text, const Tokenizer& tok) {
  std::vector<int32_t> extra = tok.encode(text);
  std::vector<int32_t> all;
  all.reserve(tokens_.size() + extra.size());
  // keep the hold-out at the tail
  all.insert(all.end(), tokens_.begin(), tokens_.begin() + train_end_);
  all.insert(all.end(), extra.begin(), extra.end());
  all.insert(all.end(), tokens_.begin() + train_end_, tokens_.end());
  const int64_t hold = static_cast<int64_t>(tokens_.size()) - train_end_;
  tokens_ = std::move(all);
  train_end_ = static_cast<int64_t>(tokens_.size()) - hold;
}

Batch TokenDataset::sample_batch(int64_t B, int64_t T, Rng& rng) const {
  Batch b;
  b.B = B;
  b.T = T;
  b.ids.resize(static_cast<size_t>(B * T));
  b.targets.resize(static_cast<size_t>(B * T));
  const int64_t limit = train_end_ - T - 1;
  for (int64_t i = 0; i < B; ++i) {
    const int64_t off = limit > 0 ? static_cast<int64_t>(rng.below(static_cast<uint64_t>(limit))) : 0;
    for (int64_t t = 0; t < T; ++t) {
      const size_t src = static_cast<size_t>(off + t);
      b.ids[static_cast<size_t>(i * T + t)] =
          src < tokens_.size() ? tokens_[src] : 0;
      b.targets[static_cast<size_t>(i * T + t)] =
          src + 1 < tokens_.size() ? tokens_[src + 1] : -100;
    }
  }
  return b;
}

std::vector<Batch> TokenDataset::holdout_batches(int64_t B, int64_t T, int count) const {
  std::vector<Batch> out;
  const int64_t begin = train_end_;
  const int64_t avail = static_cast<int64_t>(tokens_.size()) - begin - 1;
  if (avail < T + 1) return out;
  const int64_t per_batch = B * T;
  int64_t cursor = begin;
  for (int c = 0; c < count; ++c) {
    Batch b;
    b.B = B;
    b.T = T;
    b.ids.resize(static_cast<size_t>(per_batch));
    b.targets.resize(static_cast<size_t>(per_batch));
    for (int64_t i = 0; i < B; ++i) {
      int64_t off = cursor + i * T;
      if (off + T + 1 > static_cast<int64_t>(tokens_.size()))
        off = begin + ((off - begin) % std::max<int64_t>(1, avail - T));
      for (int64_t t = 0; t < T; ++t) {
        const size_t src = static_cast<size_t>(off + t);
        b.ids[static_cast<size_t>(i * T + t)] = src < tokens_.size() ? tokens_[src] : 0;
        b.targets[static_cast<size_t>(i * T + t)] =
            src + 1 < tokens_.size() ? tokens_[src + 1] : -100;
      }
    }
    out.push_back(std::move(b));
    cursor += per_batch;
    if (cursor + T + 1 > static_cast<int64_t>(tokens_.size())) cursor = begin;
  }
  return out;
}

std::vector<Batch> TokenDataset::batches_from_sequences(
    const std::vector<std::vector<int32_t>>& seqs, int64_t B, int64_t T) {
  std::vector<Batch> out;
  for (size_t s = 0; s < seqs.size(); s += static_cast<size_t>(B)) {
    Batch b;
    b.B = B;
    b.T = T;
    b.ids.assign(static_cast<size_t>(B * T), 0);
    b.targets.assign(static_cast<size_t>(B * T), -100);
    bool any = false;
    for (int64_t i = 0; i < B; ++i) {
      const size_t si = s + static_cast<size_t>(i);
      if (si >= seqs.size()) break;
      const std::vector<int32_t>& seq = seqs[si];
      const int64_t len = std::min<int64_t>(static_cast<int64_t>(seq.size()), T + 1);
      for (int64_t t = 0; t + 1 < len; ++t) {
        b.ids[static_cast<size_t>(i * T + t)] = seq[static_cast<size_t>(t)];
        b.targets[static_cast<size_t>(i * T + t)] = seq[static_cast<size_t>(t + 1)];
        any = true;
      }
    }
    if (any) out.push_back(std::move(b));
  }
  return out;
}

}  // namespace slm
