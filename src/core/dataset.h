// SPDX-License-Identifier: Apache-2.0
//
// Token corpora with deterministic hold-out splits.
//
//  * TokenDataset    one corpus, tail split off as hold-out
//  * MixtureDataset  several named/weighted sources (fa / en / py), each with
//                    its *own* hold-out.
//
// The per-source hold-outs are the backbone of the multilingual safety story:
// the coordinator gates every update on *each* language separately, so a round
// that improves Persian while quietly destroying Python is rejected.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/rng.h"
#include "core/text.h"

namespace slm {

class Tokenizer;

// ---------------------------------------------------------------------------
// Token storage.
//
// A corpus is tokenised once into a compact binary file (uint16 when the vocab
// fits, uint32 otherwise) and afterwards *memory mapped*: a 10 GB corpus costs
// no resident memory and starting a run is instant instead of minutes of BPE.
// This is what makes training on hundreds of millions of tokens practical on a
// 16 GB laptop.
class TokenStore {
 public:
  TokenStore() = default;
  ~TokenStore();
  TokenStore(const TokenStore&) = delete;
  TokenStore& operator=(const TokenStore&) = delete;

  // "SLMTOKB1" | u8 dtype | u64 count | u64 chars | u64 docs | payload@64
  static bool write_bin(const std::string& path, const std::vector<int32_t>& tokens,
                        int32_t vocab_size, uint64_t chars, uint64_t docs,
                        std::string* err);
  bool load_bin(const std::string& path, std::string* err);
  void set_vector(std::vector<int32_t> v);
  void set_chars(uint64_t c) { chars_ = c; }

  size_t size() const { return n_; }
  bool empty() const { return n_ == 0; }
  uint64_t chars() const { return chars_; }
  uint64_t docs() const { return docs_; }
  bool mapped() const { return map_ != nullptr; }

  inline int32_t at(size_t i) const {
    if (u16_) return static_cast<int32_t>(u16_[i]);
    if (u32_) return static_cast<int32_t>(u32_[i]);
    return owned_[i];
  }

 private:
  void reset();
  void* map_ = nullptr;
  size_t map_bytes_ = 0;
  int fd_ = -1;
  const uint16_t* u16_ = nullptr;
  const uint32_t* u32_ = nullptr;
  std::vector<int32_t> owned_;
  size_t n_ = 0;
  uint64_t chars_ = 0, docs_ = 0;
};

struct Batch {
  std::vector<int32_t> ids;      // B*T
  std::vector<int32_t> targets;  // B*T (-100 == ignore)
  int64_t B = 0, T = 0;
  Lang lang = Lang::kUnknown;
  int source = -1;
};

class TokenDataset {
 public:
  bool load_text_file(const std::string& path, const Tokenizer& tok,
                      std::string* err = nullptr);
  // Loads a pre-tokenised .bin (no tokenizer needed, memory mapped).
  bool load_bin(const std::string& path, std::string* err = nullptr,
                float holdout_frac = 0.02f);
  void set_tokens(std::vector<int32_t> tokens, float holdout_frac = 0.05f);
  void append_text(const std::string& text, const Tokenizer& tok);

  size_t num_tokens() const { return store_.size(); }
  int64_t train_tokens() const { return train_end_; }
  int64_t holdout_tokens() const { return static_cast<int64_t>(store_.size()) - train_end_; }
  bool empty() const { return store_.size() < 8; }
  const TokenStore& tokens() const { return store_; }
  uint64_t chars() const { return store_.chars(); }

  Batch sample_batch(int64_t B, int64_t T, Rng& rng) const;
  std::vector<Batch> holdout_batches(int64_t B, int64_t T, int count) const;

  static std::vector<Batch> batches_from_sequences(
      const std::vector<std::vector<int32_t>>& seqs, int64_t B, int64_t T);

  void set_holdout(float frac);

 private:
  TokenStore store_;
  int64_t train_end_ = 0;
  float holdout_frac_ = 0.05f;
};

// ---------------------------------------------------------------- mixture
struct SourceInfo {
  std::string name;   // "fa", "en", "py", ...
  std::string path;
  Lang lang = Lang::kUnknown;
  double weight = 1.0;
  size_t bytes = 0;
  size_t tokens = 0;
  double chars_per_token = 0.0;
};

class MixtureDataset {
 public:
  // spec: "fa=data/fa.txt:0.35,en=data/en.txt:0.35,py=data/py.txt:0.30"
  // The weight is optional (defaults to an equal share) and the language is
  // taken from the name when it is fa/en/py, otherwise it is auto-detected.
  bool add_spec(const std::string& spec, const Tokenizer& tok, std::string* err);
  // `path` may be a text file or a pre-tokenised .bin produced by `slm tokenize`.
  bool add_source(const std::string& name, const std::string& path, Lang lang,
                  double weight, const Tokenizer& tok, std::string* err);

  int num_sources() const { return static_cast<int>(sources_.size()); }
  bool empty() const { return sources_.empty(); }
  const SourceInfo& info(int i) const { return sources_[static_cast<size_t>(i)].info; }
  const TokenDataset& data(int i) const { return *sources_[static_cast<size_t>(i)].data; }
  int find_source(const std::string& name) const;
  int find_lang(Lang l) const;  // first source with this language, or -1

  // Weighted draw over sources, then a random window inside that source.
  Batch sample_batch(int64_t B, int64_t T, Rng& rng) const;
  Batch sample_batch_from(int source, int64_t B, int64_t T, Rng& rng) const;
  // One batch per source, so a replay mix always touches every language.
  std::vector<Batch> sample_round_robin(int64_t B, int64_t T, Rng& rng,
                                        int per_source = 1) const;
  std::vector<Batch> holdout_batches(int source, int64_t B, int64_t T, int count) const;

  size_t total_tokens() const;
  std::string describe() const;
  // Effective share of each source in the sampling distribution.
  std::vector<double> normalised_weights() const;

 private:
  struct Entry {
    SourceInfo info;
    std::unique_ptr<TokenDataset> data;
  };
  std::vector<Entry> sources_;
};

}  // namespace slm
