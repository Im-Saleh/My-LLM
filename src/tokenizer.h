// SPDX-License-Identifier: Apache-2.0
//
// Byte-level BPE tokenizer.
//
// Layout of the id space:
//   0 .. kNumSpecial-1   special/control tokens
//   kNumSpecial .. +255  the 256 raw bytes
//   then                 one id per learned merge
//
// Working on raw bytes means the tokenizer never fails on arbitrary UTF-8
// (Persian, emoji, code, ...) while still learning multi-byte units.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/text.h"

namespace slm {

class Tokenizer {
 public:
  // Control tokens (kept stable across checkpoints).
  static constexpr int32_t kEot = 0;        // <|endoftext|>
  static constexpr int32_t kUser = 1;       // <|user|>
  static constexpr int32_t kAssistant = 2;  // <|assistant|>
  static constexpr int32_t kSystem = 3;     // <|system|>
  static constexpr int32_t kNumSpecial = 4;
  static constexpr int32_t kFirstByte = kNumSpecial;
  static constexpr int32_t kBaseVocab = kNumSpecial + 256;

  Tokenizer();

  // Persian/multilingual normalisation applied to *every* string before it is
  // encoded, and to the corpus before merges are learned.  Persisting the flag
  // inside the tokenizer file is what keeps training and inference consistent.
  void set_normalize(bool on) { normalize_ = on; }
  bool normalize() const { return normalize_; }
  std::string preprocess(const std::string& text) const;

  // Learn merges from `text`. Progress callback receives (done, total).
  void train(const std::string& text, int32_t vocab_size, int min_pair_freq = 2,
             const std::function<void(int32_t, int32_t)>& progress = nullptr);

  std::vector<int32_t> encode(const std::string& text) const;
  std::string decode(const std::vector<int32_t>& ids) const;

  // Human readable form of one token (specials shown as <|..|>, control bytes
  // escaped) - used by the GUI token distribution panel.
  std::string token_display(int32_t id) const;

  int32_t vocab_size() const { return static_cast<int32_t>(pieces_.size()); }
  size_t num_merges() const { return merges_.size(); }

  bool save(const std::string& path) const;
  bool load(const std::string& path);

  // Split text the way training does (exposed for tests/tools).
  // Code-point aware: Persian punctuation (، ؛ ؟ « ») never glues to a word,
  // ZWNJ stays *inside* a word, digits are emitted one by one and a run of
  // spaces/newlines (Python indentation) is one piece.
  static std::vector<std::string> pretokenize(const std::string& text);

  // Per-language statistics for a corpus: tokens, bytes/token, and the share
  // of tokens that are a single byte (the symptom of a vocabulary that does not
  // cover a language).
  struct FertilityReport {
    size_t bytes = 0, tokens = 0, single_byte_tokens = 0, chars = 0;
    double bytes_per_token() const { return tokens ? static_cast<double>(bytes) / tokens : 0.0; }
    double chars_per_token() const { return tokens ? static_cast<double>(chars) / tokens : 0.0; }
    double single_byte_share() const {
      return tokens ? static_cast<double>(single_byte_tokens) / tokens : 0.0;
    }
  };
  FertilityReport fertility(const std::string& text) const;

 private:
  void rebuild_pieces();
  void apply_merges(std::vector<int32_t>& ids) const;

  struct PairHash {
    size_t operator()(const uint64_t& k) const noexcept {
      return std::hash<uint64_t>()(k);
    }
  };
  static uint64_t pack(int32_t a, int32_t b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
           static_cast<uint32_t>(b);
  }

  bool normalize_ = true;
  std::vector<std::pair<int32_t, int32_t>> merges_;  // id kBaseVocab+i
  std::unordered_map<uint64_t, int32_t> rank_;       // pair -> merge index
  std::vector<std::string> pieces_;                  // id -> byte string
};

}  // namespace slm
