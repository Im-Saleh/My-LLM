// SPDX-License-Identifier: Apache-2.0
#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace slm {
namespace {

enum CharClass { kSpace = 0, kLetter = 1, kDigit = 2, kOther = 3 };

inline int char_class(unsigned char c) {
  if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f')
    return kSpace;
  if (c >= 0x80) return kLetter;  // keep UTF-8 sequences together
  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') return kLetter;
  if (c >= '0' && c <= '9') return kDigit;
  return kOther;
}

constexpr size_t kMaxChunk = 48;  // hard cap so no single token explodes

const char* kSpecialNames[Tokenizer::kNumSpecial] = {
    "<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"};

}  // namespace

Tokenizer::Tokenizer() { rebuild_pieces(); }

void Tokenizer::rebuild_pieces() {
  pieces_.clear();
  pieces_.reserve(static_cast<size_t>(kBaseVocab) + merges_.size());
  for (int i = 0; i < kNumSpecial; ++i) pieces_.push_back(kSpecialNames[i]);
  for (int b = 0; b < 256; ++b) pieces_.push_back(std::string(1, static_cast<char>(b)));
  rank_.clear();
  for (size_t i = 0; i < merges_.size(); ++i) {
    const auto& m = merges_[i];
    rank_[pack(m.first, m.second)] = static_cast<int32_t>(i);
    pieces_.push_back(pieces_[static_cast<size_t>(m.first)] +
                      pieces_[static_cast<size_t>(m.second)]);
  }
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) {
  std::vector<std::string> out;
  const size_t n = text.size();
  size_t i = 0;
  while (i < n) {
    const size_t start = i;
    if (text[i] == ' ' && i + 1 < n &&
        char_class(static_cast<unsigned char>(text[i + 1])) != kSpace)
      ++i;
    const int c = char_class(static_cast<unsigned char>(text[i]));
    if (c == kSpace) {
      while (i < n && char_class(static_cast<unsigned char>(text[i])) == kSpace &&
             i - start < kMaxChunk)
        ++i;
    } else {
      while (i < n && char_class(static_cast<unsigned char>(text[i])) == c &&
             i - start < kMaxChunk)
        ++i;
    }
    out.emplace_back(text, start, i - start);
  }
  return out;
}

void Tokenizer::train(const std::string& text, int32_t vocab_size, int min_pair_freq,
                      const std::function<void(int32_t, int32_t)>& progress) {
  merges_.clear();
  rebuild_pieces();
  if (vocab_size <= kBaseVocab) return;

  // 1) word frequencies
  std::unordered_map<std::string, int64_t> freq;
  for (const std::string& w : pretokenize(text)) ++freq[w];

  std::vector<std::vector<int32_t>> words;
  std::vector<int64_t> wfreq;
  words.reserve(freq.size());
  wfreq.reserve(freq.size());
  for (const auto& kv : freq) {
    std::vector<int32_t> ids;
    ids.reserve(kv.first.size());
    for (char ch : kv.first)
      ids.push_back(kFirstByte + static_cast<int32_t>(static_cast<unsigned char>(ch)));
    words.push_back(std::move(ids));
    wfreq.push_back(kv.second);
  }

  // 2) pair statistics with an inverted index so merges are incremental
  std::unordered_map<uint64_t, int64_t> counts;
  std::unordered_map<uint64_t, std::unordered_set<int32_t>> where;
  auto touch = [&](int32_t wi, int64_t sign) {
    const std::vector<int32_t>& w = words[static_cast<size_t>(wi)];
    for (size_t k = 0; k + 1 < w.size(); ++k) {
      const uint64_t p = pack(w[k], w[k + 1]);
      const int64_t c = (counts[p] += sign * wfreq[static_cast<size_t>(wi)]);
      if (sign > 0)
        where[p].insert(wi);
      else if (c <= 0)
        counts.erase(p);
    }
    if (sign < 0)
      for (size_t k = 0; k + 1 < w.size(); ++k) {
        auto it = where.find(pack(w[k], w[k + 1]));
        if (it != where.end()) it->second.erase(wi);
      }
  };
  for (int32_t wi = 0; wi < static_cast<int32_t>(words.size()); ++wi) touch(wi, +1);

  const int32_t target = vocab_size - kBaseVocab;
  for (int32_t step = 0; step < target; ++step) {
    // best pair: highest count, deterministic tie-break
    uint64_t best = 0;
    int64_t best_c = 0;
    for (const auto& kv : counts) {
      if (kv.second > best_c || (kv.second == best_c && kv.first < best)) {
        best = kv.first;
        best_c = kv.second;
      }
    }
    if (best_c < min_pair_freq) break;
    const int32_t a = static_cast<int32_t>(best >> 32);
    const int32_t b = static_cast<int32_t>(best & 0xffffffffu);
    const int32_t new_id = static_cast<int32_t>(pieces_.size());

    std::vector<int32_t> touched(where[best].begin(), where[best].end());
    for (int32_t wi : touched) touch(wi, -1);
    for (int32_t wi : touched) {
      std::vector<int32_t>& w = words[static_cast<size_t>(wi)];
      std::vector<int32_t> nw;
      nw.reserve(w.size());
      for (size_t k = 0; k < w.size();) {
        if (k + 1 < w.size() && w[k] == a && w[k + 1] == b) {
          nw.push_back(new_id);
          k += 2;
        } else {
          nw.push_back(w[k]);
          ++k;
        }
      }
      w.swap(nw);
    }
    merges_.emplace_back(a, b);
    rank_[best] = static_cast<int32_t>(merges_.size()) - 1;
    pieces_.push_back(pieces_[static_cast<size_t>(a)] + pieces_[static_cast<size_t>(b)]);
    for (int32_t wi : touched) touch(wi, +1);
    counts.erase(best);
    where.erase(best);
    if (progress && (step % 64 == 0)) progress(step, target);
  }
  if (progress) progress(target, target);
}

void Tokenizer::apply_merges(std::vector<int32_t>& ids) const {
  while (ids.size() >= 2) {
    int32_t best_rank = -1;
    size_t best_i = 0;
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
      auto it = rank_.find(pack(ids[i], ids[i + 1]));
      if (it == rank_.end()) continue;
      if (best_rank < 0 || it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_rank < 0) break;
    ids[best_i] = kBaseVocab + best_rank;
    ids.erase(ids.begin() + static_cast<long>(best_i) + 1);
  }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
  std::vector<int32_t> out;
  out.reserve(text.size() / 3 + 8);
  std::unordered_map<std::string, std::vector<int32_t>> cache;
  size_t i = 0;
  const size_t n = text.size();
  while (i < n) {
    // Inline detection of the literal special-token spellings.
    if (text[i] == '<' && i + 1 < n && text[i + 1] == '|') {
      bool matched = false;
      for (int s = 0; s < kNumSpecial; ++s) {
        const std::string& name = pieces_[static_cast<size_t>(s)];
        if (text.compare(i, name.size(), name) == 0) {
          out.push_back(s);
          i += name.size();
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }
    // Otherwise consume one pre-token chunk.
    size_t j = i;
    if (text[j] == ' ' && j + 1 < n &&
        char_class(static_cast<unsigned char>(text[j + 1])) != kSpace)
      ++j;
    const int c = char_class(static_cast<unsigned char>(text[j]));
    if (c == kSpace)
      while (j < n && char_class(static_cast<unsigned char>(text[j])) == kSpace &&
             j - i < kMaxChunk)
        ++j;
    else
      while (j < n && char_class(static_cast<unsigned char>(text[j])) == c &&
             j - i < kMaxChunk) {
        if (text[j] == '<' && j > i && j + 1 < n && text[j + 1] == '|') break;
        ++j;
      }
    if (j == i) ++j;
    const std::string chunk(text, i, j - i);
    auto it = cache.find(chunk);
    if (it == cache.end()) {
      std::vector<int32_t> ids;
      ids.reserve(chunk.size());
      for (char ch : chunk)
        ids.push_back(kFirstByte + static_cast<int32_t>(static_cast<unsigned char>(ch)));
      apply_merges(ids);
      it = cache.emplace(chunk, std::move(ids)).first;
    }
    out.insert(out.end(), it->second.begin(), it->second.end());
    i = j;
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string out;
  for (int32_t id : ids) {
    if (id < 0 || id >= static_cast<int32_t>(pieces_.size())) continue;
    if (id < kNumSpecial) continue;  // control tokens are not rendered
    out += pieces_[static_cast<size_t>(id)];
  }
  return out;
}

std::string Tokenizer::token_display(int32_t id) const {
  if (id < 0 || id >= static_cast<int32_t>(pieces_.size())) return "<?>";
  if (id < kNumSpecial) return pieces_[static_cast<size_t>(id)];
  std::string s = pieces_[static_cast<size_t>(id)];
  std::string out;
  for (unsigned char c : s) {
    if (c == '\n')
      out += "\\n";
    else if (c == '\t')
      out += "\\t";
    else if (c == '\r')
      out += "\\r";
    else if (c == ' ')
      out += "\xc2\xb7";  // middle dot for visible spaces
    else if (c < 0x20)
      out += '?';
    else
      out += static_cast<char>(c);
  }
  return out;
}

bool Tokenizer::save(const std::string& path) const {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << "SLMTOK 1\n";
  f << "specials " << kNumSpecial << "\n";
  f << "merges " << merges_.size() << "\n";
  for (const auto& m : merges_) f << m.first << " " << m.second << "\n";
  return f.good();
}

bool Tokenizer::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::string magic;
  int version = 0;
  f >> magic >> version;
  if (magic != "SLMTOK" || version != 1) return false;
  std::string key;
  int32_t specials = 0;
  size_t nmerges = 0;
  f >> key >> specials;
  if (key != "specials" || specials != kNumSpecial) return false;
  f >> key >> nmerges;
  if (key != "merges") return false;
  merges_.clear();
  merges_.reserve(nmerges);
  for (size_t i = 0; i < nmerges; ++i) {
    int32_t a = 0, b = 0;
    if (!(f >> a >> b)) return false;
    merges_.emplace_back(a, b);
  }
  rebuild_pieces();
  return true;
}

}  // namespace slm
