// SPDX-License-Identifier: Apache-2.0
#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include "core/text.h"

namespace slm {
namespace {

constexpr size_t kMaxChunkChars = 24;  // hard cap so that no token explodes
constexpr size_t kMaxDigitRun = 1;     // one token per digit -> better arithmetic

const char* kSpecialNames[Tokenizer::kNumSpecial] = {
    "<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"};

// Returns the end offset of the pre-token that starts at byte offset `i`.
//
// Code-point aware on purpose:
//   * one leading space joins the following word (GPT-2 behaviour),
//   * a run of spaces/newlines is one piece (Python indentation),
//   * ZWNJ (U+200C) has the Arabic class, so "می‌رود" stays a single piece,
//   * Persian punctuation (، ؛ ؟ « ») is its own piece and never glues to a word,
//   * digits are emitted one by one.
size_t chunk_end(const std::string& t, size_t i) {
  const size_t n = t.size();
  size_t j = i;
  uint32_t cp = utf8_next(t, &j);
  size_t after_first = j;
  if (cp == ' ' && j < n) {
    size_t k = j;
    const uint32_t nxt = utf8_next(t, &k);
    if (classify(nxt) != CharClass::kSpace) {
      cp = nxt;
      after_first = k;
    }
  }
  const CharClass cls = classify(cp);
  size_t chars = 1;
  size_t end = after_first;
  const size_t cap = (cls == CharClass::kDigit)
                         ? kMaxDigitRun
                         : (cls == CharClass::kPunct ? 1 : kMaxChunkChars);
  while (end < n && chars < cap) {
    size_t k = end;
    const uint32_t next = utf8_next(t, &k);
    if (next == '<' && k < n && t[k] == '|') break;  // control token spelling
    if (classify(next) != cls) break;
    end = k;
    ++chars;
  }
  return end;
}

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

std::string Tokenizer::preprocess(const std::string& text) const {
  return normalize_ ? normalize_persian(text) : text;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < text.size()) {
    const size_t e = chunk_end(text, i);
    out.emplace_back(text, i, e - i);
    i = e;
  }
  return out;
}

void Tokenizer::train(const std::string& raw_text, int32_t vocab_size, int min_pair_freq,
                      const std::function<void(int32_t, int32_t)>& progress) {
  merges_.clear();
  rebuild_pieces();
  if (vocab_size <= kBaseVocab) return;
  const std::string text = preprocess(raw_text);

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

  // 2) pair statistics with an inverted index so merges stay incremental
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

std::vector<int32_t> Tokenizer::encode(const std::string& raw) const {
  const std::string text = preprocess(raw);
  std::vector<int32_t> out;
  out.reserve(text.size() / 3 + 8);
  std::unordered_map<std::string, std::vector<int32_t>> cache;
  size_t i = 0;
  const size_t n = text.size();
  while (i < n) {
    if (text[i] == '<' && i + 1 < n && text[i + 1] == '|') {
      bool matched = false;
      for (int sp = 0; sp < kNumSpecial; ++sp) {
        const std::string& name = pieces_[static_cast<size_t>(sp)];
        if (text.compare(i, name.size(), name) == 0) {
          out.push_back(sp);
          i += name.size();
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }
    const size_t e = chunk_end(text, i);
    const std::string chunk(text, i, e - i);
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
    i = e;
  }
  return out;
}

Tokenizer::FertilityReport Tokenizer::fertility(const std::string& raw) const {
  const std::string text = preprocess(raw);
  FertilityReport r;
  r.bytes = text.size();
  r.chars = utf8_length(text);
  const std::vector<int32_t> ids = encode(raw);
  r.tokens = ids.size();
  for (int32_t id : ids)
    if (id >= kFirstByte && id < kBaseVocab) ++r.single_byte_tokens;
  return r;
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
  const std::string& s = pieces_[static_cast<size_t>(id)];
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    const size_t start = i;
    const uint32_t cp = utf8_next(s, &i);
    if (cp == '\n') out += "\\n";
    else if (cp == '\t') out += "\\t";
    else if (cp == '\r') out += "\\r";
    else if (cp == ' ') out += "\xc2\xb7";       // visible space
    else if (cp == 0x200C) out += "\xe2\x80\xa2";  // visible ZWNJ (bullet)
    else if (cp >= 0xDC80 && cp <= 0xDCFF) out += '?';  // half a UTF-8 sequence
    else if (cp < 0x20) out += '?';
    else out.append(s, start, i - start);
  }
  return out;
}

bool Tokenizer::save(const std::string& path) const {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << "SLMTOK 2\n";
  f << "specials " << kNumSpecial << "\n";
  f << "normalize " << (normalize_ ? 1 : 0) << "\n";
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
  if (magic != "SLMTOK" || (version != 1 && version != 2)) return false;
  std::string key;
  int32_t specials = 0;
  size_t nmerges = 0;
  f >> key >> specials;
  if (key != "specials" || specials != kNumSpecial) return false;
  normalize_ = false;  // v1 files predate normalisation
  f >> key;
  if (key == "normalize") {
    int flag = 0;
    f >> flag;
    normalize_ = flag != 0;
    f >> key;
  }
  if (key != "merges") return false;
  f >> nmerges;
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
