// SPDX-License-Identifier: Apache-2.0
#include "memory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "core/text.h"

namespace slm {
namespace {

double now_wall() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string json_escape(const std::string& raw) {
  const std::string s = utf8_sanitize(raw);
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
  }
  return o;
}

// Minimal field extraction: the file is written by us, so a full JSON parser is
// unnecessary.  Unknown/missing fields fall back to defaults.
std::string json_get_str(const std::string& line, const std::string& key) {
  const std::string pat = "\"" + key + "\":\"";
  size_t p = line.find(pat);
  if (p == std::string::npos) return {};
  p += pat.size();
  std::string out;
  while (p < line.size()) {
    if (line[p] == '\\' && p + 1 < line.size()) {
      const char c = line[p + 1];
      if (c == 'n') out += '\n';
      else if (c == 't') out += '\t';
      else if (c == 'r') out += '\r';
      else out += c;
      p += 2;
      continue;
    }
    if (line[p] == '"') break;
    out += line[p++];
  }
  return out;
}

double json_get_num(const std::string& line, const std::string& key, double def) {
  const std::string pat = "\"" + key + "\":";
  size_t p = line.find(pat);
  if (p == std::string::npos) return def;
  p += pat.size();
  try {
    return std::stod(line.substr(p, 32));
  } catch (...) {
    return def;
  }
}

}  // namespace

std::vector<float> MemoryStore::embed(const std::string& text) {
  // Hashed character trigrams over the normalised text, plus whole-word
  // features.  L2 normalised so cosine is a dot product.
  std::vector<float> v(static_cast<size_t>(kDim), 0.0f);
  const std::string norm = normalize_persian(text);
  std::vector<uint32_t> cps;
  size_t i = 0;
  while (i < norm.size()) {
    uint32_t cp = utf8_next(norm, &i);
    if (cp >= 'A' && cp <= 'Z') cp += 32;  // fold ASCII case
    if (classify(cp) == CharClass::kSpace) cp = ' ';
    cps.push_back(cp);
  }
  auto bump = [&v](uint64_t h, float w) {
    v[static_cast<size_t>(h % static_cast<uint64_t>(kDim))] += w;
  };
  auto mix = [](uint64_t h, uint64_t x) {
    h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h;
  };
  for (size_t k = 0; k + 2 < cps.size(); ++k) {
    uint64_t h = 1469598103934665603ull;
    h = mix(h, cps[k]);
    h = mix(h, cps[k + 1]);
    h = mix(h, cps[k + 2]);
    bump(h, 1.0f);
  }
  // word level features (helps exact-name recall: "علی", "fib", "PostgreSQL")
  std::string word;
  for (size_t k = 0; k <= cps.size(); ++k) {
    const bool end = k == cps.size();
    const uint32_t cp = end ? ' ' : cps[k];
    if (cp == ' ' || classify(cp) == CharClass::kPunct) {
      if (word.size() >= 2) {
        uint64_t h = 14695981039346656037ull;
        for (unsigned char c : word) h = mix(h, c);
        bump(h, 2.0f);
      }
      word.clear();
    } else {
      utf8_append(cp, &word);
    }
    if (end) break;
  }
  double n = 0.0;
  for (float x : v) n += static_cast<double>(x) * x;
  n = std::sqrt(n);
  if (n > 0.0)
    for (float& x : v) x = static_cast<float>(x / n);
  return v;
}

float MemoryStore::cosine(const std::vector<float>& a, const std::vector<float>& b) {
  const size_t n = std::min(a.size(), b.size());
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
  return static_cast<float>(s);
}

bool MemoryStore::open(const std::string& path) {
  std::lock_guard<std::mutex> g(m_);
  path_ = path;
  items_.clear();
  emb_.clear();
  next_id_ = 1;
  std::ifstream f(path);
  if (f) {
    std::string line;
    while (std::getline(f, line)) {
      if (line.size() < 8 || line[0] != '{') continue;
      MemoryItem it;
      it.id = static_cast<int64_t>(json_get_num(line, "id", 0));
      it.text = json_get_str(line, "text");
      if (it.text.empty()) continue;
      it.tags = json_get_str(line, "tags");
      it.created = json_get_num(line, "t", 0);
      it.uses = static_cast<int64_t>(json_get_num(line, "uses", 0));
      it.taught = static_cast<int64_t>(json_get_num(line, "taught", 0));
      it.importance = static_cast<float>(json_get_num(line, "importance", 1.0));
      // a later line with the same id replaces the earlier one
      auto old = std::find_if(items_.begin(), items_.end(),
                              [&](const MemoryItem& o) { return o.id == it.id; });
      if (old != items_.end()) {
        const size_t idx = static_cast<size_t>(old - items_.begin());
        items_[idx] = it;
        emb_[idx] = embed(it.text);
      } else {
        items_.push_back(it);
        emb_.push_back(embed(it.text));
      }
      next_id_ = std::max(next_id_, it.id + 1);
    }
  }
  // tombstones: entries whose text is exactly "\x01forget" mark deletions
  for (size_t i = items_.size(); i-- > 0;)
    if (items_[i].text == "\x01forget") {
      items_.erase(items_.begin() + static_cast<long>(i));
      emb_.erase(emb_.begin() + static_cast<long>(i));
    }
  return true;
}

bool MemoryStore::append_line(const MemoryItem& it) {
  if (path_.empty()) return false;
  std::ofstream f(path_, std::ios::app);
  if (!f) return false;
  f << "{\"id\":" << it.id << ",\"t\":" << static_cast<int64_t>(it.created)
    << ",\"importance\":" << it.importance << ",\"uses\":" << it.uses
    << ",\"taught\":" << it.taught << ",\"tags\":\"" << json_escape(it.tags)
    << "\",\"text\":\"" << json_escape(it.text) << "\"}\n";
  return f.good();
}

bool MemoryStore::rewrite_all() {
  if (path_.empty()) return false;
  const std::string tmp = path_ + ".tmp";
  {
    std::ofstream f(tmp, std::ios::trunc);
    if (!f) return false;
  }
  std::string keep = path_;
  path_ = tmp;
  for (const MemoryItem& it : items_) append_line(it);
  path_ = keep;
  std::rename(tmp.c_str(), path_.c_str());
  return true;
}

int64_t MemoryStore::add(const std::string& text, const std::string& tags,
                         float importance) {
  const std::string clean = normalize_persian(text);
  if (clean.empty()) return -1;
  std::lock_guard<std::mutex> g(m_);
  // Exact duplicates just bump importance instead of piling up.
  for (size_t i = 0; i < items_.size(); ++i)
    if (items_[i].text == clean) {
      items_[i].importance = std::max(items_[i].importance, importance);
      return items_[i].id;
    }
  MemoryItem it;
  it.id = next_id_++;
  it.text = clean;
  it.tags = tags;
  it.created = now_wall();
  it.importance = importance;
  items_.push_back(it);
  emb_.push_back(embed(clean));
  append_line(it);
  return it.id;
}

bool MemoryStore::forget(int64_t id) {
  std::lock_guard<std::mutex> g(m_);
  auto it = std::find_if(items_.begin(), items_.end(),
                         [&](const MemoryItem& o) { return o.id == id; });
  if (it == items_.end()) return false;
  const size_t idx = static_cast<size_t>(it - items_.begin());
  items_.erase(it);
  emb_.erase(emb_.begin() + static_cast<long>(idx));
  rewrite_all();
  return true;
}

size_t MemoryStore::size() const {
  std::lock_guard<std::mutex> g(m_);
  return items_.size();
}

std::vector<MemoryItem> MemoryStore::all() const {
  std::lock_guard<std::mutex> g(m_);
  return items_;
}

MemoryItem MemoryStore::get(int64_t id) const {
  std::lock_guard<std::mutex> g(m_);
  for (const MemoryItem& it : items_)
    if (it.id == id) return it;
  return MemoryItem{};
}

std::vector<MemoryStore::Hit> MemoryStore::search(const std::string& query,
                                                  int k) const {
  const std::vector<float> q = embed(query);
  std::vector<Hit> hits;
  {
    std::lock_guard<std::mutex> g(m_);
    hits.reserve(items_.size());
    for (size_t i = 0; i < items_.size(); ++i) {
      Hit h;
      h.item = items_[i];
      const float cos = cosine(q, emb_[i]);
      // importance boosts, and a memory that has never been used gets a small
      // exploration bonus so new notes are not buried by old popular ones.
      h.score = cos * (0.75f + 0.25f * std::min(3.0f, items_[i].importance)) +
                (items_[i].uses == 0 ? 0.01f : 0.0f);
      hits.push_back(std::move(h));
    }
  }
  std::sort(hits.begin(), hits.end(),
            [](const Hit& a, const Hit& b) { return a.score > b.score; });
  if (static_cast<int>(hits.size()) > k) hits.resize(static_cast<size_t>(k));
  return hits;
}

std::string MemoryStore::context_block(const std::string& query, int k,
                                       size_t max_chars, float min_score) {
  std::vector<Hit> hits = search(query, k);
  std::string body;
  std::vector<int64_t> used;
  for (const Hit& h : hits) {
    if (h.score < min_score) continue;
    const std::string line = "- " + h.item.text + "\n";
    if (body.size() + line.size() > max_chars) break;
    body += line;
    used.push_back(h.item.id);
  }
  if (body.empty()) return {};
  {
    std::lock_guard<std::mutex> g(m_);
    for (int64_t id : used)
      for (MemoryItem& it : items_)
        if (it.id == id) ++it.uses;
  }
  return "<|system|>" + std::string("حافظه / memory:\n") + body;
}

std::vector<std::string> MemoryStore::consolidate(size_t max_items) {
  std::vector<MemoryItem> sorted;
  {
    std::lock_guard<std::mutex> g(m_);
    sorted = items_;
  }
  std::sort(sorted.begin(), sorted.end(), [](const MemoryItem& a, const MemoryItem& b) {
    if (a.taught != b.taught) return a.taught < b.taught;
    return a.importance > b.importance;
  });
  std::vector<std::string> out;
  last_consolidated_.clear();
  for (const MemoryItem& it : sorted) {
    if (out.size() >= max_items) break;
    out.push_back(it.text);
    last_consolidated_.push_back(it.id);
  }
  return out;
}

void MemoryStore::mark_taught(const std::vector<int64_t>& ids) {
  {
    std::lock_guard<std::mutex> g(m_);
    for (int64_t id : ids)
      for (MemoryItem& it : items_)
        if (it.id == id) ++it.taught;
  }
  std::lock_guard<std::mutex> g(m_);
  rewrite_all();
}

}  // namespace slm
