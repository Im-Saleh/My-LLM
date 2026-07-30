// SPDX-License-Identifier: Apache-2.0
//
// Long term memory.
//
// The model has three kinds of memory, and they are deliberately separate:
//
//   1. context      what fits in the prompt window right now (volatile)
//   2. retrieval    this file: an explicit, editable, on-disk store.  Facts you
//                   write here are found by similarity and injected into the
//                   prompt as a <|system|> block, so the model can use them
//                   *immediately* without any training.
//   3. weights      the continual-learning thread consolidates memories into
//                   the parameters, so eventually the model knows them without
//                   retrieval.  `consolidate()` feeds them into that pipeline.
//
// Retrieval uses hashed character trigrams instead of a learned embedding on
// purpose: it is language agnostic (works for Persian, English and code),
// deterministic, needs no forward pass, and stays useful while the model itself
// is still changing under self-training.
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace slm {

struct MemoryItem {
  int64_t id = 0;
  std::string text;
  std::string tags;
  double created = 0.0;
  int64_t uses = 0;         // how often it was retrieved
  int64_t taught = 0;       // how often it was consolidated into the weights
  float importance = 1.0f;  // user supplied multiplier for ranking
};

class MemoryStore {
 public:
  static constexpr int kDim = 2048;

  // Opens (and creates) a JSONL file.  Existing entries are loaded.
  bool open(const std::string& path);
  const std::string& path() const { return path_; }
  bool is_open() const { return !path_.empty(); }

  int64_t add(const std::string& text, const std::string& tags = "",
              float importance = 1.0f);
  bool forget(int64_t id);
  size_t size() const;
  std::vector<MemoryItem> all() const;
  MemoryItem get(int64_t id) const;

  struct Hit {
    float score = 0.0f;
    MemoryItem item;
  };
  // Top-k by cosine similarity, blended with importance and usage recency.
  std::vector<Hit> search(const std::string& query, int k) const;

  // A ready-to-prepend prompt block, or "" when nothing is relevant enough.
  std::string context_block(const std::string& query, int k = 3,
                            size_t max_chars = 600, float min_score = 0.08f);

  // Texts to hand to the continual-learning thread (least-taught first).
  std::vector<std::string> consolidate(size_t max_items);

  // Marks ids as taught (called after they were pushed to the trainer).
  void mark_taught(const std::vector<int64_t>& ids);

  static std::vector<float> embed(const std::string& text);
  static float cosine(const std::vector<float>& a, const std::vector<float>& b);

 private:
  bool append_line(const MemoryItem& it);
  bool rewrite_all();

  mutable std::mutex m_;
  std::string path_;
  std::vector<MemoryItem> items_;
  std::vector<std::vector<float>> emb_;
  std::vector<int64_t> last_consolidated_;
  int64_t next_id_ = 1;
};

}  // namespace slm
