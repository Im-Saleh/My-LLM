// SPDX-License-Identifier: Apache-2.0
//
// Local codebase index: scan a repository, chunk it by structure, retrieve with
// a hybrid of lexical and vector search, and hand the model a context block that
// fits its window.
//
// Chunking: symbols, not lines
// ----------------------------
// Splitting code every N lines is the single biggest quality mistake in code
// RAG: it cuts function bodies in half, so neither half retrieves and neither
// half is understandable.  This indexer walks the file and cuts at *structural*
// boundaries instead - a chunk is normally one function, method or class, taken
// together with the comment block immediately above it (which is usually the
// best description of it that exists).  Brace languages are cut on
// nesting-depth-0 boundaries, Python on `def`/`class` at the same indentation.
// Oversized bodies are split further at statement level, and each piece keeps a
// header line naming the file and the enclosing symbol so a retrieved fragment
// is never anonymous.
//
// Retrieval: hybrid, because code is half prose and half identifiers
// -----------------------------------------------------------------
//   * BM25 over subword tokens.  Unbeatable when the query contains an exact
//     identifier - and in a codebase most queries do ("where is qpack_synthesise
//     defined").  Okapi BM25 with k1=1.2, b=0.75 over a token stream that splits
//     camelCase and snake_case, so `qpack_synthesise` also matches `qpack` and
//     `synthesise`.
//   * Dense vectors for the paraphrase half ("what does this project do", "the
//     part that decides whether an update is accepted"), where the query shares
//     no identifier with the answer.
//   * Fusion by Reciprocal Rank Fusion, score = sum 1/(60 + rank).  RRF needs no
//     score calibration between the two very differently-scaled retrievers,
//     which is exactly why it is the standard choice.
//   * Then a light structural re-rank: exact symbol-name match, path match, and
//     a small penalty for very long chunks.
//
// Embeddings on 16 GB / 2 GB VRAM
// ------------------------------
// Default is `HashEmbedder`: subword hashing into a fixed dimension with signed
// random projections, L2 normalised.  It is not semantic - it is a smoothed
// lexical space - but it costs nothing, needs no download, is deterministic, and
// combined with BM25 through RRF it covers most real queries.  When a GGUF
// embedding model is configured (`embedder.gguf`), `GgufEmbedder` uses it through
// llama.cpp instead; bge-small-en-v1.5 (~60 MB Q8) or nomic-embed-text-v1.5
// (~140 MB Q8) both fit trivially next to the two chat models and give real
// semantic matching.  The index records which embedder produced it and refuses
// to mix them.
//
// Vector search
// -------------
// Exact, SIMD, brute force.  For 50 k chunks at 256 dimensions one query is
// 12.8 M multiply-adds, i.e. well under 10 ms on one core - an ANN structure
// like HNSW would add a dependency, a build step and a recall cliff in exchange
// for nothing.  `needs_ann()` reports when the index has grown to the point
// (~1 M chunks) where that stops being true.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace slm {

// ------------------------------------------------------------------ embedders
class Embedder {
 public:
  virtual ~Embedder() = default;
  virtual int dim() const = 0;
  virtual std::string name() const = 0;
  // Writes `dim()` L2-normalised floats.
  virtual void embed(const std::string& text, float* out) = 0;
  virtual void embed_batch(const std::vector<std::string>& texts, float* out);
};
using EmbedderPtr = std::shared_ptr<Embedder>;

// Deterministic, dependency-free, ~1 GB/s.  Hashes character 4-grams and
// identifier subwords into `dim` buckets with a sign from a second hash.
class HashEmbedder : public Embedder {
 public:
  explicit HashEmbedder(int dim = 256) : dim_(dim) {}
  int dim() const override { return dim_; }
  std::string name() const override { return "hash-" + std::to_string(dim_); }
  void embed(const std::string& text, float* out) override;

 private:
  int dim_;
};

// Optional: a real embedding model through llama.cpp.  Returns nullptr when the
// build has no llama.cpp or the file cannot be loaded.
EmbedderPtr make_gguf_embedder(const std::string& gguf_path, int threads,
                               std::string* err);

// ------------------------------------------------------------------- chunking
struct CodeChunk {
  int32_t id = 0;
  std::string path;         // relative to the index root
  std::string lang;         // "cpp", "python", "markdown", ...
  std::string symbol;       // enclosing function/class, when known
  std::string kind;         // "function" | "class" | "block" | "doc" | "file"
  int32_t start_line = 0;   // 1-based, inclusive
  int32_t end_line = 0;
  std::string text;
  int32_t length = 0;       // tokens in the BM25 stream
  std::string header() const;  // "src/foo.cpp:120-190  Foo::bar()"
};

struct ScanOptions {
  std::string root;
  int64_t max_file_bytes = 1 << 20;   // skip anything bigger (generated blobs)
  int max_chunk_lines = 160;          // hard cap before a body is split
  int min_chunk_lines = 3;
  bool use_gitignore = true;          // honours .gitignore at every level
  bool skip_hidden = true;
  bool skip_binary = true;
  std::vector<std::string> extra_ignores;   // glob-ish suffix/substring rules
  std::vector<std::string> only_exts;       // empty = every text extension
  int64_t max_total_chunks = 200000;
};

// Splits one file into structural chunks.  Exposed for tests.
std::vector<CodeChunk> chunk_source(const std::string& path, const std::string& text,
                                    const ScanOptions& opt);
std::string language_of(const std::string& path);
// camelCase/snake_case aware tokeniser used by BM25 and the hash embedder.
std::vector<std::string> code_tokens(const std::string& text, bool split_identifiers);

enum class SearchMode { kHybrid = 0, kLexical, kDense };

struct SearchHit {
  int32_t chunk = 0;
  double score = 0.0;
  double bm25 = 0.0;
  double dense = 0.0;
  int lexical_rank = -1;
  int dense_rank = -1;
  std::string why;           // "exact symbol match", "identifier overlap", ...
};

struct IndexStats {
  int64_t files = 0, chunks = 0, bytes = 0, tokens = 0;
  int64_t skipped_ignored = 0, skipped_binary = 0, skipped_large = 0;
  double scan_seconds = 0.0, embed_seconds = 0.0;
  size_t memory_bytes = 0;
  std::string embedder;
  std::string root;
  std::vector<std::pair<std::string, int64_t>> by_language;
};

class CodebaseIndex {
 public:
  CodebaseIndex();
  ~CodebaseIndex();

  void set_embedder(EmbedderPtr e);
  std::string embedder_name() const;

  // Full (re)scan.  `progress(files_done, files_total, current_path)` is called
  // from the scanning thread; returning false from `cancel` aborts cleanly.
  bool scan(const ScanOptions& opt,
            const std::function<void(int64_t, int64_t, const std::string&)>& progress,
            std::atomic<bool>* cancel, std::string* err);

  // Re-indexes only files whose mtime/size changed since the last scan.
  bool refresh(const std::function<void(int64_t, int64_t, const std::string&)>& progress,
               std::atomic<bool>* cancel, std::string* err);

  std::vector<SearchHit> search(const std::string& query, size_t k,
                                SearchMode mode = SearchMode::kHybrid) const;
  const CodeChunk* chunk(int32_t id) const;
  std::vector<const CodeChunk*> chunks_of(const std::string& path) const;

  // Ready-to-inject context: the top chunks, de-duplicated by file, trimmed to
  // `budget_chars`, each preceded by its header line.
  std::string context_block(const std::string& query, size_t budget_chars,
                            size_t max_chunks = 12) const;

  // Definition lookup by symbol name (exact, then case-insensitive, then
  // substring).  This is what answers "where is X defined".
  std::vector<SearchHit> find_symbol(const std::string& name, size_t k) const;

  // A short structural digest of the whole repository: languages, biggest
  // files/directories, entry points, and the top-level README headings.  This is
  // what answers "what does this project do" without retrieving anything.
  std::string overview(size_t budget_chars) const;

  IndexStats stats() const;
  bool empty() const;
  bool needs_ann() const;      // true past ~1 M chunks
  const ScanOptions& options() const;

  bool save(const std::string& path, std::string* err) const;
  bool load(const std::string& path, std::string* err);
  void clear();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  mutable std::mutex m_;
};

// Fuses ranked lists with Reciprocal Rank Fusion (Cormack et al., 2009).
// Exposed for tests and reused by the web-search ranker.
std::vector<std::pair<int32_t, double>> reciprocal_rank_fusion(
    const std::vector<std::vector<int32_t>>& ranked_lists, double k = 60.0,
    const std::vector<double>& list_weights = {});

}  // namespace slm
