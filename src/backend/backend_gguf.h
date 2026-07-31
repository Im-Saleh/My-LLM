// SPDX-License-Identifier: Apache-2.0
//
// ModelBackendGGUF - any GGUF model through llama.cpp, behind IModelBackend.
//
// The intended occupant is OLMo 3 7B Instruct at Q4_K_M (~4.5 GB).  Note the
// choice of *Instruct* over *Base*: the debate engine asks a model to criticise
// another model's answer and then to synthesise, and a base model cannot follow
// that instruction - it continues text.  A base checkpoint is fine as a
// participant that only drafts, and there is a flag for that.
//
// Why one context per session instead of one per model
// ---------------------------------------------------
// llama.cpp keeps its KV cache inside the context, addressed by sequence id.  A
// debate participant needs its cache to survive across rounds, so every session
// slot gets its own sequence id inside one shared context (n_seq_max), which is
// both cheaper than several contexts and what makes `generate_many` a single
// batched decode loop instead of N serial ones - the "batch the rounds into one
// call" optimisation.  A llama_context is not thread safe, so all access is
// serialised by IModelBackend::lock(); the parallelism that matters here is
// SPT-on-one-thread while OLMo-decodes-on-another, not two threads inside OLMo.
//
// Quantisation choice on 16 GB / 2 GB VRAM (measured guidance in the docs):
//   Q4_K_M  4.5 GB  the default: best quality per byte at this size
//   Q5_K_M  5.4 GB  ~1-2% better perplexity, ~20% slower, still fits 16 GB
//   Q4_K_S  4.2 GB  marginally faster, measurably worse
//   Q3_K_M  3.6 GB  only if RAM is genuinely tight; the quality drop is visible
// AWQ is deliberately not an option here: it is a GPU-oriented format and
// llama.cpp does not consume it, so on this hardware it is not on the table.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backend/model_backend.h"

namespace slm {

class Telemetry;

struct GgufBackendOptions {
  std::string path;
  std::string id = "olmo";
  std::string display_name = "OLMo 3 7B";
  int n_ctx = 4096;
  int n_threads = 0;          // 0 = hardware_concurrency
  int n_gpu_layers = 0;       // 0 = pure CPU; a 2 GB card can hold a few layers
  int n_seq_max = 4;          // debate participants sharing this context
  int n_batch = 512;
  bool use_mmap = true;
  bool use_mlock = false;
  bool flash_attn = true;
  bool kv_q8 = false;         // int8 KV cache: halves it, costs ~nothing
  bool base_model = false;    // true = cannot follow instructions, draft only
  bool embeddings = false;    // load as an embedding model instead
  Telemetry* tel = nullptr;
};

// Always constructible.  When the build has no llama.cpp, load() fails with a
// clear message and everything else degrades to empty - so the GUI can still
// list the backend and explain why it is unavailable.
class ModelBackendGGUF : public IModelBackend {
 public:
  explicit ModelBackendGGUF(const GgufBackendOptions& o);
  ~ModelBackendGGUF() override;

  static bool compiled_in();
  static std::string llama_version();

  std::string id() const override;
  std::string display_name() const override;
  std::string runtime() const override;
  BackendCaps caps() const override;
  BackendStatus status() const override;

  bool load(std::string* err) override;
  void unload() override;
  bool loaded() const override;

  int64_t context_limit() const override;
  std::vector<int32_t> tokenize(const std::string& text) const override;
  std::string detokenize(const std::vector<int32_t>& ids) const override;
  std::string format_chat(const std::string& system,
                          const std::vector<std::pair<std::string, std::string>>& turns,
                          bool add_generation_prompt) const override;

  int open_session() override;
  void close_session(int slot) override;
  void reset_session(int slot) override;
  int64_t session_tokens(int slot) const override;

  GenResponse generate(const GenRequest& req, const ChunkFn& on_chunk,
                       std::atomic<bool>* cancel) override;
  std::vector<GenResponse> generate_many(const std::vector<GenRequest>& reqs,
                                         const ChunkFn& on_chunk,
                                         std::atomic<bool>* cancel) override;
  bool score(const std::string& context, const std::string& text,
             double* mean_logprob, std::string* err) override;

  // Mean-pooled embedding of `text`; only works when opened with embeddings=true.
  bool embed(const std::string& text, std::vector<float>* out, std::string* err);
  int embedding_dim() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace slm
