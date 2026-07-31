// SPDX-License-Identifier: Apache-2.0
//
// IModelBackend - the one interface the agent, the debate engine and the GUI
// talk to.  Everything above this line is model agnostic; everything below it is
// a specific runtime.
//
//   ModelBackendSPT   our own model.  Either the *live* self-training weights
//                     (read from the coordinator, so it improves while you use
//                     it) or a static .slm / .slmq file.  Small enough to fit
//                     in 2 GB of VRAM, and the only backend that can be trained.
//   ModelBackendGGUF  llama.cpp.  Loads any GGUF (OLMo 3 7B Q4_K_M is the
//                     intended one), CPU or partially offloaded, mmap'd.
//
// Why an interface at all: the debate engine must be able to put two *different*
// runtimes in the same argument, and also the *same* runtime twice with
// different personas.  Neither is possible if the caller knows the concrete type.
//
// ------------------------------------------------------------------ sessions
// A "session" is a KV cache slot.  Debate rounds re-send a prompt that shares a
// long prefix with the previous one (the question, the peer answers, the
// persona).  Re-encoding that prefix on a 7B CPU model costs seconds, so a
// backend is required to keep per-slot token history and only process the tokens
// that actually changed:
//
//   previous slot tokens : [ system | question | peer answers r1 | ask ]
//   new prompt           : [ system | question | peer answers r1 | ask | reply | peer answers r2 | ask ]
//                            \-------------- reused from the KV cache -------/ \--- newly encoded ---/
//
// The caller always passes the *whole* prompt; finding the shared prefix is the
// backend's job (`Session::common_prefix`).  That keeps the debate engine simple
// and lets each runtime use its native cache-trimming primitive
// (`KVCache::T`, `QGenState::pos`, `llama_memory_seq_rm`).
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace slm {

// ------------------------------------------------------------------ sampling
// Superset of GenOptions (model.h).  min_p and stop strings are new: min_p is
// the best single-knob truncation for small models, and debate prompts need
// stop strings because a base-ish model happily keeps writing the next turn.
struct SamplingParams {
  int max_new_tokens = 256;
  float temperature = 0.8f;
  int top_k = 40;
  float top_p = 0.95f;
  // Relative probability floor: keep tokens with p >= min_p * p_max.  Unlike
  // top_p it adapts to how peaked the distribution is, which is why it survives
  // high temperatures far better.  0 disables it.
  float min_p = 0.05f;
  float repetition_penalty = 1.08f;
  float presence_penalty = 0.0f;
  uint64_t seed = 0;
  bool stop_on_eot = true;
  std::vector<std::string> stop;  // generation ends when any of these appears
  int64_t max_prompt_tokens = 0;  // 0 = the backend's context limit

  static SamplingParams creative(uint64_t seed);   // round 0: diversity
  static SamplingParams critical(uint64_t seed);   // critique: focused
  static SamplingParams decisive(uint64_t seed);   // synthesis: near greedy
};

struct BackendCaps {
  bool trainable = false;         // weights can be updated (SPT only)
  bool attention_capture = false; // can export attention maps for the GUI
  bool logprobs = false;          // can score an arbitrary string
  bool batched = false;           // generate_many() truly decodes in parallel
  bool gpu = false;
  int max_parallel = 1;           // concurrent sessions that make sense
};

struct BackendStatus {
  bool loaded = false;
  std::string detail;             // "OLMo-3-7B-Instruct Q4_K_M, 28L, ctx 4096"
  std::string error;              // non-empty when load() failed
  int64_t params = 0;
  size_t weight_bytes = 0;        // resident weights
  size_t kv_bytes = 0;            // all live sessions
  double last_latency_s = 0.0;    // wall time of the last generate()
  double last_prompt_tps = 0.0;   // prompt (prefill) tokens/s
  double last_decode_tps = 0.0;   // generated tokens/s
  int64_t total_prompt_tokens = 0;
  int64_t total_gen_tokens = 0;
  int64_t calls = 0;
  int64_t cache_hit_tokens = 0;   // prefix tokens saved by session reuse
  double busy_seconds = 0.0;
};

struct GenChunk {
  std::string text;               // incremental piece (may be empty mid-UTF8)
  int32_t token = 0;
  float logprob = 0.0f;
  int index = 0;                  // which request, for generate_many
};

struct GenRequest {
  int slot = -1;                  // session slot; -1 = stateless (fresh cache)
  std::string prompt;
  SamplingParams sp;
  int tag = 0;                    // echoed back, for the caller's bookkeeping
};

struct GenResponse {
  std::string text;
  std::string error;
  int prompt_tokens = 0;
  int reused_tokens = 0;          // prefix taken from the KV cache
  int gen_tokens = 0;
  double seconds = 0.0;
  double mean_logprob = 0.0;      // fluency of what it produced
  bool truncated = false;         // hit max_new_tokens
  bool cancelled = false;
  int tag = 0;
  bool ok() const { return error.empty(); }
  double decode_tps() const { return seconds > 0 ? gen_tokens / seconds : 0.0; }
};

// A callback returning false cancels that generation.
using ChunkFn = std::function<bool(const GenChunk&)>;

class IModelBackend {
 public:
  virtual ~IModelBackend() = default;

  virtual std::string id() const = 0;            // stable key: "spt", "olmo"
  virtual std::string display_name() const = 0;  // for the GUI
  virtual std::string runtime() const = 0;       // "native/spt", "llama.cpp"
  virtual BackendCaps caps() const = 0;
  virtual BackendStatus status() const = 0;

  virtual bool load(std::string* err) = 0;
  virtual void unload() = 0;
  virtual bool loaded() const = 0;

  virtual int64_t context_limit() const = 0;
  virtual std::vector<int32_t> tokenize(const std::string& text) const = 0;
  virtual std::string detokenize(const std::vector<int32_t>& ids) const = 0;
  int64_t count_tokens(const std::string& text) const {
    return static_cast<int64_t>(tokenize(text).size());
  }

  // Wrap a conversation in whatever format this model expects.  SPT uses its
  // <|user|>/<|assistant|> control tokens; a GGUF model uses the chat template
  // stored in the file.  Returning a plain concatenation is a valid fallback.
  virtual std::string format_chat(const std::string& system,
                                  const std::vector<std::pair<std::string, std::string>>& turns,
                                  bool add_generation_prompt) const = 0;

  // ---------------------------------------------------------------- sessions
  virtual int open_session() = 0;
  virtual void close_session(int slot) = 0;
  virtual void reset_session(int slot) = 0;
  virtual int64_t session_tokens(int slot) const = 0;

  // ------------------------------------------------------------- generation
  virtual GenResponse generate(const GenRequest& req, const ChunkFn& on_chunk,
                               std::atomic<bool>* cancel) = 0;

  // Backends with caps().batched run these concurrently in one decode loop;
  // the default implementation is a sequential loop, which is always correct.
  virtual std::vector<GenResponse> generate_many(const std::vector<GenRequest>& reqs,
                                                 const ChunkFn& on_chunk,
                                                 std::atomic<bool>* cancel);

  // Mean log-probability per token of `text` (optionally conditioned on
  // `context`).  Used for perplexity-based answer selection, which is the
  // cheapest reliable judge signal available.
  virtual bool score(const std::string& context, const std::string& text,
                     double* mean_logprob, std::string* err) = 0;

  // Serialises calls that share hardware.  The debate engine locks *different*
  // backends in parallel and the same backend never concurrently.
  std::mutex& lock() { return mu_; }

 protected:
  std::mutex mu_;
};

using BackendPtr = std::shared_ptr<IModelBackend>;

// ------------------------------------------------------------------ registry
// Owns the backends and is the single place the GUI reads status from.
class BackendRegistry {
 public:
  void add(BackendPtr b);
  BackendPtr get(const std::string& id) const;
  std::vector<BackendPtr> all() const;
  std::vector<std::string> ids() const;
  size_t size() const;

  // Total resident bytes of every loaded backend - the number that decides
  // whether two models plus a KV cache plus an index fit in 16 GB.
  size_t weight_bytes() const;
  size_t kv_bytes() const;

 private:
  mutable std::mutex m_;
  std::vector<BackendPtr> v_;
};

}  // namespace slm
