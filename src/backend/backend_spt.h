// SPDX-License-Identifier: Apache-2.0
//
// ModelBackendSPT - our own model behind IModelBackend.
//
// Three sources, one class, because the debate engine must not care which one it
// got:
//
//   kLive        weights are read from the Coordinator.  The model therefore
//                improves *while* the session runs: every published version is
//                picked up between turns (never in the middle of one, which is
//                the RCU read side ChatEngine already uses).  This is the only
//                mode where SPT is the same object the self-training threads are
//                busy improving.
//   kCheckpoint  a static .slm file, float32 weights in the heap.
//   kQuant       a static .slmq file, int4/int8 weights mmap'd (qmodel.h).
//                This is the deployment mode: ~7 MB resident for a 14 M model.
//
// The sampler lives here rather than in GPT::generate because the backend has to
// enforce things the model layer knows nothing about: stop *strings* (which need
// detokenisation), and KV-cache prefix reuse across debate rounds.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backend/model_backend.h"
#include "core/rng.h"
#include "model.h"
#include "qmodel.h"

namespace slm {

class Tokenizer;
class Telemetry;

// How the live backend obtains the current weights.  A callback rather than a
// Coordinator* on purpose: the backend layer must not depend on the training
// layer, otherwise the dependency graph is a cycle and the agent cannot be
// built without the trainers.
using WeightSource = std::function<ParamStorePtr(uint64_t* version)>;

struct SptBackendOptions {
  enum Source { kLive = 0, kCheckpoint, kQuant };
  Source source = kCheckpoint;
  std::string path;               // .slm or .slmq (unused for kLive)
  std::string id = "spt";
  std::string display_name = "SPT";
  WeightSource weights;           // kLive only
  GPTConfig cfg;                  // kLive only (the live architecture)
  const Tokenizer* tok = nullptr; // required for kLive/kQuant; kCheckpoint may
                                  // read the path recorded in the checkpoint
  std::string tokenizer_path;     // used when `tok` is null
  Telemetry* tel = nullptr;
  int max_sessions = 8;
};

class ModelBackendSPT : public IModelBackend {
 public:
  explicit ModelBackendSPT(const SptBackendOptions& o);
  ~ModelBackendSPT() override;

  std::string id() const override { return opt_.id; }
  std::string display_name() const override { return opt_.display_name; }
  std::string runtime() const override;
  BackendCaps caps() const override;
  BackendStatus status() const override;

  bool load(std::string* err) override;
  void unload() override;
  bool loaded() const override { return loaded_; }

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
  bool score(const std::string& context, const std::string& text,
             double* mean_logprob, std::string* err) override;

  // Number of parameters right now (grows with progressive depth growth).
  int64_t param_count() const;

 private:
  struct Session {
    std::vector<int32_t> tokens;  // exactly what the KV cache holds
    KVCache kv;                   // float path
    QGenState qs;                 // quantised path
    bool primed = false;
  };

  // Pulls a newer published snapshot if one exists.  kLive only.
  void resync_weights();
  // Feeds `ids` from position `pos` and returns the logits of the last position.
  bool run_prefix(Session& s, const std::vector<int32_t>& ids, int64_t pos,
                  std::vector<float>* logits, std::string* err);
  int32_t sample(const std::vector<float>& logits, const SamplingParams& sp,
                 const std::vector<int32_t>& history, Rng& rng, float* logprob) const;
  Session* session(int slot);

  SptBackendOptions opt_;
  std::unique_ptr<GPT> model_;      // kLive / kCheckpoint
  std::unique_ptr<QModel> qmodel_;  // kQuant
  std::unique_ptr<Tokenizer> owned_tok_;
  const Tokenizer* tok_ = nullptr;
  GPTConfig cfg_;
  bool loaded_ = false;
  uint64_t weight_version_ = 0;
  std::map<int, Session> sessions_;
  int next_slot_ = 1;
  mutable BackendStatus st_;
};

}  // namespace slm
