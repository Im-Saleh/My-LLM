// SPDX-License-Identifier: Apache-2.0
//
// Common machinery for the three self-training threads.
//
// A trainer never touches the live weights.  Each round is:
//
//    1. sync()      copy the currently published weights into the local replica
//                   and remember them as the base point,
//    2. train       a few local steps on whatever data this trainer owns,
//    3. submit()    hand the flat difference to the coordinator and go to sleep.
//
// Everything is driven by a single thread per trainer, and every trainer can be
// paused/resumed independently (or killed globally by Emergency Stop) without
// affecting the others.
#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/dataset.h"
#include "core/optim.h"
#include "core/rng.h"
#include "coordinator.h"
#include "model.h"
#include "telemetry.h"
#include "tokenizer.h"

namespace slm {

struct TrainerConfig {
  FreezePolicy freeze;
  float lr = 2e-5f;
  float weight_decay = 0.0f;
  float grad_clip = 0.5f;
  int64_t local_steps = 3;
  int64_t batch = 2;
  int64_t ctx = 128;
  double min_interval_s = 3.0;
  int replay_percent = 50;  // share of replay batches mixed into a round
  std::string note;
};

class TrainerBase {
 public:
  TrainerBase(Source src, Coordinator* coord, Telemetry* tel, const GPTConfig& mcfg,
              const TrainerConfig& tcfg, const Tokenizer* tok,
              const TokenDataset* corpus, uint64_t seed);
  virtual ~TrainerBase();

  void start();
  void stop();
  Source source() const { return src_; }
  const TrainerConfig& tcfg() const { return tcfg_; }

 protected:
  // ------------------------------------------------------- to be implemented
  virtual bool ready() = 0;  // is there work to do right now?
  virtual void round() = 0;  // one full round

  // ------------------------------------------------------------- utilities
  void sync();
  // Runs one optimiser step over `b`; returns the batch loss.
  float step_on(const Batch& b);
  // CE training over a list of batches, `local_steps` passes.
  float train_batches(const std::vector<Batch>& batches, int64_t* steps_done);
  // Mixes replay batches from the base corpus (anti forgetting).
  void add_replay(std::vector<Batch>* batches, int64_t how_many);
  void submit_delta(float loss_start, float loss_end, int64_t samples,
                    int64_t steps, const std::string& note);
  void set_status(const std::string& s);
  void bump_state();
  bool should_stop() const;

  Source src_;
  Coordinator* coord_;
  Telemetry* tel_;
  const Tokenizer* tok_;
  const TokenDataset* corpus_;
  TrainerConfig tcfg_;
  GPTConfig mcfg_;
  std::unique_ptr<GPT> model_;
  std::unique_ptr<AdamW> opt_;
  Rng rng_;

  std::vector<float> base_flat_;
  uint64_t base_version_ = 0;
  TrainerState state_;

 private:
  void loop();
  std::thread th_;
  std::atomic<bool> run_{false};
  double last_round_ = -1e9;
};

}  // namespace slm
