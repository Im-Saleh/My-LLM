// SPDX-License-Identifier: Apache-2.0
//
// (a) Continual learning thread.
//
// Buffers whatever the user types, and every `min_samples` items (or after
// `max_wait_s` seconds) runs one very small fine-tuning round on them.  Two
// safeguards against catastrophic forgetting:
//   * the learning rate is one to two orders of magnitude below pre-training,
//   * every round mixes replay batches from the original corpus, so the update
//     direction is never dominated by the new samples alone.
#pragma once

#include <deque>
#include <string>

#include "interaction.h"
#include "trainer.h"

namespace slm {

struct ContinualConfig {
  int64_t min_samples = 2;
  double max_wait_s = 25.0;
  int64_t buffer_cap = 512;
  static ContinualConfig from_config(const Config& c);
};

class ContinualTrainer : public TrainerBase {
 public:
  ContinualTrainer(Coordinator* coord, Telemetry* tel, InteractionHub* hub,
                   const GPTConfig& mcfg, const TrainerConfig& tcfg,
                   const ContinualConfig& ccfg, const Tokenizer* tok,
                   const MixtureDataset* corpus, uint64_t seed);

 protected:
  bool ready() override;
  void round() override;

 private:
  InteractionHub* hub_;
  ContinualConfig ccfg_;
  std::deque<std::string> buffer_;
  double first_seen_ = 0.0;
};

}  // namespace slm
