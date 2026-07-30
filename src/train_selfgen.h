// SPDX-License-Identifier: Apache-2.0
//
// (b) Self-generated data thread.
//
// Periodically samples the *current* model, then filters what it produced
// before it is allowed anywhere near a gradient step:
//
//   * perplexity band            reject degenerate loops (ppl too low) and
//                                incoherent noise (ppl too high),
//   * n-gram repetition ratio    reject "the the the the",
//   * length                     reject stubs,
//   * novelty                    reject samples that are near duplicates of
//                                what was already accepted.
//
// Only the top `keep_best` survivors of a round enter the augmentation buffer,
// and every decision (accepted or not, with the reason and the score) goes into
// the audit log.
#pragma once

#include <deque>
#include <string>
#include <unordered_set>
#include <vector>

#include "trainer.h"

namespace slm {

struct SelfGenConfig {
  int samples_per_round = 6;
  int max_new_tokens = 48;
  float temperature = 0.95f;
  int top_k = 40;
  float top_p = 0.95f;
  // Absolute backstops ...
  float ppl_min = 1.0005f;
  float ppl_max = 30.0f;
  // ... and the band that actually matters: relative to the perplexity the
  // model currently has on the hold-out set.  A fixed absolute band is wrong
  // because "degenerate" means "far more confident than this model usually is",
  // which changes as the model trains.
  float ppl_min_ratio = 0.30f;
  float ppl_max_ratio = 3.0f;
  float max_repeat_ratio = 0.34f;
  int min_tokens = 8;
  int keep_best = 3;
  int64_t buffer_cap = 192;
  bool require_novel = true;
  static SelfGenConfig from_config(const Config& c);
};

class SelfGenTrainer : public TrainerBase {
 public:
  SelfGenTrainer(Coordinator* coord, Telemetry* tel, const GPTConfig& mcfg,
                 const TrainerConfig& tcfg, const SelfGenConfig& scfg,
                 const Tokenizer* tok, const TokenDataset* corpus, uint64_t seed);

  size_t buffer_size() const { return buffer_.size(); }

 protected:
  bool ready() override;
  void round() override;

 private:
  struct Candidate {
    std::vector<int32_t> ids;  // prompt + response
    size_t response_start = 0;
    float ppl = 0.0f;
    float repeat = 0.0f;
    float score = 0.0f;
    uint64_t hash = 0;
    std::string text;
    std::string verdict;
  };

  void build_prompt_pool();
  std::vector<int32_t> pick_prompt();
  float response_logprob(const std::vector<int32_t>& full, size_t resp_start,
                         int64_t* ntok);
  static float repetition_ratio(const std::vector<int32_t>& ids, size_t from);

  static uint64_t sample_hash(const std::vector<int32_t>& ids, size_t from);

  SelfGenConfig scfg_;
  std::unordered_set<uint64_t> accepted_hashes_;
  std::vector<std::vector<int32_t>> prompts_;
  std::deque<std::vector<int32_t>> buffer_;
  int64_t accepted_total_ = 0;
  int64_t rejected_total_ = 0;
};

}  // namespace slm
