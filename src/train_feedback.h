// SPDX-License-Identifier: Apache-2.0
//
// (c) Feedback thread - simplified RLHF.
//
// Why DPO and not PPO at this scale
// ---------------------------------
// PPO needs a separate reward model, a value head, rollouts and careful KL
// control: four moving parts, each of which can destabilise a 10-100M parameter
// model trained on a handful of ratings.  Direct Preference Optimisation
// replaces all of that with a single closed-form loss on *pairs*:
//
//   L = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
//                            - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )
//
// The reference policy pi_ref is simply the published snapshot the round
// started from, so its log probabilities are computed once, before the first
// gradient step, and cached - no second model in memory.
//
// When a prompt only has a single rating (no pair to compare against) the
// thread falls back to reward-weighted fine-tuning: plain cross entropy scaled
// by the normalised score, which is the score-conditioned special case of the
// same objective.  A small SFT term on the preferred answer is kept alongside
// the DPO loss because it measurably stabilises tiny models.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "interaction.h"
#include "trainer.h"

namespace slm {

struct FeedbackConfig {
  float dpo_beta = 0.1f;
  float sft_weight = 0.25f;
  // Every feedback round also replays the base corpus in *all* languages.  Even
  // if 100% of the ratings arrive in Persian, the gradient still contains
  // English and Python, so the preference signal cannot quietly delete a
  // language.  (The coordinator's per-language gate is the second line of
  // defence.)
  float replay_weight = 0.5f;
  int replay_batches = 0;      // 0 -> one batch per corpus source
  bool balance_languages = true;
  float good_score = 3.5f;      // >= this is "preferred" for reward weighting
  float min_score_gap = 0.5f;   // minimum gap to form a preference pair
  int max_pairs_per_round = 4;
  int max_rwft_per_round = 6;
  int64_t bank_cap = 512;
  bool use_dpo = true;
  static FeedbackConfig from_config(const Config& c);
};

class FeedbackTrainer : public TrainerBase {
 public:
  FeedbackTrainer(Coordinator* coord, Telemetry* tel, InteractionHub* hub,
                  const GPTConfig& mcfg, const TrainerConfig& tcfg,
                  const FeedbackConfig& fcfg, const Tokenizer* tok,
                  const MixtureDataset* corpus, uint64_t seed);

 protected:
  bool ready() override;
  void round() override;

 private:
  struct Item {
    std::vector<int32_t> ids;  // prompt + response
    size_t response_start = 0;
    float score = 0.0f;
    Lang lang = Lang::kUnknown;
    std::string response;
  };
  struct Pair {
    const Item* win;
    const Item* lose;
    float ref_win = 0.0f;
    float ref_lose = 0.0f;
  };

  Item make_item(const RatedSample& s) const;
  // masked targets: only the response positions are scored
  void masked_targets(const Item& it, std::vector<int32_t>* ids,
                      std::vector<int32_t>* tgt, int64_t* T) const;
  float logprob_of(const Item& it);  // no graph
  Tensor logprob_tensor(const Item& it);

  InteractionHub* hub_;
  FeedbackConfig fcfg_;
  std::unordered_map<std::string, std::vector<Item>> bank_;
  int64_t bank_items_ = 0;
  int64_t pairs_total_ = 0;
  int64_t rwft_total_ = 0;
  int64_t ratings_lang_[kNumLangs] = {};
  int64_t pairs_lang_[kNumLangs] = {};
};

}  // namespace slm
