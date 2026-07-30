// SPDX-License-Identifier: Apache-2.0
//
// (b) Self-generated data thread - one quality standard per language.
//
// The generator samples the *current* model and then applies filters that are
// deliberately different per language, because "bad output" means different
// things in Persian prose and in Python source:
//
//   shared     perplexity band (relative to the model's own hold-out
//              perplexity, not a magic constant), 4-gram repetition, length,
//              novelty against everything already accepted
//   fa / en    code-switching guard: a Persian sample stuffed with Latin words
//              (or the reverse) is rejected, which is exactly the interference
//              failure mode of a bilingual model
//   py         the sample must survive a structural Python check (balanced
//              brackets and quotes, consistent indentation, real statements)
//              and contain at least one definition or return
//
// Prompts are mined per source, so every round asks the model to produce data
// in *every* language it is supposed to know.
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
  // Absolute backstops.  Keep the floor almost at 1.0: "degenerate" is decided
  // by the *relative* rule below plus the repetition/novelty filters, because a
  // model that reaches ppl 1.07 on real text also produces ~1.0002 on its own
  // correct output - that is memorisation, not degeneration.
  float ppl_min = 1.000001f;
  float ppl_max = 30.0f;
  // The band that matters: relative to the current hold-out perplexity.
  float ppl_min_ratio = 0.30f;
  float ppl_max_ratio = 3.0f;
  float max_repeat_ratio = 0.34f;
  int min_tokens = 8;
  int keep_best = 3;
  int64_t buffer_cap = 192;
  bool require_novel = true;
  // Language specific
  bool balance_languages = true;      // generate for every source, round robin
  float max_code_switch = 0.25f;      // fa/en interference guard
  bool require_valid_python = true;   // py structural check
  float py_min_comment_ratio = 0.0f;  // set >0 to demand commented code
  static SelfGenConfig from_config(const Config& c);
};

class SelfGenTrainer : public TrainerBase {
 public:
  SelfGenTrainer(Coordinator* coord, Telemetry* tel, const GPTConfig& mcfg,
                 const TrainerConfig& tcfg, const SelfGenConfig& scfg,
                 const Tokenizer* tok, const MixtureDataset* corpus, uint64_t seed);

  size_t buffer_size() const { return buffer_.size(); }

 protected:
  bool ready() override;
  void round() override;

 private:
  struct Candidate {
    std::vector<int32_t> ids;  // prompt + response
    size_t response_start = 0;
    Lang lang = Lang::kUnknown;
    int source = -1;
    float ppl = 0.0f;
    float repeat = 0.0f;
    float switch_ratio = 0.0f;
    float score = 0.0f;
    uint64_t hash = 0;
    std::string text;
    std::string verdict;  // empty == passed every filter
  };
  struct Pool {
    int source = -1;
    Lang lang = Lang::kUnknown;
    std::vector<std::vector<int32_t>> prompts;
  };

  void build_prompt_pools();
  const Pool* pick_pool(int index);
  float response_logprob(const std::vector<int32_t>& full, size_t resp_start, int64_t* ntok);
  void judge(Candidate* c, float ppl_lo, float ppl_hi) const;
  static float repetition_ratio(const std::vector<int32_t>& ids, size_t from);
  static uint64_t sample_hash(const std::vector<int32_t>& ids, size_t from);
  static std::string extract_code(const std::string& text);

  SelfGenConfig scfg_;
  std::vector<Pool> pools_;
  std::unordered_set<uint64_t> accepted_hashes_;
  std::deque<std::pair<Lang, std::vector<int32_t>>> buffer_;
  int64_t accepted_total_ = 0;
  int64_t rejected_total_ = 0;
  int64_t accepted_lang_[kNumLangs] = {};
  int64_t rejected_lang_[kNumLangs] = {};
  int rr_ = 0;
};

}  // namespace slm
