// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
//  The Coordinator - how three concurrent learners share one set of weights
// ============================================================================
//
// Every trainer works on its *own* replica of the model and never writes to the
// live weights.  What it produces is a proposal: the flat difference
//
//      delta_i = theta_local_after_training - theta_base_when_it_started
//
// The coordinator is the single writer.  For every round it merges the pending
// proposals into one update and then decides whether that update is allowed to
// become the new published state.  The merge is a pipeline of six stages:
//
//   1. Fisher damping (EWC-lite)
//        delta[j] <- delta[j] / (1 + lambda * F[j])
//      F is a diagonal Fisher estimate (EMA of squared hold-out gradients).
//      Coordinates that matter for the *original* task are hard to move, which
//      is what keeps catastrophic forgetting away.
//
//   2. Per-source trust region
//        ||delta_i|| <= rho_i * ||theta||
//      A single learner can never move the model by more than a fixed relative
//      distance, no matter how long it trained or how wrong its data was.
//
//   3. TIES trimming
//      Keep only the top-k% coordinates by magnitude of each proposal.  Small,
//      noisy coordinates are exactly the ones that cause interference.
//
//   4. PCGrad projection
//      For every conflicting pair (cos < 0) project one out of the other:
//        delta_i <- delta_i - (<delta_i,delta_j>/||delta_j||^2) * delta_j
//      After this step no pair of proposals actively fights each other.
//
//   5. Sign election + weighted mean (TIES merge)
//      Per coordinate elect the sign with the larger weighted mass, then
//      average only the proposals that agree with it.  Weights are
//        w_i = priority(source) * credit(source)
//      where credit is a bandit-style score that grows for sources whose
//      updates keep being accepted and shrinks for the ones that get rejected.
//
//   6. Global trust region + rate limiter (token bucket)
//      Bounds how much the model may drift *per minute*, which is the direct
//      defence against model collapse from self-generated data.
//
// The merged update is then applied speculatively with a backtracking line
// search over alpha in {1, 1/2, 1/4}: the candidate weights are evaluated on a
// fixed hold-out set (loss *and* predictive entropy).  The first alpha that
// does not hurt either metric is committed; if none qualifies the round is
// rejected and nothing changes.  Because a candidate is always a *new*
// ParamStore and publishing is a pointer swap, rollback is instantaneous and
// cannot leave the live weights in a half-updated state.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/config.h"
#include "core/dataset.h"
#include "core/params.h"
#include "model.h"
#include "telemetry.h"

namespace slm {

struct UpdateProposal {
  Source source = Source::kContinual;
  int64_t id = 0;
  double created_at = 0.0;
  uint64_t base_version = 0;
  std::vector<float> delta;  // flat over Coordinator::flat()
  float loss_start = 0.0f;
  float loss_end = 0.0f;
  int64_t samples = 0;
  int64_t steps = 0;
  float lr = 0.0f;
  std::string note;
};

struct CoordinatorConfig {
  // merge
  bool enable_fisher = true;
  bool enable_ties = true;
  bool enable_pcgrad = true;
  float fisher_lambda = 4.0f;
  int fisher_refresh_rounds = 6;
  float fisher_ema = 0.25f;
  float ties_keep = 0.30f;  // fraction of coordinates kept per proposal
  float trust_ratio = 0.015f;
  float global_trust_ratio = 0.02f;
  float priority[kNumSources] = {1.0f, 0.6f, 1.4f};
  // acceptance
  bool per_language_gate = true;       // every language must pass, not just the mean
  float accept_tolerance = 0.003f;     // allowed step-local hold-out increase
  // Ratchet: the model may never end up worse than the best state seen this
  // session plus this budget.  Without it, a long series of "only 0.002 worse"
  // updates walks the model downhill for free.
  float regression_budget = 0.01f;
  float entropy_floor_ratio = 0.75f;   // collapse guard vs. the session baseline
  float anchor_drift_max = 0.12f;      // ||theta - anchor|| / ||anchor||
  float anchor_ema = 0.05f;
  std::vector<float> alphas = {1.0f, 0.5f, 0.25f};
  // rate limiting
  float rate_per_minute = 0.06f;  // relative norm budget added per minute
  float rate_burst = 0.12f;
  // evaluation
  int holdout_batches = 2;
  int64_t holdout_batch = 4;
  int64_t holdout_ctx = 0;  // 0 -> model block_size
  // bookkeeping
  double round_period_s = 0.25;
  // How long to wait for the *other* sources once the first proposal arrives.
  // This is what makes real multi-source merges (and therefore conflict
  // resolution) happen instead of degenerating into sequential updates.
  double merge_window_s = 4.0;
  int max_disk_backups = 6;
  int max_memory_backups = 2;
  int save_every_rounds = 5;
  bool save_accepted_checkpoints = true;
  static CoordinatorConfig from_config(const Config& c);
};

struct EvalResult {
  float loss = 0.0f;     // mean over every hold-out batch
  float entropy = 0.0f;  // mean predictive entropy (collapse detector)
  bool present[kNumLangs] = {};
  float lang_loss[kNumLangs] = {};
  float lang_entropy[kNumLangs] = {};
};

// Thread-safe RCU-style holder for the published weights.
class WeightRegistry {
 public:
  void publish(ParamStorePtr p);
  ParamStorePtr current(uint64_t* version = nullptr) const;
  uint64_t version() const { return version_.load(std::memory_order_acquire); }

 private:
  mutable std::shared_mutex m_;
  ParamStorePtr cur_;
  std::atomic<uint64_t> version_{0};
};

class Coordinator {
 public:
  Coordinator(const GPTConfig& mcfg, ParamStorePtr initial,
              std::vector<std::string> merge_space, const MixtureDataset* data,
              Telemetry* tel, const CoordinatorConfig& cfg, std::string workdir);
  ~Coordinator();

  void start();
  void stop();

  // ------------------------------------------------------------- weights
  ParamStorePtr snapshot(uint64_t* version = nullptr) const {
    return weights_.current(version);
  }
  uint64_t version() const { return weights_.version(); }
  const FlatSpec& flat() const { return flat_; }

  // ------------------------------------------------------------ proposals
  void submit(UpdateProposal p);
  size_t queue_length() const;
  int queue_sources() const;  // number of distinct sources waiting

  // ------------------------------------------------------------ inspection
  EvalResult baseline() const;
  CoordinatorStats stats() const;
  // Reverts to the best hold-out snapshot seen this session.
  bool restore_best();
  double credit(Source s) const;

 private:
  void loop();
  void round();
  EvalResult evaluate(const ParamStore& candidate);
  void refresh_fisher();
  void save_backup(const ParamStore& ps, int64_t round, float val);
  std::vector<UpdateProposal> collect();

  GPTConfig mcfg_;
  CoordinatorConfig cfg_;
  Telemetry* tel_;
  const MixtureDataset* data_;
  std::string workdir_;

  WeightRegistry weights_;
  FlatSpec flat_;
  std::unique_ptr<GPT> eval_model_;
  std::vector<Batch> holdout_;

  std::vector<float> fisher_;
  std::vector<float> anchor_;
  double credit_[kNumSources] = {1.0, 1.0, 1.0};

  mutable std::mutex q_m_;
  std::condition_variable q_cv_;
  std::deque<UpdateProposal> queue_;
  int64_t next_id_ = 1;

  mutable std::mutex st_m_;
  CoordinatorStats st_;
  EvalResult base_eval_;
  float session_entropy_ = 0.0f;
  float session_lang_entropy_[kNumLangs] = {};
  float best_val_ = 1e30f;
  float best_lang_[kNumLangs] = {};
  ParamStorePtr best_snapshot_;
  std::deque<ParamStorePtr> mem_backups_;
  std::deque<std::string> disk_backups_;

  float rate_bucket_ = 0.0f;
  double last_rate_t_ = 0.0;

  std::thread th_;
  std::atomic<bool> run_{false};
};

}  // namespace slm
