// SPDX-License-Identifier: Apache-2.0
//
// Telemetry hub: the single place where every thread publishes what it is
// doing and the only place the dashboard reads from.
//
// Design rules
//   * writers never block on readers for long: each record type has its own
//     small mutex and the payloads are tiny (a float, a fixed size struct) or
//     double buffered (attention maps),
//   * every self-training decision is *also* appended to an on-disk JSONL
//     audit trail, so a run can be reviewed long after the process exited.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "core/text.h"
#include "model.h"

namespace slm {

enum class Source : int { kContinual = 0, kSelfGen = 1, kFeedback = 2, kCount = 3 };
constexpr int kNumSources = static_cast<int>(Source::kCount);
const char* source_name(Source s);
const char* source_short(Source s);

// Series shown on the loss chart. The three trainers plus the coordinator's
// hold-out evaluation.
enum class Stream : int {
  kContinual = 0,
  kSelfGen = 1,
  kFeedback = 2,
  kHoldout = 3,     // aggregate hold-out loss
  kHoldoutFa = 4,   // per-language hold-out loss
  kHoldoutEn = 5,
  kHoldoutPy = 6,
  kCount = 7
};
// Stream carrying the hold-out loss of one language (kUnknown -> aggregate).
Stream stream_for_lang(Lang l);
constexpr int kNumStreams = static_cast<int>(Stream::kCount);
const char* stream_name(Stream s);

struct LossPoint {
  double t = 0.0;  // seconds since process start
  float value = 0.0f;
  int64_t step = 0;
};

struct TrainerState {
  bool enabled = true;
  bool busy = false;
  float lr = 0.0f;
  float last_loss = 0.0f;
  float loss_start = 0.0f;
  int64_t local_steps = 0;
  int64_t rounds = 0;
  int64_t accepted = 0;
  int64_t rejected = 0;
  int64_t samples_seen = 0;
  int64_t pending_inputs = 0;
  double credit = 1.0;
  double last_round_t = 0.0;
  std::string status = "idle";
  std::string last_checkpoint;
  std::string policy;
};

struct CoordinatorStats {
  int64_t rounds = 0;
  int64_t merged_proposals = 0;
  int64_t accepted = 0;
  int64_t rejected = 0;
  int64_t rollbacks = 0;
  int64_t queue_len = 0;
  float baseline_val = 0.0f;
  float last_val = 0.0f;
  float best_val = 0.0f;
  float last_alpha = 0.0f;
  float last_delta_norm = 0.0f;
  float theta_norm = 0.0f;
  float trust_radius = 0.0f;
  float rate_budget = 0.0f;
  float anchor_drift = 0.0f;
  float fisher_mean = 0.0f;
  int64_t fisher_updates = 0;
  float cos[kNumSources][kNumSources] = {};
  bool active[kNumSources] = {};
  // Per-language hold-out state: this is what makes "did we forget Persian?"
  // an observable number instead of a guess.
  bool lang_present[kNumLangs] = {};
  float lang_val[kNumLangs] = {};
  float lang_best[kNumLangs] = {};
  float lang_gate[kNumLangs] = {};
  float lang_entropy[kNumLangs] = {};
  std::string last_decision = "waiting";
  double last_round_time = 0.0;
  uint64_t weight_version = 0;
};

struct TokenDist {
  std::vector<std::pair<int32_t, float>> top;  // (token id, probability)
  std::vector<std::string> labels;
  std::string context;   // decoded text generated so far
  int64_t step = 0;
};

class Telemetry {
 public:
  explicit Telemetry(size_t capacity = 4096) : capacity_(capacity) {}

  // ---------------------------------------------------------------- clocks
  static double now();

  // ------------------------------------------------------------------ loss
  void push_loss(Stream s, float value, int64_t step);
  std::vector<LossPoint> loss_series(Stream s) const;
  void loss_bounds(float* lo, float* hi) const;

  // -------------------------------------------------------------- trainers
  void set_trainer(Source s, const TrainerState& st);
  TrainerState trainer(Source s) const;
  void set_trainer_enabled(Source s, bool on);
  bool trainer_enabled(Source s) const;

  // ------------------------------------------------------------ coordinator
  void set_coord(const CoordinatorStats& st);
  CoordinatorStats coord() const;

  // -------------------------------------------------------------- attention
  void set_attention(const AttentionCapture& cap);
  bool attention(AttentionCapture* out) const;

  // ------------------------------------------------------- token distribution
  void set_token_dist(const TokenDist& d);
  TokenDist token_dist() const;

  // ------------------------------------------------------------------- logs
  void open_audit(const std::string& path);
  // level: info | accept | reject | rollback | warn | error
  void log(const std::string& level, const std::string& source,
           const std::string& message,
           const std::vector<std::pair<std::string, std::string>>& fields = {});
  std::vector<std::string> recent_logs(size_t max_lines) const;

  // --------------------------------------------------------- global controls
  void emergency_stop();
  void clear_emergency_stop();
  bool stopped() const { return estop_.load(std::memory_order_relaxed); }

  void set_chat_busy(bool b) { chat_busy_.store(b); }
  bool chat_busy() const { return chat_busy_.load(); }

  // Master switch for the three self-training threads.  Off by default: a model
  // that starts rewriting its own weights the moment the dashboard opens is not
  // what someone installing a package expects, and it makes every measurement
  // taken afterwards irreproducible.  The threads stay alive and idle so the
  // switch takes effect within a fraction of a second.
  void set_self_training_enabled(bool on) { self_train_.store(on); }
  bool self_training_enabled() const { return self_train_.load(); }

 private:
  size_t capacity_;
  mutable std::mutex loss_m_;
  std::deque<LossPoint> loss_[kNumStreams];

  mutable std::mutex tr_m_;
  TrainerState trainers_[kNumSources];

  mutable std::mutex co_m_;
  CoordinatorStats coord_;

  mutable std::mutex att_m_;
  AttentionCapture att_;
  bool att_valid_ = false;

  mutable std::mutex td_m_;
  TokenDist td_;

  mutable std::mutex log_m_;
  std::deque<std::string> logs_;
  std::string audit_path_;

  std::atomic<bool> estop_{false};
  std::atomic<bool> chat_busy_{false};
  std::atomic<bool> self_train_{false};
};

}  // namespace slm
