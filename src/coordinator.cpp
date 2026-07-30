// SPDX-License-Identifier: Apache-2.0
#include "coordinator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <sstream>

#include "core/config.h"
#include "core/serialize.h"

namespace slm {
namespace {

std::string fmt(double v, int prec = 4) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
  return buf;
}

// loss and mean predictive entropy of one batch, without building a graph
EvalResult eval_batch(GPT& m, const Batch& b) {
  NoGradGuard ng;
  Tensor logits = m.forward(b.ids, b.B, b.T, ForwardOptions());
  const int64_t V = m.config().vocab_size;
  const int64_t N = b.B * b.T;
  const float* x = logits.host_ptr();
  double loss = 0.0, ent = 0.0;
  int64_t n = 0;
  for (int64_t r = 0; r < N; ++r) {
    const int32_t tgt = b.targets[static_cast<size_t>(r)];
    if (tgt < 0) continue;
    const float* row = x + r * V;
    float mx = row[0];
    for (int64_t j = 1; j < V; ++j) mx = std::max(mx, row[j]);
    double s = 0.0;
    for (int64_t j = 0; j < V; ++j) s += std::exp(static_cast<double>(row[j] - mx));
    const double lse = static_cast<double>(mx) + std::log(s);
    double px = 0.0;
    for (int64_t j = 0; j < V; ++j)
      px += std::exp(static_cast<double>(row[j]) - lse) * static_cast<double>(row[j]);
    loss += lse - static_cast<double>(row[tgt]);
    ent += lse - px;
    ++n;
  }
  EvalResult r;
  if (n) {
    r.loss = static_cast<float>(loss / static_cast<double>(n));
    r.entropy = static_cast<float>(ent / static_cast<double>(n));
  }
  return r;
}

}  // namespace

CoordinatorConfig CoordinatorConfig::from_config(const Config& c) {
  CoordinatorConfig k;
  k.enable_fisher = c.get_bool("coord.fisher", k.enable_fisher);
  k.enable_ties = c.get_bool("coord.ties", k.enable_ties);
  k.enable_pcgrad = c.get_bool("coord.pcgrad", k.enable_pcgrad);
  k.fisher_lambda = static_cast<float>(c.get_num("coord.fisher_lambda", k.fisher_lambda));
  k.fisher_refresh_rounds =
      static_cast<int>(c.get_int("coord.fisher_refresh_rounds", k.fisher_refresh_rounds));
  k.fisher_ema = static_cast<float>(c.get_num("coord.fisher_ema", k.fisher_ema));
  k.ties_keep = static_cast<float>(c.get_num("coord.ties_keep", k.ties_keep));
  k.trust_ratio = static_cast<float>(c.get_num("coord.trust_ratio", k.trust_ratio));
  k.global_trust_ratio =
      static_cast<float>(c.get_num("coord.global_trust_ratio", k.global_trust_ratio));
  k.priority[0] = static_cast<float>(c.get_num("coord.priority.continual", k.priority[0]));
  k.priority[1] = static_cast<float>(c.get_num("coord.priority.selfgen", k.priority[1]));
  k.priority[2] = static_cast<float>(c.get_num("coord.priority.feedback", k.priority[2]));
  k.accept_tolerance =
      static_cast<float>(c.get_num("coord.accept_tolerance", k.accept_tolerance));
  k.regression_budget =
      static_cast<float>(c.get_num("coord.regression_budget", k.regression_budget));
  k.entropy_floor_ratio =
      static_cast<float>(c.get_num("coord.entropy_floor_ratio", k.entropy_floor_ratio));
  k.anchor_drift_max =
      static_cast<float>(c.get_num("coord.anchor_drift_max", k.anchor_drift_max));
  k.anchor_ema = static_cast<float>(c.get_num("coord.anchor_ema", k.anchor_ema));
  k.rate_per_minute = static_cast<float>(c.get_num("coord.rate_per_minute", k.rate_per_minute));
  k.rate_burst = static_cast<float>(c.get_num("coord.rate_burst", k.rate_burst));
  k.holdout_batches = static_cast<int>(c.get_int("coord.holdout_batches", k.holdout_batches));
  k.holdout_batch = c.get_int("coord.holdout_batch", k.holdout_batch);
  k.holdout_ctx = c.get_int("coord.holdout_ctx", k.holdout_ctx);
  k.round_period_s = c.get_num("coord.round_period_s", k.round_period_s);
  k.merge_window_s = c.get_num("coord.merge_window_s", k.merge_window_s);
  k.max_disk_backups = static_cast<int>(c.get_int("coord.max_disk_backups", k.max_disk_backups));
  k.save_every_rounds = static_cast<int>(c.get_int("coord.save_every_rounds", k.save_every_rounds));
  k.save_accepted_checkpoints =
      c.get_bool("coord.save_accepted_checkpoints", k.save_accepted_checkpoints);
  return k;
}

// ------------------------------------------------------------ WeightRegistry
void WeightRegistry::publish(ParamStorePtr p) {
  {
    std::unique_lock<std::shared_mutex> lk(m_);
    cur_ = std::move(p);
  }
  version_.fetch_add(1, std::memory_order_release);
}

ParamStorePtr WeightRegistry::current(uint64_t* version) const {
  std::shared_lock<std::shared_mutex> lk(m_);
  if (version) *version = version_.load(std::memory_order_acquire);
  return cur_;
}

// ================================================================ Coordinator
Coordinator::Coordinator(const GPTConfig& mcfg, ParamStorePtr initial,
                         std::vector<std::string> merge_space,
                         const TokenDataset* data, Telemetry* tel,
                         const CoordinatorConfig& cfg, std::string workdir)
    : mcfg_(mcfg), cfg_(cfg), tel_(tel), data_(data), workdir_(std::move(workdir)) {
  weights_.publish(initial);
  flat_ = FlatSpec::build(*initial, merge_space);
  fisher_.assign(static_cast<size_t>(flat_.total), 0.0f);
  flat_.gather(*initial, anchor_);

  eval_model_ = std::make_unique<GPT>(mcfg_);
  eval_model_->load_params(*initial);

  const int64_t ctx = cfg_.holdout_ctx > 0
                          ? std::min<int64_t>(cfg_.holdout_ctx, mcfg_.block_size)
                          : mcfg_.block_size;
  if (data_) holdout_ = data_->holdout_batches(cfg_.holdout_batch, ctx, cfg_.holdout_batches);

  base_eval_ = evaluate(*initial);
  session_entropy_ = base_eval_.entropy;
  best_val_ = base_eval_.loss;
  best_snapshot_ = initial;
  st_.baseline_val = base_eval_.loss;
  st_.last_val = base_eval_.loss;
  st_.best_val = base_eval_.loss;
  st_.weight_version = weights_.version();
  {
    std::vector<float> theta;
    flat_.gather(*initial, theta);
    st_.theta_norm = static_cast<float>(vec_norm(theta));
    st_.trust_radius = cfg_.trust_ratio * st_.theta_norm;
  }
  rate_bucket_ = cfg_.rate_burst;
  last_rate_t_ = Telemetry::now();
  if (tel_) {
    tel_->set_coord(st_);
    tel_->log("info", "coordinator",
              "merge space " + std::to_string(flat_.names.size()) + " tensors / " +
                  std::to_string(flat_.total) + " scalars",
              {{"holdout_batches", std::to_string(holdout_.size())},
               {"baseline_loss", fmt(base_eval_.loss)},
               {"baseline_entropy", fmt(base_eval_.entropy)}});
  }
}

Coordinator::~Coordinator() { stop(); }

void Coordinator::start() {
  if (run_.exchange(true)) return;
  th_ = std::thread([this] { loop(); });
}

void Coordinator::stop() {
  if (!run_.exchange(false)) return;
  q_cv_.notify_all();
  if (th_.joinable()) th_.join();
}

void Coordinator::submit(UpdateProposal p) {
  {
    std::lock_guard<std::mutex> g(q_m_);
    p.id = next_id_++;
    queue_.push_back(std::move(p));
    while (queue_.size() > 32) queue_.pop_front();
  }
  q_cv_.notify_one();
}

size_t Coordinator::queue_length() const {
  std::lock_guard<std::mutex> g(q_m_);
  return queue_.size();
}

int Coordinator::queue_sources() const {
  bool seen[kNumSources] = {};
  std::lock_guard<std::mutex> g(q_m_);
  for (const UpdateProposal& p : queue_) seen[static_cast<int>(p.source)] = true;
  int n = 0;
  for (bool b : seen) n += b ? 1 : 0;
  return n;
}

EvalResult Coordinator::baseline() const {
  std::lock_guard<std::mutex> g(st_m_);
  return base_eval_;
}

CoordinatorStats Coordinator::stats() const {
  std::lock_guard<std::mutex> g(st_m_);
  return st_;
}

double Coordinator::credit(Source s) const {
  std::lock_guard<std::mutex> g(st_m_);
  return credit_[static_cast<int>(s)];
}

std::vector<UpdateProposal> Coordinator::collect() {
  std::vector<UpdateProposal> out;
  std::lock_guard<std::mutex> g(q_m_);
  while (!queue_.empty()) {
    out.push_back(std::move(queue_.front()));
    queue_.pop_front();
  }
  return out;
}

EvalResult Coordinator::evaluate(const ParamStore& candidate) {
  if (holdout_.empty()) return EvalResult{};
  eval_model_->load_params(candidate);
  double loss = 0.0, ent = 0.0;
  for (const Batch& b : holdout_) {
    const EvalResult r = eval_batch(*eval_model_, b);
    loss += r.loss;
    ent += r.entropy;
  }
  EvalResult r;
  r.loss = static_cast<float>(loss / static_cast<double>(holdout_.size()));
  r.entropy = static_cast<float>(ent / static_cast<double>(holdout_.size()));
  return r;
}

void Coordinator::refresh_fisher() {
  if (holdout_.empty() || !cfg_.enable_fisher) return;
  int64_t updates = 0;
  {
    std::lock_guard<std::mutex> lk(st_m_);
    updates = st_.fisher_updates;
  }
  ParamStorePtr cur = weights_.current();
  eval_model_->load_params(*cur);
  eval_model_->zero_grad();
  const Batch& b = holdout_[static_cast<size_t>(updates) % holdout_.size()];
  Tensor l = eval_model_->loss(b.ids, b.targets, b.B, b.T, nullptr, nullptr, false);
  l.backward();
  std::vector<float> g;
  double mean = 0.0;
  for (size_t k = 0; k < flat_.names.size(); ++k) {
    Tensor* t = eval_model_->param(flat_.names[k]);
    if (!t) continue;
    t->copy_grad_to_host(g);
    const int64_t base = flat_.offsets[k];
    for (size_t j = 0; j < g.size(); ++j) {
      float& f = fisher_[static_cast<size_t>(base) + j];
      f = (1.0f - cfg_.fisher_ema) * f + cfg_.fisher_ema * g[j] * g[j];
      mean += f;
    }
  }
  std::lock_guard<std::mutex> lk(st_m_);
  st_.fisher_mean = static_cast<float>(mean / std::max<int64_t>(1, flat_.total));
  ++st_.fisher_updates;
}

void Coordinator::save_backup(const ParamStore& ps, int64_t round, float val) {
  if (!cfg_.save_accepted_checkpoints || workdir_.empty()) return;
  std::error_code ec;
  std::filesystem::create_directories(workdir_, ec);
  char name[256];
  std::snprintf(name, sizeof(name), "%s/round_%06lld.slm", workdir_.c_str(),
                static_cast<long long>(round));
  CheckpointMeta m;
  m.step = round;
  mcfg_.write_to(m.extra);
  m.extra.set("coord.holdout_loss", fmt(val));
  m.extra.set("origin", "coordinator");
  if (!save_checkpoint(name, ps, m, Dtype::F16)) return;
  disk_backups_.push_back(name);
  while (static_cast<int>(disk_backups_.size()) > cfg_.max_disk_backups) {
    std::filesystem::remove(disk_backups_.front(), ec);
    disk_backups_.pop_front();
  }
}

bool Coordinator::restore_best() {
  ParamStorePtr best;
  {
    std::lock_guard<std::mutex> g(st_m_);
    best = best_snapshot_;
  }
  if (!best) return false;
  weights_.publish(best);
  EvalResult e = evaluate(*best);
  std::lock_guard<std::mutex> g(st_m_);
  base_eval_ = e;
  st_.baseline_val = e.loss;
  st_.last_val = e.loss;
  st_.weight_version = weights_.version();
  ++st_.rollbacks;
  st_.last_decision = "manual restore of best snapshot";
  if (tel_) tel_->log("rollback", "coordinator", "restored best snapshot",
                      {{"holdout_loss", fmt(e.loss)}});
  return true;
}

void Coordinator::loop() {
  while (run_.load()) {
    {
      std::unique_lock<std::mutex> lk(q_m_);
      q_cv_.wait_for(lk, std::chrono::duration<double>(cfg_.round_period_s),
                     [this] { return !queue_.empty() || !run_.load(); });
    }
    if (!run_.load()) break;

    // refill the rate limiter bucket
    const double t = Telemetry::now();
    const double dt = std::max(0.0, t - last_rate_t_);
    last_rate_t_ = t;
    rate_bucket_ = std::min(cfg_.rate_burst,
                            rate_bucket_ + static_cast<float>(dt / 60.0) * cfg_.rate_per_minute);

    if (queue_length() > 0) {
      // Gather window: give the other learners a chance to land their proposal
      // for the same round so that the merge sees the real conflict picture.
      int want = 0;
      for (int i = 0; i < kNumSources; ++i)
        if (!tel_ || tel_->trainer_enabled(static_cast<Source>(i))) ++want;
      const double deadline = Telemetry::now() + cfg_.merge_window_s;
      while (run_.load() && queue_sources() < want && Telemetry::now() < deadline) {
        std::unique_lock<std::mutex> lk(q_m_);
        q_cv_.wait_for(lk, std::chrono::milliseconds(50));
      }
      round();
    }

    std::lock_guard<std::mutex> g(st_m_);
    st_.queue_len = static_cast<int64_t>(queue_length());
    st_.rate_budget = rate_bucket_;
    st_.weight_version = weights_.version();
    if (tel_) tel_->set_coord(st_);
  }
}

void Coordinator::round() {
  std::vector<UpdateProposal> props = collect();
  if (props.empty()) return;
  const double t_start = Telemetry::now();

  if (tel_ && tel_->stopped()) {
    tel_->log("warn", "coordinator",
              "dropped " + std::to_string(props.size()) +
                  " proposals: emergency stop is engaged");
    return;
  }

  uint64_t version = 0;
  ParamStorePtr base = weights_.current(&version);
  std::vector<float> theta;
  flat_.gather(*base, theta);
  const double theta_norm = std::max(1e-8, vec_norm(theta));

  // ------------------------------------------------- one slot per source
  std::vector<float> d[kNumSources];
  bool has[kNumSources] = {};
  int64_t samples[kNumSources] = {};
  int64_t steps[kNumSources] = {};
  float loss_start[kNumSources] = {};
  float loss_end[kNumSources] = {};
  double staleness[kNumSources] = {1.0, 1.0, 1.0};
  int merged_count = 0;
  for (UpdateProposal& p : props) {
    const int i = static_cast<int>(p.source);
    if (static_cast<int64_t>(p.delta.size()) != flat_.total) {
      if (tel_) tel_->log("error", source_name(p.source), "proposal size mismatch, dropped");
      continue;
    }
    ++merged_count;
    const double stale = 1.0 / (1.0 + static_cast<double>(version - p.base_version));
    if (!has[i]) {
      d[i] = std::move(p.delta);
      vec_scale(d[i], static_cast<float>(stale));
      has[i] = true;
      loss_start[i] = p.loss_start;
    } else {
      vec_axpy(static_cast<float>(stale), p.delta, d[i]);
    }
    samples[i] += p.samples;
    steps[i] += p.steps;
    loss_end[i] = p.loss_end;
    staleness[i] = std::min(staleness[i], stale);
  }
  if (!merged_count) return;

  // ---------------------------------------- stage 1..3 : damp, bound, trim
  const float fmean = std::max(1e-12f, [&] {
    double s = 0.0;
    for (float f : fisher_) s += f;
    return static_cast<float>(s / std::max<size_t>(1, fisher_.size()));
  }());
  bool fisher_ready = false;
  int64_t round_index = 0;
  {
    std::lock_guard<std::mutex> lk(st_m_);
    fisher_ready = st_.fisher_updates > 0;
    round_index = st_.rounds;
  }
  (void)round_index;
  std::vector<float> mag;
  float pre_norm[kNumSources] = {};
  for (int i = 0; i < kNumSources; ++i) {
    if (!has[i]) continue;
    std::vector<float>& v = d[i];
    pre_norm[i] = static_cast<float>(vec_norm(v));
    if (cfg_.enable_fisher && fisher_ready) {
      for (size_t j = 0; j < v.size(); ++j)
        v[j] /= (1.0f + cfg_.fisher_lambda * fisher_[j] / fmean);
    }
    const double n = vec_norm(v);
    const double cap = cfg_.trust_ratio * theta_norm;
    if (n > cap && n > 0.0) vec_scale(v, static_cast<float>(cap / n));
    if (cfg_.enable_ties && cfg_.ties_keep < 1.0f) {
      mag.resize(v.size());
      for (size_t j = 0; j < v.size(); ++j) mag[j] = std::fabs(v[j]);
      const size_t keep = std::max<size_t>(
          1, static_cast<size_t>(static_cast<double>(mag.size()) * cfg_.ties_keep));
      std::nth_element(mag.begin(), mag.begin() + static_cast<long>(mag.size() - keep),
                       mag.end());
      const float thresh = mag[mag.size() - keep];
      for (float& x : v)
        if (std::fabs(x) < thresh) x = 0.0f;
    }
  }

  // ------------------------------------------------- pairwise similarities
  float cosm[kNumSources][kNumSources] = {};
  for (int i = 0; i < kNumSources; ++i)
    for (int j = 0; j < kNumSources; ++j) {
      if (!has[i] || !has[j]) continue;
      const double ni = vec_norm(d[i]), nj = vec_norm(d[j]);
      cosm[i][j] = (ni > 0 && nj > 0)
                       ? static_cast<float>(vec_dot(d[i], d[j]) / (ni * nj))
                       : 0.0f;
    }

  // -------------------------------------------- stage 4 : PCGrad projection
  int projections = 0;
  if (cfg_.enable_pcgrad) {
    for (int i = 0; i < kNumSources; ++i) {
      if (!has[i]) continue;
      for (int j = 0; j < kNumSources; ++j) {
        if (i == j || !has[j]) continue;
        const double dot = vec_dot(d[i], d[j]);
        if (dot >= 0.0) continue;
        const double nj2 = vec_dot(d[j], d[j]);
        if (nj2 <= 0.0) continue;
        vec_axpy(static_cast<float>(-dot / nj2), d[j], d[i]);
        ++projections;
      }
    }
  }

  // --------------------------- stage 5 : sign election + weighted averaging
  double w[kNumSources] = {};
  {
    std::lock_guard<std::mutex> g(st_m_);
    for (int i = 0; i < kNumSources; ++i)
      w[i] = has[i] ? cfg_.priority[i] * credit_[i] : 0.0;
  }
  std::vector<float> merged(static_cast<size_t>(flat_.total), 0.0f);
  if (cfg_.enable_ties) {
    for (int64_t j = 0; j < flat_.total; ++j) {
      double signed_mass = 0.0;
      for (int i = 0; i < kNumSources; ++i)
        if (has[i]) signed_mass += w[i] * static_cast<double>(d[i][static_cast<size_t>(j)]);
      if (signed_mass == 0.0) continue;
      const double s = signed_mass > 0.0 ? 1.0 : -1.0;
      double num = 0.0, den = 0.0;
      for (int i = 0; i < kNumSources; ++i) {
        if (!has[i]) continue;
        const double v = d[i][static_cast<size_t>(j)];
        if (v == 0.0 || (v > 0.0 ? 1.0 : -1.0) != s) continue;
        num += w[i] * v;
        den += w[i];
      }
      if (den > 0.0) merged[static_cast<size_t>(j)] = static_cast<float>(num / den);
    }
  } else {
    double wsum = 0.0;
    for (int i = 0; i < kNumSources; ++i) wsum += w[i];
    for (int i = 0; i < kNumSources; ++i)
      if (has[i]) vec_axpy(static_cast<float>(w[i] / std::max(1e-9, wsum)), d[i], merged);
  }

  // ------------------------------ stage 6 : global trust region + rate limit
  double mnorm = vec_norm(merged);
  const double gcap = cfg_.global_trust_ratio * theta_norm;
  if (mnorm > gcap && mnorm > 0.0) {
    vec_scale(merged, static_cast<float>(gcap / mnorm));
    mnorm = gcap;
  }
  const double rel = mnorm / theta_norm;
  double rate_scale = 1.0;
  if (rel > 0.0 && rel > rate_bucket_) rate_scale = rate_bucket_ / rel;
  if (rate_scale < 1.0) vec_scale(merged, static_cast<float>(rate_scale));

  // ------------------------------------------------ speculative line search
  EvalResult before;
  {
    std::lock_guard<std::mutex> g(st_m_);
    before = base_eval_;
  }
  const float ent_floor = cfg_.entropy_floor_ratio * session_entropy_;
  // two-sided gate: local tolerance *and* a session ratchet against the best
  const float gate = std::min(before.loss + cfg_.accept_tolerance,
                              best_val_ + cfg_.regression_budget);
  bool accepted = false;
  std::shared_ptr<ParamStore> cand_store;
  float used_alpha = 0.0f;
  EvalResult after;
  ParamStorePtr committed;
  std::string reason = "no alpha passed the hold-out gate";

  for (float alpha : cfg_.alphas) {
    if (tel_ && tel_->stopped()) {
      reason = "emergency stop during evaluation";
      break;
    }
    // drift guard decides the largest legal alpha
    std::vector<float> cand_flat = theta;
    vec_axpy(alpha, merged, cand_flat);
    double drift = 0.0, anorm = std::max(1e-8, vec_norm(anchor_));
    {
      std::vector<float> diff = cand_flat;
      vec_axpy(-1.0f, anchor_, diff);
      drift = vec_norm(diff) / anorm;
    }
    if (drift > cfg_.anchor_drift_max) {
      reason = "anchor drift " + fmt(drift, 3) + " > " + fmt(cfg_.anchor_drift_max, 3);
      continue;
    }
    if (!cand_store) cand_store = base->clone();
    flat_.scatter(cand_flat, *cand_store);
    after = evaluate(*cand_store);
    const bool loss_ok = after.loss <= gate;
    const bool ent_ok = after.entropy >= ent_floor;
    if (loss_ok && ent_ok) {
      accepted = true;
      used_alpha = alpha;
      committed = cand_store;
      cand_store.reset();  // ownership moves to the published snapshot
      reason = "holdout " + fmt(before.loss) + " -> " + fmt(after.loss) +
               " (gate " + fmt(gate) + ")";
      break;
    }
    if (!ent_ok)
      reason = "entropy collapse guard: " + fmt(after.entropy, 3) + " < " + fmt(ent_floor, 3);
    else
      reason = "holdout regression " + fmt(before.loss) + " -> " + fmt(after.loss) +
               " > gate " + fmt(gate);
  }

  // ---------------------------------------------------------------- commit
  std::string decision;
  int64_t round_no = 0;
  std::vector<std::pair<std::string, std::string>> fields;
  for (int i = 0; i < kNumSources; ++i)
    if (has[i])
      fields.emplace_back(source_short(static_cast<Source>(i)),
                          "n=" + std::to_string(samples[i]) + ",steps=" +
                              std::to_string(steps[i]) + ",|d|=" + fmt(pre_norm[i], 5) +
                              ",loss " + fmt(loss_start[i], 3) + "->" + fmt(loss_end[i], 3));
  fields.emplace_back("projections", std::to_string(projections));
  fields.emplace_back("merged_norm", fmt(mnorm, 6));
  fields.emplace_back("rel_step", fmt(rel * rate_scale, 6));
  fields.emplace_back("rate_left", fmt(rate_bucket_, 4));
  fields.emplace_back("gate", fmt(gate));

  if (accepted && committed) {
    const double consumed = rel * static_cast<double>(used_alpha) * rate_scale;
    rate_bucket_ = std::max(0.0f, rate_bucket_ - static_cast<float>(consumed));
    weights_.publish(committed);

    // slow EMA anchor: the reference the drift guard measures against
    std::vector<float> newtheta;
    flat_.gather(*committed, newtheta);
    for (size_t j = 0; j < anchor_.size(); ++j)
      anchor_[j] = (1.0f - cfg_.anchor_ema) * anchor_[j] + cfg_.anchor_ema * newtheta[j];

    std::lock_guard<std::mutex> g(st_m_);
    base_eval_ = after;
    ++st_.rounds;
    ++st_.accepted;
    st_.merged_proposals += merged_count;
    st_.baseline_val = before.loss;
    st_.last_val = after.loss;
    st_.last_alpha = used_alpha;
    st_.last_delta_norm = static_cast<float>(mnorm * used_alpha);
    st_.theta_norm = static_cast<float>(theta_norm);
    st_.trust_radius = static_cast<float>(cfg_.trust_ratio * theta_norm);
    st_.anchor_drift = static_cast<float>([&] {
      std::vector<float> diff = newtheta;
      vec_axpy(-1.0f, anchor_, diff);
      return vec_norm(diff) / std::max(1e-8, vec_norm(anchor_));
    }());
    st_.last_decision = "accepted (alpha=" + fmt(used_alpha, 2) + ") " + reason;
    st_.last_round_time = Telemetry::now() - t_start;
    st_.weight_version = weights_.version();
    for (int i = 0; i < kNumSources; ++i) {
      st_.active[i] = has[i];
      for (int j = 0; j < kNumSources; ++j) st_.cos[i][j] = cosm[i][j];
    }
    // bandit credit: reward sources aligned with the accepted direction
    const double improvement = static_cast<double>(before.loss - after.loss);
    for (int i = 0; i < kNumSources; ++i) {
      if (!has[i]) continue;
      const double ni = vec_norm(d[i]), nm = vec_norm(merged);
      const double align = (ni > 0 && nm > 0) ? vec_dot(d[i], merged) / (ni * nm) : 0.0;
      credit_[i] = std::min(2.5, std::max(0.05, credit_[i] * (1.0 + 0.10 * align) +
                                                    2.0 * improvement * std::max(0.0, align)));
    }
    if (after.loss < best_val_) {
      best_val_ = after.loss;
      best_snapshot_ = committed;
      st_.best_val = best_val_;
    }
    mem_backups_.push_back(committed);
    while (static_cast<int>(mem_backups_.size()) > cfg_.max_memory_backups)
      mem_backups_.pop_front();
    decision = st_.last_decision;
    round_no = st_.rounds;
  } else {
    std::lock_guard<std::mutex> g(st_m_);
    ++st_.rounds;
    ++st_.rejected;
    ++st_.rollbacks;
    st_.merged_proposals += merged_count;
    st_.last_decision = "auto-rollback, rejected: " + reason;
    st_.last_round_time = Telemetry::now() - t_start;
    st_.last_alpha = 0.0f;
    st_.last_delta_norm = 0.0f;
    for (int i = 0; i < kNumSources; ++i) {
      st_.active[i] = has[i];
      for (int j = 0; j < kNumSources; ++j) st_.cos[i][j] = cosm[i][j];
      if (has[i]) credit_[i] = std::max(0.05, credit_[i] * 0.8);
    }
    decision = st_.last_decision;
    round_no = st_.rounds;
  }

  // Telemetry, logging and disk I/O happen *outside* the stats lock so the
  // dashboard never waits on a file write.
  if (tel_) {
    tel_->set_coord(stats());
    if (accepted) {
      tel_->push_loss(Stream::kHoldout, after.loss, round_no);
      tel_->log("accept", "coordinator", decision, fields);
    } else {
      tel_->log("reject", "coordinator", decision, fields);
    }
  }
  if (accepted && committed && cfg_.save_every_rounds > 0 &&
      round_no % cfg_.save_every_rounds == 0)
    save_backup(*committed, round_no, after.loss);

  if (cfg_.enable_fisher && cfg_.fisher_refresh_rounds > 0 &&
      round_no % cfg_.fisher_refresh_rounds == 0)
    refresh_fisher();
}

}  // namespace slm
