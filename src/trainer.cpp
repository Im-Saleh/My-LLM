// SPDX-License-Identifier: Apache-2.0
#include "trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

namespace slm {

TrainerBase::TrainerBase(Source src, Coordinator* coord, Telemetry* tel,
                         const GPTConfig& mcfg, const TrainerConfig& tcfg,
                         const Tokenizer* tok, const MixtureDataset* corpus,
                         uint64_t seed)
    : src_(src),
      coord_(coord),
      tel_(tel),
      tok_(tok),
      corpus_(corpus),
      tcfg_(tcfg),
      mcfg_(mcfg),
      rng_(seed) {
  model_ = std::make_unique<GPT>(mcfg_);
  model_->set_freeze_policy(tcfg_.freeze);
  AdamWConfig oc;
  oc.lr = tcfg_.lr;
  oc.weight_decay = tcfg_.weight_decay;
  oc.grad_clip = tcfg_.grad_clip;
  opt_ = std::make_unique<AdamW>(oc);
  opt_->set_params(model_->trainable_params(), model_->trainable_names());
  state_.lr = tcfg_.lr;
  state_.policy = tcfg_.freeze.describe();
  state_.status = "starting";
  if (tel_) {
    tel_->set_trainer_enabled(src_, true);
    tel_->set_trainer(src_, state_);
  }
  sync();
}

TrainerBase::~TrainerBase() { stop(); }

void TrainerBase::start() {
  if (run_.exchange(true)) return;
  th_ = std::thread([this] { loop(); });
}

void TrainerBase::stop() {
  if (!run_.exchange(false)) return;
  if (th_.joinable()) th_.join();
}

bool TrainerBase::should_stop() const {
  return !run_.load() || (tel_ && tel_->stopped());
}

void TrainerBase::set_status(const std::string& s) {
  state_.status = s;
  bump_state();
}

void TrainerBase::bump_state() {
  if (!tel_) return;
  state_.credit = coord_ ? coord_->credit(src_) : 1.0;
  tel_->set_trainer(src_, state_);
}

void TrainerBase::sync() {
  if (!coord_) return;
  ParamStorePtr base = coord_->snapshot(&base_version_);
  if (!base) return;
  model_->load_params(*base);
  coord_->flat().gather(*base, base_flat_);
  // A fresh optimiser moment estimate per round keeps the update honest: the
  // delta we hand to the coordinator is then a function of *this* round's data
  // only.
  opt_->reset_state();
}

float TrainerBase::step_on(const Batch& b) {
  model_->zero_grad();
  float lv = 0.0f;
  Tensor l = model_->loss(b.ids, b.targets, b.B, b.T, &lv, nullptr);
  l.backward();
  opt_->step(tcfg_.lr);
  return lv;
}

float TrainerBase::train_batches(const std::vector<Batch>& batches, int64_t* steps_done) {
  float last = 0.0f;
  int64_t steps = 0;
  for (int64_t pass = 0; pass < tcfg_.local_steps; ++pass) {
    for (const Batch& b : batches) {
      if (should_stop()) break;
      last = step_on(b);
      ++steps;
      state_.local_steps++;
      state_.last_loss = last;
      if (tel_) tel_->push_loss(static_cast<Stream>(static_cast<int>(src_)), last,
                                state_.local_steps);
      bump_state();
    }
    if (should_stop()) break;
  }
  if (steps_done) *steps_done = steps;
  return last;
}

void TrainerBase::add_replay(std::vector<Batch>* batches, int64_t how_many) {
  if (!corpus_ || corpus_->empty() || how_many <= 0) return;
  const int64_t ctx = std::min<int64_t>(tcfg_.ctx, mcfg_.block_size);
  const int n = corpus_->num_sources();
  // At least one batch per language, then top up in round-robin order.
  const int64_t rounds = std::max<int64_t>(1, (how_many + n - 1) / n);
  for (int64_t r = 0; r < rounds; ++r)
    for (int s = 0; s < n; ++s)
      batches->push_back(corpus_->sample_batch_from(s, tcfg_.batch, ctx, rng_));
}

void TrainerBase::submit_delta(float loss_start, float loss_end, int64_t samples,
                               int64_t steps, const std::string& note) {
  if (!coord_) return;
  UpdateProposal p;
  p.source = src_;
  p.created_at = Telemetry::now();
  p.base_version = base_version_;
  p.loss_start = loss_start;
  p.loss_end = loss_end;
  p.samples = samples;
  p.steps = steps;
  p.lr = tcfg_.lr;
  p.note = note;

  ParamStorePtr local = model_->snapshot();
  std::vector<float> now;
  coord_->flat().gather(*local, now);
  p.delta.resize(now.size());
  for (size_t i = 0; i < now.size(); ++i)
    p.delta[i] = now[i] - (i < base_flat_.size() ? base_flat_[i] : 0.0f);

  double dn = 0.0;
  for (float v : p.delta) dn += static_cast<double>(v) * v;
  dn = std::sqrt(dn);

  state_.rounds++;
  state_.samples_seen += samples;
  state_.loss_start = loss_start;
  state_.last_loss = loss_end;
  state_.last_round_t = Telemetry::now();
  bump_state();

  if (tel_)
    tel_->log("propose", source_name(src_), note,
              {{"samples", std::to_string(samples)},
               {"steps", std::to_string(steps)},
               {"loss", std::to_string(loss_start) + "->" + std::to_string(loss_end)},
               {"delta_norm", std::to_string(dn)},
               {"lr", std::to_string(tcfg_.lr)}});
  coord_->submit(std::move(p));
}

void TrainerBase::loop() {
  set_status("idle");
  while (run_.load()) {
    if (tel_ && tel_->stopped()) {
      set_status("halted (emergency stop)");
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    if (tel_ && !tel_->trainer_enabled(src_)) {
      set_status("disabled");
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    const double now = Telemetry::now();
    if (now - last_round_ < tcfg_.min_interval_s) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    if (!ready()) {
      set_status("waiting for data");
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    state_.busy = true;
    bump_state();
    try {
      round();
    } catch (const std::exception& e) {
      if (tel_) tel_->log("error", source_name(src_), std::string("round failed: ") + e.what());
      set_status(std::string("error: ") + e.what());
    }
    state_.busy = false;
    last_round_ = Telemetry::now();
    set_status("idle");
  }
  set_status("stopped");
}

}  // namespace slm
