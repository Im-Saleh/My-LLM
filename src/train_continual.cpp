// SPDX-License-Identifier: Apache-2.0
#include "train_continual.h"

#include <algorithm>
#include <cstdio>

namespace slm {

ContinualConfig ContinualConfig::from_config(const Config& c) {
  ContinualConfig k;
  k.min_samples = c.get_int("continual.min_samples", k.min_samples);
  k.max_wait_s = c.get_num("continual.max_wait_s", k.max_wait_s);
  k.buffer_cap = c.get_int("continual.buffer_cap", k.buffer_cap);
  return k;
}

ContinualTrainer::ContinualTrainer(Coordinator* coord, Telemetry* tel,
                                   InteractionHub* hub, const GPTConfig& mcfg,
                                   const TrainerConfig& tcfg,
                                   const ContinualConfig& ccfg, const Tokenizer* tok,
                                   const MixtureDataset* corpus, uint64_t seed)
    : TrainerBase(Source::kContinual, coord, tel, mcfg, tcfg, tok, corpus, seed),
      hub_(hub),
      ccfg_(ccfg) {}

bool ContinualTrainer::ready() {
  if (hub_) {
    for (std::string& s : hub_->drain_texts(64)) {
      if (buffer_.empty()) first_seen_ = Telemetry::now();
      buffer_.push_back(std::move(s));
      while (static_cast<int64_t>(buffer_.size()) > ccfg_.buffer_cap) buffer_.pop_front();
    }
  }
  state_.pending_inputs = static_cast<int64_t>(buffer_.size());
  bump_state();
  if (buffer_.empty()) return false;
  if (static_cast<int64_t>(buffer_.size()) >= ccfg_.min_samples) return true;
  return Telemetry::now() - first_seen_ > ccfg_.max_wait_s;
}

void ContinualTrainer::round() {
  set_status("fine-tuning on new user data");
  sync();

  std::vector<std::string> take(buffer_.begin(), buffer_.end());
  buffer_.clear();
  state_.pending_inputs = 0;

  const int64_t ctx = std::min<int64_t>(tcfg_.ctx, mcfg_.block_size);
  std::vector<std::vector<int32_t>> seqs;
  size_t chars = 0;
  int per_lang[kNumLangs] = {};
  for (const std::string& s : take) {
    ++per_lang[static_cast<int>(detect_language(s))];
    std::vector<int32_t> ids = tok_->encode(s);
    if (ids.empty()) continue;
    ids.push_back(Tokenizer::kEot);
    if (static_cast<int64_t>(ids.size()) > ctx + 1) ids.resize(static_cast<size_t>(ctx + 1));
    seqs.push_back(std::move(ids));
    chars += s.size();
  }
  if (seqs.empty()) return;

  std::vector<Batch> batches = TokenDataset::batches_from_sequences(seqs, tcfg_.batch, ctx);
  const int64_t n_new = static_cast<int64_t>(batches.size());
  const int64_t n_replay =
      std::max<int64_t>(0, (n_new * tcfg_.replay_percent) / std::max(1, 100 - tcfg_.replay_percent));
  add_replay(&batches, n_replay);
  for (size_t i = batches.size(); i > 1; --i) {
    const size_t j = static_cast<size_t>(rng_.below(i));
    std::swap(batches[i - 1], batches[j]);
  }

  // measure before / after on the *new* data only
  float before = 0.0f;
  {
    for (const Batch& b : batches) {
      before = model_->eval_loss(b.ids, b.targets, b.B, b.T);
      break;
    }
  }
  int64_t steps = 0;
  const float after = train_batches(batches, &steps);

  std::string langs;
  for (int l = 0; l < kNumLangs; ++l)
    if (per_lang[l]) {
      if (!langs.empty()) langs += "/";
      langs += std::string(lang_code(static_cast<Lang>(l))) + ":" + std::to_string(per_lang[l]);
    }
  char note[320];
  std::snprintf(note, sizeof(note),
                "%zu user samples [%s] (%zu chars) + %lld replay batches, %lld steps",
                seqs.size(), langs.c_str(), chars, static_cast<long long>(n_replay),
                static_cast<long long>(steps));
  submit_delta(before, after, static_cast<int64_t>(seqs.size()), steps, note);
}

}  // namespace slm
