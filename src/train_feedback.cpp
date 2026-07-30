// SPDX-License-Identifier: Apache-2.0
#include "train_feedback.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace slm {

FeedbackConfig FeedbackConfig::from_config(const Config& c) {
  FeedbackConfig k;
  k.dpo_beta = static_cast<float>(c.get_num("feedback.dpo_beta", k.dpo_beta));
  k.sft_weight = static_cast<float>(c.get_num("feedback.sft_weight", k.sft_weight));
  k.good_score = static_cast<float>(c.get_num("feedback.good_score", k.good_score));
  k.min_score_gap = static_cast<float>(c.get_num("feedback.min_score_gap", k.min_score_gap));
  k.max_pairs_per_round =
      static_cast<int>(c.get_int("feedback.max_pairs_per_round", k.max_pairs_per_round));
  k.max_rwft_per_round =
      static_cast<int>(c.get_int("feedback.max_rwft_per_round", k.max_rwft_per_round));
  k.bank_cap = c.get_int("feedback.bank_cap", k.bank_cap);
  k.use_dpo = c.get_bool("feedback.use_dpo", k.use_dpo);
  return k;
}

FeedbackTrainer::FeedbackTrainer(Coordinator* coord, Telemetry* tel, InteractionHub* hub,
                                 const GPTConfig& mcfg, const TrainerConfig& tcfg,
                                 const FeedbackConfig& fcfg, const Tokenizer* tok,
                                 const TokenDataset* corpus, uint64_t seed)
    : TrainerBase(Source::kFeedback, coord, tel, mcfg, tcfg, tok, corpus, seed),
      hub_(hub),
      fcfg_(fcfg) {}

FeedbackTrainer::Item FeedbackTrainer::make_item(const RatedSample& s) const {
  Item it;
  std::vector<int32_t> p = tok_->encode("<|user|>" + s.prompt + "<|assistant|>");
  std::vector<int32_t> r = tok_->encode(s.response);
  r.push_back(Tokenizer::kEot);
  const int64_t cap = mcfg_.block_size + 1;
  if (static_cast<int64_t>(p.size()) > cap / 2)
    p.erase(p.begin(), p.end() - static_cast<long>(cap / 2));
  it.ids = p;
  it.response_start = p.size();
  it.ids.insert(it.ids.end(), r.begin(), r.end());
  if (static_cast<int64_t>(it.ids.size()) > cap) it.ids.resize(static_cast<size_t>(cap));
  it.score = s.score;
  it.response = s.response;
  return it;
}

void FeedbackTrainer::masked_targets(const Item& it, std::vector<int32_t>* ids,
                                     std::vector<int32_t>* tgt, int64_t* T) const {
  const int64_t len = std::min<int64_t>(static_cast<int64_t>(it.ids.size()) - 1,
                                        mcfg_.block_size);
  *T = std::max<int64_t>(1, len);
  ids->assign(it.ids.begin(), it.ids.begin() + *T);
  tgt->assign(static_cast<size_t>(*T), -100);
  for (int64_t i = 0; i < *T; ++i) {
    const size_t next = static_cast<size_t>(i) + 1;
    if (next >= it.response_start && next < it.ids.size())
      (*tgt)[static_cast<size_t>(i)] = it.ids[next];
  }
}

Tensor FeedbackTrainer::logprob_tensor(const Item& it) {
  std::vector<int32_t> ids, tgt;
  int64_t T = 0;
  masked_targets(it, &ids, &tgt, &T);
  Tensor logits = model_->forward(ids, 1, T, ForwardOptions());
  return seq_logprob(logits, tgt, -100);
}

float FeedbackTrainer::logprob_of(const Item& it) {
  NoGradGuard ng;
  return logprob_tensor(it).host_ptr()[0];
}

bool FeedbackTrainer::ready() {
  if (hub_) {
    for (const RatedSample& s : hub_->drain_ratings(64)) {
      Item it = make_item(s);
      if (it.ids.size() < it.response_start + 2) continue;
      std::vector<Item>& v = bank_[s.prompt];
      // replace an identical response instead of piling up duplicates
      auto dup = std::find_if(v.begin(), v.end(), [&](const Item& o) {
        return o.response == it.response;
      });
      if (dup != v.end())
        *dup = std::move(it);
      else {
        v.push_back(std::move(it));
        ++bank_items_;
      }
      if (tel_)
        tel_->log("info", source_name(src_), "rating stored",
                  {{"score", std::to_string(s.score)},
                   {"prompt", s.prompt.substr(0, 80)},
                   {"response", s.response.substr(0, 80)}});
    }
    while (bank_items_ > fcfg_.bank_cap && !bank_.empty()) {
      auto it = bank_.begin();
      bank_items_ -= static_cast<int64_t>(it->second.size());
      bank_.erase(it);
    }
  }
  state_.pending_inputs = bank_items_;
  bump_state();
  if (bank_items_ == 0) return false;
  // work exists if we can build at least one pair or one reward-weighted sample
  for (const auto& kv : bank_) {
    if (kv.second.size() >= 2) return true;
    for (const Item& i : kv.second)
      if (i.score >= fcfg_.good_score) return true;
  }
  return false;
}

void FeedbackTrainer::round() {
  set_status("applying preference feedback");
  sync();

  // ---------------------------------------------------------------- pairs
  std::vector<Pair> pairs;
  std::vector<const Item*> rwft;
  for (const auto& kv : bank_) {
    const std::vector<Item>& v = kv.second;
    if (v.size() >= 2) {
      const Item* best = &v[0];
      const Item* worst = &v[0];
      for (const Item& i : v) {
        if (i.score > best->score) best = &i;
        if (i.score < worst->score) worst = &i;
      }
      if (best != worst && best->score - worst->score >= fcfg_.min_score_gap) {
        pairs.push_back(Pair{best, worst, 0.0f, 0.0f});
        continue;
      }
    }
    for (const Item& i : v)
      if (i.score >= fcfg_.good_score) rwft.push_back(&i);
  }
  if (static_cast<int>(pairs.size()) > fcfg_.max_pairs_per_round)
    pairs.resize(static_cast<size_t>(fcfg_.max_pairs_per_round));
  if (static_cast<int>(rwft.size()) > fcfg_.max_rwft_per_round)
    rwft.resize(static_cast<size_t>(fcfg_.max_rwft_per_round));
  if (pairs.empty() && rwft.empty()) {
    set_status("no usable preference signal");
    return;
  }

  // Reference log probabilities: the model currently *is* the reference, so
  // this is the only moment they can be captured for free.
  for (Pair& p : pairs) {
    p.ref_win = logprob_of(*p.win);
    p.ref_lose = logprob_of(*p.lose);
  }

  const int64_t ctx = std::min<int64_t>(tcfg_.ctx, mcfg_.block_size);
  float first_loss = 0.0f, last_loss = 0.0f;
  int64_t steps = 0;
  const bool use_dpo = fcfg_.use_dpo && !pairs.empty();

  for (int64_t pass = 0; pass < tcfg_.local_steps && !should_stop(); ++pass) {
    model_->zero_grad();
    Tensor total;
    float value = 0.0f;

    if (use_dpo) {
      for (Pair& p : pairs) {
        Tensor lw = logprob_tensor(*p.win);
        Tensor ll = logprob_tensor(*p.lose);
        // z = beta * [(lw - ref_w) - (ll - ref_l)]
        Tensor konst = Tensor::full({1}, p.ref_lose - p.ref_win);
        Tensor z = lw.add(ll.scale(-1.0f)).add(konst).scale(fcfg_.dpo_beta);
        Tensor term = logsigmoid(z).scale(-1.0f);
        total = total.defined() ? total.add(term) : term;
      }
      // [1] -> scalar so the optional SFT term (a scalar) can be added
      total = total.scale(1.0f / static_cast<float>(pairs.size())).sum_all();
      // small SFT anchor on the preferred answers
      if (fcfg_.sft_weight > 0.0f) {
        std::vector<std::vector<int32_t>> seqs;
        for (const Pair& p : pairs) seqs.push_back(p.win->ids);
        std::vector<Batch> bs = TokenDataset::batches_from_sequences(seqs, 1, ctx);
        for (const Batch& b : bs) {
          float lv = 0.0f;
          Tensor ce = model_->loss(b.ids, b.targets, b.B, b.T, &lv, nullptr, false);
          total = total.add(ce.scale(fcfg_.sft_weight / static_cast<float>(bs.size())));
        }
      }
    } else {
      // reward weighted fine-tuning
      float wsum = 0.0f;
      for (const Item* i : rwft) wsum += std::max(0.1f, i->score / 5.0f);
      for (const Item* i : rwft) {
        std::vector<int32_t> ids, tgt;
        int64_t T = 0;
        masked_targets(*i, &ids, &tgt, &T);
        Tensor logits = model_->forward(ids, 1, T, ForwardOptions());
        Tensor ce = cross_entropy(logits.reshape({T, mcfg_.vocab_size}), tgt, -100,
                                  nullptr, nullptr);
        const float w = std::max(0.1f, i->score / 5.0f) / std::max(1e-6f, wsum);
        Tensor term = ce.scale(w);
        total = total.defined() ? total.add(term) : term;
      }
    }
    if (!total.defined()) break;
    value = total.host_ptr()[0];
    total.backward();
    opt_->step(tcfg_.lr);
    if (pass == 0) first_loss = value;
    last_loss = value;
    ++steps;
    state_.local_steps++;
    state_.last_loss = value;
    if (tel_) tel_->push_loss(Stream::kFeedback, value, state_.local_steps);
    bump_state();
  }
  if (steps == 0) return;

  if (use_dpo) pairs_total_ += static_cast<int64_t>(pairs.size());
  else rwft_total_ += static_cast<int64_t>(rwft.size());

  char note[256];
  std::snprintf(note, sizeof(note),
                "%s: %zu %s, beta=%.3f, %lld steps (session pairs=%lld rwft=%lld)",
                use_dpo ? "DPO" : "reward-weighted", use_dpo ? pairs.size() : rwft.size(),
                use_dpo ? "preference pairs" : "rated samples", fcfg_.dpo_beta,
                static_cast<long long>(steps), static_cast<long long>(pairs_total_),
                static_cast<long long>(rwft_total_));
  submit_delta(first_loss, last_loss,
               static_cast<int64_t>(use_dpo ? pairs.size() : rwft.size()), steps, note);
}

}  // namespace slm
