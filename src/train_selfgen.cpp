// SPDX-License-Identifier: Apache-2.0
#include "train_selfgen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

namespace slm {

SelfGenConfig SelfGenConfig::from_config(const Config& c) {
  SelfGenConfig k;
  k.samples_per_round = static_cast<int>(c.get_int("selfgen.samples_per_round", k.samples_per_round));
  k.max_new_tokens = static_cast<int>(c.get_int("selfgen.max_new_tokens", k.max_new_tokens));
  k.temperature = static_cast<float>(c.get_num("selfgen.temperature", k.temperature));
  k.top_k = static_cast<int>(c.get_int("selfgen.top_k", k.top_k));
  k.top_p = static_cast<float>(c.get_num("selfgen.top_p", k.top_p));
  k.ppl_min = static_cast<float>(c.get_num("selfgen.ppl_min", k.ppl_min));
  k.ppl_max = static_cast<float>(c.get_num("selfgen.ppl_max", k.ppl_max));
  k.ppl_min_ratio = static_cast<float>(c.get_num("selfgen.ppl_min_ratio", k.ppl_min_ratio));
  k.ppl_max_ratio = static_cast<float>(c.get_num("selfgen.ppl_max_ratio", k.ppl_max_ratio));
  k.require_novel = c.get_bool("selfgen.require_novel", k.require_novel);
  k.max_repeat_ratio = static_cast<float>(c.get_num("selfgen.max_repeat_ratio", k.max_repeat_ratio));
  k.min_tokens = static_cast<int>(c.get_int("selfgen.min_tokens", k.min_tokens));
  k.keep_best = static_cast<int>(c.get_int("selfgen.keep_best", k.keep_best));
  k.buffer_cap = c.get_int("selfgen.buffer_cap", k.buffer_cap);
  return k;
}

SelfGenTrainer::SelfGenTrainer(Coordinator* coord, Telemetry* tel,
                               const GPTConfig& mcfg, const TrainerConfig& tcfg,
                               const SelfGenConfig& scfg, const Tokenizer* tok,
                               const TokenDataset* corpus, uint64_t seed)
    : TrainerBase(Source::kSelfGen, coord, tel, mcfg, tcfg, tok, corpus, seed),
      scfg_(scfg) {
  build_prompt_pool();
}

// Prompts are mined straight out of the corpus: every <|user|> ... <|assistant|>
// span becomes a seed.  Corpora without control tokens fall back to random
// token windows.
void SelfGenTrainer::build_prompt_pool() {
  if (!corpus_) return;
  const std::vector<int32_t>& t = corpus_->tokens();
  const int64_t limit = corpus_->train_tokens();
  for (int64_t i = 0; i < limit && static_cast<int>(prompts_.size()) < 2048; ++i) {
    if (t[static_cast<size_t>(i)] != Tokenizer::kUser) continue;
    int64_t j = i + 1;
    while (j < limit && t[static_cast<size_t>(j)] != Tokenizer::kAssistant &&
           j - i < 64)
      ++j;
    if (j >= limit || t[static_cast<size_t>(j)] != Tokenizer::kAssistant) continue;
    std::vector<int32_t> p(t.begin() + i, t.begin() + j + 1);  // includes markers
    if (p.size() >= 3) prompts_.push_back(std::move(p));
    i = j;
  }
  if (prompts_.empty() && !t.empty()) {
    for (int i = 0; i < 256; ++i) {
      const int64_t off = static_cast<int64_t>(rng_.below(static_cast<uint64_t>(std::max<int64_t>(1, limit - 24))));
      prompts_.emplace_back(t.begin() + off, t.begin() + std::min<int64_t>(limit, off + 12));
    }
  }
  if (tel_)
    tel_->log("info", source_name(src_),
              "prompt pool built: " + std::to_string(prompts_.size()) + " seeds");
}

std::vector<int32_t> SelfGenTrainer::pick_prompt() {
  if (prompts_.empty()) return {Tokenizer::kEot};
  return prompts_[static_cast<size_t>(rng_.below(prompts_.size()))];
}

float SelfGenTrainer::response_logprob(const std::vector<int32_t>& full,
                                       size_t resp_start, int64_t* ntok) {
  NoGradGuard ng;
  const int64_t T = std::min<int64_t>(static_cast<int64_t>(full.size()) - 1,
                                      mcfg_.block_size);
  if (T <= 0) {
    if (ntok) *ntok = 0;
    return 0.0f;
  }
  std::vector<int32_t> ids(full.begin(), full.begin() + T);
  std::vector<int32_t> tgt(static_cast<size_t>(T));
  int64_t n = 0;
  for (int64_t i = 0; i < T; ++i) {
    const size_t next = static_cast<size_t>(i) + 1;
    if (next >= resp_start && next < full.size()) {
      tgt[static_cast<size_t>(i)] = full[next];
      ++n;
    } else {
      tgt[static_cast<size_t>(i)] = -100;
    }
  }
  if (n == 0) {
    if (ntok) *ntok = 0;
    return 0.0f;
  }
  Tensor logits = model_->forward(ids, 1, T, ForwardOptions());
  Tensor lp = seq_logprob(logits, tgt, -100);
  if (ntok) *ntok = n;
  return lp.host_ptr()[0];
}

uint64_t SelfGenTrainer::sample_hash(const std::vector<int32_t>& ids, size_t from) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = from; i < ids.size(); ++i) {
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(ids[i]));
    h *= 1099511628211ull;
  }
  return h;
}

float SelfGenTrainer::repetition_ratio(const std::vector<int32_t>& ids, size_t from) {
  if (ids.size() < from + 8) return 0.0f;
  std::set<uint64_t> seen;
  int64_t total = 0, dup = 0;
  for (size_t i = from; i + 3 < ids.size(); ++i) {
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(ids[i])) << 48) ^
                         (static_cast<uint64_t>(static_cast<uint32_t>(ids[i + 1])) << 32) ^
                         (static_cast<uint64_t>(static_cast<uint32_t>(ids[i + 2])) << 16) ^
                         static_cast<uint64_t>(static_cast<uint32_t>(ids[i + 3]));
    ++total;
    if (!seen.insert(key).second) ++dup;
  }
  return total ? static_cast<float>(dup) / static_cast<float>(total) : 0.0f;
}

bool SelfGenTrainer::ready() {
  state_.pending_inputs = static_cast<int64_t>(buffer_.size());
  bump_state();
  return !prompts_.empty();
}

void SelfGenTrainer::round() {
  set_status("generating candidates");
  sync();

  GenOptions go;
  go.max_new_tokens = scfg_.max_new_tokens;
  go.temperature = scfg_.temperature;
  go.top_k = scfg_.top_k;
  go.top_p = scfg_.top_p;
  go.stop_on_eot = true;

  // Quality band, anchored to what this model can currently do.
  const float ref_ppl =
      coord_ ? std::exp(std::min(6.0f, coord_->baseline().loss)) : 2.0f;
  const float ppl_lo = std::max(scfg_.ppl_min, ref_ppl * scfg_.ppl_min_ratio);
  const float ppl_hi = std::min(scfg_.ppl_max, ref_ppl * scfg_.ppl_max_ratio);

  std::vector<Candidate> cands;
  for (int i = 0; i < scfg_.samples_per_round && !should_stop(); ++i) {
    Candidate c;
    std::vector<int32_t> prompt = pick_prompt();
    go.seed = rng_.next_u64() | 1ull;
    std::vector<int32_t> resp =
        model_->generate(prompt, go, [this](const GenStep&) { return !should_stop(); });
    if (static_cast<int>(resp.size()) < scfg_.min_tokens) {
      c.verdict = "too short (" + std::to_string(resp.size()) + " tokens)";
      c.text = tok_->decode(resp);
      c.ids = resp;
      cands.push_back(std::move(c));
      continue;
    }
    c.ids = prompt;
    c.response_start = prompt.size();
    c.ids.insert(c.ids.end(), resp.begin(), resp.end());
    if (static_cast<int64_t>(c.ids.size()) > mcfg_.block_size + 1)
      c.ids.resize(static_cast<size_t>(mcfg_.block_size) + 1);
    int64_t n = 0;
    const float lp = response_logprob(c.ids, c.response_start, &n);
    c.ppl = n > 0 ? std::exp(-lp / static_cast<float>(n)) : 1e9f;
    c.repeat = repetition_ratio(c.ids, c.response_start);
    c.text = tok_->decode(std::vector<int32_t>(c.ids.begin() + static_cast<long>(c.response_start),
                                               c.ids.end()));
    const uint64_t hash = sample_hash(c.ids, c.response_start);
    if (c.ppl < ppl_lo)
      c.verdict = "degenerate (ppl " + std::to_string(c.ppl) + " < " +
                  std::to_string(ppl_lo) + ")";
    else if (c.ppl > ppl_hi)
      c.verdict = "incoherent (ppl " + std::to_string(c.ppl) + " > " +
                  std::to_string(ppl_hi) + ")";
    else if (c.repeat > scfg_.max_repeat_ratio)
      c.verdict = "repetitive (4-gram dup ratio " + std::to_string(c.repeat) + ")";
    else if (scfg_.require_novel && accepted_hashes_.count(hash))
      c.verdict = "duplicate of an already accepted sample";
    else
      c.verdict.clear();
    c.hash = hash;
    // prefer mid-band perplexity and low repetition
    const float center = 0.5f * (ppl_lo + ppl_hi);
    c.score = -std::fabs(c.ppl - center) / center - 2.0f * c.repeat;
    cands.push_back(std::move(c));
  }

  std::vector<Candidate*> good;
  for (Candidate& c : cands)
    if (c.verdict.empty()) good.push_back(&c);
  std::sort(good.begin(), good.end(),
            [](const Candidate* a, const Candidate* b) { return a->score > b->score; });
  if (static_cast<int>(good.size()) > scfg_.keep_best)
    good.resize(static_cast<size_t>(scfg_.keep_best));

  for (Candidate& c : cands) {
    const bool kept = std::find(good.begin(), good.end(), &c) != good.end();
    if (kept) ++accepted_total_;
    else ++rejected_total_;
    if (tel_)
      tel_->log(kept ? "augment" : "filtered", source_name(src_),
                kept ? "accepted synthetic sample" : ("dropped: " + (c.verdict.empty() ? "not in top-k" : c.verdict)),
                {{"ppl", std::to_string(c.ppl)},
                 {"repeat", std::to_string(c.repeat)},
                 {"tokens", std::to_string(c.ids.size())},
                 {"text", c.text.substr(0, 160)}});
  }
  for (const Candidate* c : good) {
    accepted_hashes_.insert(c->hash);
    if (accepted_hashes_.size() > 8192) accepted_hashes_.clear();
    buffer_.push_back(c->ids);
    while (static_cast<int64_t>(buffer_.size()) > scfg_.buffer_cap) buffer_.pop_front();
  }
  state_.pending_inputs = static_cast<int64_t>(buffer_.size());
  if (buffer_.empty()) {
    set_status("no sample passed the quality filter");
    return;
  }

  set_status("training on augmentation buffer");
  const int64_t ctx = std::min<int64_t>(tcfg_.ctx, mcfg_.block_size);
  std::vector<std::vector<int32_t>> seqs;
  const size_t take = std::min<size_t>(buffer_.size(), static_cast<size_t>(tcfg_.batch) * 4);
  for (size_t i = buffer_.size() - take; i < buffer_.size(); ++i) seqs.push_back(buffer_[i]);
  std::vector<Batch> batches = TokenDataset::batches_from_sequences(seqs, tcfg_.batch, ctx);
  const int64_t n_new = static_cast<int64_t>(batches.size());
  add_replay(&batches, std::max<int64_t>(1, (n_new * tcfg_.replay_percent) /
                                                std::max(1, 100 - tcfg_.replay_percent)));
  float before = batches.empty() ? 0.0f
                                 : model_->eval_loss(batches[0].ids, batches[0].targets,
                                                     batches[0].B, batches[0].T);
  int64_t steps = 0;
  const float after = train_batches(batches, &steps);

  char note[256];
  std::snprintf(note, sizeof(note),
                "%zu/%zu synthetic samples kept (total %lld kept / %lld dropped), %lld steps",
                good.size(), cands.size(), static_cast<long long>(accepted_total_),
                static_cast<long long>(rejected_total_), static_cast<long long>(steps));
  submit_delta(before, after, static_cast<int64_t>(good.size()), steps, note);
}

}  // namespace slm
