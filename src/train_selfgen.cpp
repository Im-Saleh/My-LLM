// SPDX-License-Identifier: Apache-2.0
#include "train_selfgen.h"

#include <algorithm>

#include "core/text.h"
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
  k.max_repeat_ratio = static_cast<float>(c.get_num("selfgen.max_repeat_ratio", k.max_repeat_ratio));
  k.min_tokens = static_cast<int>(c.get_int("selfgen.min_tokens", k.min_tokens));
  k.keep_best = static_cast<int>(c.get_int("selfgen.keep_best", k.keep_best));
  k.buffer_cap = c.get_int("selfgen.buffer_cap", k.buffer_cap);
  k.require_novel = c.get_bool("selfgen.require_novel", k.require_novel);
  k.balance_languages = c.get_bool("selfgen.balance_languages", k.balance_languages);
  k.max_code_switch = static_cast<float>(c.get_num("selfgen.max_code_switch", k.max_code_switch));
  k.require_valid_python = c.get_bool("selfgen.require_valid_python", k.require_valid_python);
  k.py_min_comment_ratio =
      static_cast<float>(c.get_num("selfgen.py_min_comment_ratio", k.py_min_comment_ratio));
  return k;
}

SelfGenTrainer::SelfGenTrainer(Coordinator* coord, Telemetry* tel, const GPTConfig& mcfg,
                               const TrainerConfig& tcfg, const SelfGenConfig& scfg,
                               const Tokenizer* tok, const MixtureDataset* corpus,
                               uint64_t seed)
    : TrainerBase(Source::kSelfGen, coord, tel, mcfg, tcfg, tok, corpus, seed),
      scfg_(scfg) {
  build_prompt_pools();
}

// One prompt pool per corpus source: every <|user|> ... <|assistant|> span
// becomes a seed.  Sources without control tokens fall back to random windows.
void SelfGenTrainer::build_prompt_pools() {
  if (!corpus_) return;
  for (int si = 0; si < corpus_->num_sources(); ++si) {
    Pool pool;
    pool.source = si;
    pool.lang = corpus_->info(si).lang;
    const std::vector<int32_t>& t = corpus_->data(si).tokens();
    const int64_t limit = corpus_->data(si).train_tokens();
    for (int64_t i = 0; i < limit && pool.prompts.size() < 1024; ++i) {
      if (t[static_cast<size_t>(i)] != Tokenizer::kUser) continue;
      int64_t j = i + 1;
      while (j < limit && t[static_cast<size_t>(j)] != Tokenizer::kAssistant && j - i < 64) ++j;
      if (j >= limit || t[static_cast<size_t>(j)] != Tokenizer::kAssistant) continue;
      std::vector<int32_t> p(t.begin() + i, t.begin() + j + 1);  // markers included
      if (p.size() >= 3) pool.prompts.push_back(std::move(p));
      i = j;
    }
    if (pool.prompts.empty() && limit > 32) {
      for (int i = 0; i < 128; ++i) {
        const int64_t off = static_cast<int64_t>(
            rng_.below(static_cast<uint64_t>(std::max<int64_t>(1, limit - 24))));
        pool.prompts.emplace_back(t.begin() + off,
                                  t.begin() + std::min<int64_t>(limit, off + 12));
      }
    }
    if (!pool.prompts.empty()) pools_.push_back(std::move(pool));
  }
  if (tel_) {
    std::string desc;
    for (const Pool& p : pools_)
      desc += std::string(lang_code(p.lang)) + ":" + std::to_string(p.prompts.size()) + " ";
    tel_->log("info", source_name(src_), "prompt pools built", {{"seeds", desc}});
  }
}

const SelfGenTrainer::Pool* SelfGenTrainer::pick_pool(int index) {
  if (pools_.empty()) return nullptr;
  if (scfg_.balance_languages)
    return &pools_[static_cast<size_t>((rr_ + index) % static_cast<int>(pools_.size()))];
  return &pools_[static_cast<size_t>(rng_.below(pools_.size()))];
}

float SelfGenTrainer::response_logprob(const std::vector<int32_t>& full, size_t resp_start,
                                       int64_t* ntok) {
  NoGradGuard ng;
  const int64_t T = std::min<int64_t>(static_cast<int64_t>(full.size()) - 1, mcfg_.block_size);
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

uint64_t SelfGenTrainer::sample_hash(const std::vector<int32_t>& ids, size_t from) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = from; i < ids.size(); ++i) {
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(ids[i]));
    h *= 1099511628211ull;
  }
  return h;
}

// Pulls the body out of a ```python ... ``` fence when the model produced one.
std::string SelfGenTrainer::extract_code(const std::string& text) {
  const size_t open = text.find("```");
  if (open == std::string::npos) return text;
  size_t body = text.find('\n', open);
  if (body == std::string::npos) return text;
  ++body;
  const size_t close = text.find("```", body);
  return text.substr(body, close == std::string::npos ? std::string::npos : close - body);
}

void SelfGenTrainer::judge(Candidate* c, float ppl_lo, float ppl_hi) const {
  c->verdict.clear();
  if (c->ppl < ppl_lo) {
    c->verdict = "degenerate (ppl " + std::to_string(c->ppl) + " < " + std::to_string(ppl_lo) + ")";
    return;
  }
  if (c->ppl > ppl_hi) {
    c->verdict = "incoherent (ppl " + std::to_string(c->ppl) + " > " + std::to_string(ppl_hi) + ")";
    return;
  }
  if (c->repeat > scfg_.max_repeat_ratio) {
    c->verdict = "repetitive (4-gram dup " + std::to_string(c->repeat) + ")";
    return;
  }
  if (scfg_.require_novel && accepted_hashes_.count(c->hash)) {
    c->verdict = "duplicate of an already accepted sample";
    return;
  }
  if (c->lang == Lang::kPython) {
    const std::string code = extract_code(c->text);
    const CodeCheck chk = check_python(code);
    if (scfg_.require_valid_python && !chk.ok) {
      c->verdict = "invalid python: " + chk.reason;
      return;
    }
    if (code.find("def ") == std::string::npos && code.find("return") == std::string::npos &&
        code.find('=') == std::string::npos) {
      c->verdict = "python sample has no statement";
      return;
    }
    if (chk.comment_ratio < scfg_.py_min_comment_ratio) {
      c->verdict = "python sample is undocumented (" + std::to_string(chk.comment_ratio) + ")";
      return;
    }
  } else if (c->lang == Lang::kPersian || c->lang == Lang::kEnglish) {
    c->switch_ratio = static_cast<float>(code_switch_ratio(c->text, c->lang));
    if (c->switch_ratio > scfg_.max_code_switch) {
      c->verdict = "language interference (foreign letters " +
                   std::to_string(c->switch_ratio) + ")";
      return;
    }
  }
}

bool SelfGenTrainer::ready() {
  state_.pending_inputs = static_cast<int64_t>(buffer_.size());
  bump_state();
  return !pools_.empty();
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

  // Quality band anchored to what this model can currently do; per language
  // when the coordinator reports a per-language hold-out loss.
  const CoordinatorStats cs = coord_ ? coord_->stats() : CoordinatorStats{};
  auto band = [&](Lang l, float* lo, float* hi) {
    const int li = static_cast<int>(l);
    const float loss = (li < kNumLangs && cs.lang_present[li] && cs.lang_val[li] > 0.0f)
                           ? cs.lang_val[li]
                           : (coord_ ? coord_->baseline().loss : 1.0f);
    const float ref = std::exp(std::min(6.0f, loss));
    *lo = std::max(scfg_.ppl_min, ref * scfg_.ppl_min_ratio);
    *hi = std::min(scfg_.ppl_max, ref * scfg_.ppl_max_ratio);
  };

  std::vector<Candidate> cands;
  for (int i = 0; i < scfg_.samples_per_round && !should_stop(); ++i) {
    const Pool* pool = pick_pool(i);
    if (!pool) break;
    Candidate c;
    c.lang = pool->lang;
    c.source = pool->source;
    std::vector<int32_t> prompt =
        pool->prompts[static_cast<size_t>(rng_.below(pool->prompts.size()))];
    go.seed = rng_.next_u64() | 1ull;
    std::vector<int32_t> resp =
        model_->generate(prompt, go, [this](const GenStep&) { return !should_stop(); });
    c.ids = prompt;
    c.response_start = prompt.size();
    c.ids.insert(c.ids.end(), resp.begin(), resp.end());
    if (static_cast<int64_t>(c.ids.size()) > mcfg_.block_size + 1)
      c.ids.resize(static_cast<size_t>(mcfg_.block_size) + 1);
    c.text = tok_->decode(std::vector<int32_t>(
        c.ids.begin() + static_cast<long>(c.response_start), c.ids.end()));
    if (static_cast<int>(resp.size()) < scfg_.min_tokens) {
      c.verdict = "too short (" + std::to_string(resp.size()) + " tokens)";
      cands.push_back(std::move(c));
      continue;
    }
    int64_t n = 0;
    const float lp = response_logprob(c.ids, c.response_start, &n);
    c.ppl = n > 0 ? std::exp(-lp / static_cast<float>(n)) : 1e9f;
    c.repeat = repetition_ratio(c.ids, c.response_start);
    c.hash = sample_hash(c.ids, c.response_start);
    float lo = 0.0f, hi = 0.0f;
    band(c.lang, &lo, &hi);
    judge(&c, lo, hi);
    const float center = 0.5f * (lo + hi);
    c.score = -std::fabs(c.ppl - center) / std::max(1e-3f, center) - 2.0f * c.repeat -
              1.5f * c.switch_ratio;
    cands.push_back(std::move(c));
  }
  rr_ = (rr_ + scfg_.samples_per_round) % std::max<int>(1, static_cast<int>(pools_.size()));

  // Keep the best survivors, but never let one language take every slot.
  std::vector<Candidate*> good;
  for (Candidate& c : cands)
    if (c.verdict.empty()) good.push_back(&c);
  std::sort(good.begin(), good.end(),
            [](const Candidate* a, const Candidate* b) { return a->score > b->score; });
  const int per_lang_cap =
      std::max(1, (scfg_.keep_best + static_cast<int>(pools_.size()) - 1) /
                      std::max<int>(1, static_cast<int>(pools_.size())));
  std::vector<Candidate*> kept;
  int taken[kNumLangs] = {};
  for (Candidate* c : good) {
    const int li = static_cast<int>(c->lang);
    if (taken[li] >= per_lang_cap) continue;
    kept.push_back(c);
    ++taken[li];
    if (static_cast<int>(kept.size()) >= scfg_.keep_best) break;
  }
  for (Candidate* c : good) {  // fill the remaining slots if some language had none
    if (static_cast<int>(kept.size()) >= scfg_.keep_best) break;
    if (std::find(kept.begin(), kept.end(), c) == kept.end()) kept.push_back(c);
  }

  for (Candidate& c : cands) {
    const bool ok = std::find(kept.begin(), kept.end(), &c) != kept.end();
    const int li = static_cast<int>(c.lang);
    if (ok) {
      ++accepted_total_;
      ++accepted_lang_[li];
    } else {
      ++rejected_total_;
      ++rejected_lang_[li];
    }
    if (tel_)
      tel_->log(ok ? "augment" : "filtered", source_name(src_),
                ok ? "accepted synthetic sample"
                   : ("dropped: " + (c.verdict.empty() ? "not in top-k" : c.verdict)),
                {{"lang", lang_code(c.lang)},
                 {"ppl", std::to_string(c.ppl)},
                 {"repeat", std::to_string(c.repeat)},
                 {"switch", std::to_string(c.switch_ratio)},
                 {"tokens", std::to_string(c.ids.size())},
                 {"text", utf8_truncate(c.text, 160)}});
  }
  for (const Candidate* c : kept) {
    accepted_hashes_.insert(c->hash);
    if (accepted_hashes_.size() > 8192) accepted_hashes_.clear();
    buffer_.emplace_back(c->lang, c->ids);
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
  for (size_t i = buffer_.size() - take; i < buffer_.size(); ++i)
    seqs.push_back(buffer_[i].second);
  std::vector<Batch> batches = TokenDataset::batches_from_sequences(seqs, tcfg_.batch, ctx);
  const int64_t n_new = static_cast<int64_t>(batches.size());
  add_replay(&batches, std::max<int64_t>(1, (n_new * tcfg_.replay_percent) /
                                                std::max(1, 100 - tcfg_.replay_percent)));
  float before = batches.empty() ? 0.0f
                                 : model_->eval_loss(batches[0].ids, batches[0].targets,
                                                     batches[0].B, batches[0].T);
  int64_t steps = 0;
  const float after = train_batches(batches, &steps);

  std::string per_lang;
  for (int l = 0; l < kNumLangs; ++l)
    if (accepted_lang_[l] || rejected_lang_[l])
      per_lang += std::string(lang_code(static_cast<Lang>(l))) + " " +
                  std::to_string(accepted_lang_[l]) + "/" +
                  std::to_string(accepted_lang_[l] + rejected_lang_[l]) + "  ";
  char note[384];
  std::snprintf(note, sizeof(note),
                "%zu/%zu synthetic samples kept (session kept/seen per language: %s), %lld steps",
                kept.size(), cands.size(), per_lang.c_str(), static_cast<long long>(steps));
  submit_delta(before, after, static_cast<int64_t>(kept.size()), steps, note);
}

}  // namespace slm
