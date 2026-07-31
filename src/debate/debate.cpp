// SPDX-License-Identifier: Apache-2.0
#include "debate/debate.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "core/text.h"
#include "telemetry.h"

namespace slm {
namespace {

// The marker the critique round asks for.  A short ASCII marker survives a small
// model's sampling far better than JSON or a heading.
constexpr const char* kAnswerMarker = "ANSWER:";

std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// Word tokens for the similarity measure: lowercased, alphanumeric, and bytes
// >= 0x80 kept together so Persian words survive as words.
std::vector<std::string> words_of(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (unsigned char c : s) {
    const bool keep = std::isalnum(c) || c >= 0x80;
    if (keep) {
      cur.push_back(static_cast<char>(std::tolower(c)));
    } else if (!cur.empty()) {
      out.push_back(cur);
      cur.clear();
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

std::string first_line(const std::string& s, size_t max_chars) {
  const size_t nl = s.find('\n');
  std::string t = nl == std::string::npos ? s : s.substr(0, nl);
  return utf8_truncate(trim(t), max_chars);
}

}  // namespace

// ======================================================== similarity, clusters
double DebateEngine::answer_similarity(const std::string& a, const std::string& b) {
  // F1 over the multiset of words.  Chosen over edit distance because two
  // correct answers to the same question routinely say the same things in a
  // different order, and over cosine-on-embeddings because this must be exact,
  // dependency free and fast enough to run on every pair every round.
  const std::vector<std::string> wa = words_of(a), wb = words_of(b);
  if (wa.empty() || wb.empty()) return (wa.empty() && wb.empty()) ? 1.0 : 0.0;
  std::unordered_map<std::string, int> ca;
  for (const std::string& w : wa) ++ca[w];
  int overlap = 0;
  for (const std::string& w : wb) {
    auto it = ca.find(w);
    if (it != ca.end() && it->second > 0) {
      --it->second;
      ++overlap;
    }
  }
  if (overlap == 0) return 0.0;
  const double p = static_cast<double>(overlap) / static_cast<double>(wb.size());
  const double r = static_cast<double>(overlap) / static_cast<double>(wa.size());
  return 2.0 * p * r / (p + r);
}

std::vector<int> DebateEngine::cluster_answers(const std::vector<std::string>& texts,
                                              double threshold) {
  const size_t n = texts.size();
  std::vector<int> cl(n, -1);
  int next = 0;
  // Single-link agglomeration in one pass: an answer joins the first cluster it
  // is close enough to, otherwise it starts its own.  O(n^2) similarity calls on
  // a handful of answers is nothing, and single-link is the right linkage here
  // because "says the same thing" is transitive enough at these thresholds.
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (cl[j] < 0) continue;
      if (answer_similarity(texts[i], texts[j]) >= threshold) {
        cl[i] = cl[j];
        break;
      }
    }
    if (cl[i] < 0) cl[i] = next++;
  }
  return cl;
}

bool DebateEngine::parse_judge_score(const std::string& reply, double* score) {
  // Accepts "SCORE: 7.5", "score 7", "7/10", "**8**" and a bare leading number,
  // because that is the full set of things models actually emit.
  const std::string low = [&] {
    std::string t = reply;
    for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return t;
  }();
  size_t at = low.find("score");
  size_t from = 0;
  if (at != std::string::npos) from = at + 5;
  for (size_t i = from; i < reply.size(); ++i) {
    const char c = reply[i];
    if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '.')) continue;
    size_t j = i;
    while (j < reply.size() &&
           (std::isdigit(static_cast<unsigned char>(reply[j])) || reply[j] == '.'))
      ++j;
    // Whatever happens below, never restart inside this run of digits: "42" must
    // be rejected as 42, not silently re-read as the "2" that follows.
    struct Skip {
      size_t& i;
      size_t to;
      ~Skip() { i = to > 0 ? to - 1 : i; }
    } skip{i, j};
    try {
      double v = std::stod(reply.substr(i, j - i));
      // "8/10" and "0.8" both mean the same thing; normalise onto 0..10.
      if (j < reply.size() && reply[j] == '/') {
        size_t k = j + 1;
        size_t s = k;
        while (k < reply.size() && std::isdigit(static_cast<unsigned char>(reply[k]))) ++k;
        if (k > s) {
          const double den = std::stod(reply.substr(s, k - s));
          if (den > 0) v = v * 10.0 / den;
        }
      } else if (v <= 1.0 && reply.substr(i, j - i).find('.') != std::string::npos) {
        v *= 10.0;
      }
      if (v < 0.0 || v > 10.0) continue;
      if (score) *score = v;
      return true;
    } catch (...) {
      continue;
    }
  }
  return false;
}

// =============================================================== configuration
int DebateConfig::planned_rounds() const {
  if (rounds > 0) return rounds;
  int m = 1;
  for (const DebateParticipant& p : participants) m = std::max(m, p.multiplier);
  return m;
}

void DebateConfig::normalise() {
  int idx = 0;
  for (DebateParticipant& p : participants) {
    ++idx;
    if (p.multiplier < 1) p.multiplier = 1;
    if (p.multiplier > 8) p.multiplier = 8;   // beyond this it is only cost
    if (p.draft_cap <= 0) p.draft_cap = p.multiplier;
    if (p.vote_weight <= 0.0f) p.vote_weight = static_cast<float>(p.multiplier);
    if (p.name.empty())
      p.name = p.backend_id + "-" + std::string(1, static_cast<char>('A' + idx - 1));
  }
  if (rounds <= 0) rounds = planned_rounds();
  rounds = std::max(1, std::min(rounds, 6));
  max_answer_tokens = std::max(32, max_answer_tokens);
  max_critique_tokens = std::max(32, max_critique_tokens);
  max_final_tokens = std::max(32, max_final_tokens);
  cluster_threshold = std::min(0.95f, std::max(0.10f, cluster_threshold));
  const float sum = w_agreement + w_judge + w_fluency;
  if (sum > 0.0f) {
    w_agreement /= sum;
    w_judge /= sum;
    w_fluency /= sum;
  } else {
    w_agreement = w_judge = w_fluency = 1.0f / 3.0f;
  }
}

DebateConfig DebateConfig::two_model(const std::string& fast_id, int fast_mult,
                                     const std::string& strong_id, int strong_mult) {
  DebateConfig c;
  DebateParticipant a;
  a.backend_id = fast_id;
  a.name = "SPT";
  a.multiplier = std::max(1, fast_mult);
  a.persona =
      "You are concise and concrete. Prefer short, checkable statements over "
      "general ones.";
  a.cost_rank = 0.0f;
  DebateParticipant b;
  b.backend_id = strong_id;
  b.name = "OLMo";
  b.multiplier = std::max(1, strong_mult);
  // One draft only: extra participation from the expensive model is spent on
  // reviewing and synthesising, where it is worth far more per second.
  b.draft_cap = 1;
  b.persona =
      "You are careful and sceptical. Point out unsupported claims and missing "
      "cases before agreeing.";
  b.cost_rank = 1.0f;
  c.participants = {a, b};
  c.normalise();
  return c;
}

DebateConfig DebateConfig::self_debate(const std::string& backend_id, int voices,
                                       int multiplier) {
  DebateConfig c;
  // Two instances of the *same* weights still disagree usefully, as long as they
  // are decorrelated: different session slots (so different KV caches), different
  // seeds, and different personas. Without the persona split they collapse onto
  // the same answer and the debate degenerates into self-agreement.
  static const char* kPersonas[4] = {
      "You argue for the most direct solution and defend it with concrete detail.",
      "You look for what the other answer got wrong: edge cases, wrong "
      "assumptions, missing steps.",
      "You optimise for correctness over completeness; say less, be right.",
      "You optimise for completeness; make sure nothing important is left out."};
  const int n = std::max(2, std::min(4, voices));
  for (int i = 0; i < n; ++i) {
    DebateParticipant p;
    p.backend_id = backend_id;
    p.name = backend_id + "-" + std::string(1, static_cast<char>('A' + i));
    p.persona = kPersonas[i % 4];
    p.multiplier = std::max(1, multiplier);
    p.cost_rank = 0.0f;
    c.participants.push_back(p);
  }
  c.normalise();
  return c;
}

// ================================================================= transcript
const DebateAnswer* DebateTranscript::best() const {
  const DebateAnswer* b = nullptr;
  for (const DebateRound& r : rounds)
    for (const DebateAnswer& a : r.answers) {
      if (a.superseded) continue;
      if (!b || a.score > b->score) b = &a;
    }
  return b;
}

std::string DebateTranscript::summary() const {
  std::ostringstream os;
  int answers = 0;
  for (const DebateRound& r : rounds) answers += static_cast<int>(r.answers.size());
  os << rounds.size() << " rounds, " << answers << " answers, "
     << (escalated ? "escalated" : "cheap path") << ", synthesised by "
     << (synthesised_by.empty() ? "none" : synthesised_by) << ", " << seconds << " s";
  return os.str();
}

// ==================================================================== engine
DebateEngine::DebateEngine(BackendRegistry* reg, Telemetry* tel)
    : reg_(reg), tel_(tel) {}

namespace {

struct Actor {
  DebateParticipant spec;
  BackendPtr be;
  int slot = -1;
  int drafts = 0;         // round-0 drafts this actor produces
  double weight = 1.0;
  // Estimated decode speed, used only to weight the progress bar.
  double tps = 0.0;
};

// Rounds in which an actor engages.  With rounds=R and multiplier=m the actor is
// active in m of the R critique rounds, spread as evenly as possible - so a 2x
// participant really does take part twice as often as a 1x one, which is the
// whole point of the multiplier.
bool engages(int round_index, int rounds, int multiplier) {
  if (multiplier >= rounds) return true;
  if (multiplier <= 0) return false;
  // Bresenham-style spread: active when the count crosses the next boundary.
  const int a = (round_index * multiplier) / rounds;
  const int b = ((round_index + 1) * multiplier) / rounds;
  return b > a;
}

std::string digest_of(const std::vector<const DebateAnswer*>& peers, size_t per_peer) {
  std::string out;
  char label = 'A';
  for (const DebateAnswer* a : peers) {
    out += std::string("[") + label + "] " + a->participant + ": " +
           utf8_truncate(trim(a->text), per_peer) + "\n";
    ++label;
  }
  return out;
}

// Splits a critique reply into (critique, revised answer).
void split_reply(const std::string& reply, std::string* critique, std::string* answer) {
  const size_t at = reply.find(kAnswerMarker);
  if (at == std::string::npos) {
    // The model ignored the format.  Treating the whole reply as the answer is
    // the safe default: a missing critique costs a little signal, a missing
    // answer costs the round.
    *critique = "";
    *answer = trim(reply);
    return;
  }
  *critique = trim(reply.substr(0, at));
  *answer = trim(reply.substr(at + std::strlen(kAnswerMarker)));
  if (answer->empty()) *answer = *critique;
}

double normalise_fluency(double lp, double lo, double hi) {
  if (hi <= lo) return 0.5;
  const double t = (lp - lo) / (hi - lo);
  return std::min(1.0, std::max(0.0, t));
}

}  // namespace

struct DebateEngine::Plan {
  double total = 0.0;
  double done = 0.0;
  double unit(double tokens, double tps) const {
    // Predicted seconds, floored so an unmeasured backend still contributes.
    return tokens / std::max(4.0, tps);
  }
};

DebateTranscript DebateEngine::run(const std::string& question,
                                   const DebateConfig& cfg_in,
                                   std::atomic<bool>* cancel,
                                   const DebateObserver& on_update) {
  DebateConfig cfg = cfg_in;
  cfg.normalise();
  DebateTranscript tr;
  tr.question = question;
  const double t_start = Telemetry::now();

  // ------------------------------------------------------------- resolve actors
  std::vector<Actor> actors;
  for (const DebateParticipant& p : cfg.participants) {
    BackendPtr be = reg_ ? reg_->get(p.backend_id) : nullptr;
    if (!be) {
      tr.decision_log += "skipped " + p.name + ": no backend '" + p.backend_id + "'\n";
      continue;
    }
    if (!be->loaded()) {
      std::string err;
      if (!be->load(&err)) {
        tr.decision_log += "skipped " + p.name + ": " + err + "\n";
        continue;
      }
    }
    Actor a;
    a.spec = p;
    a.be = be;
    a.drafts = std::max(1, std::min(p.draft_cap, p.multiplier));
    a.weight = p.vote_weight;
    a.tps = be->status().last_decode_tps;
    actors.push_back(std::move(a));
  }
  if (actors.empty()) {
    tr.error = "no usable participant";
    tr.seconds = Telemetry::now() - t_start;
    return tr;
  }

  // One session per actor for the whole debate: this is what makes the shared
  // prefix (persona + question + context) survive every round.
  for (Actor& a : actors) a.slot = a.be->open_session();

  // ------------------------------------------------------------------- planning
  // Real progress needs an estimate of the *cost* of each planned unit, not a
  // count of units: one OLMo answer is worth fifty SPT answers in seconds.
  Plan plan;
  const int R = cfg.rounds;
  for (const Actor& a : actors) {
    plan.total += plan.unit(cfg.max_answer_tokens * a.drafts, a.tps);
    for (int r = 1; r < R; ++r)
      if (engages(r - 1, std::max(1, R - 1), a.spec.multiplier))
        plan.total += plan.unit(cfg.max_critique_tokens + cfg.max_answer_tokens, a.tps);
    if (a.spec.can_judge) plan.total += plan.unit(16, a.tps) * actors.size();
  }
  if (cfg.synthesise) {
    const Actor* judge = &actors.front();
    for (const Actor& a : actors)
      if (a.spec.can_synthesise && a.spec.cost_rank >= judge->spec.cost_rank) judge = &a;
    plan.total += plan.unit(cfg.max_final_tokens, judge->tps);
  }
  if (plan.total <= 0.0) plan.total = 1.0;

  auto publish = [&](void) {
    tr.progress = std::min(1.0, plan.done / plan.total);
    tr.seconds = Telemetry::now() - t_start;
    if (on_update) on_update(tr);
  };

  // Usage accounting per backend.
  std::map<std::string, DebateTranscript::Usage> usage;
  std::mutex usage_m;
  auto account = [&](const std::string& id, const GenResponse& g) {
    std::lock_guard<std::mutex> lk(usage_m);
    DebateTranscript::Usage& u = usage[id];
    u.backend_id = id;
    u.calls += 1;
    u.prompt_tokens += g.prompt_tokens;
    u.reused_tokens += g.reused_tokens;
    u.gen_tokens += g.gen_tokens;
    u.seconds += g.seconds;
  };

  const std::string ctx_block =
      cfg.context.empty()
          ? std::string()
          : "Reference material you may rely on:\n" + cfg.context + "\n";

  auto system_for = [&](const Actor& a) {
    std::string s = cfg.system_prompt;
    if (!s.empty()) s += " ";
    s += a.spec.persona;
    return trim(s);
  };

  // ------------------------------------------------------- round 0: divergence
  {
    DebateRound round;
    round.index = 0;
    round.kind = "draft";
    const double r0 = Telemetry::now();

    // Group by backend so one thread drives one runtime: two threads inside one
    // llama_context would be a data race, two threads on different hardware is
    // the parallelism we actually want.
    std::map<std::string, std::vector<size_t>> by_backend;
    for (size_t i = 0; i < actors.size(); ++i)
      by_backend[actors[i].be->id()].push_back(i);

    std::mutex out_m;
    std::vector<DebateAnswer> collected;

    auto run_group = [&](const std::vector<size_t>& idx) {
      for (size_t ai : idx) {
        Actor& a = actors[ai];
        std::vector<GenRequest> reqs;
        for (int d = 0; d < a.drafts; ++d) {
          GenRequest q;
          // Only the first draft may reuse the session; the others must be
          // independent samples, so they run stateless.
          q.slot = d == 0 ? a.slot : -1;
          q.sp = SamplingParams::creative(cfg.seed + 1000ull * (ai + 1) + d);
          q.sp.max_new_tokens = cfg.max_answer_tokens;
          q.sp.stop = {"<|user|>", "\n<|", kAnswerMarker};
          q.tag = static_cast<int>(d);
          std::vector<std::pair<std::string, std::string>> turns;
          turns.emplace_back(ctx_block + question, std::string());
          q.prompt = a.be->format_chat(system_for(a), turns, true);
          reqs.push_back(std::move(q));
        }
        std::lock_guard<std::mutex> hw(a.be->lock());
        const std::vector<GenResponse> res =
            reqs.size() > 1 && a.be->caps().batched
                ? a.be->generate_many(reqs, nullptr, cancel)
                : a.be->generate_many(reqs, nullptr, cancel);
        for (size_t k = 0; k < res.size(); ++k) {
          DebateAnswer ans;
          ans.participant = a.spec.name;
          ans.backend_id = a.be->id();
          ans.draft = static_cast<int>(k);
          ans.round = 0;
          ans.text = trim(res[k].text);
          ans.fluency = res[k].mean_logprob;
          ans.vote_weight = a.weight;
          ans.seconds = res[k].seconds;
          ans.gen_tokens = res[k].gen_tokens;
          ans.reused_tokens = res[k].reused_tokens;
          account(a.be->id(), res[k]);
          if (ans.text.empty()) continue;
          std::lock_guard<std::mutex> lk(out_m);
          collected.push_back(std::move(ans));
        }
        {
          std::lock_guard<std::mutex> lk(out_m);
          plan.done += plan.unit(cfg.max_answer_tokens * a.drafts, a.tps);
        }
        publish();
      }
    };

    if (cfg.parallel_first_round && by_backend.size() > 1) {
      std::vector<std::thread> th;
      for (const auto& kv : by_backend)
        th.emplace_back([&run_group, idx = kv.second] { run_group(idx); });
      for (std::thread& t : th) t.join();
    } else {
      for (const auto& kv : by_backend) run_group(kv.second);
    }
    // Deterministic order regardless of which thread finished first.
    std::sort(collected.begin(), collected.end(),
              [](const DebateAnswer& x, const DebateAnswer& y) {
                if (x.participant != y.participant) return x.participant < y.participant;
                return x.draft < y.draft;
              });
    round.answers = std::move(collected);
    round.seconds = Telemetry::now() - r0;
    tr.rounds.push_back(std::move(round));
    publish();
  }
  if (tr.rounds.front().answers.empty()) {
    tr.error = "every participant returned an empty answer";
    for (Actor& a : actors) a.be->close_session(a.slot);
    tr.seconds = Telemetry::now() - t_start;
    return tr;
  }

  // ------------------------------------------------- scoring of a round's answers
  auto score_round = [&](DebateRound* round, bool allow_judging) {
    std::vector<std::string> texts;
    for (const DebateAnswer& a : round->answers) texts.push_back(a.text);
    const std::vector<int> cl = cluster_answers(texts, cfg.cluster_threshold);
    std::map<int, double> mass;
    double total_w = 0.0;
    for (size_t i = 0; i < round->answers.size(); ++i) {
      mass[cl[i]] += round->answers[i].vote_weight;
      total_w += round->answers[i].vote_weight;
    }
    if (total_w <= 0.0) total_w = 1.0;
    double lo = 1e9, hi = -1e9;
    for (const DebateAnswer& a : round->answers) {
      lo = std::min(lo, a.fluency);
      hi = std::max(hi, a.fluency);
    }

    // Judging: every judge-capable actor scores every answer that is not its
    // own.  Cheap for SPT (16 tokens per verdict), and it is the only signal
    // that is not just "how many models said the same thing".
    if (allow_judging) {
      for (Actor& a : actors) {
        if (!a.spec.can_judge) continue;
        if (cancel && cancel->load()) break;
        std::vector<GenRequest> reqs;
        std::vector<size_t> targets;
        for (size_t i = 0; i < round->answers.size(); ++i) {
          if (round->answers[i].participant == a.spec.name) continue;
          GenRequest q;
          q.slot = -1;
          q.sp = SamplingParams::decisive(cfg.seed + 7777ull + i);
          q.sp.max_new_tokens = 12;
          q.sp.stop = {"\n"};
          std::vector<std::pair<std::string, std::string>> turns;
          turns.emplace_back(
              "Question: " + question + "\n\nAnswer:\n" +
                  utf8_truncate(round->answers[i].text, 900) +
                  "\n\nRate this answer from 0 to 10 for correctness and "
                  "usefulness. Reply with exactly: SCORE: <number>",
              std::string());
          q.prompt = a.be->format_chat("You are a strict grader.", turns, true);
          q.tag = static_cast<int>(i);
          reqs.push_back(std::move(q));
          targets.push_back(i);
        }
        if (reqs.empty()) continue;
        std::vector<GenResponse> res;
        {
          std::lock_guard<std::mutex> hw(a.be->lock());
          res = a.be->generate_many(reqs, nullptr, cancel);
        }
        for (size_t k = 0; k < res.size() && k < targets.size(); ++k) {
          account(a.be->id(), res[k]);
          double sc = 0.0;
          if (parse_judge_score(res[k].text, &sc)) {
            DebateAnswer& t = round->answers[targets[k]];
            t.judge_score = (t.judge_score * t.judge_votes + sc) / (t.judge_votes + 1);
            t.judge_votes += 1;
          }
        }
        plan.done += plan.unit(16, a.tps) * static_cast<double>(reqs.size());
        publish();
      }
    }

    for (size_t i = 0; i < round->answers.size(); ++i) {
      DebateAnswer& a = round->answers[i];
      a.cluster = cl[i];
      a.cluster_mass = mass[cl[i]] / total_w;
      const double judge = a.judge_votes ? a.judge_score / 10.0 : 0.5;
      const double flu = normalise_fluency(a.fluency, lo, hi);
      a.score = cfg.w_agreement * a.cluster_mass + cfg.w_judge * judge +
                cfg.w_fluency * flu;
    }
    std::sort(round->answers.begin(), round->answers.end(),
              [](const DebateAnswer& x, const DebateAnswer& y) {
                return x.score > y.score;
              });
    return mass;
  };

  std::map<int, double> mass = score_round(&tr.rounds.back(), true);
  publish();

  // Weighted margin between the best and the runner-up cluster.  This is the
  // cascade's decision variable: wide margin means the cheap models already
  // agree and the expensive one has nothing to add.
  auto cluster_margin = [](const std::map<int, double>& m) {
    if (m.size() < 2) return 1.0;
    std::vector<double> v;
    for (const auto& kv : m) v.push_back(kv.second);
    std::sort(v.rbegin(), v.rend());
    double sum = 0.0;
    for (double x : v) sum += x;
    if (sum <= 0.0) return 1.0;
    return (v[0] - v[1]) / sum;
  };

  // ------------------------------------------------- rounds 1..R-1: critique
  for (int r = 1; r < R; ++r) {
    if (cancel && cancel->load()) {
      tr.cancelled = true;
      break;
    }
    const double margin = cluster_margin(mass);
    if (cfg.cascade && margin >= cfg.escalate_margin && r > 1) {
      char note[160];
      std::snprintf(note, sizeof(note),
                    "stopped after round %d: cluster margin %.2f >= %.2f, further "
                    "debate would not change the winner",
                    r - 1, margin, static_cast<double>(cfg.escalate_margin));
      tr.decision_log += std::string(note) + "\n";
      break;
    }

    DebateRound round;
    round.index = r;
    round.kind = "critique";
    const double rt0 = Telemetry::now();

    // Who engages this round.  A participant whose multiplier is lower than the
    // round count sits some rounds out - that is the cost saving that makes a
    // 1x 7B affordable next to a 3x SPT.
    std::vector<size_t> active;
    for (size_t i = 0; i < actors.size(); ++i)
      if (engages(r - 1, std::max(1, R - 1), actors[i].spec.multiplier))
        active.push_back(i);
    if (active.empty()) break;

    const DebateRound& prev = tr.rounds.back();
    bool used_expensive = false;
    for (size_t ai : active) {
      Actor& a = actors[ai];
      if (cancel && cancel->load()) break;
      // The peers this actor should react to: the strongest answers that are not
      // its own, capped so the prompt stays inside a small context window.
      std::vector<const DebateAnswer*> peers;
      for (const DebateAnswer& x : prev.answers) {
        if (x.participant == a.spec.name) continue;
        peers.push_back(&x);
        if (peers.size() >= 3) break;
      }
      std::string own;
      for (const DebateAnswer& x : prev.answers)
        if (x.participant == a.spec.name) {
          own = x.text;
          break;
        }
      if (peers.empty()) continue;

      std::string ask = ctx_block + "Question: " + question + "\n\n";
      if (!own.empty())
        ask += "Your previous answer:\n" + utf8_truncate(own, cfg.peer_digest_chars) +
               "\n\n";
      ask += "Other answers:\n" + digest_of(peers, cfg.peer_digest_chars) + "\n";
      ask +=
          "First list the concrete mistakes or gaps in the other answers, at most "
          "three short bullets. Then write your improved answer on a new line "
          "starting with " +
          std::string(kAnswerMarker);

      GenRequest q;
      q.slot = a.slot;  // the KV cache from this actor's previous turns is reused
      q.sp = SamplingParams::critical(cfg.seed + 31337ull * r + ai);
      q.sp.max_new_tokens = cfg.max_critique_tokens + cfg.max_answer_tokens;
      q.sp.stop = {"<|user|>"};
      std::vector<std::pair<std::string, std::string>> turns;
      turns.emplace_back(ask, std::string());
      q.prompt = a.be->format_chat(system_for(a), turns, true);

      GenResponse res;
      {
        std::lock_guard<std::mutex> hw(a.be->lock());
        res = a.be->generate(q, nullptr, cancel);
      }
      account(a.be->id(), res);
      if (a.spec.cost_rank > 0.5f) used_expensive = true;

      DebateAnswer ans;
      ans.participant = a.spec.name;
      ans.backend_id = a.be->id();
      ans.round = r;
      ans.vote_weight = a.weight;
      ans.fluency = res.mean_logprob;
      ans.seconds = res.seconds;
      ans.gen_tokens = res.gen_tokens;
      ans.reused_tokens = res.reused_tokens;
      split_reply(res.text, &ans.critique, &ans.text);
      if (ans.text.empty()) ans.text = own;
      if (!ans.text.empty()) round.answers.push_back(std::move(ans));

      plan.done += plan.unit(cfg.max_critique_tokens + cfg.max_answer_tokens, a.tps);
      publish();
    }
    if (round.answers.empty()) break;

    // Carry forward the best previous answer of any participant that sat this
    // round out, so a 1x participant's vote is not silently deleted.
    for (const DebateAnswer& x : prev.answers) {
      bool present = false;
      for (const DebateAnswer& y : round.answers)
        if (y.participant == x.participant) present = true;
      if (present) continue;
      bool active_now = false;
      for (size_t ai : active)
        if (actors[ai].spec.name == x.participant) active_now = true;
      if (active_now) continue;
      DebateAnswer carried = x;
      carried.round = r;
      carried.critique.clear();
      round.answers.push_back(std::move(carried));
    }

    if (used_expensive) tr.escalated = true;
    round.seconds = Telemetry::now() - rt0;
    round.note = "round " + std::to_string(r) + ": " +
                 std::to_string(active.size()) + " of " +
                 std::to_string(actors.size()) + " participants engaged";
    tr.rounds.push_back(std::move(round));
    mass = score_round(&tr.rounds.back(), true);
    publish();
  }

  // Mark everything but the last round as superseded so best() looks at the
  // final state of the debate.
  for (size_t i = 0; i + 1 < tr.rounds.size(); ++i)
    for (DebateAnswer& a : tr.rounds[i].answers) a.superseded = true;

  // ----------------------------------------------------------- synthesis
  const DebateRound& last = tr.rounds.back();
  const DebateAnswer* winner = last.answers.empty() ? nullptr : &last.answers.front();
  if (winner) {
    tr.final_answer = winner->text;
    tr.synthesised_by = winner->participant;
    char note[256];
    std::snprintf(note, sizeof(note),
                  "winner %s: agreement %.2f, judge %.1f/10 (%d votes), score %.3f\n",
                  winner->participant.c_str(), winner->cluster_mass,
                  winner->judge_score, winner->judge_votes, winner->score);
    tr.decision_log += note;
  }

  const bool worth_synthesising =
      cfg.synthesise && !(cancel && cancel->load()) && last.answers.size() > 1;
  if (worth_synthesising) {
    // Mixture-of-Agents aggregator: the strongest model that is allowed to
    // synthesise sees the top answers and the critiques and writes one reply.
    Actor* agg = nullptr;
    for (Actor& a : actors) {
      if (!a.spec.can_synthesise) continue;
      if (!agg || a.spec.cost_rank > agg->spec.cost_rank) agg = &a;
    }
    const double margin = cluster_margin(mass);
    const bool skip_expensive =
        cfg.cascade && agg && agg->spec.cost_rank > 0.5f && margin >= cfg.escalate_margin;
    if (skip_expensive) {
      char note[200];
      std::snprintf(note, sizeof(note),
                    "synthesis skipped the expensive model: margin %.2f >= %.2f\n",
                    margin, static_cast<double>(cfg.escalate_margin));
      tr.decision_log += note;
      for (Actor& a : actors)
        if (a.spec.can_synthesise && a.spec.cost_rank <= 0.5f) {
          agg = &a;
          break;
        }
    }
    if (agg) {
      std::string ask = ctx_block + "Question: " + question + "\n\nCandidate answers:\n";
      char label = 'A';
      size_t shown = 0;
      for (const DebateAnswer& x : last.answers) {
        if (shown >= 3) break;
        ask += std::string("[") + label + "] (agreement " +
               std::to_string(static_cast<int>(x.cluster_mass * 100)) + "%, judge " +
               std::to_string(static_cast<int>(x.judge_score)) + "/10) " +
               utf8_truncate(x.text, cfg.peer_digest_chars) + "\n";
        if (!x.critique.empty())
          ask += "    critique from " + x.participant + ": " +
                 first_line(x.critique, 200) + "\n";
        ++label;
        ++shown;
      }
      ask +=
          "\nWrite the single best final answer to the question. Take the correct "
          "parts of the candidates, drop the wrong ones, and do not mention the "
          "candidates or this process.";

      GenRequest q;
      q.slot = agg->slot;
      q.sp = SamplingParams::decisive(cfg.seed + 999983ull);
      q.sp.max_new_tokens = cfg.max_final_tokens;
      q.sp.stop = {"<|user|>"};
      std::vector<std::pair<std::string, std::string>> turns;
      turns.emplace_back(ask, std::string());
      q.prompt = agg->be->format_chat(
          "You produce one final answer by combining several drafts.", turns, true);
      GenResponse res;
      {
        std::lock_guard<std::mutex> hw(agg->be->lock());
        res = agg->be->generate(q, nullptr, cancel);
      }
      account(agg->be->id(), res);
      plan.done += plan.unit(cfg.max_final_tokens, agg->tps);
      if (!trim(res.text).empty()) {
        DebateRound sr;
        sr.index = static_cast<int>(tr.rounds.size());
        sr.kind = "synthesis";
        DebateAnswer fa;
        fa.participant = agg->spec.name;
        fa.backend_id = agg->be->id();
        fa.round = sr.index;
        fa.text = trim(res.text);
        fa.fluency = res.mean_logprob;
        fa.seconds = res.seconds;
        fa.gen_tokens = res.gen_tokens;
        fa.reused_tokens = res.reused_tokens;
        fa.vote_weight = agg->weight;
        fa.score = 1.0;
        sr.answers.push_back(fa);
        sr.seconds = res.seconds;
        tr.rounds.push_back(std::move(sr));
        tr.final_answer = fa.text;
        tr.synthesised_by = agg->spec.name + " (synthesis)";
        if (agg->spec.cost_rank > 0.5f) tr.escalated = true;
      }
      publish();
    }
  }

  for (Actor& a : actors) a.be->close_session(a.slot);
  for (const auto& kv : usage) tr.usage.push_back(kv.second);
  plan.done = plan.total;
  tr.progress = 1.0;
  tr.seconds = Telemetry::now() - t_start;
  if (on_update) on_update(tr);

  if (tel_) {
    std::vector<std::pair<std::string, std::string>> fields = {
        {"rounds", std::to_string(tr.rounds.size())},
        {"escalated", tr.escalated ? "true" : "false"},
        {"seconds", std::to_string(tr.seconds)},
        {"winner", tr.synthesised_by}};
    for (const DebateTranscript::Usage& u : tr.usage)
      fields.emplace_back(u.backend_id,
                          std::to_string(u.gen_tokens) + " tok / " +
                              std::to_string(u.reused_tokens) + " reused / " +
                              std::to_string(u.seconds) + " s");
    tel_->log("info", "debate", tr.summary(), fields);
  }
  return tr;
}

}  // namespace slm
