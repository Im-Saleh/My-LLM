// SPDX-License-Identifier: Apache-2.0
//
// Tests for the debate engine.
//
// The point of a mock backend here is that it makes the *algorithm* testable
// without a model: participation counts per round, vote weighting, cluster
// agreement, the cascade that skips the expensive participant, KV-cache prefix
// reuse, cancellation and progress monotonicity are all properties of the
// engine, not of any model's output quality.  A real model would make every one
// of these assertions flaky.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "backend/model_backend.h"
#include "debate/debate.h"

using namespace slm;

namespace {

int g_pass = 0, g_fail = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
  if (ok) {
    ++g_pass;
    std::printf("  ok   %s%s%s\n", what.c_str(), detail.empty() ? "" : "  ",
                detail.c_str());
  } else {
    ++g_fail;
    std::printf("  FAIL %s%s%s\n", what.c_str(), detail.empty() ? "" : "  ",
                detail.c_str());
  }
}

bool contains(const std::string& h, const std::string& n) {
  return h.find(n) != std::string::npos;
}

// --------------------------------------------------------------- mock backend
// Deterministic: the reply is a function of the prompt, so clustering and
// scoring are reproducible.  It also records everything the engine did to it.
class MockBackend : public IModelBackend {
 public:
  MockBackend(std::string id, std::string reply, double tps, bool judge_high)
      : id_(std::move(id)), reply_(std::move(reply)), tps_(tps), judge_high_(judge_high) {}

  std::string id() const override { return id_; }
  std::string display_name() const override { return id_; }
  std::string runtime() const override { return "mock"; }
  BackendCaps caps() const override {
    BackendCaps c;
    c.logprobs = true;
    c.batched = batched_;
    c.max_parallel = 8;
    return c;
  }
  BackendStatus status() const override {
    BackendStatus s;
    s.loaded = loaded_;
    s.last_decode_tps = tps_;
    s.calls = calls;
    return s;
  }
  bool load(std::string*) override {
    loaded_ = true;
    return true;
  }
  void unload() override { loaded_ = false; }
  bool loaded() const override { return loaded_; }
  int64_t context_limit() const override { return 4096; }

  // One token per whitespace-delimited word: enough for prefix arithmetic.
  std::vector<int32_t> tokenize(const std::string& t) const override {
    std::vector<int32_t> out;
    int32_t h = 0;
    for (char c : t) {
      if (c == ' ' || c == '\n') {
        if (h) out.push_back(h);
        h = 0;
      } else {
        h = h * 31 + static_cast<int32_t>(c);
      }
    }
    if (h) out.push_back(h);
    return out;
  }
  std::string detokenize(const std::vector<int32_t>&) const override { return ""; }
  std::string format_chat(const std::string& sys,
                          const std::vector<std::pair<std::string, std::string>>& turns,
                          bool) const override {
    std::string out = sys + "\n";
    for (const auto& t : turns) out += t.first;
    return out;
  }

  int open_session() override {
    const int s = next_slot_++;
    sessions_[s] = {};
    ++sessions_opened;
    return s;
  }
  void close_session(int s) override { sessions_.erase(s); }
  void reset_session(int s) override { sessions_[s].clear(); }
  int64_t session_tokens(int s) const override {
    auto it = sessions_.find(s);
    return it == sessions_.end() ? 0 : static_cast<int64_t>(it->second.size());
  }

  GenResponse generate(const GenRequest& req, const ChunkFn&,
                       std::atomic<bool>* cancel) override {
    GenResponse r;
    r.tag = req.tag;
    ++calls;
    prompts.push_back(req.prompt);
    slots_used.push_back(req.slot);
    if (cancel && cancel->load()) {
      r.cancelled = true;
      return r;
    }
    const std::vector<int32_t> ids = tokenize(req.prompt);
    r.prompt_tokens = static_cast<int>(ids.size());
    if (req.slot >= 0) {
      std::vector<int32_t>& hist = sessions_[req.slot];
      size_t common = 0;
      while (common < hist.size() && common < ids.size() && hist[common] == ids[common])
        ++common;
      r.reused_tokens = static_cast<int>(common);
      reused_total += static_cast<int64_t>(common);
      hist = ids;
    }
    // A grading request is recognised by the rubric line the engine sends.
    if (contains(req.prompt, "Rate this answer")) {
      ++judge_calls;
      // Bias so the two mocks disagree about quality unless told otherwise.
      r.text = judge_high_ ? "SCORE: 9" : "SCORE: 4";
    } else if (contains(req.prompt, "single best final answer")) {
      ++synth_calls;
      r.text = "FINAL from " + id_;
    } else if (contains(req.prompt, "Other answers:")) {
      ++critique_calls;
      r.text = "- one gap\nANSWER: " + reply_;
    } else {
      ++draft_calls;
      r.text = reply_;
    }
    r.gen_tokens = static_cast<int>(tokenize(r.text).size());
    r.mean_logprob = -1.0 - 0.1 * static_cast<double>(id_.size());
    r.seconds = 0.001;
    return r;
  }

  bool score(const std::string&, const std::string&, double* lp,
             std::string*) override {
    if (lp) *lp = -1.0;
    return true;
  }

  void set_batched(bool b) { batched_ = b; }

  // Observations the tests assert on.
  int64_t calls = 0, draft_calls = 0, critique_calls = 0, judge_calls = 0,
          synth_calls = 0, sessions_opened = 0, reused_total = 0;
  std::vector<std::string> prompts;
  std::vector<int> slots_used;

 private:
  std::string id_, reply_;
  double tps_;
  bool judge_high_;
  bool loaded_ = false;
  bool batched_ = false;
  std::map<int, std::vector<int32_t>> sessions_;
  int next_slot_ = 1;
};

std::shared_ptr<MockBackend> mk(const std::string& id, const std::string& reply,
                               double tps = 100.0, bool judge_high = true) {
  return std::make_shared<MockBackend>(id, reply, tps, judge_high);
}

// Counts how many answers a participant contributed in a given round kind.
int answers_by(const DebateTranscript& tr, const std::string& name, int round) {
  if (round < 0 || round >= static_cast<int>(tr.rounds.size())) return 0;
  int n = 0;
  for (const DebateAnswer& a : tr.rounds[static_cast<size_t>(round)].answers)
    if (a.participant == name) ++n;
  return n;
}

// ------------------------------------------------------------------- [1] utils
void test_similarity() {
  std::printf("[1] similarity, clustering, score parsing\n");
  check(DebateEngine::answer_similarity("the cat sat", "the cat sat") > 0.999,
        "identical strings score 1");
  check(DebateEngine::answer_similarity("the cat sat", "a dog ran") < 0.2,
        "disjoint strings score ~0");
  const double reordered =
      DebateEngine::answer_similarity("use a mutex to guard the queue",
                                      "guard the queue with a mutex");
  check(reordered > 0.7, "word order barely matters", std::to_string(reordered));
  check(DebateEngine::answer_similarity("", "") > 0.999, "two empties are equal");
  check(DebateEngine::answer_similarity("x", "") < 0.001, "empty vs non-empty is 0");
  const double partial =
      DebateEngine::answer_similarity("use a mutex", "use a mutex and also a condition "
                                                     "variable and a queue and a lock");
  check(partial > 0.2 && partial < 0.8, "length mismatch is penalised but not fatal",
        std::to_string(partial));
  // Persian must tokenise as words, not bytes.
  check(DebateEngine::answer_similarity("سلام دنیا", "سلام دنیا") > 0.999,
        "Persian text compares as words");
  check(DebateEngine::answer_similarity("سلام دنیا", "خداحافظ دنیا") < 0.7,
        "different Persian words differ");

  const std::vector<std::string> texts = {
      "use a mutex to guard the queue", "guard the queue with a mutex",
      "spawn one thread per core", "one thread per core should be spawned"};
  const std::vector<int> cl = DebateEngine::cluster_answers(texts, 0.55);
  check(cl.size() == 4 && cl[0] == cl[1] && cl[2] == cl[3] && cl[0] != cl[2],
        "clustering groups paraphrases and separates topics",
        std::to_string(cl[0]) + std::to_string(cl[1]) + std::to_string(cl[2]) +
            std::to_string(cl[3]));
  const std::vector<int> strict = DebateEngine::cluster_answers(texts, 0.99);
  check(strict[0] != strict[1], "a high threshold splits paraphrases");

  double sc = 0.0;
  check(DebateEngine::parse_judge_score("SCORE: 7.5", &sc) && std::abs(sc - 7.5) < 1e-9,
        "parses SCORE: 7.5");
  check(DebateEngine::parse_judge_score("score 8", &sc) && std::abs(sc - 8.0) < 1e-9,
        "parses lowercase score 8");
  check(DebateEngine::parse_judge_score("I would say 6/10 overall", &sc) &&
            std::abs(sc - 6.0) < 1e-9,
        "parses 6/10");
  check(DebateEngine::parse_judge_score("**9**", &sc) && std::abs(sc - 9.0) < 1e-9,
        "parses a decorated number");
  check(DebateEngine::parse_judge_score("0.8", &sc) && std::abs(sc - 8.0) < 1e-9,
        "rescales a 0..1 fraction");
  check(!DebateEngine::parse_judge_score("no number here", &sc),
        "reports failure when there is no score");
  check(!DebateEngine::parse_judge_score("SCORE: 42", &sc),
        "rejects an out-of-range score");
}

// ------------------------------------------------------------ [2] config rules
void test_config() {
  std::printf("[2] configuration\n");
  DebateConfig c = DebateConfig::two_model("spt", 3, "olmo", 1);
  check(c.planned_rounds() == 3, "rounds default to the largest multiplier",
        std::to_string(c.planned_rounds()));
  check(c.participants.size() == 2, "two participants");
  check(c.participants[0].vote_weight == 3.0f && c.participants[1].vote_weight == 1.0f,
        "vote weight defaults to the multiplier");
  check(c.participants[1].draft_cap == 1,
        "the expensive participant is capped to one draft");
  check(std::abs(c.w_agreement + c.w_judge + c.w_fluency - 1.0f) < 1e-6,
        "score weights are normalised");

  DebateConfig s = DebateConfig::self_debate("spt", 3, 2);
  check(s.participants.size() == 3, "self-debate creates the requested voices");
  check(s.participants[0].backend_id == s.participants[1].backend_id,
        "all voices share one backend");
  check(s.participants[0].persona != s.participants[1].persona,
        "but not one persona - identical prompts would collapse the debate");
  check(s.participants[0].name != s.participants[1].name, "names are distinct");

  DebateConfig w;
  w.participants.resize(1);
  w.participants[0].backend_id = "x";
  w.participants[0].multiplier = 99;
  w.normalise();
  check(w.participants[0].multiplier <= 8, "an absurd multiplier is clamped");
  check(w.rounds >= 1 && w.rounds <= 6, "rounds are clamped");
}

// --------------------------------------------------- [3] a full two-model debate
void test_two_model() {
  std::printf("[3] two-model debate: participation, weighting, sessions\n");
  BackendRegistry reg;
  auto fast = mk("spt", "use a mutex to guard the queue", 200.0, true);
  auto slow = mk("olmo", "guard the queue with a mutex", 5.0, true);
  reg.add(fast);
  reg.add(slow);
  DebateEngine eng(&reg, nullptr);

  DebateConfig cfg = DebateConfig::two_model("spt", 3, "olmo", 1);
  cfg.cascade = false;  // exercise the full schedule here
  std::atomic<bool> cancel(false);
  std::vector<double> progress;
  const DebateTranscript tr =
      eng.run("How do I make the queue thread safe?", cfg, &cancel,
              [&](const DebateTranscript& t) { progress.push_back(t.progress); });

  check(tr.error.empty(), "the debate completed", tr.error);
  check(!tr.final_answer.empty(), "a final answer was produced");
  check(tr.rounds.size() >= 3, "round 0 plus critique rounds plus synthesis",
        std::to_string(tr.rounds.size()) + " rounds");
  check(tr.rounds[0].kind == "draft", "round 0 is the drafting round");
  check(answers_by(tr, "SPT", 0) == 3,
        "the 3x participant produced three independent drafts",
        std::to_string(answers_by(tr, "SPT", 0)));
  check(answers_by(tr, "OLMo", 0) == 1,
        "the 1x expensive participant produced one draft");

  // Participation: SPT must engage in every critique round, OLMo in fewer.
  int spt_crit = 0, olmo_crit = 0;
  for (const DebateRound& r : tr.rounds) {
    if (r.kind != "critique") continue;
    for (const DebateAnswer& a : r.answers) {
      if (a.critique.empty()) continue;  // carried-forward answers do not count
      if (a.participant == "SPT") ++spt_crit;
      if (a.participant == "OLMo") ++olmo_crit;
    }
  }
  check(spt_crit > olmo_crit,
        "the 3x participant engages in more critique rounds than the 1x one",
        "SPT " + std::to_string(spt_crit) + " vs OLMo " + std::to_string(olmo_crit));
  check(olmo_crit >= 1, "but the 1x participant is not shut out entirely");

  // Sessions: one per participant, opened once, and reused across rounds.
  check(fast->sessions_opened == 1 && slow->sessions_opened == 1,
        "one session per participant for the whole debate");
  check(fast->reused_total > 0,
        "the KV cache prefix was reused across rounds",
        std::to_string(fast->reused_total) + " tokens reused");

  // Progress must be monotonic and finish at 1.
  bool mono = true;
  for (size_t i = 1; i < progress.size(); ++i)
    if (progress[i] + 1e-9 < progress[i - 1]) mono = false;
  check(mono, "progress never goes backwards");
  check(!progress.empty() && std::abs(progress.back() - 1.0) < 1e-9,
        "progress ends at exactly 1.0");
  check(tr.progress == 1.0 && tr.seconds > 0.0, "the transcript reports completion");

  // Usage accounting must add up.
  int64_t calls = 0;
  for (const DebateTranscript::Usage& u : tr.usage) calls += u.calls;
  check(calls == fast->calls + slow->calls, "usage accounting matches the backends",
        std::to_string(calls) + " vs " + std::to_string(fast->calls + slow->calls));
  check(tr.usage.size() == 2, "both backends appear in the usage table");

  // Weighted agreement: with two answers in the same cluster, mass is 1.
  bool masses_ok = true;
  for (const DebateRound& r : tr.rounds)
    for (const DebateAnswer& a : r.answers)
      if (a.cluster_mass < 0.0 || a.cluster_mass > 1.0001) masses_ok = false;
  check(masses_ok, "cluster masses are valid shares");
}

// ------------------------------------------------------------- [4] the cascade
void test_cascade() {
  std::printf("[4] cascade: the expensive model is skipped when the cheap agree\n");
  BackendRegistry reg;
  // Three cheap voices that all say the same thing -> one cluster -> margin 1.0.
  auto fast = mk("spt", "always use a mutex", 200.0, true);
  auto slow = mk("olmo", "something else entirely different", 5.0, true);
  reg.add(fast);
  reg.add(slow);
  DebateEngine eng(&reg, nullptr);

  DebateConfig cfg;
  DebateParticipant a;
  a.backend_id = "spt";
  a.name = "SPT";
  a.multiplier = 3;
  a.cost_rank = 0.0f;
  DebateParticipant b;
  b.backend_id = "olmo";
  b.name = "OLMo";
  b.multiplier = 1;
  b.draft_cap = 1;
  b.cost_rank = 1.0f;   // marks it as the expensive one
  cfg.participants = {a, b};
  cfg.cascade = true;
  cfg.escalate_margin = 0.15f;
  cfg.rounds = 3;
  std::atomic<bool> cancel(false);
  const DebateTranscript tr = eng.run("q", cfg, &cancel, nullptr);

  check(tr.error.empty(), "cascade debate completed", tr.error);
  check(contains(tr.decision_log, "margin") || !tr.escalated,
        "the decision log explains the cascade", tr.decision_log);
  const int64_t expensive_gen = slow->critique_calls + slow->synth_calls;
  std::printf("       olmo: %lld drafts, %lld critiques, %lld syntheses\n",
              static_cast<long long>(slow->draft_calls),
              static_cast<long long>(slow->critique_calls),
              static_cast<long long>(slow->synth_calls));
  check(expensive_gen <= 2,
        "the expensive model does not run in every round",
        std::to_string(expensive_gen) + " expensive generation calls");

  // With cascade off, the same configuration must do strictly more work.
  auto fast2 = mk("spt", "always use a mutex", 200.0, true);
  auto slow2 = mk("olmo", "something else entirely different", 5.0, true);
  BackendRegistry reg2;
  reg2.add(fast2);
  reg2.add(slow2);
  DebateEngine eng2(&reg2, nullptr);
  DebateConfig off = cfg;
  off.cascade = false;
  const DebateTranscript tr2 = eng2.run("q", off, &cancel, nullptr);
  check(tr2.rounds.size() >= tr.rounds.size(),
        "disabling the cascade runs at least as many rounds",
        std::to_string(tr2.rounds.size()) + " vs " + std::to_string(tr.rounds.size()));
  check(fast2->calls + slow2->calls >= fast->calls + slow->calls,
        "and costs at least as many calls",
        std::to_string(fast2->calls + slow2->calls) + " vs " +
            std::to_string(fast->calls + slow->calls));
}

// -------------------------------------------------- [5] two identical models
void test_self_debate() {
  std::printf("[5] self-debate: one backend, several independent voices\n");
  BackendRegistry reg;
  auto only = mk("spt", "the answer is forty two", 150.0, true);
  reg.add(only);
  DebateEngine eng(&reg, nullptr);

  DebateConfig cfg = DebateConfig::self_debate("spt", 2, 2);
  cfg.cascade = false;
  std::atomic<bool> cancel(false);
  const DebateTranscript tr = eng.run("what is the answer?", cfg, &cancel, nullptr);
  check(tr.error.empty(), "self-debate completed", tr.error);
  check(only->sessions_opened == 2,
        "two voices on one backend hold two independent sessions",
        std::to_string(only->sessions_opened) + " sessions");

  // The personas must actually reach the model, otherwise the voices are clones.
  bool p0 = false, p1 = false;
  for (const std::string& p : only->prompts) {
    if (contains(p, "most direct solution")) p0 = true;
    if (contains(p, "what the other answer got wrong")) p1 = true;
  }
  check(p0 && p1, "each voice was prompted with its own persona");
  check(tr.usage.size() == 1, "usage collapses onto the single backend");
}

// ---------------------------------------------------------- [6] cancellation
void test_cancel() {
  std::printf("[6] cancellation\n");
  BackendRegistry reg;
  auto be = mk("spt", "an answer", 100.0, true);
  reg.add(be);
  DebateEngine eng(&reg, nullptr);
  DebateConfig cfg = DebateConfig::self_debate("spt", 2, 3);

  std::atomic<bool> cancel(true);  // cancelled before it starts
  const DebateTranscript tr = eng.run("q", cfg, &cancel, nullptr);
  check(tr.rounds.size() <= 2, "a pre-cancelled debate stops almost immediately",
        std::to_string(tr.rounds.size()) + " rounds");

  // Cancel from another thread partway through.
  auto be2 = mk("spt", "an answer", 100.0, true);
  BackendRegistry reg2;
  reg2.add(be2);
  DebateEngine eng2(&reg2, nullptr);
  std::atomic<bool> c2(false);
  std::atomic<bool> seen(false);
  DebateTranscript out;
  std::thread th([&] {
    out = eng2.run("q", cfg, &c2, [&](const DebateTranscript&) { seen.store(true); });
  });
  while (!seen.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  c2.store(true);
  th.join();
  check(out.progress <= 1.0, "a mid-flight cancel leaves a consistent transcript",
        "progress " + std::to_string(out.progress));
  check(be2->calls > 0, "some work happened before the cancel");
}

// ------------------------------------------------------------ [7] degradation
void test_missing_backend() {
  std::printf("[7] missing and unloadable backends\n");
  BackendRegistry reg;
  auto ok = mk("spt", "an answer", 100.0, true);
  reg.add(ok);
  DebateEngine eng(&reg, nullptr);

  DebateConfig cfg = DebateConfig::two_model("spt", 2, "does-not-exist", 1);
  std::atomic<bool> cancel(false);
  const DebateTranscript tr = eng.run("q", cfg, &cancel, nullptr);
  check(tr.error.empty(), "the debate proceeds with the participants it has", tr.error);
  check(contains(tr.decision_log, "does-not-exist"),
        "the missing backend is reported in the decision log");
  check(!tr.final_answer.empty(), "and still yields an answer");

  BackendRegistry empty;
  DebateEngine eng2(&empty, nullptr);
  const DebateTranscript none = eng2.run("q", cfg, &cancel, nullptr);
  check(!none.error.empty(), "no usable participant is an error, not a crash",
        none.error);
}

}  // namespace

int main() {
  std::printf("debate engine tests\n\n");
  test_similarity();
  test_config();
  test_two_model();
  test_cascade();
  test_self_debate();
  test_cancel();
  test_missing_backend();
  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
