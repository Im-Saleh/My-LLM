// SPDX-License-Identifier: Apache-2.0
//
// Weighted multi-round debate.
//
// The question this file answers: given two models of very different cost and
// quality (a ~10-50 M SPT that runs at hundreds of tokens/s, and a 7 B GGUF that
// runs at a handful of tokens/s on CPU), how do you combine them so the answer
// is better than either alone *without* paying 7 B latency N times over?
//
// The pipeline, and where each piece comes from
// --------------------------------------------
//   round 0   divergent drafting
//             Every participant answers independently at high temperature.
//             Sampling several answers and then agreeing on one is
//             Self-Consistency (Wang et al., 2023, "Self-Consistency Improves
//             Chain of Thought Reasoning"); the multi-model version of the same
//             idea is Mixture-of-Agents (Wang et al., 2024) where several
//             "proposers" feed an "aggregator".
//
//   round 1..N-1   critique and revise
//             Each participant is shown a digest of the other answers and asked
//             to criticise them and then improve its own.  This is Multi-Agent
//             Debate (Du et al., 2023, "Improving Factuality and Reasoning in
//             Language Models through Multiagent Debate"); the empirical finding
//             there is that most of the gain arrives in the first two rounds,
//             which is why the default N is small and why N is tied to the
//             multiplier instead of being large by default.
//
//   selection   weighted agreement
//             Answers are clustered by similarity and each cluster accumulates
//             the vote weight of its members.  Cluster mass is the open-ended
//             analogue of majority voting over exact answers.
//
//   synthesis   the aggregator writes the final answer
//             Mixture-of-Agents style: the best available model sees the top
//             answers plus the critiques and produces one reply.
//
// The cost control, which is the actual engineering
// ------------------------------------------------
//   1. multiplier -> participation, not just weight.  With multipliers 2x (SPT)
//      and 1x (OLMo) the debate runs max(2,1) = 2 critique rounds, SPT engages
//      in both, OLMo in one.  A participant's vote weight always counts in full;
//      its *generation* count is capped by `draft_cap`, because three
//      independent 7 B drafts cost three times the seconds while three votes
//      cost nothing.
//
//   2. cascade escalation.  The expensive model is not consulted when the cheap
//      ones already agree: if the weighted margin between the best and the
//      runner-up cluster exceeds `escalate_margin`, the strong model is skipped
//      entirely.  Only ambiguous questions pay for it.  This is the LLM-cascade
//      pattern from FrugalGPT (Chen et al., 2023) applied inside one query, and
//      it is the same shape as speculative decoding: something cheap proposes,
//      something expensive only verifies, and only when verification matters.
//
//   3. KV cache retention.  Debate prompts grow by appending, so every round
//      shares a long prefix with the previous one.  Each participant keeps its
//      own session slot for the whole debate and the backend re-encodes only the
//      changed suffix (see model_backend.h).
//
//   4. parallelism.  Round 0 has no data dependencies, so participants that live
//      on different hardware run at the same time; backends that support batched
//      decoding get all their drafts in one decode loop.
//
// Two identical models can debate each other: give two participants the same
// backend id and different personas.  They then hold different session slots, so
// their KV caches, and therefore their reasoning traces, are genuinely
// independent - only the weights are shared.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "backend/model_backend.h"

namespace slm {

class Telemetry;

// --------------------------------------------------------------- participants
struct DebateParticipant {
  std::string backend_id;       // key into the BackendRegistry
  std::string name;             // label in the transcript ("SPT-A")
  std::string persona;          // injected as a system line; decorrelates clones
  int multiplier = 1;           // rounds of engagement AND vote weight
  int draft_cap = 0;            // max drafts in round 0 (0 = multiplier)
  float vote_weight = 0.0f;     // 0 = use multiplier
  bool can_judge = true;        // may score other answers
  bool can_synthesise = true;   // may write the final answer
  float cost_rank = 0.0f;       // 0 = cheap/fast, 1 = expensive/strong
};

struct DebateConfig {
  std::vector<DebateParticipant> participants;
  int rounds = 0;               // 0 = max(multiplier); critique rounds after round 0
  int max_answer_tokens = 320;
  int max_critique_tokens = 200;
  int max_final_tokens = 420;
  int peer_digest_chars = 700;  // per peer answer shown during critique
  uint64_t seed = 1234;

  bool parallel_first_round = true;
  bool cascade = true;          // skip the expensive model when the cheap agree
  float escalate_margin = 0.15f;// weighted cluster-mass gap that ends the debate
  bool synthesise = true;       // false = pick the winning answer verbatim
  float cluster_threshold = 0.55f;  // similarity above which answers are "the same"

  // Score weights; normalised internally.
  float w_agreement = 0.45f;    // cluster mass (self-consistency)
  float w_judge = 0.35f;        // rubric score from the judges
  float w_fluency = 0.20f;      // mean logprob under the answering model

  std::string system_prompt;    // shared framing, may be empty
  std::string context;          // retrieved documents / codebase excerpts

  static DebateConfig two_model(const std::string& fast_id, int fast_mult,
                                const std::string& strong_id, int strong_mult);
  static DebateConfig self_debate(const std::string& backend_id, int voices,
                                  int multiplier);
  int planned_rounds() const;
  void normalise();             // fills defaults, clamps, resolves vote weights
};

// ------------------------------------------------------------------ transcript
struct DebateAnswer {
  std::string participant;      // DebateParticipant::name
  std::string backend_id;
  int draft = 0;                // which independent draft of that participant
  int round = 0;
  std::string text;
  std::string critique;         // what this participant said about the others
  double fluency = 0.0;         // mean logprob of `text` under its own model
  double judge_score = 0.0;     // 0..10, averaged over judges
  int judge_votes = 0;
  double cluster_mass = 0.0;    // weighted share of the agreement cluster
  int cluster = -1;
  double vote_weight = 1.0;
  double score = 0.0;           // final combined score
  double seconds = 0.0;
  int gen_tokens = 0;
  int reused_tokens = 0;
  bool superseded = false;      // revised in a later round
};

struct DebateRound {
  int index = 0;
  std::string kind;             // "draft" | "critique" | "synthesis"
  std::vector<DebateAnswer> answers;
  std::string note;             // e.g. "escalated to olmo: margin 0.04 < 0.15"
  double seconds = 0.0;
};

struct DebateTranscript {
  std::string question;
  std::vector<DebateRound> rounds;
  std::string final_answer;
  std::string synthesised_by;
  std::string decision_log;     // why this answer won, in one paragraph
  double seconds = 0.0;
  double progress = 0.0;        // 0..1, real work done / planned work
  bool cancelled = false;
  bool escalated = false;       // the expensive model was actually used
  std::string error;

  struct Usage {
    std::string backend_id;
    int calls = 0;
    int prompt_tokens = 0;
    int reused_tokens = 0;
    int gen_tokens = 0;
    double seconds = 0.0;
  };
  std::vector<Usage> usage;
  const DebateAnswer* best() const;
  std::string summary() const;  // one line for the log
};

// Called with a snapshot after every finished generation so the GUI can render
// the debate as it happens.  Must not block for long.
using DebateObserver = std::function<void(const DebateTranscript&)>;

class DebateEngine {
 public:
  DebateEngine(BackendRegistry* reg, Telemetry* tel);

  DebateTranscript run(const std::string& question, const DebateConfig& cfg,
                       std::atomic<bool>* cancel, const DebateObserver& on_update);

  // Exposed because they are independently useful and independently testable.
  // Order-insensitive token-overlap similarity in [0,1]; F1 over the multiset of
  // normalised tokens, which is robust to reordering and to length differences
  // in a way that raw edit distance is not.
  static double answer_similarity(const std::string& a, const std::string& b);
  // Greedy single-link clustering at `threshold`; returns cluster id per answer.
  static std::vector<int> cluster_answers(const std::vector<std::string>& texts,
                                          double threshold);
  // Parses "SCORE: 7.5" / "7/10" / a bare number out of a judge reply.
  static bool parse_judge_score(const std::string& reply, double* score);

 private:
  struct Plan;
  BackendRegistry* reg_;
  Telemetry* tel_;
};

}  // namespace slm
