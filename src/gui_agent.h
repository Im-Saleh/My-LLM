// SPDX-License-Identifier: Apache-2.0
//
// AgentController - everything the dashboard needs from the agent, made safe to
// call from the render thread.
//
// The GUI must never block: a debate round with a 7 B model on CPU takes tens of
// seconds, and indexing a large repository or streaming a dataset into the
// trainers takes longer than that.  So every long operation runs on its own
// worker thread and the GUI only ever reads an immutable snapshot taken under a
// mutex, and writes requests.  Cancellation is an atomic flag the workers poll,
// which is what makes the "cancel debate" button honest rather than cosmetic.
#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "agent/runtime.h"

namespace slm {

class InteractionHub;
class Telemetry;

struct AgentTurn {
  std::string question;
  std::string answer;
  AskMode mode = AskMode::kFast;
  DebateTranscript debate;
  std::vector<ToolTrace> tools;
  std::string context_used;
  double seconds = 0.0;
  int prompt_tokens = 0, gen_tokens = 0, reused_tokens = 0;
  bool was_debate = false;
  std::string error;
};

struct AgentSnapshot {
  bool busy = false;
  std::string question;         // the one in flight
  std::string partial;          // streamed text so far
  double progress = 0.0;        // debate progress, 0..1
  DebateTranscript live;        // the debate as it stands
  std::vector<AgentTurn> history;
  std::vector<std::string> status;   // one line per backend
  size_t pending_approvals = 0;
};

struct IndexJob {
  bool running = false;
  bool done = false;
  int64_t files_done = 0, files_total = 0;
  std::string current;
  std::string root;
  std::string error;
  double seconds = 0.0;
};

struct TrainJob {
  bool running = false;
  bool done = false;
  int64_t lines = 0, queued = 0, bytes = 0;
  std::string path;
  std::string error;
  double seconds = 0.0;
};

class AgentController {
 public:
  AgentController();
  ~AgentController();

  void attach(AgentRuntime* rt, InteractionHub* hub, Telemetry* tel);
  bool attached() const;
  AgentRuntime* runtime() const;

  // ---------------------------------------------------------------- asking
  void ask(const std::string& question, AskMode mode, int fast_mult, int strong_mult,
           int voices, bool use_tools, bool use_codebase, int max_tokens);
  void cancel();
  bool busy() const;
  AgentSnapshot snapshot() const;
  void clear_history();

  // ------------------------------------------------------------- codebase
  void start_index(const std::string& root);
  IndexJob index_job() const;

  // -------------------------------------------------------------- dataset
  // Streams a text file into the continual-learning thread.  This is the safe
  // way to attach data at run time: the trainers already consume the hub under a
  // mutex, so nothing else has to become thread safe for it to work, and the
  // coordinator's accept/reject gate still guards every resulting update.
  void teach_dataset(const std::string& path, int64_t max_lines, int64_t chunk_chars);
  TrainJob train_job() const;
  void stop_jobs();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace slm
