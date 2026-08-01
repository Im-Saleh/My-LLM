// SPDX-License-Identifier: Apache-2.0
//
// AgentRuntime - assembles the pieces into the thing the user actually talks to.
//
//   backends   SPT (live or static) and optionally a GGUF model
//   tools      web search, page fetch, shell, files, codebase search
//   codebase   the local index, used both as a tool and as automatic context
//   debate     the multi-round weighted debate over whichever backends exist
//
// Everything above IModelBackend is shared: the tool loop, the retrieval and the
// debate do not know which runtime answered, which is the property that lets a
// 30 M model and a 7 B model be used interchangeably - and against each other.
//
// The tool loop is the standard reason-act cycle: generate, look for tool calls
// in the reply, run the allowed ones, append their output, generate again, up to
// a step limit.  Small models emit malformed calls often, so a broken call gets
// one corrective nudge before the loop gives up and returns the prose.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "agent/codebase.h"
#include "agent/tools.h"
#include "backend/backend_gguf.h"
#include "backend/backend_spt.h"
#include "backend/model_backend.h"
#include "debate/debate.h"

namespace slm {

class Telemetry;

enum class AskMode {
  kFast = 0,     // SPT only
  kStrong,       // the GGUF model only
  kDebate,       // both, weighted
  kSelfDebate,   // several voices of the same model
};
const char* ask_mode_name(AskMode m);

// ------------------------------------------------------------ the tool router
// Whether a question needs something the weights cannot contain.
//
// This is deterministic and lives outside the model because asking the model to
// emit tool-call syntax does not work at these sizes: SPT-30M never produces a
// well-formed call, and OLMo sometimes writes plausible *fake* search results
// inside the call block instead of making it.  The decision is taken here from
// the question, and the tool then runs whether or not the model would have asked.
//
// Public because it decides what the agent does before it thinks, which makes it
// the part most worth testing directly.
struct ToolPlan {
  bool search = false;     // run a web search
  bool fetch = false;      // fetch `url` directly
  std::string url;         // explicit URL found in the question
  bool codebase = false;   // retrieve from the indexed repository
  std::string reason;      // shown in the UI under "thinking"
};
ToolPlan plan_tools(const std::string& question, bool have_index, bool web_ok);

struct AgentRuntimeOptions {
  // SPT: exactly one of these three.
  std::string spt_ckpt;                 // .slm
  std::string spt_quant;                // .slmq (preferred for pure inference)
  WeightSource spt_live;                // live coordinator weights
  GPTConfig spt_cfg;                    // required with spt_live
  std::string tokenizer;
  const Tokenizer* tok = nullptr;

  // The strong model (optional).
  std::string gguf;
  int gguf_ctx = 4096;
  int gguf_threads = 0;
  int gguf_gpu_layers = 0;
  bool gguf_kv_q8 = false;
  bool gguf_base = false;

  std::string workspace = ".";          // the only tree the tools may touch
  std::string workdir = "runs";
  bool enable_web = true;
  bool enable_shell = true;
  bool enable_codebase = true;
  bool index_workspace = false;         // scan the workspace at startup
  std::string index_cache;              // where to persist the index
  Telemetry* tel = nullptr;
};

struct AskRequest {
  std::string question;
  AskMode mode = AskMode::kFast;
  int fast_multiplier = 2;
  int strong_multiplier = 1;
  int voices = 2;                       // kSelfDebate
  bool use_tools = true;
  bool use_codebase = true;             // inject retrieved context automatically
  // Run the retrieval tools from a deterministic plan instead of waiting for the
  // model to emit tool-call syntax.  A 32M model never emits it correctly and a
  // 7B model sometimes hallucinates the *results* instead of the call, so on by
  // default: it is the only way tools work with a small model.
  bool auto_tools = true;
  int max_tool_steps = 4;
  int max_tokens = 320;
  uint64_t seed = 0;
  std::string system_prompt;
};

struct ToolTrace {
  std::string tool;
  std::string args;
  bool ok = false;
  bool denied = false;
  double seconds = 0.0;
  std::string output;   // truncated for display
  // What actually happened, in one human sentence: the query that was searched,
  // the host that was read, the files that were retrieved.  The GUI shows this;
  // "web_search query=..." is not what someone wants to read.
  std::string detail;
  std::vector<std::string> sources;   // URLs or file paths this call used
};

struct AskResult {
  std::string answer;
  std::string context_used;             // what retrieval injected, for inspection
  std::string thinking;                 // reasoning/critique, collapsed in the UI
  std::vector<std::string> sources;     // every URL and file the answer rests on
  std::vector<ToolTrace> tools;
  DebateTranscript debate;              // populated in the two debate modes
  bool was_debate = false;
  double seconds = 0.0;
  int prompt_tokens = 0, gen_tokens = 0, reused_tokens = 0;
  bool cancelled = false;
  std::string error;
};

// Streaming: partial text, and a snapshot of the debate as it evolves.
struct AskObserver {
  std::function<void(const std::string& piece)> on_text;
  std::function<void(const DebateTranscript&)> on_debate;
  std::function<void(const ToolTrace&)> on_tool;
  // Live "what am I doing right now", e.g. "searching the web: cmake latest
  // version" or "reading src/qmodel.cpp:417-445".  Called from the worker thread.
  std::function<void(const std::string& status)> on_status;
};

class AgentRuntime {
 public:
  AgentRuntime();
  ~AgentRuntime();

  bool init(const AgentRuntimeOptions& o, std::string* err);
  const AgentRuntimeOptions& options() const;

  BackendRegistry& backends();
  ToolRegistry& tools();
  ApprovalGate& gate();
  ToolPolicy& policy();
  CodebaseIndex& codebase();
  DebateEngine& debate();

  // Which modes are usable right now (kStrong needs a loaded GGUF).
  bool mode_available(AskMode m) const;
  std::string spt_id() const;
  std::string strong_id() const;

  // Loads the GGUF model on demand; safe to call repeatedly.
  bool load_strong(std::string* err);
  void unload_strong();

  // Indexes `root` (default: the workspace).  Blocking; report progress through
  // the callback.  Safe to call while the rest of the runtime is in use.
  bool index_codebase(const std::string& root,
                      const std::function<void(int64_t, int64_t, const std::string&)>& progress,
                      std::atomic<bool>* cancel, std::string* err);

  AskResult ask(const AskRequest& req, std::atomic<bool>* cancel,
                const AskObserver& obs = AskObserver());

  // One line per backend, for the GUI status strip.
  std::vector<std::string> status_lines() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace slm
