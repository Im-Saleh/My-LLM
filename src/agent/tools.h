// SPDX-License-Identifier: Apache-2.0
//
// The tool layer: one registry, model independent.
//
// Both backends see exactly the same tools, because the tools live above
// IModelBackend.  A tool call is text the model emits; the registry parses it,
// runs it if policy allows, and returns text to append to the conversation.
//
// Call syntax (all three accepted, because a 30 M model and a 7 B model are not
// equally good at structured output):
//
//   [[tool:web_search]]
//   query: persian nlp datasets
//   [[/tool]]
//
//   [[tool:shell cmd="ls -la src"]]
//
//   {"tool": "read_file", "path": "src/main.cpp", "lines": "1-40"}
//
// Safety model
// ------------
// Every tool declares a risk level.  `ToolPolicy` maps risk to one of
// auto-allow / ask / deny, and anything that is not auto-allowed goes through
// `ApprovalGate`, which blocks the agent thread until the user answers in the
// GUI (or the terminal) or the request times out.  Every decision and every
// invocation is written to the same JSONL audit log the trainers use, so the
// record of "what did it run and why" is one file.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace slm {

class Telemetry;
class CodebaseIndex;

enum class ToolRisk {
  kSafe = 0,     // pure computation / reads inside the workspace
  kNetwork,      // talks to the internet
  kWrite,        // modifies files
  kDangerous,    // arbitrary command execution
};
const char* tool_risk_name(ToolRisk r);

enum class ToolDecision { kAllow = 0, kDeny, kTimeout };

struct ToolParam {
  std::string name;
  std::string description;
  bool required = false;
  std::string def;
};

struct ToolSpec {
  std::string name;
  std::string summary;      // one line, shown to the model
  std::string usage;        // example call, shown to the model
  std::vector<ToolParam> params;
  ToolRisk risk = ToolRisk::kSafe;
  bool enabled = true;
};

struct ToolCall {
  std::string name;
  std::map<std::string, std::string> args;
  std::string raw;          // the exact text that produced this call
  uint64_t id = 0;
  std::string arg(const std::string& k, const std::string& def = "") const;
  int64_t arg_int(const std::string& k, int64_t def) const;
};

struct ToolResult {
  bool ok = false;
  std::string output;       // what gets appended to the conversation
  std::string error;
  std::string display;      // richer text for the GUI (may be empty)
  double seconds = 0.0;
  int64_t bytes_before_truncation = 0;
  bool truncated = false;
  bool denied = false;
};

// What a tool is allowed to touch.
struct ToolContext {
  std::string workspace;          // shell/read/write are confined to this root
  std::string workdir;            // scratch (logs, caches)
  Telemetry* tel = nullptr;
  CodebaseIndex* codebase = nullptr;
  int64_t output_budget = 8000;   // characters returned to the model
  std::atomic<bool>* cancel = nullptr;
};

class Tool {
 public:
  virtual ~Tool() = default;
  virtual ToolSpec spec() const = 0;
  virtual ToolResult run(const ToolCall& call, ToolContext& ctx) = 0;
  // Short human-readable description of what this call will do, shown in the
  // approval dialog.  Must not have side effects.
  virtual std::string preview(const ToolCall& call) const;
};
using ToolPtr = std::shared_ptr<Tool>;

// ------------------------------------------------------------- approval gate
struct PendingApproval {
  uint64_t id = 0;
  std::string tool;
  std::string preview;
  std::string detail;
  ToolRisk risk = ToolRisk::kSafe;
  double requested_at = 0.0;
  double timeout_s = 0.0;
};

struct ToolPolicy {
  // Per risk level: 0 = ask, 1 = auto-allow, 2 = deny outright.
  int safe = 1;
  int network = 1;
  int write = 0;
  int dangerous = 0;
  double timeout_s = 120.0;
  bool remember_allowed = true;   // "allow this exact command again"
  int mode_for(ToolRisk r) const;
};

class ApprovalGate {
 public:
  // Agent side: blocks until the user decides or the request times out.
  ToolDecision request(const ToolCall& call, const std::string& preview,
                       ToolRisk risk, const ToolPolicy& policy);

  // UI side.
  std::vector<PendingApproval> pending() const;
  void decide(uint64_t id, bool allow, bool remember);
  void deny_all();
  size_t pending_count() const;

  // Approvals remembered by "tool\0canonical-args".
  void forget_all();
  size_t remembered() const;

 private:
  struct Waiter {
    PendingApproval info;
    ToolDecision decision = ToolDecision::kTimeout;
    bool answered = false;
  };
  mutable std::mutex m_;
  std::condition_variable cv_;
  std::map<uint64_t, Waiter> waiting_;
  std::vector<std::string> remembered_;
  uint64_t next_id_ = 1;
};

// ------------------------------------------------------------------ registry
class ToolRegistry {
 public:
  void add(ToolPtr t);
  std::vector<ToolSpec> specs() const;
  bool set_enabled(const std::string& name, bool on);
  bool enabled(const std::string& name) const;
  ToolPtr get(const std::string& name) const;

  // The block of text describing the tools, injected into the system prompt.
  std::string catalogue(bool compact) const;

  ToolResult invoke(const ToolCall& call, ToolContext& ctx, ApprovalGate* gate,
                    const ToolPolicy& policy);

  // Tolerant parser: finds every tool call in a model reply, in any of the three
  // supported syntaxes, and reports where each call started so the caller can
  // keep the prose that preceded it.
  static std::vector<ToolCall> parse(const std::string& reply);
  // True when the reply looks like it *wanted* to call a tool but got the syntax
  // wrong; used to send a corrective nudge instead of giving up.
  static bool looks_like_broken_call(const std::string& reply);

  struct Stat {
    std::string name;
    int64_t calls = 0, failures = 0, denials = 0;
    double seconds = 0.0;
  };
  std::vector<Stat> stats() const;

 private:
  mutable std::mutex m_;
  std::vector<ToolPtr> tools_;
  std::map<std::string, bool> enabled_;
  std::map<std::string, Stat> stats_;
};

// ------------------------------------------------------------------ factories
// Registers the standard set.  `web` needs HttpClient::available().
void register_builtin_tools(ToolRegistry* reg, bool with_web, bool with_shell,
                            bool with_codebase);

// Ranks and de-duplicates search results before they reach the context window.
struct WebResult {
  std::string title, url, snippet, host;
  double score = 0.0;
};
// Scoring: query-term coverage in title+snippet (BM25-ish, saturating), a
// domain-quality prior, a penalty for near-duplicate snippets, and a small bonus
// for hosts that appear more than once across engines.  Returns at most `k`.
std::vector<WebResult> rank_web_results(std::vector<WebResult> in,
                                        const std::string& query, size_t k);
// Strips tags/scripts/nav from HTML and keeps the main text.
std::string html_to_text(const std::string& html, std::string* title);
// Extractive compression to fit a byte budget: keeps the sentences that carry
// the most query terms and rare words, in original order.
std::string summarise_extractive(const std::string& text, const std::string& query,
                                 size_t budget_chars);

}  // namespace slm
