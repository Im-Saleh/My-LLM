// SPDX-License-Identifier: Apache-2.0
//
// Codebase tools: the retrieval side of the index exposed to the model.
//
// These are the tools that let either model answer "where is X defined" and
// "what does this project do" about a repository far larger than its context
// window.  They are all read-only and confined to whatever was indexed, so they
// are ToolRisk::kSafe and run without an approval prompt.
//
// Retrieval is also applied automatically by AgentRuntime for questions that look
// like code questions; these tools exist for the cases where the model wants to
// look something *else* up after reading the first result, which is exactly what
// a small model needs to be able to do to recover from a bad first retrieval.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "agent/codebase.h"
#include "agent/tools.h"
#include "core/text.h"

namespace slm {
namespace {

std::string no_index() {
  return "no codebase has been indexed yet - run `slm agent --index DIR`, or press "
         "'index' in the codebase panel";
}

// One hit rendered for a model: the header line locates it, the body is trimmed
// so several hits fit in a small context window.
std::string render_hit(const CodebaseIndex& idx, const SearchHit& h, size_t budget) {
  const CodeChunk* c = idx.chunk(h.chunk);
  if (!c) return {};
  char head[256];
  std::snprintf(head, sizeof(head), "%s  [%s, score %.2f, %s]\n", c->header().c_str(),
                c->kind.c_str(), h.score, h.why.c_str());
  return std::string(head) + utf8_truncate(c->text, budget) + "\n";
}

// ------------------------------------------------------------------ code_search
class CodeSearchTool : public Tool {
 public:
  ToolSpec spec() const override {
    ToolSpec s;
    s.name = "code_search";
    s.summary = "search the indexed codebase (hybrid keyword + vector)";
    s.usage = "[[tool:code_search]]\nquery: how are updates accepted\nk: 3\n[[/tool]]";
    s.risk = ToolRisk::kSafe;
    s.params = {{"query", "what to look for", true, ""},
                {"k", "how many results (default 4, max 10)", false, "4"},
                {"mode", "hybrid | lexical | dense", false, "hybrid"}};
    return s;
  }
  ToolResult run(const ToolCall& call, ToolContext& ctx) override {
    ToolResult r;
    if (!ctx.codebase || ctx.codebase->empty()) {
      r.error = no_index();
      return r;
    }
    const std::string q = call.arg("query", call.arg("_"));
    if (q.empty()) {
      r.error = "code_search needs a query";
      return r;
    }
    const size_t k = static_cast<size_t>(
        std::max<int64_t>(1, std::min<int64_t>(10, call.arg_int("k", 4))));
    const std::string m = call.arg("mode", "hybrid");
    SearchMode mode = SearchMode::kHybrid;
    if (m == "lexical") mode = SearchMode::kLexical;
    else if (m == "dense") mode = SearchMode::kDense;

    const std::vector<SearchHit> hits = ctx.codebase->search(q, k, mode);
    if (hits.empty()) {
      r.ok = true;
      r.output = "no match for: " + q;
      return r;
    }
    // Split the output budget between the hits so one huge chunk cannot crowd
    // out the rest of the answer.
    const size_t per = std::max<size_t>(
        200, static_cast<size_t>(ctx.output_budget) / (hits.size() + 1));
    std::string out;
    for (const SearchHit& h : hits) out += render_hit(*ctx.codebase, h, per);
    r.ok = true;
    r.output = out;
    return r;
  }
};

// ----------------------------------------------------------------- find_symbol
class FindSymbolTool : public Tool {
 public:
  ToolSpec spec() const override {
    ToolSpec s;
    s.name = "find_symbol";
    s.summary = "locate the definition of a function, class or heading by name";
    s.usage = "[[tool:find_symbol]]\nname: qpack_synthesise\n[[/tool]]";
    s.risk = ToolRisk::kSafe;
    s.params = {{"name", "exact or partial symbol name", true, ""},
                {"k", "how many candidates (default 3)", false, "3"}};
    return s;
  }
  ToolResult run(const ToolCall& call, ToolContext& ctx) override {
    ToolResult r;
    if (!ctx.codebase || ctx.codebase->empty()) {
      r.error = no_index();
      return r;
    }
    const std::string name = call.arg("name", call.arg("_"));
    if (name.empty()) {
      r.error = "find_symbol needs a name";
      return r;
    }
    const size_t k = static_cast<size_t>(
        std::max<int64_t>(1, std::min<int64_t>(10, call.arg_int("k", 3))));
    const std::vector<SearchHit> hits = ctx.codebase->find_symbol(name, k);
    if (hits.empty()) {
      r.ok = true;
      r.output = "no symbol named " + name;
      return r;
    }
    const size_t per = std::max<size_t>(
        200, static_cast<size_t>(ctx.output_budget) / (hits.size() + 1));
    std::string out;
    for (const SearchHit& h : hits) out += render_hit(*ctx.codebase, h, per);
    r.ok = true;
    r.output = out;
    return r;
  }
};

// --------------------------------------------------------------- repo_overview
class RepoOverviewTool : public Tool {
 public:
  ToolSpec spec() const override {
    ToolSpec s;
    s.name = "repo_overview";
    s.summary = "structure of the indexed repository: languages, layout, entry points";
    s.usage = "[[tool:repo_overview]]";
    s.risk = ToolRisk::kSafe;
    return s;
  }
  ToolResult run(const ToolCall& call, ToolContext& ctx) override {
    (void)call;
    ToolResult r;
    if (!ctx.codebase || ctx.codebase->empty()) {
      r.error = no_index();
      return r;
    }
    r.ok = true;
    r.output = ctx.codebase->overview(static_cast<size_t>(ctx.output_budget));
    return r;
  }
};

}  // namespace

ToolPtr make_code_search_tool() { return std::make_shared<CodeSearchTool>(); }
ToolPtr make_find_symbol_tool() { return std::make_shared<FindSymbolTool>(); }
ToolPtr make_repo_overview_tool() { return std::make_shared<RepoOverviewTool>(); }

}  // namespace slm
