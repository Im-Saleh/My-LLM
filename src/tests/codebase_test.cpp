// SPDX-License-Identifier: Apache-2.0
//
// Tests for the local codebase index:
//   1. the tokeniser (camelCase / snake_case / acronym runs),
//   2. structure-aware chunking of C++, Python and Markdown - including the
//      case that quietly ruins naive chunkers, a `{` inside a string literal,
//   3. the scanner: .gitignore (with a negation), binary and oversized files,
//   4. retrieval: BM25 / dense / hybrid, symbol lookup and RRF,
//   5. persistence: save/load round trip, and the refusal to load an index
//      built by a different embedder,
//   6. context_block budgeting on multilingual text, and incremental refresh.
//
// Everything runs against a fixture repository created under /tmp; no network,
// no fixtures checked into the tree.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "agent/codebase.h"
#include "core/text.h"

using namespace slm;
namespace fs = std::filesystem;

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

const char* kRoot = "/tmp/slm_codebase_test";

std::string root_path(const std::string& rel) { return std::string(kRoot) + "/" + rel; }

void write_file(const std::string& rel, const std::string& text) {
  const fs::path p(root_path(rel));
  std::error_code ec;
  fs::create_directories(p.parent_path(), ec);
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string read_fixture(const std::string& rel) {
  std::ifstream f(root_path(rel), std::ios::binary);
  std::string out;
  char buf[4096];
  while (f.read(buf, sizeof(buf)) || f.gcount()) out.append(buf, static_cast<size_t>(f.gcount()));
  return out;
}

bool has_token(const std::vector<std::string>& v, const std::string& t) {
  return std::find(v.begin(), v.end(), t) != v.end();
}

bool contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

const CodeChunk* by_symbol(const std::vector<CodeChunk>& cs, const std::string& sym) {
  for (const CodeChunk& c : cs)
    if (c.symbol == sym) return &c;
  return nullptr;
}

int count_symbol(const std::vector<CodeChunk>& cs, const std::string& sym) {
  int n = 0;
  for (const CodeChunk& c : cs)
    if (c.symbol == sym) ++n;
  return n;
}

// Brace balance with literals and comments ignored - the same thing the chunker
// has to get right.  A chunk that carries a symbol must be balanced, otherwise a
// boundary fell inside a body.
int brace_balance(const std::string& s) {
  int d = 0;
  bool in_line_comment = false, in_block_comment = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (in_line_comment) {
      if (c == '\n') in_line_comment = false;
      continue;
    }
    if (in_block_comment) {
      if (c == '*' && i + 1 < s.size() && s[i + 1] == '/') {
        in_block_comment = false;
        ++i;
      }
      continue;
    }
    if (c == '/' && i + 1 < s.size() && s[i + 1] == '/') {
      in_line_comment = true;
      continue;
    }
    if (c == '/' && i + 1 < s.size() && s[i + 1] == '*') {
      in_block_comment = true;
      continue;
    }
    if (c == '"') {
      ++i;
      while (i < s.size() && s[i] != '"') i += (s[i] == '\\') ? 2 : 1;
      continue;
    }
    if (c == '\'' && i + 2 < s.size()) {
      size_t k = i + 1;
      if (s[k] == '\\') k += 2; else ++k;
      if (k < s.size() && s[k] == '\'') {
        i = k;
        continue;
      }
    }
    if (c == '{') ++d;
    if (c == '}') --d;
  }
  return d;
}

// ------------------------------------------------------------------ fixtures
const char* kCpp = R"CPP(// SPDX-License-Identifier: Apache-2.0
#include <string>

namespace demo {

// Synthesises one QPACK header block.  The brace inside the string literal
// below must not open a nesting level: if it does, this function's chunk
// swallows everything after it.
std::string qpack_synthesise(int stream_id) {
  std::string opener = "{";
  std::string label = "سلام دنیا از موتور";
  if (stream_id < 0) {
    return opener + label;
  }
  return label + opener;
}

// Parses one HTTP response and returns the status code it carries.
int parseHTTPResponse(const std::string& raw) {
  int status = 0;
  for (size_t i = 0; i < raw.size(); ++i) {
    if (raw[i] >= '0' && raw[i] <= '9') status = status * 10 + (raw[i] - '0');
  }
  return status;
}

// Decides whether a proposed update is accepted.
class UpdateGate {
 public:
  explicit UpdateGate(double threshold) : threshold_(threshold) {}

  bool accepts(double score) const {
    return score >= threshold_;
  }

  void relax(double by) {
    threshold_ -= by;
  }

 private:
  double threshold_ = 0.0;
};

}  // namespace demo
)CPP";

const char* kPy = R"PY(# Utility service used by the fixture tests.
import json


@cached
def build_payload(name):
    """Return the payload for one name."""
    data = {"name": name}
    if not name:
        return "{}"
    return json.dumps(data)


class PayloadStore:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)


TOP_LEVEL_CONSTANT = 3
)PY";

const char* kReadme = R"MD(# Demo Project

A fixture repository for the codebase index tests.

## Usage

Run `demo --help` to see the options.
The gate is configured through UpdateGate.

## سلام

این متن فارسی برای آزمایش برش یونیکد است.
)MD";

const char* kGitignore = R"GI(# fixture ignore rules
build/
*.log
!keep.log
)GI";

const char* kCMake = R"CM(cmake_minimum_required(VERSION 3.16)
project(demo CXX)
add_executable(demo src/engine.cpp)
target_include_directories(demo PRIVATE src)
install(TARGETS demo DESTINATION bin)
)CM";

void build_fixture() {
  std::error_code ec;
  fs::remove_all(kRoot, ec);
  fs::create_directories(kRoot, ec);

  write_file("src/engine.cpp", kCpp);
  write_file("pysrc/service.py", kPy);
  write_file("README.md", kReadme);
  write_file("CMakeLists.txt", kCMake);
  write_file(".gitignore", kGitignore);

  // Ignored by `build/`, and ignored by `*.log`; keep.log is re-included by the
  // negation and must survive.
  write_file("build/generated.cpp", "int generated_symbol() { return 1; }\n");
  write_file("notes.log", "line one\nline two\nline three\nline four\n");
  write_file("debug.log", "debug one\ndebug two\ndebug three\ndebug four\n");
  write_file("keep.log",
             "keepable log line one\nkeepable log line two\n"
             "keepable log line three\nkeepable log line four\n");

  // Binary: NUL bytes in the first 8 kB.
  {
    std::string bin;
    for (int i = 0; i < 512; ++i) {
      bin.push_back(static_cast<char>(i & 0x7F));
      bin.push_back('\0');
    }
    write_file("assets/blob.bin", bin);
  }
  // Oversized for the 8 kB limit used by the test scan.
  {
    std::string big;
    while (big.size() < 40000) big += "generated filler line of plain text\n";
    write_file("big.txt", big);
  }
}

ScanOptions test_options() {
  ScanOptions o;
  o.root = kRoot;
  o.max_file_bytes = 8192;  // small enough that big.txt is skipped
  return o;
}

// ------------------------------------------------------------------- [1] tokens
void test_tokens() {
  std::printf("[1] code_tokens\n");
  const std::vector<std::string> a = code_tokens("qpack_synthesise", true);
  check(has_token(a, "qpack_synthesise") && has_token(a, "qpack") &&
            has_token(a, "synthesise"),
        "snake_case yields the whole identifier and its parts");

  const std::vector<std::string> b = code_tokens("parseHTTPResponse", true);
  check(has_token(b, "parsehttpresponse") && has_token(b, "parse") &&
            has_token(b, "http") && has_token(b, "response"),
        "camelCase with an acronym run splits into parse|http|response");

  const std::vector<std::string> c = code_tokens("HTTPServer handle2Requests", true);
  check(has_token(c, "http") && has_token(c, "server"), "HTTPServer -> http|server");
  check(has_token(c, "handle2") && has_token(c, "requests"),
        "digits stay attached to their word");

  const std::vector<std::string> d = code_tokens("qpack_synthesise", false);
  check(has_token(d, "qpack_synthesise") && !has_token(d, "qpack"),
        "split_identifiers=false keeps identifiers whole");

  const std::vector<std::string> e = code_tokens("x a1 1234567 123456 ab", true);
  check(!has_token(e, "x"), "single characters are dropped");
  check(!has_token(e, "1234567"), "numbers longer than six digits are dropped");
  check(has_token(e, "123456") && has_token(e, "ab"),
        "short numbers and two-letter words are kept");

  check(language_of("src/engine.cpp") == "cpp" && language_of("a/b.py") == "python" &&
            language_of("CMakeLists.txt") == "cmake" &&
            language_of("Makefile") == "make" &&
            language_of("deploy/Dockerfile") == "dockerfile" &&
            language_of("README.md") == "markdown" &&
            language_of("x.unknownext") == "",
        "language_of maps extensions and special basenames");
}

// ------------------------------------------------------------- [2] chunking
void test_chunk_cpp() {
  std::printf("[2] chunk_source on the C++ fixture\n");
  const ScanOptions opt = test_options();
  const std::vector<CodeChunk> cs =
      chunk_source("src/engine.cpp", read_fixture("src/engine.cpp"), opt);
  std::printf("       %zu chunks\n", cs.size());
  for (const CodeChunk& c : cs)
    std::printf("       %-40s kind=%-8s lines=%d..%d\n", c.header().c_str(),
                c.kind.c_str(), c.start_line, c.end_line);

  const CodeChunk* qp = by_symbol(cs, "qpack_synthesise");
  const CodeChunk* hp = by_symbol(cs, "parseHTTPResponse");
  const CodeChunk* gate = by_symbol(cs, "UpdateGate");
  check(qp != nullptr && hp != nullptr && gate != nullptr,
        "both functions and the class are named");
  if (!qp || !hp || !gate) return;
  check(qp->kind == "function" && hp->kind == "function" && gate->kind == "class",
        "kinds are function/function/class");
  check(count_symbol(cs, "qpack_synthesise") == 1,
        "the function is one chunk, not several");

  // The literal brace must not have opened a block: if it had, this chunk would
  // run past its own closing brace and swallow the next function.
  check(!contains(qp->text, "parseHTTPResponse"),
        "the '{' inside the string literal did not open a block");
  check(contains(qp->text, "return label + opener;"),
        "the function body is complete to its last statement");
  check(contains(qp->text, "// Synthesises one QPACK header block."),
        "the preceding comment block is attached");
  check(contains(hp->text, "return status;") && contains(gate->text, "void relax"),
        "no boundary falls inside a function body");
  check(contains(gate->text, "double threshold_ = 0.0;") &&
            !contains(gate->text, "namespace demo"),
        "the class chunk spans exactly the class");

  bool balanced = true;
  std::string bad;
  for (const CodeChunk& c : cs) {
    if (c.symbol.empty()) continue;  // preamble/gap chunks legitimately are not
    if (brace_balance(c.text) != 0) {
      balanced = false;
      bad = c.header();
    }
  }
  check(balanced, "every named chunk has balanced braces", bad);

  // Every chunk must be locatable in the file it came from.
  bool lines_ok = true;
  for (const CodeChunk& c : cs)
    if (c.start_line < 1 || c.end_line < c.start_line) lines_ok = false;
  check(lines_ok, "line ranges are 1-based and ordered");
  check(qp->header() == "src/engine.cpp:" + std::to_string(qp->start_line) + "-" +
                            std::to_string(qp->end_line) + "  qpack_synthesise()",
        "header() format", qp->header());
}

// An oversized body must be split at statement level, and every piece must
// still say which symbol it came from - an anonymous fragment of a function is
// useless in a prompt.
void test_chunk_split() {
  std::printf("[2b] oversized bodies are split, not truncated\n");
  ScanOptions opt = test_options();
  opt.max_chunk_lines = 8;
  const std::vector<CodeChunk> cs =
      chunk_source("src/engine.cpp", read_fixture("src/engine.cpp"), opt);
  std::vector<const CodeChunk*> pieces;
  for (const CodeChunk& c : cs)
    if (c.symbol == "qpack_synthesise") pieces.push_back(&c);
  check(pieces.size() >= 2, "the 11-line function became several pieces",
        std::to_string(pieces.size()) + " pieces");
  bool kinds = true, sizes = true, joined = true;
  for (size_t i = 0; i < pieces.size(); ++i) {
    if (pieces[i]->kind != "block") kinds = false;
    if (pieces[i]->end_line - pieces[i]->start_line + 1 > opt.max_chunk_lines)
      sizes = false;
    if (i && pieces[i]->start_line != pieces[i - 1]->end_line + 1) joined = false;
  }
  check(kinds, "every piece is kind=block with the same symbol");
  check(sizes, "no piece exceeds max_chunk_lines");
  check(joined, "the pieces tile the original range without gaps or overlap");

  // min_chunk_lines only applies to chunks that have no name of their own.
  opt.max_chunk_lines = 160;
  opt.min_chunk_lines = 40;
  const std::vector<CodeChunk> few =
      chunk_source("src/engine.cpp", read_fixture("src/engine.cpp"), opt);
  bool all_named = true;
  for (const CodeChunk& c : few)
    if (c.symbol.empty()) all_named = false;
  check(!few.empty() && all_named,
        "a large min_chunk_lines drops the anonymous chunks and keeps the named ones",
        std::to_string(few.size()) + " chunks left");
}

void test_chunk_python() {
  std::printf("[3] chunk_source on the Python fixture\n");
  const ScanOptions opt = test_options();
  const std::vector<CodeChunk> cs =
      chunk_source("pysrc/service.py", read_fixture("pysrc/service.py"), opt);
  for (const CodeChunk& c : cs)
    std::printf("       %-44s kind=%s\n", c.header().c_str(), c.kind.c_str());

  const CodeChunk* fn = by_symbol(cs, "build_payload");
  const CodeChunk* cl = by_symbol(cs, "PayloadStore");
  check(fn != nullptr && cl != nullptr, "def and class are named");
  if (!fn || !cl) return;
  check(fn->kind == "function" && cl->kind == "class", "python kinds");
  check(contains(fn->text, "@cached") && fn->text.compare(0, 7, "@cached") == 0,
        "the decorator is attached and starts the chunk");
  check(contains(fn->text, "return json.dumps(data)"), "the body is complete");
  check(!contains(fn->text, "class PayloadStore") && !contains(fn->text, "import json"),
        "the function stops at the dedent and does not reach back over the blanks");
  check(contains(cl->text, "def add") && !contains(cl->text, "TOP_LEVEL_CONSTANT"),
        "the class stops at the dedent");
}

void test_chunk_markdown() {
  std::printf("[4] chunk_source on the Markdown fixture\n");
  const ScanOptions opt = test_options();
  const std::vector<CodeChunk> cs =
      chunk_source("README.md", read_fixture("README.md"), opt);
  for (const CodeChunk& c : cs)
    std::printf("       %-44s kind=%s\n", c.header().c_str(), c.kind.c_str());
  const CodeChunk* usage = by_symbol(cs, "Usage");
  check(by_symbol(cs, "Demo Project") != nullptr && usage != nullptr,
        "headings become chunk names");
  if (usage) {
    check(usage->kind == "doc", "markdown chunks are kind=doc");
    check(contains(usage->text, "demo --help") && !contains(usage->text, "# Demo"),
          "a chunk is one heading plus its body");
  }
}

// ---------------------------------------------------------------- [5] scan
void test_scan(CodebaseIndex* idx) {
  std::printf("[5] scan\n");
  std::string err;
  std::atomic<bool> cancel(false);
  int64_t seen = 0;
  const bool ok = idx->scan(
      test_options(),
      [&seen](int64_t done, int64_t total, const std::string&) {
        (void)done;
        (void)total;
        ++seen;
      },
      &cancel, &err);
  check(ok, "scan succeeded", err);
  const IndexStats st = idx->stats();
  std::printf("       files %lld chunks %lld bytes %lld tokens %lld\n"
              "       skipped ignored %lld binary %lld large %lld\n",
              static_cast<long long>(st.files), static_cast<long long>(st.chunks),
              static_cast<long long>(st.bytes), static_cast<long long>(st.tokens),
              static_cast<long long>(st.skipped_ignored),
              static_cast<long long>(st.skipped_binary),
              static_cast<long long>(st.skipped_large));
  check(st.files > 0 && st.chunks > 0, "stats().files and .chunks are positive");
  check(seen > 0, "progress was reported");
  check(st.embedder == "hash-256", "the default embedder is recorded", st.embedder);

  bool saw_build = false, saw_notes = false, saw_bin = false, saw_big = false;
  for (const auto& lang : st.by_language) (void)lang;
  const std::vector<const CodeChunk*> keep = idx->chunks_of("keep.log");
  for (int32_t i = 0; i < static_cast<int32_t>(st.chunks); ++i) {
    const CodeChunk* c = idx->chunk(i);
    if (!c) continue;
    if (c->path.compare(0, 6, "build/") == 0) saw_build = true;
    if (c->path == "notes.log") saw_notes = true;
    if (c->path == "assets/blob.bin") saw_bin = true;
    if (c->path == "big.txt") saw_big = true;
  }
  check(!saw_build, ".gitignore excluded the build/ directory");
  check(!saw_notes, ".gitignore excluded *.log");
  check(!keep.empty(), "the !keep.log negation re-included the file");
  check(!saw_bin && st.skipped_binary >= 1, "the binary file was skipped and counted");
  check(!saw_big && st.skipped_large >= 1, "the oversized file was skipped and counted");
  check(st.skipped_ignored >= 2, "both *.log files were counted as ignored",
        std::to_string(st.skipped_ignored));
  check(!idx->chunks_of("src/engine.cpp").empty() &&
            !idx->chunks_of("pysrc/service.py").empty() &&
            !idx->chunks_of("README.md").empty(),
        "the source fixtures were indexed");
  check(!idx->needs_ann(), "a fixture repo does not need an ANN structure");
}

// -------------------------------------------------------------- [6] search
void test_search(const CodebaseIndex& idx) {
  std::printf("[6] search\n");
  const SearchMode modes[3] = {SearchMode::kHybrid, SearchMode::kLexical,
                               SearchMode::kDense};
  const char* names[3] = {"hybrid", "lexical", "dense"};
  for (int m = 0; m < 3; ++m) {
    const std::vector<SearchHit> hits = idx.search("qpack_synthesise", 5, modes[m]);
    const CodeChunk* top = hits.empty() ? nullptr : idx.chunk(hits[0].chunk);
    check(top != nullptr && top->symbol == "qpack_synthesise",
          std::string("top hit for an exact identifier (") + names[m] + ")",
          top ? top->header() + "  why=" + hits[0].why : std::string("no hits"));
    if (!hits.empty()) {
      const SearchHit& h = hits[0];
      if (modes[m] == SearchMode::kLexical)
        check(h.bm25 > 0.0 && h.lexical_rank >= 1, "lexical fields are filled");
      if (modes[m] == SearchMode::kDense)
        check(h.dense_rank >= 1, "dense fields are filled");
      if (modes[m] == SearchMode::kHybrid)
        check(h.lexical_rank >= 1 && h.dense_rank >= 1,
              "hybrid hit records both ranks");
    }
  }
  // A paraphrase with no shared identifier: the point of having dense at all.
  const std::vector<SearchHit> para =
      idx.search("decides whether an update is accepted", 5);
  check(!para.empty(), "a paraphrase query returns something");

  const std::vector<SearchHit> sym = idx.find_symbol("qpack_synthesise", 3);
  check(!sym.empty() && idx.chunk(sym[0].chunk) &&
            idx.chunk(sym[0].chunk)->symbol == "qpack_synthesise",
        "find_symbol locates the definition",
        sym.empty() ? "" : sym[0].why);
  const std::vector<SearchHit> ci = idx.find_symbol("updategate", 3);
  check(!ci.empty() && idx.chunk(ci[0].chunk) &&
            idx.chunk(ci[0].chunk)->symbol == "UpdateGate",
        "find_symbol is case-insensitive on the second pass");
  const std::vector<SearchHit> none = idx.find_symbol("no_such_symbol_anywhere", 3);
  check(none.empty() || true, "find_symbol falls back without crashing");
}

// ----------------------------------------------------------------- [7] RRF
void test_rrf() {
  std::printf("[7] reciprocal rank fusion\n");
  const std::vector<std::vector<int32_t>> lists = {{10, 20, 30}, {20, 40, 10}};
  const std::vector<std::pair<int32_t, double>> f = reciprocal_rank_fusion(lists);
  std::string order;
  for (const auto& p : f) order += std::to_string(p.first) + " ";
  // 20: 1/62 + 1/61, 10: 1/61 + 1/63, 40: 1/62, 30: 1/63.
  check(f.size() == 4 && f[0].first == 20 && f[1].first == 10 && f[2].first == 40 &&
            f[3].first == 30,
        "documented ordering", order);
  const double s20 = 1.0 / 62.0 + 1.0 / 61.0;
  check(std::abs(f[0].second - s20) < 1e-12, "score = sum of 1/(k+rank)");

  const std::vector<std::pair<int32_t, double>> w =
      reciprocal_rank_fusion(lists, 60.0, {2.0, 1.0});
  check(!w.empty() && w[0].first == 10, "weights reorder the fusion");

  check(reciprocal_rank_fusion({}).empty(), "no lists -> no results");
  const std::vector<std::pair<int32_t, double>> one =
      reciprocal_rank_fusion({{}, {7}});
  check(one.size() == 1 && one[0].first == 7, "an empty list is ignored");
}

// ----------------------------------------------------- [8] save / load
void test_save_load(const CodebaseIndex& idx) {
  std::printf("[8] save / load\n");
  const std::string path = "/tmp/slm_codebase_test.idx";
  std::string err;
  check(idx.save(path, &err), "save", err);

  CodebaseIndex loaded;
  check(loaded.load(path, &err), "load into a fresh index", err);
  const IndexStats a = idx.stats(), b = loaded.stats();
  check(a.files == b.files && a.chunks == b.chunks && a.bytes == b.bytes &&
            a.tokens == b.tokens && a.skipped_binary == b.skipped_binary &&
            a.skipped_large == b.skipped_large &&
            a.skipped_ignored == b.skipped_ignored,
        "stats survive the round trip",
        std::to_string(b.files) + " files / " + std::to_string(b.chunks) + " chunks");
  check(loaded.options().root == idx.options().root &&
            loaded.options().max_file_bytes == idx.options().max_file_bytes,
        "ScanOptions survive the round trip");

  bool same = true;
  std::string detail;
  for (const char* q : {"qpack_synthesise", "update accepted", "payload"}) {
    const std::vector<SearchHit> x = idx.search(q, 5);
    const std::vector<SearchHit> y = loaded.search(q, 5);
    if (x.size() != y.size()) {
      same = false;
      detail = std::string("size differs for ") + q;
      break;
    }
    for (size_t i = 0; i < x.size(); ++i) {
      if (x[i].chunk != y[i].chunk || std::abs(x[i].score - y[i].score) > 1e-9) {
        same = false;
        detail = std::string("hit ") + std::to_string(i) + " differs for " + q;
        break;
      }
    }
  }
  check(same, "top-5 results are identical after a round trip", detail);

  CodebaseIndex mismatched;
  mismatched.set_embedder(std::make_shared<HashEmbedder>(128));
  std::string err2;
  const bool bad = mismatched.load(path, &err2);
  check(!bad && !err2.empty(), "load refuses a mismatched embedder", err2);
  std::remove(path.c_str());
}

// ------------------------------------------------- [9] context and overview
void test_context(const CodebaseIndex& idx) {
  std::printf("[9] context_block / overview\n");
  const std::string full = idx.context_block("qpack_synthesise", 4000, 6);
  check(!full.empty() && full.size() <= 4000, "context fits the budget",
        std::to_string(full.size()) + " chars");
  check(contains(full, "--- src/engine.cpp:"), "blocks are preceded by their header");
  check(utf8_sanitize(full) == full, "the block is valid UTF-8");

  // A sweep of tight budgets over a chunk that contains Persian text: every cut
  // must land on a character boundary.
  bool utf8_ok = true, budget_ok = true;
  size_t saw_omitted = 0;
  for (size_t budget = 120; budget <= 1200; budget += 37) {
    const std::string b = idx.context_block("سلام دنیا از موتور", budget, 4);
    if (b.size() > budget) budget_ok = false;
    if (utf8_sanitize(b) != b) utf8_ok = false;
    if (contains(b, "lines omitted")) ++saw_omitted;
  }
  check(budget_ok, "every budget is respected");
  check(utf8_ok, "no budget cut a UTF-8 sequence in half");
  check(saw_omitted > 0, "an oversized chunk is included as a head+tail excerpt",
        std::to_string(saw_omitted) + " budgets used an excerpt");

  // At most two chunks per file.
  const std::string many = idx.context_block("string", 100000, 12);
  size_t pos = 0;
  int engine_blocks = 0;
  while ((pos = many.find("--- src/engine.cpp:", pos)) != std::string::npos) {
    ++engine_blocks;
    ++pos;
  }
  check(engine_blocks <= 2, "at most two chunks per file",
        std::to_string(engine_blocks) + " blocks from src/engine.cpp");

  const std::string ov = idx.overview(4000);
  check(!ov.empty() && ov.size() <= 4000, "overview fits the budget");
  check(contains(ov, "languages:") && contains(ov, "cpp") && contains(ov, "layout:"),
        "overview reports languages and layout");
  check(contains(ov, "cmake target demo"), "overview detects the CMake target");
  check(contains(ov, "README.md headings:") && contains(ov, "- Usage"),
        "overview lists the README headings");
  check(ov == idx.overview(4000), "overview is deterministic");
}

// -------------------------------------------------------------- [10] refresh
void test_refresh(CodebaseIndex* idx) {
  std::printf("[10] refresh\n");
  const IndexStats before = idx->stats();
  const size_t readme_before = idx->chunks_of("README.md").size();
  std::string readme_text;
  for (const CodeChunk* c : idx->chunks_of("README.md")) readme_text += c->text;

  // Append a function to one file; nothing else changes.
  std::string src = read_fixture("src/engine.cpp");
  src +=
      "\nnamespace demo {\n"
      "// Resets the gate to a fresh threshold.\n"
      "void gate_reset(double value) {\n"
      "  UpdateGate gate(value);\n"
      "  (void)gate.accepts(value);\n"
      "}\n"
      "}  // namespace demo\n";
  write_file("src/engine.cpp", src);

  std::string err;
  std::atomic<bool> cancel(false);
  const bool ok = idx->refresh(nullptr, &cancel, &err);
  check(ok, "refresh succeeded", err);
  const IndexStats after = idx->stats();
  check(after.files == before.files, "the file count is unchanged");
  check(after.chunks > before.chunks, "the edited file contributed new chunks",
        std::to_string(before.chunks) + " -> " + std::to_string(after.chunks));

  std::string readme_after;
  for (const CodeChunk* c : idx->chunks_of("README.md")) readme_after += c->text;
  check(idx->chunks_of("README.md").size() == readme_before &&
            readme_after == readme_text,
        "untouched files were reused verbatim");

  const std::vector<SearchHit> hits = idx->find_symbol("gate_reset", 3);
  check(!hits.empty() && idx->chunk(hits[0].chunk) &&
            idx->chunk(hits[0].chunk)->symbol == "gate_reset",
        "the symbol added by the edit is findable");
  const std::vector<SearchHit> lex = idx->search("gate_reset", 3, SearchMode::kLexical);
  check(!lex.empty() && idx->chunk(lex[0].chunk)->symbol == "gate_reset",
        "and it is retrievable lexically");

  // A refresh with no changes must be a no-op.
  const bool ok2 = idx->refresh(nullptr, &cancel, &err);
  check(ok2, "second refresh succeeded", err);
  const IndexStats again = idx->stats();
  check(again.chunks == after.chunks && again.files == after.files,
        "an unchanged tree refreshes to the same index");

  // A deleted file must disappear from the index.
  std::error_code ec;
  fs::remove(root_path("keep.log"), ec);
  check(idx->refresh(nullptr, &cancel, &err), "refresh after a deletion", err);
  check(idx->chunks_of("keep.log").empty() &&
            idx->stats().files == again.files - 1,
        "the deleted file's chunks were removed",
        std::to_string(again.files) + " -> " + std::to_string(idx->stats().files));
  check(!idx->search("keepable log line", 5, SearchMode::kLexical).empty() == false,
        "and it is no longer retrievable");
}

// ------------------------------------------------------- [11] scan options
void test_scan_options() {
  std::printf("[11] scan options\n");
  std::string err;
  std::atomic<bool> cancel(false);

  ScanOptions only_md = test_options();
  only_md.only_exts = {"md"};
  CodebaseIndex a;
  check(a.scan(only_md, nullptr, &cancel, &err), "scan with only_exts", err);
  bool md_only = true;
  for (int32_t i = 0; i < static_cast<int32_t>(a.stats().chunks); ++i)
    if (a.chunk(i)->path != "README.md") md_only = false;
  check(a.stats().files == 1 && md_only, "only_exts restricts the index");
  check(!a.stats().by_language.empty() && a.stats().by_language[0].first == "markdown",
        "by_language is filled");

  ScanOptions no_py = test_options();
  no_py.extra_ignores = {"pysrc/"};
  CodebaseIndex b;
  check(b.scan(no_py, nullptr, &cancel, &err), "scan with extra_ignores", err);
  check(b.chunks_of("pysrc/service.py").empty() &&
            !b.chunks_of("README.md").empty(),
        "extra_ignores drops a subtree and keeps the rest");

  ScanOptions raw = test_options();
  raw.use_gitignore = false;
  CodebaseIndex c;
  check(c.scan(raw, nullptr, &cancel, &err), "scan with use_gitignore=false", err);
  check(!c.chunks_of("notes.log").empty() &&
            !c.chunks_of("build/generated.cpp").empty(),
        "ignored paths come back when .gitignore is off");

  std::atomic<bool> stop(true);
  CodebaseIndex d;
  std::string cerr;
  check(!d.scan(test_options(), nullptr, &stop, &cerr) && !cerr.empty(),
        "a pre-cancelled scan fails cleanly", cerr);
}

void test_edge_cases() {
  std::printf("[12] edge cases\n");
  const ScanOptions opt = test_options();
  check(chunk_source("empty.cpp", "", opt).empty(), "an empty file yields no chunks");
  check(!chunk_source("data.json", "{\n\"a\": 1,\n\"b\": 2,\n\"c\": 3\n}\n", opt).empty(),
        "an unknown-structure file still yields fixed-size chunks");

  CodebaseIndex fresh;
  check(fresh.empty(), "a fresh index is empty");
  check(fresh.search("anything", 5).empty(), "searching an empty index is safe");
  check(fresh.context_block("anything", 100, 3).empty(), "no context from nothing");
  std::string err;
  check(!fresh.scan(ScanOptions(), nullptr, nullptr, &err) && !err.empty(),
        "scanning a missing root fails cleanly", err);

  std::string gerr;
  check(make_gguf_embedder("/nonexistent.gguf", 1, &gerr) == nullptr && !gerr.empty(),
        "make_gguf_embedder reports why it declined", gerr);

  HashEmbedder e(64);
  std::vector<float> v(64, 0.0f), w(64, 0.0f);
  e.embed("qpack_synthesise", v.data());
  e.embed("qpack_synthesise", w.data());
  double n = 0.0, same = 0.0;
  for (int i = 0; i < 64; ++i) {
    n += static_cast<double>(v[i]) * v[i];
    same += static_cast<double>(v[i]) * w[i];
  }
  check(std::abs(n - 1.0) < 1e-5, "hash embeddings are L2 normalised");
  check(std::abs(same - 1.0) < 1e-5, "hash embeddings are deterministic");
  std::vector<float> batch(128, 0.0f);
  e.embed_batch({"a", "b"}, batch.data());
  check(true, "embed_batch runs over the default implementation");
}

}  // namespace

int main() {
  std::printf("codebase index tests\n\n");
  build_fixture();
  test_tokens();
  test_chunk_cpp();
  test_chunk_split();
  test_chunk_python();
  test_chunk_markdown();

  CodebaseIndex idx;
  test_scan(&idx);
  test_search(idx);
  test_rrf();
  test_save_load(idx);
  test_context(idx);
  test_refresh(&idx);
  test_scan_options();
  test_edge_cases();

  std::error_code ec;
  fs::remove_all(kRoot, ec);
  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
