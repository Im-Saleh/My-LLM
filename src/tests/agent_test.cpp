// SPDX-License-Identifier: Apache-2.0
//
// Tests for the agent tool layer.  Nothing here touches the network: the two web
// tools are exercised through their pure parts (ranking, HTML extraction,
// summarisation, URL validation), which is where the bugs that matter live.
//
// The tests that matter most are the safety ones: path containment, "policy=deny
// really means the tool never ran", and the shell timeout actually killing the
// process group.  Those are the properties that keep a wrong model output from
// becoming a wrong action.
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "agent/http.h"
#include "agent/tools.h"
#include "core/text.h"
#include "telemetry.h"

namespace fs = std::filesystem;
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

bool has(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

// ------------------------------------------------------------------- parsing
void test_parse() {
  std::printf("[1] tolerant call parsing\n");

  const std::string reply =
      "Let me look that up first.\n"
      "[[tool:web_search]]\n"
      "query: persian nlp datasets\n"
      "[[/tool]]\n"
      "and then a quick listing\n"
      "[[tool:shell cmd=\"ls -la src\"]]\n"
      "and finally {\"tool\": \"read_file\", \"path\": \"src/main.cpp\", "
      "\"lines\": \"1-40\"}\n";
  const std::vector<ToolCall> calls = ToolRegistry::parse(reply);
  check(calls.size() == 3, "three calls in one reply",
        "got " + std::to_string(calls.size()));
  if (calls.size() == 3) {
    check(calls[0].name == "web_search" &&
              calls[0].arg("query") == "persian nlp datasets",
          "block form", calls[0].name + " query='" + calls[0].arg("query") + "'");
    check(calls[1].name == "shell" && calls[1].arg("cmd") == "ls -la src",
          "inline form with a quoted value containing spaces",
          calls[1].name + " cmd='" + calls[1].arg("cmd") + "'");
    check(calls[2].name == "read_file" && calls[2].arg("path") == "src/main.cpp" &&
              calls[2].arg("lines") == "1-40",
          "json form", calls[2].name + " path='" + calls[2].arg("path") + "'");
    check(calls[0].id == 1 && calls[2].id == 3, "ids are assigned in order");
    check(has(calls[1].raw, "[[tool:shell"), "raw text is captured per call");
  }

  const std::vector<ToolCall> unterminated =
      ToolRegistry::parse("[[tool:read_file]]\npath: src/model.h\nlines: 5-9\n");
  check(unterminated.size() == 1 && unterminated[0].name == "read_file" &&
            unterminated[0].arg("path") == "src/model.h" &&
            unterminated[0].arg("lines") == "5-9",
        "unterminated block ends at end of string");

  const std::vector<ToolCall> multiline = ToolRegistry::parse(
      "[[tool:write_file]]\npath: notes.md\ncontent: line one\nline two\n[[/tool]]");
  check(multiline.size() == 1 && multiline[0].arg("content") == "line one\nline two",
        "a value continues over following lines",
        "content='" + (multiline.empty() ? "" : multiline[0].arg("content")) + "'");

  const std::vector<ToolCall> bare = ToolRegistry::parse("[[tool:shell ls -la]]");
  check(bare.size() == 1 && bare[0].arg("cmd") == "ls -la",
        "a single unnamed argument is offered to any parameter");

  const std::vector<ToolCall> nested = ToolRegistry::parse(
      "{\"name\": \"list_dir\", \"arguments\": {\"path\": \"src\", \"depth\": 2}}");
  check(nested.size() == 1 && nested[0].name == "list_dir" &&
            nested[0].arg("path") == "src" && nested[0].arg_int("depth", 0) == 2,
        "json with nested arguments and a numeric value");

  check(ToolRegistry::parse("no calls here, just prose about [[brackets]].").empty(),
        "prose produces no calls");

  check(ToolRegistry::looks_like_broken_call("I will [[tool web_search query=x"),
        "a malformed marker is reported as a broken call");
  check(ToolRegistry::looks_like_broken_call("<tool>web_search</tool>"),
        "an xml-ish attempt is reported as a broken call");
  check(!ToolRegistry::looks_like_broken_call(reply),
        "a reply that parses is not reported as broken");
  check(!ToolRegistry::looks_like_broken_call("plain prose, no tools"),
        "prose is not reported as broken");
}

// ------------------------------------------------------------ approval gate
ToolCall make_call(const std::string& name, const std::string& k,
                   const std::string& v) {
  ToolCall c;
  c.name = name;
  c.args[k] = v;
  return c;
}

bool wait_for_pending(ApprovalGate* gate, size_t n) {
  for (int i = 0; i < 400; ++i) {  // 2 s worst case
    if (gate->pending_count() == n) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

void test_gate() {
  std::printf("[2] approval gate\n");
  ToolPolicy pol;
  pol.timeout_s = 5.0;
  ApprovalGate gate;

  const ToolCall call = make_call("shell", "cmd", "ls -la");
  for (int round = 0; round < 2; ++round) {
    const bool allow = round == 0;
    ToolDecision got = ToolDecision::kTimeout;
    std::thread th(
        [&] { got = gate.request(call, "preview", ToolRisk::kDangerous, pol); });
    const bool arrived = wait_for_pending(&gate, 1);
    check(arrived, allow ? "request appears as pending (allow round)"
                         : "request appears as pending (deny round)");
    if (arrived) {
      const std::vector<PendingApproval> p = gate.pending();
      check(p.size() == 1 && p[0].tool == "shell" && p[0].preview == "preview",
            "pending carries the tool and preview");
      gate.decide(p[0].id, allow, false);
    }
    th.join();
    check(got == (allow ? ToolDecision::kAllow : ToolDecision::kDeny),
          allow ? "decide(allow) returns kAllow" : "decide(deny) returns kDeny");
  }

  // remember: the same tool + args must not ask again.
  {
    ToolDecision got = ToolDecision::kTimeout;
    std::thread th(
        [&] { got = gate.request(call, "preview", ToolRisk::kDangerous, pol); });
    if (wait_for_pending(&gate, 1)) {
      const std::vector<PendingApproval> p = gate.pending();
      gate.decide(p[0].id, true, true);
    }
    th.join();
    check(got == ToolDecision::kAllow && gate.remembered() == 1,
          "allow + remember records the canonical call");
    const double t0 = Telemetry::now();
    const ToolDecision again =
        gate.request(call, "preview", ToolRisk::kDangerous, pol);
    check(again == ToolDecision::kAllow && Telemetry::now() - t0 < 0.5,
          "a remembered call is allowed without asking");
    gate.forget_all();
    check(gate.remembered() == 0, "forget_all clears remembered approvals");
  }

  // deny_all must wake a blocked requester.
  {
    ToolDecision got = ToolDecision::kAllow;
    std::thread th([&] {
      got = gate.request(make_call("shell", "cmd", "rm -rf /"), "nope",
                         ToolRisk::kDangerous, pol);
    });
    const bool arrived = wait_for_pending(&gate, 1);
    gate.deny_all();
    th.join();
    check(arrived && got == ToolDecision::kDeny, "deny_all wakes a waiter with kDeny");
    check(gate.pending_count() == 0, "no pending requests are left behind");
  }

  // timeout
  {
    ToolPolicy quick;
    quick.timeout_s = 0.15;
    const double t0 = Telemetry::now();
    const ToolDecision got =
        gate.request(make_call("shell", "cmd", "sleep 1"), "p", ToolRisk::kDangerous,
                     quick);
    const double dt = Telemetry::now() - t0;
    check(got == ToolDecision::kTimeout && dt >= 0.1 && dt < 3.0,
          "an unanswered request times out", "after " + std::to_string(dt) + "s");
  }
}

// --------------------------------------------------------------- policy gate
class FakeTool : public Tool {
 public:
  explicit FakeTool(ToolRisk risk, std::string name, size_t out_bytes = 4)
      : risk_(risk), name_(std::move(name)), out_bytes_(out_bytes) {}
  bool ran = false;

  ToolSpec spec() const override {
    ToolSpec s;
    s.name = name_;
    s.summary = "test tool";
    s.usage = "[[tool:" + name_ + "]]";
    s.risk = risk_;
    s.params = {{"x", "anything", false, ""}};
    return s;
  }
  ToolResult run(const ToolCall&, ToolContext&) override {
    ran = true;
    ToolResult r;
    r.ok = true;
    r.output = std::string(out_bytes_, 'z');
    return r;
  }

 private:
  ToolRisk risk_;
  std::string name_;
  size_t out_bytes_;
};

void test_policy_and_invoke() {
  std::printf("[3] policy and invoke\n");
  ToolPolicy pol;  // defaults: safe/network auto, write/dangerous ask
  check(pol.mode_for(ToolRisk::kSafe) == 1, "default policy auto-allows safe");
  check(pol.mode_for(ToolRisk::kNetwork) == 1, "default policy auto-allows network");
  check(pol.mode_for(ToolRisk::kWrite) == 0, "default policy asks for write");
  check(pol.mode_for(ToolRisk::kDangerous) == 0, "default policy asks for dangerous");
  pol.safe = 2;
  pol.network = 0;
  pol.write = 1;
  pol.dangerous = 2;
  check(pol.mode_for(ToolRisk::kSafe) == 2 && pol.mode_for(ToolRisk::kNetwork) == 0 &&
            pol.mode_for(ToolRisk::kWrite) == 1 &&
            pol.mode_for(ToolRisk::kDangerous) == 2,
        "mode_for reports every configured risk level");
  check(std::string(tool_risk_name(ToolRisk::kDangerous)) == "dangerous" &&
            std::string(tool_risk_name(ToolRisk::kSafe)) == "safe",
        "risk names");

  auto danger = std::make_shared<FakeTool>(ToolRisk::kDangerous, "fake_danger");
  auto safe = std::make_shared<FakeTool>(ToolRisk::kSafe, "fake_safe", 200);
  ToolRegistry reg;
  reg.add(danger);
  reg.add(safe);

  Telemetry tel;
  ToolContext ctx;
  ctx.workspace = ".";
  ctx.tel = &tel;

  ToolPolicy deny;
  deny.dangerous = 2;
  const ToolResult denied = reg.invoke(make_call("fake_danger", "x", "1"), ctx, nullptr,
                                       deny);
  check(denied.denied && !denied.ok && !danger->ran,
        "policy=deny refuses without running the tool",
        "error='" + denied.error + "'");

  // ask + no gate connected must also refuse rather than run.
  ToolPolicy ask;
  ask.dangerous = 0;
  const ToolResult no_gate =
      reg.invoke(make_call("fake_danger", "x", "1"), ctx, nullptr, ask);
  check(no_gate.denied && !danger->ran, "ask with no approval channel refuses");

  ToolPolicy allow;
  allow.dangerous = 1;
  const ToolResult ok = reg.invoke(make_call("fake_danger", "x", "1"), ctx, nullptr,
                                   allow);
  check(ok.ok && danger->ran, "policy=auto runs the tool");

  ctx.output_budget = 20;
  const ToolResult cut = reg.invoke(make_call("fake_safe", "x", "1"), ctx, nullptr,
                                    allow);
  check(cut.truncated && cut.bytes_before_truncation == 200 &&
            cut.output.size() < 200,
        "output is truncated to the context budget",
        "kept " + std::to_string(cut.output.size()) + " of 200");
  ctx.output_budget = 8000;

  const ToolResult unknown = reg.invoke(make_call("nope", "x", "1"), ctx, nullptr,
                                        allow);
  check(!unknown.ok && has(unknown.error, "unknown tool"), "unknown tool is an error");

  check(reg.set_enabled("fake_safe", false) && !reg.enabled("fake_safe"),
        "set_enabled toggles a tool");
  const ToolResult off = reg.invoke(make_call("fake_safe", "x", "1"), ctx, nullptr,
                                    allow);
  check(!off.ok && off.denied, "a disabled tool refuses to run");
  reg.set_enabled("fake_safe", true);

  bool saw_calls = false;
  for (const ToolRegistry::Stat& s : reg.stats())
    if (s.name == "fake_danger") saw_calls = s.calls == 1 && s.denials == 2;
  check(saw_calls, "stats count calls and denials");

  const std::string compact = reg.catalogue(true);
  const std::string full = reg.catalogue(false);
  check(has(compact, "fake_danger(x?) - test tool") &&
            compact.find('\n') != std::string::npos,
        "compact catalogue is one line per tool");
  check(has(full, "dangerous") && has(full, "[[tool:fake_danger]]"),
        "full catalogue includes risk and usage");
  check(!tel.recent_logs(32).empty(), "every invocation is audited");
}

// ------------------------------------------------------------- local tools
struct Workspace {
  fs::path root;
  Workspace() {
    std::error_code ec;
    root = fs::temp_directory_path(ec) /
           ("slm_agent_test_" + std::to_string(::getpid()));
    fs::create_directories(root / "sub", ec);
    root = fs::weakly_canonical(root, ec);
    std::ofstream(root / "hello.txt") << "hello world\nsecond line\nthird line\n";
    std::ofstream(root / "sub" / "inner.txt") << "inner\n";
  }
  ~Workspace() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
};

void test_containment(const Workspace& ws) {
  std::printf("[4] path containment\n");
  ToolRegistry reg;
  register_builtin_tools(&reg, /*with_web=*/false, /*with_shell=*/true,
                         /*with_codebase=*/false);
  ToolPolicy pol;
  pol.safe = 1;
  pol.write = 1;
  pol.dangerous = 1;
  ToolContext ctx;
  ctx.workspace = ws.root.string();

  check(reg.get("read_file") && reg.get("write_file") && reg.get("list_dir") &&
            reg.get("shell"),
        "builtin tools are registered");

  for (const char* bad : {"../../etc/passwd", "/etc/passwd", "sub/../../../etc/passwd"}) {
    ToolCall c;
    c.name = "read_file";
    c.args["path"] = bad;
    const ToolResult r = reg.invoke(c, ctx, nullptr, pol);
    check(!r.ok && has(r.error, "outside the workspace"),
          std::string("read_file refuses ") + bad, r.error);
  }
  {
    ToolCall c;
    c.name = "write_file";
    c.args["path"] = "../slm_escape_probe.txt";
    c.args["content"] = "nope";
    const ToolResult r = reg.invoke(c, ctx, nullptr, pol);
    const bool leaked = fs::exists(ws.root.parent_path() / "slm_escape_probe.txt");
    check(!r.ok && !leaked, "write_file refuses to escape the workspace", r.error);
  }
  {
    ToolCall c;
    c.name = "list_dir";
    c.args["path"] = "..";
    const ToolResult r = reg.invoke(c, ctx, nullptr, pol);
    check(!r.ok && has(r.error, "outside the workspace"),
          "list_dir refuses the parent directory");
  }

  // The happy paths, so the containment check is not simply refusing everything.
  {
    ToolCall c;
    c.name = "read_file";
    c.args["path"] = "hello.txt";
    const ToolResult r = reg.invoke(c, ctx, nullptr, pol);
    check(r.ok && has(r.output, "hello world") && has(r.output, "hello.txt:1-3"),
          "read_file reads an in-workspace file with numbered lines");
    c.args["lines"] = "2-2";
    const ToolResult ranged = reg.invoke(c, ctx, nullptr, pol);
    check(ranged.ok && has(ranged.output, "second line") &&
              !has(ranged.output, "third line"),
          "read_file honours a line range");
  }
  {
    ToolCall c;
    c.name = "write_file";
    c.args["path"] = "sub/new.txt";
    c.args["content"] = "written by the test";
    const ToolResult r = reg.invoke(c, ctx, nullptr, pol);
    std::ifstream f((ws.root / "sub" / "new.txt").string());
    std::string body;
    std::getline(f, body);
    check(r.ok && body == "written by the test",
          "write_file writes inside the workspace");
    const std::string prev = reg.get("write_file")->preview(c);
    check(has(prev, "sub/new.txt") && has(prev, "19 bytes") &&
              has(prev, "written by the test"),
          "write_file preview shows path, size and a content excerpt", prev);
  }
  {
    ToolCall c;
    c.name = "list_dir";
    c.args["path"] = ".";
    c.args["depth"] = "2";
    const ToolResult r = reg.invoke(c, ctx, nullptr, pol);
    check(r.ok && has(r.output, "hello.txt") && has(r.output, "sub/inner.txt"),
          "list_dir walks to the requested depth");
  }
}

void test_shell(const Workspace& ws) {
  std::printf("[5] shell tool\n");
  ToolRegistry reg;
  register_builtin_tools(&reg, false, true, false);
  ToolPolicy pol;
  pol.dangerous = 1;
  ToolContext ctx;
  ctx.workspace = ws.root.string();

  {
    ToolCall c;
    c.name = "shell";
    c.args["cmd"] = "echo hello";
    const ToolResult r = reg.invoke(c, ctx, nullptr, pol);
    check(r.ok && has(r.output, "hello") && has(r.output, "[exit 0]"),
          "echo hello runs and reports exit 0");
    const std::string prev = reg.get("shell")->preview(c);
    check(has(prev, "echo hello") && has(prev, ws.root.string()),
          "shell preview shows the command and the cwd", prev);
  }
  {
    ToolCall c;
    c.name = "shell";
    c.args["cmd"] = "pwd";
    const ToolResult r = reg.invoke(c, ctx, nullptr, pol);
    check(r.ok && has(r.output, ws.root.string()),
          "the workspace is the working directory", r.output);
  }
  {
    ToolCall c;
    c.name = "shell";
    c.args["cmd"] = "echo out; echo err 1>&2; exit 3";
    const ToolResult r = reg.invoke(c, ctx, nullptr, pol);
    check(!r.ok && has(r.output, "out") && has(r.output, "err") &&
              has(r.output, "[exit 3]"),
          "stdout and stderr are merged and the exit code is reported");
  }
  {
    ToolCall c;
    c.name = "shell";
    c.args["cmd"] = "sleep 5";
    c.args["timeout"] = "1";
    const double t0 = Telemetry::now();
    const ToolResult r = reg.invoke(c, ctx, nullptr, pol);
    const double dt = Telemetry::now() - t0;
    check(!r.ok && has(r.error, "timeout") && has(r.output, "timeout"),
          "a hanging command is reported as a timeout", r.error);
    check(dt < 4.0, "the timeout kills the process instead of waiting",
          "took " + std::to_string(dt) + "s");
  }
}

// ---------------------------------------------------------------- html/text
void test_html() {
  std::printf("[6] html_to_text\n");
  const std::string html =
      "<html><head><title>Test &amp; Page</title>"
      "<style>body{color:red}</style></head>\n"
      "<body><script>var secret = \"DO_NOT_SHOW\";</script>\n"
      "<h1>Hello</h1><p>a &lt;b&gt; &#65;&#x42;&nbsp;spaced</p>\n"
      "<div><ul><li>one</li><li>two</li></ul></div>\n"
      "<nav>menu junk</nav><footer>footer junk</footer>\n"
      "<p>unterminated <b";
  std::string title;
  const std::string text = html_to_text(html, &title);
  check(title == "Test & Page", "title is extracted and decoded", "'" + title + "'");
  check(!has(text, "DO_NOT_SHOW") && !has(text, "var secret"),
        "script bodies never reach the text");
  check(!has(text, "color:red"), "style bodies never reach the text");
  check(!has(text, "menu junk") && !has(text, "footer junk"),
        "nav and footer are dropped");
  check(has(text, "a <b> AB spaced"), "entities decode (named, decimal, hex, nbsp)",
        "'" + text + "'");
  check(has(text, "one") && has(text, "two") && has(text, "\n"),
        "list items become separate lines");
  check(has(text, "unterminated"),
        "an unterminated tag does not lose the text before it");
  check(text.find("<script") == std::string::npos &&
            text.find("</p>") == std::string::npos,
        "no markup survives");
  check(text.find("\n\n\n") == std::string::npos, "blank line runs are collapsed");
}

void test_summarise() {
  std::printf("[7] summarise_extractive\n");
  const std::vector<std::string> sents = {
      "Alpha opens the document with a general statement.",
      "The weather in Tehran was pleasant that week.",
      "Nothing in this line is worth remembering at all.",
      "Quantisation packs the weights into four bit groups.",
      "A final unrelated remark closes the document."};
  std::string doc;
  for (const std::string& s : sents) doc += s + " ";
  const size_t budget = 140;
  const std::string out = summarise_extractive(doc, "quantisation four bit weights",
                                               budget);
  check(out.size() <= budget, "the summary respects the byte budget",
        std::to_string(out.size()) + " <= " + std::to_string(budget));
  check(has(out, "Quantisation packs the weights"),
        "the query-relevant sentence is kept", "'" + out + "'");

  size_t kept = 0, last = 0;
  bool ordered = true;
  for (const std::string& s : sents) {
    const size_t p = out.find(s.substr(0, s.size() - 1));  // '.' may be the cut point
    if (p == std::string::npos) continue;
    ++kept;
    if (p < last) ordered = false;
    last = p;
  }
  check(kept >= 2, "more than one sentence is kept", "kept " + std::to_string(kept));
  check(ordered, "kept sentences stay in document order");

  const std::string fa =
      "این پروژه یک مدل کوچک را آموزش می‌دهد. کوانتیزه‌سازی وزن‌ها را فشرده می‌کند؟ "
      "هوا در تهران خوب بود.";
  const std::string fa_out = summarise_extractive(fa, "کوانتیزه‌سازی", 80);
  check(fa_out.size() <= 80 && utf8_sanitize(fa_out) == fa_out,
        "a Persian summary is valid UTF-8 inside the budget",
        std::to_string(fa_out.size()) + " bytes");
  check(summarise_extractive("short text", "q", 4000) == "short text",
        "text that already fits is returned unchanged");
}

void test_ranking() {
  std::printf("[8] rank_web_results\n");
  std::vector<WebResult> in;
  in.push_back({"Quantisation guide",
                "https://example.org/quant?utm_source=x&utm_medium=y",
                "how to quantise weights into four bit groups", "", 0.0});
  in.push_back({"Quantisation guide", "https://example.org/quant/",
                "how to quantise weights into four bit groups", "", 0.0});
  in.push_back({"Cake recipes", "https://example.org/cake",
                "sponge cake, icing and a cup of tea", "", 0.0});

  const std::vector<WebResult> out = rank_web_results(in, "quantisation weights", 5);
  check(out.size() == 2, "the same URL with different utm_* params is de-duplicated",
        "got " + std::to_string(out.size()));
  check(!out.empty() && out[0].title == "Quantisation guide",
        "the query-relevant result outranks the irrelevant one");
  check(out.size() > 1 && out[0].score > out[1].score, "results are sorted by score");
  check(!out.empty() && out[0].host == "example.org", "the host is filled in");
  check(rank_web_results(in, "quantisation weights", 1).size() == 1,
        "at most k results are returned");
  check(rank_web_results({}, "anything", 5).empty(), "no input, no results");
}

void test_urls() {
  std::printf("[9] url validation and encoding\n");
  check(HttpClient::url_encode("a b") == "a%20b", "space is percent-encoded");
  check(HttpClient::url_encode("q=1&x/y") == "q%3D1%26x%2Fy",
        "reserved characters are percent-encoded");
  check(HttpClient::url_encode("azAZ09-_.~") == "azAZ09-_.~",
        "the unreserved set is left alone");
  check(HttpClient::url_encode("سلام").find('%') == 0, "utf-8 is encoded byte by byte");

  std::string why;
  check(HttpClient::url_is_safe("https://example.com/a?b=c", &why), "https is accepted");
  check(HttpClient::url_is_safe("http://example.com", &why), "http is accepted");
  check(!HttpClient::url_is_safe("file:///etc/passwd", &why), "file:// is rejected", why);
  check(!HttpClient::url_is_safe("javascript:alert(1)", &why), "javascript: is rejected",
        why);
  check(!HttpClient::url_is_safe("https://example.com/a\nb", &why),
        "an embedded newline is rejected", why);
  check(!HttpClient::url_is_safe("https://example.com/a b", &why),
        "an embedded space is rejected", why);
  check(!HttpClient::url_is_safe("", &why), "the empty URL is rejected", why);
  check(HttpClient::host_of("https://user:pw@Example.COM:8443/x") == "example.com",
        "host_of drops userinfo and port and lowercases");
  check(HttpClient::host_of("not a url").empty(), "host_of on nonsense is empty");
}

}  // namespace

int main() {
  std::printf("agent tool layer tests (http backend %s, available %d)\n\n",
              HttpClient::backend_name(), HttpClient::available() ? 1 : 0);
  Workspace ws;
  test_parse();
  test_gate();
  test_policy_and_invoke();
  test_containment(ws);
  test_shell(ws);
  test_html();
  test_summarise();
  test_ranking();
  test_urls();
  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
