// SPDX-License-Identifier: Apache-2.0
//
// The local tools: shell, read_file, write_file, list_dir.
//
// Every path argument goes through `resolve_in_workspace`, which is the only
// containment mechanism in the agent: the model is assumed to be adversarially
// bad at paths (it will happily write "../../etc/passwd"), so containment is
// checked on the *resolved* path rather than by looking for ".." in the string,
// which symlinks and "a/../../b" both defeat.
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "agent/tools.h"
#include "core/text.h"

namespace fs = std::filesystem;

namespace slm {
namespace {

constexpr int64_t kMaxReadBytes = 4 * 1024 * 1024;  // refuse bigger files
constexpr size_t kMaxListEntries = 400;

// `Tool::preview` is called before `run`, so it has no ToolContext and no way to
// learn the workspace - yet the approval dialog is useless without the cwd.  The
// last workspace a local tool actually ran in is therefore remembered here; the
// process cwd is the (correct, for a single-workspace session) initial guess.
std::mutex& hint_mutex() {
  static std::mutex m;
  return m;
}

std::string* hint_slot() {
  static std::string hint;
  return &hint;
}

std::string workspace_hint() {
  std::lock_guard<std::mutex> g(hint_mutex());
  std::string* h = hint_slot();
  if (h->empty()) {
    std::error_code ec;
    const fs::path p = fs::current_path(ec);
    *h = ec ? std::string(".") : p.string();
  }
  return *h;
}

void remember_workspace(const std::string& ws) {
  if (ws.empty()) return;
  std::lock_guard<std::mutex> g(hint_mutex());
  *hint_slot() = ws;
}

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  const auto sp = [](char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
  };
  while (a < b && sp(s[a])) ++a;
  while (b > a && sp(s[b - 1])) --b;
  return s.substr(a, b - a);
}

std::string workspace_root(const ToolContext& ctx, std::string* err) {
  std::error_code ec;
  const std::string ws = ctx.workspace.empty() ? std::string(".") : ctx.workspace;
  fs::path root = fs::absolute(fs::path(ws), ec);
  if (ec) {
    if (err) *err = "cannot resolve the workspace path '" + ws + "'";
    return std::string();
  }
  root = fs::weakly_canonical(root, ec);
  if (ec) {
    if (err) *err = "cannot resolve the workspace path '" + ws + "'";
    return std::string();
  }
  return root.generic_string();
}

// Resolves `rel` (relative to the workspace, or absolute) and refuses anything
// outside the workspace subtree.
bool resolve_in_workspace(const ToolContext& ctx, const std::string& rel, fs::path* out,
                          std::string* err) {
  const std::string arg = trim(rel);
  if (arg.empty()) {
    *err = "missing 'path' argument";
    return false;
  }
  const std::string root = workspace_root(ctx, err);
  if (root.empty()) return false;

  std::error_code ec;
  fs::path want(arg);
  if (!want.is_absolute()) want = fs::path(root) / want;
  const fs::path canon = fs::weakly_canonical(want, ec);
  if (ec) {
    *err = "cannot resolve path '" + arg + "'";
    return false;
  }
  const std::string cs = canon.generic_string();
  const bool inside =
      cs == root ||
      (cs.size() > root.size() && cs.compare(0, root.size(), root) == 0 &&
       (root.back() == '/' || cs[root.size()] == '/'));
  if (!inside) {
    *err = "refused: '" + arg + "' resolves to '" + cs +
           "', which is outside the workspace '" + root + "'";
    return false;
  }
  *out = canon;
  return true;
}

std::string display_path(const ToolContext& ctx, const fs::path& p) {
  const std::string root = workspace_root(ctx, nullptr);
  if (root.empty()) return p.generic_string();
  const std::string rel = p.lexically_relative(fs::path(root)).generic_string();
  return rel.empty() || rel == "." ? p.filename().generic_string() : rel;
}

ToolResult fail(const std::string& msg) {
  ToolResult r;
  r.ok = false;
  r.error = msg;
  r.output = "error: " + msg;
  return r;
}

// ------------------------------------------------------------------- shell
// A minimal environment: inheriting the parent's makes the agent's behaviour
// depend on whatever happened to be exported, and leaks tokens held in env vars
// to anything the model decides to run.
std::vector<std::string> minimal_env() {
  const auto pick = [](const char* key, const char* def) {
    const char* v = std::getenv(key);
    return std::string(key) + "=" + ((v && *v) ? v : def);
  };
  return {pick("PATH", "/usr/local/bin:/usr/bin:/bin"), pick("HOME", "/tmp"),
          pick("LANG", "C.UTF-8"), pick("TERM", "dumb")};
}

struct RunOutcome {
  std::string out;
  int exit_code = -1;
  bool timed_out = false;
  bool cancelled = false;
  bool spawn_failed = false;
  int64_t total_bytes = 0;
};

RunOutcome run_shell(const std::string& cmd, const std::string& cwd, double timeout_s,
                     size_t cap, std::atomic<bool>* cancel) {
  RunOutcome o;
  int fds[2];
  if (::pipe(fds) != 0) {
    o.spawn_failed = true;
    return o;
  }
  const std::vector<std::string> envs = minimal_env();
  std::vector<char*> envp;
  for (const std::string& e : envs) envp.push_back(const_cast<char*>(e.c_str()));
  envp.push_back(nullptr);
  std::string sh = "/bin/sh", flag = "-c", body = cmd;
  char* argv[] = {&sh[0], &flag[0], &body[0], nullptr};

  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    o.spawn_failed = true;
    return o;
  }
  if (pid == 0) {
    // Own process group: a timeout must kill the whole tree ("sleep 5 | cat"),
    // not just the shell.
    ::setpgid(0, 0);
    if (!cwd.empty() && ::chdir(cwd.c_str()) != 0) ::_exit(126);
    const int devnull = ::open("/dev/null", O_RDONLY);
    if (devnull >= 0) {
      ::dup2(devnull, STDIN_FILENO);
      ::close(devnull);
    }
    ::close(fds[0]);
    ::dup2(fds[1], STDOUT_FILENO);
    ::dup2(fds[1], STDERR_FILENO);  // merged: the model needs to see errors
    ::close(fds[1]);
    ::execve("/bin/sh", argv, envp.data());
    ::_exit(127);
  }
  ::close(fds[1]);
  ::setpgid(pid, pid);  // also here: whichever call wins, the group exists

  using clock = std::chrono::steady_clock;
  const auto budget =
      std::chrono::milliseconds(static_cast<long long>(timeout_s * 1000.0));
  const auto deadline = clock::now() + budget;
  for (;;) {
    if (cancel && cancel->load()) {
      ::killpg(pid, SIGKILL);
      o.cancelled = true;
      break;
    }
    const double left = std::chrono::duration<double>(deadline - clock::now()).count();
    if (left <= 0.0) {
      ::killpg(pid, SIGKILL);
      o.timed_out = true;
      break;
    }
    struct pollfd pf = {fds[0], POLLIN, 0};
    const int pr = ::poll(&pf, 1, std::min(200, static_cast<int>(left * 1000.0) + 1));
    if (pr < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (pr == 0) continue;
    char buf[16 * 1024];
    const ssize_t got = ::read(fds[0], buf, sizeof(buf));
    if (got <= 0) break;  // EOF: the command finished
    o.total_bytes += got;
    if (o.out.size() < cap)
      o.out.append(buf, std::min(static_cast<size_t>(got), cap - o.out.size()));
  }
  // Drain whatever is still buffered after a kill so the reason is visible.
  if (o.timed_out || o.cancelled) {
    char buf[4096];
    for (;;) {
      struct pollfd pf = {fds[0], POLLIN, 0};
      if (::poll(&pf, 1, 20) <= 0) break;
      const ssize_t got = ::read(fds[0], buf, sizeof(buf));
      if (got <= 0) break;
      o.total_bytes += got;
      if (o.out.size() < cap)
        o.out.append(buf, std::min(static_cast<size_t>(got), cap - o.out.size()));
    }
  }
  ::close(fds[0]);
  int status = 0;
  ::waitpid(pid, &status, 0);
  if (WIFEXITED(status)) o.exit_code = WEXITSTATUS(status);
  else if (WIFSIGNALED(status)) o.exit_code = 128 + WTERMSIG(status);
  return o;
}

class ShellTool : public Tool {
 public:
  ToolSpec spec() const override {
    ToolSpec s;
    s.name = "shell";
    s.summary = "run a shell command in the workspace and return its output";
    s.usage = "[[tool:shell cmd=\"ls -la src\"]]";
    s.risk = ToolRisk::kDangerous;
    s.params = {{"cmd", "the command line, run with /bin/sh -c", true, ""},
                {"timeout", "seconds before the process group is killed", false, "30"}};
    return s;
  }

  std::string preview(const ToolCall& call) const override {
    const std::string cmd = command_of(call);
    return "run in " + workspace_hint() + ":\n  $ " + utf8_truncate(cmd, 600);
  }

  ToolResult run(const ToolCall& call, ToolContext& ctx) override {
    const std::string cmd = command_of(call);
    if (cmd.empty()) return fail("missing 'cmd' argument");
    std::string err;
    const std::string root = workspace_root(ctx, &err);
    if (root.empty()) return fail(err);
    remember_workspace(root);

    int64_t t = call.arg_int("timeout", 30);
    if (t < 1) t = 1;
    if (t > 600) t = 600;  // an agent that needs longer should split the work
    const size_t cap = static_cast<size_t>(
        std::max<int64_t>(4096, std::min<int64_t>(ctx.output_budget * 4, 256000)));

    const RunOutcome o =
        run_shell(cmd, root, static_cast<double>(t), cap, ctx.cancel);
    ToolResult r;
    if (o.spawn_failed) return fail("cannot start /bin/sh");

    r.output = "$ " + cmd + "\n" + utf8_sanitize(o.out);
    if (!r.output.empty() && r.output.back() != '\n') r.output += "\n";
    if (o.cancelled) {
      r.output += "[cancelled; process group killed]\n";
      r.error = "cancelled";
    } else if (o.timed_out) {
      r.output += "[timeout after " + std::to_string(t) +
                  "s; process group killed]\n";
      r.error = "timeout after " + std::to_string(t) + "s";
    } else {
      r.output += "[exit " + std::to_string(o.exit_code) + "]\n";
      if (o.exit_code != 0)
        r.error = "command exited with status " + std::to_string(o.exit_code);
    }
    if (o.total_bytes > static_cast<int64_t>(cap)) {
      r.truncated = true;
      r.bytes_before_truncation = o.total_bytes;
      r.output += "[output capped at " + std::to_string(cap) + " bytes of " +
                  std::to_string(o.total_bytes) + "]\n";
    }
    r.ok = r.error.empty();
    r.display = r.output;
    return r;
  }

 private:
  static std::string command_of(const ToolCall& call) {
    std::string cmd = call.arg("cmd", "");
    if (cmd.empty()) cmd = call.arg("command", "");
    if (cmd.empty()) cmd = call.arg("_", "");
    return trim(cmd);
  }
};

// --------------------------------------------------------------- read_file
bool parse_range(const std::string& spec, int64_t* from, int64_t* to) {
  const std::string s = trim(spec);
  if (s.empty()) return false;
  const size_t dash = s.find('-');
  if (dash == std::string::npos) {
    *from = *to = std::strtoll(s.c_str(), nullptr, 10);
    return *from > 0;
  }
  *from = std::strtoll(s.substr(0, dash).c_str(), nullptr, 10);
  const std::string tail = trim(s.substr(dash + 1));
  *to = tail.empty() ? -1 : std::strtoll(tail.c_str(), nullptr, 10);
  if (*from <= 0) *from = 1;
  return true;
}

class ReadFileTool : public Tool {
 public:
  ToolSpec spec() const override {
    ToolSpec s;
    s.name = "read_file";
    s.summary = "read a text file from the workspace, optionally one line range";
    s.usage = "[[tool:read_file path=\"src/main.cpp\" lines=\"1-40\"]]";
    s.risk = ToolRisk::kSafe;
    s.params = {{"path", "path inside the workspace", true, ""},
                {"lines", "\"START-END\", 1-based and inclusive", false, ""}};
    return s;
  }

  ToolResult run(const ToolCall& call, ToolContext& ctx) override {
    fs::path p;
    std::string err;
    if (!resolve_in_workspace(ctx, call.arg("path", ""), &p, &err)) return fail(err);
    remember_workspace(workspace_root(ctx, nullptr));

    std::error_code ec;
    if (!fs::exists(p, ec)) return fail("no such file: " + display_path(ctx, p));
    if (fs::is_directory(p, ec))
      return fail(display_path(ctx, p) + " is a directory; use list_dir");
    const std::uintmax_t sz = fs::file_size(p, ec);
    if (ec) return fail("cannot stat " + display_path(ctx, p));
    if (static_cast<int64_t>(sz) > kMaxReadBytes)
      return fail(display_path(ctx, p) + " is " + std::to_string(sz) +
                  " bytes, over the 4 MB read limit");

    int64_t from = 1, to = -1;
    const bool ranged = parse_range(call.arg("lines", ""), &from, &to);
    if (!ranged) {
      from = 1;
      to = -1;
    }
    std::ifstream f(p.string(), std::ios::binary);
    if (!f) return fail("cannot open " + display_path(ctx, p));

    std::string body, line;
    int64_t n = 0, first = 0, last = 0;
    while (std::getline(f, line)) {
      ++n;
      if (n < from) continue;
      if (to > 0 && n > to) break;
      if (!first) first = n;
      last = n;
      if (!line.empty() && line.back() == '\r') line.pop_back();
      char num[24];
      std::snprintf(num, sizeof(num), "%6lld| ", static_cast<long long>(n));
      body += num;
      body += line;
      body += "\n";
    }
    ToolResult r;
    r.ok = true;
    r.output = display_path(ctx, p) + ":" + std::to_string(first) + "-" +
               std::to_string(last) + "\n" + utf8_sanitize(body);
    if (first == 0)
      r.output = display_path(ctx, p) + ":0-0\n[no lines in that range; the file has " +
                 std::to_string(n) + " lines]\n";
    r.display = r.output;
    return r;
  }
};

// -------------------------------------------------------------- write_file
class WriteFileTool : public Tool {
 public:
  ToolSpec spec() const override {
    ToolSpec s;
    s.name = "write_file";
    s.summary = "create or overwrite a file inside the workspace";
    s.usage = "[[tool:write_file]]\npath: notes.md\ncontent: hello\n[[/tool]]";
    s.risk = ToolRisk::kWrite;
    s.params = {{"path", "path inside the workspace", true, ""},
                {"content", "the bytes to write", true, ""},
                {"append", "\"true\" to append instead of replacing", false, "false"}};
    return s;
  }

  std::string preview(const ToolCall& call) const override {
    const std::string c = call.arg("content", "");
    const bool app = truthy(call.arg("append", "false"));
    return std::string(app ? "append " : "write ") + std::to_string(c.size()) +
           " bytes to " + call.arg("path", "?") + " (in " + workspace_hint() + ")\n  " +
           utf8_truncate(c, 200) + (c.size() > 200 ? " ..." : "");
  }

  ToolResult run(const ToolCall& call, ToolContext& ctx) override {
    fs::path p;
    std::string err;
    if (!resolve_in_workspace(ctx, call.arg("path", ""), &p, &err)) return fail(err);
    remember_workspace(workspace_root(ctx, nullptr));
    const std::string content = call.arg("content", "");
    const bool app = truthy(call.arg("append", "false"));

    std::error_code ec;
    if (fs::is_directory(p, ec)) return fail(display_path(ctx, p) + " is a directory");
    if (p.has_parent_path()) {
      fs::create_directories(p.parent_path(), ec);
      if (ec) return fail("cannot create " + p.parent_path().generic_string());
    }
    std::ofstream f(p.string(), app ? (std::ios::binary | std::ios::app)
                                    : (std::ios::binary | std::ios::trunc));
    if (!f) return fail("cannot open " + display_path(ctx, p) + " for writing");
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
    f.close();
    if (!f) return fail("write to " + display_path(ctx, p) + " failed");

    ToolResult r;
    r.ok = true;
    r.output = std::string(app ? "appended " : "wrote ") +
               std::to_string(content.size()) + " bytes to " + display_path(ctx, p) +
               "\n";
    r.display = r.output;
    return r;
  }

 private:
  static bool truthy(const std::string& v) {
    const std::string s = trim(v);
    return s == "1" || s == "true" || s == "True" || s == "TRUE" || s == "yes";
  }
};

// ---------------------------------------------------------------- list_dir
bool skip_dir_name(const std::string& n) {
  return n == ".git" || n == "node_modules" || n == "__pycache__" ||
         n == ".venv" || n == "build";
}

class ListDirTool : public Tool {
 public:
  ToolSpec spec() const override {
    ToolSpec s;
    s.name = "list_dir";
    s.summary = "list files and directories inside the workspace";
    s.usage = "[[tool:list_dir path=\"src\" depth=\"2\"]]";
    s.risk = ToolRisk::kSafe;
    s.params = {{"path", "directory inside the workspace", false, "."},
                {"depth", "1..3 levels of recursion", false, "1"}};
    return s;
  }

  ToolResult run(const ToolCall& call, ToolContext& ctx) override {
    std::string arg = call.arg("path", ".");
    if (trim(arg).empty()) arg = ".";
    fs::path p;
    std::string err;
    if (!resolve_in_workspace(ctx, arg, &p, &err)) return fail(err);
    remember_workspace(workspace_root(ctx, nullptr));

    std::error_code ec;
    if (!fs::exists(p, ec)) return fail("no such directory: " + display_path(ctx, p));
    if (!fs::is_directory(p, ec))
      return fail(display_path(ctx, p) + " is not a directory");
    int64_t depth = call.arg_int("depth", 1);
    if (depth < 1) depth = 1;
    if (depth > 3) depth = 3;

    std::vector<std::pair<std::string, std::string>> rows;  // (path, formatted row)
    size_t total = 0, dirs = 0, files = 0;
    fs::recursive_directory_iterator it(
        p, fs::directory_options::skip_permission_denied, ec);
    if (ec) return fail("cannot list " + display_path(ctx, p));
    const fs::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
      if (ec) break;
      const fs::directory_entry& e = *it;
      const std::string name = e.path().filename().string();
      const bool is_dir = e.is_directory(ec);
      if (is_dir && (skip_dir_name(name) || it.depth() + 1 >= depth))
        it.disable_recursion_pending();
      if (is_dir && skip_dir_name(name)) continue;
      ++total;
      if (total > kMaxListEntries) continue;
      const std::string rel = e.path().lexically_relative(p).generic_string();
      char row[64];
      if (is_dir) {
        ++dirs;
        std::snprintf(row, sizeof(row), "d %9s  ", "-");
      } else {
        ++files;
        const std::uintmax_t sz = e.is_regular_file(ec) ? fs::file_size(e.path(), ec) : 0;
        std::snprintf(row, sizeof(row), "- %9lld  ", static_cast<long long>(ec ? 0 : sz));
      }
      rows.emplace_back(rel, std::string(row) + rel + (is_dir ? "/" : ""));
    }
    // recursive_directory_iterator order is unspecified; a stable listing makes
    // the output diffable between calls.
    std::sort(rows.begin(), rows.end());

    ToolResult r;
    r.ok = true;
    r.output = display_path(ctx, p) + "/  (" + std::to_string(dirs) + " dirs, " +
               std::to_string(files) + " files, depth " + std::to_string(depth) + ")\n";
    for (const auto& kv : rows) r.output += kv.second + "\n";
    if (total > kMaxListEntries) {
      r.truncated = true;
      r.output += "[... " + std::to_string(total - kMaxListEntries) +
                  " more entries; narrow the path or lower depth]\n";
    }
    r.display = r.output;
    return r;
  }
};

}  // namespace

ToolPtr make_shell_tool() { return std::make_shared<ShellTool>(); }
ToolPtr make_read_file_tool() { return std::make_shared<ReadFileTool>(); }
ToolPtr make_write_file_tool() { return std::make_shared<WriteFileTool>(); }
ToolPtr make_list_dir_tool() { return std::make_shared<ListDirTool>(); }

}  // namespace slm
