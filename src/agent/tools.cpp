// SPDX-License-Identifier: Apache-2.0
#include "agent/tools.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "agent/http.h"
#include "core/text.h"
#include "telemetry.h"

namespace slm {

// Defined in tool_shell.cpp / tool_web.cpp.  They are declared here rather than
// in a header because nothing outside `register_builtin_tools` may construct a
// builtin directly: the registry is the only supported entry point.
ToolPtr make_shell_tool();
ToolPtr make_read_file_tool();
ToolPtr make_write_file_tool();
ToolPtr make_list_dir_tool();
ToolPtr make_web_search_tool();
ToolPtr make_web_fetch_tool();
// Defined in tool_codebase.cpp; declared weakly here so this translation unit
// still has no CodebaseIndex dependency.
ToolPtr make_code_search_tool();
ToolPtr make_find_symbol_tool();
ToolPtr make_repo_overview_tool();

namespace {

// --------------------------------------------------------------- tiny helpers
std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && is_space(s[a])) ++a;
  while (b > a && is_space(s[b - 1])) --b;
  return s.substr(a, b - a);
}

std::string rtrim(const std::string& s) {
  size_t b = s.size();
  while (b > 0 && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
  return s.substr(0, b);
}

bool is_name_char(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' ||
         c == '.';
}

// A token character for scoring: bytes >= 0x80 are kept so Persian words are
// single tokens instead of being shattered per byte.
bool is_word_byte(unsigned char c) {
  return std::isalnum(c) || c == '_' || c >= 0x80;
}

std::vector<std::string> words_of(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (unsigned char c : s) {
    if (is_word_byte(c)) {
      cur += static_cast<char>(std::tolower(c));
    } else if (!cur.empty()) {
      out.push_back(cur);
      cur.clear();
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

bool is_stopword(const std::string& w) {
  static const std::unordered_set<std::string> sw = {
      "the", "a", "an", "of", "in", "to", "and", "or", "for", "is", "are",
      "on",  "at", "by", "it", "as", "be", "how", "what", "with", "from"};
  return sw.count(w) > 0 || w.size() < 2;
}

bool ends_with_host(const std::string& host, const std::string& dom) {
  if (host == dom) return true;
  return host.size() > dom.size() &&
         host.compare(host.size() - dom.size() - 1, dom.size() + 1, "." + dom) == 0;
}

std::string strip_quotes(std::string v) {
  if (v.size() >= 2) {
    const char a = v.front(), b = v.back();
    if ((a == '"' && b == '"') || (a == '\'' && b == '\'') || (a == '`' && b == '`'))
      v = v.substr(1, v.size() - 2);
  }
  return v;
}

// ---------------------------------------------------------------- arg parsing
// Inline form: key="value with spaces" key2=bare, tolerant about commas and
// about a missing '=' (see the "_" fallback in ToolCall::arg).
void parse_inline_args(const std::string& s,
                       std::map<std::string, std::string>* out) {
  size_t i = 0;
  bool saw_kv = false;
  while (i < s.size()) {
    while (i < s.size() && (is_space(s[i]) || s[i] == ',')) ++i;
    const size_t k0 = i;
    while (i < s.size() && is_name_char(s[i])) ++i;
    if (i == k0) {  // not a key at all - skip one byte and keep going
      ++i;
      continue;
    }
    const std::string key = lower(s.substr(k0, i - k0));
    size_t j = i;
    while (j < s.size() && is_space(s[j])) ++j;
    if (j >= s.size() || (s[j] != '=' && s[j] != ':')) continue;  // bare word
    ++j;
    while (j < s.size() && is_space(s[j])) ++j;
    std::string val;
    if (j < s.size() && (s[j] == '"' || s[j] == '\'' || s[j] == '`')) {
      const char q = s[j++];
      while (j < s.size() && s[j] != q) {
        if (s[j] == '\\' && j + 1 < s.size()) ++j;
        val += s[j++];
      }
      if (j < s.size()) ++j;
    } else {
      while (j < s.size() && !is_space(s[j]) && s[j] != ',') val += s[j++];
    }
    (*out)[key] = val;
    saw_kv = true;
    i = j;
  }
  // `[[tool:shell ls -la]]`: no key=value anywhere, so the whole remainder is
  // the single argument the model meant to pass.
  if (!saw_kv) {
    const std::string t = trim(s);
    if (!t.empty()) (*out)["_"] = strip_quotes(t);
  }
}

// Block form: "key: value" per line.  A line without a key continues the
// previous value, which is what makes multi-line `content:` for write_file work.
void parse_block_args(const std::string& body,
                      std::map<std::string, std::string>* out) {
  std::string cur;
  size_t i = 0;
  while (i <= body.size()) {
    const size_t nl = body.find('\n', i);
    const size_t stop = (nl == std::string::npos) ? body.size() : nl;
    const std::string line = body.substr(i, stop - i);
    i = (nl == std::string::npos) ? body.size() + 1 : nl + 1;

    size_t k = 0;
    while (k < line.size() && (line[k] == ' ' || line[k] == '\t')) ++k;
    size_t k1 = k;
    while (k1 < line.size() && is_name_char(line[k1])) ++k1;
    const bool is_key = k1 > k && k1 < line.size() && line[k1] == ':' &&
                        (k1 + 1 == line.size() || line[k1 + 1] == ' ' ||
                         line[k1 + 1] == '\t' || line[k1 + 1] == '\r');
    if (is_key) {
      cur = lower(line.substr(k, k1 - k));
      (*out)[cur] = strip_quotes(trim(line.substr(k1 + 1)));
    } else if (!cur.empty()) {
      std::string& v = (*out)[cur];
      if (v.empty() && trim(line).empty()) continue;  // no leading blank lines
      if (!v.empty()) v += "\n";
      v += rtrim(line);
    }
  }
  // Trailing blank lines are an artefact of the block layout, never part of an
  // argument (the closing marker is always on its own line).
  for (auto& kv : *out) {
    std::string& v = kv.second;
    size_t b = v.size();
    while (b > 0 && is_space(v[b - 1])) --b;
    v.resize(b);
  }
}

// ------------------------------------------------------------ minimal JSON-ish
// Deliberately minimal: enough for {"tool":"read_file","path":"a","lines":"1-2"}
// and one level of nesting ("arguments": {...}, which is flattened into the same
// argument map).  Not a JSON parser - no arrays of objects, no \u decoding
// beyond the BMP escape being replaced by '?'.
bool json_string(const std::string& s, size_t* i, std::string* out) {
  if (*i >= s.size() || s[*i] != '"') return false;
  size_t j = *i + 1;
  out->clear();
  while (j < s.size() && s[j] != '"') {
    if (s[j] == '\\' && j + 1 < s.size()) {
      const char e = s[j + 1];
      j += 2;
      switch (e) {
        case 'n': *out += '\n'; break;
        case 't': *out += '\t'; break;
        case 'r': *out += '\r'; break;
        case 'u':
          *out += '?';  // rare in tool arguments; not worth a decoder
          j = std::min(s.size(), j + 4);
          break;
        default: *out += e;
      }
      continue;
    }
    *out += s[j++];
  }
  if (j >= s.size()) return false;  // unterminated string
  *i = j + 1;
  return true;
}

void json_skip_ws(const std::string& s, size_t* i) {
  while (*i < s.size() && (is_space(s[*i]) || s[*i] == ',')) ++*i;
}

// Skips a bracketed region ('[' or '{') honouring nesting and strings.
size_t json_skip_bracket(const std::string& s, size_t i) {
  const char open = s[i], close = (open == '[') ? ']' : '}';
  int depth = 0;
  for (; i < s.size(); ++i) {
    if (s[i] == '"') {
      std::string tmp;
      if (!json_string(s, &i, &tmp)) return s.size();
      --i;
      continue;
    }
    if (s[i] == open) ++depth;
    else if (s[i] == close && --depth == 0) return i + 1;
  }
  return s.size();
}

bool json_object(const std::string& s, size_t start, size_t* end,
                 std::vector<std::pair<std::string, std::string>>* out, int depth) {
  if (start >= s.size() || s[start] != '{') return false;
  size_t i = start + 1;
  bool any = false;
  for (;;) {
    json_skip_ws(s, &i);
    if (i >= s.size()) {
      *end = s.size();
      return any;  // unterminated object: keep whatever we understood
    }
    if (s[i] == '}') {
      *end = i + 1;
      return any;
    }
    std::string key;
    if (s[i] == '"') {
      if (!json_string(s, &i, &key)) {
        *end = s.size();
        return any;
      }
    } else {  // bare key, which small models emit often enough to accept
      const size_t k0 = i;
      while (i < s.size() && is_name_char(s[i])) ++i;
      if (i == k0) {
        *end = i + 1;
        return any;
      }
      key = s.substr(k0, i - k0);
    }
    json_skip_ws(s, &i);
    if (i >= s.size() || s[i] != ':') {
      *end = s.size();
      return any;
    }
    ++i;
    while (i < s.size() && is_space(s[i])) ++i;
    if (i >= s.size()) {
      *end = s.size();
      return any;
    }
    key = lower(key);
    if (s[i] == '"') {
      std::string val;
      if (!json_string(s, &i, &val)) {
        *end = s.size();
        return any;
      }
      out->emplace_back(key, val);
      any = true;
    } else if (s[i] == '{') {
      // Nested objects are flattened, so {"name":"f","arguments":{"p":1}} and
      // {"tool":"f","p":1} land in the same argument map.
      size_t sub_end = 0;
      if (depth < 2) json_object(s, i, &sub_end, out, depth + 1);
      i = json_skip_bracket(s, i);
      any = true;
    } else if (s[i] == '[') {
      const size_t e = json_skip_bracket(s, i);
      out->emplace_back(key, s.substr(i, e - i));
      i = e;
      any = true;
    } else {
      const size_t v0 = i;
      while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != '\n') ++i;
      out->emplace_back(key, trim(s.substr(v0, i - v0)));
      any = true;
    }
  }
}

// ------------------------------------------------------------ marker scanning
// Matches "[[tool:" case-insensitively and tolerates "[[ tool :".
bool tool_marker_at(const std::string& s, size_t i, size_t* name_start) {
  if (s.compare(i, 2, "[[") != 0) return false;
  size_t j = i + 2;
  while (j < s.size() && is_space(s[j])) ++j;
  if (lower(s.substr(j, 4)) != "tool") return false;
  j += 4;
  while (j < s.size() && is_space(s[j])) ++j;
  if (j >= s.size() || s[j] != ':') return false;
  ++j;
  while (j < s.size() && is_space(s[j])) ++j;
  *name_start = j;
  return true;
}

// Finds "[[/tool...]]" and returns its start; *after is set past the marker.
size_t find_close_marker(const std::string& s, size_t from, size_t* after) {
  const size_t p = s.find("[[/", from);
  if (p == std::string::npos) return std::string::npos;
  if (lower(s.substr(p + 3, 4)) != "tool") return std::string::npos;
  const size_t e = s.find("]]", p);
  *after = (e == std::string::npos) ? s.size() : e + 2;
  return p;
}

std::string canonical_args(const ToolCall& call) {
  // std::map is already ordered by key, which is exactly the canonical form the
  // "remember this approval" key needs.
  std::string s;
  for (const auto& kv : call.args) {
    if (!s.empty()) s += "\n";
    s += kv.first + "=" + kv.second;
  }
  return utf8_sanitize(s);
}

std::string remember_key(const std::string& tool, const std::string& canon) {
  return tool + std::string(1, '\0') + canon;
}

std::string fmt(double v, int prec) {
  char b[64];
  std::snprintf(b, sizeof(b), "%.*f", prec, v);
  return b;
}

}  // namespace

// ------------------------------------------------------------------- basics
const char* tool_risk_name(ToolRisk r) {
  switch (r) {
    case ToolRisk::kSafe: return "safe";
    case ToolRisk::kNetwork: return "network";
    case ToolRisk::kWrite: return "write";
    case ToolRisk::kDangerous: return "dangerous";
  }
  return "?";
}

std::string ToolCall::arg(const std::string& k, const std::string& def) const {
  const auto it = args.find(k);
  if (it != args.end()) return it->second;
  // A model that wrote `[[tool:shell ls -la]]` meant the tool's main argument;
  // the parser stored it as "_" because it cannot know the parameter names.
  if (args.size() == 1) {
    const auto only = args.find("_");
    if (only != args.end()) return only->second;
  }
  return def;
}

int64_t ToolCall::arg_int(const std::string& k, int64_t def) const {
  const std::string v = trim(arg(k, ""));
  if (v.empty()) return def;
  errno = 0;
  char* end = nullptr;
  const long long n = std::strtoll(v.c_str(), &end, 10);
  if (end == v.c_str() || errno == ERANGE) return def;
  return static_cast<int64_t>(n);
}

std::string Tool::preview(const ToolCall& call) const {
  std::string s = call.name + "(";
  bool first = true;
  for (const auto& kv : call.args) {
    if (!first) s += ", ";
    first = false;
    s += kv.first + "=" + utf8_truncate(kv.second, 120);
  }
  return s + ")";
}

int ToolPolicy::mode_for(ToolRisk r) const {
  int m = 0;
  switch (r) {
    case ToolRisk::kSafe: m = safe; break;
    case ToolRisk::kNetwork: m = network; break;
    case ToolRisk::kWrite: m = write; break;
    case ToolRisk::kDangerous: m = dangerous; break;
  }
  return m < 0 ? 0 : (m > 2 ? 2 : m);
}

// -------------------------------------------------------------- approval gate
ToolDecision ApprovalGate::request(const ToolCall& call, const std::string& preview,
                                   ToolRisk risk, const ToolPolicy& policy) {
  const std::string canon = canonical_args(call);
  const std::string key = remember_key(call.name, canon);
  std::unique_lock<std::mutex> lk(m_);
  if (std::find(remembered_.begin(), remembered_.end(), key) != remembered_.end())
    return ToolDecision::kAllow;

  const uint64_t id = next_id_++;
  Waiter w;
  w.info.id = id;
  w.info.tool = call.name;
  w.info.preview = preview;
  w.info.detail = canon;  // also the "remember" key, so decide() needs no extra state
  w.info.risk = risk;
  w.info.requested_at = Telemetry::now();
  w.info.timeout_s = policy.timeout_s;
  waiting_.emplace(id, w);
  cv_.notify_all();  // wakes the UI thread polling pending()

  const double timeout = policy.timeout_s > 0.0 ? policy.timeout_s : 0.001;
  const auto ms = std::chrono::milliseconds(static_cast<long long>(timeout * 1000.0));
  const auto deadline = std::chrono::steady_clock::now() + ms;
  ToolDecision out = ToolDecision::kTimeout;
  while (true) {
    const auto it = waiting_.find(id);
    if (it == waiting_.end()) break;  // cleared by deny_all()
    if (it->second.answered) {
      out = it->second.decision;
      break;
    }
    if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) {
      const auto it2 = waiting_.find(id);
      if (it2 != waiting_.end() && it2->second.answered) out = it2->second.decision;
      break;
    }
  }
  waiting_.erase(id);
  return out;
}

std::vector<PendingApproval> ApprovalGate::pending() const {
  std::lock_guard<std::mutex> g(m_);
  std::vector<PendingApproval> out;
  out.reserve(waiting_.size());
  for (const auto& kv : waiting_)
    if (!kv.second.answered) out.push_back(kv.second.info);
  return out;
}

void ApprovalGate::decide(uint64_t id, bool allow, bool remember) {
  {
    std::lock_guard<std::mutex> g(m_);
    const auto it = waiting_.find(id);
    if (it == waiting_.end()) return;
    it->second.decision = allow ? ToolDecision::kAllow : ToolDecision::kDeny;
    it->second.answered = true;
    if (allow && remember) {
      const std::string key = remember_key(it->second.info.tool, it->second.info.detail);
      if (std::find(remembered_.begin(), remembered_.end(), key) == remembered_.end())
        remembered_.push_back(key);
    }
  }
  cv_.notify_all();
}

void ApprovalGate::deny_all() {
  {
    std::lock_guard<std::mutex> g(m_);
    for (auto& kv : waiting_) {
      kv.second.decision = ToolDecision::kDeny;
      kv.second.answered = true;
    }
  }
  cv_.notify_all();
}

size_t ApprovalGate::pending_count() const {
  std::lock_guard<std::mutex> g(m_);
  size_t n = 0;
  for (const auto& kv : waiting_)
    if (!kv.second.answered) ++n;
  return n;
}

void ApprovalGate::forget_all() {
  std::lock_guard<std::mutex> g(m_);
  remembered_.clear();
}

size_t ApprovalGate::remembered() const {
  std::lock_guard<std::mutex> g(m_);
  return remembered_.size();
}

// ------------------------------------------------------------------ registry
void ToolRegistry::add(ToolPtr t) {
  if (!t) return;
  const ToolSpec s = t->spec();
  std::lock_guard<std::mutex> g(m_);
  for (size_t i = 0; i < tools_.size(); ++i) {
    if (tools_[i]->spec().name == s.name) {  // re-registering replaces
      tools_[i] = t;
      enabled_[s.name] = s.enabled;
      return;
    }
  }
  tools_.push_back(t);
  enabled_[s.name] = s.enabled;
  stats_[s.name].name = s.name;
}

std::vector<ToolSpec> ToolRegistry::specs() const {
  std::lock_guard<std::mutex> g(m_);
  std::vector<ToolSpec> out;
  out.reserve(tools_.size());
  for (const ToolPtr& t : tools_) {
    ToolSpec s = t->spec();
    const auto it = enabled_.find(s.name);
    s.enabled = (it == enabled_.end()) ? s.enabled : it->second;
    out.push_back(std::move(s));
  }
  return out;
}

bool ToolRegistry::set_enabled(const std::string& name, bool on) {
  std::lock_guard<std::mutex> g(m_);
  const auto it = enabled_.find(name);
  if (it == enabled_.end()) return false;
  it->second = on;
  return true;
}

bool ToolRegistry::enabled(const std::string& name) const {
  std::lock_guard<std::mutex> g(m_);
  const auto it = enabled_.find(name);
  return it != enabled_.end() && it->second;
}

ToolPtr ToolRegistry::get(const std::string& name) const {
  std::lock_guard<std::mutex> g(m_);
  for (const ToolPtr& t : tools_)
    if (t->spec().name == name) return t;
  return nullptr;
}

std::string ToolRegistry::catalogue(bool compact) const {
  const std::vector<ToolSpec> all = specs();
  std::string out;
  if (!compact)
    out +=
        "Tools. Call one per reply:\n[[tool:NAME]]\nkey: value\n[[/tool]]\n"
        "Wait for the result before answering.\n\n";
  for (const ToolSpec& s : all) {
    if (!s.enabled) continue;
    std::string params;
    for (const ToolParam& p : s.params) {
      if (!params.empty()) params += ", ";
      params += p.name;
      if (!p.required) params += "?";
    }
    if (compact) {
      // One line per tool: a 30 M model's context is measured in hundreds of
      // tokens, so the catalogue cannot afford examples.
      out += s.name + "(" + params + ") - " + s.summary + "\n";
    } else {
      out += "* " + s.name + "(" + params + ")  [" + tool_risk_name(s.risk) + "]\n  " +
             s.summary + "\n";
      if (!s.usage.empty()) out += "  " + s.usage + "\n";
    }
  }
  return out;
}

std::vector<ToolCall> ToolRegistry::parse(const std::string& reply) {
  std::vector<ToolCall> calls;
  size_t i = 0;
  while (i < reply.size()) {
    size_t name_start = 0;
    if (tool_marker_at(reply, i, &name_start)) {
      size_t j = name_start;
      while (j < reply.size() && is_name_char(reply[j])) ++j;
      const std::string name = reply.substr(name_start, j - name_start);
      const size_t hdr_end = reply.find("]]", j);
      if (name.empty() || hdr_end == std::string::npos) {  // junk; move past "[["
        i += 2;
        continue;
      }
      ToolCall call;
      call.name = lower(name);
      parse_inline_args(reply.substr(j, hdr_end - j), &call.args);
      const bool inline_args = !call.args.empty();

      size_t body_start = hdr_end + 2, end = body_start;
      size_t close_after = 0;
      const size_t close = find_close_marker(reply, body_start, &close_after);
      size_t next = std::string::npos;
      for (size_t p = body_start; p + 1 < reply.size(); ++p) {
        size_t dummy = 0;
        if (tool_marker_at(reply, p, &dummy)) {
          next = p;
          break;
        }
      }
      const bool has_body = (close != std::string::npos &&
                            (next == std::string::npos || close < next)) ||
                           !inline_args;
      if (has_body) {
        // An unterminated block ends at the next call or at end of string: a
        // model that forgets [[/tool]] still gets its call executed.
        const size_t stop = (close != std::string::npos &&
                             (next == std::string::npos || close < next))
                                ? close
                                : (next == std::string::npos ? reply.size() : next);
        parse_block_args(reply.substr(body_start, stop - body_start), &call.args);
        end = (close != std::string::npos && stop == close) ? close_after : stop;
      } else {
        end = hdr_end + 2;
      }
      call.args.erase("_");  // only meaningful when it is the sole argument
      if (call.args.empty()) parse_inline_args(reply.substr(j, hdr_end - j), &call.args);
      call.raw = reply.substr(i, end - i);
      calls.push_back(std::move(call));
      i = end;
      continue;
    }
    if (reply[i] == '{') {
      std::vector<std::pair<std::string, std::string>> kv;
      size_t end = i + 1;
      if (json_object(reply, i, &end, &kv, 0)) {
        std::string name;
        for (const auto& p : kv)
          if (p.first == "tool" || p.first == "name" || p.first == "tool_name" ||
              p.first == "function")
            name = p.second;
        if (!name.empty()) {
          ToolCall call;
          call.name = lower(trim(name));
          for (const auto& p : kv) {
            if (p.first == "tool" || p.first == "name" || p.first == "tool_name" ||
                p.first == "function" || p.first == "id")
              continue;
            call.args[p.first] = p.second;
          }
          call.raw = reply.substr(i, std::min(end, reply.size()) - i);
          calls.push_back(std::move(call));
        }
      }
      i = std::max(end, i + 1);
      continue;
    }
    ++i;
  }
  for (size_t n = 0; n < calls.size(); ++n) calls[n].id = static_cast<uint64_t>(n + 1);
  return calls;
}

bool ToolRegistry::looks_like_broken_call(const std::string& reply) {
  const bool marker = reply.find("[[tool") != std::string::npos ||
                      reply.find("[[ tool") != std::string::npos ||
                      reply.find("\"tool\":") != std::string::npos ||
                      reply.find("\"tool\" :") != std::string::npos ||
                      reply.find("<tool") != std::string::npos;
  return marker && parse(reply).empty();
}

ToolResult ToolRegistry::invoke(const ToolCall& call, ToolContext& ctx,
                                ApprovalGate* gate, const ToolPolicy& policy) {
  const double t0 = Telemetry::now();
  ToolResult r;
  const std::string canon = canonical_args(call);

  // Every exit path below goes through this, so "what did it run and why" is
  // always one grep in the same JSONL the trainers write.
  const auto audit = [&](const char* level, const char* decision) {
    if (!ctx.tel) return;
    ctx.tel->log(level, "tool", call.name + ": " + decision,
                 {{"tool", call.name},
                  {"args", utf8_truncate(canon, 400)},
                  {"decision", decision},
                  {"seconds", fmt(r.seconds, 3)},
                  {"bytes", std::to_string(static_cast<long long>(r.output.size()))},
                  {"ok", r.ok ? "1" : "0"}});
  };

  // Refusals are counted the same way wherever they happen, so "how often did
  // policy stop this tool" is answerable from stats() alone.
  const auto bump_denial = [this](const std::string& name, double seconds) {
    std::lock_guard<std::mutex> g(m_);
    Stat& st = stats_[name];
    st.name = name;
    ++st.denials;
    st.seconds += seconds;
  };

  ToolPtr t = get(call.name);
  if (!t) {
    std::string names;
    for (const ToolSpec& s : specs()) {
      if (!s.enabled) continue;
      if (!names.empty()) names += ", ";
      names += s.name;
    }
    r.error = "unknown tool '" + call.name + "'. Available: " + names;
    r.seconds = Telemetry::now() - t0;
    audit("warn", "unknown-tool");
    return r;
  }
  const ToolSpec spec = t->spec();
  if (!enabled(spec.name)) {
    r.error = "tool '" + spec.name + "' is disabled";
    r.denied = true;
    r.seconds = Telemetry::now() - t0;
    bump_denial(spec.name, r.seconds);
    audit("warn", "disabled");
    return r;
  }
  if (ctx.cancel && ctx.cancel->load()) {
    r.error = "cancelled before the tool ran";
    r.denied = true;
    r.seconds = Telemetry::now() - t0;
    audit("warn", "cancelled");
    return r;
  }

  const int mode = policy.mode_for(spec.risk);
  if (mode == 2) {
    r.error = "policy denies every " + std::string(tool_risk_name(spec.risk)) +
              " tool, so '" + spec.name + "' was not run";
    r.denied = true;
    r.seconds = Telemetry::now() - t0;
    bump_denial(spec.name, r.seconds);
    audit("warn", "policy-deny");
    return r;
  }
  if (mode == 0) {
    if (!gate) {
      r.error = "'" + spec.name + "' needs approval but no approval channel is "
                "connected";
      r.denied = true;
      r.seconds = Telemetry::now() - t0;
      bump_denial(spec.name, r.seconds);
      audit("warn", "no-gate");
      return r;
    }
    std::string prev;
    try {
      prev = t->preview(call);
    } catch (const std::exception&) {
      prev = call.name;
    }
    const ToolDecision d = gate->request(call, prev, spec.risk, policy);
    if (d != ToolDecision::kAllow) {
      r.denied = true;
      r.error = (d == ToolDecision::kTimeout)
                    ? "approval timed out after " + fmt(policy.timeout_s, 0) + "s"
                    : "the user denied '" + spec.name + "'";
      r.seconds = Telemetry::now() - t0;
      bump_denial(spec.name, r.seconds);
      audit("warn", d == ToolDecision::kTimeout ? "timeout" : "user-deny");
      return r;
    }
  }

  // Tools are ordinary C++ and may throw (std::filesystem does); the contract of
  // this boundary is that errors come back in the struct.
  try {
    r = t->run(call, ctx);
  } catch (const std::exception& e) {
    r = ToolResult();
    r.error = std::string("tool threw: ") + e.what();
  } catch (...) {
    r = ToolResult();
    r.error = "tool threw an unknown exception";
  }
  r.seconds = Telemetry::now() - t0;

  const size_t budget = ctx.output_budget > 0 ? static_cast<size_t>(ctx.output_budget)
                                              : r.output.size();
  r.bytes_before_truncation = static_cast<int64_t>(r.output.size());
  if (r.output.size() > budget) {
    const int64_t total = static_cast<int64_t>(r.output.size());
    r.output = utf8_truncate(r.output, budget);
    r.output += "\n[... truncated, " + std::to_string(static_cast<long long>(total)) +
                " bytes total]";
    r.truncated = true;
  }

  {
    std::lock_guard<std::mutex> g(m_);
    Stat& st = stats_[spec.name];
    st.name = spec.name;
    ++st.calls;
    st.seconds += r.seconds;
    if (!r.ok) ++st.failures;
    if (r.denied) ++st.denials;
  }
  audit(r.ok ? "info" : "warn", r.ok ? "ran" : "failed");
  return r;
}

std::vector<ToolRegistry::Stat> ToolRegistry::stats() const {
  std::lock_guard<std::mutex> g(m_);
  std::vector<Stat> out;
  out.reserve(stats_.size());
  for (const auto& kv : stats_) out.push_back(kv.second);
  return out;
}

// ------------------------------------------------------------------ factories
void register_builtin_tools(ToolRegistry* reg, bool with_web, bool with_shell,
                            bool with_codebase) {
  if (!reg) return;
  reg->add(make_read_file_tool());
  reg->add(make_list_dir_tool());
  if (with_shell) {
    reg->add(make_write_file_tool());
    reg->add(make_shell_tool());
  }
  if (with_web && HttpClient::available()) {
    reg->add(make_web_search_tool());
    reg->add(make_web_fetch_tool());
  }
  if (with_codebase) {
    reg->add(make_code_search_tool());
    reg->add(make_find_symbol_tool());
    reg->add(make_repo_overview_tool());
  }
}

// ---------------------------------------------------------------- web ranking
namespace {

// Heuristic only, and deliberately tiny: primary reference/documentation hosts
// get a nudge up, hosts that mostly republish other people's answers get a nudge
// down.  It is a prior, not a filter - a penalised host still wins if it is the
// only relevant hit.
double domain_prior(const std::string& host) {
  static const char* good[] = {"docs.python.org", "developer.mozilla.org",
                               "cppreference.com", "wikipedia.org", "arxiv.org",
                               "man7.org", "pytorch.org", "kernel.org",
                               "rfc-editor.org", "w3.org", "huggingface.co",
                               "readthedocs.io", "gnu.org"};
  static const char* code[] = {"github.com", "gitlab.com", "stackoverflow.com",
                               "stackexchange.com", "sourceforge.net",
                               "bitbucket.org"};
  static const char* farm[] = {"w3schools.com", "tutorialspoint.com",
                               "javatpoint.com", "geeksforgeeks.org",
                               "pinterest.com", "quora.com", "scribd.com",
                               "coursehero.com", "blogspot.com", "medium.com"};
  for (const char* d : good)
    if (ends_with_host(host, d)) return 0.25;
  for (const char* d : code)
    if (ends_with_host(host, d)) return 0.20;
  for (const char* d : farm)
    if (ends_with_host(host, d)) return -0.30;
  return 0.0;
}

// Normalised for de-duplication only: tracking parameters and a trailing slash
// change the URL string without changing the page.
std::string normalise_url(const std::string& url) {
  std::string u = url;
  const size_t hash = u.find('#');
  if (hash != std::string::npos) u = u.substr(0, hash);
  const size_t q = u.find('?');
  std::string base = u.substr(0, q == std::string::npos ? u.size() : q);
  std::string keep;
  if (q != std::string::npos) {
    const std::string qs = u.substr(q + 1);
    size_t i = 0;
    while (i < qs.size()) {
      const size_t amp = qs.find('&', i);
      const size_t stop = (amp == std::string::npos) ? qs.size() : amp;
      const std::string kvs = qs.substr(i, stop - i);
      i = (amp == std::string::npos) ? qs.size() : amp + 1;
      const std::string k = lower(kvs.substr(0, kvs.find('=')));
      const bool tracking = k.compare(0, 4, "utm_") == 0 || k == "fbclid" ||
                            k == "gclid" || k == "ref" || k == "ref_src" ||
                            k == "igshid";
      if (kvs.empty() || tracking) continue;
      keep += (keep.empty() ? "?" : "&") + kvs;
    }
  }
  // Scheme and host are case-insensitive; the path is not.
  const size_t slash = base.find('/', base.find("://") == std::string::npos
                                          ? 0
                                          : base.find("://") + 3);
  if (slash != std::string::npos) {
    base = lower(base.substr(0, slash)) + base.substr(slash);
  } else {
    base = lower(base);
  }
  if (base.size() > 1 && base.back() == '/') base.pop_back();
  const std::string www = "://www.";
  const size_t w = base.find(www);
  if (w != std::string::npos && w < 8) base = base.substr(0, w + 3) + base.substr(w + 7);
  return base + keep;
}

double jaccard(const std::unordered_set<std::string>& a,
               const std::unordered_set<std::string>& b) {
  if (a.empty() || b.empty()) return 0.0;
  size_t inter = 0;
  const auto& small = a.size() < b.size() ? a : b;
  const auto& big = a.size() < b.size() ? b : a;
  for (const std::string& w : small)
    if (big.count(w)) ++inter;
  const size_t uni = a.size() + b.size() - inter;
  return uni == 0 ? 0.0 : static_cast<double>(inter) / static_cast<double>(uni);
}

}  // namespace

std::vector<WebResult> rank_web_results(std::vector<WebResult> in,
                                        const std::string& query, size_t k) {
  std::vector<std::string> terms;
  for (const std::string& w : words_of(query))
    if (!is_stopword(w)) terms.push_back(w);
  if (terms.empty()) terms = words_of(query);

  // "the same host answered twice" is weak evidence that the host is the right
  // place to look, so it is counted before de-duplication removes the twin.
  std::unordered_map<std::string, int> host_count;
  for (WebResult& r : in) {
    if (r.host.empty()) r.host = HttpClient::host_of(r.url);
    if (!r.host.empty()) ++host_count[r.host];
  }

  std::vector<WebResult> uniq;
  std::unordered_set<std::string> seen;
  for (WebResult& r : in) {
    if (r.url.empty()) continue;
    if (!seen.insert(normalise_url(r.url)).second) continue;
    const std::string title = lower(r.title), snip = lower(r.snippet);
    double cover = 0.0;
    for (const std::string& t : terms) {
      double tf = 0.0;
      for (size_t p = title.find(t); p != std::string::npos; p = title.find(t, p + 1))
        tf += 2.0;  // a term in the title says more than one in the snippet
      for (size_t p = snip.find(t); p != std::string::npos; p = snip.find(t, p + 1))
        tf += 1.0;
      cover += tf / (tf + 1.2);  // BM25-style saturation: the 5th hit is noise
    }
    if (!terms.empty()) cover /= static_cast<double>(terms.size());
    r.score = cover + domain_prior(r.host);
    if (host_count[r.host] > 1) r.score += 0.05;
    uniq.push_back(r);
  }

  const auto by_score = [](const WebResult& a, const WebResult& b) {
    return a.score > b.score;
  };
  std::stable_sort(uniq.begin(), uniq.end(), by_score);

  std::vector<WebResult> kept, second;
  std::vector<std::unordered_set<std::string>> kept_tokens;
  for (WebResult& r : uniq) {
    const std::vector<std::string> ws = words_of(r.title + " " + r.snippet);
    std::unordered_set<std::string> tok(ws.begin(), ws.end());
    bool dup = false;
    for (const auto& prev : kept_tokens)
      if (jaccard(tok, prev) > 0.8) {
        dup = true;
        break;
      }
    if (dup) {
      r.score -= 0.5;  // penalised, not dropped: it may still beat the tail
      second.push_back(r);
      continue;
    }
    kept.push_back(r);
    kept_tokens.push_back(std::move(tok));
    if (kept.size() >= k) break;
  }
  for (WebResult& r : second) {
    if (kept.size() >= k) break;
    kept.push_back(r);
  }
  std::stable_sort(kept.begin(), kept.end(), by_score);
  if (kept.size() > k) kept.resize(k);
  return kept;
}

// -------------------------------------------------------------- html to text
namespace {

bool ci_equal(const std::string& a, const char* b) { return lower(a) == b; }

bool is_skip_tag(const std::string& n) {
  static const char* t[] = {"script", "style",  "head",     "nav",  "footer",
                            "aside",  "svg",    "noscript", "form", "iframe",
                            "template", "canvas"};
  for (const char* s : t)
    if (ci_equal(n, s)) return true;
  return false;
}

bool is_break_tag(const std::string& n) {
  static const char* t[] = {"br", "p",  "div", "li", "tr", "h1", "h2",   "h3",
                            "h4", "h5", "h6",  "hr", "ul", "ol", "table", "section",
                            "article", "blockquote", "pre", "td", "dd", "dt"};
  for (const char* s : t)
    if (ci_equal(n, s)) return true;
  return false;
}

// Case-insensitive find that does not copy the haystack: html_to_text calls it
// once per skipped element and pages are megabytes.
size_t find_ci(const std::string& hay, const std::string& needle, size_t from) {
  if (needle.empty() || from > hay.size()) return std::string::npos;
  const auto eq = [](char a, char b) {
    return std::tolower(static_cast<unsigned char>(a)) ==
           std::tolower(static_cast<unsigned char>(b));
  };
  const auto it = std::search(hay.begin() + static_cast<long>(from), hay.end(),
                              needle.begin(), needle.end(), eq);
  if (it == hay.end()) return std::string::npos;
  return static_cast<size_t>(it - hay.begin());
}

const std::unordered_map<std::string, const char*>& entities() {
  static const std::unordered_map<std::string, const char*> m = {
      {"amp", "&"},      {"lt", "<"},       {"gt", ">"},      {"quot", "\""},
      {"apos", "'"},     {"nbsp", " "},     {"mdash", "-"},   {"ndash", "-"},
      {"hellip", "..."}, {"rsquo", "'"},    {"lsquo", "'"},   {"ldquo", "\""},
      {"rdquo", "\""},   {"copy", "(c)"},   {"reg", "(R)"},   {"trade", "(TM)"},
      {"laquo", "\""},   {"raquo", "\""},   {"middot", "-"},  {"bull", "-"},
      {"deg", " deg"},   {"euro", "EUR"},   {"pound", "GBP"}, {"zwnj", "\xE2\x80\x8C"},
      {"times", "x"},    {"divide", "/"},   {"shy", ""},      {"#39", "'"}};
  return m;
}

// Returns true when an entity was consumed; *i then points past it.
bool decode_entity(const std::string& s, size_t* i, std::string* out) {
  const size_t semi = s.find(';', *i);
  if (semi == std::string::npos || semi - *i > 10) return false;
  const std::string body = s.substr(*i + 1, semi - *i - 1);
  if (body.empty()) return false;
  if (body[0] == '#') {
    uint32_t cp = 0;
    if (body.size() > 2 && (body[1] == 'x' || body[1] == 'X'))
      cp = static_cast<uint32_t>(std::strtoul(body.c_str() + 2, nullptr, 16));
    else
      cp = static_cast<uint32_t>(std::strtoul(body.c_str() + 1, nullptr, 10));
    if (cp == 0 || cp > 0x10FFFF) return false;
    utf8_append(cp, out);
    *i = semi + 1;
    return true;
  }
  const auto it = entities().find(lower(body));
  if (it == entities().end()) return false;
  *out += it->second;
  *i = semi + 1;
  return true;
}

// Collapses whitespace: runs of spaces to one, runs of blank lines to at most
// two newlines, trailing spaces per line dropped.
std::string tidy(const std::string& raw) {
  std::string out;
  out.reserve(raw.size());
  int newlines = 0;
  bool space_pending = false;
  for (size_t i = 0; i < raw.size(); ++i) {
    const char c = raw[i];
    if (c == '\n' || c == '\r') {
      if (c == '\r' && i + 1 < raw.size() && raw[i + 1] == '\n') continue;
      space_pending = false;
      if (newlines < 2) {
        while (!out.empty() && (out.back() == ' ' || out.back() == '\t')) out.pop_back();
        out += '\n';
        ++newlines;
      }
      continue;
    }
    if (c == ' ' || c == '\t' || c == '\f' || c == '\v') {
      if (!out.empty() && newlines == 0) space_pending = true;
      continue;
    }
    if (space_pending) out += ' ';
    space_pending = false;
    newlines = 0;
    out += c;
  }
  return trim(out);
}

}  // namespace

std::string html_to_text(const std::string& html, std::string* title) {
  if (title) title->clear();
  if (title) {
    // Pulled out first: <head> is dropped wholesale below.
    const size_t t0 = find_ci(html, "<title", 0);
    if (t0 != std::string::npos) {
      const size_t gt = html.find('>', t0);
      const size_t t1 = gt == std::string::npos ? std::string::npos
                                                : find_ci(html, "</title", gt);
      if (gt != std::string::npos) {
        const std::string inner =
            html.substr(gt + 1, (t1 == std::string::npos ? html.size() : t1) - gt - 1);
        std::string dec;
        for (size_t i = 0; i < inner.size();) {
          if (inner[i] == '&' && decode_entity(inner, &i, &dec)) continue;
          dec += inner[i++];
        }
        *title = tidy(dec);
      }
    }
  }

  std::string raw;
  raw.reserve(html.size() / 2);
  size_t i = 0;
  while (i < html.size()) {
    const char c = html[i];
    if (c == '<') {
      // Comments and doctype/CDATA are skipped as a block.
      if (html.compare(i, 4, "<!--") == 0) {
        const size_t e = html.find("-->", i + 4);
        i = (e == std::string::npos) ? html.size() : e + 3;
        continue;
      }
      size_t j = i + 1;
      const bool closing = j < html.size() && html[j] == '/';
      if (closing) ++j;
      const size_t n0 = j;
      while (j < html.size() && (std::isalnum(static_cast<unsigned char>(html[j])) ||
                                 html[j] == '!'))
        ++j;
      const std::string name = html.substr(n0, j - n0);
      if (!closing && is_skip_tag(name)) {
        // Drop the whole element; an unterminated one drops the rest of the
        // document, which is the safe direction (never leak script source).
        const size_t close = find_ci(html, "</" + lower(name), j);
        if (close == std::string::npos) break;
        const size_t gt = html.find('>', close);
        i = (gt == std::string::npos) ? html.size() : gt + 1;
        raw += '\n';
        continue;
      }
      const size_t gt = html.find('>', i + 1);
      if (gt == std::string::npos) break;  // unterminated tag: drop the tail
      if (is_break_tag(name)) raw += '\n';
      i = gt + 1;
      continue;
    }
    if (c == '&' && decode_entity(html, &i, &raw)) continue;
    raw += c;
    ++i;
  }
  return tidy(raw);
}

// ------------------------------------------------------------ summarisation
namespace {

struct Sentence {
  size_t index = 0;
  std::string text;
  double score = 0.0;
};

bool is_ascii_digit(const std::string& s, size_t pos) {
  return pos < s.size() && s[pos] >= '0' && s[pos] <= '9';
}

std::vector<Sentence> split_sentences(const std::string& text) {
  std::vector<Sentence> out;
  std::string cur;
  size_t i = 0;
  while (i < text.size()) {
    const size_t start = i;
    const uint32_t cp = utf8_next(text, &i);
    cur.append(text, start, i - start);
    bool cut = false;
    if (cp == '.' || cp == '!' || cp == '?' || cp == 0x061F /* ؟ */ ||
        cp == 0x06D4 /* ۔ */) {
      // "3.14" and "v1.2" are not sentence ends.
      const bool numeric = cp == '.' && start > 0 && is_ascii_digit(text, start - 1) &&
                           is_ascii_digit(text, i);
      cut = !numeric;
    } else if (cp == '\n') {
      cut = true;
    } else if (cp == 0x060C /* ، */ && cur.size() > 240) {
      // Persian prose often runs on with commas; splitting only very long
      // segments keeps sentences useful without shredding normal text.
      cut = true;
    }
    if (cut) {
      const std::string t = trim(cur);
      if (!t.empty()) out.push_back({out.size(), t, 0.0});
      cur.clear();
    }
  }
  const std::string t = trim(cur);
  if (!t.empty()) out.push_back({out.size(), t, 0.0});
  return out;
}

}  // namespace

std::string summarise_extractive(const std::string& text, const std::string& query,
                                 size_t budget_chars) {
  if (budget_chars == 0) return std::string();
  const std::string clean = utf8_sanitize(text);
  if (clean.size() <= budget_chars) return clean;

  std::vector<Sentence> sents = split_sentences(clean);
  if (sents.empty()) return utf8_truncate(clean, budget_chars);

  std::vector<std::string> qterms;
  for (const std::string& w : words_of(query))
    if (!is_stopword(w)) qterms.push_back(w);

  // Document frequency over sentences: a word that appears everywhere carries no
  // information about which sentence to keep.
  std::unordered_map<std::string, int> df;
  std::vector<std::vector<std::string>> toks(sents.size());
  for (size_t s = 0; s < sents.size(); ++s) {
    toks[s] = words_of(sents[s].text);
    std::unordered_set<std::string> uniq(toks[s].begin(), toks[s].end());
    for (const std::string& w : uniq) ++df[w];
  }
  const double N = static_cast<double>(sents.size());
  for (size_t s = 0; s < sents.size(); ++s) {
    double q = 0.0, idf = 0.0;
    std::unordered_set<std::string> uniq(toks[s].begin(), toks[s].end());
    for (const std::string& w : uniq) {
      for (const std::string& t : qterms)
        if (w == t) q += 3.0;
      if (!is_stopword(w)) idf += std::log(1.0 + N / (1.0 + df[w]));
    }
    // Normalised by length so a long sentence does not win on volume alone.
    const double len = std::sqrt(static_cast<double>(std::max<size_t>(1, uniq.size())));
    sents[s].score = q + idf / len + (s < 2 ? 0.5 : 0.0);
  }

  std::vector<size_t> order(sents.size());
  for (size_t s = 0; s < order.size(); ++s) order[s] = s;
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return sents[a].score > sents[b].score; });

  std::vector<bool> take(sents.size(), false);
  size_t used = 0;
  for (size_t idx : order) {
    const size_t cost = sents[idx].text.size() + 1;
    if (used + cost > budget_chars) {
      if (used > 0) continue;  // keep looking for a sentence that fits
      take[idx] = true;
      used += cost;
      break;
    }
    take[idx] = true;
    used += cost;
  }

  std::string out;
  for (size_t s = 0; s < sents.size(); ++s) {
    if (!take[s]) continue;
    if (!out.empty()) out += " ";
    out += sents[s].text;
  }
  return utf8_truncate(out, budget_chars);
}

}  // namespace slm
