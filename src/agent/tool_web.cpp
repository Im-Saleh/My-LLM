// SPDX-License-Identifier: Apache-2.0
//
// The two network tools: web_search and web_fetch.
//
// Search has no single free API that is always available, so the tool walks a
// chain of engines and stops at the first one that answers: a self-hosted
// SearxNG (SLM_SEARCH_URL), Brave (SLM_BRAVE_KEY), the DuckDuckGo HTML
// endpoint, and finally the Wikipedia opensearch API.  The chain is what makes
// the tool useful with no keys configured at all.
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "agent/http.h"
#include "agent/tools.h"
#include "core/text.h"

namespace slm {
namespace {

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool contains(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
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

int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Needed to undo DuckDuckGo's redirect wrapper (?uddg=<percent-encoded url>).
std::string url_decode(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      const int hi = hexval(s[i + 1]), lo = hexval(s[i + 2]);
      if (hi >= 0 && lo >= 0) {
        o += static_cast<char>(hi * 16 + lo);
        i += 2;
        continue;
      }
    }
    o += (s[i] == '+') ? ' ' : s[i];
  }
  return o;
}

ToolResult fail(const std::string& msg) {
  ToolResult r;
  r.ok = false;
  r.error = msg;
  r.output = "error: " + msg;
  return r;
}

// --------------------------------------------------------------- mini JSON
// Deliberately minimal: the two JSON APIs used here return string fields inside
// a flat array of objects, and that is the only shape these helpers understand.
// Anything more (numbers, nesting, unicode escapes beyond one level) is either
// ignored or returned verbatim - a real parser would be a new dependency for no
// gain.
size_t skip_json_string(const std::string& s, size_t i) {
  if (i >= s.size() || s[i] != '"') return i;
  for (++i; i < s.size(); ++i) {
    if (s[i] == '\\') {
      ++i;
      continue;
    }
    if (s[i] == '"') return i + 1;
  }
  return s.size();
}

std::string json_unescape(const std::string& raw) {
  std::string o;
  for (size_t i = 0; i < raw.size(); ++i) {
    if (raw[i] != '\\' || i + 1 >= raw.size()) {
      o += raw[i];
      continue;
    }
    const char e = raw[++i];
    switch (e) {
      case 'n': o += '\n'; break;
      case 't': o += ' '; break;
      case 'r': break;
      case 'u': {
        uint32_t cp = 0;
        bool ok = i + 4 < raw.size();
        for (int k = 1; ok && k <= 4; ++k) {
          const int h = hexval(raw[i + static_cast<size_t>(k)]);
          if (h < 0) ok = false;
          else cp = cp * 16 + static_cast<uint32_t>(h);
        }
        if (ok) {
          // Surrogate halves cannot be decoded in isolation; '?' keeps the text
          // valid UTF-8, which the audit log depends on.
          if (cp >= 0xD800 && cp <= 0xDFFF) o += '?';
          else utf8_append(cp, &o);
          i += 4;
        } else {
          o += '?';
        }
        break;
      }
      default: o += e;
    }
  }
  return o;
}

// Returns the value position of "key": ... starting at `from`, or npos.
size_t json_value_pos(const std::string& s, const std::string& key, size_t from) {
  const std::string pat = "\"" + key + "\"";
  size_t p = from;
  while ((p = s.find(pat, p)) != std::string::npos) {
    size_t i = p + pat.size();
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) ++i;
    if (i < s.size() && s[i] == ':') {
      ++i;
      while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) ++i;
      return i;
    }
    p = i;
  }
  return std::string::npos;
}

std::string json_string_field(const std::string& s, const std::string& key) {
  const size_t v = json_value_pos(s, key, 0);
  if (v == std::string::npos || v >= s.size() || s[v] != '"') return std::string();
  const size_t e = skip_json_string(s, v);
  return json_unescape(s.substr(v + 1, e - v - 2));
}

// Splits the array that follows "key": [ ... ] into its top-level objects.
std::vector<std::string> json_objects_in_array(const std::string& s,
                                               const std::string& key) {
  std::vector<std::string> out;
  const size_t v = json_value_pos(s, key, 0);
  if (v == std::string::npos || v >= s.size() || s[v] != '[') return out;
  int depth = 0;
  size_t start = 0;
  for (size_t i = v + 1; i < s.size(); ++i) {
    if (s[i] == '"') {
      i = skip_json_string(s, i) - 1;
      continue;
    }
    if (s[i] == '{') {
      if (depth++ == 0) start = i;
    } else if (s[i] == '}') {
      if (--depth == 0) out.push_back(s.substr(start, i - start + 1));
    } else if (s[i] == ']' && depth == 0) {
      break;
    }
  }
  return out;
}

// Wikipedia's opensearch reply is an array of arrays of strings; this collects
// them in order.
std::vector<std::vector<std::string>> json_string_arrays(const std::string& s) {
  std::vector<std::vector<std::string>> out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '"') {
      i = skip_json_string(s, i) - 1;
      continue;
    }
    if (s[i] != '[') continue;
    if (i == 0 && out.empty()) continue;  // the outer array itself
    std::vector<std::string> items;
    size_t j = i + 1;
    bool nested = false;
    for (; j < s.size() && s[j] != ']'; ++j) {
      if (s[j] == '[') {
        nested = true;
        break;
      }
      if (s[j] != '"') continue;
      const size_t e = skip_json_string(s, j);
      items.push_back(json_unescape(s.substr(j + 1, e - j - 2)));
      j = e - 1;
    }
    if (nested) continue;
    out.push_back(items);
    i = j;
  }
  return out;
}

// ----------------------------------------------------------------- engines
struct EngineOut {
  std::vector<WebResult> results;
  std::string engine;
  std::string error;
};

void from_json_results(const std::string& body, const char* array_key,
                       const char* snippet_key, std::vector<WebResult>* out) {
  for (const std::string& obj : json_objects_in_array(body, array_key)) {
    WebResult r;
    r.title = trim(json_string_field(obj, "title"));
    r.url = trim(json_string_field(obj, "url"));
    r.snippet = trim(json_string_field(obj, snippet_key));
    if (r.snippet.empty()) r.snippet = trim(json_string_field(obj, "description"));
    if (r.url.empty() || r.url.compare(0, 4, "http") != 0) continue;
    r.host = HttpClient::host_of(r.url);
    // Snippets from these APIs contain <strong> highlighting.
    r.snippet = html_to_text(r.snippet, nullptr);
    r.title = html_to_text(r.title, nullptr);
    out->push_back(r);
  }
}

bool searx_engine(const std::string& query, EngineOut* eo) {
  const char* base = std::getenv("SLM_SEARCH_URL");
  if (!base || !*base) return false;
  std::string url = base;
  url += (url.find('?') == std::string::npos ? "?" : "&");
  url += "q=" + HttpClient::url_encode(query) + "&format=json";
  const HttpResponse resp = HttpClient::get(url, 15.0);
  eo->engine = "searxng (SLM_SEARCH_URL)";
  if (!resp.ok()) {
    eo->error = resp.error.empty() ? ("HTTP " + std::to_string(resp.status)) : resp.error;
    return false;
  }
  from_json_results(resp.body, "results", "content", &eo->results);
  return !eo->results.empty();
}

bool brave_engine(const std::string& query, size_t k, EngineOut* eo) {
  const char* key = std::getenv("SLM_BRAVE_KEY");
  if (!key || !*key) return false;
  HttpRequest req;
  req.url = "https://api.search.brave.com/res/v1/web/search?q=" +
            HttpClient::url_encode(query) + "&count=" + std::to_string(k);
  req.headers.push_back({"Accept", "application/json"});
  req.headers.push_back({"X-Subscription-Token", key});
  req.accept_html = false;
  req.timeout_s = 15.0;
  const HttpResponse resp = HttpClient::fetch(req);
  eo->engine = "brave search api";
  if (!resp.ok()) {
    eo->error = resp.error.empty() ? ("HTTP " + std::to_string(resp.status)) : resp.error;
    return false;
  }
  from_json_results(resp.body, "results", "description", &eo->results);
  return !eo->results.empty();
}

// DuckDuckGo's HTML endpoint needs no key and no JavaScript, which is why it is
// the default.  It also wraps every link in a redirect, so the real URL has to be
// pulled back out of the uddg parameter.
std::string ddg_real_url(std::string href) {
  const size_t u = href.find("uddg=");
  if (u != std::string::npos) {
    const size_t e = href.find('&', u);
    const size_t stop = (e == std::string::npos) ? href.size() : e;
    href = url_decode(href.substr(u + 5, stop - u - 5));
  } else if (href.compare(0, 2, "//") == 0) {
    href = "https:" + href;
  }
  return href;
}

std::string attr_value(const std::string& tag, const std::string& name) {
  const size_t p = tag.find(name + "=");
  if (p == std::string::npos) return std::string();
  size_t i = p + name.size() + 1;
  if (i < tag.size() && (tag[i] == '"' || tag[i] == '\'')) {
    const char q = tag[i++];
    const size_t e = tag.find(q, i);
    return tag.substr(i, (e == std::string::npos ? tag.size() : e) - i);
  }
  const size_t e = tag.find_first_of(" \t>", i);
  return tag.substr(i, (e == std::string::npos ? tag.size() : e) - i);
}

bool ddg_engine(const std::string& query, EngineOut* eo) {
  const std::string url =
      "https://html.duckduckgo.com/html/?q=" + HttpClient::url_encode(query);
  const HttpResponse resp = HttpClient::get(url, 20.0);
  eo->engine = "duckduckgo html";
  if (!resp.ok()) {
    eo->error = resp.error.empty() ? ("HTTP " + std::to_string(resp.status)) : resp.error;
    return false;
  }
  const std::string& h = resp.body;
  size_t pos = 0;
  while ((pos = h.find("result__a", pos)) != std::string::npos) {
    const size_t tag_start = h.rfind('<', pos);
    const size_t tag_end = h.find('>', pos);
    pos += 9;
    if (tag_start == std::string::npos || tag_end == std::string::npos) continue;
    const std::string tag = h.substr(tag_start, tag_end - tag_start + 1);
    WebResult r;
    r.url = ddg_real_url(attr_value(tag, "href"));
    if (r.url.compare(0, 4, "http") != 0) continue;
    const size_t a_end = h.find("</a", tag_end);
    const size_t title_end = (a_end == std::string::npos) ? tag_end + 1 : a_end;
    r.title = html_to_text(h.substr(tag_end + 1, title_end - tag_end - 1), nullptr);
    const size_t sn = h.find("result__snippet", tag_end);
    if (sn != std::string::npos) {
      const size_t sn_gt = h.find('>', sn);
      const size_t sn_end = sn_gt == std::string::npos ? std::string::npos
                                                       : h.find("</a", sn_gt);
      if (sn_gt != std::string::npos && sn_end != std::string::npos &&
          sn_end - sn_gt < 4000)
        r.snippet = html_to_text(h.substr(sn_gt + 1, sn_end - sn_gt - 1), nullptr);
    }
    r.host = HttpClient::host_of(r.url);
    eo->results.push_back(r);
    pos = (a_end == std::string::npos) ? pos : a_end;
  }
  return !eo->results.empty();
}

// Last resort: always up, always answers something, and for factual queries the
// answer is often the right one.
bool wikipedia_engine(const std::string& query, size_t k, EngineOut* eo) {
  const std::string url =
      "https://en.wikipedia.org/w/api.php?action=opensearch&format=json&limit=" +
      std::to_string(k) + "&namespace=0&search=" + HttpClient::url_encode(query);
  const HttpResponse resp = HttpClient::get(url, 15.0);
  eo->engine = "wikipedia opensearch";
  if (!resp.ok()) {
    eo->error = resp.error.empty() ? ("HTTP " + std::to_string(resp.status)) : resp.error;
    return false;
  }
  const std::vector<std::vector<std::string>> arrays = json_string_arrays(resp.body);
  if (arrays.empty()) return false;
  const std::vector<std::string>& titles = arrays[0];
  const std::vector<std::string>* descs = arrays.size() > 1 ? &arrays[1] : nullptr;
  const std::vector<std::string>* urls = arrays.size() > 2 ? &arrays[2] : nullptr;
  for (size_t i = 0; i < titles.size(); ++i) {
    WebResult r;
    r.title = titles[i];
    if (descs && i < descs->size()) r.snippet = (*descs)[i];
    if (urls && i < urls->size()) r.url = (*urls)[i];
    if (r.url.compare(0, 4, "http") != 0) continue;
    r.host = HttpClient::host_of(r.url);
    eo->results.push_back(r);
  }
  return !eo->results.empty();
}

class WebSearchTool : public Tool {
 public:
  ToolSpec spec() const override {
    ToolSpec s;
    s.name = "web_search";
    s.summary = "search the web and return ranked titles, URLs and snippets";
    s.usage = "[[tool:web_search]]\nquery: persian nlp datasets\n[[/tool]]";
    s.risk = ToolRisk::kNetwork;
    s.params = {{"query", "what to search for", true, ""},
                {"k", "how many results, 1..10", false, "5"}};
    return s;
  }

  ToolResult run(const ToolCall& call, ToolContext& ctx) override {
    const std::string query = trim(call.arg("query", call.arg("q", "")));
    if (query.empty()) return fail("missing 'query' argument");
    if (!HttpClient::available())
      return fail("no network: no HTTP backend (install libcurl or the curl binary)");
    int64_t k = call.arg_int("k", 5);
    if (k < 1) k = 1;
    if (k > 10) k = 10;
    const size_t want = static_cast<size_t>(k);

    // First engine that actually returns results wins; the others' failures are
    // reported only if every one of them fails.
    EngineOut eo;
    std::string tried;
    bool got = false;
    for (int stage = 0; stage < 4 && !got; ++stage) {
      EngineOut cur;
      switch (stage) {
        case 0: got = searx_engine(query, &cur); break;
        case 1: got = brave_engine(query, want, &cur); break;
        case 2: got = ddg_engine(query, &cur); break;
        default: got = wikipedia_engine(query, want, &cur); break;
      }
      if (!cur.engine.empty()) {
        if (!tried.empty()) tried += ", ";
        tried += cur.engine + (got ? "" : (cur.error.empty() ? ": no results"
                                                            : ": " + cur.error));
      }
      if (got) eo = cur;
      if (ctx.cancel && ctx.cancel->load()) break;
    }
    if (!got)
      return fail("no search engine answered (" +
                  (tried.empty() ? std::string("no engine configured") : tried) + ")");

    const std::vector<WebResult> ranked = rank_web_results(eo.results, query, want);
    if (ranked.empty()) return fail("no usable results for '" + query + "'");

    std::string out = "web_search \"" + query + "\" via " + eo.engine + " (" +
                      std::to_string(ranked.size()) + " results)\n";
    for (size_t i = 0; i < ranked.size(); ++i) {
      const WebResult& r = ranked[i];
      out += std::to_string(i + 1) + ". " +
             (r.title.empty() ? std::string("(untitled)") : r.title) + " - " + r.host +
             "\n   " + r.url + "\n";
      if (!r.snippet.empty()) out += "   " + utf8_truncate(r.snippet, 400) + "\n";
    }
    ToolResult res;
    res.ok = true;
    res.output = utf8_sanitize(out);
    res.display = res.output;
    return res;
  }
};

// ---------------------------------------------------------------- web_fetch
bool textual_content_type(const std::string& ct) {
  if (ct.empty()) return true;  // servers that omit it are usually serving HTML
  return ct.compare(0, 5, "text/") == 0 || contains(ct, "json") ||
         contains(ct, "xml") || contains(ct, "javascript") ||
         contains(ct, "x-www-form-urlencoded");
}

class WebFetchTool : public Tool {
 public:
  ToolSpec spec() const override {
    ToolSpec s;
    s.name = "web_fetch";
    s.summary = "fetch one URL and return its readable text, compressed to a budget";
    s.usage = "[[tool:web_fetch url=\"https://example.com/page\"]]";
    s.risk = ToolRisk::kNetwork;
    s.params = {{"url", "http:// or https:// URL", true, ""},
                {"budget", "characters of text to return, max 20000", false, "4000"},
                {"query", "what to focus the summary on", false, ""}};
    return s;
  }

  ToolResult run(const ToolCall& call, ToolContext& ctx) override {
    const std::string url = trim(call.arg("url", call.arg("link", "")));
    if (url.empty()) return fail("missing 'url' argument");
    std::string why;
    if (!HttpClient::url_is_safe(url, &why)) return fail(why);
    if (!HttpClient::available())
      return fail("no network: no HTTP backend (install libcurl or the curl binary)");
    int64_t budget = call.arg_int("budget", 4000);
    if (budget < 200) budget = 200;
    if (budget > 20000) budget = 20000;

    HttpRequest req;
    req.url = url;
    req.timeout_s = 20.0;
    req.max_bytes = 3 * 1024 * 1024;  // enough for any article, not for a tarball
    const HttpResponse resp = HttpClient::fetch(req);
    if (!resp.error.empty()) return fail("fetch failed: " + resp.error);
    if (resp.status >= 400)
      return fail("HTTP " + std::to_string(resp.status) + " for " + resp.final_url);
    const std::string ct = lower(resp.content_type);
    if (!textual_content_type(ct))
      return fail("refusing content type '" + ct + "': web_fetch returns text only");

    std::string title, text;
    const bool html = contains(ct, "html") || contains(ct, "xml") ||
                      (ct.empty() && resp.body.find("<html") != std::string::npos);
    if (html) {
      text = html_to_text(resp.body, &title);
    } else {
      text = utf8_sanitize(resp.body);
    }
    if (trim(text).empty())
      return fail("nothing readable at " + resp.final_url + " (" +
                  std::to_string(resp.body.size()) + " bytes of " +
                  (ct.empty() ? "unknown type" : ct) + ")");

    const std::string focus = call.arg("query", "");
    const std::string body =
        summarise_extractive(text, focus, static_cast<size_t>(budget));

    ToolResult r;
    r.ok = true;
    r.output = (title.empty() ? std::string("(untitled)") : title) + "\n" +
               resp.final_url + "\n";
    if (resp.truncated) r.output += "[body capped while downloading]\n";
    r.output += "\n" + body + "\n";
    r.output = utf8_sanitize(r.output);
    r.display = r.output;
    r.bytes_before_truncation = static_cast<int64_t>(text.size());
    r.truncated = text.size() > body.size();
    (void)ctx;
    return r;
  }
};

}  // namespace

ToolPtr make_web_search_tool() { return std::make_shared<WebSearchTool>(); }
ToolPtr make_web_fetch_tool() { return std::make_shared<WebFetchTool>(); }

}  // namespace slm
