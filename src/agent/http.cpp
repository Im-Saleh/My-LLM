// SPDX-License-Identifier: Apache-2.0
#include "agent/http.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#ifdef SLM_WITH_CURL
#include <curl/curl.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <vector>
#endif

namespace slm {
namespace {

double mono_now() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
}

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool starts_with_ci(const std::string& s, const char* prefix) {
  const size_t n = std::strlen(prefix);
  if (s.size() < n) return false;
  for (size_t i = 0; i < n; ++i) {
    const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
    if (a != prefix[i]) return false;
  }
  return true;
}

// One Accept header for both backends, so a server cannot answer differently
// depending on which one happened to fetch the page.
const char* const kAcceptHeader =
    "Accept: text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8";

// The scheme check is the security boundary of this file: everything downstream
// (libcurl protocol restrictions, the argv-only fallback) assumes the URL is
// already known to be an http(s) URL and nothing else.
bool scheme_ok(const std::string& url) {
  return starts_with_ci(url, "http://") || starts_with_ci(url, "https://");
}

std::string trim_ws(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
  return s.substr(a, b - a);
}

// A header value with a newline in it could inject extra headers, and a header
// without a colon confuses curl(1); both are dropped rather than repaired.
bool header_is_sane(const std::pair<std::string, std::string>& h) {
  if (h.first.empty()) return false;
  for (const std::string* s : {&h.first, &h.second})
    for (char c : *s)
      if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) return false;
  return h.first.find(':') == std::string::npos;
}

#ifndef SLM_WITH_CURL
// --------------------------------------------------------------- curl(1) path
// Locating the binary is cached: the tools call fetch() in a loop over search
// engines and stat()ing four paths every time is pointless.
const std::string& curl_binary() {
  static const std::string path = [] {
    for (const char* p : {"/usr/bin/curl", "/bin/curl", "/usr/local/bin/curl"})
      if (::access(p, X_OK) == 0) return std::string(p);
    // Fall back to a PATH search, still by absolute path so execvp cannot be
    // tricked by a relative PATH entry.
    const char* env = std::getenv("PATH");
    if (!env) return std::string();
    std::string acc;
    for (const char* c = env;; ++c) {
      if (*c == ':' || *c == '\0') {
        if (!acc.empty() && acc[0] == '/') {
          const std::string cand = acc + "/curl";
          if (::access(cand.c_str(), X_OK) == 0) return cand;
        }
        acc.clear();
        if (*c == '\0') break;
      } else {
        acc += *c;
      }
    }
    return std::string();
  }();
  return path;
}

// Response bodies go to a file instead of the pipe so that the -w trailer (the
// only place the status code comes from) can never be pushed out of reach by a
// large body or by the max_bytes cap.
struct TempFile {
  std::string path;
  bool ok = false;
  explicit TempFile(const char* tag) {
    std::string t = std::string("/tmp/slm-http-") + tag + "-XXXXXX";
    std::vector<char> buf(t.begin(), t.end());
    buf.push_back('\0');
    const int fd = ::mkstemp(buf.data());
    if (fd >= 0) {
      ::close(fd);
      path.assign(buf.data());
      ok = true;
    }
  }
  ~TempFile() {
    if (ok) ::unlink(path.c_str());
  }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
};

std::string read_capped(const std::string& path, int64_t max_bytes, bool* truncated) {
  std::string out;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return out;
  char buf[64 * 1024];
  while (out.size() < static_cast<size_t>(max_bytes)) {
    const size_t want =
        std::min(sizeof(buf), static_cast<size_t>(max_bytes) - out.size());
    const size_t got = std::fread(buf, 1, want, f);
    if (got == 0) break;
    out.append(buf, got);
  }
  if (out.size() >= static_cast<size_t>(max_bytes) && std::fgetc(f) != EOF)
    *truncated = true;
  std::fclose(f);
  return out;
}

HttpResponse fetch_via_binary(const HttpRequest& req) {
  HttpResponse r;
  const double t0 = mono_now();
  const std::string& bin = curl_binary();
  if (bin.empty()) {
    r.error = "no HTTP backend: neither libcurl nor a curl binary is available";
    return r;
  }
  TempFile body("body"), post("post");
  if (!body.ok) {
    r.error = "cannot create a temporary file for the response body";
    return r;
  }

  const double timeout = req.timeout_s > 0.5 ? req.timeout_s : 0.5;
  char tbuf[32], rbuf[32];
  std::snprintf(tbuf, sizeof(tbuf), "%.0f", timeout < 1.0 ? 1.0 : timeout);
  std::snprintf(rbuf, sizeof(rbuf), "%d", req.max_redirects < 0 ? 0 : req.max_redirects);

  // Owned storage for the argv strings; argv itself is built afterwards so the
  // pointers stay valid (never a shell string, so a URL is never parsed as a
  // command).
  std::vector<std::string> a;
  a.push_back(bin);
  a.push_back("-sS");
  a.push_back("-L");
  a.push_back("--max-redirs");
  a.push_back(rbuf);
  a.push_back("--max-time");
  a.push_back(tbuf);
  a.push_back("--proto");
  a.push_back("=http,https");
  a.push_back("--proto-redir");
  a.push_back("=http,https");
  a.push_back("--compressed");
  a.push_back("-A");
  a.push_back(req.user_agent);
  if (req.accept_html) {
    a.push_back("-H");
    a.push_back(kAcceptHeader);
  }
  for (const auto& h : req.headers) {
    if (!header_is_sane(h)) continue;
    a.push_back("-H");
    a.push_back(h.first + ": " + h.second);
  }
  const std::string method = req.method.empty() ? "GET" : req.method;
  if (method != "GET") {
    a.push_back("-X");
    a.push_back(method);
  }
  if (!req.body.empty() && post.ok) {
    FILE* f = std::fopen(post.path.c_str(), "wb");
    if (f) {
      std::fwrite(req.body.data(), 1, req.body.size(), f);
      std::fclose(f);
    }
    a.push_back("--data-binary");
    a.push_back("@" + post.path);
  }
  a.push_back("-o");
  a.push_back(body.path);
  a.push_back("-w");
  a.push_back("%{http_code}\t%{content_type}\t%{url_effective}");
  a.push_back("--");
  a.push_back(req.url);

  std::vector<char*> argv;
  argv.reserve(a.size() + 1);
  for (std::string& s : a) argv.push_back(&s[0]);
  argv.push_back(nullptr);

  int fds[2];
  if (::pipe(fds) != 0) {
    r.error = "pipe() failed";
    return r;
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    r.error = "fork() failed";
    return r;
  }
  if (pid == 0) {
    ::setpgid(0, 0);  // so a timeout kill reaches curl's children too
    ::close(fds[0]);
    ::dup2(fds[1], STDOUT_FILENO);
    ::dup2(fds[1], STDERR_FILENO);
    ::close(fds[1]);
    ::execvp(argv[0], argv.data());
    ::_exit(127);
  }
  ::close(fds[1]);

  // The timeout is enforced here as well as with --max-time: curl can hang in
  // ways --max-time does not cover (DNS on some libc versions).
  const double deadline = mono_now() + timeout + 2.0;
  std::string trailer;
  bool killed = false;
  for (;;) {
    const double left = deadline - mono_now();
    if (left <= 0.0) {
      ::killpg(pid, SIGKILL);
      killed = true;
      break;
    }
    struct pollfd pf = {fds[0], POLLIN, 0};
    const int pr = ::poll(&pf, 1, static_cast<int>(left * 1000.0) + 1);
    if (pr < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (pr == 0) continue;
    char buf[8192];
    const ssize_t got = ::read(fds[0], buf, sizeof(buf));
    if (got <= 0) break;
    if (trailer.size() < 8192) trailer.append(buf, static_cast<size_t>(got));
  }
  ::close(fds[0]);
  int status = 0;
  ::waitpid(pid, &status, 0);
  r.seconds = mono_now() - t0;

  r.body = read_capped(body.path, req.max_bytes, &r.truncated);
  // The trailer is the last line; curl's own diagnostics land before it.
  const size_t nl = trailer.find_last_of('\n');
  std::string last = nl == std::string::npos ? trailer : trailer.substr(nl + 1);
  size_t t1 = last.find('\t');
  size_t t2 = t1 == std::string::npos ? t1 : last.find('\t', t1 + 1);
  if (t1 != std::string::npos && t2 != std::string::npos) {
    r.status = std::strtol(last.substr(0, t1).c_str(), nullptr, 10);
    r.content_type = lower(trim_ws(last.substr(t1 + 1, t2 - t1 - 1)));
    r.final_url = trim_ws(last.substr(t2 + 1));
  }
  if (killed) {
    r.error = "timeout after " + std::string(tbuf) + "s";
  } else if (r.status == 0) {
    // No status line means curl never got a response; its own message on stderr
    // is the only useful diagnostic.
    const std::string diag =
        (nl == std::string::npos) ? std::string() : trim_ws(trailer.substr(0, nl));
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    r.error = diag.empty() ? ("curl exited with status " + std::to_string(code)) : diag;
  }
  if (r.final_url.empty()) r.final_url = req.url;
  return r;
}
#endif  // !SLM_WITH_CURL

#ifdef SLM_WITH_CURL
// ---------------------------------------------------------------- libcurl path
bool curl_ready() {
  static const bool ok = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
  return ok;
}

struct Sink {
  std::string* body = nullptr;
  int64_t max = 0;
  bool truncated = false;
};

size_t on_write(char* ptr, size_t size, size_t nmemb, void* user) {
  Sink* s = static_cast<Sink*>(user);
  const size_t n = size * nmemb;
  const size_t room = s->body->size() >= static_cast<size_t>(s->max)
                          ? 0
                          : static_cast<size_t>(s->max) - s->body->size();
  if (n > room) {
    s->body->append(ptr, room);
    s->truncated = true;
    return 0;  // aborts the transfer; reported as CURLE_WRITE_ERROR
  }
  s->body->append(ptr, n);
  return n;
}

HttpResponse fetch_via_libcurl(const HttpRequest& req) {
  HttpResponse r;
  const double t0 = mono_now();
  if (!curl_ready()) {
    r.error = "libcurl global init failed";
    return r;
  }
  CURL* c = curl_easy_init();
  if (!c) {
    r.error = "curl_easy_init failed";
    return r;
  }
  Sink sink;
  sink.body = &r.body;
  sink.max = req.max_bytes > 0 ? req.max_bytes : 1;

  curl_easy_setopt(c, CURLOPT_URL, req.url.c_str());
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, on_write);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, &sink);
  curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(c, CURLOPT_MAXREDIRS, static_cast<long>(req.max_redirects));
  curl_easy_setopt(c, CURLOPT_TIMEOUT_MS,
                   static_cast<long>(req.timeout_s * 1000.0) + 1);
  curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS,
                   static_cast<long>(req.timeout_s * 1000.0 * 0.6) + 1);
  curl_easy_setopt(c, CURLOPT_USERAGENT, req.user_agent.c_str());
  curl_easy_setopt(c, CURLOPT_ACCEPT_ENCODING, "");  // gzip/deflate/br if built in
  curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);         // safe in a threaded process
  curl_easy_setopt(c, CURLOPT_FAILONERROR, 0L);
#if LIBCURL_VERSION_NUM >= 0x075500
  curl_easy_setopt(c, CURLOPT_PROTOCOLS_STR, "http,https");
  curl_easy_setopt(c, CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
#else
  curl_easy_setopt(c, CURLOPT_PROTOCOLS,
                   static_cast<long>(CURLPROTO_HTTP | CURLPROTO_HTTPS));
  curl_easy_setopt(c, CURLOPT_REDIR_PROTOCOLS,
                   static_cast<long>(CURLPROTO_HTTP | CURLPROTO_HTTPS));
#endif
  const std::string method = req.method.empty() ? "GET" : req.method;
  if (!req.body.empty()) {
    curl_easy_setopt(c, CURLOPT_POSTFIELDS, req.body.c_str());
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(req.body.size()));
  }
  if (method != "GET") curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, method.c_str());

  struct curl_slist* hs = nullptr;
  if (req.accept_html) hs = curl_slist_append(hs, kAcceptHeader);
  for (const auto& h : req.headers) {
    if (!header_is_sane(h)) continue;
    hs = curl_slist_append(hs, (h.first + ": " + h.second).c_str());
  }
  if (hs) curl_easy_setopt(c, CURLOPT_HTTPHEADER, hs);

  const CURLcode rc = curl_easy_perform(c);
  long code = 0;
  curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
  r.status = code;
  char* ct = nullptr;
  if (curl_easy_getinfo(c, CURLINFO_CONTENT_TYPE, &ct) == CURLE_OK && ct)
    r.content_type = lower(trim_ws(ct));
  char* eff = nullptr;
  if (curl_easy_getinfo(c, CURLINFO_EFFECTIVE_URL, &eff) == CURLE_OK && eff)
    r.final_url = eff;
  r.truncated = sink.truncated;
  // Hitting max_bytes aborts the write callback; that is a success with a short
  // body, not a transport failure.
  if (rc != CURLE_OK && !(rc == CURLE_WRITE_ERROR && sink.truncated))
    r.error = curl_easy_strerror(rc);

  if (hs) curl_slist_free_all(hs);
  curl_easy_cleanup(c);
  if (r.final_url.empty()) r.final_url = req.url;
  r.seconds = mono_now() - t0;
  return r;
}
#endif  // SLM_WITH_CURL

}  // namespace

bool HttpClient::available() {
#ifdef SLM_WITH_CURL
  return curl_ready();
#else
  return !curl_binary().empty();
#endif
}

const char* HttpClient::backend_name() {
#ifdef SLM_WITH_CURL
  return curl_ready() ? "libcurl" : "none";
#else
  return curl_binary().empty() ? "none" : "curl(1)";
#endif
}

HttpResponse HttpClient::fetch(const HttpRequest& req) {
  HttpResponse r;
  std::string why;
  if (!url_is_safe(req.url, &why)) {
    r.error = "refused URL: " + why;
    return r;
  }
  if (!available()) {
    r.error = "no HTTP backend available (install libcurl or the curl binary)";
    return r;
  }
#ifdef SLM_WITH_CURL
  return fetch_via_libcurl(req);
#else
  return fetch_via_binary(req);
#endif
}

HttpResponse HttpClient::get(const std::string& url, double timeout_s) {
  HttpRequest req;
  req.url = url;
  req.timeout_s = timeout_s;
  return fetch(req);
}

std::string HttpClient::url_encode(const std::string& s) {
  static const char* hex = "0123456789ABCDEF";
  std::string o;
  o.reserve(s.size() * 3 / 2);
  for (unsigned char c : s) {
    const bool unreserved = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                            (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                            c == '.' || c == '~';
    if (unreserved) {
      o += static_cast<char>(c);
    } else {
      o += '%';
      o += hex[c >> 4];
      o += hex[c & 0x0f];
    }
  }
  return o;
}

bool HttpClient::url_is_safe(const std::string& url, std::string* why) {
  const auto fail = [why](const char* msg) {
    if (why) *why = msg;
    return false;
  };
  if (url.empty()) return fail("empty URL");
  if (url.size() > 4096) return fail("URL longer than 4096 bytes");
  if (!scheme_ok(url)) return fail("only http:// and https:// URLs are allowed");
  for (unsigned char c : url) {
    if (c < 0x20 || c == 0x7f) return fail("URL contains a control character");
    if (c == ' ') return fail("URL contains a space");
  }
  const std::string host = host_of(url);
  if (host.empty()) return fail("URL has no host");
  if (why) why->clear();
  return true;
}

std::string HttpClient::host_of(const std::string& url) {
  const size_t s = url.find("://");
  if (s == std::string::npos) return std::string();
  size_t i = s + 3;
  size_t end = url.size();
  for (size_t j = i; j < url.size(); ++j)
    if (url[j] == '/' || url[j] == '?' || url[j] == '#') {
      end = j;
      break;
    }
  std::string auth = url.substr(i, end - i);
  const size_t at = auth.rfind('@');  // drop user:password@
  if (at != std::string::npos) auth = auth.substr(at + 1);
  if (!auth.empty() && auth[0] == '[') {  // IPv6 literal keeps its brackets
    const size_t rb = auth.find(']');
    if (rb != std::string::npos) return lower(auth.substr(0, rb + 1));
  }
  const size_t colon = auth.find(':');
  if (colon != std::string::npos) auth = auth.substr(0, colon);
  return lower(auth);
}

}  // namespace slm
