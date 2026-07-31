// SPDX-License-Identifier: Apache-2.0
//
// Minimal HTTP(S) client for the web tools.
//
// Two implementations, chosen at build time and reported at run time:
//   * libcurl, when the headers were found (SLM_WITH_CURL);
//   * otherwise the `curl` binary, spawned with an argv array (never a shell
//     string, so a URL can never be interpreted as a command).
//
// The fallback exists because a package that hard-depends on libcurl-dev is
// annoying to install, while /usr/bin/curl is present on essentially every
// machine.  Both paths enforce the same limits: total timeout, maximum response
// size, redirect cap, and no protocol other than http/https.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace slm {

struct HttpRequest {
  std::string url;
  std::string method = "GET";
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;
  double timeout_s = 20.0;
  int64_t max_bytes = 4 * 1024 * 1024;
  int max_redirects = 5;
  std::string user_agent =
      "Mozilla/5.0 (X11; Linux x86_64) slm-agent/0.4 (+https://github.com/Im-Saleh/My-LLM)";
  bool accept_html = true;
};

struct HttpResponse {
  long status = 0;
  std::string body;
  std::string content_type;
  std::string final_url;
  std::string error;
  double seconds = 0.0;
  bool truncated = false;
  bool ok() const { return error.empty() && status >= 200 && status < 300; }
};

class HttpClient {
 public:
  static bool available();
  static const char* backend_name();  // "libcurl" | "curl(1)" | "none"

  static HttpResponse fetch(const HttpRequest& req);
  static HttpResponse get(const std::string& url, double timeout_s = 20.0);

  // Percent-encodes everything outside the unreserved set.
  static std::string url_encode(const std::string& s);
  // Rejects anything that is not http/https, or that contains control
  // characters - the only URL validation the tools need.
  static bool url_is_safe(const std::string& url, std::string* why = nullptr);
  static std::string host_of(const std::string& url);
};

}  // namespace slm
