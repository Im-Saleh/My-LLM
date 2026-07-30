// SPDX-License-Identifier: Apache-2.0
//
// Dependency-free ANSI dashboard.  Same data as the ImGui one; usable over SSH,
// inside containers and in CI (that is also how the soak test runs).
#include <sys/select.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "gui.h"

namespace slm {
namespace {

const char* kReset = "\033[0m";
const char* kBold = "\033[1m";
const char* kDim = "\033[2m";
const char* kRed = "\033[31m";
const char* kGreen = "\033[32m";
const char* kYellow = "\033[33m";
const char* kBlue = "\033[36m";
const char* kMagenta = "\033[35m";

const char* stream_color(Stream s) {
  switch (s) {
    case Stream::kContinual:
      return kBlue;
    case Stream::kSelfGen:
      return kYellow;
    case Stream::kFeedback:
      return kMagenta;
    default:
      return kGreen;
  }
}

std::string sparkline(const std::vector<LossPoint>& pts, size_t width) {
  static const char* blocks[8] = {"\u2581", "\u2582", "\u2583", "\u2584",
                                  "\u2585", "\u2586", "\u2587", "\u2588"};
  if (pts.empty()) return std::string(width, ' ');
  const size_t n = std::min(width, pts.size());
  std::vector<float> v;
  v.reserve(n);
  for (size_t i = pts.size() - n; i < pts.size(); ++i) v.push_back(pts[i].value);
  float lo = v[0], hi = v[0];
  for (float x : v) {
    lo = std::min(lo, x);
    hi = std::max(hi, x);
  }
  std::string out;
  for (float x : v) {
    const float t = (hi > lo) ? (x - lo) / (hi - lo) : 0.5f;
    out += blocks[std::min(7, static_cast<int>(t * 7.999f))];
  }
  if (n < width) out += std::string(width - n, ' ');
  return out;
}

std::string trunc(const std::string& s, size_t n) {
  std::string t;
  // keep it byte-safe for UTF-8 by cutting at n bytes and dropping partials
  size_t i = 0;
  while (i < s.size() && i < n) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    size_t len = 1;
    if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xF8) == 0xF0) len = 4;
    if (i + len > n || i + len > s.size()) break;
    for (size_t k = 0; k < len; ++k) t += s[i + k];
    i += len;
  }
  if (i < s.size()) t += "...";
  for (char& ch : t)
    if (ch == '\n' || ch == '\r') ch = ' ';
  return t;
}

void handle_command(DashboardContext& ctx, const std::string& line) {
  Telemetry& tel = *ctx.tel;
  const char cmd = line[0];
  const std::string rest = line.size() > 2 ? line.substr(2) : std::string();
  if (cmd == 'q') {
    ctx.quit->store(true);
  } else if (cmd == 'x') {
    if (tel.stopped())
      tel.clear_emergency_stop();
    else
      tel.emergency_stop();
  } else if (cmd >= '1' && cmd <= '3') {
    const Source s = static_cast<Source>(cmd - '1');
    tel.set_trainer_enabled(s, !tel.trainer_enabled(s));
    tel.log("info", "control",
            std::string(tel.trainer_enabled(s) ? "enabled " : "disabled ") + source_name(s));
  } else if (cmd == 'b') {
    ctx.coord->restore_best();
  } else if (cmd == 'a') {
    ctx.chat->ask(rest);
  } else if (cmd == 'r') {
    std::vector<ChatTurn> h = ctx.chat->history();
    if (!h.empty()) ctx.chat->rate(h.size() - 1, static_cast<float>(std::atof(rest.c_str())));
  }
}

}  // namespace

int run_terminal_dashboard(DashboardContext& ctx) {
  Telemetry& tel = *ctx.tel;
  Coordinator& coord = *ctx.coord;

  // Keyboard commands.  stdin is polled with a timeout instead of a blocking
  // getline: a blocking read in a detached thread can deadlock the process at
  // exit (the stream lock is held while the runtime tears the iostreams down),
  // and this way the thread is joinable and always terminates.
  std::thread input([&] {
    std::string line;
    while (!ctx.quit->load()) {
      fd_set rd;
      FD_ZERO(&rd);
      FD_SET(STDIN_FILENO, &rd);
      struct timeval tv{0, 200000};  // 200 ms
      const int r = select(STDIN_FILENO + 1, &rd, nullptr, nullptr, &tv);
      if (r <= 0) continue;
      char buf[512];
      const ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
      if (n <= 0) return;  // EOF (not a terminal / redirected from /dev/null)
      for (ssize_t i = 0; i < n; ++i) {
        if (buf[i] != '\n') {
          line += buf[i];
          continue;
        }
        if (!line.empty()) handle_command(ctx, line);
        line.clear();
      }
    }
  });

  const double t0 = Telemetry::now();
  while (!ctx.quit->load()) {
    std::ostringstream o;
    o << "\033[H\033[2J";  // home + clear
    const CoordinatorStats cs = coord.stats();
    o << kBold << "slm live self-training" << kReset << "   " << ctx.mcfg->describe()
      << "\n"
      << kDim << "backend " << backend_name() << "  |  weights v" << cs.weight_version
      << "  |  uptime " << static_cast<int>(Telemetry::now() - t0) << "s"
      << "  |  tensor mem "
      << (backend_allocated_bytes() / (1024.0 * 1024.0)) << " MiB" << kReset << "\n\n";

    if (tel.stopped())
      o << kRed << kBold << "  *** EMERGENCY STOP ENGAGED - all training halted ***"
        << kReset << "\n\n";

    // ------------------------------------------------------------ trainers
    for (int i = 0; i < kNumSources; ++i) {
      const Source s = static_cast<Source>(i);
      const TrainerState st = tel.trainer(s);
      const Stream stream = static_cast<Stream>(i);
      o << stream_color(stream) << kBold << "[" << (i + 1) << "] "
        << source_name(s) << kReset << stream_color(stream)
        << (st.enabled ? "" : "  (disabled)") << kReset << "\n";
      char buf[512];
      std::snprintf(buf, sizeof(buf),
                    "    lr %.2e  loss %7.4f  steps %5lld  rounds %3lld  pending %4lld  credit %.2f\n"
                    "    policy: %-28s status: %s\n",
                    st.lr, st.last_loss, static_cast<long long>(st.local_steps),
                    static_cast<long long>(st.rounds),
                    static_cast<long long>(st.pending_inputs), st.credit,
                    st.policy.c_str(), st.status.c_str());
      o << buf << "    " << stream_color(stream)
        << sparkline(tel.loss_series(stream), 64) << kReset << "\n";
    }

    // --------------------------------------------------------- coordinator
    o << "\n" << kGreen << kBold << "coordinator" << kReset << "\n";
    {
      char buf[900];
      std::snprintf(buf, sizeof(buf),
                    "    rounds %lld  accepted %lld  rejected %lld  rollbacks %lld  queue %lld\n"
                    "    holdout loss %.4f (best %.4f)  alpha %.2f  |delta| %.5f  trust %.5f\n"
                    "    rate budget %.4f  anchor drift %.4f  fisher mean %.3e (%lld updates)\n"
                    "    cos(CL,SG) %+.3f  cos(CL,FB) %+.3f  cos(SG,FB) %+.3f\n"
                    "    last: %s\n",
                    static_cast<long long>(cs.rounds), static_cast<long long>(cs.accepted),
                    static_cast<long long>(cs.rejected), static_cast<long long>(cs.rollbacks),
                    static_cast<long long>(cs.queue_len), cs.last_val, cs.best_val,
                    cs.last_alpha, cs.last_delta_norm, cs.trust_radius, cs.rate_budget,
                    cs.anchor_drift, cs.fisher_mean,
                    static_cast<long long>(cs.fisher_updates), cs.cos[0][1], cs.cos[0][2],
                    cs.cos[1][2], cs.last_decision.c_str());
      o << buf;
      o << "    " << kGreen << sparkline(tel.loss_series(Stream::kHoldout), 64) << kReset
        << "\n";
    }

    // ---------------------------------------------------------------- chat
    o << "\n" << kBold << "chat" << kReset;
    if (ctx.chat->busy()) o << kYellow << "  (generating...)" << kReset;
    o << "\n";
    {
      std::vector<ChatTurn> h = ctx.chat->history();
      const size_t show = std::min<size_t>(h.size(), 3);
      for (size_t i = h.size() - show; i < h.size(); ++i)
        o << "    " << kDim << "#" << i << " you>" << kReset << " " << trunc(h[i].prompt, 90)
          << "\n      slm> " << trunc(h[i].response, 110)
          << (h[i].rated ? ("  " + std::string(kGreen) + "[rated " +
                            std::to_string(static_cast<int>(h[i].score)) + "]" + kReset)
                         : "")
          << "\n";
      const std::string p = ctx.chat->partial();
      if (!p.empty()) o << "      " << kYellow << trunc(p, 110) << kReset << "\n";
    }
    {
      const TokenDist d = tel.token_dist();
      if (!d.top.empty()) {
        o << "    next token: ";
        for (size_t i = 0; i < std::min<size_t>(d.top.size(), 6); ++i) {
          char buf[96];
          std::snprintf(buf, sizeof(buf), "%s %.2f   ", d.labels[i].c_str(), d.top[i].second);
          o << buf;
        }
        o << "\n";
      }
      AttentionCapture cap;
      if (tel.attention(&cap) && !cap.layers.empty())
        o << "    attention captured: " << cap.n_layer << " layers x " << cap.n_head
          << " heads, " << cap.Tq << "x" << cap.Tk << "\n";
    }

    // ---------------------------------------------------------------- logs
    o << "\n" << kBold << "audit log" << kReset << "\n";
    for (const std::string& l : tel.recent_logs(9)) o << "    " << kDim << trunc(l, 150) << kReset << "\n";

    o << "\n" << kBold << "keys" << kReset
      << ": [1|2|3] toggle a trainer   [x] emergency stop   [b] restore best   "
         "[a <text>] ask   [r <score>] rate last   [q] quit\n";
    std::fputs(o.str().c_str(), stdout);
    std::fflush(stdout);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (ctx.opt->seconds > 0.0 && Telemetry::now() - t0 > ctx.opt->seconds) break;
  }
  ctx.quit->store(true);
  if (input.joinable()) input.join();
  std::printf("\nshutting down...\n");
  return 0;
}

}  // namespace slm
