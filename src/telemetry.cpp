// SPDX-License-Identifier: Apache-2.0
#include "telemetry.h"

#include "core/text.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>

namespace slm {

const char* source_name(Source s) {
  switch (s) {
    case Source::kContinual:
      return "continual";
    case Source::kSelfGen:
      return "self-generated";
    case Source::kFeedback:
      return "feedback";
    default:
      return "?";
  }
}

const char* source_short(Source s) {
  switch (s) {
    case Source::kContinual:
      return "CL";
    case Source::kSelfGen:
      return "SG";
    case Source::kFeedback:
      return "FB";
    default:
      return "??";
  }
}

const char* stream_name(Stream s) {
  switch (s) {
    case Stream::kContinual:
      return "continual";
    case Stream::kSelfGen:
      return "self-gen";
    case Stream::kFeedback:
      return "feedback";
    case Stream::kHoldout:
      return "holdout";
    case Stream::kHoldoutFa:
      return "holdout fa";
    case Stream::kHoldoutEn:
      return "holdout en";
    case Stream::kHoldoutPy:
      return "holdout py";
    default:
      return "?";
  }
}

Stream stream_for_lang(Lang l) {
  switch (l) {
    case Lang::kPersian:
      return Stream::kHoldoutFa;
    case Lang::kEnglish:
      return Stream::kHoldoutEn;
    case Lang::kPython:
      return Stream::kHoldoutPy;
    default:
      return Stream::kHoldout;
  }
}

double Telemetry::now() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
}

void Telemetry::push_loss(Stream s, float value, int64_t step) {
  const int i = static_cast<int>(s);
  std::lock_guard<std::mutex> g(loss_m_);
  loss_[i].push_back({now(), value, step});
  while (loss_[i].size() > capacity_) loss_[i].pop_front();
}

std::vector<LossPoint> Telemetry::loss_series(Stream s) const {
  const int i = static_cast<int>(s);
  std::lock_guard<std::mutex> g(loss_m_);
  return std::vector<LossPoint>(loss_[i].begin(), loss_[i].end());
}

void Telemetry::loss_bounds(float* lo, float* hi) const {
  float mn = 1e30f, mx = -1e30f;
  {
    std::lock_guard<std::mutex> g(loss_m_);
    for (int i = 0; i < kNumStreams; ++i)
      for (const LossPoint& p : loss_[i]) {
        mn = std::min(mn, p.value);
        mx = std::max(mx, p.value);
      }
  }
  if (mn > mx) {
    mn = 0.0f;
    mx = 1.0f;
  }
  *lo = mn;
  *hi = mx;
}

void Telemetry::set_trainer(Source s, const TrainerState& st) {
  std::lock_guard<std::mutex> g(tr_m_);
  const bool enabled = trainers_[static_cast<int>(s)].enabled;
  trainers_[static_cast<int>(s)] = st;
  trainers_[static_cast<int>(s)].enabled = enabled;  // owned by the UI
}

TrainerState Telemetry::trainer(Source s) const {
  std::lock_guard<std::mutex> g(tr_m_);
  return trainers_[static_cast<int>(s)];
}

void Telemetry::set_trainer_enabled(Source s, bool on) {
  std::lock_guard<std::mutex> g(tr_m_);
  trainers_[static_cast<int>(s)].enabled = on;
}

bool Telemetry::trainer_enabled(Source s) const {
  std::lock_guard<std::mutex> g(tr_m_);
  return trainers_[static_cast<int>(s)].enabled;
}

void Telemetry::set_coord(const CoordinatorStats& st) {
  std::lock_guard<std::mutex> g(co_m_);
  coord_ = st;
}

CoordinatorStats Telemetry::coord() const {
  std::lock_guard<std::mutex> g(co_m_);
  return coord_;
}

void Telemetry::set_attention(const AttentionCapture& cap) {
  std::lock_guard<std::mutex> g(att_m_);
  att_ = cap;
  att_valid_ = !cap.layers.empty();
}

bool Telemetry::attention(AttentionCapture* out) const {
  std::lock_guard<std::mutex> g(att_m_);
  if (!att_valid_) return false;
  *out = att_;
  return true;
}

void Telemetry::set_token_dist(const TokenDist& d) {
  std::lock_guard<std::mutex> g(td_m_);
  td_ = d;
}

TokenDist Telemetry::token_dist() const {
  std::lock_guard<std::mutex> g(td_m_);
  return td_;
}

void Telemetry::open_audit(const std::string& path) {
  std::lock_guard<std::mutex> g(log_m_);
  audit_path_ = path;
  std::ofstream f(path, std::ios::app);
  if (f) {
    const std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    f << "{\"event\":\"session_start\",\"wall\":\"" << buf << "\"}\n";
  }
}

namespace {
std::string json_escape(const std::string& raw) {
  // Always emit valid UTF-8: a Persian string cut mid-sequence anywhere upstream
  // would otherwise produce an audit log that no JSON parser can read.
  const std::string s = utf8_sanitize(raw);
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"':
        o += "\\\"";
        break;
      case '\\':
        o += "\\\\";
        break;
      case '\n':
        o += "\\n";
        break;
      case '\r':
        o += "\\r";
        break;
      case '\t':
        o += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20)
          o += ' ';
        else
          o += c;
    }
  }
  return o;
}
}  // namespace

void Telemetry::log(const std::string& level, const std::string& source,
                    const std::string& message,
                    const std::vector<std::pair<std::string, std::string>>& fields) {
  const double t = now();
  char head[64];
  std::snprintf(head, sizeof(head), "[%8.2fs] %-8s %-12s ", t, level.c_str(),
                source.c_str());
  std::string line = std::string(head) + message;
  for (const auto& f : fields) line += "  " + f.first + "=" + f.second;

  std::ostringstream js;
  js << "{\"t\":" << t << ",\"level\":\"" << json_escape(level) << "\",\"source\":\""
     << json_escape(source) << "\",\"message\":\"" << json_escape(message) << "\"";
  for (const auto& f : fields)
    js << ",\"" << json_escape(f.first) << "\":\"" << json_escape(f.second) << "\"";
  js << "}";

  std::string path;
  {
    std::lock_guard<std::mutex> g(log_m_);
    logs_.push_back(line);
    while (logs_.size() > 512) logs_.pop_front();
    path = audit_path_;
  }
  if (!path.empty()) {
    std::ofstream f(path, std::ios::app);
    if (f) f << js.str() << "\n";
  }
}

std::vector<std::string> Telemetry::recent_logs(size_t max_lines) const {
  std::lock_guard<std::mutex> g(log_m_);
  const size_t n = std::min(max_lines, logs_.size());
  return std::vector<std::string>(logs_.end() - static_cast<long>(n), logs_.end());
}

void Telemetry::emergency_stop() {
  estop_.store(true, std::memory_order_relaxed);
  log("warn", "control", "EMERGENCY STOP engaged: all training halted");
}

void Telemetry::clear_emergency_stop() {
  estop_.store(false, std::memory_order_relaxed);
  log("info", "control", "emergency stop released");
}

}  // namespace slm
