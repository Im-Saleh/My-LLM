// SPDX-License-Identifier: Apache-2.0
//
// The bridge between the user (GUI / autopilot) and the learning threads.
// Everything is copied under a mutex; the payloads are strings, so contention
// is irrelevant compared to a training step.
#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace slm {

struct RatedSample {
  std::string prompt;    // raw user text (without control tokens)
  std::string response;  // model output
  float score = 0.0f;    // 1..5 stars, or -1/+1 for thumbs
  double t = 0.0;
};

class InteractionHub {
 public:
  // ------------------------------------------------------- continual learning
  void push_text(const std::string& text) {
    std::lock_guard<std::mutex> g(m_);
    if (text.empty()) return;
    texts_.push_back(text);
    while (texts_.size() > 4096) texts_.pop_front();
    ++total_texts_;
  }
  std::vector<std::string> drain_texts(size_t max_items) {
    std::lock_guard<std::mutex> g(m_);
    std::vector<std::string> out;
    while (!texts_.empty() && out.size() < max_items) {
      out.push_back(std::move(texts_.front()));
      texts_.pop_front();
    }
    return out;
  }
  size_t pending_texts() const {
    std::lock_guard<std::mutex> g(m_);
    return texts_.size();
  }

  // ------------------------------------------------------------ RLHF signal
  void push_rating(const RatedSample& s) {
    std::lock_guard<std::mutex> g(m_);
    rated_.push_back(s);
    while (rated_.size() > 4096) rated_.pop_front();
    ++total_ratings_;
  }
  std::vector<RatedSample> drain_ratings(size_t max_items) {
    std::lock_guard<std::mutex> g(m_);
    std::vector<RatedSample> out;
    while (!rated_.empty() && out.size() < max_items) {
      out.push_back(std::move(rated_.front()));
      rated_.pop_front();
    }
    return out;
  }
  size_t pending_ratings() const {
    std::lock_guard<std::mutex> g(m_);
    return rated_.size();
  }

  int64_t total_texts() const {
    std::lock_guard<std::mutex> g(m_);
    return total_texts_;
  }
  int64_t total_ratings() const {
    std::lock_guard<std::mutex> g(m_);
    return total_ratings_;
  }

 private:
  mutable std::mutex m_;
  std::deque<std::string> texts_;
  std::deque<RatedSample> rated_;
  int64_t total_texts_ = 0;
  int64_t total_ratings_ = 0;
};

}  // namespace slm
