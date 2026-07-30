// SPDX-License-Identifier: Apache-2.0
#include "chat.h"

#include <algorithm>

#include "core/text.h"
#include <chrono>

namespace slm {

ChatEngine::ChatEngine(const GPTConfig& mcfg, Coordinator* coord, Telemetry* tel,
                       InteractionHub* hub, const Tokenizer* tok, const GenOptions& go)
    : mcfg_(mcfg), coord_(coord), tel_(tel), hub_(hub), tok_(tok), go_(go) {
  model_ = std::make_unique<GPT>(mcfg_);
  uint64_t v = 0;
  if (coord_) {
    ParamStorePtr p = coord_->snapshot(&v);
    if (p) model_->load_params(*p);
  }
  weight_version_.store(v);
}

ChatEngine::~ChatEngine() { stop(); }

void ChatEngine::start() {
  if (run_.exchange(true)) return;
  th_ = std::thread([this] { loop(); });
}

void ChatEngine::stop() {
  if (!run_.exchange(false)) return;
  cv_.notify_all();
  if (th_.joinable()) th_.join();
}

void ChatEngine::ask(const std::string& text) {
  if (text.empty()) return;
  {
    std::lock_guard<std::mutex> g(m_);
    pending_.push_back(text);
  }
  // The raw user text also feeds the continual-learning buffer.
  if (hub_) hub_->push_text(text);
  cv_.notify_one();
}

size_t ChatEngine::queued() const {
  std::lock_guard<std::mutex> g(m_);
  return pending_.size();
}

std::string ChatEngine::partial() const {
  std::lock_guard<std::mutex> g(m_);
  return partial_;
}

std::string ChatEngine::last_memory_block() const {
  std::lock_guard<std::mutex> g(m_);
  return last_mem_;
}

std::vector<ChatTurn> ChatEngine::history() const {
  std::lock_guard<std::mutex> g(m_);
  return history_;
}

void ChatEngine::rate(size_t index, float score) {
  RatedSample s;
  {
    std::lock_guard<std::mutex> g(m_);
    if (index >= history_.size()) return;
    history_[index].rated = true;
    history_[index].score = score;
    s.prompt = history_[index].prompt;
    s.response = history_[index].response;
  }
  s.score = score;
  s.t = Telemetry::now();
  if (hub_) hub_->push_rating(s);
  if (tel_)
    tel_->log("info", "user", "rated a response",
              {{"score", std::to_string(score)}, {"prompt", utf8_truncate(s.prompt, 80)}});
}

void ChatEngine::loop() {
  while (run_.load()) {
    std::string req;
    {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait_for(lk, std::chrono::milliseconds(200),
                   [this] { return !pending_.empty() || !run_.load(); });
      if (!run_.load()) break;
      if (pending_.empty()) continue;
      req = std::move(pending_.front());
      pending_.pop_front();
    }
    busy_.store(true);
    if (tel_) tel_->set_chat_busy(true);
    try {
      handle(req);
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> g(m_);
      history_.push_back(ChatTurn{req, std::string("[generation failed: ") + e.what() + "]",
                                  Telemetry::now(), false, 0.0f, 0, 0.0});
    }
    busy_.store(false);
    if (tel_) tel_->set_chat_busy(false);
  }
}

void ChatEngine::handle(const std::string& text) {
  // Re-sync weights between turns (RCU read side).
  if (coord_) {
    uint64_t v = 0;
    ParamStorePtr p = coord_->snapshot(&v);
    if (p && v != weight_version_.load()) {
      model_->load_params(*p);
      weight_version_.store(v);
    }
  }
  // A fresh seed per turn: asking the same question twice must be able to
  // produce two different answers, otherwise the feedback thread can never
  // build a preference pair.
  go_.seed = rng_.next_u64() | 1ull;
  // Retrieval augmented prompt: what the model was *told* to remember comes
  // first, then the user turn.
  std::string mem_block;
  if (mem_ && mem_->size() > 0) mem_block = mem_->context_block(text, mem_k_);
  {
    std::lock_guard<std::mutex> g(m_);
    last_mem_ = mem_block;
  }
  const std::vector<int32_t> prompt =
      tok_->encode(mem_block + "<|user|>" + text + "<|assistant|>");
  {
    std::lock_guard<std::mutex> g(m_);
    partial_.clear();
  }
  AttentionCapture cap;
  std::string out;
  int64_t step = 0;
  const double t0 = Telemetry::now();
  std::vector<int32_t> ids = model_->generate(
      prompt, go_,
      [&](const GenStep& s) {
        out += tok_->decode({s.token});
        {
          std::lock_guard<std::mutex> g(m_);
          partial_ = out;
        }
        if (tel_) {
          TokenDist d;
          d.top = s.top;
          d.labels.reserve(s.top.size());
          for (const auto& kv : s.top) d.labels.push_back(tok_->token_display(kv.first));
          d.context = out;
          d.step = ++step;
          tel_->set_token_dist(d);
        }
        return !(tel_ && tel_->stopped());
      },
      &cap);
  const double dt = std::max(1e-6, Telemetry::now() - t0);
  if (tel_ && !cap.layers.empty()) tel_->set_attention(cap);

  ChatTurn turn;
  turn.prompt = text;
  turn.response = out;
  turn.t = Telemetry::now();
  turn.tokens = static_cast<int>(ids.size());
  turn.tokens_per_s = static_cast<double>(ids.size()) / dt;
  std::lock_guard<std::mutex> g(m_);
  history_.push_back(std::move(turn));
  while (history_.size() > 256) history_.erase(history_.begin());
  partial_.clear();
}

}  // namespace slm
