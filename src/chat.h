// SPDX-License-Identifier: Apache-2.0
//
// Inference worker.
//
// Generation runs on its own thread so the dashboard stays at 60 FPS while the
// model is decoding.  The engine re-syncs its weights from the coordinator
// between turns only (never in the middle of a turn), which is exactly the RCU
// read side: it takes a shared_ptr to a published snapshot and that snapshot is
// kept alive for as long as it is in use.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/rng.h"
#include "coordinator.h"
#include "interaction.h"
#include "memory.h"
#include "model.h"
#include "telemetry.h"
#include "tokenizer.h"

namespace slm {

struct ChatTurn {
  std::string prompt;
  std::string response;
  double t = 0.0;
  bool rated = false;
  float score = 0.0f;
  int tokens = 0;
  double tokens_per_s = 0.0;
};

class ChatEngine {
 public:
  ChatEngine(const GPTConfig& mcfg, Coordinator* coord, Telemetry* tel,
             InteractionHub* hub, const Tokenizer* tok, const GenOptions& go);

  // Optional long-term memory: relevant entries are retrieved and injected as a
  // <|system|> block before every turn (no training needed), and can later be
  // consolidated into the weights by the continual thread.
  void set_memory(MemoryStore* mem, int top_k = 3) {
    mem_ = mem;
    mem_k_ = top_k;
  }
  MemoryStore* memory() const { return mem_; }
  std::string last_memory_block() const;
  ~ChatEngine();

  void start();
  void stop();

  void ask(const std::string& text);
  bool busy() const { return busy_.load(); }
  size_t queued() const;
  std::string partial() const;
  std::vector<ChatTurn> history() const;
  void rate(size_t index, float score);
  uint64_t weight_version() const { return weight_version_.load(); }

  GenOptions& gen_options() { return go_; }

 private:
  void loop();
  void handle(const std::string& text);

  GPTConfig mcfg_;
  Coordinator* coord_;
  Telemetry* tel_;
  InteractionHub* hub_;
  const Tokenizer* tok_;
  GenOptions go_;
  Rng rng_{0x9e3779b97f4a7c15ull};
  std::unique_ptr<GPT> model_;

  mutable std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::string> pending_;
  std::vector<ChatTurn> history_;
  std::string partial_;

  MemoryStore* mem_ = nullptr;
  int mem_k_ = 3;
  std::string last_mem_;
  std::thread th_;
  std::atomic<bool> run_{false};
  std::atomic<bool> busy_{false};
  std::atomic<uint64_t> weight_version_{0};
};

}  // namespace slm
