// SPDX-License-Identifier: Apache-2.0
// Entry point of the live self-training application (GUI or terminal).
#pragma once

#include <string>

#include "core/config.h"

namespace slm {

struct AppOptions {
  std::string ckpt;       // base checkpoint (required)
  std::string tokenizer;  // tokenizer file (required)
  // Corpus for replay + per-language hold-out gating.  Either a single file or
  // a mixture spec:  "fa=data/fa.txt:0.35,en=data/en.txt:0.35,py=data/py.txt:0.30"
  std::string data;
  std::string workdir = "runs";
  bool headless = false;  // terminal dashboard instead of ImGui
  double seconds = 0.0;   // auto-exit after N seconds (0 = run forever)
  int threads = 0;
  bool cuda = false;
  uint64_t seed = 1234;
  bool autopilot = false;  // synthesise user turns + ratings (demo / soak test)
  // Optional second model (a GGUF for llama.cpp) and the root the agent tools are
  // allowed to touch.  An empty gguf simply means "one model".
  std::string gguf;
  std::string workspace = ".";
  Config cfg;              // merged configuration
};

// Runs the coordinator, the three training threads and the dashboard.
int run_self_training(const AppOptions& opt);

}  // namespace slm
