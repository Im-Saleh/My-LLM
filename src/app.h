// SPDX-License-Identifier: Apache-2.0
// Entry point of the live self-training application (GUI or terminal).
#pragma once

#include <string>

#include "core/config.h"

namespace slm {

struct AppOptions {
  std::string ckpt;       // base checkpoint (required)
  std::string tokenizer;  // tokenizer file (required)
  std::string data;       // corpus used for replay + hold-out gating
  std::string workdir = "runs";
  bool headless = false;  // terminal dashboard instead of ImGui
  double seconds = 0.0;   // auto-exit after N seconds (0 = run forever)
  int threads = 0;
  bool cuda = false;
  uint64_t seed = 1234;
  bool autopilot = false;  // synthesise user turns + ratings (demo / soak test)
  Config cfg;              // merged configuration
};

// Runs the coordinator, the three training threads and the dashboard.
int run_self_training(const AppOptions& opt);

}  // namespace slm
