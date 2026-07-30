// SPDX-License-Identifier: Apache-2.0
//
// Dashboard front ends.
//
// Both front ends read exclusively from Telemetry and write only through
// InteractionHub / ChatEngine / the enable flags, so neither of them can ever
// stall or corrupt a training thread.
//
//   run_imgui_dashboard     Dear ImGui + GLFW + OpenGL 3 (built when
//                           SLM_WITH_GUI is defined)
//   run_terminal_dashboard  ANSI terminal fallback, no dependencies, works over
//                           SSH and in containers
#pragma once

#include <atomic>
#include <vector>

#include "app.h"
#include "chat.h"
#include "coordinator.h"
#include "interaction.h"
#include "memory.h"
#include "telemetry.h"
#include "tokenizer.h"
#include "trainer.h"

namespace slm {

struct DashboardContext {
  Telemetry* tel = nullptr;
  Coordinator* coord = nullptr;
  InteractionHub* hub = nullptr;
  ChatEngine* chat = nullptr;
  const Tokenizer* tok = nullptr;
  std::vector<TrainerBase*> trainers;
  const AppOptions* opt = nullptr;
  const GPTConfig* mcfg = nullptr;
  std::atomic<bool>* quit = nullptr;
  MemoryStore* memory = nullptr;
};

// Pushes up to `n` memories into the continual-learning buffer so they end up in
// the weights instead of only in the prompt.
int teach_memories(DashboardContext& ctx, size_t n);

int run_terminal_dashboard(DashboardContext& ctx);
int run_imgui_dashboard(DashboardContext& ctx);
bool gui_available();

}  // namespace slm
