// SPDX-License-Identifier: Apache-2.0
//
// Dear ImGui dashboard.
//
// Why Dear ImGui and not Qt for this project
// ------------------------------------------
//  * The panels are pure functions of live numeric state that changes every
//    frame (loss curves, attention matrices, token distributions).  An
//    immediate-mode UI redraws from the current state; a retained-mode toolkit
//    like Qt would need models, signals and manual invalidation for the same
//    result.
//  * Drawing a 256x256 attention heat map per frame is a handful of
//    ImDrawList::AddRectFilled calls.  In Qt it means a custom QWidget with its
//    own paintEvent and pixmap caching.
//  * It is a ~10 file source drop compiled into the binary: no moc, no uic, no
//    extra runtime, no LGPL considerations, trivial static linking.
//  * Qt is the better choice for a document oriented desktop app with native
//    widgets, menus and accessibility.  This is a telemetry cockpit, so ImGui
//    wins on both latency and maintenance cost.
#include "gui.h"

#ifdef SLM_WITH_GUI

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "agent/runtime.h"
#include "gui_agent.h"
#include "qmodel.h"

#include "gui_text.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
// GLFW must come after the ImGui backend header.
#include <GLFW/glfw3.h>

namespace slm {
namespace {

const ImU32 kColors[kNumStreams] = {
    IM_COL32(90, 170, 255, 255),   // continual   - blue
    IM_COL32(255, 200, 80, 255),   // self-gen    - amber
    IM_COL32(220, 120, 255, 255),  // feedback    - violet
    IM_COL32(110, 230, 140, 255),  // holdout all - green
    IM_COL32(255, 130, 130, 255),  // holdout fa  - red
    IM_COL32(150, 220, 255, 255),  // holdout en  - light blue
    IM_COL32(200, 255, 150, 255),  // holdout py  - lime
};

ShapedText g_shaper;

}  // namespace

bool shaping_ready() { return g_shaper.ready(); }

namespace {

// Draws one line of text, using the Persian shaper when the string contains
// Arabic script.  Falls back to plain ImGui text otherwise (and when the
// shaper is unavailable), so the dashboard never depends on it.
void ui_line(const std::string& s, const ImVec4* color = nullptr) {
  if (s.empty()) {
    ImGui::TextUnformatted("");
    return;
  }
  if (g_shaper.ready() && needs_shaping(s)) {
    const ShapedText::Image* img = g_shaper.image(s);
    if (img && img->texture) {
      const ImVec4 tint = color ? *color : ImGui::GetStyleColorVec4(ImGuiCol_Text);
      ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(img->texture)),
                   ImVec2(static_cast<float>(img->width), static_cast<float>(img->height)),
                   ImVec2(0, 0), ImVec2(1, 1), tint);
      return;
    }
  }
  if (color)
    ImGui::TextColored(*color, "%s", s.c_str());
  else
    ImGui::TextUnformatted(s.c_str());
}

// Word-wrapped variant: shaped strings are measured word by word so a long
// Persian answer wraps like any other paragraph.
void ui_wrapped(const std::string& text, const ImVec4* color = nullptr) {
  const float avail = ImGui::GetContentRegionAvail().x - 4.0f;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t nl = text.find('\n', start);
    const std::string line =
        text.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
    if (!g_shaper.ready() || !needs_shaping(line)) {
      if (color)
        ImGui::TextColored(*color, "%s", line.c_str());
      else
        ImGui::TextWrapped("%s", line.c_str());
    } else if (g_shaper.measure(line) <= avail) {
      ui_line(line, color);
    } else {
      // greedy word wrapping
      std::string cur;
      size_t i = 0;
      while (i < line.size()) {
        const size_t sp = line.find(' ', i);
        const std::string word =
            line.substr(i, sp == std::string::npos ? std::string::npos : sp - i);
        const std::string candidate = cur.empty() ? word : cur + " " + word;
        if (!cur.empty() && g_shaper.measure(candidate) > avail) {
          ui_line(cur, color);
          cur = word;
        } else {
          cur = candidate;
        }
        if (sp == std::string::npos) break;
        i = sp + 1;
      }
      if (!cur.empty()) ui_line(cur, color);
    }
    if (nl == std::string::npos) break;
    start = nl + 1;
  }
}

void glfw_error(int code, const char* msg) {
  std::fprintf(stderr, "glfw error %d: %s\n", code, msg);
}

// viridis-like colormap for the attention heat map
ImU32 heat(float t) {
  t = std::min(1.0f, std::max(0.0f, t));
  const float r = std::min(1.0f, std::max(0.0f, 1.30f * t - 0.30f));
  const float g = std::min(1.0f, std::max(0.0f, 1.10f * std::sqrt(t) - 0.05f));
  const float b = std::min(1.0f, std::max(0.0f, 0.85f - 0.75f * t + 0.35f * (1.0f - t) * (1.0f - t)));
  return IM_COL32(static_cast<int>(255 * r), static_cast<int>(255 * g),
                  static_cast<int>(255 * b), 255);
}

// Multi-series line chart drawn straight into the window draw list.
void plot_losses(Telemetry& tel, float height, float window_seconds, bool log_scale,
                 const bool* visible) {
  std::vector<LossPoint> series[kNumStreams];
  for (int i = 0; i < kNumStreams; ++i)
    series[i] = tel.loss_series(static_cast<Stream>(i));

  const double now = Telemetry::now();
  float lo = 1e30f, hi = -1e30f;
  double t_min = now - window_seconds;
  for (int i = 0; i < kNumStreams; ++i) {
    if (!visible[i]) continue;
    for (const LossPoint& p : series[i]) {
      if (p.t < t_min) continue;
      float v = p.value;
      if (log_scale) v = std::log10(std::max(1e-4f, v));
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  }
  if (lo > hi) {
    lo = 0.0f;
    hi = 1.0f;
  }
  const float pad = 0.08f * (hi - lo + 1e-3f);
  lo -= pad;
  hi += pad;

  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  const float width = ImGui::GetContentRegionAvail().x;
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(18, 20, 26, 255));
  dl->AddRect(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(60, 65, 80, 255));

  // grid + y labels
  for (int g = 0; g <= 4; ++g) {
    const float y = p0.y + height * static_cast<float>(g) / 4.0f;
    dl->AddLine(ImVec2(p0.x, y), ImVec2(p0.x + width, y), IM_COL32(45, 48, 60, 255));
    const float value = hi - (hi - lo) * static_cast<float>(g) / 4.0f;
    char buf[32];
    std::snprintf(buf, sizeof(buf), log_scale ? "1e%.2f" : "%.3f", value);
    dl->AddText(ImVec2(p0.x + 4, y + 1), IM_COL32(140, 145, 160, 255), buf);
  }

  for (int i = 0; i < kNumStreams; ++i) {
    if (!visible[i] || series[i].size() < 2) continue;
    ImVec2 prev(0, 0);
    bool have_prev = false;
    for (const LossPoint& p : series[i]) {
      if (p.t < t_min) continue;
      float v = p.value;
      if (log_scale) v = std::log10(std::max(1e-4f, v));
      const float x = p0.x + width * static_cast<float>((p.t - t_min) / std::max(1e-6, now - t_min));
      const float y = p0.y + height * (1.0f - (v - lo) / std::max(1e-6f, hi - lo));
      const ImVec2 cur(x, y);
      if (have_prev) dl->AddLine(prev, cur, kColors[i], 1.8f);
      prev = cur;
      have_prev = true;
    }
    if (have_prev) dl->AddCircleFilled(prev, 3.0f, kColors[i]);
  }
  ImGui::Dummy(ImVec2(width, height));
}

void legend(const char* label, ImU32 color, bool* visible) {
  ImGui::PushStyleColor(ImGuiCol_Text, color);
  ImGui::Checkbox(label, visible);
  ImGui::PopStyleColor();
}

void attention_panel(Telemetry& tel, const GPTConfig& mcfg, int* layer, int* head) {
  AttentionCapture cap;
  if (!tel.attention(&cap) || cap.layers.empty()) {
    ImGui::TextDisabled("no attention captured yet - send a chat message");
    return;
  }
  *layer = std::min(*layer, static_cast<int>(cap.layers.size()) - 1);
  *head = std::min(*head, static_cast<int>(cap.n_head) - 1);
  ImGui::SetNextItemWidth(140);
  ImGui::SliderInt("layer", layer, 0, static_cast<int>(cap.layers.size()) - 1);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(140);
  ImGui::SliderInt("head", head, 0, static_cast<int>(cap.n_head) - 1);
  ImGui::SameLine();
  ImGui::TextDisabled("%lldx%lld  (%lld layers x %lld heads)", static_cast<long long>(cap.Tq),
                      static_cast<long long>(cap.Tk),
                      static_cast<long long>(cap.layers.size()),
                      static_cast<long long>(cap.n_head));
  (void)mcfg;

  const float* a = cap.at(*layer, *head);
  const int64_t Tq = cap.Tq, Tk = cap.Tk;
  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  const float avail_w = ImGui::GetContentRegionAvail().x;
  const float avail_h = std::max(80.0f, ImGui::GetContentRegionAvail().y - 8.0f);
  const float cell = std::max(1.0f, std::min(avail_w / static_cast<float>(Tk),
                                             avail_h / static_cast<float>(Tq)));
  ImDrawList* dl = ImGui::GetWindowDrawList();
  float mx = 1e-6f;
  for (int64_t i = 0; i < Tq * Tk; ++i) mx = std::max(mx, a[i]);
  for (int64_t q = 0; q < Tq; ++q)
    for (int64_t k = 0; k <= std::min(Tk - 1, q + (Tk - Tq)); ++k) {
      const float v = a[q * Tk + k] / mx;
      const ImVec2 r0(p0.x + static_cast<float>(k) * cell, p0.y + static_cast<float>(q) * cell);
      dl->AddRectFilled(r0, ImVec2(r0.x + cell, r0.y + cell), heat(std::pow(v, 0.6f)));
    }
  const ImVec2 size(cell * static_cast<float>(Tk), cell * static_cast<float>(Tq));
  ImGui::Dummy(size);
  if (ImGui::IsItemHovered()) {
    const ImVec2 m = ImGui::GetIO().MousePos;
    const int64_t k = static_cast<int64_t>((m.x - p0.x) / cell);
    const int64_t q = static_cast<int64_t>((m.y - p0.y) / cell);
    if (q >= 0 && q < Tq && k >= 0 && k < Tk)
      ImGui::SetTooltip("query %lld -> key %lld : %.4f", static_cast<long long>(q),
                        static_cast<long long>(k), a[q * Tk + k]);
  }
}

void token_panel(Telemetry& tel) {
  const TokenDist d = tel.token_dist();
  if (d.top.empty()) {
    ImGui::TextDisabled("no sampling step recorded yet");
    return;
  }
  ImGui::TextDisabled("step %lld", static_cast<long long>(d.step));
  for (size_t i = 0; i < d.top.size(); ++i) {
    const float p = d.top[i].second;
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                          i == 0 ? IM_COL32(110, 230, 140, 255) : IM_COL32(90, 140, 220, 255));
    char label[64];
    std::snprintf(label, sizeof(label), "%.3f", p);
    ImGui::ProgressBar(p, ImVec2(-120, 0), label);
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ui_line(d.labels[i]);
  }
}

void trainer_panel(DashboardContext& ctx, Source s) {
  Telemetry& tel = *ctx.tel;
  const int i = static_cast<int>(s);
  const TrainerState st = tel.trainer(s);
  ImGui::PushID(i);
  ImGui::PushStyleColor(ImGuiCol_Text, kColors[i]);
  ImGui::Text("%s", source_name(s));
  ImGui::PopStyleColor();
  bool enabled = st.enabled;
  if (ImGui::Checkbox("active", &enabled)) {
    tel.set_trainer_enabled(s, enabled);
    tel.log("info", "control",
            std::string(enabled ? "enabled " : "disabled ") + source_name(s));
  }
  ImGui::SameLine();
  ImGui::TextDisabled("%s", st.busy ? "[running]" : "[idle]");
  ImGui::Text("lr %.2e   loss %.4f   credit %.2f", st.lr, st.last_loss, st.credit);
  ImGui::Text("steps %lld   rounds %lld   pending %lld", static_cast<long long>(st.local_steps),
              static_cast<long long>(st.rounds), static_cast<long long>(st.pending_inputs));
  ImGui::TextWrapped("policy: %s", st.policy.c_str());
  ImGui::TextWrapped("status: %s", st.status.c_str());
  ImGui::Separator();
  ImGui::PopID();
}

void coordinator_panel(DashboardContext& ctx) {
  const CoordinatorStats cs = ctx.coord->stats();
  ImGui::Text("rounds %lld   accepted %lld   rejected %lld   rollbacks %lld",
              static_cast<long long>(cs.rounds), static_cast<long long>(cs.accepted),
              static_cast<long long>(cs.rejected), static_cast<long long>(cs.rollbacks));
  ImGui::Text("holdout %.4f   best %.4f   alpha %.2f", cs.last_val, cs.best_val, cs.last_alpha);
  ImGui::Text("|delta| %.5f   trust %.5f   drift %.4f", cs.last_delta_norm, cs.trust_radius,
              cs.anchor_drift);
  ImGui::Text("rate budget");
  ImGui::SameLine();
  ImGui::ProgressBar(std::min(1.0f, cs.rate_budget / 0.12f), ImVec2(-1, 0));
  ImGui::Text("fisher mean %.3e (%lld updates)", cs.fisher_mean,
              static_cast<long long>(cs.fisher_updates));
  ImGui::Text("weights v%llu   queue %lld", static_cast<unsigned long long>(cs.weight_version),
              static_cast<long long>(cs.queue_len));
  ImGui::Separator();
  ImGui::TextDisabled("pairwise proposal similarity");
  const char* names[kNumSources] = {"CL", "SG", "FB"};
  for (int i = 0; i < kNumSources; ++i) {
    for (int j = 0; j < kNumSources; ++j) {
      const float c = cs.cos[i][j];
      const ImU32 col = c < -0.02f ? IM_COL32(230, 90, 90, 255)
                                   : (c > 0.02f ? IM_COL32(110, 220, 130, 255)
                                                : IM_COL32(150, 150, 150, 255));
      ImGui::PushStyleColor(ImGuiCol_Text, col);
      ImGui::Text("%s/%s %+.3f", names[i], names[j], c);
      ImGui::PopStyleColor();
      if (j + 1 < kNumSources) ImGui::SameLine();
    }
  }
  ImGui::Separator();
  ImGui::TextWrapped("last decision: %s", cs.last_decision.c_str());
}

// Long term memory: write, inspect, and push into the weights.
void memory_panel(DashboardContext& ctx, char* buf, size_t buf_size) {
  if (!ctx.memory) {
    ImGui::TextDisabled("memory store not available");
    return;
  }
  ImGui::Text("%zu memories", ctx.memory->size());
  ImGui::SameLine();
  if (ImGui::SmallButton("teach the weights")) {
    const int n = teach_memories(ctx, 8);
    ctx.tel->log("info", "memory",
                 "queued " + std::to_string(n) + " memories for the continual thread");
  }
  ImGui::SetNextItemWidth(-70);
  const bool enter =
      ImGui::InputText("##mem", buf, buf_size, ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  const bool add = ImGui::Button("remember", ImVec2(64, 0));
  if ((enter || add) && buf[0] != '\0') {
    ctx.memory->add(buf, "user");
    buf[0] = '\0';
  }
  if (buf[0] != '\0' && needs_shaping(buf)) ui_line(buf);
  const std::string block = ctx.chat->last_memory_block();
  if (!block.empty()) {
    ImGui::TextDisabled("last retrieved:");
    const ImVec4 dim(0.6f, 0.75f, 0.9f, 1.0f);
    ui_wrapped(block.substr(std::min<size_t>(block.size(), 10)), &dim);
  }
  ImGui::BeginChild("memlist", ImVec2(0, 120), ImGuiChildFlags_Border);
  for (const MemoryItem& it : ctx.memory->all()) {
    ImGui::PushID(static_cast<int>(it.id));
    if (ImGui::SmallButton("x")) ctx.memory->forget(it.id);
    ImGui::SameLine();
    ImGui::TextDisabled("#%lld t%lld", static_cast<long long>(it.id),
                        static_cast<long long>(it.taught));
    ImGui::SameLine();
    ui_wrapped(it.text);
    ImGui::PopID();
  }
  ImGui::EndChild();
}

// ======================================================== agent / debate panels
// UI state that has to survive between frames.  Kept in one struct so the frame
// loop stays readable and so nothing here is accidentally re-initialised.
struct AgentUi {
  int mode = 0;                 // index into kModeNames
  int fast_mult = 2;
  int strong_mult = 1;
  int voices = 2;
  bool use_tools = true;
  bool use_codebase = true;
  int max_tokens = 320;
  char input[2048] = {0};
  char index_root[512] = {0};
  char dataset[512] = {0};
  int dataset_max_lines = 0;    // 0 = the whole file
  int dataset_chunk = 800;
  int open_round = -1;
  bool show_context = false;
};

const char* const kModeNames[4] = {"fast (SPT only)", "strong (GGUF only)",
                                   "debate (both, weighted)", "self-debate (SPT x N)"};
AskMode mode_of(int i) {
  switch (i) {
    case 1: return AskMode::kStrong;
    case 2: return AskMode::kDebate;
    case 3: return AskMode::kSelfDebate;
    default: return AskMode::kFast;
  }
}

// The model selector: which model answers, and with what weight.  This is the
// one control that changes the cost of everything else, so it shows the live
// state of each backend right next to the choice.
void model_selector(DashboardContext& ctx, AgentUi& ui) {
  if (!ctx.agent) {
    ImGui::TextDisabled("agent runtime unavailable");
    return;
  }
  ImGui::SetNextItemWidth(230);
  ImGui::Combo("model", &ui.mode, kModeNames, 4);
  const AskMode m = mode_of(ui.mode);
  const bool needs_strong = (m == AskMode::kStrong || m == AskMode::kDebate);
  if (needs_strong && !ctx.agent->mode_available(m)) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.4f, 1.0f), "no GGUF configured");
  }
  if (m == AskMode::kDebate) {
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("SPT x", &ui.fast_mult, 1, 5);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("GGUF x", &ui.strong_mult, 1, 5);
    ImGui::SameLine();
    ImGui::TextDisabled("%d rounds", std::max(ui.fast_mult, ui.strong_mult));
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip(
          "The multiplier is both the vote weight and how often a model takes "
          "part: with 2x and 1x the debate runs 2 critique rounds, the 2x model "
          "joins both, the 1x model one.");
  } else if (m == AskMode::kSelfDebate) {
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("voices", &ui.voices, 2, 4);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("rounds x", &ui.fast_mult, 1, 4);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip(
          "The same weights debating themselves: separate KV caches, separate "
          "seeds and different personas, so the voices genuinely disagree.");
  }
  ImGui::Checkbox("tools", &ui.use_tools);
  ImGui::SameLine();
  ImGui::Checkbox("codebase context", &ui.use_codebase);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(110);
  ImGui::SliderInt("max tok", &ui.max_tokens, 64, 1024);

  ImGui::Separator();
  for (const BackendPtr& b : ctx.agent->backends().all()) {
    const BackendStatus s = b->status();
    const ImVec4 col = s.loaded ? ImVec4(0.6f, 0.95f, 0.7f, 1.0f)
                                : ImVec4(0.7f, 0.7f, 0.75f, 1.0f);
    ImGui::TextColored(col, "%s", s.loaded ? "*" : "o");
    ImGui::SameLine();
    ImGui::Text("%-14s %s", b->display_name().c_str(), b->runtime().c_str());
    if (s.loaded) {
      ImGui::SameLine();
      ImGui::TextDisabled("%.1fM  %.0f MiB%s%s", s.params / 1e6,
                          s.weight_bytes / (1024.0 * 1024.0),
                          s.kv_bytes ? "  +KV " : "",
                          s.kv_bytes ? human_bytes(static_cast<double>(s.kv_bytes)).c_str()
                                     : "");
      if (s.last_decode_tps > 0.0) {
        ImGui::SameLine();
        ImGui::TextDisabled("%.0f tok/s", s.last_decode_tps);
      }
      if (s.cache_hit_tokens > 0) {
        ImGui::SameLine();
        ImGui::TextDisabled("| %lld cached",
                            static_cast<long long>(s.cache_hit_tokens));
      }
    } else {
      ImGui::SameLine();
      if (ImGui::SmallButton((std::string("load##") + b->id()).c_str())) {
        std::string err;
        if (!b->load(&err)) ctx.tel->log("warn", "agent", "load failed: " + err);
      }
      if (!s.error.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.45f, 1.0f), "%s",
                           utf8_truncate(s.error, 60).c_str());
      }
    }
  }
  const size_t w = ctx.agent->backends().weight_bytes();
  const size_t kv = ctx.agent->backends().kv_bytes();
  ImGui::TextDisabled("resident: %s weights + %s KV", human_bytes(static_cast<double>(w)).c_str(),
                      human_bytes(static_cast<double>(kv)).c_str());
}

// The debate view: what each model said, what it criticised, and the scores that
// decided the winner.  Progress is the engine's own estimate of work completed,
// weighted by each backend's measured speed, not a count of rounds.
void debate_panel(DashboardContext& ctx, AgentUi& ui) {
  if (!ctx.actrl || !ctx.actrl->attached()) {
    ImGui::TextDisabled("agent not attached");
    return;
  }
  const AgentSnapshot snap = ctx.actrl->snapshot();

  if (snap.busy) {
    ImGui::ProgressBar(static_cast<float>(snap.progress), ImVec2(-90, 0));
    ImGui::SameLine();
    if (ImGui::Button("cancel", ImVec2(80, 0))) ctx.actrl->cancel();
    ImGui::TextDisabled("%s", utf8_truncate(snap.question, 90).c_str());
  }

  ImGui::BeginChild("debatebody", ImVec2(0, -64), ImGuiChildFlags_Border);
  const DebateTranscript* live = nullptr;
  if (snap.busy && !snap.live.rounds.empty()) live = &snap.live;
  else if (!snap.history.empty() && snap.history.back().was_debate)
    live = &snap.history.back().debate;

  if (live) {
    for (const DebateRound& r : live->rounds) {
      const ImVec4 col = r.kind == "draft"    ? ImVec4(0.6f, 0.8f, 1.0f, 1.0f)
                         : r.kind == "critique" ? ImVec4(1.0f, 0.85f, 0.5f, 1.0f)
                                                : ImVec4(0.6f, 0.95f, 0.7f, 1.0f);
      ImGui::PushStyleColor(ImGuiCol_Text, col);
      const bool open = ImGui::TreeNodeEx(
          reinterpret_cast<const void*>(static_cast<intptr_t>(r.index + 1)),
          r.index == live->rounds.size() - 1 ? ImGuiTreeNodeFlags_DefaultOpen : 0,
          "round %d - %s (%zu answers, %.2fs)", r.index, r.kind.c_str(),
          r.answers.size(), r.seconds);
      ImGui::PopStyleColor();
      if (!open) continue;
      if (!r.note.empty()) ImGui::TextDisabled("%s", r.note.c_str());
      for (size_t i = 0; i < r.answers.size(); ++i) {
        const DebateAnswer& a = r.answers[i];
        ImGui::PushID(static_cast<int>(r.index * 100 + i));
        ImGui::TextColored(ImVec4(0.75f, 0.85f, 1.0f, 1.0f), "%s#%d", a.participant.c_str(),
                           a.draft);
        ImGui::SameLine();
        ImGui::TextDisabled("score %.3f | agree %.0f%% | judge %.1f/10 (%d) | %d tok",
                            a.score, 100.0 * a.cluster_mass, a.judge_score,
                            a.judge_votes, a.gen_tokens);
        if (a.reused_tokens > 0) {
          ImGui::SameLine();
          ImGui::TextColored(ImVec4(0.55f, 0.9f, 0.6f, 1.0f), "| %d cached",
                             a.reused_tokens);
        }
        if (!a.critique.empty()) {
          const ImVec4 amber(1.0f, 0.85f, 0.45f, 1.0f);
          ui_wrapped("critique: " + a.critique, &amber);
        }
        ui_wrapped(a.text);
        ImGui::Separator();
        ImGui::PopID();
      }
      ImGui::TreePop();
    }
    if (!live->decision_log.empty()) {
      const ImVec4 dim(0.65f, 0.8f, 0.95f, 1.0f);
      ui_wrapped(live->decision_log, &dim);
    }
    if (!live->usage.empty()) {
      for (const DebateTranscript::Usage& u : live->usage)
        ImGui::TextDisabled("%s: %d calls, %d generated, %d reused, %.2fs",
                            u.backend_id.c_str(), u.calls, u.gen_tokens,
                            u.reused_tokens, u.seconds);
    }
  } else if (!snap.partial.empty()) {
    const ImVec4 amber(1.0f, 0.85f, 0.4f, 1.0f);
    ui_wrapped(snap.partial, &amber);
  } else if (!snap.history.empty()) {
    const AgentTurn& t = snap.history.back();
    ImGui::TextColored(ImVec4(0.55f, 0.75f, 1.0f, 1.0f), "you:");
    ImGui::SameLine();
    ui_wrapped(t.question);
    ImGui::TextColored(ImVec4(0.65f, 0.95f, 0.7f, 1.0f), "%s:",
                       ask_mode_name(t.mode));
    ui_wrapped(t.answer.empty() ? ("(" + t.error + ")") : t.answer);
    ImGui::TextDisabled("%.2fs, %d prompt (%d reused), %d generated", t.seconds,
                        t.prompt_tokens, t.reused_tokens, t.gen_tokens);
    if (!t.tools.empty()) {
      for (const ToolTrace& tt : t.tools)
        ImGui::TextDisabled("  tool %s %s -> %s (%.2fs)", tt.tool.c_str(),
                            tt.args.c_str(),
                            tt.denied ? "denied" : (tt.ok ? "ok" : "failed"),
                            tt.seconds);
    }
    if (!t.context_used.empty()) {
      ImGui::Checkbox("show retrieved context", &ui.show_context);
      if (ui.show_context) {
        const ImVec4 dim(0.6f, 0.7f, 0.8f, 1.0f);
        ui_wrapped(t.context_used, &dim);
      }
    }
  } else {
    ImGui::TextDisabled(
        "ask something. 'fast' uses SPT alone; 'debate' has both models answer, "
        "criticise each other and synthesise, weighted by the multipliers.");
  }
  if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 40.0f) ImGui::SetScrollHereY(1.0f);
  ImGui::EndChild();

  ImGui::SetNextItemWidth(-150);
  const bool enter = ImGui::InputText("##agentin", ui.input, sizeof(ui.input),
                                      ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  const bool send = ImGui::Button("ask", ImVec2(60, 0));
  ImGui::SameLine();
  if (ImGui::Button("clear", ImVec2(60, 0))) ctx.actrl->clear_history();
  if ((enter || send) && ui.input[0] != '\0' && !snap.busy) {
    ctx.actrl->ask(ui.input, mode_of(ui.mode), ui.fast_mult, ui.strong_mult, ui.voices,
                   ui.use_tools, ui.use_codebase, ui.max_tokens);
    ui.input[0] = '\0';
  }
  if (ui.input[0] != '\0' && needs_shaping(ui.input)) ui_line(ui.input);
}

// Codebase panel: index a folder, see what is in the index, and search it.
void codebase_panel(DashboardContext& ctx, AgentUi& ui) {
  if (!ctx.agent || !ctx.actrl) {
    ImGui::TextDisabled("agent runtime unavailable");
    return;
  }
  if (ui.index_root[0] == '\0')
    std::snprintf(ui.index_root, sizeof(ui.index_root), "%s",
                  ctx.agent->options().workspace.c_str());
  ImGui::SetNextItemWidth(-150);
  ImGui::InputText("folder", ui.index_root, sizeof(ui.index_root));
  ImGui::SameLine();
  const IndexJob job = ctx.actrl->index_job();
  if (job.running) {
    if (ImGui::Button("stop", ImVec2(60, 0))) ctx.actrl->stop_jobs();
  } else if (ImGui::Button("index", ImVec2(60, 0))) {
    ctx.actrl->start_index(ui.index_root);
  }
  if (job.running) {
    const float f = job.files_total > 0
                        ? static_cast<float>(job.files_done) / static_cast<float>(job.files_total)
                        : 0.0f;
    ImGui::ProgressBar(f, ImVec2(-1, 0));
    ImGui::TextDisabled("%lld/%lld  %s", static_cast<long long>(job.files_done),
                        static_cast<long long>(job.files_total),
                        utf8_truncate(job.current, 60).c_str());
  } else if (!job.error.empty()) {
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.45f, 1.0f), "%s", job.error.c_str());
  }

  const IndexStats st = ctx.agent->codebase().stats();
  if (st.chunks > 0) {
    ImGui::Text("%lld files, %lld chunks, %lld tokens", static_cast<long long>(st.files),
                static_cast<long long>(st.chunks), static_cast<long long>(st.tokens));
    ImGui::TextDisabled("%s embeddings, %.1f MiB, %.2fs scan + %.2fs embed",
                        st.embedder.c_str(), st.memory_bytes / (1024.0 * 1024.0),
                        st.scan_seconds, st.embed_seconds);
    if (st.skipped_ignored || st.skipped_binary || st.skipped_large)
      ImGui::TextDisabled("skipped: %lld ignored, %lld binary, %lld too large",
                          static_cast<long long>(st.skipped_ignored),
                          static_cast<long long>(st.skipped_binary),
                          static_cast<long long>(st.skipped_large));
    ImGui::Separator();
    ImGui::BeginChild("langs", ImVec2(0, 90), ImGuiChildFlags_Border);
    for (const auto& l : st.by_language)
      ImGui::TextDisabled("  %-12s %lld chunks", l.first.c_str(),
                          static_cast<long long>(l.second));
    ImGui::EndChild();
    ImGui::TextWrapped(
        "The index is used automatically for questions that look like code "
        "questions, and through the code_search / find_symbol / repo_overview "
        "tools for anything else.");
  } else {
    ImGui::TextDisabled("nothing indexed yet");
  }
}

// Dataset panel: point it at a file and the continual thread learns from it.
void dataset_panel(DashboardContext& ctx, AgentUi& ui) {
  if (!ctx.actrl) {
    ImGui::TextDisabled("agent not attached");
    return;
  }
  ImGui::TextWrapped(
      "Give the model a dataset. The file is split into paragraph-sized samples "
      "and fed to the continual-learning thread; the coordinator still validates "
      "every resulting update on the hold-out set and rolls it back if quality "
      "drops, so a bad dataset cannot silently damage the weights.");
  ImGui::Separator();
  ImGui::SetNextItemWidth(-150);
  ImGui::InputText("file", ui.dataset, sizeof(ui.dataset));
  ImGui::SameLine();
  const TrainJob job = ctx.actrl->train_job();
  if (job.running) {
    if (ImGui::Button("stop", ImVec2(60, 0))) ctx.actrl->stop_jobs();
  } else if (ImGui::Button("train", ImVec2(60, 0))) {
    ctx.actrl->teach_dataset(ui.dataset, ui.dataset_max_lines, ui.dataset_chunk);
  }
  ImGui::SetNextItemWidth(150);
  ImGui::InputInt("max lines (0=all)", &ui.dataset_max_lines);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(150);
  ImGui::SliderInt("sample chars", &ui.dataset_chunk, 200, 4000);

  if (job.running || job.done) {
    ImGui::Separator();
    ImGui::Text("%s", utf8_truncate(job.path, 70).c_str());
    ImGui::Text("%lld lines read, %lld samples queued, %s", static_cast<long long>(job.lines),
                static_cast<long long>(job.queued),
                human_bytes(static_cast<double>(job.bytes)).c_str());
    if (job.running) ImGui::TextDisabled("streaming...");
    else ImGui::TextDisabled("finished in %.1fs", job.seconds);
  }
  if (!job.error.empty())
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.45f, 1.0f), "%s", job.error.c_str());

  ImGui::Separator();
  const TrainerState cl = ctx.tel->trainer(Source::kContinual);
  ImGui::Text("continual thread: %lld pending, %lld steps, loss %.4f",
              static_cast<long long>(cl.pending_inputs),
              static_cast<long long>(cl.local_steps), cl.last_loss);
  if (ctx.corpus && !ctx.corpus->empty())
    ImGui::TextDisabled("corpus: %lld tokens across %d sources",
                        static_cast<long long>(ctx.corpus->total_tokens()),
                        ctx.corpus->num_sources());
  ImGui::TextWrapped(
      "For a full pre-training run on a large corpus use the command line "
      "(slm pretrain --data F --steps N): it is much faster per token than the "
      "continual path, which is designed for a steady trickle of new data.");
}

// Tool approval: the confirmation gate, as a modal so it cannot be missed.
void approval_modal(DashboardContext& ctx) {
  if (!ctx.agent) return;
  const std::vector<PendingApproval> pend = ctx.agent->gate().pending();
  if (pend.empty()) return;
  ImGui::OpenPopup("tool approval");
  if (ImGui::BeginPopupModal("tool approval", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    const PendingApproval& p = pend.front();
    ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.4f, 1.0f), "%s wants to run %s",
                       "the agent", p.tool.c_str());
    ImGui::TextDisabled("risk: %s", tool_risk_name(p.risk));
    ImGui::Separator();
    ImGui::PushTextWrapPos(560.0f);
    ImGui::TextUnformatted(p.preview.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Separator();
    if (ImGui::Button("allow", ImVec2(110, 0)))
      ctx.agent->gate().decide(p.id, true, false);
    ImGui::SameLine();
    if (ImGui::Button("allow always", ImVec2(120, 0)))
      ctx.agent->gate().decide(p.id, true, true);
    ImGui::SameLine();
    if (ImGui::Button("deny", ImVec2(110, 0)))
      ctx.agent->gate().decide(p.id, false, false);
    ImGui::SameLine();
    if (ImGui::Button("deny all", ImVec2(110, 0))) ctx.agent->gate().deny_all();
    if (pend.size() > 1)
      ImGui::TextDisabled("%zu more waiting", pend.size() - 1);
    ImGui::EndPopup();
  }
}

// Which tools exist, and the policy for each risk level.
void tools_panel(DashboardContext& ctx) {
  if (!ctx.agent) {
    ImGui::TextDisabled("agent runtime unavailable");
    return;
  }
  ToolPolicy& pol = ctx.agent->policy();
  const char* kModes[3] = {"ask", "allow", "deny"};
  ImGui::TextDisabled("policy per risk level");
  ImGui::SetNextItemWidth(110);
  ImGui::Combo("safe", &pol.safe, kModes, 3);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(110);
  ImGui::Combo("network", &pol.network, kModes, 3);
  ImGui::SetNextItemWidth(110);
  ImGui::Combo("write", &pol.write, kModes, 3);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(110);
  ImGui::Combo("shell", &pol.dangerous, kModes, 3);
  ImGui::Separator();
  for (const ToolSpec& s : ctx.agent->tools().specs()) {
    bool on = s.enabled;
    ImGui::PushID(s.name.c_str());
    if (ImGui::Checkbox("##on", &on)) ctx.agent->tools().set_enabled(s.name, on);
    ImGui::SameLine();
    ImGui::Text("%-14s", s.name.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("[%s] %s", tool_risk_name(s.risk), s.summary.c_str());
    ImGui::PopID();
  }
  ImGui::Separator();
  for (const ToolRegistry::Stat& st : ctx.agent->tools().stats())
    ImGui::TextDisabled("%-14s %lld calls, %lld failed, %lld denied, %.2fs",
                        st.name.c_str(), static_cast<long long>(st.calls),
                        static_cast<long long>(st.failures),
                        static_cast<long long>(st.denials), st.seconds);
}

void chat_panel(DashboardContext& ctx, char* input, size_t input_size) {
  ImGui::BeginChild("history", ImVec2(0, -70), ImGuiChildFlags_Border);
  std::vector<ChatTurn> h = ctx.chat->history();
  for (size_t i = 0; i < h.size(); ++i) {
    ImGui::PushID(static_cast<int>(i));
    ImGui::TextColored(ImVec4(0.55f, 0.75f, 1.0f, 1.0f), "you:");
    ImGui::SameLine();
    ui_wrapped(h[i].prompt);
    ImGui::TextColored(ImVec4(0.65f, 0.95f, 0.7f, 1.0f), "slm:");
    ImGui::SameLine();
    ui_wrapped(h[i].response);
    ImGui::TextDisabled("%d tokens, %.1f tok/s", h[i].tokens, h[i].tokens_per_s);
    ImGui::SameLine();
    if (h[i].rated) {
      ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f), "rated %.1f", h[i].score);
    } else {
      for (int s = 1; s <= 5; ++s) {
        char b[8];
        std::snprintf(b, sizeof(b), "%d", s);
        if (ImGui::SmallButton(b)) ctx.chat->rate(i, static_cast<float>(s));
        if (s < 5) ImGui::SameLine();
      }
    }
    ImGui::Separator();
    ImGui::PopID();
  }
  const std::string partial = ctx.chat->partial();
  if (!partial.empty()) {
    ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.4f, 1.0f), "slm:");
    ImGui::SameLine();
    const ImVec4 amber(1.0f, 0.85f, 0.4f, 1.0f);
    ui_wrapped(partial, &amber);
  }
  if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 40.0f) ImGui::SetScrollHereY(1.0f);
  ImGui::EndChild();

  ImGui::SetNextItemWidth(-90);
  const bool enter =
      ImGui::InputText("##input", input, input_size, ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  const bool send = ImGui::Button("send", ImVec2(80, 0));
  if ((enter || send) && input[0] != '\0') {
    ctx.chat->ask(input);
    input[0] = '\0';
  }
  if (input[0] != '\0' && needs_shaping(input)) {
    ImGui::TextDisabled("preview:");
    ImGui::SameLine();
    ui_line(input);  // the input widget itself cannot shape Persian
  }
  ImGui::TextDisabled(
      "rating a reply feeds the feedback (DPO) thread; every message also feeds "
      "the continual thread");
}

// ============================================================== visual design
// A dark theme with one accent colour, generous padding and rounded surfaces.
// The old layout was six fixed windows fighting over 1600x950; this is a single
// window with a sidebar and one view at a time, which is what makes room for the
// chat to be the main thing rather than a 34%-wide column.
namespace theme {

constexpr ImVec4 kBg = ImVec4(0.055f, 0.063f, 0.082f, 1.00f);
constexpr ImVec4 kSurface = ImVec4(0.086f, 0.098f, 0.125f, 1.00f);
constexpr ImVec4 kSurface2 = ImVec4(0.118f, 0.133f, 0.165f, 1.00f);
constexpr ImVec4 kBorder = ImVec4(0.180f, 0.200f, 0.243f, 1.00f);
constexpr ImVec4 kText = ImVec4(0.878f, 0.898f, 0.925f, 1.00f);
constexpr ImVec4 kTextDim = ImVec4(0.514f, 0.553f, 0.616f, 1.00f);
constexpr ImVec4 kAccent = ImVec4(0.365f, 0.522f, 0.988f, 1.00f);
constexpr ImVec4 kAccentDim = ImVec4(0.365f, 0.522f, 0.988f, 0.24f);
constexpr ImVec4 kGood = ImVec4(0.290f, 0.800f, 0.510f, 1.00f);
constexpr ImVec4 kWarn = ImVec4(0.980f, 0.741f, 0.286f, 1.00f);
constexpr ImVec4 kBad = ImVec4(0.937f, 0.376f, 0.376f, 1.00f);
constexpr ImVec4 kUser = ImVec4(0.475f, 0.702f, 0.996f, 1.00f);
constexpr ImVec4 kModel = ImVec4(0.427f, 0.867f, 0.706f, 1.00f);

void apply() {
  ImGuiStyle& s = ImGui::GetStyle();
  s.WindowRounding = 0.0f;
  s.ChildRounding = 10.0f;
  s.FrameRounding = 8.0f;
  s.PopupRounding = 10.0f;
  s.ScrollbarRounding = 10.0f;
  s.GrabRounding = 8.0f;
  s.TabRounding = 8.0f;
  s.WindowPadding = ImVec2(18, 16);
  s.FramePadding = ImVec2(12, 7);
  s.ItemSpacing = ImVec2(10, 9);
  s.ItemInnerSpacing = ImVec2(8, 6);
  s.ScrollbarSize = 11.0f;
  s.GrabMinSize = 11.0f;
  s.WindowBorderSize = 0.0f;
  s.ChildBorderSize = 1.0f;
  s.FrameBorderSize = 0.0f;
  s.SeparatorTextBorderSize = 1.0f;
  s.SeparatorTextPadding = ImVec2(18, 6);
  s.WindowTitleAlign = ImVec2(0.0f, 0.5f);

  ImVec4* c = s.Colors;
  c[ImGuiCol_WindowBg] = kBg;
  c[ImGuiCol_ChildBg] = kSurface;
  c[ImGuiCol_PopupBg] = kSurface2;
  c[ImGuiCol_Border] = kBorder;
  c[ImGuiCol_Text] = kText;
  c[ImGuiCol_TextDisabled] = kTextDim;
  c[ImGuiCol_FrameBg] = kSurface2;
  c[ImGuiCol_FrameBgHovered] = ImVec4(0.157f, 0.176f, 0.216f, 1.0f);
  c[ImGuiCol_FrameBgActive] = ImVec4(0.196f, 0.220f, 0.271f, 1.0f);
  c[ImGuiCol_TitleBg] = kSurface;
  c[ImGuiCol_TitleBgActive] = kSurface;
  c[ImGuiCol_MenuBarBg] = kSurface;
  c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
  c[ImGuiCol_ScrollbarGrab] = ImVec4(0.243f, 0.271f, 0.325f, 1.0f);
  c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.310f, 0.345f, 0.408f, 1.0f);
  c[ImGuiCol_CheckMark] = kAccent;
  c[ImGuiCol_SliderGrab] = kAccent;
  c[ImGuiCol_SliderGrabActive] = ImVec4(0.478f, 0.616f, 1.0f, 1.0f);
  c[ImGuiCol_Button] = kSurface2;
  c[ImGuiCol_ButtonHovered] = ImVec4(0.196f, 0.227f, 0.290f, 1.0f);
  c[ImGuiCol_ButtonActive] = ImVec4(0.243f, 0.282f, 0.361f, 1.0f);
  c[ImGuiCol_Header] = kAccentDim;
  c[ImGuiCol_HeaderHovered] = ImVec4(0.365f, 0.522f, 0.988f, 0.38f);
  c[ImGuiCol_HeaderActive] = ImVec4(0.365f, 0.522f, 0.988f, 0.52f);
  c[ImGuiCol_Separator] = kBorder;
  c[ImGuiCol_SeparatorHovered] = kAccent;
  c[ImGuiCol_Tab] = kSurface2;
  c[ImGuiCol_TabHovered] = kAccentDim;
  c[ImGuiCol_TabSelected] = ImVec4(0.365f, 0.522f, 0.988f, 0.42f);
  c[ImGuiCol_PlotLines] = kAccent;
  c[ImGuiCol_PlotHistogram] = kAccent;
  c[ImGuiCol_TableHeaderBg] = kSurface2;
  c[ImGuiCol_TableBorderLight] = kBorder;
  c[ImGuiCol_TableBorderStrong] = kBorder;
  c[ImGuiCol_ModalWindowDimBg] = ImVec4(0.02f, 0.02f, 0.03f, 0.72f);
}

// A rounded status pill.  Used everywhere a boolean or a short value has to be
// legible at a glance rather than read.
void pill(const char* text, const ImVec4& col, bool filled = false) {
  const ImVec2 sz = ImGui::CalcTextSize(text);
  const ImVec2 p = ImGui::GetCursorScreenPos();
  const float h = sz.y + 8.0f, w = sz.x + 18.0f;
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImU32 bg = ImGui::ColorConvertFloat4ToU32(
      filled ? col : ImVec4(col.x, col.y, col.z, 0.16f));
  dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), bg, h * 0.5f);
  if (!filled)
    dl->AddRect(p, ImVec2(p.x + w, p.y + h),
                ImGui::ColorConvertFloat4ToU32(ImVec4(col.x, col.y, col.z, 0.45f)),
                h * 0.5f);
  dl->AddText(ImVec2(p.x + 9.0f, p.y + 4.0f),
              ImGui::ColorConvertFloat4ToU32(filled ? ImVec4(0.05f, 0.06f, 0.08f, 1.0f)
                                                   : col),
              text);
  ImGui::Dummy(ImVec2(w, h));
}

void heading(const char* text) {
  ImGui::PushStyleColor(ImGuiCol_Text, kText);
  ImGui::TextUnformatted(text);
  ImGui::PopStyleColor();
  ImGui::Spacing();
}

void muted(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
  ImGui::TextV(fmt, ap);
  ImGui::PopStyleColor();
  va_end(ap);
}

// A card: a titled child window with padding.  Returns false when collapsed
// (never, currently) so call sites read like a scope.
bool begin_card(const char* id, const ImVec2& size, const char* title = nullptr) {
  ImGui::PushStyleColor(ImGuiCol_ChildBg, kSurface);
  ImGui::BeginChild(id, size, ImGuiChildFlags_Border);
  ImGui::PopStyleColor();
  if (title) {
    ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    ImGui::Spacing();
  }
  return true;
}
void end_card() { ImGui::EndChild(); }

}  // namespace theme

// ================================================================= navigation
enum class View { kChat = 0, kDebate, kModels, kCodebase, kTraining, kMemory, kLogs };

struct NavItem {
  View view;
  const char* label;
  const char* hint;
};
const NavItem kNav[] = {
    {View::kChat, "Chat", "ask, with web + codebase + shell"},
    {View::kDebate, "Debate", "both models argue, then synthesise"},
    {View::kModels, "Models", "SPT and OLMo: load, size, speed"},
    {View::kCodebase, "Codebase", "index a folder and search it"},
    {View::kTraining, "Training", "self-training (off by default)"},
    {View::kMemory, "Memory", "long term memory"},
    {View::kLogs, "Logs", "audit trail"},
};

// One compact status card per backend, shown in the sidebar.
void sidebar_model_card(DashboardContext& ctx, const BackendPtr& b) {
  const BackendStatus s = b->status();
  ImGui::PushID(b->id().c_str());
  theme::begin_card((b->id() + "_card").c_str(), ImVec2(-1, 74));
  ImGui::PushStyleColor(ImGuiCol_Text, s.loaded ? theme::kModel : theme::kTextDim);
  ImGui::TextUnformatted(b->display_name().c_str());
  ImGui::PopStyleColor();
  if (s.loaded) {
    theme::muted("%.1fM  %s", s.params / 1e6,
                 human_bytes(static_cast<double>(s.weight_bytes)).c_str());
    if (s.last_decode_tps > 0.0)
      theme::muted("%.0f tok/s", s.last_decode_tps);
    else
      theme::muted("ready");
  } else {
    theme::muted("not loaded");
    if (ImGui::SmallButton("load")) {
      std::string err;
      if (!b->load(&err) && ctx.tel)
        ctx.tel->log("warn", "agent", "load failed: " + err);
    }
  }
  theme::end_card();
  ImGui::PopID();
}

void sidebar(DashboardContext& ctx, View* view, double uptime) {
  ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kSurface);
  ImGui::BeginChild("sidebar", ImVec2(228, -1), ImGuiChildFlags_None);
  ImGui::PopStyleColor();

  ImGui::PushStyleColor(ImGuiCol_Text, theme::kAccent);
  ImGui::SetWindowFontScale(1.25f);
  ImGui::TextUnformatted("SPT");
  ImGui::SetWindowFontScale(1.0f);
  ImGui::PopStyleColor();
  ImGui::SameLine();
  theme::muted("self-improving");
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  for (const NavItem& n : kNav) {
    const bool sel = (*view == n.view);
    if (sel) {
      ImGui::PushStyleColor(ImGuiCol_Button, theme::kAccentDim);
      ImGui::PushStyleColor(ImGuiCol_Text, theme::kAccent);
    } else {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
      ImGui::PushStyleColor(ImGuiCol_Text, theme::kText);
    }
    if (ImGui::Button(n.label, ImVec2(-1, 36))) *view = n.view;
    ImGui::PopStyleColor(2);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", n.hint);
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  if (ctx.agent)
    for (const BackendPtr& b : ctx.agent->backends().all()) sidebar_model_card(ctx, b);

  // Bottom block: the global stop and the session facts.
  const float bottom = 132.0f;
  ImGui::Dummy(ImVec2(1, std::max(0.0f, ImGui::GetContentRegionAvail().y - bottom)));
  const bool stopped = ctx.tel->stopped();
  ImGui::PushStyleColor(ImGuiCol_Button,
                        stopped ? ImVec4(0.15f, 0.45f, 0.22f, 1.0f)
                                : ImVec4(0.62f, 0.14f, 0.14f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                        stopped ? ImVec4(0.20f, 0.55f, 0.28f, 1.0f)
                                : ImVec4(0.80f, 0.18f, 0.18f, 1.0f));
  if (ImGui::Button(stopped ? "RESUME TRAINING" : "EMERGENCY STOP", ImVec2(-1, 40))) {
    if (stopped) ctx.tel->clear_emergency_stop();
    else ctx.tel->emergency_stop();
  }
  ImGui::PopStyleColor(2);
  theme::muted("%.0f M params  |  up %.0fs", ctx.mcfg->param_count() / 1e6, uptime);
  theme::muted("%s", backend_name());
  if (ImGui::Button("quit", ImVec2(-1, 28))) ctx.quit->store(true);
  ImGui::EndChild();
}

// ==================================================================== chat view
// The main event.  Everything the agent can do is reachable from here: the model
// dropdown chooses SPT, OLMo or a debate, and the two toggles decide whether a
// question may reach the web and the indexed codebase.
void chat_view(DashboardContext& ctx, AgentUi& ui) {
  if (!ctx.actrl || !ctx.actrl->attached()) {
    ImGui::TextColored(theme::kBad, "the agent runtime is not available");
    theme::muted("check the audit log for why the model failed to load");
    return;
  }
  const AgentSnapshot snap = ctx.actrl->snapshot();

  // ---- header: model + capabilities
  theme::begin_card("chathdr", ImVec2(-1, 56));
  ImGui::AlignTextToFramePadding();
  theme::muted("model");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(210);
  ImGui::Combo("##mode", &ui.mode, kModeNames, 4);
  const AskMode m = mode_of(ui.mode);
  if ((m == AskMode::kStrong || m == AskMode::kDebate) && ctx.agent &&
      !ctx.agent->mode_available(m)) {
    ImGui::SameLine();
    theme::pill("OLMo not installed", theme::kWarn);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("run:  slm fetch-model olmo");
  }
  if (m == AskMode::kDebate) {
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::SliderInt("SPT x", &ui.fast_mult, 1, 5);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::SliderInt("OLMo x", &ui.strong_mult, 1, 5);
  } else if (m == AskMode::kSelfDebate) {
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::SliderInt("voices", &ui.voices, 2, 4);
  }
  ImGui::SameLine(0, 20);
  ImGui::Checkbox("web + shell", &ui.use_tools);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip(
        "lets the answer use web_search, web_fetch, read_file, list_dir and "
        "shell.  Anything risky still asks you first.");
  ImGui::SameLine();
  ImGui::Checkbox("codebase", &ui.use_codebase);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("retrieves from the indexed folder before answering");
  theme::end_card();

  if (!shaping_ready()) {
    ImGui::PushStyleColor(ImGuiCol_Text, theme::kWarn);
    ImGui::TextWrapped(
        "No Arabic-capable font found, so Persian will render as '???'. Install "
        "one and restart:  Debian/Ubuntu: fonts-noto-core  |  Fedora: "
        "google-noto-sans-arabic-fonts  |  Arch: noto-fonts");
    ImGui::PopStyleColor();
    ImGui::Spacing();
  }

  // ---- transcript
  const float input_h = 86.0f;
  theme::begin_card("chatbody", ImVec2(-1, -input_h));
  for (size_t i = 0; i < snap.history.size(); ++i) {
    const AgentTurn& t = snap.history[i];
    ImGui::PushID(static_cast<int>(i));
    ImGui::PushStyleColor(ImGuiCol_Text, theme::kUser);
    ImGui::TextUnformatted("you");
    ImGui::PopStyleColor();
    ui_wrapped(t.question);
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, theme::kModel);
    ImGui::TextUnformatted(t.was_debate ? "debate" : (t.mode == AskMode::kStrong ? "OLMo" : "SPT"));
    ImGui::PopStyleColor();
    if (!t.tools.empty()) {
      for (const ToolTrace& tt : t.tools) {
        const ImVec4 col = tt.denied ? theme::kWarn : (tt.ok ? theme::kTextDim : theme::kBad);
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        ImGui::Text("  %s %s  (%.2fs)%s", tt.tool.c_str(),
                    utf8_truncate(tt.args, 60).c_str(), tt.seconds,
                    tt.denied ? "  denied" : (tt.ok ? "" : "  failed"));
        ImGui::PopStyleColor();
      }
    }
    ui_wrapped(t.answer.empty() ? ("(" + t.error + ")") : t.answer);
    theme::muted("%.2fs  |  %d prompt (%d cached)  |  %d generated", t.seconds,
                 t.prompt_tokens, t.reused_tokens, t.gen_tokens);
    // Ratings feed the feedback (DPO) thread, exactly as the old chat did.
    ImGui::SameLine();
    if (ctx.chat) {
      for (int sc = 1; sc <= 5; ++sc) {
        char b[8];
        std::snprintf(b, sizeof(b), "%d", sc);
        if (ImGui::SmallButton(b)) {
          RatedSample rs;
          rs.prompt = t.question;
          rs.response = t.answer;
          rs.score = static_cast<float>(sc);
          rs.t = Telemetry::now();
          ctx.hub->push_rating(rs);
          ctx.tel->log("info", "feedback", "rated an agent answer",
                       {{"score", std::to_string(sc)}});
        }
        if (sc < 5) ImGui::SameLine();
      }
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::PopID();
  }
  if (snap.busy) {
    ImGui::PushStyleColor(ImGuiCol_Text, theme::kUser);
    ImGui::TextUnformatted("you");
    ImGui::PopStyleColor();
    ui_wrapped(snap.question);
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, theme::kWarn);
    ImGui::TextUnformatted(snap.progress > 0.0 ? "debating..." : "thinking...");
    ImGui::PopStyleColor();
    if (snap.progress > 0.0) {
      ImGui::ProgressBar(static_cast<float>(snap.progress), ImVec2(-1, 6), "");
      if (!snap.live.rounds.empty()) {
        const DebateRound& r = snap.live.rounds.back();
        theme::muted("round %d (%s), %zu answers", r.index, r.kind.c_str(),
                     r.answers.size());
      }
    }
    if (!snap.partial.empty()) {
      const ImVec4 amber = theme::kWarn;
      ui_wrapped(snap.partial, &amber);
    }
  }
  if (snap.history.empty() && !snap.busy) {
    const ImVec4 dim = theme::kTextDim;
    theme::muted("Ask anything - Persian, English or Python.");
    ImGui::Spacing();
    ui_wrapped("۲۳۴ ضربدر ۵۶ چند است؟", &dim);
    ui_wrapped("این پروژه چه کاری انجام می‌دهد؟", &dim);
    theme::muted("        (needs an indexed folder)");
    theme::muted("latest release of CMake?");
    theme::muted("        (needs web enabled)");
  }
  if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 40.0f) ImGui::SetScrollHereY(1.0f);
  theme::end_card();

  // ---- input
  ImGui::Spacing();
  const bool busy = snap.busy;
  ImGui::SetNextItemWidth(-190);
  const bool enter = ImGui::InputTextWithHint(
      "##ask", "ask SPT anything...", ui.input, sizeof(ui.input),
      ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  bool send = false;
  if (busy) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.16f, 0.16f, 1.0f));
    if (ImGui::Button("stop", ImVec2(84, 0))) ctx.actrl->cancel();
    ImGui::PopStyleColor();
  } else {
    ImGui::PushStyleColor(ImGuiCol_Button, theme::kAccent);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.05f, 0.06f, 0.08f, 1.0f));
    send = ImGui::Button("send", ImVec2(84, 0));
    ImGui::PopStyleColor(2);
  }
  ImGui::SameLine();
  if (ImGui::Button("clear", ImVec2(76, 0))) ctx.actrl->clear_history();
  if ((enter || send) && ui.input[0] != '\0' && !busy) {
    ctx.actrl->ask(ui.input, m, ui.fast_mult, ui.strong_mult, ui.voices, ui.use_tools,
                   ui.use_codebase, ui.max_tokens);
    ui.input[0] = '\0';
  }
  if (ui.input[0] != '\0' && needs_shaping(ui.input)) {
    theme::muted("preview:");
    ImGui::SameLine();
    ui_line(ui.input);
  }
}

// ================================================================ models view
void models_view(DashboardContext& ctx, AgentUi& ui) {
  if (!ctx.agent) {
    ImGui::TextColored(theme::kBad, "the agent runtime is not available");
    return;
  }
  theme::heading("Models");
  theme::muted(
      "SPT is your model: it trains, it can be taught, and it is the one the "
      "self-training threads improve.  OLMo is the ready-made model: no training, "
      "much stronger, much slower.");
  ImGui::Spacing();

  for (const BackendPtr& b : ctx.agent->backends().all()) {
    const BackendStatus s = b->status();
    ImGui::PushID(b->id().c_str());
    theme::begin_card((b->id() + "_big").c_str(), ImVec2(-1, 168));
    ImGui::SetWindowFontScale(1.15f);
    ImGui::PushStyleColor(ImGuiCol_Text, s.loaded ? theme::kModel : theme::kTextDim);
    ImGui::TextUnformatted(b->display_name().c_str());
    ImGui::PopStyleColor();
    ImGui::SetWindowFontScale(1.0f);
    ImGui::SameLine();
    theme::pill(s.loaded ? "loaded" : "not loaded",
                s.loaded ? theme::kGood : theme::kTextDim, s.loaded);
    ImGui::SameLine();
    theme::pill(b->runtime().c_str(), theme::kAccent);
    if (b->caps().trainable) {
      ImGui::SameLine();
      theme::pill("trainable", theme::kWarn);
    }
    ImGui::Spacing();
    if (s.loaded) {
      ImGui::Text("%s", s.detail.c_str());
      theme::muted("%lld parameters  |  %s weights  |  %s KV cache",
                   static_cast<long long>(s.params),
                   human_bytes(static_cast<double>(s.weight_bytes)).c_str(),
                   human_bytes(static_cast<double>(s.kv_bytes)).c_str());
      theme::muted("%lld calls  |  %.0f tok/s decode  |  %.0f tok/s prompt  |  "
                   "%lld tokens served from cache",
                   static_cast<long long>(s.calls), s.last_decode_tps,
                   s.last_prompt_tps, static_cast<long long>(s.cache_hit_tokens));
      if (!b->caps().trainable && ImGui::Button("unload", ImVec2(120, 0))) b->unload();
    } else {
      if (!s.error.empty()) ImGui::TextColored(theme::kBad, "%s", s.error.c_str());
      else theme::muted("%s", s.detail.c_str());
      ImGui::Spacing();
      if (ImGui::Button("load now", ImVec2(140, 0))) {
        std::string err;
        if (!b->load(&err))
          ctx.tel->log("warn", "agent", "load failed: " + err);
      }
      if (b->id() == "olmo") {
        ImGui::SameLine();
        theme::muted("or from a terminal:  slm fetch-model olmo");
      }
    }
    theme::end_card();
    ImGui::PopID();
    ImGui::Spacing();
  }

  theme::begin_card("membudget", ImVec2(-1, 108), "memory");
  const size_t w = ctx.agent->backends().weight_bytes();
  const size_t kv = ctx.agent->backends().kv_bytes();
  ImGui::Text("weights %s   +   KV cache %s",
              human_bytes(static_cast<double>(w)).c_str(),
              human_bytes(static_cast<double>(kv)).c_str());
  if (!ctx.agent->codebase().empty()) {
    const IndexStats st = ctx.agent->codebase().stats();
    ImGui::Text("codebase index %s  (%lld chunks)",
                human_bytes(static_cast<double>(st.memory_bytes)).c_str(),
                static_cast<long long>(st.chunks));
  }
  theme::muted("GGUF weights are memory mapped, so the pages are shared and "
               "evictable rather than counted twice.");
  theme::end_card();
  ImGui::Spacing();
  ImGui::SetNextItemWidth(220);
  ImGui::SliderInt("answer length (tokens)", &ui.max_tokens, 64, 1024);
}

// =============================================================== training view
// Self-training is OFF by default; this is where it is turned on, and the switch
// says plainly what it will do.
void training_view(DashboardContext& ctx, AgentUi& ui, bool* visible, bool* log_scale,
                   float* window_seconds, int* layer, int* head) {
  theme::heading("Self-training");
  const bool on = ctx.tel->self_training_enabled();
  theme::begin_card("swcard", ImVec2(-1, 96));
  bool sw = on;
  if (ImGui::Checkbox("  let the model keep training itself", &sw))
    ctx.tel->set_self_training_enabled(sw);
  theme::muted(
      "Off by default.  When on, three threads learn from your chats, from their "
      "own filtered output and from your 1-5 ratings, and a coordinator validates "
      "every change on a hold-out set before it is published - a change that makes "
      "quality worse is rolled back automatically.");
  ImGui::SameLine();
  theme::pill(on ? "running" : "paused", on ? theme::kGood : theme::kTextDim, on);
  theme::end_card();
  ImGui::Spacing();

  if (ImGui::BeginTabBar("traintabs")) {
    if (ImGui::BeginTabItem("dataset")) {
      dataset_panel(ctx, ui);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("loss")) {
      for (int i = 0; i < kNumStreams; ++i) {
        legend(stream_name(static_cast<Stream>(i)), kColors[i], &visible[i]);
        if (i + 1 < kNumStreams) ImGui::SameLine();
      }
      ImGui::SetNextItemWidth(180);
      ImGui::SliderFloat("window (s)", window_seconds, 20.0f, 1800.0f, "%.0f");
      ImGui::SameLine();
      ImGui::Checkbox("log10", log_scale);
      plot_losses(*ctx.tel, ImGui::GetContentRegionAvail().y - 8.0f, *window_seconds,
                  *log_scale, visible);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("threads")) {
      for (int i = 0; i < kNumSources; ++i) trainer_panel(ctx, static_cast<Source>(i));
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("coordinator")) {
      coordinator_panel(ctx);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("attention")) {
      attention_panel(*ctx.tel, *ctx.mcfg, layer, head);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("next token")) {
      token_panel(*ctx.tel);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

}  // namespace

bool gui_available() { return true; }

int run_imgui_dashboard(DashboardContext& ctx) {
  glfwSetErrorCallback(glfw_error);
  if (!glfwInit()) {
    std::fprintf(stderr, "cannot initialise GLFW (no display?), falling back\n");
    return run_terminal_dashboard(ctx);
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* win = glfwCreateWindow(1600, 950, "slm - hybrid self-training monitor",
                                     nullptr, nullptr);
  if (!win) {
    std::fprintf(stderr, "cannot create a window, falling back to the terminal dashboard\n");
    glfwTerminate();
    return run_terminal_dashboard(ctx);
  }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  ImGui::StyleColorsDark();
  theme::apply();
  ImGui_ImplGlfw_InitForOpenGL(win, true);
  ImGui_ImplOpenGL3_Init("#version 150");

  // Best effort: use a system font with wider glyph coverage when one exists.
  // NOTE: the path *must* be checked first - ImGui 1.91 raises an internal
  // error (and crashes outside of a frame) when the file cannot be opened.
  for (const char* path : {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                           "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                           "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                           "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"}) {
    if (!std::filesystem::exists(path)) continue;
    if (io.Fonts->AddFontFromFileTTF(path, 18.5f)) break;
  }

  // The shaper needs a live GL context, so it is initialised here.
  {
    std::vector<std::string> fonts;
    if (ctx.opt && !ctx.opt->cfg.get_str("gui.font", "").empty())
      fonts.push_back(ctx.opt->cfg.get_str("gui.font", ""));
    if (g_shaper.init(fonts, 18.0f))
      std::printf("persian text shaping: %s  (latin fallback: %s)\n",
                  g_shaper.font().c_str(),
                  g_shaper.latin_font().empty() ? "none" : g_shaper.latin_font().c_str());
    else
      std::printf("persian text shaping unavailable (%s)\n",
                  text_shaping_compiled() ? "no suitable font found"
                                          : "built without freetype/harfbuzz");
  }

  bool visible[kNumStreams] = {};
  for (int i = 0; i < kNumStreams; ++i) visible[i] = true;
  bool log_scale = false;
  float window_seconds = 180.0f;
  int layer = 0, head = 0;
  char mem_input[512] = {0};
  AgentUi aui;
  View view = View::kChat;
  const double t0 = Telemetry::now();

  while (!glfwWindowShouldClose(win) && !ctx.quit->load()) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    const ImVec2 disp = io.DisplaySize;

    // One full-screen window: a sidebar plus whichever view is selected.  Six
    // competing fixed windows is what made the old dashboard feel cramped.
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(disp, ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14, 14));
    ImGui::Begin("##shell", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);
    sidebar(ctx, &view, Telemetry::now() - t0);
    ImGui::SameLine();

    ImGui::BeginChild("main", ImVec2(-1, -1), ImGuiChildFlags_None);
    switch (view) {
      case View::kChat:
        chat_view(ctx, aui);
        break;
      case View::kDebate:
        debate_panel(ctx, aui);
        break;
      case View::kModels:
        models_view(ctx, aui);
        break;
      case View::kCodebase:
        codebase_panel(ctx, aui);
        break;
      case View::kTraining:
        training_view(ctx, aui, visible, &log_scale, &window_seconds, &layer, &head);
        break;
      case View::kMemory:
        theme::heading("Long term memory");
        theme::muted(
            "What you write here is injected into the prompt; \"teach the weights\" "
            "also queues it for the continual thread so it outlives the context "
            "window.");
        ImGui::Spacing();
        memory_panel(ctx, mem_input, sizeof(mem_input));
        break;
      case View::kLogs:
        theme::heading("Audit log");
        theme::muted("every self-training decision and every tool call, also "
                     "appended to %s/audit.jsonl",
                     ctx.opt ? ctx.opt->workdir.c_str() : "runs");
        ImGui::Spacing();
        theme::begin_card("logbody", ImVec2(-1, -1));
        for (const std::string& l : ctx.tel->recent_logs(300)) {
          ImVec4 col = theme::kTextDim;
          if (l.find(" accept ") != std::string::npos) col = theme::kGood;
          else if (l.find(" reject ") != std::string::npos) col = theme::kBad;
          else if (l.find(" warn ") != std::string::npos) col = theme::kWarn;
          else if (l.find(" error ") != std::string::npos) col = theme::kBad;
          else if (l.find(" tool ") != std::string::npos) col = theme::kAccent;
          else if (l.find(" agent ") != std::string::npos) col = theme::kModel;
          ui_wrapped(l, &col);
        }
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 40.0f)
          ImGui::SetScrollHereY(1.0f);
        theme::end_card();
        break;
    }
    ImGui::EndChild();
    ImGui::End();
    ImGui::PopStyleVar();
    approval_modal(ctx);

    ImGui::Render();
    int w = 0, h = 0;
    glfwGetFramebufferSize(win, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.07f, 0.08f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(win);

    if (ctx.opt->seconds > 0.0 && Telemetry::now() - t0 > ctx.opt->seconds) break;
  }
  ctx.quit->store(true);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(win);
  glfwTerminate();
  return 0;
}

}  // namespace slm

#else  // SLM_WITH_GUI

namespace slm {
bool gui_available() { return false; }
int run_imgui_dashboard(DashboardContext& ctx) { return run_terminal_dashboard(ctx); }
}  // namespace slm

#endif
