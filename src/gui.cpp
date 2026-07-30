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
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

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
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 6.0f;
  style.FrameRounding = 4.0f;
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
    if (io.Fonts->AddFontFromFileTTF(path, 17.0f)) break;
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
  char input[1024] = {0};
  const double t0 = Telemetry::now();

  while (!glfwWindowShouldClose(win) && !ctx.quit->load()) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    const ImVec2 disp = io.DisplaySize;
    const float col_w = disp.x * 0.34f;

    // ------------------------------------------------------------- header
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(disp.x, 62), ImGuiCond_Always);
    ImGui::Begin("header", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);
    const CoordinatorStats cs = ctx.coord->stats();
    ImGui::Text("%s   |   backend %s   |   weights v%llu   |   uptime %.0fs   |   tensors %.1f MiB",
                ctx.mcfg->describe().c_str(), backend_name(),
                static_cast<unsigned long long>(cs.weight_version), Telemetry::now() - t0,
                backend_allocated_bytes() / (1024.0 * 1024.0));
    ImGui::SameLine(disp.x - 430);
    if (ctx.tel->stopped()) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.45f, 0.2f, 1.0f));
      if (ImGui::Button("RESUME TRAINING", ImVec2(200, 34))) ctx.tel->clear_emergency_stop();
      ImGui::PopStyleColor();
    } else {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.70f, 0.10f, 0.10f, 1.0f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.90f, 0.15f, 0.15f, 1.0f));
      if (ImGui::Button("EMERGENCY STOP", ImVec2(200, 34))) ctx.tel->emergency_stop();
      ImGui::PopStyleColor(2);
    }
    ImGui::SameLine();
    if (ImGui::Button("restore best", ImVec2(120, 34))) ctx.coord->restore_best();
    ImGui::SameLine();
    if (ImGui::Button("quit", ImVec2(70, 34))) ctx.quit->store(true);
    ImGui::End();

    // ---------------------------------------------------------- loss chart
    ImGui::SetNextWindowPos(ImVec2(0, 62), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(disp.x - 2 * col_w, 330), ImGuiCond_Always);
    ImGui::Begin("training loss", nullptr, ImGuiWindowFlags_NoMove);
    for (int i = 0; i < kNumStreams; ++i) {
      legend(stream_name(static_cast<Stream>(i)), kColors[i], &visible[i]);
      if (i + 1 < kNumStreams) ImGui::SameLine();
    }
    ImGui::SetNextItemWidth(200);
    ImGui::SliderFloat("window (s)", &window_seconds, 20.0f, 1800.0f, "%.0f");
    ImGui::SameLine();
    ImGui::Checkbox("log10", &log_scale);
    plot_losses(*ctx.tel, ImGui::GetContentRegionAvail().y - 8.0f, window_seconds, log_scale,
                visible);
    ImGui::End();

    // ----------------------------------------------------------- attention
    ImGui::SetNextWindowPos(ImVec2(0, 392), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(disp.x - 2 * col_w, disp.y - 392), ImGuiCond_Always);
    ImGui::Begin("attention heat map", nullptr, ImGuiWindowFlags_NoMove);
    attention_panel(*ctx.tel, *ctx.mcfg, &layer, &head);
    ImGui::End();

    // -------------------------------------------------- self-training panel
    ImGui::SetNextWindowPos(ImVec2(disp.x - 2 * col_w, 62), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(col_w, disp.y - 62), ImGuiCond_Always);
    ImGui::Begin("self-training", nullptr, ImGuiWindowFlags_NoMove);
    for (int i = 0; i < kNumSources; ++i) trainer_panel(ctx, static_cast<Source>(i));
    ImGui::TextColored(ImVec4(0.7f, 0.95f, 0.75f, 1.0f), "coordinator");
    coordinator_panel(ctx);
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.9f, 1.0f), "next token distribution");
    token_panel(*ctx.tel);
    ImGui::End();

    // ----------------------------------------------------- chat + audit log
    ImGui::SetNextWindowPos(ImVec2(disp.x - col_w, 62), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(col_w, (disp.y - 62) * 0.55f), ImGuiCond_Always);
    ImGui::Begin("chat", nullptr, ImGuiWindowFlags_NoMove);
    chat_panel(ctx, input, sizeof(input));
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(disp.x - col_w, 62 + (disp.y - 62) * 0.55f),
                            ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(col_w, (disp.y - 62) * 0.45f), ImGuiCond_Always);
    ImGui::Begin("audit log", nullptr, ImGuiWindowFlags_NoMove);
    for (const std::string& l : ctx.tel->recent_logs(120)) {
      ImVec4 col(0.75f, 0.75f, 0.8f, 1.0f);
      if (l.find(" accept ") != std::string::npos) col = ImVec4(0.5f, 0.95f, 0.6f, 1.0f);
      else if (l.find(" reject ") != std::string::npos) col = ImVec4(1.0f, 0.6f, 0.5f, 1.0f);
      else if (l.find(" warn ") != std::string::npos) col = ImVec4(1.0f, 0.85f, 0.4f, 1.0f);
      else if (l.find(" error ") != std::string::npos) col = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
      else if (l.find(" augment ") != std::string::npos) col = ImVec4(0.6f, 0.85f, 1.0f, 1.0f);
      ui_wrapped(l, &col);
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 40.0f) ImGui::SetScrollHereY(1.0f);
    ImGui::End();

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
