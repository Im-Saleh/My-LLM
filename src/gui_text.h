// SPDX-License-Identifier: Apache-2.0
//
// Correct Persian/Arabic text rendering for the ImGui dashboard.
//
// Why this file exists
// --------------------
// ImGui draws one glyph per code point from a font atlas.  That is fine for
// Latin, but Persian needs:
//   * contextual shaping   (each letter has isolated/initial/medial/final forms
//                           and there are mandatory ligatures such as لا),
//   * right-to-left layout (with LTR islands for numbers, code and Latin words),
//   * mark positioning     (diacritics).
// The Unicode "presentation forms" trick that many ImGui apps use cannot work
// for Persian, because پ چ ژ گ simply have no presentation form code points.
//
// So instead of fighting the atlas, a shaped string is rasterised once with
// HarfBuzz + FreeType into an RGBA texture and drawn as an image, then cached.
// Cost: one texture per distinct string, uploaded on first use.
//
// Built only when FreeType and HarfBuzz are available (SLM_WITH_TEXT_SHAPING);
// the dashboard degrades to plain ImGui text otherwise.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace slm {

// True when this build can shape Persian text.
bool text_shaping_compiled();

class ShapedText {
 public:
  ShapedText();
  ~ShapedText();
  ShapedText(const ShapedText&) = delete;
  ShapedText& operator=(const ShapedText&) = delete;

  // Tries every path in order plus a built-in list of common locations.
  bool init(const std::vector<std::string>& font_paths, float pixel_size);
  bool ready() const;
  const std::string& font() const;
  // Fallback face used for Latin runs when the primary font has no Latin glyphs.
  const std::string& latin_font() const;
  float line_height() const;

  struct Image {
    unsigned int texture = 0;
    int width = 0;
    int height = 0;
  };
  // Shapes + rasterises (cached).  Returns nullptr when unavailable.
  const Image* image(const std::string& utf8);
  // Width in pixels a string would need.
  float measure(const std::string& utf8);
  void clear_cache();
  size_t cache_size() const;

 private:
  struct Impl;
  Impl* impl_;
};

// True when the string contains Arabic-script letters (so it needs shaping).
bool needs_shaping(const std::string& utf8);

}  // namespace slm
