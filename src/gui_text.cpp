// SPDX-License-Identifier: Apache-2.0
#include "gui_text.h"

#include "core/text.h"

#ifdef SLM_WITH_TEXT_SHAPING

#include <ft2build.h>
#include FT_FREETYPE_H
#include <hb-ft.h>
#include <hb.h>

#include <GL/gl.h>

#include <algorithm>
#include <cstring>
#include <list>
#include <unordered_map>

namespace slm {
namespace {

const char* kFontCandidates[] = {
    // Persian first: these have the full Arabic shaping tables.
    "/usr/share/fonts/vazirmatn/Vazirmatn-Regular.ttf",
    "/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallback.ttf",
};

struct Run {
  size_t begin = 0, end = 0;
  bool rtl = false;
};

// Minimal bidi: split into runs of Arabic-script vs everything else.  Neutral
// characters (spaces, punctuation, digits) join the previous run, which is what
// a reader expects for "سلام def foo() است".
std::vector<Run> split_runs(const std::string& s, bool* base_rtl) {
  std::vector<Run> runs;
  bool first_strong_set = false;
  bool cur_rtl = false;
  size_t i = 0, run_start = 0;
  *base_rtl = false;
  while (i < s.size()) {
    const size_t start = i;
    const uint32_t cp = utf8_next(s, &i);
    const CharClass cls = classify(cp);
    bool strong = false, rtl = cur_rtl;
    // Arabic punctuation (، ؛ ؟ ٪ « ») is treated as RTL-strong: it belongs to
    // the Persian run both for ordering and for font selection.
    const bool arabic_punct = cp == 0x060C || cp == 0x061B || cp == 0x061F ||
                              cp == 0x066A || cp == 0x06D4 || cp == 0x00AB || cp == 0x00BB;
    if ((cls == CharClass::kArabic && !is_zwnj(cp)) || arabic_punct) {
      strong = true;
      rtl = true;
    } else if (cls == CharClass::kLatin) {
      strong = true;
      rtl = false;
    }
    if (strong && !first_strong_set) {
      first_strong_set = true;
      *base_rtl = rtl;
      cur_rtl = rtl;
      run_start = 0;
    }
    if (strong && rtl != cur_rtl) {
      runs.push_back(Run{run_start, start, cur_rtl});
      run_start = start;
      cur_rtl = rtl;
    }
  }
  runs.push_back(Run{run_start, s.size(), cur_rtl});
  return runs;
}

}  // namespace

struct ShapedText::Impl {
  FT_Library lib = nullptr;
  // Two faces: an Arabic-capable one for RTL runs and a Latin one for the rest.
  // Many Arabic fonts (Noto Sans Arabic for instance) carry no Latin glyphs at
  // all, which is why mixed log lines need a real fallback instead of one face.
  FT_Face face = nullptr;       // primary / Arabic
  FT_Face latin = nullptr;      // fallback for Latin, digits, punctuation
  hb_font_t* hb = nullptr;
  hb_font_t* hb_latin = nullptr;
  std::string path;
  std::string latin_path;
  float px = 17.0f;
  int ascent = 0, descent = 0;
  bool ok = false;

  // string -> texture, with an LRU list so long sessions stay bounded
  struct Cached {
    Image img;
    std::list<std::string>::iterator lru;
  };
  std::unordered_map<std::string, Cached> cache;
  std::list<std::string> lru;
  size_t max_entries = 1024;

  ~Impl() {
    for (auto& kv : cache)
      if (kv.second.img.texture) glDeleteTextures(1, &kv.second.img.texture);
    if (hb) hb_font_destroy(hb);
    if (hb_latin) hb_font_destroy(hb_latin);
    if (face) FT_Done_Face(face);
    if (latin) FT_Done_Face(latin);
    if (lib) FT_Done_FreeType(lib);
  }

  // Does the primary face cover plain ASCII?  If not, LTR runs go to `latin`.
  bool primary_has_latin() const {
    return face && FT_Get_Char_Index(face, 'A') != 0 && FT_Get_Char_Index(face, 'a') != 0;
  }

  // Lays out the string and optionally rasterises it into `rgba`.
  bool layout(const std::string& s, int* out_w, int* out_h, std::vector<uint8_t>* rgba) {
    if (!ok) return false;
    bool base_rtl = false;
    const std::vector<Run> runs = split_runs(s, &base_rtl);

    struct Shaped {
      hb_buffer_t* buf = nullptr;
      unsigned count = 0;
      hb_glyph_info_t* info = nullptr;
      hb_glyph_position_t* pos = nullptr;
      int width = 0;
      FT_Face face = nullptr;
    };
    const bool use_fallback = latin && !primary_has_latin();
    std::vector<Shaped> shaped;
    int total_w = 0;
    for (const Run& r : runs) {
      if (r.end <= r.begin) continue;
      Shaped sh;
      const bool latin_run = !r.rtl && use_fallback;
      sh.face = latin_run ? latin : face;
      sh.buf = hb_buffer_create();
      hb_buffer_add_utf8(sh.buf, s.data() + r.begin, static_cast<int>(r.end - r.begin), 0,
                         static_cast<int>(r.end - r.begin));
      hb_buffer_set_direction(sh.buf, r.rtl ? HB_DIRECTION_RTL : HB_DIRECTION_LTR);
      hb_buffer_set_script(sh.buf, r.rtl ? HB_SCRIPT_ARABIC : HB_SCRIPT_LATIN);
      hb_buffer_set_language(sh.buf, hb_language_from_string(r.rtl ? "fa" : "en", -1));
      hb_shape(latin_run ? hb_latin : hb, sh.buf, nullptr, 0);
      sh.info = hb_buffer_get_glyph_infos(sh.buf, &sh.count);
      sh.pos = hb_buffer_get_glyph_positions(sh.buf, &sh.count);
      for (unsigned g = 0; g < sh.count; ++g) sh.width += sh.pos[g].x_advance >> 6;
      total_w += sh.width;
      shaped.push_back(sh);
    }
    const int h = ascent + descent + 2;
    *out_w = std::max(1, total_w + 2);
    *out_h = std::max(1, h);
    if (rgba) {
      rgba->assign(static_cast<size_t>(*out_w) * static_cast<size_t>(*out_h) * 4, 0);
      // Visual order: for an RTL paragraph the runs are placed right to left.
      std::vector<size_t> order(shaped.size());
      for (size_t i = 0; i < shaped.size(); ++i) order[i] = i;
      if (base_rtl) std::reverse(order.begin(), order.end());
      int pen_x = 1;
      for (size_t oi : order) {
        const Shaped& sh = shaped[oi];
        for (unsigned g = 0; g < sh.count; ++g) {
          FT_Face f = sh.face ? sh.face : face;
          if (FT_Load_Glyph(f, sh.info[g].codepoint, FT_LOAD_RENDER)) continue;
          const FT_GlyphSlot sl = f->glyph;
          const int gx = pen_x + (sh.pos[g].x_offset >> 6) + sl->bitmap_left;
          const int gy = ascent - (sh.pos[g].y_offset >> 6) - sl->bitmap_top + 1;
          for (unsigned row = 0; row < sl->bitmap.rows; ++row) {
            const int py = gy + static_cast<int>(row);
            if (py < 0 || py >= *out_h) continue;
            for (unsigned col = 0; col < sl->bitmap.width; ++col) {
              const int px_ = gx + static_cast<int>(col);
              if (px_ < 0 || px_ >= *out_w) continue;
              const uint8_t a = sl->bitmap.buffer[row * static_cast<unsigned>(sl->bitmap.pitch) + col];
              if (!a) continue;
              uint8_t* dst = rgba->data() + (static_cast<size_t>(py) * *out_w + px_) * 4;
              const uint8_t prev = dst[3];
              const uint8_t v = a > prev ? a : prev;
              dst[0] = dst[1] = dst[2] = 255;
              dst[3] = v;
            }
          }
          pen_x += sh.pos[g].x_advance >> 6;
        }
      }
    }
    for (Shaped& sh : shaped) hb_buffer_destroy(sh.buf);
    return true;
  }
};

ShapedText::ShapedText() : impl_(new Impl) {}
ShapedText::~ShapedText() { delete impl_; }

bool ShapedText::init(const std::vector<std::string>& font_paths, float pixel_size) {
  impl_->px = pixel_size;
  if (FT_Init_FreeType(&impl_->lib)) return false;
  std::vector<std::string> candidates = font_paths;
  for (const char* p : kFontCandidates) candidates.emplace_back(p);
  for (const std::string& p : candidates) {
    if (p.empty()) continue;
    if (FT_New_Face(impl_->lib, p.c_str(), 0, &impl_->face) == 0) {
      impl_->path = p;
      break;
    }
    impl_->face = nullptr;
  }
  if (!impl_->face) return false;
  FT_Set_Pixel_Sizes(impl_->face, 0, static_cast<FT_UInt>(pixel_size));
  impl_->ascent = static_cast<int>(impl_->face->size->metrics.ascender >> 6);
  impl_->descent = static_cast<int>(-(impl_->face->size->metrics.descender >> 6));
  impl_->hb = hb_ft_font_create_referenced(impl_->face);
  impl_->ok = impl_->hb != nullptr;

  // Latin fallback (only used when the primary face has no Latin coverage).
  static const char* kLatinCandidates[] = {
      "/usr/share/fonts/dejavu/DejaVuSans.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
      "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
      "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
      "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
      "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf",
  };
  for (const char* p : kLatinCandidates) {
    if (FT_New_Face(impl_->lib, p, 0, &impl_->latin) == 0) {
      impl_->latin_path = p;
      FT_Set_Pixel_Sizes(impl_->latin, 0, static_cast<FT_UInt>(pixel_size));
      impl_->hb_latin = hb_ft_font_create_referenced(impl_->latin);
      break;
    }
    impl_->latin = nullptr;
  }
  return impl_->ok;
}

bool ShapedText::ready() const { return impl_->ok; }
const std::string& ShapedText::font() const { return impl_->path; }
const std::string& ShapedText::latin_font() const { return impl_->latin_path; }
float ShapedText::line_height() const {
  return static_cast<float>(impl_->ascent + impl_->descent + 2);
}
size_t ShapedText::cache_size() const { return impl_->cache.size(); }

void ShapedText::clear_cache() {
  for (auto& kv : impl_->cache)
    if (kv.second.img.texture) glDeleteTextures(1, &kv.second.img.texture);
  impl_->cache.clear();
  impl_->lru.clear();
}

const ShapedText::Image* ShapedText::image(const std::string& utf8) {
  if (!impl_->ok || utf8.empty()) return nullptr;
  auto it = impl_->cache.find(utf8);
  if (it != impl_->cache.end()) {
    impl_->lru.splice(impl_->lru.end(), impl_->lru, it->second.lru);
    return &it->second.img;
  }
  int w = 0, h = 0;
  std::vector<uint8_t> rgba;
  if (!impl_->layout(utf8, &w, &h, &rgba)) return nullptr;

  Image img;
  img.width = w;
  img.height = h;
  glGenTextures(1, &img.texture);
  glBindTexture(GL_TEXTURE_2D, img.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

  while (impl_->cache.size() >= impl_->max_entries && !impl_->lru.empty()) {
    const std::string& victim = impl_->lru.front();
    auto vit = impl_->cache.find(victim);
    if (vit != impl_->cache.end()) {
      if (vit->second.img.texture) glDeleteTextures(1, &vit->second.img.texture);
      impl_->cache.erase(vit);
    }
    impl_->lru.pop_front();
  }
  auto lit = impl_->lru.insert(impl_->lru.end(), utf8);
  Impl::Cached c;
  c.img = img;
  c.lru = lit;
  return &impl_->cache.emplace(utf8, c).first->second.img;
}

float ShapedText::measure(const std::string& utf8) {
  const Image* img = image(utf8);
  return img ? static_cast<float>(img->width) : 0.0f;
}

bool text_shaping_compiled() { return true; }

bool needs_shaping(const std::string& utf8) {
  size_t i = 0;
  while (i < utf8.size()) {
    const uint32_t cp = utf8_next(utf8, &i);
    if (classify(cp) == CharClass::kArabic && !is_zwnj(cp)) return true;
  }
  return false;
}

}  // namespace slm

#else  // no FreeType / HarfBuzz

namespace slm {

struct ShapedText::Impl {};
ShapedText::ShapedText() : impl_(nullptr) {}
ShapedText::~ShapedText() {}
bool ShapedText::init(const std::vector<std::string>&, float) { return false; }
bool ShapedText::ready() const { return false; }
const std::string& ShapedText::font() const {
  static const std::string empty;
  return empty;
}
const std::string& ShapedText::latin_font() const {
  static const std::string empty;
  return empty;
}
float ShapedText::line_height() const { return 0.0f; }
const ShapedText::Image* ShapedText::image(const std::string&) { return nullptr; }
float ShapedText::measure(const std::string&) { return 0.0f; }
void ShapedText::clear_cache() {}
size_t ShapedText::cache_size() const { return 0; }
bool text_shaping_compiled() { return false; }

bool needs_shaping(const std::string& utf8) {
  size_t i = 0;
  while (i < utf8.size()) {
    const uint32_t cp = utf8_next(utf8, &i);
    if (classify(cp) == CharClass::kArabic && !is_zwnj(cp)) return true;
  }
  return false;
}

}  // namespace slm

#endif
