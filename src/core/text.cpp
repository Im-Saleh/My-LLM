// SPDX-License-Identifier: Apache-2.0
#include "core/text.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace slm {
namespace {

// Arabic presentation forms (U+FE70..U+FEFC) back to their base letters.
// Scraped Persian text is full of these; keeping them would double the
// effective alphabet and shatter the BPE vocabulary.
const std::unordered_map<uint32_t, uint32_t>& forms_to_base() {
  static const std::unordered_map<uint32_t, uint32_t> m = {
      {0xFE80, 0x0621}, {0xFE81, 0x0622}, {0xFE82, 0x0622}, {0xFE83, 0x0623},
      {0xFE84, 0x0623}, {0xFE85, 0x0624}, {0xFE86, 0x0624}, {0xFE87, 0x0625},
      {0xFE88, 0x0625}, {0xFE89, 0x0626}, {0xFE8A, 0x0626}, {0xFE8B, 0x0626},
      {0xFE8C, 0x0626}, {0xFE8D, 0x0627}, {0xFE8E, 0x0627}, {0xFE8F, 0x0628},
      {0xFE90, 0x0628}, {0xFE91, 0x0628}, {0xFE92, 0x0628}, {0xFE93, 0x0629},
      {0xFE94, 0x0629}, {0xFE95, 0x062A}, {0xFE96, 0x062A}, {0xFE97, 0x062A},
      {0xFE98, 0x062A}, {0xFE99, 0x062B}, {0xFE9A, 0x062B}, {0xFE9B, 0x062B},
      {0xFE9C, 0x062B}, {0xFE9D, 0x062C}, {0xFE9E, 0x062C}, {0xFE9F, 0x062C},
      {0xFEA0, 0x062C}, {0xFEA1, 0x062D}, {0xFEA2, 0x062D}, {0xFEA3, 0x062D},
      {0xFEA4, 0x062D}, {0xFEA5, 0x062E}, {0xFEA6, 0x062E}, {0xFEA7, 0x062E},
      {0xFEA8, 0x062E}, {0xFEA9, 0x062F}, {0xFEAA, 0x062F}, {0xFEAB, 0x0630},
      {0xFEAC, 0x0630}, {0xFEAD, 0x0631}, {0xFEAE, 0x0631}, {0xFEAF, 0x0632},
      {0xFEB0, 0x0632}, {0xFEB1, 0x0633}, {0xFEB2, 0x0633}, {0xFEB3, 0x0633},
      {0xFEB4, 0x0633}, {0xFEB5, 0x0634}, {0xFEB6, 0x0634}, {0xFEB7, 0x0634},
      {0xFEB8, 0x0634}, {0xFEB9, 0x0635}, {0xFEBA, 0x0635}, {0xFEBB, 0x0635},
      {0xFEBC, 0x0635}, {0xFEBD, 0x0636}, {0xFEBE, 0x0636}, {0xFEBF, 0x0636},
      {0xFEC0, 0x0636}, {0xFEC1, 0x0637}, {0xFEC2, 0x0637}, {0xFEC3, 0x0637},
      {0xFEC4, 0x0637}, {0xFEC5, 0x0638}, {0xFEC6, 0x0638}, {0xFEC7, 0x0638},
      {0xFEC8, 0x0638}, {0xFEC9, 0x0639}, {0xFECA, 0x0639}, {0xFECB, 0x0639},
      {0xFECC, 0x0639}, {0xFECD, 0x063A}, {0xFECE, 0x063A}, {0xFECF, 0x063A},
      {0xFED0, 0x063A}, {0xFED1, 0x0641}, {0xFED2, 0x0641}, {0xFED3, 0x0641},
      {0xFED4, 0x0641}, {0xFED5, 0x0642}, {0xFED6, 0x0642}, {0xFED7, 0x0642},
      {0xFED8, 0x0642}, {0xFED9, 0x0643}, {0xFEDA, 0x0643}, {0xFEDB, 0x0643},
      {0xFEDC, 0x0643}, {0xFEDD, 0x0644}, {0xFEDE, 0x0644}, {0xFEDF, 0x0644},
      {0xFEE0, 0x0644}, {0xFEE1, 0x0645}, {0xFEE2, 0x0645}, {0xFEE3, 0x0645},
      {0xFEE4, 0x0645}, {0xFEE5, 0x0646}, {0xFEE6, 0x0646}, {0xFEE7, 0x0646},
      {0xFEE8, 0x0646}, {0xFEE9, 0x0647}, {0xFEEA, 0x0647}, {0xFEEB, 0x0647},
      {0xFEEC, 0x0647}, {0xFEED, 0x0648}, {0xFEEE, 0x0648}, {0xFEEF, 0x0649},
      {0xFEF0, 0x0649}, {0xFEF1, 0x064A}, {0xFEF2, 0x064A}, {0xFEF3, 0x064A},
      {0xFEF4, 0x064A},
  };
  return m;
}

bool is_harakat(uint32_t cp) {
  return (cp >= 0x064B && cp <= 0x065F) || cp == 0x0670 ||
         (cp >= 0x06D6 && cp <= 0x06ED) || cp == 0x0653 || cp == 0x0654 ||
         cp == 0x0655 || cp == 0x0640 /*tatweel handled separately*/;
}

bool is_bidi_control(uint32_t cp) {
  return cp == 0x200D || cp == 0x200E || cp == 0x200F ||
         (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2066 && cp <= 0x2069) ||
         cp == 0xFEFF;
}

const char* kPyKeywords[] = {"def ",   "class ",  "import ", "from ",  "return",
                            "self.",  "print(",  "for ",    "while ", "if ",
                            "elif ",  "else:",   "lambda ", "yield ", "except",
                            "try:",   "with ",   "async ",  "await ", "None",
                            "True",   "False",   "assert ", "raise "};

}  // namespace

// =============================================================== UTF-8
uint32_t utf8_next(const std::string& s, size_t* i) {
  const size_t n = s.size();
  if (*i >= n) return 0;
  const unsigned char c = static_cast<unsigned char>(s[*i]);
  auto cont = [&](size_t k) {
    return *i + k < n && (static_cast<unsigned char>(s[*i + k]) & 0xC0) == 0x80;
  };
  if (c < 0x80) {
    ++*i;
    return c;
  }
  if ((c & 0xE0) == 0xC0 && cont(1)) {
    const uint32_t cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[*i + 1]) & 0x3Fu);
    *i += 2;
    return cp;
  }
  if ((c & 0xF0) == 0xE0 && cont(1) && cont(2)) {
    const uint32_t cp = ((c & 0x0Fu) << 12) |
                        ((static_cast<unsigned char>(s[*i + 1]) & 0x3Fu) << 6) |
                        (static_cast<unsigned char>(s[*i + 2]) & 0x3Fu);
    *i += 3;
    return cp;
  }
  if ((c & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
    const uint32_t cp = ((c & 0x07u) << 18) |
                        ((static_cast<unsigned char>(s[*i + 1]) & 0x3Fu) << 12) |
                        ((static_cast<unsigned char>(s[*i + 2]) & 0x3Fu) << 6) |
                        (static_cast<unsigned char>(s[*i + 3]) & 0x3Fu);
    *i += 4;
    return cp;
  }
  ++*i;
  return 0xDC00u + c;  // lone byte, preserved
}

void utf8_append(uint32_t cp, std::string* out) {
  if (cp >= 0xDC80 && cp <= 0xDCFF) {  // restored raw byte
    out->push_back(static_cast<char>(cp - 0xDC00));
    return;
  }
  if (cp < 0x80) {
    out->push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

size_t utf8_length(const std::string& s) {
  size_t i = 0, n = 0;
  while (i < s.size()) {
    utf8_next(s, &i);
    ++n;
  }
  return n;
}

size_t utf8_offset(const std::string& s, size_t nchars) {
  size_t i = 0, n = 0;
  while (i < s.size() && n < nchars) {
    utf8_next(s, &i);
    ++n;
  }
  return i;
}

std::string utf8_truncate(const std::string& s, size_t max_bytes) {
  if (s.size() <= max_bytes) return s;
  size_t i = 0, last = 0;
  while (i < s.size()) {
    const size_t start = i;
    utf8_next(s, &i);
    if (i > max_bytes) break;
    last = i;
    (void)start;
  }
  return s.substr(0, last);
}

std::string utf8_sanitize(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    const size_t start = i;
    const uint32_t cp = utf8_next(s, &i);
    if (cp >= 0xDC80 && cp <= 0xDCFF)
      out += '?';  // lone byte
    else
      out.append(s, start, i - start);
  }
  return out;
}

bool is_zwnj(uint32_t cp) { return cp == 0x200C; }

bool is_arabic_letter(uint32_t cp) {
  return (cp >= 0x0621 && cp <= 0x064A) ||    // Arabic letters
         (cp >= 0x0660 && cp <= 0x066F) ||    // Arabic-Indic digits/punct
         (cp >= 0x0671 && cp <= 0x06D5) ||    // extended (پ چ ژ گ ک ی ...)
         (cp >= 0x06F0 && cp <= 0x06FF) ||    // Persian digits + extras
         (cp >= 0xFB50 && cp <= 0xFDFF) ||    // presentation forms A
         (cp >= 0xFE70 && cp <= 0xFEFF) ||    // presentation forms B
         is_zwnj(cp);
}

CharClass classify(uint32_t cp) {
  if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x00A0 ||
      cp == 0x2000 || cp == 0x2028 || cp == 0x2029 || cp == 0x3000)
    return CharClass::kSpace;
  if (cp >= '0' && cp <= '9') return CharClass::kDigit;
  if ((cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9))
    return CharClass::kDigit;
  if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || cp == '_')
    return CharClass::kLatin;
  if ((cp >= 0x00C0 && cp <= 0x024F)) return CharClass::kLatin;
  // Arabic punctuation must NOT be glued to words.
  if (cp == 0x060C || cp == 0x061B || cp == 0x061F || cp == 0x066A ||
      cp == 0x066B || cp == 0x066C || cp == 0x066D || cp == 0x06D4 ||
      cp == 0x00AB || cp == 0x00BB)
    return CharClass::kPunct;
  if (is_arabic_letter(cp)) return CharClass::kArabic;
  if (cp < 0x80) return CharClass::kPunct;  // ASCII symbols
  if ((cp >= 0x2010 && cp <= 0x205E)) return CharClass::kPunct;
  return CharClass::kOther;
}

// ======================================================== normalisation
NormalizeOptions NormalizeOptions::none() {
  NormalizeOptions o;
  o.arabic_to_persian = false;
  o.strip_harakat = false;
  o.strip_tatweel = false;
  o.digits_to_ascii = false;
  o.decompose_presentation = false;
  o.drop_bidi_controls = false;
  o.collapse_spaces = false;
  o.strip_trailing_ws = false;
  o.fix_zwnj = false;
  return o;
}

std::string normalize_text(const std::string& in, const NormalizeOptions& opt) {
  std::string out;
  out.reserve(in.size());
  size_t i = 0;
  bool pending_space = false;
  size_t pending_spaces = 0;
  bool pending_zwnj = false;
  auto flush = [&] {
    if (pending_spaces) {
      out.append(pending_spaces, ' ');
      pending_spaces = 0;
    }
    // A space next to a ZWNJ is always a typing artefact: keep the ZWNJ only.
    if (pending_zwnj) {
      utf8_append(0x200C, &out);
      pending_zwnj = false;
      pending_space = false;
      return;
    }
    if (pending_space) {
      out.push_back(' ');
      pending_space = false;
    }
  };
  while (i < in.size()) {
    uint32_t cp = utf8_next(in, &i);
    if (opt.drop_bidi_controls && is_bidi_control(cp)) continue;
    if (opt.decompose_presentation) {
      auto it = forms_to_base().find(cp);
      if (it != forms_to_base().end()) cp = it->second;
      if (cp == 0xFEFB || cp == 0xFEFC) {  // lam-alef ligature
        flush();
        utf8_append(0x0644, &out);
        utf8_append(0x0627, &out);
        continue;
      }
    }
    if (opt.strip_tatweel && cp == 0x0640) continue;
    if (opt.strip_harakat && is_harakat(cp)) continue;
    if (opt.arabic_to_persian) {
      switch (cp) {
        case 0x0643: cp = 0x06A9; break;  // ك -> ک
        case 0x064A: case 0x0649: case 0x06CD: cp = 0x06CC; break;  // ي ى -> ی
        case 0x0629: cp = 0x0647; break;  // ة -> ه
        case 0x0623: case 0x0625: case 0x0671: case 0x0672: case 0x0673:
          cp = 0x0627; break;             // أ إ ... -> ا
        case 0x06C0: cp = 0x0647; break;  // ۀ -> ه (keep it simple)
        default: break;
      }
    }
    if (opt.digits_to_ascii) {
      if (cp >= 0x0660 && cp <= 0x0669) cp = '0' + (cp - 0x0660);
      else if (cp >= 0x06F0 && cp <= 0x06F9) cp = '0' + (cp - 0x06F0);
    }
    if (opt.fix_zwnj && is_zwnj(cp)) {
      pending_zwnj = true;   // absorbs any spaces around it
      pending_space = false;
      pending_spaces = 0;
      continue;
    }
    if (cp == 0x00A0) cp = ' ';  // NBSP -> plain space
    if (cp == ' ' && (opt.collapse_spaces || opt.strip_trailing_ws)) {
      // Hold the space back: it is emitted only if real content follows on the
      // same line (this strips end-of-line whitespace), and when
      // `collapse_spaces` is off a run of spaces is emitted verbatim.
      if (opt.collapse_spaces) {
        pending_space = true;
      } else {
        ++pending_spaces;
      }
      continue;
    }
    if (cp == '\r') continue;
    if (cp == '\n') {
      pending_space = false;
      pending_spaces = 0;  // drop trailing whitespace
      pending_zwnj = false;
      out.push_back('\n');
      continue;
    }
    flush();
    utf8_append(cp, &out);
  }
  if (pending_zwnj) utf8_append(0x200C, &out);
  return out;
}

std::string normalize_persian(const std::string& in) {
  return normalize_text(in, NormalizeOptions());
}

// ============================================================== language
const char* lang_name(Lang l) {
  switch (l) {
    case Lang::kPersian: return "persian";
    case Lang::kEnglish: return "english";
    case Lang::kPython: return "python";
    default: return "unknown";
  }
}
const char* lang_code(Lang l) {
  switch (l) {
    case Lang::kPersian: return "fa";
    case Lang::kEnglish: return "en";
    case Lang::kPython: return "py";
    default: return "??";
  }
}
bool parse_lang(const std::string& s, Lang* out) {
  if (s == "fa" || s == "persian" || s == "farsi") *out = Lang::kPersian;
  else if (s == "en" || s == "english") *out = Lang::kEnglish;
  else if (s == "py" || s == "python" || s == "code") *out = Lang::kPython;
  else return false;
  return true;
}

TextStats analyse(const std::string& text) {
  TextStats st;
  size_t i = 0;
  while (i < text.size()) {
    const uint32_t cp = utf8_next(text, &i);
    ++st.chars;
    if (cp == '\n') ++st.lines;
    switch (classify(cp)) {
      case CharClass::kArabic: ++st.arabic; break;
      case CharClass::kLatin: ++st.latin; break;
      case CharClass::kDigit: ++st.digits; break;
      default: break;
    }
  }
  if (!text.empty()) ++st.lines;
  for (const char* kw : kPyKeywords) {
    size_t pos = 0;
    while ((pos = text.find(kw, pos)) != std::string::npos) {
      ++st.code_signals;
      pos += std::strlen(kw);
      if (st.code_signals > 4096) break;
    }
  }
  // indentation and structural punctuation are strong code signals
  size_t indented = 0, colon_eol = 0;
  size_t line_start = 0;
  while (line_start <= text.size()) {
    const size_t nl = text.find('\n', line_start);
    const std::string line = text.substr(line_start, nl == std::string::npos
                                                         ? std::string::npos
                                                         : nl - line_start);
    if (line.rfind("    ", 0) == 0 || (!line.empty() && line[0] == '\t')) ++indented;
    size_t e = line.size();
    while (e > 0 && (line[e - 1] == ' ' || line[e - 1] == '\r')) --e;
    if (e > 0 && line[e - 1] == ':') ++colon_eol;
    if (nl == std::string::npos) break;
    line_start = nl + 1;
  }
  st.code_signals += indented + 2 * colon_eol;
  const double denom = std::max<double>(1.0, static_cast<double>(st.chars));
  st.arabic_ratio = static_cast<double>(st.arabic) / denom;
  st.latin_ratio = static_cast<double>(st.latin) / denom;
  st.code_score = std::min(
      1.0, static_cast<double>(st.code_signals) /
               std::max(1.0, static_cast<double>(st.lines) * 0.8 + 2.0));
  return st;
}

Lang detect_language(const TextStats& st) {
  if (st.chars == 0) return Lang::kUnknown;
  if (st.arabic_ratio > 0.15) return Lang::kPersian;
  if (st.code_score > 0.55) return Lang::kPython;
  if (st.latin_ratio > 0.20) return Lang::kEnglish;
  return Lang::kUnknown;
}

Lang detect_language(const std::string& text) { return detect_language(analyse(text)); }

double code_switch_ratio(const std::string& text, Lang expected) {
  const TextStats st = analyse(text);
  const double letters = static_cast<double>(st.arabic + st.latin);
  if (letters < 4.0) return 0.0;
  if (expected == Lang::kPersian) return static_cast<double>(st.latin) / letters;
  if (expected == Lang::kEnglish) return static_cast<double>(st.arabic) / letters;
  return 0.0;
}

// ========================================================= python checks
CodeCheck check_python(const std::string& code) {
  CodeCheck r;
  int round = 0, square = 0, curly = 0;
  bool in_s = false, in_d = false;
  bool triple_s = false, triple_d = false;
  int lines = 0, comments = 0, defs = 0, max_indent = 0;
  bool has_tab_indent = false, has_space_indent = false;
  size_t i = 0;
  size_t line_start = 0;
  bool line_has_code = false;
  auto starts_with_at = [&](const char* p, size_t at) {
    return code.compare(at, std::strlen(p), p) == 0;
  };
  while (i < code.size()) {
    const char c = code[i];
    if (c == '\n') {
      ++lines;
      line_start = i + 1;
      line_has_code = false;
      ++i;
      continue;
    }
    if (i == line_start) {  // indentation of this line
      size_t sp = 0;
      while (i + sp < code.size() && (code[i + sp] == ' ' || code[i + sp] == '\t')) {
        if (code[i + sp] == '\t') has_tab_indent = true;
        else has_space_indent = true;
        ++sp;
      }
      max_indent = std::max(max_indent, static_cast<int>(sp));
      if (sp) {
        i += sp;
        continue;
      }
    }
    if (!in_s && !in_d && !triple_s && !triple_d) {
      if (c == '#') {
        if (!line_has_code) ++comments;
        while (i < code.size() && code[i] != '\n') ++i;
        continue;
      }
      if (starts_with_at("def ", i)) ++defs;
      switch (c) {
        case '(': ++round; break;
        case ')': --round; break;
        case '[': ++square; break;
        case ']': --square; break;
        case '{': ++curly; break;
        case '}': --curly; break;
        default: break;
      }
      if (starts_with_at("\"\"\"", i)) {
        triple_d = !triple_d;
        i += 3;
        line_has_code = true;
        continue;
      }
      if (starts_with_at("'''", i)) {
        triple_s = !triple_s;
        i += 3;
        line_has_code = true;
        continue;
      }
      if (c == '"') in_d = true;
      else if (c == '\'') in_s = true;
      if (c != ' ') line_has_code = true;
    } else if (!triple_s && !triple_d) {
      if (c == '\\') {
        i += 2;
        continue;
      }
      if (c == '"' && in_d) in_d = false;
      else if (c == '\'' && in_s) in_s = false;
    } else {
      if (triple_d && starts_with_at("\"\"\"", i)) {
        triple_d = false;
        i += 3;
        continue;
      }
      if (triple_s && starts_with_at("'''", i)) {
        triple_s = false;
        i += 3;
        continue;
      }
    }
    if (round < 0 || square < 0 || curly < 0) {
      r.reason = "unbalanced closing bracket";
      return r;
    }
    ++i;
  }
  if (!code.empty() && code.back() != '\n') ++lines;
  r.lines = lines;
  r.comment_lines = comments;
  r.def_count = defs;
  r.max_indent = max_indent;
  r.comment_ratio = lines ? static_cast<double>(comments) / lines : 0.0;
  if (round || square || curly) {
    r.reason = "unbalanced brackets";
    return r;
  }
  if (in_s || in_d) {
    r.reason = "unterminated string literal";
    return r;
  }
  if (triple_s || triple_d) {
    r.reason = "unterminated triple quoted string";
    return r;
  }
  if (has_tab_indent && has_space_indent) {
    r.reason = "mixed tab/space indentation";
    return r;
  }
  if (lines < 2) {
    r.reason = "too few lines to be a code sample";
    return r;
  }
  r.ok = true;
  return r;
}

}  // namespace slm
