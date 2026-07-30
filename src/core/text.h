// SPDX-License-Identifier: Apache-2.0
//
// Unicode / multilingual text utilities.
//
// Everything the rest of the project needs to treat Persian, English and
// Python source as first class citizens:
//   * UTF-8 iteration and code point classification,
//   * Persian normalisation (the single most important preprocessing step for
//     Persian: ZWNJ, Arabic vs Persian letter shapes, harakat, digits),
//   * language identification (fa / en / py) used by the mixture dataset, the
//     per-language evaluation and the self-training filters,
//   * a code-switching metric that answers "is the model leaking English into
//     Persian sentences?".
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace slm {

// ------------------------------------------------------------------- UTF-8
// Decodes one code point; advances *i. Invalid bytes are returned as
// 0xDC00+byte (surrogate escape) so round-tripping never loses data.
uint32_t utf8_next(const std::string& s, size_t* i);
void utf8_append(uint32_t cp, std::string* out);
size_t utf8_length(const std::string& s);
// Byte offset of the first byte of the code point at character index `n`.
size_t utf8_offset(const std::string& s, size_t n);
// Truncates to at most `max_bytes` *without* cutting a UTF-8 sequence in half
// (cutting one produces invalid UTF-8, which then breaks the JSONL audit log).
std::string utf8_truncate(const std::string& s, size_t max_bytes);
// Replaces malformed byte sequences with '?' so the result is always valid UTF-8.
std::string utf8_sanitize(const std::string& s);

enum class CharClass : uint8_t {
  kSpace,
  kLatin,       // a-z A-Z and Latin-1 letters
  kArabic,      // Arabic/Persian letters, including ZWNJ (word internal)
  kDigit,       // ASCII digits (after normalisation Persian digits become these)
  kPunct,       // ASCII + Arabic punctuation
  kOther,       // everything else (CJK, emoji, symbols, ...)
};
CharClass classify(uint32_t cp);
bool is_arabic_letter(uint32_t cp);
bool is_zwnj(uint32_t cp);

// -------------------------------------------------------- Persian normalising
struct NormalizeOptions {
  bool arabic_to_persian = true;   // ي->ی  ك->ک  ة->ه  أإ->ا
  bool strip_harakat = true;       // remove short vowels / tashkeel
  bool strip_tatweel = true;       // remove U+0640
  bool digits_to_ascii = true;     // ۰-۹ and ٠-٩  ->  0-9
  bool decompose_presentation = true;  // U+FE70..U+FEFF -> base letters
  bool drop_bidi_controls = true;  // LRM/RLM/LRE..PDF, ZWJ, BOM
  // Off by default on purpose: collapsing runs of spaces destroys Python
  // indentation, and the same normaliser runs over prose *and* source code.
  bool collapse_spaces = false;
  bool strip_trailing_ws = true;   // trailing spaces at end of line (safe for code)
  bool fix_zwnj = true;            // " ‌ " -> "‌", duplicate ZWNJ -> one
  static NormalizeOptions none();
};

std::string normalize_text(const std::string& in, const NormalizeOptions& opt);
// Convenience: the default Persian pipeline.
std::string normalize_persian(const std::string& in);

// ------------------------------------------------------------------ language
enum class Lang : uint8_t { kUnknown = 0, kPersian = 1, kEnglish = 2, kPython = 3, kCount = 4 };
constexpr int kNumLangs = static_cast<int>(Lang::kCount);
const char* lang_name(Lang l);
const char* lang_code(Lang l);  // "fa" / "en" / "py"
bool parse_lang(const std::string& s, Lang* out);

struct TextStats {
  size_t chars = 0;
  size_t arabic = 0;
  size_t latin = 0;
  size_t digits = 0;
  size_t lines = 0;
  size_t code_signals = 0;   // def/import/return/self/:/=/() and indentation
  double code_score = 0.0;   // 0..1
  double arabic_ratio = 0.0;
  double latin_ratio = 0.0;
};
TextStats analyse(const std::string& text);
Lang detect_language(const std::string& text);
Lang detect_language(const TextStats& st);

// Share of Latin letters inside an otherwise Persian text (and vice versa).
// This is the interference / code-switching metric: a Persian answer that
// suddenly contains English words scores high.
double code_switch_ratio(const std::string& text, Lang expected);

// ------------------------------------------------------------- python checks
struct CodeCheck {
  bool ok = false;
  std::string reason;
  int lines = 0;
  int comment_lines = 0;
  int def_count = 0;
  double comment_ratio = 0.0;
  int max_indent = 0;
};
// Structural sanity check for generated Python: balanced brackets and quotes,
// consistent indentation, no stray tabs, plausible statement shape.
CodeCheck check_python(const std::string& code);

}  // namespace slm
