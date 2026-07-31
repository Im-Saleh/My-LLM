// SPDX-License-Identifier: Apache-2.0
//
// Implementation of the local codebase index declared in codebase.h.
//
// The three parts that carry the quality of the whole thing:
//
//   1. `code_tokens` - one pass over the bytes, emitting the whole identifier
//      plus its camelCase / snake_case pieces.  It runs over every byte of the
//      repository during a scan and over every query afterwards, so it does no
//      regex, no substr and no temporary vectors.
//   2. the chunkers - a brace-language scanner that tracks nesting depth while
//      *ignoring* braces inside strings, char literals and comments (a `{` in a
//      string that opens a phantom block shifts every following boundary in the
//      file, which is the classic way structural chunking silently degrades into
//      random line slicing), plus an indentation scanner for Python and a
//      heading scanner for prose.
//   3. retrieval - Okapi BM25 over an inverted index, exact cosine over one
//      contiguous L2-normalised matrix, fused with RRF and re-ranked with a
//      symbol/path/length prior.
//
// Vector search uses a plain `for` loop over contiguous floats rather than
// explicit AVX2 intrinsics: at -O3 (and even -O2 with the loop shaped like this)
// gcc/clang vectorise it, and the scalar source stays portable to arm64 with no
// `#if defined(__x86_64__)` maze and no runtime dispatch to keep correct.
#include "agent/codebase.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

#include "core/text.h"

namespace slm {
namespace fs = std::filesystem;

namespace {

constexpr uint32_t kIndexVersion = 1;
constexpr const char kIndexMagic[8] = {'S', 'L', 'M', 'I', 'D', 'X', '0', '1'};
// Embedding a 5 MB generated file would cost more than it can ever return, and
// the tail of such a file says nothing the head did not.
constexpr size_t kEmbedInputBytes = 8192;
constexpr size_t kBinarySniffBytes = 8192;

// ------------------------------------------------------------ tiny primitives
inline char lower_ascii(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}
inline bool is_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }
inline bool is_lower(unsigned char c) { return c >= 'a' && c <= 'z'; }
inline bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
inline bool is_alnum(unsigned char c) { return is_upper(c) || is_lower(c) || is_digit(c); }

// Identifier bytes.  '_' is included so `qpack_synthesise` survives as one token
// (its pieces are emitted separately), and bytes >= 0x80 are included so Persian
// and other non-Latin words in comments and docs are searchable at all.
inline bool is_word_byte(unsigned char c) { return is_alnum(c) || c == '_' || c >= 0x80; }

std::string lower_str(const std::string& s) {
  std::string o(s.size(), '\0');
  for (size_t i = 0; i < s.size(); ++i) o[i] = lower_ascii(s[i]);
  return o;
}

bool starts_with(const std::string& s, const char* p) {
  const size_t n = std::strlen(p);
  return s.size() >= n && std::memcmp(s.data(), p, n) == 0;
}
bool ends_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && std::memcmp(s.data() + s.size() - p.size(), p.data(),
                                             p.size()) == 0;
}

std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' ||
                   s[e - 1] == '\n'))
    --e;
  return s.substr(b, e - b);
}

std::string basename_of(const std::string& p) {
  const size_t s = p.find_last_of("/\\");
  return s == std::string::npos ? p : p.substr(s + 1);
}

std::vector<std::string> split_lines(const std::string& text) {
  std::vector<std::string> out;
  out.reserve(text.size() / 32 + 4);
  size_t b = 0;
  while (b <= text.size()) {
    const size_t e = text.find('\n', b);
    if (e == std::string::npos) {
      if (b < text.size()) out.push_back(text.substr(b));
      break;
    }
    size_t len = e - b;
    if (len && text[b + len - 1] == '\r') --len;  // CRLF checkouts
    out.push_back(text.substr(b, len));
    b = e + 1;
  }
  return out;
}

std::string join_lines(const std::vector<std::string>& lines, size_t from, size_t to) {
  size_t n = 0;
  for (size_t i = from; i <= to && i < lines.size(); ++i) n += lines[i].size() + 1;
  std::string out;
  out.reserve(n);
  for (size_t i = from; i <= to && i < lines.size(); ++i) {
    if (i > from) out.push_back('\n');
    out += lines[i];
  }
  return out;
}

// ------------------------------------------------------------------- hashing
inline uint64_t fnv1a(const char* p, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<unsigned char>(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

}  // namespace

// =========================================================== tokenisation
namespace {

// Appends s[b,e) lowercased as one token, unless it is noise.
inline void push_token(const std::string& s, size_t b, size_t e,
                       std::vector<std::string>* out) {
  if (e - b < 2) return;  // single characters match everything and mean nothing
  bool all_digits = true;
  for (size_t i = b; i < e && all_digits; ++i)
    if (!is_digit(static_cast<unsigned char>(s[i]))) all_digits = false;
  if (all_digits && e - b > 6) return;  // timestamps, hashes, generated ids
  out->emplace_back();
  std::string& t = out->back();
  t.resize(e - b);
  for (size_t i = b; i < e; ++i) t[i - b] = lower_ascii(s[i]);
}

}  // namespace

std::vector<std::string> code_tokens(const std::string& text, bool split_identifiers) {
  std::vector<std::string> out;
  // One token per ~6 bytes of source is a good enough guess to avoid most of
  // the vector growth; this function is on the hot path of a full scan.
  out.reserve(text.size() / 6 + 8);
  const size_t n = text.size();
  size_t i = 0;
  while (i < n) {
    if (!is_word_byte(static_cast<unsigned char>(text[i]))) {
      ++i;
      continue;
    }
    const size_t b = i;
    while (i < n && is_word_byte(static_cast<unsigned char>(text[i]))) ++i;
    push_token(text, b, i, &out);  // the whole identifier, always
    if (!split_identifiers) continue;

    // Sub-words: '_' separators and camelCase / acronym-run boundaries.
    size_t s = b;
    for (size_t j = b; j <= i; ++j) {
      const bool at_end = (j == i);
      const bool sep = !at_end && text[j] == '_';
      bool camel = false;
      if (!at_end && !sep && j > s) {
        const unsigned char c = static_cast<unsigned char>(text[j]);
        const unsigned char p = static_cast<unsigned char>(text[j - 1]);
        if (is_upper(c) && (is_lower(p) || is_digit(p))) {
          camel = true;  // parseHttp -> parse | Http
        } else if (is_upper(c) && is_upper(p) && j + 1 < i &&
                   is_lower(static_cast<unsigned char>(text[j + 1]))) {
          camel = true;  // HTTPServer -> HTTP | Server
        }
      }
      if (at_end || sep || camel) {
        // Skip the piece that spans the whole identifier: already emitted.
        if (j > s && !(s == b && j == i)) push_token(text, s, j, &out);
        s = sep ? j + 1 : j;
      }
    }
  }
  return out;
}

// ============================================================== languages
std::string language_of(const std::string& path) {
  const std::string lb = lower_str(basename_of(path));
  // Basenames first: "CMakeLists.txt" would otherwise be plain text, and
  // Makefile / Dockerfile have no extension at all.
  if (lb == "cmakelists.txt") return "cmake";
  if (lb == "makefile" || lb == "gnumakefile") return "make";
  if (lb == "dockerfile" || starts_with(lb, "dockerfile.")) return "dockerfile";
  const size_t dot = lb.find_last_of('.');
  if (dot == std::string::npos || dot + 1 == lb.size()) return "";
  const std::string e = lb.substr(dot + 1);

  static const std::map<std::string, std::string> kByExt = {
      {"c", "cpp"},        {"cc", "cpp"},         {"cpp", "cpp"},
      {"cxx", "cpp"},      {"h", "cpp"},          {"hh", "cpp"},
      {"hpp", "cpp"},      {"hxx", "cpp"},        {"inl", "cpp"},
      {"py", "python"},    {"pyi", "python"},     {"js", "javascript"},
      {"jsx", "javascript"}, {"mjs", "javascript"}, {"cjs", "javascript"},
      {"ts", "typescript"}, {"tsx", "typescript"}, {"go", "go"},
      {"rs", "rust"},      {"java", "java"},      {"kt", "kotlin"},
      {"cs", "csharp"},    {"rb", "ruby"},        {"php", "php"},
      {"swift", "swift"},  {"scala", "scala"},    {"sh", "shell"},
      {"bash", "shell"},   {"zsh", "shell"},      {"sql", "sql"},
      {"md", "markdown"},  {"markdown", "markdown"}, {"rst", "rst"},
      {"txt", "text"},     {"json", "json"},      {"yml", "yaml"},
      {"yaml", "yaml"},    {"toml", "toml"},      {"ini", "ini"},
      {"xml", "xml"},      {"html", "html"},      {"htm", "html"},
      {"css", "css"},      {"cmake", "cmake"},    {"mk", "make"},
      {"dockerfile", "dockerfile"}, {"proto", "proto"},
  };
  const auto it = kByExt.find(e);
  return it == kByExt.end() ? std::string() : it->second;
}

namespace {

bool is_brace_lang(const std::string& l) {
  return l == "cpp" || l == "javascript" || l == "typescript" || l == "go" ||
         l == "rust" || l == "java" || l == "kotlin" || l == "csharp" || l == "php" ||
         l == "swift" || l == "scala" || l == "css";
}
bool is_prose_lang(const std::string& l) {
  return l == "markdown" || l == "rst" || l == "text";
}

// ------------------------------------------------------- brace line scanner
struct LineInfo {
  int depth_before = 0;
  int depth_after = 0;
  int depth_min = 0;      // lowest depth reached inside the line
  bool blank = true;      // no code at all (may still be a comment)
  bool comment = false;   // comment-only line
  bool ends_semi = false;
  bool ends_brace = false;
  bool has_brace = false;  // an unquoted '{' occurred, even if closed again here
  std::string code;        // comments removed, literal bodies blanked out
};

// Walks the file once, tracking nesting depth while skipping comments and
// literals.  Everything downstream (chunk boundaries, symbol names) reads
// `code`, never the raw line, so a `{` inside a string can never open a block.
std::vector<LineInfo> scan_braces(const std::vector<std::string>& lines,
                                  const std::string& lang) {
  const bool backtick = (lang == "javascript" || lang == "typescript");
  const bool hash_comment = (lang == "php");
  std::vector<LineInfo> li(lines.size());
  int depth = 0;
  bool in_block = false;  // inside /* ... */
  bool in_tmpl = false;   // inside a `template literal`
  for (size_t i = 0; i < lines.size(); ++i) {
    LineInfo& L = li[i];
    const std::string& s = lines[i];
    L.depth_before = depth;
    L.depth_min = depth;
    L.code.reserve(s.size());
    bool any_code = false;
    size_t j = 0;
    while (j < s.size()) {
      const char c = s[j];
      if (in_block) {
        if (c == '*' && j + 1 < s.size() && s[j + 1] == '/') {
          in_block = false;
          j += 2;
        } else {
          ++j;
        }
        continue;
      }
      if (in_tmpl) {
        if (c == '\\') {
          j += 2;
        } else if (c == '`') {
          in_tmpl = false;
          ++j;
          L.code += "\"\"";
          any_code = true;
        } else {
          ++j;
        }
        continue;
      }
      if (c == '/' && j + 1 < s.size() && s[j + 1] == '/') break;  // line comment
      if (hash_comment && c == '#') break;
      if (c == '/' && j + 1 < s.size() && s[j + 1] == '*') {
        in_block = true;
        j += 2;
        continue;
      }
      if (c == '"') {
        ++j;
        while (j < s.size()) {
          if (s[j] == '\\') {
            j += 2;
            continue;
          }
          if (s[j] == '"') {
            ++j;
            break;
          }
          ++j;
        }
        L.code += "\"\"";
        any_code = true;
        continue;
      }
      if (c == '\'') {
        // Only a char literal when it closes within a couple of characters;
        // otherwise this is a Rust lifetime ('a) or an apostrophe in prose.
        size_t k = j + 1;
        if (k < s.size() && s[k] == '\\') k += 2; else ++k;
        if (k < s.size() && s[k] == '\'') {
          j = k + 1;
          L.code += "''";
          any_code = true;
          continue;
        }
        L.code.push_back(c);
        ++j;
        any_code = true;
        continue;
      }
      if (backtick && c == '`') {
        in_tmpl = true;
        ++j;
        continue;
      }
      if (c == '{') {
        ++depth;
        L.has_brace = true;
      } else if (c == '}') {
        if (depth > 0) --depth;
        if (depth < L.depth_min) L.depth_min = depth;
      }
      L.code.push_back(c);
      ++j;
      if (c != ' ' && c != '\t') any_code = true;
    }
    L.depth_after = depth;
    L.code = trim(L.code);
    L.blank = L.code.empty();
    L.comment = L.blank && !trim(s).empty();
    if (!L.blank) {
      L.ends_semi = L.code.back() == ';';
      L.ends_brace = L.code.back() == '}';
    }
    (void)any_code;
  }
  return li;
}

// ----------------------------------------------------------- symbol naming
bool word_in(const char* const* set, size_t n, const std::string& w) {
  for (size_t i = 0; i < n; ++i)
    if (w == set[i]) return true;
  return false;
}

const char* const kClassKw[] = {"class",  "struct", "union",     "enum",  "interface",
                                "trait",  "impl",   "object",    "record", "protocol",
                                "module", "package"};
const char* const kModifierKw[] = {
    "public",    "private",  "protected", "static",   "final",    "abstract",
    "sealed",    "export",   "default",   "pub",      "inline",   "virtual",
    "explicit",  "constexpr", "consteval", "friend",  "extern",   "const",
    "async",     "open",     "data",      "partial",  "override", "internal",
    "unsafe",    "mutable",  "operator",  "readonly", "suspend",  "declare"};
const char* const kControlKw[] = {
    "if",     "else",   "elif",   "for",    "while",  "switch",  "case",
    "catch",  "try",    "do",     "return", "sizeof", "alignof", "new",
    "delete", "throw",  "using",  "typedef", "decltype", "static_assert",
    "goto",   "break",  "continue", "match", "when",  "with",    "assert",
    "defer",  "select", "go",     "await",  "yield",  "print",   "printf",
    "foreach", "lock",  "unless", "until",  "loop",   "in",      "of"};
// Words that introduce a definition but are never the definition's name.
const char* const kDefKw[] = {"fn", "func", "def", "function", "fun", "sub", "method",
                              "proc", "let", "var", "val", "type", "typealias"};

bool is_class_kw(const std::string& w) {
  return word_in(kClassKw, sizeof(kClassKw) / sizeof(*kClassKw), w);
}
bool is_modifier_kw(const std::string& w) {
  return word_in(kModifierKw, sizeof(kModifierKw) / sizeof(*kModifierKw), w);
}
bool is_control_kw(const std::string& w) {
  return word_in(kControlKw, sizeof(kControlKw) / sizeof(*kControlKw), w);
}
bool is_def_kw(const std::string& w) {
  return word_in(kDefKw, sizeof(kDefKw) / sizeof(*kDefKw), w);
}

// Drops a leading `template <...>`, `[[attr]]`, `#[attr]`, `@Anno` so the words
// that follow are the ones that name the definition.
std::string strip_prologue(const std::string& in) {
  std::string s = trim(in);
  for (int guard = 0; guard < 4; ++guard) {
    if (starts_with(s, "template")) {
      const size_t lt = s.find('<');
      if (lt == std::string::npos) break;
      int d = 0;
      size_t k = lt;
      for (; k < s.size(); ++k) {
        if (s[k] == '<') ++d;
        else if (s[k] == '>' && --d == 0) break;
      }
      if (k >= s.size()) break;
      s = trim(s.substr(k + 1));
      continue;
    }
    if (starts_with(s, "[[") || starts_with(s, "#[")) {
      const size_t close = s.find(']', 2);
      if (close == std::string::npos) break;
      size_t k = close;
      while (k + 1 < s.size() && s[k + 1] == ']') ++k;
      s = trim(s.substr(k + 1));
      continue;
    }
    if (!s.empty() && s[0] == '@') {
      const size_t sp = s.find(' ');
      if (sp == std::string::npos) break;
      s = trim(s.substr(sp + 1));
      continue;
    }
    break;
  }
  return s;
}

// Identifier-with-qualifier bytes: `Foo::bar`, `~Foo`, `self.method`.
inline bool is_name_byte(unsigned char c) {
  return is_alnum(c) || c == '_' || c == ':' || c == '~' || c == '$';
}
// Plain identifier bytes, for languages where ':' terminates the definition
// line instead of qualifying a name (`class PayloadStore:`).
inline bool is_plain_name_byte(unsigned char c) { return is_alnum(c) || c == '_'; }

// `class Foo: public Bar` and `::foo` both leave a stray colon behind.
void clean_symbol(std::string* s) {
  while (!s->empty() && s->back() == ':') s->pop_back();
  while (!s->empty() && s->front() == ':') s->erase(s->begin());
}

void extract_symbol(const std::string& def_line, std::string* sym, std::string* kind) {
  sym->clear();
  *kind = "block";
  const std::string s = strip_prologue(def_line);
  if (s.empty()) return;

  // Word list, so leading modifiers can be skipped cheaply.
  std::vector<std::string> words;
  {
    size_t i = 0;
    while (i < s.size() && words.size() < 8) {
      if (!is_name_byte(static_cast<unsigned char>(s[i]))) {
        ++i;
        continue;
      }
      const size_t b = i;
      while (i < s.size() && is_name_byte(static_cast<unsigned char>(s[i]))) ++i;
      words.push_back(s.substr(b, i - b));
    }
  }
  size_t wi = 0;
  while (wi < words.size() && is_modifier_kw(lower_str(words[wi]))) ++wi;

  // A class-family keyword names its type, and that beats the function path:
  // `class Foo { int f(); }` is a class, not a function called f.
  if (wi < words.size() && is_class_kw(lower_str(words[wi]))) {
    for (size_t k = wi + 1; k < words.size(); ++k) {
      const std::string lw = lower_str(words[k]);
      if (is_modifier_kw(lw) || is_class_kw(lw)) continue;
      *sym = words[k];
      clean_symbol(sym);
      *kind = "class";
      return;
    }
    return;
  }

  // Otherwise: the identifier in front of the first '(' that is not a keyword.
  // Scanning left to right handles Go's `func (r *T) Name(` (the receiver's '('
  // is preceded by the keyword `func`) as well as C++ return types.
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '(') continue;
    size_t e = i;
    while (e > 0 && (s[e - 1] == ' ' || s[e - 1] == '\t')) --e;
    size_t b = e;
    while (b > 0 && is_name_byte(static_cast<unsigned char>(s[b - 1]))) --b;
    if (b == e) continue;
    std::string cand = s.substr(b, e - b);
    while (!cand.empty() && cand.front() == ':') cand.erase(cand.begin());
    const std::string lw = lower_str(cand);
    if (cand.empty() || is_control_kw(lw) || is_def_kw(lw) || is_modifier_kw(lw))
      continue;
    if (is_digit(static_cast<unsigned char>(cand[0]))) continue;
    *sym = cand;
    clean_symbol(sym);
    *kind = "function";
    return;
  }
  // `fn name<T>(` style where the '(' is far away, or `def name:`.
  if (!words.empty() && is_def_kw(lower_str(words[0])) && words.size() > 1) {
    *sym = words[1];
    clean_symbol(sym);
    *kind = "function";
  }
}

// ------------------------------------------------------------ chunk emitter
struct Emitter {
  const std::string& path;
  const std::string& lang;
  const std::vector<std::string>& lines;
  const std::vector<LineInfo>* li;  // brace languages only
  const ScanOptions& opt;
  std::vector<CodeChunk> out;

  int max_lines() const { return opt.max_chunk_lines > 4 ? opt.max_chunk_lines : 4; }

  bool blank(size_t i) const { return trim(lines[i]).empty(); }

  void push(size_t s, size_t e, const std::string& sym, const std::string& kind) {
    if (s > e || e >= lines.size()) return;
    const int n = static_cast<int>(e - s) + 1;
    // A three-line anonymous fragment is retrieval noise; a three-line function
    // is exactly what someone is looking for.
    if (n < opt.min_chunk_lines && sym.empty()) return;
    CodeChunk c;
    c.id = static_cast<int32_t>(out.size());
    c.path = path;
    c.lang = lang;
    c.symbol = sym;
    c.kind = kind;
    c.start_line = static_cast<int32_t>(s) + 1;
    c.end_line = static_cast<int32_t>(e) + 1;
    c.text = join_lines(lines, s, e);
    if (trim(c.text).empty()) return;
    c.length = static_cast<int32_t>(code_tokens(c.text, true).size());
    out.push_back(std::move(c));
  }

  // Where to cut an oversized body: a blank line first (that is a statement
  // group boundary the author put there), else the shallowest `;`/`}`.
  size_t find_cut(size_t lo, size_t hi) const {
    for (size_t j = hi; j > lo; --j)
      if (blank(j)) return j;
    int best_depth = INT_MAX;
    size_t best = hi;
    for (size_t j = hi; j > lo; --j) {
      if (!li) break;
      const LineInfo& L = (*li)[j];
      if (L.blank || !(L.ends_semi || L.ends_brace)) continue;
      if (L.depth_after < best_depth) {
        best_depth = L.depth_after;
        best = j;
      }
    }
    return best;
  }

  void emit(size_t s, size_t e, const std::string& sym, const std::string& kind) {
    if (s >= lines.size()) return;
    if (e >= lines.size()) e = lines.size() - 1;
    while (s <= e && blank(s)) ++s;
    while (e > s && blank(e)) --e;
    if (s > e || (s == e && blank(s))) return;
    const size_t cap = static_cast<size_t>(max_lines());
    if (e - s + 1 <= cap) {
      push(s, e, sym, kind);
      return;
    }
    // Oversized: split, but every piece keeps the symbol so a retrieved
    // fragment still says which function it came from.
    size_t cur = s;
    while (cur <= e) {
      size_t limit = std::min(e, cur + cap - 1);
      size_t cut = limit;
      if (limit < e) cut = find_cut(cur + cap / 2, limit);
      if (cut < cur) cut = limit;
      push(cur, cut, sym, "block");
      cur = cut + 1;
    }
  }
};

// --------------------------------------------------------- brace chunking
bool opens_block_within(const std::vector<LineInfo>& li, size_t i, int d0,
                        size_t* open_line) {
  const size_t lookahead = std::min(li.size(), i + 4);
  for (size_t j = i; j < lookahead; ++j) {
    if (li[j].depth_after > d0) {
      *open_line = j;
      return true;
    }
    if (j > i && (li[j].ends_semi || li[j].blank)) return false;  // prototype / decl
  }
  return false;
}

bool is_container_line(const std::string& code) {
  // `namespace X {` and `extern "C" {` must not become one giant chunk holding
  // the whole file; descend into them and chunk their members instead.
  const std::string s = strip_prologue(code);
  if (starts_with(s, "namespace")) return true;
  if (starts_with(s, "extern") && s.find('(') == std::string::npos &&
      s.find('{') != std::string::npos)
    return true;
  return false;
}

bool looks_like_definition(const std::vector<LineInfo>& li, size_t i,
                           const std::string& lang, size_t* open_line) {
  const LineInfo& L = li[i];
  if (L.blank) return false;
  const char c0 = L.code[0];
  if (c0 == '}' || c0 == ')' || c0 == '#' || c0 == '*' || c0 == ':' || c0 == ',')
    return false;
  const bool multi = opens_block_within(li, i, L.depth_before, open_line);
  // `int f() { return 1; }` and `struct P { int a; };` open and close on one
  // line; they are still definitions and still deserve their symbol.
  const bool single = !multi && L.has_brace && (L.ends_brace || L.ends_semi);
  if (!multi && !single) return false;
  if (single) *open_line = i;
  const std::string s = strip_prologue(L.code);
  if (s.empty()) return false;
  // First word.
  size_t w = 0;
  while (w < s.size() && is_name_byte(static_cast<unsigned char>(s[w]))) ++w;
  const std::string first = lower_str(s.substr(0, w));
  if (is_control_kw(first)) return false;
  if (is_class_kw(first) || is_def_kw(first)) return true;
  const size_t brace = s.find('{');
  const size_t paren = s.find('(');
  if (paren != std::string::npos && (brace == std::string::npos || paren < brace)) {
    // `x = f(...)` and `foo(bar) {` differ by an '=' before the parenthesis:
    // the former is an initialiser, not a definition.
    const size_t eq = s.find('=');
    if (eq != std::string::npos && eq < paren) return false;
    // On one line, a function definition ends in '}'; `handler({1, 2});` is a
    // call with a braced argument, not a definition.
    if (single && !L.ends_brace) return false;
    return true;
  }
  if (lang == "css") return true;  // a selector list plus its block
  return false;
}

// The comment block and the template/attribute/decorator lines immediately above
// a definition are the best description of it that exists; they belong with it.
size_t attach_prefix(const std::vector<std::string>& lines,
                     const std::vector<LineInfo>* li, size_t def, size_t floor_line) {
  size_t s = def;
  while (s > floor_line) {
    const size_t j = s - 1;
    const std::string t = trim(lines[j]);
    if (t.empty()) break;  // a blank line ends the association
    bool ok = false;
    if (li) {
      ok = (*li)[j].comment;
      if (!ok) {
        const std::string& code = (*li)[j].code;
        ok = starts_with(code, "template") || starts_with(code, "[[") ||
             starts_with(code, "#[") || (!code.empty() && code[0] == '@');
      }
    } else {
      ok = t[0] == '#' || t[0] == '@';
    }
    if (!ok) break;
    s = j;
  }
  return s;
}

std::vector<CodeChunk> chunk_braces(const std::string& path, const std::string& lang,
                                    const std::vector<std::string>& lines,
                                    const ScanOptions& opt) {
  const std::vector<LineInfo> li = scan_braces(lines, lang);
  Emitter em{path, lang, lines, &li, opt, {}};
  const size_t n = lines.size();
  size_t gap = 0;  // start of the run of lines not claimed by any definition
  size_t i = 0;
  while (i < n) {
    const LineInfo& L = li[i];
    // Depth 0 is a free function; depth 1 a member of a class or namespace;
    // depth 2 covers `namespace X { class Y { ... } }`.
    if (L.blank || L.depth_before > 2 || is_container_line(L.code)) {
      ++i;
      continue;
    }
    size_t open_line = i;
    if (!looks_like_definition(li, i, lang, &open_line)) {
      ++i;
      continue;
    }
    const int d0 = L.depth_before;
    size_t end = open_line;
    while (end < n && li[end].depth_after > d0) ++end;
    if (end >= n) end = n - 1;
    if (end + 1 < n && li[end + 1].code == ";") ++end;  // `}\n;`

    const size_t start = attach_prefix(lines, &li, i, gap);
    if (start > gap) em.emit(gap, start - 1, "", gap == 0 ? "file" : "block");
    std::string sym, kind;
    std::string def = li[i].code;
    if (open_line > i)
      for (size_t k = i + 1; k <= open_line; ++k) def += " " + li[k].code;
    extract_symbol(def, &sym, &kind);
    em.emit(start, end, sym, kind);
    i = end + 1;
    gap = i;
  }
  if (gap < n) em.emit(gap, n - 1, "", gap == 0 ? "file" : "block");
  return std::move(em.out);
}

// -------------------------------------------------------- python chunking
int indent_of(const std::string& s) {
  int n = 0;
  for (char c : s) {
    if (c == ' ') ++n;
    else if (c == '\t') n += 4;
    else break;
  }
  return n;
}

// Marks the lines that begin inside a triple-quoted string, so a `def` inside a
// docstring does not start a chunk.
std::vector<bool> python_docstring_map(const std::vector<std::string>& lines) {
  std::vector<bool> inside(lines.size(), false);
  bool in_str = false;
  char q = '"';
  for (size_t i = 0; i < lines.size(); ++i) {
    inside[i] = in_str;
    const std::string& s = lines[i];
    size_t j = 0;
    while (j < s.size()) {
      if (in_str) {
        if (s[j] == q && s.compare(j, 3, std::string(3, q)) == 0) {
          in_str = false;
          j += 3;
        } else {
          ++j;
        }
        continue;
      }
      if (s[j] == '#') break;
      if ((s[j] == '"' || s[j] == '\'')) {
        const char c = s[j];
        if (s.compare(j, 3, std::string(3, c)) == 0) {
          q = c;
          in_str = true;
          j += 3;
          continue;
        }
        ++j;  // single-quoted string: skip its body on this line
        while (j < s.size()) {
          if (s[j] == '\\') { j += 2; continue; }
          if (s[j] == c) { ++j; break; }
          ++j;
        }
        continue;
      }
      ++j;
    }
  }
  return inside;
}

std::vector<CodeChunk> chunk_python(const std::string& path, const std::string& lang,
                                    const std::vector<std::string>& lines,
                                    const ScanOptions& opt) {
  Emitter em{path, lang, lines, nullptr, opt, {}};
  const std::vector<bool> in_doc = python_docstring_map(lines);
  const size_t n = lines.size();
  size_t gap = 0, i = 0;
  while (i < n) {
    const std::string t = trim(lines[i]);
    const bool is_def = !in_doc[i] &&
                        (starts_with(t, "def ") || starts_with(t, "class ") ||
                         starts_with(t, "async def "));
    if (!is_def) {
      ++i;
      continue;
    }
    const int ind = indent_of(lines[i]);
    // The body runs until the first non-blank line that dedents to or past the
    // definition's own indentation.
    size_t last = i;
    size_t j = i + 1;
    for (; j < n; ++j) {
      if (trim(lines[j]).empty()) continue;
      if (indent_of(lines[j]) <= ind && !in_doc[j]) break;
      last = j;
    }
    const size_t start = attach_prefix(lines, nullptr, i, gap);
    if (start > gap) em.emit(gap, start - 1, "", gap == 0 ? "file" : "block");

    std::string sym, kind = "function";
    {
      std::string d = t;
      if (starts_with(d, "async ")) d = trim(d.substr(6));
      const bool cls = starts_with(d, "class ");
      d = trim(d.substr(cls ? 6 : 4));
      size_t k = 0;
      while (k < d.size() && is_plain_name_byte(static_cast<unsigned char>(d[k]))) ++k;
      sym = d.substr(0, k);
      kind = cls ? "class" : "function";
    }
    em.emit(start, last, sym, kind);
    i = last + 1;
    gap = i;
  }
  if (gap < n) em.emit(gap, n - 1, "", gap == 0 ? "file" : "block");
  return std::move(em.out);
}

// --------------------------------------------------------- prose chunking
bool is_underline(const std::string& s) {
  const std::string t = trim(s);
  if (t.size() < 2) return false;
  const char c = t[0];
  if (std::strchr("=-~^\"'#*+`_", c) == nullptr) return false;
  for (char x : t)
    if (x != c) return false;
  return true;
}

std::vector<CodeChunk> chunk_prose(const std::string& path, const std::string& lang,
                                   const std::vector<std::string>& lines,
                                   const ScanOptions& opt) {
  Emitter em{path, lang, lines, nullptr, opt, {}};
  const size_t n = lines.size();
  std::vector<size_t> heads;
  std::vector<std::string> titles;
  bool in_fence = false;
  for (size_t i = 0; i < n; ++i) {
    const std::string t = trim(lines[i]);
    if (starts_with(t, "```") || starts_with(t, "~~~")) {
      in_fence = !in_fence;
      continue;
    }
    if (in_fence) continue;
    if (!t.empty() && t[0] == '#' && !is_underline(t)) {
      size_t h = 0;
      while (h < t.size() && t[h] == '#') ++h;
      if (h <= 6 && h < t.size() && (t[h] == ' ' || t[h] == '\t')) {
        heads.push_back(i);
        std::string title = trim(t.substr(h));
        while (!title.empty() && title.back() == '#') title.pop_back();
        titles.push_back(trim(title));
      }
      continue;
    }
    // Setext / rst underline style: text on one line, ==== under it.
    if (!t.empty() && i + 1 < n && is_underline(lines[i + 1]) &&
        trim(lines[i + 1]).size() + 2 >= t.size() && !is_underline(t)) {
      heads.push_back(i);
      titles.push_back(t);
    }
  }
  if (heads.empty()) {
    em.emit(0, n ? n - 1 : 0, "", "doc");
    return std::move(em.out);
  }
  if (heads[0] > 0) em.emit(0, heads[0] - 1, "", "doc");
  for (size_t h = 0; h < heads.size(); ++h) {
    const size_t s = heads[h];
    const size_t e = (h + 1 < heads.size()) ? heads[h + 1] - 1 : n - 1;
    // The heading text is stored as the chunk's symbol: it is the only name a
    // prose chunk has, and `find_symbol("Installation")` should reach it.
    em.emit(s, e, titles[h], "doc");
  }
  return std::move(em.out);
}

// ------------------------------------------------------ fixed-size fallback
std::vector<CodeChunk> chunk_fixed(const std::string& path, const std::string& lang,
                                   const std::vector<std::string>& lines,
                                   const ScanOptions& opt) {
  Emitter em{path, lang, lines, nullptr, opt, {}};
  const size_t n = lines.size();
  const size_t cap = static_cast<size_t>(em.max_lines());
  const size_t step = cap > 2 ? cap - 2 : 1;  // two lines of overlap
  for (size_t s = 0; s < n; s += step) {
    const size_t e = std::min(n - 1, s + cap - 1);
    em.emit(s, e, "", "block");
    if (e == n - 1) break;
  }
  return std::move(em.out);
}

}  // namespace

std::string CodeChunk::header() const {
  std::string h = path + ":" + std::to_string(start_line) + "-" +
                  std::to_string(end_line);
  if (!symbol.empty()) h += "  " + symbol + "()";
  return h;
}

std::vector<CodeChunk> chunk_source(const std::string& path, const std::string& text,
                                    const ScanOptions& opt) {
  const std::string lang = language_of(path);
  const std::vector<std::string> lines = split_lines(text);
  if (lines.empty()) return {};
  std::vector<CodeChunk> out;
  if (is_brace_lang(lang)) out = chunk_braces(path, lang, lines, opt);
  else if (lang == "python" || lang == "ruby") out = chunk_python(path, lang, lines, opt);
  else if (is_prose_lang(lang)) out = chunk_prose(path, lang, lines, opt);
  else out = chunk_fixed(path, lang, lines, opt);
  for (size_t i = 0; i < out.size(); ++i) out[i].id = static_cast<int32_t>(i);
  return out;
}

// ============================================================== embedders
void Embedder::embed_batch(const std::vector<std::string>& texts, float* out) {
  const size_t d = static_cast<size_t>(dim());
  for (size_t i = 0; i < texts.size(); ++i) embed(texts[i], out + i * d);
}

void HashEmbedder::embed(const std::string& text, float* out) {
  if (dim_ <= 0) return;
  const size_t d = static_cast<size_t>(dim_);
  std::fill(out, out + d, 0.0f);

  // Bounded work per chunk: the head of a chunk carries its identity, and a
  // 200 kB generated blob must not cost 200 kB of hashing.
  std::string t = utf8_truncate(text, kEmbedInputBytes);
  NormalizeOptions no;
  no.collapse_spaces = true;  // whitespace runs carry no signal for a bag of grams
  t = normalize_text(t, no);
  for (char& c : t) c = lower_ascii(c);

  const std::vector<std::string> toks = code_tokens(t, true);
  const float wt = 1.0f / std::sqrt(static_cast<float>(toks.size()) + 1.0f);
  for (const std::string& tk : toks) {
    const uint64_t h1 = fnv1a(tk.data(), tk.size());
    const uint64_t h2 = splitmix64(h1);
    out[h1 % d] += (h2 & 1) ? wt : -wt;
  }
  // Character 4-grams over bytes (not code points): they are only hash inputs,
  // so cutting inside a multi-byte character is harmless and deterministic, and
  // they are what gives the space its robustness to typos and morphology.
  if (t.size() >= 4) {
    const size_t ng = t.size() - 3;
    const float wg = 0.5f / std::sqrt(static_cast<float>(ng) + 1.0f);
    for (size_t i = 0; i < ng; ++i) {
      const uint64_t h1 = fnv1a(t.data() + i, 4);
      const uint64_t h2 = splitmix64(h1);
      out[h1 % d] += (h2 & 1) ? wg : -wg;
    }
  }
  double ss = 0.0;
  for (size_t i = 0; i < d; ++i) ss += static_cast<double>(out[i]) * out[i];
  if (ss > 0.0) {
    const float inv = static_cast<float>(1.0 / std::sqrt(ss));
    for (size_t i = 0; i < d; ++i) out[i] *= inv;
  }
}

EmbedderPtr make_gguf_embedder(const std::string& gguf_path, int threads,
                               std::string* err) {
  (void)gguf_path;
  (void)threads;
  // The llama.cpp side lives behind the backend seam on purpose: this file must
  // compile (and its tests must run) with no llama.cpp headers anywhere.
#ifndef SLM_WITH_LLAMA
  if (err) *err = "built without llama.cpp";
#else
  if (err) *err = "gguf embedder is wired up in backend_gguf.cpp";
#endif
  return nullptr;
}

// ==================================================================== RRF
std::vector<std::pair<int32_t, double>> reciprocal_rank_fusion(
    const std::vector<std::vector<int32_t>>& ranked_lists, double k,
    const std::vector<double>& list_weights) {
  std::unordered_map<int32_t, double> acc;
  for (size_t l = 0; l < ranked_lists.size(); ++l) {
    const double w = l < list_weights.size() ? list_weights[l] : 1.0;
    const std::vector<int32_t>& list = ranked_lists[l];
    for (size_t r = 0; r < list.size(); ++r)
      acc[list[r]] += w / (k + static_cast<double>(r + 1));  // ranks are 1-based
  }
  std::vector<std::pair<int32_t, double>> out(acc.begin(), acc.end());
  // Ties broken by id so the output is reproducible across runs and platforms.
  std::sort(out.begin(), out.end(),
            [](const std::pair<int32_t, double>& a,
               const std::pair<int32_t, double>& b) {
              if (a.second != b.second) return a.second > b.second;
              return a.first < b.first;
            });
  return out;
}


// ============================================================ ignore rules
namespace {

struct IgnoreRule {
  std::string pat;
  bool negate = false;
  bool dir_only = false;
  bool anchored = false;   // pattern had a leading '/'
  bool has_slash = false;  // pattern matches against a path, not a basename
  bool wild = false;
};

struct IgnoreFile {
  std::string dir;  // directory the rules are relative to ("" = index root)
  std::vector<IgnoreRule> rules;
};

std::vector<IgnoreRule> parse_gitignore(const std::string& text) {
  std::vector<IgnoreRule> out;
  for (const std::string& raw : split_lines(text)) {
    std::string s = raw;
    // Leading whitespace is not significant; trailing whitespace is stripped
    // unless escaped (we do not support the escape - nobody writes it).
    s = trim(s);
    if (s.empty() || s[0] == '#') continue;
    IgnoreRule r;
    if (s[0] == '!') {
      r.negate = true;
      s.erase(s.begin());
    }
    if (!s.empty() && s[0] == '/') {
      r.anchored = true;
      s.erase(s.begin());
    }
    if (!s.empty() && s.back() == '/') {
      r.dir_only = true;
      s.pop_back();
    }
    if (s.empty()) continue;
    r.has_slash = s.find('/') != std::string::npos;
    r.wild = s.find_first_of("*?[") != std::string::npos;
    r.pat = s;
    out.push_back(std::move(r));
  }
  return out;
}

// Glob matching with gitignore semantics: '*' and '?' stop at '/', '**' spans
// segments.  Recursive, but only on wildcards, so the depth is bounded by the
// number of '*' in the pattern.
bool glob_match(const char* p, const char* s, const char* s_begin) {
  while (*p) {
    if (*p == '*') {
      if (p[1] == '*') {
        const char* q = p + 2;
        if (*q == '/') ++q;
        if (!*q) return true;  // "a/**" matches everything below a
        for (const char* t = s;; ++t) {
          const bool at_boundary = (t == s_begin) || (t > s_begin && t[-1] == '/');
          if (at_boundary && glob_match(q, t, s_begin)) return true;
          if (!*t) return false;
        }
      }
      ++p;
      for (const char* t = s;; ++t) {
        if (glob_match(p, t, s_begin)) return true;
        if (!*t || *t == '/') return false;
      }
    }
    if (!*s) return false;
    if (*p == '?') {
      if (*s == '/') return false;
    } else if (*p != *s) {
      return false;
    }
    ++p;
    ++s;
  }
  return *s == '\0';
}

bool glob_match(const std::string& pat, const std::string& s) {
  return glob_match(pat.c_str(), s.c_str(), s.c_str());
}

bool rule_matches(const IgnoreRule& r, const std::string& rel, bool is_dir) {
  if (r.dir_only && !is_dir) return false;
  const std::string base = basename_of(rel);
  if (r.anchored || r.has_slash) {
    if (glob_match(r.pat, rel)) return true;
    if (!r.anchored) {
      // A non-anchored pattern with a slash still matches at any depth here;
      // git is stricter, but being lenient only ever ignores more of what the
      // user meant to ignore.
      for (size_t i = 0; i + 1 < rel.size(); ++i)
        if (rel[i] == '/' && glob_match(r.pat, rel.substr(i + 1))) return true;
    }
    return false;
  }
  if (glob_match(r.pat, base)) return true;
  // Plain (wildcard-free) patterns also act as suffix rules, which is how most
  // people expect ".o" or "min.js" to behave.
  if (!r.wild && ends_with(base, r.pat)) return true;
  return false;
}

// -1 = no rule matched, 1 = ignored, 0 = explicitly re-included.
int match_rules(const IgnoreFile& gi, const std::string& rel, bool is_dir) {
  if (!gi.dir.empty()) {
    if (rel.size() <= gi.dir.size() + 1) return -1;
    if (rel.compare(0, gi.dir.size(), gi.dir) != 0) return -1;
  }
  const std::string sub =
      gi.dir.empty() ? rel : rel.substr(gi.dir.size() + 1);
  int verdict = -1;  // the last matching rule in the file wins
  for (const IgnoreRule& r : gi.rules)
    if (rule_matches(r, sub, is_dir)) verdict = r.negate ? 0 : 1;
  return verdict;
}

bool path_ignored(const std::vector<IgnoreFile>& stack, const std::string& rel,
                  bool is_dir) {
  // Deepest .gitignore wins, which is why the stack is walked backwards.
  for (size_t k = stack.size(); k-- > 0;) {
    const int v = match_rules(stack[k], rel, is_dir);
    if (v >= 0) return v == 1;
  }
  return false;
}

bool extra_ignored(const ScanOptions& opt, const std::string& rel) {
  for (const std::string& g : opt.extra_ignores) {
    if (g.empty()) continue;
    if (g.find_first_of("*?") != std::string::npos) {
      if (glob_match(g, rel) || glob_match(g, basename_of(rel))) return true;
    } else if (rel.find(g) != std::string::npos) {
      return true;  // substring / suffix rule
    }
  }
  return false;
}

bool ext_allowed(const ScanOptions& opt, const std::string& rel) {
  if (opt.only_exts.empty()) return true;
  const std::string lb = lower_str(basename_of(rel));
  const size_t dot = lb.find_last_of('.');
  const std::string ext = dot == std::string::npos ? std::string() : lb.substr(dot + 1);
  for (const std::string& e : opt.only_exts) {
    std::string want = lower_str(e);
    if (!want.empty() && want[0] == '.') want.erase(want.begin());
    if (!ext.empty() && ext == want) return true;
    if (lb == want) return true;  // "makefile", "cmakelists.txt"
  }
  return false;
}

// A NUL byte is decisive; beyond that, control characters are what separates a
// UTF-8 text file from a .png.  Bytes >= 0x80 are *not* counted: they are the
// normal case for Persian source comments.
bool looks_binary(const std::string& head) {
  size_t bad = 0, n = std::min(head.size(), kBinarySniffBytes);
  for (size_t i = 0; i < n; ++i) {
    const unsigned char c = static_cast<unsigned char>(head[i]);
    if (c == 0) return true;
    if (c < 0x20 && c != '\t' && c != '\n' && c != '\r' && c != '\f' && c != '\v')
      ++bad;
    else if (c == 0x7F)
      ++bad;
  }
  return n > 0 && bad * 100 > n * 30;
}

bool read_whole_file(const std::string& path, std::string* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  if (n < 0) return false;
  f.seekg(0, std::ios::beg);
  out->resize(static_cast<size_t>(n));
  if (n > 0) f.read(&(*out)[0], n);
  return true;
}

struct FileEntry {
  std::string rel;
  std::string abs;
  int64_t size = 0;
  int64_t mtime = 0;
};

void walk_dir(const fs::path& dir, const std::string& rel_prefix,
              const ScanOptions& opt, std::vector<IgnoreFile>* stack,
              std::vector<FileEntry>* out, IndexStats* st,
              std::atomic<bool>* cancel) {
  if (cancel && cancel->load()) return;
  bool pushed = false;
  if (opt.use_gitignore) {
    std::error_code ec;
    const fs::path gi = dir / ".gitignore";
    if (fs::is_regular_file(gi, ec)) {
      std::string text;
      if (read_whole_file(gi.string(), &text)) {
        stack->push_back(IgnoreFile{rel_prefix, parse_gitignore(text)});
        pushed = true;
      }
    }
  }
  std::vector<fs::directory_entry> entries;
  {
    std::error_code ec;
    fs::directory_iterator it(dir, ec);
    if (!ec)
      for (const fs::directory_entry& e : it) entries.push_back(e);
  }
  // Filesystem order is unspecified; sort so chunk ids and every derived
  // artefact (context blocks, overview, saved index) are reproducible.
  std::sort(entries.begin(), entries.end(),
            [](const fs::directory_entry& a, const fs::directory_entry& b) {
              return a.path().filename().string() < b.path().filename().string();
            });

  for (const fs::directory_entry& e : entries) {
    if (cancel && cancel->load()) break;
    const std::string name = e.path().filename().string();
    if (name == ".git") continue;  // never, under any option
    std::error_code ec;
    const bool is_link = fs::is_symlink(e.symlink_status(ec));
    const bool is_dir = e.is_directory(ec);
    if (opt.skip_hidden && !name.empty() && name[0] == '.') continue;
    const std::string rel = rel_prefix.empty() ? name : rel_prefix + "/" + name;
    if (path_ignored(*stack, rel, is_dir)) {
      if (!is_dir) ++st->skipped_ignored;
      continue;
    }
    if (is_dir) {
      if (is_link) continue;  // symlinked directories are how a walk loops forever
      walk_dir(e.path(), rel, opt, stack, out, st, cancel);
      continue;
    }
    if (!e.is_regular_file(ec)) continue;
    if (extra_ignored(opt, rel) || !ext_allowed(opt, rel)) {
      ++st->skipped_ignored;
      continue;
    }
    const int64_t size = static_cast<int64_t>(fs::file_size(e.path(), ec));
    if (ec) continue;
    if (opt.max_file_bytes > 0 && size > opt.max_file_bytes) {
      ++st->skipped_large;
      continue;
    }
    int64_t mtime = 0;
    {
      std::error_code ec2;
      const fs::file_time_type t = fs::last_write_time(e.path(), ec2);
      if (!ec2) mtime = static_cast<int64_t>(t.time_since_epoch().count());
    }
    out->push_back(FileEntry{rel, e.path().string(), size, mtime});
  }
  if (pushed) stack->pop_back();
}

// ------------------------------------------------------------- binary layout
template <typename T>
void wpod(std::ostream& o, const T& v) {
  o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
bool rpod(std::istream& i, T* v) {
  i.read(reinterpret_cast<char*>(v), sizeof(T));
  return static_cast<bool>(i);
}
void wstr(std::ostream& o, const std::string& s) {
  const uint32_t n = static_cast<uint32_t>(s.size());
  wpod(o, n);
  if (n) o.write(s.data(), n);
}
bool rstr(std::istream& i, std::string* s) {
  uint32_t n = 0;
  if (!rpod(i, &n)) return false;
  if (n > (1u << 28)) return false;  // corrupt length: refuse to allocate 4 GB
  s->assign(n, '\0');
  if (n) i.read(&(*s)[0], n);
  return static_cast<bool>(i);
}
void wvec_str(std::ostream& o, const std::vector<std::string>& v) {
  wpod(o, static_cast<uint32_t>(v.size()));
  for (const std::string& s : v) wstr(o, s);
}
bool rvec_str(std::istream& i, std::vector<std::string>* v) {
  uint32_t n = 0;
  if (!rpod(i, &n)) return false;
  if (n > (1u << 20)) return false;
  v->assign(n, std::string());
  for (uint32_t k = 0; k < n; ++k)
    if (!rstr(i, &(*v)[k])) return false;
  return true;
}

}  // namespace

// ================================================================== index
struct CodebaseIndex::Impl {
  ScanOptions opt;
  EmbedderPtr emb;              // never null: the ctor installs HashEmbedder(256)
  int dim = 0;
  std::vector<CodeChunk> chunks;
  std::vector<float> dense;     // chunks.size() x dim, row-major, L2-normalised
  std::unordered_map<std::string, std::vector<std::pair<int32_t, int32_t>>> inv;
  double avgdl = 1.0;
  IndexStats st;

  struct FileRec {
    std::string path;
    int64_t mtime = 0, size = 0, lines = 0;
  };
  std::vector<FileRec> files;
  std::unordered_map<std::string, std::vector<int32_t>> by_path;

  // ---------------------------------------------------------------- building
  void reset() {
    chunks.clear();
    dense.clear();
    inv.clear();
    by_path.clear();
    files.clear();
    avgdl = 1.0;
    st = IndexStats();
  }

  void build_by_path() {
    by_path.clear();
    for (size_t i = 0; i < chunks.size(); ++i) {
      chunks[i].id = static_cast<int32_t>(i);
      by_path[chunks[i].path].push_back(static_cast<int32_t>(i));
    }
  }

  // Term frequencies, document frequencies and avgdl.  Serial on purpose: the
  // postings order is part of the index's determinism.
  void build_lexical() {
    inv.clear();
    st.tokens = 0;
    double total = 0.0;
    std::unordered_map<std::string, int32_t> tf;
    for (size_t i = 0; i < chunks.size(); ++i) {
      const std::vector<std::string> toks = code_tokens(chunks[i].text, true);
      chunks[i].length = static_cast<int32_t>(toks.size());
      total += static_cast<double>(toks.size());
      st.tokens += static_cast<int64_t>(toks.size());
      tf.clear();
      for (const std::string& t : toks) ++tf[t];
      for (const auto& kv : tf)
        inv[kv.first].emplace_back(static_cast<int32_t>(i), kv.second);
    }
    avgdl = chunks.empty() ? 1.0 : std::max(1.0, total / static_cast<double>(chunks.size()));
  }

  // Embeds the chunks listed in `todo`.  The embedder must be re-entrant;
  // HashEmbedder is stateless, and the gguf one gets one context per thread.
  void embed_rows(const std::vector<int32_t>& todo) {
    if (todo.empty() || dim <= 0) return;
    const size_t d = static_cast<size_t>(dim);
    dense.resize(chunks.size() * d, 0.0f);
    const int64_t n = static_cast<int64_t>(todo.size());
    Embedder* e = emb.get();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t j = 0; j < n; ++j) {
      const int32_t i = todo[static_cast<size_t>(j)];
      // The header line carries the path and the symbol name, which is exactly
      // the signal a "where is X" query has and the body often lacks.
      std::string t = chunks[static_cast<size_t>(i)].header();
      t += "\n";
      t += chunks[static_cast<size_t>(i)].text;
      e->embed(t, dense.data() + static_cast<size_t>(i) * d);
    }
  }

  void finish_stats() {
    st.files = static_cast<int64_t>(files.size());
    st.chunks = static_cast<int64_t>(chunks.size());
    st.bytes = 0;
    for (const FileRec& f : files) st.bytes += f.size;
    st.embedder = emb->name();
    st.root = opt.root;
    std::map<std::string, int64_t> hist;
    for (const CodeChunk& c : chunks) hist[c.lang.empty() ? "other" : c.lang] += 1;
    st.by_language.assign(hist.begin(), hist.end());
    size_t mem = chunks.size() * sizeof(CodeChunk) + dense.size() * sizeof(float);
    for (const CodeChunk& c : chunks) mem += c.text.size() + c.path.size() + 48;
    for (const auto& kv : inv)
      mem += kv.first.size() + kv.second.size() * sizeof(std::pair<int32_t, int32_t>) + 48;
    st.memory_bytes = mem;
  }

  // Chunks `entries` (in order), reusing chunks and embedding rows of files that
  // did not change when `incremental`.  This is the body of both scan() and
  // refresh(); they only differ in whether the previous state is consulted.
  bool ingest(const std::vector<FileEntry>& entries, bool incremental,
              const std::function<void(int64_t, int64_t, const std::string&)>& progress,
              std::atomic<bool>* cancel, std::string* err) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<CodeChunk> old_chunks = std::move(chunks);
    std::vector<float> old_dense = std::move(dense);
    std::unordered_map<std::string, std::vector<int32_t>> old_by_path =
        std::move(by_path);
    std::unordered_map<std::string, FileRec> old_files;
    for (const FileRec& f : files) old_files[f.path] = f;
    const int old_dim = dim;
    chunks.clear();
    dense.clear();
    by_path.clear();
    files.clear();

    dim = emb->dim();
    const int64_t total = static_cast<int64_t>(entries.size());
    std::vector<int32_t> src_old;  // new chunk index -> old index, -1 if fresh
    bool full = false;

    const size_t kBatch = 64;  // enough work for the threads, bounded memory
    for (size_t base = 0; base < entries.size() && !full; base += kBatch) {
      if (cancel && cancel->load()) {
        if (err) *err = "cancelled";
        return false;
      }
      if (progress)
        progress(static_cast<int64_t>(base), total, entries[base].rel);

      const size_t hi = std::min(entries.size(), base + kBatch);
      const int64_t nb = static_cast<int64_t>(hi - base);
      std::vector<std::vector<CodeChunk>> got(static_cast<size_t>(nb));
      std::vector<int> status(static_cast<size_t>(nb), 0);  // 1 binary, 2 unreadable
      std::vector<int64_t> nlines(static_cast<size_t>(nb), 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int64_t j = 0; j < nb; ++j) {
        const FileEntry& fe = entries[base + static_cast<size_t>(j)];
        const auto it = incremental ? old_files.find(fe.rel) : old_files.end();
        if (it != old_files.end() && it->second.size == fe.size &&
            it->second.mtime == fe.mtime) {
          status[static_cast<size_t>(j)] = 3;  // unchanged: reuse
          nlines[static_cast<size_t>(j)] = it->second.lines;
          continue;
        }
        std::string text;
        if (!read_whole_file(fe.abs, &text)) {
          status[static_cast<size_t>(j)] = 2;
          continue;
        }
        if (opt.skip_binary && looks_binary(text)) {
          status[static_cast<size_t>(j)] = 1;
          continue;
        }
        nlines[static_cast<size_t>(j)] =
            static_cast<int64_t>(std::count(text.begin(), text.end(), '\n')) + 1;
        got[static_cast<size_t>(j)] = chunk_source(fe.rel, text, opt);
      }

      // Serial merge, in entry order, so ids stay deterministic.
      for (int64_t j = 0; j < nb; ++j) {
        const FileEntry& fe = entries[base + static_cast<size_t>(j)];
        const int s = status[static_cast<size_t>(j)];
        if (s == 1) {
          ++st.skipped_binary;
          continue;
        }
        if (s == 2) continue;
        if (opt.max_total_chunks > 0 &&
            static_cast<int64_t>(chunks.size()) >= opt.max_total_chunks) {
          full = true;
          break;
        }
        if (s == 3) {
          const auto oit = old_by_path.find(fe.rel);
          if (oit != old_by_path.end()) {
            for (int32_t oid : oit->second) {
              chunks.push_back(old_chunks[static_cast<size_t>(oid)]);
              src_old.push_back(oid);
            }
          }
        } else {
          for (CodeChunk& c : got[static_cast<size_t>(j)]) {
            chunks.push_back(std::move(c));
            src_old.push_back(-1);
          }
        }
        files.push_back(FileRec{fe.rel, fe.mtime, fe.size,
                                nlines[static_cast<size_t>(j)]});
      }
    }
    build_by_path();
    build_lexical();

    const auto t1 = std::chrono::steady_clock::now();
    st.scan_seconds = std::chrono::duration<double>(t1 - t0).count();

    // Copy the embeddings we can keep, then embed the rest.
    const size_t d = static_cast<size_t>(dim);
    dense.assign(chunks.size() * d, 0.0f);
    std::vector<int32_t> todo;
    const bool reusable = (old_dim == dim) && !old_dense.empty();
    for (size_t i = 0; i < chunks.size(); ++i) {
      const int32_t oid = src_old[i];
      if (reusable && oid >= 0 &&
          (static_cast<size_t>(oid) + 1) * d <= old_dense.size()) {
        std::memcpy(dense.data() + i * d, old_dense.data() + static_cast<size_t>(oid) * d,
                    d * sizeof(float));
      } else {
        todo.push_back(static_cast<int32_t>(i));
      }
    }
    embed_rows(todo);
    const auto t2 = std::chrono::steady_clock::now();
    st.embed_seconds = std::chrono::duration<double>(t2 - t1).count();
    finish_stats();
    if (progress) progress(total, total, std::string());
    return true;
  }

  // --------------------------------------------------------------- retrieval
  std::vector<std::pair<int32_t, double>> rank_lexical(const std::string& query,
                                                       size_t limit) const {
    std::vector<std::pair<int32_t, double>> out;
    if (chunks.empty()) return out;
    const std::vector<std::string> qt = code_tokens(query, true);
    if (qt.empty()) return out;
    std::unordered_map<std::string, int> qtf;
    for (const std::string& t : qt) ++qtf[t];

    const double k1 = 1.2, b = 0.75;
    const double N = static_cast<double>(chunks.size());
    std::unordered_map<int32_t, double> acc;
    for (const auto& kv : qtf) {
      const auto it = inv.find(kv.first);
      if (it == inv.end()) continue;
      const double df = static_cast<double>(it->second.size());
      const double idf = std::log(1.0 + (N - df + 0.5) / (df + 0.5));
      for (const auto& p : it->second) {
        const double tf = static_cast<double>(p.second);
        const double dl = static_cast<double>(std::max<int32_t>(1, chunks[static_cast<size_t>(p.first)].length));
        acc[p.first] += static_cast<double>(kv.second) * idf * (tf * (k1 + 1.0)) /
                        (tf + k1 * (1.0 - b + b * dl / avgdl));
      }
    }
    out.assign(acc.begin(), acc.end());
    std::sort(out.begin(), out.end(),
              [](const std::pair<int32_t, double>& a, const std::pair<int32_t, double>& b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;
              });
    if (out.size() > limit) out.resize(limit);
    return out;
  }

  std::vector<std::pair<int32_t, double>> rank_dense(const std::string& query,
                                                      size_t limit) const {
    std::vector<std::pair<int32_t, double>> out;
    const size_t d = static_cast<size_t>(dim);
    if (chunks.empty() || dim <= 0 || dense.size() < chunks.size() * d) return out;
    std::vector<float> q(d, 0.0f);
    emb->embed(query, q.data());
    out.resize(chunks.size());
    const float* qp = q.data();
    for (size_t i = 0; i < chunks.size(); ++i) {
      const float* row = dense.data() + i * d;
      // Flat, contiguous, no aliasing, no branches: this is the loop the
      // vectoriser wants.  Both sides are L2-normalised, so the dot product
      // *is* the cosine.
      float s = 0.0f;
      for (size_t j = 0; j < d; ++j) s += row[j] * qp[j];
      out[i] = {static_cast<int32_t>(i), static_cast<double>(s)};
    }
    std::sort(out.begin(), out.end(),
              [](const std::pair<int32_t, double>& a, const std::pair<int32_t, double>& b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;
              });
    if (out.size() > limit) out.resize(limit);
    return out;
  }

  std::vector<SearchHit> search(const std::string& query, size_t k,
                                SearchMode mode) const {
    std::vector<SearchHit> hits;
    if (chunks.empty() || k == 0 || trim(query).empty()) return hits;
    const size_t pool = std::max<size_t>(50, k * 4);
    std::vector<std::pair<int32_t, double>> lex, den;
    if (mode != SearchMode::kDense) lex = rank_lexical(query, pool);
    if (mode != SearchMode::kLexical) den = rank_dense(query, pool);

    std::unordered_map<int32_t, int> lrank, drank;
    std::unordered_map<int32_t, double> lscore, dscore;
    for (size_t i = 0; i < lex.size(); ++i) {
      lrank[lex[i].first] = static_cast<int>(i) + 1;
      lscore[lex[i].first] = lex[i].second;
    }
    for (size_t i = 0; i < den.size(); ++i) {
      drank[den[i].first] = static_cast<int>(i) + 1;
      dscore[den[i].first] = den[i].second;
    }

    std::vector<std::pair<int32_t, double>> base;
    std::string how;
    if (mode == SearchMode::kHybrid) {
      std::vector<std::vector<int32_t>> lists(2);
      for (const auto& p : lex) lists[0].push_back(p.first);
      for (const auto& p : den) lists[1].push_back(p.first);
      base = reciprocal_rank_fusion(lists, 60.0, {1.0, 1.0});
      how = "lexical+vector fusion";
    } else if (mode == SearchMode::kLexical) {
      base = lex;
      how = "lexical match";
    } else {
      base = den;
      how = "vector match";
    }
    if (base.empty()) return hits;
    // Normalise to [0,1] so the structural bonuses below mean the same thing
    // whatever retriever produced the base score.
    double top = 0.0;
    for (const auto& p : base) top = std::max(top, p.second);
    if (top <= 0.0) top = 1.0;

    const std::vector<std::string> qt = code_tokens(query, true);
    const std::string qlow = lower_str(trim(query));

    hits.reserve(base.size());
    for (const auto& p : base) {
      const CodeChunk& c = chunks[static_cast<size_t>(p.first)];
      SearchHit h;
      h.chunk = p.first;
      h.bm25 = lscore.count(p.first) ? lscore[p.first] : 0.0;
      h.dense = dscore.count(p.first) ? dscore[p.first] : 0.0;
      h.lexical_rank = lrank.count(p.first) ? lrank[p.first] : -1;
      h.dense_rank = drank.count(p.first) ? drank[p.first] : -1;
      double s = p.second / top;
      std::string why = how;

      const std::string sym = lower_str(c.symbol);
      std::string tail = sym;
      const size_t cc = tail.rfind("::");
      if (cc != std::string::npos) tail = tail.substr(cc + 2);
      bool exact = false, partial = false;
      if (!sym.empty()) {
        if (sym == qlow || tail == qlow) exact = true;
        for (const std::string& t : qt) {
          if (t == sym || t == tail) exact = true;
          else if (t.size() >= 3 && sym.find(t) != std::string::npos) partial = true;
        }
      }
      if (exact) {
        s += 1.0;  // "where is X defined" must not lose to a paraphrase
        why = "exact symbol match";
      } else if (partial) {
        s += 0.2;
        why += ", symbol overlap";
      }
      const std::string plow = lower_str(c.path);
      for (const std::string& t : qt) {
        if (t.size() >= 3 && plow.find(t) != std::string::npos) {
          s += 0.25;
          why += ", path match";
          break;
        }
      }
      // Long chunks win BM25 ties for uninteresting reasons; nudge them down.
      const double excess = static_cast<double>(c.length) - 300.0;
      if (excess > 0.0) s -= 0.15 * std::min(1.0, excess / 700.0);
      h.score = s;
      h.why = why;
      hits.push_back(std::move(h));
    }
    std::sort(hits.begin(), hits.end(), [](const SearchHit& a, const SearchHit& b) {
      if (a.score != b.score) return a.score > b.score;
      return a.chunk < b.chunk;
    });
    if (hits.size() > k) hits.resize(k);
    return hits;
  }

  std::vector<SearchHit> find_symbol(const std::string& name, size_t k) const {
    std::vector<SearchHit> hits;
    const std::string want = trim(name);
    if (want.empty() || chunks.empty() || k == 0) return hits;
    const std::string wlow = lower_str(want);

    // Three passes of decreasing confidence; a definition beats a mention.
    for (int pass = 0; pass < 3 && hits.empty(); ++pass) {
      for (size_t i = 0; i < chunks.size(); ++i) {
        const CodeChunk& c = chunks[i];
        if (c.symbol.empty()) continue;
        const std::string sym = c.symbol;
        const std::string slow = lower_str(sym);
        std::string tail = slow;
        const size_t cc = tail.rfind("::");
        if (cc != std::string::npos) tail = tail.substr(cc + 2);
        bool ok = false;
        if (pass == 0) ok = (sym == want) || (slow == wlow && sym.size() == want.size());
        else if (pass == 1) ok = (slow == wlow) || (tail == wlow);
        else ok = slow.find(wlow) != std::string::npos;
        if (!ok) continue;
        SearchHit h;
        h.chunk = static_cast<int32_t>(i);
        h.score = 1.0 - 0.2 * pass;
        if (c.kind == "function" || c.kind == "class") h.score += 0.1;
        h.why = pass == 0 ? "exact symbol match"
                          : (pass == 1 ? "case-insensitive symbol match"
                                       : "symbol substring match");
        hits.push_back(std::move(h));
      }
    }
    if (hits.empty()) return search(name, k, SearchMode::kLexical);
    std::sort(hits.begin(), hits.end(), [](const SearchHit& a, const SearchHit& b) {
      if (a.score != b.score) return a.score > b.score;
      return a.chunk < b.chunk;
    });
    if (hits.size() > k) hits.resize(k);
    return hits;
  }

  // ------------------------------------------------------------------ output
  std::string excerpt(const CodeChunk& c, size_t budget) const {
    const std::string pre = "--- " + c.header() + " ---\n";
    if (budget < pre.size() + 80) return std::string();
    const std::vector<std::string> lines = split_lines(c.text);
    const size_t room = budget - pre.size();
    size_t head = 0, tail = 0, used = 40;  // 40 = room for the marker line
    while (head + tail + 1 < lines.size()) {
      const bool take_head = (head <= tail);
      const std::string& l = take_head ? lines[head] : lines[lines.size() - 1 - tail];
      if (used + l.size() + 1 > room) break;
      used += l.size() + 1;
      if (take_head) ++head; else ++tail;
    }
    if (head == 0 && tail == 0) return std::string();
    std::string out = pre;
    for (size_t i = 0; i < head; ++i) out += lines[i] + "\n";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "... %zu lines omitted ...\n",
                  lines.size() - head - tail);
    out += buf;
    for (size_t i = tail; i-- > 0;) out += lines[lines.size() - 1 - i] + "\n";
    // utf8_truncate, never a raw resize: a cut multi-byte character would make
    // the prompt (and the JSONL audit log that records it) invalid UTF-8.
    return utf8_truncate(out, budget);
  }

  std::string context_block(const std::string& query, size_t budget,
                            size_t max_chunks) const {
    std::string out;
    if (budget == 0 || max_chunks == 0) return out;
    const std::vector<SearchHit> hits =
        search(query, std::max<size_t>(max_chunks * 3, 24), SearchMode::kHybrid);
    std::unordered_map<std::string, int> per_file;
    size_t used = 0;
    for (const SearchHit& h : hits) {
      if (used >= max_chunks) break;
      const CodeChunk& c = chunks[static_cast<size_t>(h.chunk)];
      int& n = per_file[c.path];
      if (n >= 2) continue;  // one file must not crowd out the rest
      const std::string block = "--- " + c.header() + " ---\n" + c.text + "\n";
      if (out.size() + block.size() <= budget) {
        out += block;
        ++n;
        ++used;
        continue;
      }
      const std::string cut = excerpt(c, budget - out.size());
      if (!cut.empty()) {
        out += cut;
        ++n;
        ++used;
      }
      break;  // the budget is spent either way
    }
    return utf8_truncate(out, budget);
  }

  std::string overview(size_t budget) const {
    std::string o;
    o += "repository: " + (opt.root.empty() ? std::string("(none)") : opt.root) + "\n";
    char line[256];
    std::snprintf(line, sizeof(line), "%zu files, %zu chunks, %s embeddings\n",
                  files.size(), chunks.size(), emb->name().c_str());
    o += line;

    // Languages, ordered by chunk count then name so the digest is stable.
    struct LangAgg {
      int64_t files = 0, chunks = 0, lines = 0;
    };
    std::map<std::string, LangAgg> agg;
    for (const FileRec& f : files) {
      const std::string l = language_of(f.path);
      LangAgg& a = agg[l.empty() ? "other" : l];
      a.files += 1;
      a.lines += f.lines;
    }
    for (const CodeChunk& c : chunks) agg[c.lang.empty() ? "other" : c.lang].chunks += 1;
    std::vector<std::pair<std::string, LangAgg>> langs(agg.begin(), agg.end());
    std::sort(langs.begin(), langs.end(),
              [](const std::pair<std::string, LangAgg>& a,
                 const std::pair<std::string, LangAgg>& b) {
                if (a.second.chunks != b.second.chunks)
                  return a.second.chunks > b.second.chunks;
                return a.first < b.first;
              });
    o += "\nlanguages:\n";
    for (const auto& l : langs) {
      std::snprintf(line, sizeof(line), "  %-12s %5lld files  %5lld chunks  %7lld lines\n",
                    l.first.c_str(), static_cast<long long>(l.second.files),
                    static_cast<long long>(l.second.chunks),
                    static_cast<long long>(l.second.lines));
      o += line;
    }

    std::vector<FileRec> big = files;
    std::sort(big.begin(), big.end(), [](const FileRec& a, const FileRec& b) {
      if (a.size != b.size) return a.size > b.size;
      return a.path < b.path;
    });
    o += "\nlargest files:\n";
    for (size_t i = 0; i < big.size() && i < 10; ++i) {
      std::snprintf(line, sizeof(line), "  %-52s %7lld B  %5lld lines\n",
                    big[i].path.c_str(), static_cast<long long>(big[i].size),
                    static_cast<long long>(big[i].lines));
      o += line;
    }

    std::map<std::string, int64_t> dirs;
    for (const FileRec& f : files) {
      const size_t s1 = f.path.find('/');
      if (s1 == std::string::npos) {
        dirs["."] += 1;
        continue;
      }
      dirs[f.path.substr(0, s1)] += 1;
      const size_t s2 = f.path.find('/', s1 + 1);
      if (s2 != std::string::npos) dirs[f.path.substr(0, s2)] += 1;
    }
    o += "\nlayout:\n";
    for (const auto& kv : dirs) {
      std::snprintf(line, sizeof(line), "  %-52s %5lld files\n", kv.first.c_str(),
                    static_cast<long long>(kv.second));
      o += line;
    }

    o += "\nentry points:\n";
    bool any_entry = false;
    for (const CodeChunk& c : chunks) {
      const std::string s = lower_str(c.symbol);
      if (s == "main" || ends_with(s, "::main")) {
        o += "  main() at " + c.path + ":" + std::to_string(c.start_line) + "\n";
        any_entry = true;
      }
    }
    for (const FileRec& f : files) {
      const std::string b = lower_str(basename_of(f.path));
      const auto bit = by_path.find(f.path);
      if (b == "cmakelists.txt" && bit != by_path.end()) {
        for (const int32_t id : bit->second) {
          const std::string& t = chunks[static_cast<size_t>(id)].text;
          for (const char* kw : {"add_executable(", "add_library("}) {
            size_t p = 0;
            while ((p = t.find(kw, p)) != std::string::npos) {
              p += std::strlen(kw);
              size_t e = p;
              while (e < t.size() && is_name_byte(static_cast<unsigned char>(t[e]))) ++e;
              if (e > p) {
                o += "  cmake target " + t.substr(p, e - p) + " (" + f.path + ")\n";
                any_entry = true;
              }
            }
          }
        }
      } else if (b == "package.json" || b == "pyproject.toml" || b == "go.mod" ||
                 b == "cargo.toml" || b == "setup.py") {
        o += "  project manifest " + f.path + "\n";
        any_entry = true;
      }
    }
    if (!any_entry) o += "  (none detected)\n";

    for (const FileRec& f : files) {
      if (f.path.find('/') != std::string::npos) continue;  // root README only
      if (!starts_with(lower_str(basename_of(f.path)), "readme")) continue;
      const auto it = by_path.find(f.path);
      if (it == by_path.end()) break;
      o += "\n" + f.path + " headings:\n";
      int n = 0;
      for (const int32_t id : it->second) {
        const CodeChunk& c = chunks[static_cast<size_t>(id)];
        if (c.symbol.empty() || ++n > 12) continue;
        o += "  - " + c.symbol + "\n";
      }
      break;
    }
    return utf8_truncate(o, budget);
  }
};

// ------------------------------------------------------------- public shell
CodebaseIndex::CodebaseIndex() : p_(new Impl) {
  // A usable default: no download, no configuration, deterministic.
  p_->emb = std::make_shared<HashEmbedder>(256);
  p_->dim = p_->emb->dim();
}
CodebaseIndex::~CodebaseIndex() = default;

void CodebaseIndex::set_embedder(EmbedderPtr e) {
  std::lock_guard<std::mutex> lk(m_);
  p_->emb = e ? e : std::make_shared<HashEmbedder>(256);
  const int nd = p_->emb->dim();
  if (nd != p_->dim) {
    // The stored vectors live in the old embedder's space; they cannot be
    // compared with the new one's, so re-embed what we have.
    p_->dim = nd;
    p_->dense.clear();
    std::vector<int32_t> todo(p_->chunks.size());
    for (size_t i = 0; i < todo.size(); ++i) todo[i] = static_cast<int32_t>(i);
    p_->embed_rows(todo);
  }
  p_->st.embedder = p_->emb->name();
}

std::string CodebaseIndex::embedder_name() const {
  std::lock_guard<std::mutex> lk(m_);
  return p_->emb->name();
}

bool CodebaseIndex::scan(
    const ScanOptions& opt,
    const std::function<void(int64_t, int64_t, const std::string&)>& progress,
    std::atomic<bool>* cancel, std::string* err) {
  std::lock_guard<std::mutex> lk(m_);
  std::error_code ec;
  if (opt.root.empty() || !fs::is_directory(opt.root, ec)) {
    if (err) *err = "not a directory: " + opt.root;
    return false;
  }
  p_->reset();
  p_->opt = opt;
  std::vector<FileEntry> entries;
  std::vector<IgnoreFile> stack;
  walk_dir(fs::path(opt.root), std::string(), opt, &stack, &entries, &p_->st, cancel);
  if (cancel && cancel->load()) {
    if (err) *err = "cancelled";
    return false;
  }
  std::sort(entries.begin(), entries.end(),
            [](const FileEntry& a, const FileEntry& b) { return a.rel < b.rel; });
  return p_->ingest(entries, false, progress, cancel, err);
}

bool CodebaseIndex::refresh(
    const std::function<void(int64_t, int64_t, const std::string&)>& progress,
    std::atomic<bool>* cancel, std::string* err) {
  std::lock_guard<std::mutex> lk(m_);
  std::error_code ec;
  if (p_->opt.root.empty() || !fs::is_directory(p_->opt.root, ec)) {
    if (err) *err = "index has no scanned root";
    return false;
  }
  IndexStats keep = p_->st;
  p_->st.skipped_ignored = 0;
  p_->st.skipped_binary = 0;
  p_->st.skipped_large = 0;
  std::vector<FileEntry> entries;
  std::vector<IgnoreFile> stack;
  walk_dir(fs::path(p_->opt.root), std::string(), p_->opt, &stack, &entries, &p_->st,
           cancel);
  if (cancel && cancel->load()) {
    p_->st = keep;
    if (err) *err = "cancelled";
    return false;
  }
  std::sort(entries.begin(), entries.end(),
            [](const FileEntry& a, const FileEntry& b) { return a.rel < b.rel; });
  return p_->ingest(entries, true, progress, cancel, err);
}

std::vector<SearchHit> CodebaseIndex::search(const std::string& query, size_t k,
                                             SearchMode mode) const {
  std::lock_guard<std::mutex> lk(m_);
  return p_->search(query, k, mode);
}

const CodeChunk* CodebaseIndex::chunk(int32_t id) const {
  std::lock_guard<std::mutex> lk(m_);
  if (id < 0 || static_cast<size_t>(id) >= p_->chunks.size()) return nullptr;
  return &p_->chunks[static_cast<size_t>(id)];
}

std::vector<const CodeChunk*> CodebaseIndex::chunks_of(const std::string& path) const {
  std::lock_guard<std::mutex> lk(m_);
  std::vector<const CodeChunk*> out;
  const auto it = p_->by_path.find(path);
  if (it == p_->by_path.end()) return out;
  out.reserve(it->second.size());
  for (const int32_t id : it->second) out.push_back(&p_->chunks[static_cast<size_t>(id)]);
  return out;
}

std::string CodebaseIndex::context_block(const std::string& query, size_t budget_chars,
                                         size_t max_chunks) const {
  std::lock_guard<std::mutex> lk(m_);
  return p_->context_block(query, budget_chars, max_chunks);
}

std::vector<SearchHit> CodebaseIndex::find_symbol(const std::string& name,
                                                  size_t k) const {
  std::lock_guard<std::mutex> lk(m_);
  return p_->find_symbol(name, k);
}

std::string CodebaseIndex::overview(size_t budget_chars) const {
  std::lock_guard<std::mutex> lk(m_);
  return p_->overview(budget_chars);
}

IndexStats CodebaseIndex::stats() const {
  std::lock_guard<std::mutex> lk(m_);
  p_->finish_stats();
  return p_->st;
}

bool CodebaseIndex::empty() const {
  std::lock_guard<std::mutex> lk(m_);
  return p_->chunks.empty();
}

bool CodebaseIndex::needs_ann() const {
  std::lock_guard<std::mutex> lk(m_);
  return p_->chunks.size() > 1000000;
}

const ScanOptions& CodebaseIndex::options() const { return p_->opt; }

void CodebaseIndex::clear() {
  std::lock_guard<std::mutex> lk(m_);
  p_->reset();
  p_->dim = p_->emb->dim();
}

// ------------------------------------------------------------- save / load
bool CodebaseIndex::save(const std::string& path, std::string* err) const {
  std::lock_guard<std::mutex> lk(m_);
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    if (err) *err = "cannot write " + path;
    return false;
  }
  f.write(kIndexMagic, sizeof(kIndexMagic));
  wpod(f, kIndexVersion);

  const ScanOptions& o = p_->opt;
  wstr(f, o.root);
  wpod(f, o.max_file_bytes);
  wpod(f, static_cast<int32_t>(o.max_chunk_lines));
  wpod(f, static_cast<int32_t>(o.min_chunk_lines));
  wpod(f, static_cast<uint8_t>(o.use_gitignore));
  wpod(f, static_cast<uint8_t>(o.skip_hidden));
  wpod(f, static_cast<uint8_t>(o.skip_binary));
  wpod(f, static_cast<uint8_t>(0));  // padding, keeps the record 4-byte aligned
  wvec_str(f, o.extra_ignores);
  wvec_str(f, o.only_exts);
  wpod(f, o.max_total_chunks);

  wstr(f, p_->emb->name());
  wpod(f, static_cast<int32_t>(p_->dim));

  wpod(f, static_cast<uint64_t>(p_->chunks.size()));
  for (const CodeChunk& c : p_->chunks) {
    wpod(f, c.id);
    wstr(f, c.path);
    wstr(f, c.lang);
    wstr(f, c.symbol);
    wstr(f, c.kind);
    wpod(f, c.start_line);
    wpod(f, c.end_line);
    wstr(f, c.text);
    wpod(f, c.length);
  }
  wpod(f, static_cast<uint64_t>(p_->dense.size()));
  if (!p_->dense.empty())
    f.write(reinterpret_cast<const char*>(p_->dense.data()),
            static_cast<std::streamsize>(p_->dense.size() * sizeof(float)));

  wpod(f, static_cast<uint64_t>(p_->files.size()));
  for (const Impl::FileRec& r : p_->files) {
    wstr(f, r.path);
    wpod(f, r.mtime);
    wpod(f, r.size);
    wpod(f, r.lines);
  }
  const IndexStats& s = p_->st;
  wpod(f, s.skipped_ignored);
  wpod(f, s.skipped_binary);
  wpod(f, s.skipped_large);
  f.flush();
  if (!f) {
    if (err) *err = "write failed for " + path;
    return false;
  }
  return true;
}

bool CodebaseIndex::load(const std::string& path, std::string* err) {
  std::lock_guard<std::mutex> lk(m_);
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (err) *err = "cannot read " + path;
    return false;
  }
  char magic[sizeof(kIndexMagic)] = {0};
  f.read(magic, sizeof(magic));
  if (!f || std::memcmp(magic, kIndexMagic, sizeof(magic)) != 0) {
    if (err) *err = path + ": not a codebase index (bad magic)";
    return false;
  }
  uint32_t ver = 0;
  if (!rpod(f, &ver) || ver != kIndexVersion) {
    if (err)
      *err = path + ": index version " + std::to_string(ver) + ", expected " +
             std::to_string(kIndexVersion);
    return false;
  }
  ScanOptions o;
  uint8_t b_git = 0, b_hidden = 0, b_bin = 0, pad = 0;
  int32_t maxl = 0, minl = 0;
  if (!rstr(f, &o.root) || !rpod(f, &o.max_file_bytes) || !rpod(f, &maxl) ||
      !rpod(f, &minl) || !rpod(f, &b_git) || !rpod(f, &b_hidden) || !rpod(f, &b_bin) ||
      !rpod(f, &pad) || !rvec_str(f, &o.extra_ignores) || !rvec_str(f, &o.only_exts) ||
      !rpod(f, &o.max_total_chunks)) {
    if (err) *err = path + ": truncated header";
    return false;
  }
  o.max_chunk_lines = maxl;
  o.min_chunk_lines = minl;
  o.use_gitignore = b_git != 0;
  o.skip_hidden = b_hidden != 0;
  o.skip_binary = b_bin != 0;

  std::string emb_name;
  int32_t emb_dim = 0;
  if (!rstr(f, &emb_name) || !rpod(f, &emb_dim)) {
    if (err) *err = path + ": truncated embedder record";
    return false;
  }
  // The one check this format exists for: two different embedders produce two
  // incomparable vector spaces, and mixing them degrades silently instead of
  // failing.  Refuse.
  if (emb_name != p_->emb->name() || emb_dim != p_->emb->dim()) {
    if (err)
      *err = path + ": index was built with embedder '" + emb_name + "' (dim " +
             std::to_string(emb_dim) + ") but the current embedder is '" +
             p_->emb->name() + "' (dim " + std::to_string(p_->emb->dim()) +
             ") - rescan or select the matching embedder";
    return false;
  }

  uint64_t n = 0;
  if (!rpod(f, &n) || n > (1ull << 32)) {
    if (err) *err = path + ": bad chunk count";
    return false;
  }
  std::vector<CodeChunk> chunks(static_cast<size_t>(n));
  for (uint64_t i = 0; i < n; ++i) {
    CodeChunk& c = chunks[static_cast<size_t>(i)];
    if (!rpod(f, &c.id) || !rstr(f, &c.path) || !rstr(f, &c.lang) ||
        !rstr(f, &c.symbol) || !rstr(f, &c.kind) || !rpod(f, &c.start_line) ||
        !rpod(f, &c.end_line) || !rstr(f, &c.text) || !rpod(f, &c.length)) {
      if (err) *err = path + ": truncated chunk table";
      return false;
    }
  }
  uint64_t dn = 0;
  if (!rpod(f, &dn) || dn > (1ull << 34)) {
    if (err) *err = path + ": bad dense matrix size";
    return false;
  }
  std::vector<float> dense(static_cast<size_t>(dn));
  if (dn) {
    f.read(reinterpret_cast<char*>(dense.data()),
           static_cast<std::streamsize>(dn * sizeof(float)));
    if (!f) {
      if (err) *err = path + ": truncated dense matrix";
      return false;
    }
  }
  if (dn != static_cast<uint64_t>(chunks.size()) * static_cast<uint64_t>(emb_dim)) {
    if (err) *err = path + ": dense matrix does not match the chunk table";
    return false;
  }
  uint64_t fn = 0;
  if (!rpod(f, &fn) || fn > (1ull << 30)) {
    if (err) *err = path + ": bad file count";
    return false;
  }
  std::vector<Impl::FileRec> frecs(static_cast<size_t>(fn));
  for (uint64_t i = 0; i < fn; ++i) {
    Impl::FileRec& r = frecs[static_cast<size_t>(i)];
    if (!rstr(f, &r.path) || !rpod(f, &r.mtime) || !rpod(f, &r.size) ||
        !rpod(f, &r.lines)) {
      if (err) *err = path + ": truncated file table";
      return false;
    }
  }
  int64_t sk_ign = 0, sk_bin = 0, sk_large = 0;
  if (!rpod(f, &sk_ign) || !rpod(f, &sk_bin) || !rpod(f, &sk_large)) {
    if (err) *err = path + ": truncated statistics";
    return false;
  }

  p_->reset();
  p_->opt = o;
  p_->dim = emb_dim;
  p_->chunks = std::move(chunks);
  p_->dense = std::move(dense);
  p_->files = std::move(frecs);
  p_->st.skipped_ignored = sk_ign;
  p_->st.skipped_binary = sk_bin;
  p_->st.skipped_large = sk_large;
  p_->build_by_path();
  p_->build_lexical();  // cheaper to rebuild than to store, and never stale
  p_->finish_stats();
  return true;
}

}  // namespace slm
