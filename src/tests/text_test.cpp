// SPDX-License-Identifier: Apache-2.0
//
// Tests for the multilingual text layer: UTF-8, Persian normalisation,
// pre-tokenisation, language identification, Python structure checks and
// tokenizer fertility / round-tripping.
#include <cstdio>
#include <string>
#include <vector>

#include "core/text.h"
#include "tokenizer.h"

using namespace slm;

namespace {

int g_fail = 0, g_total = 0;

void check(bool ok, const std::string& name, const std::string& info = "") {
  ++g_total;
  if (!ok) ++g_fail;
  std::printf("[%s] %-46s %s\n", ok ? " OK " : "FAIL", name.c_str(), info.c_str());
}

void eq(const std::string& got, const std::string& want, const std::string& name) {
  check(got == want, name, got == want ? "" : ("got \"" + got + "\" want \"" + want + "\""));
}

const char* kFa =
    "سلام دنیا! این یک آزمایش است. کتاب‌ها را می‌خوانم و می‌نویسم، بعد از آن "
    "می‌روم. برنامه‌نویسی با پایتون خیلی جذاب است؛ آیا موافقی؟";
const char* kEn =
    "The model reads a sequence of tokens and predicts the next one. Attention "
    "lets every position look back at the earlier positions of the sequence.";
const char* kPy =
    "def fib(n):\n"
    "    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
    "    a, b = 0, 1\n"
    "    for _ in range(n):\n"
    "        a, b = b, a + b\n"
    "    return a\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    print(fib(10))\n";

void test_utf8() {
  std::string s = kFa;
  std::string round;
  size_t i = 0;
  size_t cps = 0;
  while (i < s.size()) {
    utf8_append(utf8_next(s, &i), &round);
    ++cps;
  }
  check(round == s, "utf8 round trip (persian)", std::to_string(cps) + " code points");

  std::string bad = "ok\xC3(\xFF end";  // truncated + invalid bytes
  std::string round2;
  i = 0;
  while (i < bad.size()) utf8_append(utf8_next(bad, &i), &round2);
  check(round2 == bad, "utf8 round trip (invalid bytes preserved)");

  std::string emoji = "code 🚀 fast";
  i = 0;
  size_t n = 0;
  while (i < emoji.size()) {
    utf8_next(emoji, &i);
    ++n;
  }
  check(n == 11, "utf8 length with 4-byte code point", std::to_string(n));
}

void test_normalize() {
  // Arabic Yeh/Kaf -> Persian, harakat and tatweel removed, digits to ASCII.
  eq(normalize_persian("كتاب يك"), "کتاب یک", "normalise arabic kaf/yeh");
  eq(normalize_persian("مُحَمَّد"), "محمد", "strip harakat");
  eq(normalize_persian("سلـــام"), "سلام", "strip tatweel");
  eq(normalize_persian("۱۲۳ و ٤٥٦"), "123 و 456", "persian + arabic digits to ascii");
  eq(normalize_persian("می ‌روم"), "می‌روم", "space around ZWNJ absorbed");
  eq(normalize_persian("می‌‌روم"), "می‌روم", "duplicate ZWNJ collapsed");
  eq(normalize_persian("مدرسة"), "مدرسه", "teh marbuta -> heh");
  // U+FEE3 (medial meem) + U+FE8E (final alef) must fold back to base letters.
  eq(normalize_persian("\xEF\xBB\xA3\xEF\xBA\x8E"), "ما", "presentation forms decomposed");
  eq(normalize_persian("a\xE2\x80\x8F" "b"), "ab", "bidi control dropped");
  // Latin and code must survive untouched.
  eq(normalize_persian(kPy), kPy, "python source unchanged by normalisation");
  eq(normalize_persian("x = 1   \n    y = 2\n"), "x = 1\n    y = 2\n",
     "trailing space stripped, indentation kept");
}

void test_pretokenize() {
  auto joined = [](const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) s += "|";
      s += v[i];
    }
    return s;
  };
  const std::vector<std::string> fa = Tokenizer::pretokenize(normalize_persian("می‌روم، سلام!"));
  eq(joined(fa), "می‌روم|،| سلام|!", "persian: ZWNJ inside word, punctuation split");

  const std::vector<std::string> num = Tokenizer::pretokenize("سال 1403 بود");
  eq(joined(num), "سال| 1|4|0|3| بود", "digits are emitted one by one");

  const std::vector<std::string> py = Tokenizer::pretokenize("def f(x):\n    return x");
  eq(joined(py), "def| f|(|x|)|:|\n    |return| x", "python: indentation is one piece");

  const std::vector<std::string> mix = Tokenizer::pretokenize("خط «hello» است");
  eq(joined(mix), "خط| «|hello|»| است", "guillemets are punctuation");
}

void test_language() {
  check(detect_language(kFa) == Lang::kPersian, "detect persian");
  check(detect_language(kEn) == Lang::kEnglish, "detect english");
  check(detect_language(kPy) == Lang::kPython, "detect python");
  check(detect_language("") == Lang::kUnknown, "detect empty");
  check(detect_language("سلام def x(): return 1") == Lang::kPersian,
        "persian wins over a code fragment");

  const double clean = code_switch_ratio("این یک جمله فارسی است", Lang::kPersian);
  const double dirty = code_switch_ratio("این یک sentence فارسی است with words",
                                         Lang::kPersian);
  check(clean < 0.02 && dirty > 0.25, "code switch ratio",
        "clean " + std::to_string(clean) + " dirty " + std::to_string(dirty));
}

void test_python_check() {
  CodeCheck ok = check_python(kPy);
  check(ok.ok && ok.def_count == 1, "valid python accepted",
        ok.reason + " defs=" + std::to_string(ok.def_count));
  check(!check_python("def f(:\n    return 1\n").ok, "unbalanced bracket rejected");
  check(!check_python("x = 'abc\ny = 2\n").ok, "unterminated string rejected");
  check(!check_python("def f():\n\treturn 1\n    # mixed\n").ok,
        "mixed tab/space indentation rejected");
  check(!check_python("print(1)").ok, "single line rejected");
  CodeCheck doc = check_python(
      "# helper\ndef add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n");
  check(doc.ok && doc.comment_lines == 1, "docstring + comment accepted",
        "comment_ratio " + std::to_string(doc.comment_ratio));
}

void test_tokenizer() {
  // A trilingual corpus, repeated so BPE has something to merge.
  std::string corpus;
  for (int i = 0; i < 60; ++i) {
    corpus += kFa;
    corpus += "\n";
    corpus += kEn;
    corpus += "\n";
    corpus += kPy;
    corpus += "\n";
  }
  Tokenizer tok;
  tok.set_normalize(true);
  tok.train(corpus, 1400, 2, nullptr);

  // Round trip: encode(decode) must reproduce the *normalised* text exactly.
  for (const char* sample : {kFa, kEn, kPy}) {
    const std::string norm = tok.preprocess(sample);
    const std::string back = tok.decode(tok.encode(sample));
    check(back == norm, std::string("round trip ") + lang_code(detect_language(sample)),
          back == norm ? "" : "MISMATCH");
  }

  const Tokenizer::FertilityReport fa = tok.fertility(kFa);
  const Tokenizer::FertilityReport en = tok.fertility(kEn);
  const Tokenizer::FertilityReport py = tok.fertility(kPy);
  char buf[256];
  std::snprintf(buf, sizeof(buf), "fa %.2f chars/tok, en %.2f, py %.2f",
                fa.chars_per_token(), en.chars_per_token(), py.chars_per_token());
  // Before the code-point aware pre-tokenizer Persian collapsed to ~1 char per
  // token (every letter split); anything above 2 means the vocabulary really
  // covers the language.
  check(fa.chars_per_token() > 2.0, "persian fertility is healthy", buf);
  check(fa.single_byte_share() < 0.5, "persian is not shattered into raw bytes",
        "single byte share " + std::to_string(fa.single_byte_share()));
  check(en.chars_per_token() > 2.5, "english fertility is healthy");
  check(py.chars_per_token() > 2.0, "python fertility is healthy");

  // Control tokens survive round-tripping in a chat template.
  const std::vector<int32_t> ids = tok.encode("<|user|>سلام<|assistant|>درود<|endoftext|>");
  check(ids.size() > 4 && ids[0] == Tokenizer::kUser, "chat template control tokens");
  bool has_assistant = false;
  for (int32_t id : ids) has_assistant |= (id == Tokenizer::kAssistant);
  check(has_assistant, "assistant token found");

  // save / load keeps the normalisation flag
  const std::string path = "/tmp/slm_text_test.slmtok";
  check(tok.save(path), "tokenizer save");
  Tokenizer tok2;
  check(tok2.load(path) && tok2.normalize() == tok.normalize() &&
            tok2.vocab_size() == tok.vocab_size(),
        "tokenizer load keeps normalisation flag");
  check(tok2.encode(kFa) == tok.encode(kFa), "reloaded tokenizer encodes identically");
}

}  // namespace

int main() {
  test_utf8();
  test_normalize();
  test_pretokenize();
  test_language();
  test_python_check();
  test_tokenizer();
  std::printf("\n%d/%d checks passed\n", g_total - g_fail, g_total);
  return g_fail == 0 ? 0 : 1;
}
