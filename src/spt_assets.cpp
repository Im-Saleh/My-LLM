// SPDX-License-Identifier: Apache-2.0
#include "spt_assets.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "core/serialize.h"
#include "model.h"
#include "qmodel.h"
#include "tokenizer.h"

namespace slm {
namespace fs = std::filesystem;
namespace {

std::string env(const char* k) {
  const char* v = std::getenv(k);
  return v ? std::string(v) : std::string();
}

std::string exe_dir() {
  std::error_code ec;
  const fs::path p = fs::read_symlink("/proc/self/exe", ec);
  if (ec) return {};
  return p.parent_path().string();
}

bool file_ok(const std::string& p) {
  std::error_code ec;
  return !p.empty() && fs::is_regular_file(p, ec) && fs::file_size(p, ec) > 0;
}

// Accepts a directory when it holds a tokenizer and at least one set of weights.
bool probe(const std::string& dir, SptAssets* out) {
  const std::string tok = dir + "/spt.slmtok";
  const std::string ckpt = dir + "/spt.slm";
  const std::string q4 = dir + "/spt-q4.slmq";
  if (!file_ok(tok)) return false;
  if (!file_ok(ckpt) && !file_ok(q4)) return false;
  out->tokenizer = tok;
  out->ckpt = file_ok(ckpt) ? ckpt : std::string();
  out->quant = file_ok(q4) ? q4 : std::string();
  out->dir = dir;
  return true;
}

// Older checkouts shipped several demo models; accept them so an existing clone
// keeps working, but name what was found so the rename is obvious.
bool probe_legacy(const std::string& dir, SptAssets* out) {
  static const char* kPairs[][3] = {
      {"demo-fa.slmtok", "demo-fa.slm", "demo-fa-q4.slmq"},
      {"demo-fa.slmtok", "demo-fa-base.slm", ""},
      {"demo-tri.slmtok", "demo-tri.slm", ""},
      {"tok.slmtok", "base.slm", ""},
      {"tok.slmtok", "final.slm", ""},
  };
  for (const auto& p : kPairs) {
    const std::string tok = dir + "/" + p[0];
    const std::string ckpt = dir + "/" + p[1];
    if (!file_ok(tok) || !file_ok(ckpt)) continue;
    out->tokenizer = tok;
    out->ckpt = ckpt;
    const std::string q = p[2][0] ? dir + "/" + p[2] : std::string();
    out->quant = file_ok(q) ? q : std::string();
    out->dir = dir;
    out->note = "found legacy names (" + std::string(p[1]) +
                "); rename to spt.slm/spt.slmtok to make it the canonical model";
    return true;
  }
  return false;
}

}  // namespace

std::string spt_user_dir() {
  std::string base = env("XDG_DATA_HOME");
  if (base.empty()) {
    const std::string home = env("HOME");
    base = home.empty() ? std::string("/tmp") : home + "/.local/share";
  }
  return base + "/slm";
}

std::vector<std::string> spt_search_dirs() {
  std::vector<std::string> d;
  const std::string sh = env("SLM_HOME");
  if (!sh.empty()) {
    d.push_back(sh);
    d.push_back(sh + "/models");
  }
  d.push_back("models");
  d.push_back("run");
  d.push_back(spt_user_dir());
  d.push_back("/usr/share/slm/models");
  d.push_back("/usr/local/share/slm/models");
  const std::string ed = exe_dir();
  if (!ed.empty()) {
    d.push_back(ed);
    d.push_back(ed + "/models");
    d.push_back(ed + "/../models");
    d.push_back(ed + "/../share/slm/models");
  }
  return d;
}

bool resolve_spt(SptAssets* out, bool allow_bootstrap, std::string* err,
                 const std::function<void(const std::string&)>& log) {
  SptAssets a;
  a.workdir = spt_user_dir() + "/runs";
  for (const std::string& dir : spt_search_dirs()) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) continue;
    if (probe(dir, &a)) {
      a.note = "model: " + a.dir + "/spt.*";
      *out = a;
      if (log) log(a.note);
      return true;
    }
  }
  for (const std::string& dir : spt_search_dirs()) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) continue;
    if (probe_legacy(dir, &a)) {
      *out = a;
      if (log) log(a.note);
      return true;
    }
  }
  if (!allow_bootstrap) {
    if (err)
      *err =
          "no SPT model found. Looked in: " + [&] {
            std::string s;
            for (const std::string& d : spt_search_dirs()) s += d + " ";
            return s;
          }() + "\nRun `slm up` to create a starter model, or put spt.slm and "
                "spt.slmtok in ./models";
    return false;
  }
  const std::string dir = spt_user_dir();
  if (log)
    log("no model found - creating a starter model in " + dir +
        " (tiny; point the training panel at a real dataset to improve it)");
  if (!bootstrap_spt(dir, std::string(), out, err, log)) return false;
  out->workdir = a.workdir;
  return true;
}

bool bootstrap_spt(const std::string& dir, const std::string& corpus_path,
                   SptAssets* out, std::string* err,
                   const std::function<void(const std::string&)>& log) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    if (err) *err = "cannot create " + dir + ": " + ec.message();
    return false;
  }
  // The corpus: a real file when given, otherwise whatever text the working tree
  // has, otherwise the built-in seed.
  std::string text;
  std::string source = "built-in seed corpus";
  auto slurp = [&](const std::string& p, size_t cap) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::string buf(cap, '\0');
    f.read(&buf[0], static_cast<std::streamsize>(cap));
    buf.resize(static_cast<size_t>(f.gcount()));
    if (buf.size() < 512) return false;
    text = buf;
    source = p;
    return true;
  };
  if (!corpus_path.empty() && !slurp(corpus_path, 8u << 20)) {
    if (err) *err = "cannot read corpus " + corpus_path;
    return false;
  }
  if (text.empty())
    for (const char* p : {"data/mixed.txt", "data/fa.txt", "data/sample_corpus.txt",
                          "/usr/share/slm/scripts/sample_corpus.txt"})
      if (slurp(p, 4u << 20)) break;
  if (text.empty()) text = spt_seed_corpus();
  if (log) log("bootstrap corpus: " + source + " (" + std::to_string(text.size()) + " bytes)");

  // Vocabulary sized to the corpus: a 8k vocab learned from 6 kB of text is
  // mostly single bytes, which would make the starter model worse, not better.
  int32_t vocab = 2048;
  if (text.size() > 1u << 20) vocab = 8192;
  else if (text.size() > 200000) vocab = 4096;
  else if (text.size() < 20000) vocab = 1024;

  Tokenizer tok;
  tok.set_normalize(true);
  tok.train(text, vocab, 2, nullptr);
  const std::string tok_path = dir + "/spt.slmtok";
  if (!tok.save(tok_path)) {
    if (err) *err = "cannot write " + tok_path;
    return false;
  }
  if (log)
    log("tokenizer: " + std::to_string(tok.vocab_size()) + " tokens -> " + tok_path);

  // A small but real architecture: the same modern stack the big configs use, so
  // that continuing to train it is a straight line rather than a migration.
  GPTConfig cfg;
  cfg.vocab_size = tok.vocab_size();
  cfg.n_layer = 6;
  cfg.n_head = 6;
  cfg.n_kv_head = 2;
  cfg.n_embd = 192;
  cfg.block_size = 256;
  cfg.tie_weights = true;
  cfg.validate();
  GPT model(cfg);
  model.init_weights(1234);

  ParamStore ps = *model.snapshot();
  CheckpointMeta meta;
  meta.step = 0;
  meta.tokens_seen = 0;
  cfg.write_to(meta.extra);
  meta.extra.set("tokenizer", tok_path);
  meta.extra.set("origin", "bootstrap starter model (untrained)");
  const std::string ckpt = dir + "/spt.slm";
  if (!save_checkpoint(ckpt, ps, meta, Dtype::F32)) {
    if (err) *err = "cannot write " + ckpt;
    return false;
  }
  if (log)
    log("model: " + cfg.describe() + " -> " + ckpt);

  out->tokenizer = tok_path;
  out->ckpt = ckpt;
  out->quant.clear();
  out->dir = dir;
  out->workdir = dir + "/runs";
  out->bootstrapped = true;
  out->note =
      "starter model created (untrained). Open the training panel, point it at a "
      "dataset, and press train - or run: slm pretrain --data <file>";
  return true;
}

const char* spt_seed_corpus() {
  // Deliberately small and trilingual, matching the project's three target
  // domains, and written in the <|user|>/<|assistant|> form so the chat control
  // tokens exist in the vocabulary from the first step.
  return
      "<|user|>سلام، حالت چطور است؟<|assistant|>سلام! خوبم، ممنون. چطور می‌توانم "
      "کمک کنم؟<|endoftext|>\n"
      "<|user|>پایتخت ایران کجاست؟<|assistant|>پایتخت ایران شهر تهران است.<|endoftext|>\n"
      "<|user|>زبان فارسی از راست به چپ نوشته می‌شود و نیم‌فاصله در آن اهمیت "
      "دارد.<|assistant|>درست است. نیم‌فاصله باعث می‌شود واژه‌هایی مثل «می‌روم» و "
      "«کتاب‌ها» درست نمایش داده شوند.<|endoftext|>\n"
      "<|user|>یک تابع پایتون بنویس که مجموع یک لیست را حساب کند."
      "<|assistant|>\n"
      "def total(values):\n"
      "    \"\"\"Return the sum of a list of numbers.\"\"\"\n"
      "    result = 0\n"
      "    for value in values:\n"
      "        result += value\n"
      "    return result\n"
      "<|endoftext|>\n"
      "<|user|>Write a Python function that reverses a string."
      "<|assistant|>\n"
      "def reverse(text):\n"
      "    \"\"\"Return text reversed.\"\"\"\n"
      "    return text[::-1]\n"
      "<|endoftext|>\n"
      "<|user|>What is a transformer?<|assistant|>A transformer is a neural network "
      "architecture built from attention layers. Each layer lets every position "
      "attend to every earlier position, which is how the model uses context."
      "<|endoftext|>\n"
      "<|user|>How do I read a file in Python?<|assistant|>\n"
      "with open(\"notes.txt\", \"r\", encoding=\"utf-8\") as handle:\n"
      "    text = handle.read()\n"
      "<|endoftext|>\n"
      "<|user|>مدل زبانی کوچک چه فایده‌ای دارد؟<|assistant|>مدل کوچک روی سخت‌افزار "
      "معمولی اجرا می‌شود، سریع است و می‌توان آن را روی داده‌ی خود آموزش داد. "
      "کیفیتش از مدل بزرگ کمتر است ولی هزینه‌اش هم بسیار کمتر است.<|endoftext|>\n"
      "<|user|>یک کلاس پایتون برای صف بنویس.<|assistant|>\n"
      "class Queue:\n"
      "    \"\"\"A first in, first out queue.\"\"\"\n"
      "\n"
      "    def __init__(self):\n"
      "        self.items = []\n"
      "\n"
      "    def push(self, item):\n"
      "        self.items.append(item)\n"
      "\n"
      "    def pop(self):\n"
      "        if not self.items:\n"
      "            return None\n"
      "        return self.items.pop(0)\n"
      "<|endoftext|>\n"
      "<|user|>Explain gradient descent in one paragraph.<|assistant|>Gradient "
      "descent improves a model by measuring how the loss changes with each "
      "parameter and then stepping every parameter a little way downhill. The step "
      "size is the learning rate: too large and the loss oscillates, too small and "
      "training takes forever.<|endoftext|>\n"
      "<|user|>چطور در پایتون یک دیکشنری را پیمایش کنم؟<|assistant|>\n"
      "for key, value in mapping.items():\n"
      "    print(key, value)\n"
      "<|endoftext|>\n"
      "<|user|>What does quantisation do to a model?<|assistant|>It stores the "
      "weights with fewer bits, so the model is smaller and decoding is faster "
      "because less memory has to be read for every token. Four bit weights are "
      "about eight times smaller than float32 with a small quality cost."
      "<|endoftext|>\n";
}

}  // namespace slm
