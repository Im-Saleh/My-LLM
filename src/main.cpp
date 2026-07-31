// SPDX-License-Identifier: Apache-2.0
//
// slm - command line entry point.
//
//   slm info                    environment / memory budget
//   slm tokenizer               train a BPE tokenizer
//   slm pretrain                base model training
//   slm chat                    interactive completion (terminal)
//   slm eval                    hold-out loss / perplexity
//   slm quantize                re-encode a checkpoint (f32/f16/q8)
//   slm pack                    checkpoint -> memory mappable int4/int8 .slmq
//   slm qrun / qbench / qeval   run, time and score a .slmq model
//   slm bench                   forward+backward throughput
//   slm live                    the full self-training system, terminal dashboard
//   slm dashboard               the full self-training system, ImGui dashboard
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "app.h"
#include "core/config.h"
#include "core/dataset.h"
#include "core/optim.h"
#include "core/serialize.h"
#include "core/tensor.h"
#include "core/text.h"
#include "memory.h"
#include "model.h"
#include "qmodel.h"
#include "agent/runtime.h"
#include "spt_assets.h"
#include "telemetry.h"
#include "tokenizer.h"

using namespace slm;

namespace {

// ------------------------------------------------------------- tiny arg parser
class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 0; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--set" && i + 1 < argc) {
        sets_.push_back(argv[++i]);
        continue;
      }
      if (a.rfind("--", 0) == 0) {
        const size_t eq = a.find('=');
        if (eq != std::string::npos) {
          kv_[a.substr(2, eq - 2)] = a.substr(eq + 1);
        } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
          kv_[a.substr(2)] = argv[++i];
        } else {
          kv_[a.substr(2)] = "1";
        }
      } else {
        pos_.push_back(a);
      }
    }
  }
  bool has(const std::string& k) const { return kv_.count(k) > 0; }
  std::string str(const std::string& k, const std::string& d = "") const {
    auto it = kv_.find(k);
    return it == kv_.end() ? d : it->second;
  }
  double num(const std::string& k, double d) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return d;
    try {
      return std::stod(it->second);
    } catch (...) {
      return d;
    }
  }
  int64_t integer(const std::string& k, int64_t d) const {
    return static_cast<int64_t>(num(k, static_cast<double>(d)));
  }
  bool flag(const std::string& k, bool d = false) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return d;
    return it->second == "1" || it->second == "true" || it->second == "yes";
  }
  const std::vector<std::string>& positional() const { return pos_; }
  // Applies every "--set key=value" onto a Config, in order.
  void apply_sets(Config* c) const {
    for (const std::string& s : sets_) c->set_kv(s);
  }

 private:
  std::map<std::string, std::string> kv_;
  std::vector<std::string> sets_;
  std::vector<std::string> pos_;
};

std::string read_file(const std::string& path, bool* ok = nullptr) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (ok) *ok = false;
    return {};
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  if (ok) *ok = true;
  return ss.str();
}

double now_seconds() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
}

// Builds the corpus mixture from --mix (weighted, multi language) or --data.
bool build_mixture(const Args& a, const Tokenizer& tok, MixtureDataset* mix,
                   bool required = true) {
  const std::string spec = a.has("mix") ? a.str("mix") : a.str("data");
  if (spec.empty()) {
    if (required) std::fprintf(stderr, "one of --mix or --data is required\n");
    return false;
  }
  std::string err;
  if (!mix->add_spec(spec, tok, &err)) {
    std::fprintf(stderr, "corpus: %s\n", err.c_str());
    return false;
  }
  return true;
}

void print_usage() {
  std::puts(
      "slm - a small language model with a hybrid self-training pipeline\n"
      "      (Persian + English + Python)\n"
      "\n"
      "usage: slm <command> [options]\n"
      "\n"
      "  up                                               <- START HERE\n"
      "                finds (or creates) the one SPT model and opens the dashboard.\n"
      "                no arguments needed.  [--terminal] [--gguf F.gguf] [--where]\n"
      "  agent         [--ask S] [--mode fast|strong|debate|self]\n"
      "                two models, weighted debate, tools and codebase retrieval.\n"
      "                [--gguf F.gguf] [--fast-mult N] [--strong-mult N] [--voices N]\n"
      "                [--index [DIR]] [--workspace DIR] [--transcript] [--yes]\n"
      "  info          [--config F]                       environment + memory budget\n"
      "  tokenizer     --out F [--vocab 4096]             train a byte level BPE\n"
      "                (--input F | --mix fa=F:0.4,en=F:0.3,py=F:0.3) [--no-normalize]\n"
      "  pretrain      --tokenizer F --out F              base training\n"
      "                (--data F | --mix fa=F:0.4,en=F:0.3,py=F:0.3)\n"
      "                [--config F] [--steps N] [--batch N] [--ctx N] [--lr X]\n"
      "                [--accum N] [--eval-every N] [--save-every N] [--dtype f16]\n"
      "                [--resume F] [--threads N] [--seed N]\n"
      "  chat          --ckpt F --tokenizer F [--prompt S] [--memory memory.jsonl]\n"
      "  eval          --ckpt F --tokenizer F (--data F | --mix ...) [--batch N]\n"
      "  langcheck     --ckpt F --tokenizer F --mix ...   per language quality +\n"
      "                interference (code switching) + python validity\n"
      "  tokenize      --tokenizer F --in fa=F,py=F [--out-dir data/tokens]\n"
      "                parallel BPE -> memory mappable uint16 token files\n"
      "  memory        [--file memory.jsonl] add|list|search|context|forget\n"
      "  plan          --params 7e12 [--experts N --topk N --ctx N --ram 16 --vram 2]\n"
      "                memory / compute planner for very large configurations\n"
      "  quantize      --in F --out F [--dtype q8|f16|f32]\n"
      "  pack          --in F.slm --out F.slmq [--bits 4|8] [--embed q8]\n"
      "                --synth [--layers N --dim N --heads N --kv-heads N\n"
      "                         --vocab N --ctx N]   build a big model directly\n"
      "  qrun          --model F.slmq --tokenizer F [--prompt S] [--max-new N]\n"
      "                mmap'd int4/int8 generation (no float weights)\n"
      "  qbench        --model F.slmq [--prompt-len N] [--gen N]\n"
      "                [--prefetch] [--cold]                  throughput + memory\n"
      "  qeval         --model F.slmq --tokenizer F --data F [--ckpt F.slm]\n"
      "                bits/char of the quantised model vs the f32 original\n"
      "  bench         [--config F] [--batch N] [--ctx N] [--iters N]\n"
      "  live          --ckpt F --tokenizer F --data F [--config F] [--seconds N]\n"
      "  dashboard     --ckpt F --tokenizer F --data F [--config F]\n"
      "  gradcheck                                        (see slm_gradcheck)\n");
}

// --------------------------------------------------------------------- info
int cmd_info(const Args& a) {
  Config c;
  if (a.has("config")) c.load(a.str("config"));
  // Shapes given only with --set describe a *new* model, so they must not be
  // mistaken for a pre-upgrade config file (which would silently select the
  // legacy LayerNorm/learned-position/GELU stack).
  if (!a.has("config")) c.set("model.arch_version", "2");
  a.apply_sets(&c);
  GPTConfig g = GPTConfig::from_config(c);
  std::printf("backend            : %s\n", backend_name());
  std::printf("gpu                : %s\n", backend_on_gpu() ? "yes" : "no");
  std::printf("model              : %s\n", g.describe().c_str());
  const double p = static_cast<double>(g.param_count());
  std::printf("parameters         : %lld  (%.3f M)\n",
              static_cast<long long>(g.param_count()), p / 1e6);
  std::printf("\nweights\n");
  std::printf("  f32              : %s\n", human_bytes(p * 4).c_str());
  std::printf("  f16              : %s\n", human_bytes(p * 2).c_str());
  std::printf("  int8 (group 64)  : %s\n", human_bytes(p * 8.25 / 8.0).c_str());
  std::printf("  int4 (group 64)  : %s\n", human_bytes(p * 4.25 / 8.0).c_str());
  std::printf("\ndeployment (slm pack -> mmap'd .slmq, no float weights)\n");
  std::printf("  int8 file        : %s\n",
              human_bytes(static_cast<double>(
                              qpack_estimate_bytes(g, QPackOptions{QType::Q8, QType::Q8, false, ""})))
                  .c_str());
  std::printf("  int4 file        : %s\n",
              human_bytes(static_cast<double>(qpack_estimate_bytes(g, QPackOptions())))
                  .c_str());
  std::printf("\nmemory budget (training, one replica)\n");
  std::printf("  gradients f32    : %s\n", human_bytes(p * 4).c_str());
  std::printf("  AdamW state      : %s\n", human_bytes(p * 8).c_str());
  std::printf("  total (f32 train): %s\n", human_bytes(p * 16).c_str());
  const double B = static_cast<double>(a.integer("batch", 8));
  const double T = static_cast<double>(g.block_size);
  const double act_per_layer = B * T * static_cast<double>(g.n_embd) * 12.0 * 4.0;
  const double attn = B * static_cast<double>(g.n_head) * T * T * 4.0 * 3.0;
  std::printf("\nactivations at batch=%g ctx=%g\n", B, T);
  std::printf("  no checkpointing : %s\n",
              human_bytes(act_per_layer * static_cast<double>(g.n_layer) +
                          attn * static_cast<double>(g.n_layer))
                  .c_str());
  std::printf("  checkpointing    : %s\n",
              human_bytes(act_per_layer + attn +
                          B * T * static_cast<double>(g.n_embd) * 4.0 *
                              static_cast<double>(g.n_layer))
                  .c_str());
  std::printf("  logits           : %s\n",
              human_bytes(B * T * static_cast<double>(g.vocab_size) * 4.0 * 2.0).c_str());
  return 0;
}

// ---------------------------------------------------------------- tokenizer
int cmd_tokenizer(const Args& a) {
  const std::string out = a.str("out", "tokenizer.slmtok");
  const int32_t vocab = static_cast<int32_t>(a.integer("vocab", 4096));
  Tokenizer tok;
  tok.set_normalize(!a.flag("no-normalize"));

  // The mixture is sampled *by weight* before the merges are learned: this is
  // what decides how many Persian merges the vocabulary can afford.
  struct Part {
    std::string name, text;
    double weight = 0.0;
  };
  std::vector<Part> parts;
  if (a.has("mix")) {
    const std::string spec = a.str("mix");
    size_t pos = 0;
    double wsum = 0.0;
    while (pos <= spec.size()) {
      const size_t comma = spec.find(',', pos);
      std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      if (!item.empty()) {
        std::string name, path;
        const size_t eq = item.find('=');
        if (eq == std::string::npos) path = item;
        else {
          name = item.substr(0, eq);
          path = item.substr(eq + 1);
        }
        double w = 0.0;
        const size_t colon = path.rfind(':');
        if (colon != std::string::npos &&
            path.find_first_not_of("0123456789.eE-+", colon + 1) == std::string::npos) {
          w = std::stod(path.substr(colon + 1));
          path = path.substr(0, colon);
        }
        bool ok = false;
        const std::string text = read_file(path, &ok);
        if (!ok || text.empty()) {
          std::fprintf(stderr, "tokenizer: cannot read %s\n", path.c_str());
          return 1;
        }
        parts.push_back(Part{name.empty() ? path : name, text, w});
        wsum += w;
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    for (Part& p : parts)
      if (p.weight <= 0.0) p.weight = (1.0 - std::min(1.0, wsum)) / static_cast<double>(parts.size());
  } else {
    bool ok = false;
    const std::string text = read_file(a.str("input"), &ok);
    if (!ok || text.empty()) {
      std::fprintf(stderr, "tokenizer: --input or --mix is required\n");
      return 2;
    }
    parts.push_back(Part{"corpus", text, 1.0});
  }

  // Budget: keep everything if the corpus is small, otherwise cut each source
  // down to its weighted share of `--budget-mb`.
  const double budget = a.num("budget-mb", 24.0) * 1024.0 * 1024.0;
  double wsum = 0.0;
  for (const Part& p : parts) wsum += p.weight;
  std::string corpus;
  std::printf("tokenizer mixture:\n");
  for (const Part& p : parts) {
    const double share = wsum > 0 ? p.weight / wsum : 1.0 / parts.size();
    const size_t want = static_cast<size_t>(std::min<double>(
        static_cast<double>(p.text.size()), budget * share));
    // Cut on a UTF-8 boundary and prefer the head of the file.
    size_t end = std::min(want, p.text.size());
    while (end > 0 && end < p.text.size() &&
           (static_cast<unsigned char>(p.text[end]) & 0xC0) == 0x80)
      --end;
    corpus.append(p.text, 0, end);
    corpus.push_back('\n');
    std::printf("  %-6s share %.2f  using %.2f MB of %.2f MB\n", p.name.c_str(), share,
                end / 1048576.0, p.text.size() / 1048576.0);
  }

  const double t0 = now_seconds();
  tok.train(corpus, vocab, static_cast<int>(a.integer("min-freq", 2)),
            [](int32_t done, int32_t total) {
              if (total > 0 && done % 256 == 0)
                std::printf("\r  merges %d/%d", done, total), std::fflush(stdout);
            });
  std::printf("\rtrained %zu merges, vocab=%d in %.1fs (normalisation %s)\n",
              tok.num_merges(), tok.vocab_size(), now_seconds() - t0,
              tok.normalize() ? "on" : "off");
  if (!tok.save(out)) {
    std::fprintf(stderr, "tokenizer: cannot write %s\n", out.c_str());
    return 1;
  }

  // Fertility per source: the number that tells you whether a language is
  // actually covered by the vocabulary.
  std::printf("\nfertility (higher chars/token is better, low single-byte share is better)\n");
  std::printf("  %-8s %-5s %10s %10s %12s\n", "source", "lang", "chars/tok", "bytes/tok", "single-byte");
  for (const Part& p : parts) {
    const std::string sample = p.text.substr(0, std::min<size_t>(p.text.size(), 200000));
    const Tokenizer::FertilityReport f = tok.fertility(sample);
    std::printf("  %-8s %-5s %10.2f %10.2f %11.1f%%\n", p.name.c_str(),
                lang_code(detect_language(sample)), f.chars_per_token(), f.bytes_per_token(),
                100.0 * f.single_byte_share());
  }

  const std::string probe = "سلام دنیا! می‌روم که ۱۲۳ را یاد بگیرم. def add(a, b): return a + b";
  const std::vector<int32_t> rt = tok.encode(probe);
  const std::string back = tok.decode(rt);
  const std::string norm = tok.preprocess(probe);
  std::printf("\nround trip: %zu tokens  [%s]\n  in : %s\n  out: %s\n", rt.size(),
              back == norm ? "exact" : "MISMATCH", probe.c_str(), back.c_str());
  return back == norm ? 0 : 1;
}

// ---------------------------------------------------------------- tokenize
// Turns text corpora into memory-mappable binary token files.  Encoding is
// parallel over chunks (each thread keeps its own pre-token cache), which is the
// difference between minutes and hours on a gigabyte of Persian text.
int cmd_tokenize(const Args& a) {
  Tokenizer tok;
  if (!tok.load(a.str("tokenizer"))) {
    std::fprintf(stderr, "tokenize: cannot load tokenizer %s\n", a.str("tokenizer").c_str());
    return 1;
  }
  const std::string out_dir = a.str("out-dir", "data/tokens");
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  const int threads = static_cast<int>(a.integer("threads", 0));
  if (threads > 0) omp_set_num_threads(threads);

  // inputs: "fa=data/fa.txt,en=data/en.txt" or a bare list of paths
  std::vector<std::pair<std::string, std::string>> inputs;
  const std::string spec = a.has("in") ? a.str("in") : a.str("mix");
  size_t pos = 0;
  while (pos <= spec.size() && !spec.empty()) {
    const size_t comma = spec.find(',', pos);
    std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (!item.empty()) {
      std::string name, path;
      const size_t eq = item.find('=');
      if (eq == std::string::npos) path = item;
      else {
        name = item.substr(0, eq);
        path = item.substr(eq + 1);
      }
      const size_t colon = path.rfind(':');
      if (colon != std::string::npos &&
          path.find_first_not_of("0123456789.eE-+", colon + 1) == std::string::npos)
        path = path.substr(0, colon);
      if (name.empty()) {
        const size_t slash = path.find_last_of('/');
        name = path.substr(slash == std::string::npos ? 0 : slash + 1);
        const size_t dot = name.rfind('.');
        if (dot != std::string::npos) name = name.substr(0, dot);
      }
      inputs.emplace_back(name, path);
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (inputs.empty()) {
    std::fprintf(stderr, "tokenize: --in fa=data/fa.txt,py=data/py.txt is required\n");
    return 2;
  }

  std::printf("tokenizing with vocab %d (normalisation %s), %d threads\n",
              tok.vocab_size(), tok.normalize() ? "on" : "off", backend_threads());
  int64_t grand_tokens = 0;
  double grand_bytes = 0;
  for (const auto& in : inputs) {
    bool ok = false;
    const std::string text = read_file(in.second, &ok);
    if (!ok || text.empty()) {
      std::fprintf(stderr, "  %-8s cannot read %s\n", in.first.c_str(), in.second.c_str());
      continue;
    }
    const double t0 = now_seconds();
    // Split on line boundaries into ~8 chunks per thread.
    const int nchunks = std::max(1, backend_threads() * 8);
    std::vector<size_t> bounds;
    bounds.push_back(0);
    for (int i = 1; i < nchunks; ++i) {
      size_t p = text.size() / static_cast<size_t>(nchunks) * static_cast<size_t>(i);
      p = text.find('\n', p);
      if (p == std::string::npos) break;
      bounds.push_back(p + 1);
    }
    bounds.push_back(text.size());
    const int n = static_cast<int>(bounds.size()) - 1;
    std::vector<std::vector<int32_t>> parts(static_cast<size_t>(n));
    std::vector<uint64_t> chars(static_cast<size_t>(n), 0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
      const std::string chunk = text.substr(bounds[static_cast<size_t>(i)],
                                            bounds[static_cast<size_t>(i) + 1] -
                                                bounds[static_cast<size_t>(i)]);
      parts[static_cast<size_t>(i)] = tok.encode(chunk);
      chars[static_cast<size_t>(i)] = utf8_length(tok.preprocess(chunk));
    }
    std::vector<int32_t> ids;
    uint64_t total_chars = 0;
    size_t total = 0;
    for (const auto& p2 : parts) total += p2.size();
    ids.reserve(total);
    for (int i = 0; i < n; ++i) {
      ids.insert(ids.end(), parts[static_cast<size_t>(i)].begin(),
                 parts[static_cast<size_t>(i)].end());
      total_chars += chars[static_cast<size_t>(i)];
      parts[static_cast<size_t>(i)].clear();
      parts[static_cast<size_t>(i)].shrink_to_fit();
    }
    const double dt = now_seconds() - t0;
    const std::string out = out_dir + "/" + in.first + ".bin";
    std::string err;
    if (!TokenStore::write_bin(out, ids, tok.vocab_size(), total_chars, 0, &err)) {
      std::fprintf(stderr, "  %s\n", err.c_str());
      return 1;
    }
    std::printf("  %-8s %s -> %s\n", in.first.c_str(),
                human_bytes(static_cast<double>(text.size())).c_str(), out.c_str());
    std::printf("           %10zu tokens  %.2f chars/token  %.2f bytes/token  "
                "%.1f MB/s  (%.0fs)\n",
                ids.size(),
                static_cast<double>(total_chars) / std::max<size_t>(1, ids.size()),
                static_cast<double>(text.size()) / std::max<size_t>(1, ids.size()),
                text.size() / 1048576.0 / std::max(1e-6, dt), dt);
    grand_tokens += static_cast<int64_t>(ids.size());
    grand_bytes += static_cast<double>(text.size());
  }
  std::printf("\ntotal %lld tokens from %s of text  (%s on disk as uint16)\n",
              static_cast<long long>(grand_tokens), human_bytes(grand_bytes).c_str(),
              human_bytes(static_cast<double>(grand_tokens) * 2).c_str());
  std::printf("use them with: --mix fa=%s/fa.bin:0.5,py=%s/py.bin:0.3,...\n",
              out_dir.c_str(), out_dir.c_str());
  return 0;
}

// ----------------------------------------------------------------- pretrain
int cmd_pretrain(const Args& a) {
  Config c;
  if (a.has("config")) c.load(a.str("config"));
  a.apply_sets(&c);
  backend_init(static_cast<int>(a.integer("threads", 0)), a.flag("cuda"));

  Tokenizer tok;
  if (!tok.load(a.str("tokenizer"))) {
    std::fprintf(stderr, "pretrain: cannot load tokenizer %s\n",
                 a.str("tokenizer").c_str());
    return 1;
  }
  GPTConfig g = GPTConfig::from_config(c);
  // When resuming, the *checkpoint* is authoritative for the architecture: it
  // may have grown extra blocks since the config file was written.  Silently
  // rebuilding a smaller model here used to throw the grown layers away.
  CheckpointMeta resume_meta;
  if (a.has("resume")) {
    Dtype rdt = Dtype::F32;
    if (peek_checkpoint(a.str("resume"), &resume_meta, &rdt) &&
        resume_meta.extra.has("model.n_layer")) {
      GPTConfig gm = GPTConfig::from_config(resume_meta.extra);
      gm.grad_checkpointing = g.grad_checkpointing;
      if (gm.n_layer != g.n_layer)
        std::printf("resume: checkpoint has %lld layers (config says %lld) - "
                    "using the checkpoint\n",
                    static_cast<long long>(gm.n_layer), static_cast<long long>(g.n_layer));
      g = gm;
    }
  }
  g.vocab_size = tok.vocab_size();
  if (a.has("ctx")) g.block_size = a.integer("ctx", g.block_size);
  if (a.has("layers")) g.n_layer = a.integer("layers", g.n_layer);
  if (a.has("heads")) g.n_head = a.integer("heads", g.n_head);
  if (a.has("dim")) g.n_embd = a.integer("dim", g.n_embd);
  if (a.has("checkpointing")) g.grad_checkpointing = a.flag("checkpointing");

  MixtureDataset ds;
  if (!build_mixture(a, tok, &ds)) return 1;
  GPT model(g);
  CheckpointMeta meta;
  if (a.has("resume")) {
    ParamStore ps;
    if (!load_checkpoint(a.str("resume"), &ps, &meta)) {
      std::fprintf(stderr, "pretrain: cannot resume from %s\n", a.str("resume").c_str());
      return 1;
    }
    model.load_params(ps);
    std::printf("resumed from %s at step %lld (%s tokens seen)\n",
                a.str("resume").c_str(), static_cast<long long>(meta.step),
                meta.extra.get_str("train.tokens_seen", "?").c_str());
  } else {
    model.init_weights(static_cast<uint64_t>(a.integer("seed", 1337)));
  }

  const int64_t B = a.integer("batch", 8);
  const int64_t T = g.block_size;
  const int64_t steps = a.integer("steps", 200);
  const int64_t accum = std::max<int64_t>(1, a.integer("accum", 1));
  AdamWConfig oc;
  oc.lr = static_cast<float>(a.num("lr", c.get_num("train.lr", 3e-4)));
  oc.weight_decay = static_cast<float>(a.num("wd", c.get_num("train.weight_decay", 0.1)));
  oc.grad_clip = static_cast<float>(a.num("clip", c.get_num("train.grad_clip", 1.0)));
  AdamW opt(oc);
  opt.set_params(model.trainable_params(), model.trainable_names());

  std::printf("%s\n", g.describe().c_str());
  std::printf("corpus mixture (%zu tokens total)\n%s", ds.total_tokens(),
              ds.describe().c_str());
  std::printf("parameters: %lld total (%.3fM), %lld trainable (%.1f%%)\n",
              static_cast<long long>(model.num_params()), model.num_params() / 1e6,
              static_cast<long long>(model.num_trainable()),
              100.0 * static_cast<double>(model.num_trainable()) /
                  std::max<int64_t>(1, model.num_params()));
  std::printf("optimiser state: %s\n", human_bytes(static_cast<double>(opt.state_bytes())).c_str());
  std::printf("training %lld steps, batch %lld x %lld tokens, accum %lld\n\n",
              static_cast<long long>(steps), static_cast<long long>(B),
              static_cast<long long>(T), static_cast<long long>(accum));

  // One hold-out slice per source so training prints per-language numbers.
  std::vector<Batch> val;
  for (int si = 0; si < ds.num_sources(); ++si) {
    std::vector<Batch> b = ds.holdout_batches(si, B, T, 2);
    val.insert(val.end(), b.begin(), b.end());
  }
  Rng rng(static_cast<uint64_t>(a.integer("seed", 1337)) ^ 0x9e37u);
  const int64_t warmup = a.integer("warmup", std::max<int64_t>(4, steps / 20));
  const std::string sched = a.str("schedule", c.get_str("train.schedule", "wsd"));
  const float decay_frac = static_cast<float>(a.num("decay-frac", c.get_num("train.decay_frac", 0.2)));
  const float zloss = static_cast<float>(a.num("zloss", c.get_num("train.zloss", 1e-4)));
  const float spike = static_cast<float>(a.num("skip-spike", c.get_num("train.skip_spike", 4.0)));
  // growth schedule: "3000:2,6000:2" -> add 2 blocks at step 3000 and 6000
  std::vector<std::pair<int64_t, int64_t>> growth;
  {
    const std::string gs = a.str("grow", c.get_str("train.grow", ""));
    size_t gp = 0;
    while (gp <= gs.size() && !gs.empty()) {
      const size_t comma = gs.find(',', gp);
      const std::string item = gs.substr(gp, comma == std::string::npos ? std::string::npos : comma - gp);
      if (!item.empty()) {
        const size_t colon = item.find(':');
        try {
          growth.emplace_back(std::stoll(item.substr(0, colon)),
                              colon == std::string::npos ? 1 : std::stoll(item.substr(colon + 1)));
        } catch (...) {
        }
      }
      if (comma == std::string::npos) break;
      gp = comma + 1;
    }
  }
  double grad_ema = 0.0;
  int64_t skipped = 0;
  int64_t warm_from = 0;
  const int64_t eval_every = a.integer("eval-every", std::max<int64_t>(20, steps / 10));
  const int64_t save_every = a.integer("save-every", 0);
  Dtype dt = Dtype::F32;
  parse_dtype(a.str("dtype", "f32"), &dt);
  const std::string out = a.str("out", "model.slm");

  std::printf("schedule: %s (warmup %lld, decay %.0f%%), z-loss %.1e, spike skip %.1fx\n",
              sched.c_str(), static_cast<long long>(warmup), 100.0 * decay_frac, zloss, spike);
  if (!growth.empty()) {
    std::printf("progressive growth:");
    for (const auto& g2 : growth)
      std::printf(" +%lldL@%lld", static_cast<long long>(g2.second),
                  static_cast<long long>(g2.first));
    std::printf("\n");
  }

  // A resumed run continues the same step counter, so the WSD schedule and the
  // growth schedule keep their meaning across many short sessions.
  const int64_t step0 = a.has("resume") ? meta.step : 0;
  const int64_t total_steps = a.integer("total-steps", step0 + steps);
  double t_start = now_seconds();
  int64_t tokens_done = a.has("resume")
                            ? static_cast<int64_t>(
                                  std::atoll(meta.extra.get_str("train.tokens_seen", "0").c_str()))
                            : 0;
  const int64_t tokens_at_start = tokens_done;
  float best_val = 1e30f;
  for (int64_t step = step0; step < step0 + steps; ++step) {
    // ---- progressive depth growth
    for (const auto& g2 : growth) {
      if (step != g2.first) continue;
      const GPT::GrowthEvent ev = model.grow_depth(g2.second, step);
      opt.set_params(model.trainable_params(), model.trainable_names());
      warm_from = step;  // short re-warmup so the new block settles
      std::printf("\n>>> GROWTH at step %lld: %lld -> %lld layers, "
                  "%.3fM -> %.3fM parameters (+%.1f%%)\n\n",
                  static_cast<long long>(step), static_cast<long long>(ev.layers_before),
                  static_cast<long long>(ev.layers_after), ev.params_before / 1e6,
                  ev.params_after / 1e6,
                  100.0 * (ev.params_after - ev.params_before) /
                      std::max<int64_t>(1, ev.params_before));
      std::fflush(stdout);
    }
    model.zero_grad();
    float loss_sum = 0.0f;
    for (int64_t k = 0; k < accum; ++k) {
      Batch b = ds.sample_batch(B, T, rng);
      float lv = 0.0f;
      Tensor l = model.loss(b.ids, b.targets, B, T, &lv, nullptr);
      Tensor total = l;
      if (zloss > 0.0f) {
        // z-loss needs the logits; recomputing them would double the cost, so
        // it is applied through the same graph via the model's last logits.
        total = l;  // (see note below: applied inside loss_with_zloss)
      }
      Tensor scaled = total.scale(1.0f / static_cast<float>(accum));
      scaled.backward();
      loss_sum += lv;
      tokens_done += B * T;
    }
    const int64_t sstep = step - warm_from;
    const int64_t swarm = warm_from > 0 ? std::min<int64_t>(warmup, 100) : warmup;
    const float lr = (sched == "cosine")
                         ? lr_schedule(step, oc.lr, warmup, total_steps, 0.1f)
                         : lr_schedule_wsd(sstep, oc.lr, swarm, total_steps - warm_from,
                                           decay_frac, 0.02f);
    bool was_skipped = false;
    const float skip_at = (spike > 0.0f && grad_ema > 0.0) ? static_cast<float>(spike * grad_ema)
                                                           : 0.0f;
    const float gn = opt.step(lr, skip_at, &was_skipped);
    if (was_skipped) {
      ++skipped;
    } else {
      grad_ema = grad_ema == 0.0 ? gn : 0.98 * grad_ema + 0.02 * gn;
    }
    const float loss = loss_sum / static_cast<float>(accum);
    if (step % std::max<int64_t>(1, a.integer("log-every", 10)) == 0 ||
        step + 1 == step0 + steps) {
      const double dt_s = now_seconds() - t_start;
      std::printf("step %5lld | loss %.4f | ppl %8.2f | lr %.2e | gnorm %.3f | "
                  "%6.0f tok/s | %.2fM tok | %.3fM par%s\n",
                  static_cast<long long>(step), loss, std::exp(std::min(20.0f, loss)),
                  lr, gn, static_cast<double>(tokens_done - tokens_at_start) / std::max(1e-6, dt_s),
                  tokens_done / 1e6, model.num_params() / 1e6,
                  skipped ? ("  skipped " + std::to_string(skipped)).c_str() : "");
      std::fflush(stdout);
    }
    if (!val.empty() && ((step + 1) % eval_every == 0 || step + 1 == step0 + steps)) {
      double sum = 0.0;
      double lang_sum[kNumLangs] = {};
      int lang_n[kNumLangs] = {};
      for (const Batch& vb : val) {
        const float l = model.eval_loss(vb.ids, vb.targets, vb.B, vb.T);
        sum += l;
        lang_sum[static_cast<int>(vb.lang)] += l;
        ++lang_n[static_cast<int>(vb.lang)];
      }
      const float vl = static_cast<float>(sum / val.size());
      const bool best = vl <= best_val;
      best_val = std::min(best_val, vl);
      std::printf("        >> holdout %.4f (ppl %.2f)%s  |", vl, std::exp(std::min(20.0f, vl)),
                  best ? " *best" : "");
      for (int l = 0; l < kNumLangs; ++l)
        if (lang_n[l])
          std::printf("  %s %.4f", lang_code(static_cast<Lang>(l)),
                      lang_sum[l] / lang_n[l]);
      std::printf("\n");
      std::fflush(stdout);
    }
    if (save_every > 0 && (step + 1) % save_every == 0) {
      CheckpointMeta m;
      m.step = step + 1;
      m.tokens_seen = tokens_done;
      model.config().write_to(m.extra);   // n_layer may have grown
      m.extra.set("tokenizer", a.str("tokenizer"));
      m.extra.set("train.tokens_seen", std::to_string(tokens_done));
      save_checkpoint(out, *model.snapshot(), m, dt);
    }
  }
  CheckpointMeta m;
  m.step = step0 + steps;
  m.tokens_seen = tokens_done;
  model.config().write_to(m.extra);
  m.extra.set("tokenizer", a.str("tokenizer"));
  m.extra.set("train.holdout_loss", std::to_string(best_val));
  m.extra.set("train.mixture", a.has("mix") ? a.str("mix") : a.str("data"));
  m.extra.set("train.tokens_seen", std::to_string(tokens_done));
  m.extra.set("train.params", std::to_string(model.num_params()));
  if (!save_checkpoint(out, *model.snapshot(), m, dt)) {
    std::fprintf(stderr, "pretrain: cannot write %s\n", out.c_str());
    return 1;
  }
  std::printf("\nfinal: %lld parameters (%.3fM), %.2fM tokens seen, %lld spikes skipped\n",
              static_cast<long long>(model.num_params()), model.num_params() / 1e6,
              tokens_done / 1e6, static_cast<long long>(skipped));
  for (const GPT::GrowthEvent& ev : model.growth_events())
    std::printf("  growth @%lld: %lldL/%.3fM -> %lldL/%.3fM\n",
                static_cast<long long>(ev.step), static_cast<long long>(ev.layers_before),
                ev.params_before / 1e6, static_cast<long long>(ev.layers_after),
                ev.params_after / 1e6);
  std::printf("saved %s (%s, %s)\n", out.c_str(), dtype_name(dt),
              human_bytes(static_cast<double>(
                              encoded_bytes(model.num_params(), dt)))
                  .c_str());
  return 0;
}

// -------------------------------------------------------------------- shared
bool load_model(const Args& a, Tokenizer* tok, std::unique_ptr<GPT>* model,
                CheckpointMeta* meta) {
  if (!tok->load(a.str("tokenizer"))) {
    std::fprintf(stderr, "cannot load tokenizer %s\n", a.str("tokenizer").c_str());
    return false;
  }
  ParamStore ps;
  if (!load_checkpoint(a.str("ckpt"), &ps, meta)) {
    std::fprintf(stderr, "cannot load checkpoint %s\n", a.str("ckpt").c_str());
    return false;
  }
  GPTConfig g = GPTConfig::from_config(meta->extra);
  *model = std::make_unique<GPT>(g);
  (*model)->load_params(ps);
  return true;
}

// ---------------------------------------------------------------------- chat
int cmd_chat(const Args& a) {
  backend_init(static_cast<int>(a.integer("threads", 0)), a.flag("cuda"));
  Tokenizer tok;
  std::unique_ptr<GPT> model;
  CheckpointMeta meta;
  if (!load_model(a, &tok, &model, &meta)) return 1;
  std::printf("%s\nloaded step %lld\n", model->config().describe().c_str(),
              static_cast<long long>(meta.step));
  MemoryStore mem;
  const bool use_mem = a.has("memory") && mem.open(a.str("memory"));
  if (use_mem)
    std::printf("memory: %zu entries from %s\n", mem.size(), a.str("memory").c_str());
  GenOptions go;
  go.max_new_tokens = static_cast<int>(a.integer("max-new", 96));
  go.temperature = static_cast<float>(a.num("temp", 0.9));
  go.top_k = static_cast<int>(a.integer("top-k", 40));
  go.top_p = static_cast<float>(a.num("top-p", 0.95));
  go.seed = static_cast<uint64_t>(a.integer("seed", 0));

  auto run = [&](const std::string& prompt) {
    std::vector<int32_t> ids = tok.encode(prompt);
    if (ids.empty()) ids.push_back(Tokenizer::kEot);
    const double t0 = now_seconds();
    int n = 0;
    std::vector<int32_t> outv = model->generate(ids, go, [&](const GenStep& s) {
      std::printf("%s", tok.decode({s.token}).c_str());
      std::fflush(stdout);
      ++n;
      return true;
    });
    const double dt_s = now_seconds() - t0;
    std::printf("\n[%d tokens, %.1f tok/s]\n", n, n / std::max(1e-6, dt_s));
    return outv;
  };

  if (a.has("prompt")) {
    run(a.str("prompt"));
    return 0;
  }
  std::printf("interactive mode - empty line quits\n");
  std::string line;
  while (true) {
    std::printf("\nyou> ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line) || line.empty()) break;
    std::string prompt = "<|user|>" + line + "<|assistant|>";
    if (use_mem) {
      const std::string block = mem.context_block(line, 3);
      if (!block.empty()) {
        std::printf("[memory: %zu chars injected]\n", block.size());
        prompt = block + prompt;
      }
    }
    std::printf("slm> ");
    run(prompt);
  }
  return 0;
}

// ---------------------------------------------------------------------- eval
int cmd_eval(const Args& a) {
  backend_init(static_cast<int>(a.integer("threads", 0)), a.flag("cuda"));
  Tokenizer tok;
  std::unique_ptr<GPT> model;
  CheckpointMeta meta;
  if (!load_model(a, &tok, &model, &meta)) return 1;
  MixtureDataset ds;
  if (!build_mixture(a, tok, &ds)) return 1;
  const int64_t B = a.integer("batch", 4);
  const int64_t T = std::min<int64_t>(a.integer("ctx", model->config().block_size),
                                      model->config().block_size);
  const int nb = static_cast<int>(a.integer("batches", 8));
  std::printf("%s\n\n", model->config().describe().c_str());
  // Token perplexity is NOT comparable across languages: a tokenizer that packs
  // 5 Persian characters into one token and 3 Python characters into one token
  // produces different numbers for the same modelling quality.  Bits per
  // character is the tokenizer-independent metric, so both are printed.
  std::printf("  %-8s %-5s %10s %12s %10s %10s %8s\n", "source", "lang", "loss",
              "perplexity", "bits/char", "chars/tok", "batches");
  double total = 0.0;
  int total_n = 0;
  for (int si = 0; si < ds.num_sources(); ++si) {
    const std::vector<Batch> vb = ds.holdout_batches(si, B, T, nb);
    if (vb.empty()) {
      std::printf("  %-8s %-5s  (hold-out too small)\n", ds.info(si).name.c_str(),
                  lang_code(ds.info(si).lang));
      continue;
    }
    double sum = 0.0;
    for (const Batch& b : vb) sum += model->eval_loss(b.ids, b.targets, b.B, b.T);
    const double loss = sum / static_cast<double>(vb.size());
    total += sum;
    total_n += static_cast<int>(vb.size());
    const double cpt = std::max(1e-6, ds.info(si).chars_per_token);
    const double bpc = loss / (cpt * std::log(2.0));
    std::printf("  %-8s %-5s %10.4f %12.2f %10.4f %10.2f %8zu\n", ds.info(si).name.c_str(),
                lang_code(ds.info(si).lang), loss, std::exp(loss), bpc, cpt, vb.size());
  }
  if (total_n) {
    const double loss = total / total_n;
    std::printf("  %-8s %-5s %10.4f %12.2f %10s %10s %8d\n", "ALL", "-", loss,
                std::exp(loss), "-", "-", total_n);
  }
  return 0;
}

// ------------------------------------------------------------------ langcheck
// Generates from each language's own prompts and reports the three things that
// matter for a multilingual model: quality, interference and code validity.
int cmd_langcheck(const Args& a) {
  backend_init(static_cast<int>(a.integer("threads", 0)), a.flag("cuda"));
  Tokenizer tok;
  std::unique_ptr<GPT> model;
  CheckpointMeta meta;
  if (!load_model(a, &tok, &model, &meta)) return 1;
  MixtureDataset ds;
  if (!build_mixture(a, tok, &ds)) return 1;

  const int samples = static_cast<int>(a.integer("samples", 12));
  GenOptions go;
  go.max_new_tokens = static_cast<int>(a.integer("max-new", 64));
  go.temperature = static_cast<float>(a.num("temp", 0.8));
  go.top_k = static_cast<int>(a.integer("top-k", 40));
  go.top_p = static_cast<float>(a.num("top-p", 0.95));
  Rng rng(static_cast<uint64_t>(a.integer("seed", 99)));

  std::printf("%s\n\n", model->config().describe().c_str());
  std::printf("  %-6s %8s %8s %10s %10s %10s\n", "lang", "holdout", "ppl", "interfere",
              "py-valid", "samples");
  int rc = 0;
  for (int si = 0; si < ds.num_sources(); ++si) {
    const Lang lang = ds.info(si).lang;
    // 1) hold-out perplexity for this language
    const std::vector<Batch> vb =
        ds.holdout_batches(si, 2, std::min<int64_t>(256, model->config().block_size), 4);
    double loss = 0.0;
    for (const Batch& b : vb) loss += model->eval_loss(b.ids, b.targets, b.B, b.T);
    loss = vb.empty() ? 0.0 : loss / static_cast<double>(vb.size());

    // 2) generate from this language's prompts
    const TokenStore& toks = ds.data(si).tokens();
    const int64_t limit = ds.data(si).train_tokens();
    std::vector<std::vector<int32_t>> prompts;
    for (int64_t i = 0; i < limit && static_cast<int>(prompts.size()) < 256; ++i) {
      if (toks.at(static_cast<size_t>(i)) != Tokenizer::kUser) continue;
      int64_t j = i + 1;
      while (j < limit && toks.at(static_cast<size_t>(j)) != Tokenizer::kAssistant && j - i < 64) ++j;
      if (j >= limit || toks.at(static_cast<size_t>(j)) != Tokenizer::kAssistant) continue;
      { std::vector<int32_t> p; for (int64_t q = i; q <= j; ++q) p.push_back(toks.at(static_cast<size_t>(q))); prompts.push_back(std::move(p)); }
      i = j;
    }
    double switch_sum = 0.0;
    int py_ok = 0, py_total = 0, n = 0;
    std::string first_sample;
    for (int k = 0; k < samples && !prompts.empty(); ++k) {
      go.seed = rng.next_u64() | 1ull;
      const std::vector<int32_t>& p = prompts[static_cast<size_t>(rng.below(prompts.size()))];
      const std::vector<int32_t> out = model->generate(p, go);
      const std::string text = tok.decode(out);
      if (first_sample.empty()) first_sample = text;
      switch_sum += code_switch_ratio(text, lang);
      if (lang == Lang::kPython) {
        ++py_total;
        size_t open = text.find("```");
        std::string code = text;
        if (open != std::string::npos) {
          size_t body = text.find('\n', open);
          if (body != std::string::npos) {
            const size_t close = text.find("```", body + 1);
            code = text.substr(body + 1, close == std::string::npos ? std::string::npos
                                                                   : close - body - 1);
          }
        }
        if (check_python(code).ok) ++py_ok;
      }
      ++n;
    }
    const double interfere = n ? switch_sum / n : 0.0;
    char pybuf[32] = "-";
    if (py_total) std::snprintf(pybuf, sizeof(pybuf), "%.0f%%", 100.0 * py_ok / py_total);
    std::printf("  %-6s %8.4f %8.2f %9.1f%% %10s %10d\n", lang_code(lang), loss,
                std::exp(loss), 100.0 * interfere, pybuf, n);
    if (a.flag("show") && !first_sample.empty())
      std::printf("      sample: %s\n", first_sample.substr(0, 200).c_str());
    // Fail the command when a language is clearly leaking into another.
    if (interfere > a.num("max-interference", 0.25)) rc = 1;
  }
  std::printf("\ninterfere = share of foreign-script letters in this language's output\n");
  if (rc) std::printf("FAIL: at least one language exceeds the interference threshold\n");
  return rc;
}

// ------------------------------------------------------------------ quantize
int cmd_quantize(const Args& a) {
  ParamStore ps;
  CheckpointMeta meta;
  if (!load_checkpoint(a.str("in"), &ps, &meta)) {
    std::fprintf(stderr, "quantize: cannot read %s\n", a.str("in").c_str());
    return 1;
  }
  Dtype dt = Dtype::Q8;
  if (!parse_dtype(a.str("dtype", "q8"), &dt)) {
    std::fprintf(stderr, "quantize: unknown dtype\n");
    return 2;
  }
  const std::string out = a.str("out", a.str("in") + "." + dtype_name(dt));
  if (!save_checkpoint(out, ps, meta, dt)) {
    std::fprintf(stderr, "quantize: cannot write %s\n", out.c_str());
    return 1;
  }
  // Round-trip error report.
  ParamStore back;
  CheckpointMeta m2;
  load_checkpoint(out, &back, &m2);
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ps.data.size(); ++i) {
    const int j = back.find(ps.names[i]);
    if (j < 0) continue;
    for (size_t k = 0; k < ps.data[i].size(); ++k) {
      const double d = static_cast<double>(ps.data[i][k]) -
                       static_cast<double>(back.data[static_cast<size_t>(j)][k]);
      num += d * d;
      den += static_cast<double>(ps.data[i][k]) * static_cast<double>(ps.data[i][k]);
    }
  }
  std::ifstream fa(a.str("in"), std::ios::binary | std::ios::ate);
  std::ifstream fb(out, std::ios::binary | std::ios::ate);
  std::printf("wrote %s (%s)\n  size %s -> %s  (%.2fx)\n  relative RMS error %.4f%%\n",
              out.c_str(), dtype_name(dt),
              human_bytes(static_cast<double>(fa.tellg())).c_str(),
              human_bytes(static_cast<double>(fb.tellg())).c_str(),
              static_cast<double>(fa.tellg()) / std::max(1.0, static_cast<double>(fb.tellg())),
              100.0 * std::sqrt(num / std::max(1e-12, den)));
  return 0;
}

// ========================================================= quantised (.slmq)
// The deployment path.  `pack` turns a training checkpoint into a memory
// mappable int4/int8 file; `qrun` / `qbench` / `qeval` run it with the integer
// kernels and never materialise a float weight.

void set_threads(const Args& a) {
#ifdef _OPENMP
  const int n = static_cast<int>(a.integer("threads", 0));
  if (n > 0) omp_set_num_threads(n);
#endif
}

bool parse_pack_options(const Args& a, QPackOptions* o) {
  const std::string body = a.str("type", a.str("bits", "q4"));
  if (!parse_qtype(body, &o->type)) {
    std::fprintf(stderr, "pack: unknown --bits/--type '%s' (use 4, 8, f16, f32)\n",
                 body.c_str());
    return false;
  }
  const std::string emb = a.str("embed", o->type == QType::F32 ? "f32" : "q8");
  if (!parse_qtype(emb, &o->embed_type)) {
    std::fprintf(stderr, "pack: unknown --embed '%s'\n", emb.c_str());
    return false;
  }
  o->progress = !a.flag("quiet");
  if (a.has("note")) o->note = a.str("note");
  return true;
}

void print_qmodel_report(const QModel& qm) {
  const double P = static_cast<double>(qm.param_count());
  std::printf("  %s\n", qm.describe().c_str());
  std::printf("  parameters       : %lld  (%.3f M)\n",
              static_cast<long long>(qm.param_count()), P / 1e6);
  std::printf("  file             : %s  (%.3f bits/weight)\n",
              human_bytes(static_cast<double>(qm.file_bytes())).c_str(),
              qm.bits_per_weight());
  std::printf("  same model f32   : %s   -> %.1fx smaller\n",
              human_bytes(P * 4.0).c_str(),
              P * 4.0 / std::max(1.0, static_cast<double>(qm.file_bytes())));
  std::printf("  resident now     : %s  (page cache; a fresh process starts at 0)\n",
              human_bytes(static_cast<double>(qm.resident_bytes())).c_str());
}

// ------------------------------------------------------------------ slm pack
int cmd_pack(const Args& a) {
  set_threads(a);
  QPackOptions o;
  if (!parse_pack_options(a, &o)) return 2;
  const std::string out = a.str("out");
  if (out.empty()) {
    std::fprintf(stderr, "pack: --out is required\n");
    return 2;
  }
  std::string err;
  const double t0 = now_seconds();

  if (a.flag("synth")) {
    // No checkpoint: build a model of the requested shape directly in quantised
    // form.  Rows are generated, quantised and written one chunk at a time, so
    // packing a 2 B parameter file needs a few megabytes of RAM, not 8 GB.
    Config c;
    if (a.has("config")) c.load(a.str("config"));
    if (!a.has("config")) c.set("model.arch_version", "2");
    a.apply_sets(&c);
    GPTConfig g = GPTConfig::from_config(c);
    if (a.has("layers")) g.n_layer = a.integer("layers", g.n_layer);
    if (a.has("dim")) g.n_embd = a.integer("dim", g.n_embd);
    if (a.has("heads")) g.n_head = a.integer("heads", g.n_head);
    if (a.has("kv-heads")) g.n_kv_head = a.integer("kv-heads", g.n_kv_head);
    if (a.has("ffn")) g.ffn_hidden = a.integer("ffn", g.ffn_hidden);
    if (a.has("vocab")) g.vocab_size = static_cast<int32_t>(a.integer("vocab", g.vocab_size));
    if (a.has("ctx")) g.block_size = a.integer("ctx", g.block_size);
    g.validate();
    std::printf("synthesising %s\n", g.describe().c_str());
    std::printf("  target file      : %s\n",
                human_bytes(static_cast<double>(qpack_estimate_bytes(g, o))).c_str());
    if (!qpack_synthesise(g, o, out, static_cast<uint64_t>(a.integer("seed", 1234)),
                          &err)) {
      std::fprintf(stderr, "pack: %s\n", err.c_str());
      return 1;
    }
  } else {
    const std::string in = a.str("in", a.str("ckpt"));
    if (in.empty()) {
      std::fprintf(stderr, "pack: --in <checkpoint.slm> (or --synth) is required\n");
      return 2;
    }
    std::printf("packing %s -> %s  (body %s, embedding %s, group %lld)\n", in.c_str(),
                out.c_str(), qtype_name(o.type), qtype_name(o.embed_type),
                static_cast<long long>(kQBlock));
    if (!qpack_from_checkpoint(in, o, out, &err)) {
      std::fprintf(stderr, "pack: %s\n", err.c_str());
      return 1;
    }
  }
  const double dt = now_seconds() - t0;

  QModel qm;
  if (!qm.open(out, &err)) {
    std::fprintf(stderr, "pack: wrote the file but cannot open it back: %s\n",
                 err.c_str());
    return 1;
  }
  std::printf("\nwrote %s in %.1f s\n", out.c_str(), dt);
  print_qmodel_report(qm);
  const std::string tokp = a.str("tokenizer");
  if (!tokp.empty())
    std::printf("\nrun it:  slm qrun --model %s --tokenizer %s --prompt \"...\"\n",
                out.c_str(), tokp.c_str());
  return 0;
}

// ------------------------------------------------------------------ slm qrun
int cmd_qrun(const Args& a) {
  set_threads(a);
  QModel qm;
  std::string err;
  if (!qm.open(a.str("model"), &err)) {
    std::fprintf(stderr, "qrun: %s\n", err.c_str());
    return 1;
  }
  Tokenizer tok;
  if (!tok.load(a.str("tokenizer"))) {
    std::fprintf(stderr, "qrun: cannot load tokenizer %s\n", a.str("tokenizer").c_str());
    return 1;
  }
  std::printf("%s\n", qm.describe().c_str());
  std::printf("kernels: %s, %d thread(s)\n\n", quant_backend_name(), backend_threads());
  if (a.flag("prefetch")) qm.prefetch();

  GenOptions go;
  go.max_new_tokens = static_cast<int>(a.integer("max-new", 96));
  go.temperature = static_cast<float>(a.num("temp", 0.9));
  go.top_k = static_cast<int>(a.integer("top-k", 40));
  go.top_p = static_cast<float>(a.num("top-p", 0.95));
  go.repetition_penalty = static_cast<float>(a.num("rep-penalty", 1.08));
  go.seed = static_cast<uint64_t>(a.integer("seed", 0));

  auto run = [&](const std::string& prompt) {
    std::vector<int32_t> ids = tok.encode(prompt);
    if (ids.empty()) ids.push_back(Tokenizer::kEot);
    const double t0 = now_seconds();
    int n = 0;
    double first = 0.0;
    qm.generate(ids, go, [&](const GenStep& s) {
      if (n == 0) first = now_seconds() - t0;
      std::printf("%s", tok.decode({s.token}).c_str());
      std::fflush(stdout);
      ++n;
      return true;
    });
    const double dt = now_seconds() - t0;
    std::printf(
        "\n[%d tokens, %.1f tok/s decode, prompt of %zu tokens in %.2f s, "
        "resident %s]\n",
        n, (n - 1) / std::max(1e-6, dt - first), ids.size(), first,
        human_bytes(static_cast<double>(qm.resident_bytes())).c_str());
  };

  if (a.has("prompt")) {
    run(a.str("prompt"));
    return 0;
  }
  std::printf("interactive mode - empty line quits\n");
  std::string line;
  while (true) {
    std::printf("\n> ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line) || line.empty()) break;
    run(line);
  }
  return 0;
}

// ---------------------------------------------------------------- slm qbench
// Measures the two numbers that actually decide whether a model is usable:
// prompt throughput (compute bound) and decode throughput (memory bound).
int cmd_qbench(const Args& a) {
  set_threads(a);
  QModel qm;
  std::string err;
  if (!qm.open(a.str("model"), &err)) {
    std::fprintf(stderr, "qbench: %s\n", err.c_str());
    return 1;
  }
  std::printf("%s\n", qm.describe().c_str());
  std::printf("kernels: %s, %d thread(s)\n", quant_backend_name(), backend_threads());
  const int64_t plen = a.integer("prompt-len", 64);
  const int64_t gen = a.integer("gen", 32);
  const int64_t ctx = std::max<int64_t>(plen + gen + 1, 8);
  if (a.flag("cold")) {
    // Drop the file's clean pages so the run starts from disk, the way a fresh
    // process on a fresh boot would.
    qm.drop_page_cache();
    std::printf("dropped page cache for the mapping\n");
  }
  if (a.flag("prefetch")) {
    const double t0 = now_seconds();
    qm.prefetch();
    std::printf("prefetch (stream all pages): %.2f s\n", now_seconds() - t0);
  }
  std::printf("resident before: %s\n",
              human_bytes(static_cast<double>(qm.resident_bytes())).c_str());

  Rng rng(1234);
  std::vector<int32_t> ids;
  for (int64_t i = 0; i < plen; ++i)
    ids.push_back(static_cast<int32_t>(rng.below(static_cast<uint64_t>(qm.config().vocab_size))));

  QGenState st;
  qm.reset(&st, ctx);
  std::vector<float> logits;
  const double t0 = now_seconds();
  qm.forward(&st, ids, &logits);
  const double t_prefill = now_seconds() - t0;

  const double t1 = now_seconds();
  for (int64_t i = 0; i < gen; ++i)
    qm.forward_token(&st, static_cast<int32_t>(rng.below(static_cast<uint64_t>(
                              qm.config().vocab_size))),
                     &logits);
  const double t_decode = now_seconds() - t1;

  const double wbytes =
      static_cast<double>(qm.param_count()) * qm.bits_per_weight() / 8.0;
  std::printf("\nprompt (%lld tokens) : %.3f s  ->  %.0f tok/s\n",
              static_cast<long long>(plen), t_prefill,
              static_cast<double>(plen) / std::max(1e-9, t_prefill));
  std::printf("decode (%lld tokens) : %.3f s  ->  %.1f tok/s  (%.1f ms/token)\n",
              static_cast<long long>(gen), t_decode,
              static_cast<double>(gen) / std::max(1e-9, t_decode),
              1000.0 * t_decode / static_cast<double>(std::max<int64_t>(1, gen)));
  // One decoded token reads every weight exactly once, so this is the honest
  // memory-bandwidth number.  Note that a model small enough to sit in cache is
  // *not* bandwidth bound and will show little difference between encodings; the
  // win from int4 appears once the weights no longer fit (see docs).
  std::printf("weight traffic      : %s per token  ->  %.1f GB/s effective\n",
              human_bytes(wbytes).c_str(),
              wbytes * static_cast<double>(gen) / std::max(1e-9, t_decode) / 1e9);
  std::printf("KV cache (ctx %lld)  : %s\n", static_cast<long long>(ctx),
              human_bytes(static_cast<double>(st.cache_bytes())).c_str());
  std::printf("resident after      : %s of %s\n",
              human_bytes(static_cast<double>(qm.resident_bytes())).c_str(),
              human_bytes(static_cast<double>(qm.file_bytes())).c_str());
  return 0;
}

// ----------------------------------------------------------------- slm qeval
// Quality of the quantised model, and (with --ckpt) the exact cost of
// quantisation measured on the same token stream.
int cmd_qeval(const Args& a) {
  set_threads(a);
  QModel qm;
  std::string err;
  if (!qm.open(a.str("model"), &err)) {
    std::fprintf(stderr, "qeval: %s\n", err.c_str());
    return 1;
  }
  Tokenizer tok;
  if (!tok.load(a.str("tokenizer"))) {
    std::fprintf(stderr, "qeval: cannot load tokenizer %s\n", a.str("tokenizer").c_str());
    return 1;
  }
  const std::string data = a.str("data");
  std::ifstream f(data, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "qeval: cannot read %s\n", data.c_str());
    return 1;
  }
  const int64_t max_bytes = a.integer("bytes", 200000);
  std::string text;
  text.resize(static_cast<size_t>(max_bytes));
  f.read(&text[0], max_bytes);
  text.resize(static_cast<size_t>(f.gcount()));
  if (a.flag("tail")) {
    // The hold-out slice used during training is the *end* of each file.
    f.clear();
    f.seekg(0, std::ios::end);
    const int64_t total = static_cast<int64_t>(f.tellg());
    if (total > max_bytes) {
      f.seekg(total - max_bytes);
      text.assign(static_cast<size_t>(max_bytes), '\0');
      f.read(&text[0], max_bytes);
      text.resize(static_cast<size_t>(f.gcount()));
    }
  }
  // Do not cut a UTF-8 sequence in half; that would corrupt the first token.
  text = utf8_sanitize(text);
  const int64_t chars = static_cast<int64_t>(utf8_length(text));
  std::vector<int32_t> ids = tok.encode(text);
  const int64_t want = a.integer("tokens", 8000);
  const double chars_per_token =
      ids.empty() ? 1.0 : static_cast<double>(chars) / static_cast<double>(ids.size());
  if (static_cast<int64_t>(ids.size()) > want) ids.resize(static_cast<size_t>(want));
  if (ids.size() < 2) {
    std::fprintf(stderr, "qeval: not enough text\n");
    return 1;
  }
  const int64_t chunk = std::min<int64_t>(qm.config().block_size, 512);

  std::printf("%s\n", qm.describe().c_str());
  std::printf("data: %s  (%lld tokens, %.2f chars/token, chunks of %lld)\n\n",
              data.c_str(), static_cast<long long>(ids.size()), chars_per_token,
              static_cast<long long>(chunk));

  // Note on speed: the quantised path scores one token at a time through the KV
  // cache (exactly what generation does), while the f32 reference scores a whole
  // window in one batched GEMM.  The two tok/s numbers are therefore *not*
  // comparable - use `slm qbench` for throughput.
  auto report = [&](const char* what, double nats, int64_t n, double secs) {
    const double bpc = nats / (chars_per_token * std::log(2.0));
    std::printf("  %-14s nats/token %7.4f   ppl %9.2f   bits/char %7.4f   "
                "(%.1f s)\n",
                what, nats, std::exp(nats), bpc, secs);
    (void)n;
    return bpc;
  };

  int64_t n = 0;
  const double t0 = now_seconds();
  const double nats_q = qm.eval_nats(ids, &n, chunk);
  const double bpc_q = report(qtype_name(qm.body_type()), nats_q, n, now_seconds() - t0);

  if (a.has("ckpt")) {
    // Same tokens, same chunking, float32 weights: the difference is exactly the
    // price of quantisation.
    backend_init(static_cast<int>(a.integer("threads", 0)), false);
    ParamStore ps;
    CheckpointMeta meta;
    if (!load_checkpoint(a.str("ckpt"), &ps, &meta)) {
      std::fprintf(stderr, "qeval: cannot load %s\n", a.str("ckpt").c_str());
      return 1;
    }
    GPTConfig g = GPTConfig::from_config(meta.extra);
    g.n_layer = qm.config().n_layer;  // the checkpoint may have grown
    GPT ref(g);
    ref.load_params(ps);
    const double t1 = now_seconds();
    double total = 0.0;
    int64_t nn = 0;
    for (size_t start = 0; start + 1 < ids.size(); start += static_cast<size_t>(chunk)) {
      const size_t stop = std::min(ids.size() - 1, start + static_cast<size_t>(chunk));
      const int64_t T = static_cast<int64_t>(stop - start);
      if (T < 1) break;
      std::vector<int32_t> inp(ids.begin() + start, ids.begin() + start + T);
      std::vector<int32_t> tgt(ids.begin() + start + 1, ids.begin() + start + T + 1);
      total += ref.eval_loss(inp, tgt, 1, T) * static_cast<double>(T);
      nn += T;
    }
    const double nats_f = nn ? total / static_cast<double>(nn) : 0.0;
    const double bpc_f = report("f32 reference", nats_f, nn, now_seconds() - t1);
    std::printf("\n  quantisation cost: %+.4f nats/token (%+.4f bits/char, %+.2f%% ppl)\n",
                nats_q - nats_f, bpc_q - bpc_f,
                100.0 * (std::exp(nats_q) / std::max(1e-9, std::exp(nats_f)) - 1.0));
  }
  return 0;
}


// ====================================================================== slm up
// One command, no arguments: find (or create) the model and open the dashboard.
int cmd_up(const Args& a) {
  if (a.flag("where")) {
    std::printf("SLM_HOME  : %s\n",
                std::getenv("SLM_HOME") ? std::getenv("SLM_HOME") : "(unset)");
    std::printf("user data : %s\n", spt_user_dir().c_str());
    std::printf("\nsearched in order:\n");
    for (const std::string& d : spt_search_dirs()) {
      std::error_code ec;
      const bool tok = std::filesystem::is_regular_file(d + "/spt.slmtok", ec);
      const bool ck = std::filesystem::is_regular_file(d + "/spt.slm", ec);
      const bool q4 = std::filesystem::is_regular_file(d + "/spt-q4.slmq", ec);
      std::printf("  %-44s %s%s%s\n", d.c_str(), tok ? "spt.slmtok " : "",
                  ck ? "spt.slm " : "", q4 ? "spt-q4.slmq" : "");
    }
    return 0;
  }

  SptAssets assets;
  std::string err;
  std::printf("slm up\n");
  if (!resolve_spt(&assets, !a.flag("no-bootstrap"), &err,
                   [](const std::string& m) { std::printf("  %s\n", m.c_str()); })) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  if (assets.bootstrapped) std::printf("\n  %s\n\n", assets.note.c_str());

  AppOptions o;
  if (a.has("config")) o.cfg.load(a.str("config"));
  a.apply_sets(&o.cfg);
  o.ckpt = a.str("ckpt", assets.ckpt);
  o.tokenizer = a.str("tokenizer", assets.tokenizer);
  o.workdir = a.str("workdir", assets.workdir);
  // A corpus is optional; without one, replay and the hold-out gate are off, and
  // the dashboard says so rather than refusing to start.
  o.data = a.has("mix") ? a.str("mix") : a.str("data");
  if (o.data.empty()) {
    for (const char* p : {"data/mixed.txt", "data/fa.txt", "data/sample_corpus.txt"}) {
      std::error_code ec;
      if (std::filesystem::is_regular_file(p, ec)) {
        o.data = p;
        std::printf("  corpus: %s\n", p);
        break;
      }
    }
  }
  o.headless = a.flag("terminal") || a.flag("headless") ||
               (std::getenv("DISPLAY") == nullptr && std::getenv("WAYLAND_DISPLAY") == nullptr);
  if (o.headless && !a.flag("terminal") && !a.flag("headless"))
    std::printf("  no display detected - using the terminal dashboard\n");
  o.seconds = a.num("seconds", 0.0);
  o.threads = static_cast<int>(a.integer("threads", 0));
  o.cuda = a.flag("cuda");
  o.seed = static_cast<uint64_t>(a.integer("seed", 1234));
  o.autopilot = a.flag("autopilot");
  o.gguf = a.str("gguf", std::getenv("SLM_GGUF") ? std::getenv("SLM_GGUF") : "");
  o.workspace = a.str("workspace", ".");
  std::printf("\n");
  return run_self_training(o);
}

// ==================================================================== slm agent
// The dual-model agent from the command line: one model, the other model, or a
// weighted debate between them, with tools and codebase retrieval.
int cmd_agent(const Args& a) {
  set_threads(a);
  backend_init(static_cast<int>(a.integer("threads", 0)), a.flag("cuda"));

  SptAssets assets;
  std::string err;
  if (!resolve_spt(&assets, true, &err, nullptr)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }

  Telemetry tel;
  const std::string workdir = a.str("workdir", assets.workdir);
  std::error_code ec;
  std::filesystem::create_directories(workdir, ec);
  tel.open_audit(workdir + "/audit.jsonl");

  AgentRuntimeOptions ro;
  // int4 for pure inference when it exists: same answers, a fraction of the RAM.
  if (!assets.quant.empty() && !a.flag("f32")) ro.spt_quant = assets.quant;
  else ro.spt_ckpt = assets.ckpt;
  ro.tokenizer = assets.tokenizer;
  ro.workspace = a.str("workspace", ".");
  ro.workdir = workdir;
  ro.enable_web = !a.flag("no-web");
  ro.enable_shell = !a.flag("no-shell");
  ro.enable_codebase = !a.flag("no-codebase");
  ro.index_cache = a.str("index-cache", workdir + "/codebase.idx");
  ro.tel = &tel;
  ro.gguf = a.str("gguf", std::getenv("SLM_GGUF") ? std::getenv("SLM_GGUF") : "");
  ro.gguf_ctx = static_cast<int>(a.integer("gguf-ctx", 4096));
  ro.gguf_threads = static_cast<int>(a.integer("gguf-threads", 0));
  ro.gguf_gpu_layers = static_cast<int>(a.integer("gguf-gpu-layers", 0));
  ro.gguf_kv_q8 = a.flag("kv-q8");
  ro.gguf_base = a.flag("gguf-base");

  AgentRuntime rt;
  if (!rt.init(ro, &err)) {
    std::fprintf(stderr, "agent: %s\n", err.c_str());
    return 1;
  }
  // Non-interactive runs cannot answer an approval dialog, so the policy has to
  // be explicit: --yes to allow, otherwise risky tools are refused, never hung.
  if (a.flag("yes")) {
    rt.policy().write = 1;
    rt.policy().dangerous = 1;
  } else {
    rt.policy().write = 2;
    rt.policy().dangerous = 2;
  }
  rt.policy().timeout_s = a.num("approve-timeout", 60.0);

  if (a.has("index")) {
    const std::string root = a.str("index", ro.workspace);
    std::printf("indexing %s\n", root.c_str());
    std::atomic<bool> stop{false};
    int64_t last = -1;
    if (!rt.index_codebase(
            root,
            [&](int64_t done, int64_t total, const std::string& path) {
              if (total <= 0 || done == last) return;
              last = done;
              std::printf("\r  %lld/%lld  %-56s", static_cast<long long>(done),
                          static_cast<long long>(total),
                          utf8_truncate(path, 56).c_str());
              std::fflush(stdout);
            },
            &stop, &err)) {
      std::fprintf(stderr, "\nindex: %s\n", err.c_str());
      return 1;
    }
    const IndexStats st = rt.codebase().stats();
    std::printf("\r  %lld files, %lld chunks, %lld tokens in %.2f s scan + %.2f s embed"
                "  (%s, %.1f MiB)%20s\n",
                static_cast<long long>(st.files), static_cast<long long>(st.chunks),
                static_cast<long long>(st.tokens), st.scan_seconds, st.embed_seconds,
                st.embedder.c_str(), st.memory_bytes / (1024.0 * 1024.0), "");
    for (const auto& l : st.by_language)
      std::printf("    %-12s %lld chunks\n", l.first.c_str(),
                  static_cast<long long>(l.second));
  }

  // Retrieval on its own, so the index can be inspected without a model in the
  // way: this is how you tell "the model is weak" apart from "retrieval missed".
  if (a.has("search") || a.has("symbol") || a.flag("overview")) {
    if (rt.codebase().empty()) {
      std::fprintf(stderr, "no index - pass --index DIR first\n");
      return 1;
    }
    if (a.flag("overview")) std::printf("%s\n", rt.codebase().overview(4000).c_str());
    const bool sym = a.has("symbol");
    if (sym || a.has("search")) {
      const std::string q = sym ? a.str("symbol") : a.str("search");
      const size_t k = static_cast<size_t>(a.integer("k", 5));
      const std::vector<SearchHit> hits =
          sym ? rt.codebase().find_symbol(q, k) : rt.codebase().search(q, k);
      std::printf("\n%s \"%s\": %zu hits\n", sym ? "find_symbol" : "search",
                  q.c_str(), hits.size());
      for (size_t i = 0; i < hits.size(); ++i) {
        const CodeChunk* c = rt.codebase().chunk(hits[i].chunk);
        if (!c) continue;
        std::printf("  %zu. %-52s score %.3f  bm25 %.2f  cos %.3f  [%s]\n", i + 1,
                    c->header().c_str(), hits[i].score, hits[i].bm25, hits[i].dense,
                    hits[i].why.c_str());
      }
    }
    if (!a.has("ask") && a.positional().empty()) return 0;
  }

  AskRequest req;
  const std::string mode = a.str("mode", "fast");
  if (mode == "strong" || mode == "olmo") req.mode = AskMode::kStrong;
  else if (mode == "debate" || mode == "both") req.mode = AskMode::kDebate;
  else if (mode == "self" || mode == "self-debate") req.mode = AskMode::kSelfDebate;
  else req.mode = AskMode::kFast;
  req.fast_multiplier = static_cast<int>(a.integer("fast-mult", 2));
  req.strong_multiplier = static_cast<int>(a.integer("strong-mult", 1));
  req.voices = static_cast<int>(a.integer("voices", 2));
  req.use_tools = !a.flag("no-tools");
  req.use_codebase = !a.flag("no-codebase");
  req.max_tool_steps = static_cast<int>(a.integer("tool-steps", 3));
  req.max_tokens = static_cast<int>(a.integer("max-new", 320));
  req.seed = static_cast<uint64_t>(a.integer("seed", 0));
  if (a.has("system")) req.system_prompt = a.str("system");

  std::printf("\nmode: %s\n", ask_mode_name(req.mode));
  for (const std::string& l : rt.status_lines()) std::printf("  %s\n", l.c_str());
  std::printf("  tools: ");
  for (const ToolSpec& s : rt.tools().specs()) std::printf("%s ", s.name.c_str());
  std::printf("\n");

  auto run_one = [&](const std::string& question) {
    req.question = question;
    std::atomic<bool> cancel{false};
    AskObserver obs;
    if (!a.flag("quiet") && !(req.mode == AskMode::kDebate ||
                              req.mode == AskMode::kSelfDebate)) {
      obs.on_text = [](const std::string& piece) {
        std::printf("%s", piece.c_str());
        std::fflush(stdout);
      };
    }
    obs.on_tool = [](const ToolTrace& t) {
      std::printf("\n  [tool %s %s -> %s in %.2fs]\n", t.tool.c_str(), t.args.c_str(),
                  t.denied ? "denied" : (t.ok ? "ok" : "failed"), t.seconds);
    };
    int last_round = -1;
    if (req.mode == AskMode::kDebate || req.mode == AskMode::kSelfDebate) {
      obs.on_debate = [&](const DebateTranscript& tr) {
        if (static_cast<int>(tr.rounds.size()) == last_round) return;
        last_round = static_cast<int>(tr.rounds.size());
        if (tr.rounds.empty()) return;
        const DebateRound& r = tr.rounds.back();
        std::printf("  [%3.0f%%] round %d (%s): %zu answers in %.2fs\n",
                    100.0 * tr.progress, r.index, r.kind.c_str(), r.answers.size(),
                    r.seconds);
      };
    }
    const AskResult res = rt.ask(req, &cancel, obs);
    if (!res.error.empty()) {
      std::fprintf(stderr, "\nerror: %s\n", res.error.c_str());
      return;
    }
    if (res.was_debate && a.flag("transcript")) {
      std::printf("\n--- transcript ---\n");
      for (const DebateRound& r : res.debate.rounds) {
        std::printf("\n[round %d %s]%s\n", r.index, r.kind.c_str(),
                    r.note.empty() ? "" : ("  " + r.note).c_str());
        for (const DebateAnswer& ans : r.answers) {
          std::printf("  %s#%d  score %.3f (agree %.2f, judge %.1f/10, %d tok, "
                      "%d reused, %.2fs)\n",
                      ans.participant.c_str(), ans.draft, ans.score, ans.cluster_mass,
                      ans.judge_score, ans.gen_tokens, ans.reused_tokens, ans.seconds);
          if (!ans.critique.empty())
            std::printf("    critique: %s\n", utf8_truncate(ans.critique, 300).c_str());
          std::printf("    %s\n", utf8_truncate(ans.text, 400).c_str());
        }
      }
      std::printf("\n%s", res.debate.decision_log.c_str());
      for (const DebateTranscript::Usage& u : res.debate.usage)
        std::printf("  usage %-8s %d calls, %d prompt (%d reused), %d generated, %.2fs\n",
                    u.backend_id.c_str(), u.calls, u.prompt_tokens, u.reused_tokens,
                    u.gen_tokens, u.seconds);
    }
    if (res.was_debate) std::printf("\n--- answer ---\n%s\n", res.answer.c_str());
    else std::printf("\n");
    std::printf("\n[%.2fs, %d prompt tokens (%d reused from cache), %d generated%s]\n",
                res.seconds, res.prompt_tokens, res.reused_tokens, res.gen_tokens,
                res.was_debate ? (res.debate.escalated ? ", escalated"
                                                       : ", cheap path only")
                               : "");
    if (!res.tools.empty()) {
      std::printf("tools used:");
      for (const ToolTrace& t : res.tools) std::printf(" %s", t.tool.c_str());
      std::printf("\n");
    }
  };

  if (a.has("ask")) {
    run_one(a.str("ask"));
    return 0;
  }
  if (!a.positional().empty()) {
    std::string q;
    for (const std::string& w : a.positional()) q += (q.empty() ? "" : " ") + w;
    run_one(q);
    return 0;
  }
  std::printf("\ninteractive - empty line quits\n");
  std::string line;
  while (true) {
    std::printf("\n> ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line) || line.empty()) break;
    run_one(line);
  }
  return 0;
}

// -------------------------------------------------------------------- memory
int cmd_memory(const Args& a) {
  MemoryStore mem;
  const std::string file = a.str("file", "memory.jsonl");
  if (!mem.open(file)) {
    std::fprintf(stderr, "memory: cannot open %s\n", file.c_str());
    return 1;
  }
  const std::vector<std::string>& pos = a.positional();
  const std::string sub = pos.empty() ? "list" : pos[0];
  auto rest = [&](size_t from) {
    std::string t;
    for (size_t i = from; i < pos.size(); ++i) {
      if (i > from) t += " ";
      t += pos[i];
    }
    return t;
  };

  if (sub == "add") {
    const std::string text = a.has("text") ? a.str("text") : rest(1);
    if (text.empty()) {
      std::fprintf(stderr, "usage: slm memory add \"fact\" [--tags t] [--importance 2]\n");
      return 2;
    }
    const int64_t id = mem.add(text, a.str("tags"), static_cast<float>(a.num("importance", 1.0)));
    std::printf("remembered #%lld (%zu memories in %s)\n", static_cast<long long>(id),
                mem.size(), file.c_str());
    return 0;
  }
  if (sub == "forget") {
    const int64_t id = a.integer("id", pos.size() > 1 ? std::stoll(pos[1]) : 0);
    std::printf(mem.forget(id) ? "forgot #%lld\n" : "no memory #%lld\n",
                static_cast<long long>(id));
    return 0;
  }
  if (sub == "search") {
    const std::string q = a.has("query") ? a.str("query") : rest(1);
    const std::vector<MemoryStore::Hit> hits =
        mem.search(q, static_cast<int>(a.integer("k", 5)));
    for (const MemoryStore::Hit& h : hits)
      std::printf("  %.3f  #%-4lld %s%s\n", h.score, static_cast<long long>(h.item.id),
                  h.item.tags.empty() ? "" : ("[" + h.item.tags + "] ").c_str(),
                  utf8_truncate(h.item.text, 160).c_str());
    if (hits.empty()) std::printf("  (nothing stored yet)\n");
    return 0;
  }
  if (sub == "context") {
    const std::string q = a.has("query") ? a.str("query") : rest(1);
    const std::string block = mem.context_block(q, static_cast<int>(a.integer("k", 3)));
    std::printf("%s", block.empty() ? "(no relevant memory)\n" : block.c_str());
    return 0;
  }
  // list
  std::printf("%zu memories in %s\n", mem.size(), file.c_str());
  for (const MemoryItem& it : mem.all())
    std::printf("  #%-4lld imp %.1f used %-3lld taught %-3lld %s%s\n",
                static_cast<long long>(it.id), it.importance,
                static_cast<long long>(it.uses), static_cast<long long>(it.taught),
                it.tags.empty() ? "" : ("[" + it.tags + "] ").c_str(),
                utf8_truncate(it.text, 140).c_str());
  return 0;
}

// --------------------------------------------------------------------- plan
// Memory / compute planner.  The point of this command is to replace wishful
// thinking with arithmetic before anyone starts a run.
int cmd_plan(const Args& a) {
  Config c;
  if (a.has("config")) c.load(a.str("config"));
  a.apply_sets(&c);
  GPTConfig g = GPTConfig::from_config(c);

  double P = a.num("params", 0.0);
  if (P <= 0.0) P = static_cast<double>(g.param_count());
  const double experts = std::max(1.0, a.num("experts", 1.0));
  const double topk = std::min(experts, std::max(1.0, a.num("topk", 1.0)));
  const double expert_share = a.num("expert-share", 0.67);  // FFN share of params
  const double active =
      experts > 1.0 ? P * (1.0 - expert_share) + P * expert_share * topk / experts : P;

  const int64_t ctx = a.integer("ctx", g.block_size);
  const int64_t batch = a.integer("batch", 1);
  // When only --params is given, estimate a plausible shape (aspect ratio
  // dim/layers ~ 128, P ~ 12 * L * D^2) so the activation and KV-cache numbers
  // below are not silently computed for a tiny model.
  double layers = a.num("layers", 0.0);
  double dim = a.num("dim", 0.0);
  bool estimated = false;
  if (layers <= 0.0 || dim <= 0.0) {
    if (a.has("params")) {
      layers = std::max(2.0, std::cbrt(P / 196608.0));
      dim = 128.0 * layers;
      estimated = true;
    } else {
      layers = static_cast<double>(g.n_layer);
      dim = static_cast<double>(g.n_embd);
    }
  }
  const double heads = a.num("heads", std::max(1.0, std::round(dim / 128.0)));
  const double kv_heads = a.num("kv-heads", heads);
  const double head_dim = dim / std::max(1.0, heads);

  const double ram = a.num("ram", 16.0) * 1073741824.0;
  const double vram = a.num("vram", 2.0) * 1073741824.0;
  const double gpu = a.num("gpu-mem", 80.0) * 1073741824.0;
  const double flops = a.num("throughput", 4e14);  // achieved FLOP/s of one accelerator
  const double trainable_frac = std::min(1.0, std::max(0.0, a.num("trainable", 1.0)));

  auto human = [](double b) { return human_bytes(b); };
  std::printf("parameters\n");
  std::printf("  total            : %.4g  (%.2f B)\n", P, P / 1e9);
  if (experts > 1.0) {
    std::printf("  MoE              : %.0f experts, top-%.0f routing, %.0f%% of params in experts\n",
                experts, topk, 100.0 * expert_share);
    std::printf("  active per token : %.4g  (%.2f B, %.1f%% of total)\n", active, active / 1e9,
                100.0 * active / P);
  }
  const double trainable = P * trainable_frac;
  std::printf("  trainable        : %.4g  (%.0f%%)\n", trainable, 100.0 * trainable_frac);
  std::printf("  shape%s          : %.0f layers x %.0f dim, %.0f heads (%.0f kv heads), ctx %lld\n",
              estimated ? " (est.)" : "      ", layers, dim, heads, kv_heads,
              static_cast<long long>(ctx));

  std::printf("\nweights\n");
  std::printf("  fp32             : %s\n", human(P * 4).c_str());
  std::printf("  fp16/bf16        : %s\n", human(P * 2).c_str());
  // These are the exact figures of the .slmq format: group of 64 with one f16
  // scale -> 8.25 and 4.25 bits per weight.
  std::printf("  int8 (group 64)  : %s   (8.25 bits/weight, slm pack --bits 8)\n",
              human(P * 8.25 / 8.0).c_str());
  std::printf("  int4 (group 64)  : %s   (4.25 bits/weight, slm pack --bits 4)\n",
              human(P * 4.25 / 8.0).c_str());

  const double grads = trainable * 2;          // bf16 grads
  const double adam = trainable * 8;           // fp32 m+v
  const double adam8 = trainable * 2;          // 8-bit optimiser states
  const double act_full = static_cast<double>(batch) * ctx * dim * layers * 12.0 * 2.0;
  const double act_ckpt = static_cast<double>(batch) * ctx * dim * (layers + 12.0) * 2.0;
  const double kv = 2.0 * layers * kv_heads * head_dim * static_cast<double>(ctx) *
                    static_cast<double>(batch) * 2.0;
  std::printf("\ntraining state (bf16 weights, %s)\n", trainable_frac < 1.0 ? "partial fine-tune" : "full");
  std::printf("  weights          : %s\n", human(P * 2).c_str());
  std::printf("  gradients        : %s\n", human(grads).c_str());
  std::printf("  AdamW fp32       : %s   (8-bit: %s)\n", human(adam).c_str(), human(adam8).c_str());
  std::printf("  activations      : %s   (checkpointing: %s)\n", human(act_full).c_str(),
              human(act_ckpt).c_str());
  const double train_min = P * 2 + grads + adam8 + act_ckpt;
  const double train_max = P * 2 + grads + adam + act_full;
  std::printf("  total            : %s .. %s\n", human(train_min).c_str(), human(train_max).c_str());
  std::printf("\ninference (batch %lld, ctx %lld)\n", static_cast<long long>(batch),
              static_cast<long long>(ctx));
  std::printf("  weights int8     : %s\n", human(P * 8.25 / 8.0).c_str());
  std::printf("  weights int4     : %s\n", human(P * 4.25 / 8.0).c_str());
  std::printf("  KV cache fp16    : %s\n", human(kv).c_str());
  std::printf("  total int8       : %s\n", human(P * 8.25 / 8.0 + kv).c_str());
  std::printf("  total int4       : %s   <- what `slm qrun` actually maps\n",
              human(P * 4.25 / 8.0 + kv).c_str());

  std::printf("\nverdict for your machine (--ram %.0f GiB, --vram %.0f GiB)\n", ram / 1073741824.0,
              vram / 1073741824.0);
  auto verdict = [](bool ok) { return ok ? "FITS" : "does NOT fit"; };
  std::printf("  train (min config) in RAM   : %-13s (needs %s)\n", verdict(train_min <= ram),
              human(train_min).c_str());
  std::printf("  train (min config) in VRAM  : %-13s (needs %s)\n", verdict(train_min <= vram),
              human(train_min).c_str());
  std::printf("  int8 inference in RAM       : %-13s (needs %s)\n",
              verdict(P * 1.0625 + kv <= ram), human(P * 1.0625 + kv).c_str());
  std::printf("  int8 inference in VRAM      : %-13s (needs %s)\n",
              verdict(P * 1.0625 + kv <= vram), human(P * 1.0625 + kv).c_str());
  const double per_gpu_train = train_min;
  const int gpus_train = static_cast<int>(std::ceil(per_gpu_train / gpu));
  const int gpus_infer = static_cast<int>(std::ceil((P * 2 + kv) / gpu));
  std::printf("  accelerators of %.0f GiB     : %d for sharded training, %d for fp16 inference\n",
              gpu / 1073741824.0, gpus_train, gpus_infer);

  const double tokens = a.num("tokens", 20.0 * P);  // Chinchilla-ish
  const double train_flops = 6.0 * active * tokens;
  const double secs = train_flops / std::max(1.0, flops);
  std::printf("\ncompute for a from-scratch run\n");
  std::printf("  tokens (20x params): %.3g\n", tokens);
  std::printf("  training FLOPs     : %.3g   (6 x active params x tokens)\n", train_flops);
  std::printf("  one accelerator at %.2g FLOP/s: %.3g s  =  %.3g GPU-years\n", flops, secs,
              secs / (3600.0 * 24.0 * 365.0));
  std::printf("  1000 accelerators  : %.3g days\n", secs / 1000.0 / 86400.0);
  const double cpu_flops = 5.8e10;  // measured for the bundled native backend
  std::printf("  this CPU backend (%.1e FLOP/s): %.3g years\n", cpu_flops,
              train_flops / cpu_flops / (3600.0 * 24.0 * 365.0));

  const double budget = std::min(ram, vram > 0 ? std::max(ram, vram) : ram);
  const double fit_params_train = budget / (2 + 2 + 2);  // bf16 w + grads + 8-bit adam
  const double fit_params_infer = budget / 1.0625;
  std::printf("\nwhat actually fits here\n");
  std::printf("  full training      : up to ~%.2f B parameters (bf16 + 8-bit optimiser,\n"
              "                       checkpointing, micro-batch 1)\n", fit_params_train / 1e9);
  std::printf("  int8 inference     : up to ~%.2f B parameters\n", fit_params_infer / 1e9);
  std::printf("  partial fine-tune  : a %.2f B model with 5%% trainable needs %s\n",
              fit_params_infer / 1e9,
              human(fit_params_infer * 2 + fit_params_infer * 0.05 * 4 + act_ckpt).c_str());
  return 0;
}

// --------------------------------------------------------------------- bench
int cmd_bench(const Args& a) {
  Config c;
  if (a.has("config")) c.load(a.str("config"));
  a.apply_sets(&c);
  backend_init(static_cast<int>(a.integer("threads", 0)), a.flag("cuda"));
  GPTConfig g = GPTConfig::from_config(c);
  if (a.has("ctx")) g.block_size = a.integer("ctx", g.block_size);
  if (a.has("checkpointing")) g.grad_checkpointing = a.flag("checkpointing");
  GPT model(g);
  model.init_weights(7);
  const int64_t B = a.integer("batch", 4);
  const int64_t T = g.block_size;
  const int64_t iters = a.integer("iters", 3);
  std::vector<int32_t> ids(static_cast<size_t>(B * T)), tg(static_cast<size_t>(B * T));
  Rng rng(5);
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i] = static_cast<int32_t>(rng.below(static_cast<uint64_t>(g.vocab_size)));
    tg[i] = static_cast<int32_t>(rng.below(static_cast<uint64_t>(g.vocab_size)));
  }
  std::printf("%s\nbackend %s, batch %lld, ctx %lld\n", g.describe().c_str(),
              backend_name(), static_cast<long long>(B), static_cast<long long>(T));
  AdamW opt;
  opt.set_params(model.trainable_params(), model.trainable_names());
  double fwd = 0.0, full = 0.0;
  for (int64_t i = 0; i < iters; ++i) {
    double t0 = now_seconds();
    { model.eval_loss(ids, tg, B, T); }
    fwd += now_seconds() - t0;
    t0 = now_seconds();
    model.zero_grad();
    Tensor l = model.loss(ids, tg, B, T, nullptr, nullptr);
    l.backward();
    opt.step(1e-4f);
    full += now_seconds() - t0;
  }
  const double tok = static_cast<double>(B * T * iters);
  std::printf("forward      : %.3f s/iter, %.0f tok/s\n", fwd / static_cast<double>(iters),
              tok / fwd);
  std::printf("fwd+bwd+step : %.3f s/iter, %.0f tok/s\n", full / static_cast<double>(iters),
              tok / full);
  std::printf("peak tensor memory: %s\n",
              human_bytes(static_cast<double>(backend_allocated_bytes())).c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 2;
  }
  const std::string cmd = argv[1];
  Args a(argc - 2, argv + 2);
  try {
    if (cmd == "up") return cmd_up(a);
    if (cmd == "agent") return cmd_agent(a);
    if (cmd == "info") return cmd_info(a);
    if (cmd == "tokenizer") return cmd_tokenizer(a);
    if (cmd == "pretrain") return cmd_pretrain(a);
    if (cmd == "chat") return cmd_chat(a);
    if (cmd == "eval") return cmd_eval(a);
    if (cmd == "langcheck") return cmd_langcheck(a);
    if (cmd == "quantize") return cmd_quantize(a);
    if (cmd == "pack") return cmd_pack(a);
    if (cmd == "qrun" || cmd == "qchat") return cmd_qrun(a);
    if (cmd == "qbench") return cmd_qbench(a);
    if (cmd == "qeval") return cmd_qeval(a);
    if (cmd == "bench") return cmd_bench(a);
    if (cmd == "plan") return cmd_plan(a);
    if (cmd == "tokenize") return cmd_tokenize(a);
    if (cmd == "memory") return cmd_memory(a);
    if (cmd == "live" || cmd == "dashboard") {
      AppOptions o;
      if (a.has("config")) o.cfg.load(a.str("config"));
      a.apply_sets(&o.cfg);
      o.ckpt = a.str("ckpt");
      o.tokenizer = a.str("tokenizer");
      o.data = a.has("mix") ? a.str("mix") : a.str("data");
      o.workdir = a.str("workdir", "runs");
      o.headless = (cmd == "live") || a.flag("headless");
      o.seconds = a.num("seconds", 0.0);
      o.threads = static_cast<int>(a.integer("threads", 0));
      o.cuda = a.flag("cuda");
      o.seed = static_cast<uint64_t>(a.integer("seed", 1234));
      o.autopilot = a.flag("autopilot");
      return run_self_training(o);
    }
    if (cmd == "-h" || cmd == "--help" || cmd == "help") {
      print_usage();
      return 0;
    }
    std::fprintf(stderr, "unknown command '%s'\n\n", cmd.c_str());
    print_usage();
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\nerror: %s\n", e.what());
    return 1;
  }
}
