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
//   slm bench                   forward+backward throughput
//   slm live                    the full self-training system, terminal dashboard
//   slm dashboard               the full self-training system, ImGui dashboard
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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
#include "model.h"
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

std::string human_bytes(double b) {
  const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  int i = 0;
  while (b >= 1024.0 && i < 4) {
    b /= 1024.0;
    ++i;
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f %s", b, u[i]);
  return buf;
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
      "  info          [--config F]                       environment + memory budget\n"
      "  tokenizer     --out F [--vocab 4096]             train a byte level BPE\n"
      "                (--input F | --mix fa=F:0.4,en=F:0.3,py=F:0.3) [--no-normalize]\n"
      "  pretrain      --tokenizer F --out F              base training\n"
      "                (--data F | --mix fa=F:0.4,en=F:0.3,py=F:0.3)\n"
      "                [--config F] [--steps N] [--batch N] [--ctx N] [--lr X]\n"
      "                [--accum N] [--eval-every N] [--save-every N] [--dtype f16]\n"
      "                [--resume F] [--threads N] [--seed N]\n"
      "  chat          --ckpt F --tokenizer F [--prompt S] [--max-new N] [--temp X]\n"
      "  eval          --ckpt F --tokenizer F (--data F | --mix ...) [--batch N]\n"
      "  langcheck     --ckpt F --tokenizer F --mix ...   per language quality +\n"
      "                interference (code switching) + python validity\n"
      "  plan          --params 7e12 [--experts N --topk N --ctx N --ram 16 --vram 2]\n"
      "                memory / compute planner for very large configurations\n"
      "  quantize      --in F --out F [--dtype q8|f16|f32]\n"
      "  bench         [--config F] [--batch N] [--ctx N] [--iters N]\n"
      "  live          --ckpt F --tokenizer F --data F [--config F] [--seconds N]\n"
      "  dashboard     --ckpt F --tokenizer F --data F [--config F]\n"
      "  gradcheck                                        (see slm_gradcheck)\n");
}

// --------------------------------------------------------------------- info
int cmd_info(const Args& a) {
  Config c;
  if (a.has("config")) c.load(a.str("config"));
  a.apply_sets(&c);
  GPTConfig g = GPTConfig::from_config(c);
  std::printf("backend            : %s\n", backend_name());
  std::printf("gpu                : %s\n", backend_on_gpu() ? "yes" : "no");
  std::printf("model              : %s\n", g.describe().c_str());
  const double p = static_cast<double>(g.param_count());
  std::printf("\nmemory budget (training, one replica)\n");
  std::printf("  weights f32      : %s\n", human_bytes(p * 4).c_str());
  std::printf("  weights f16      : %s\n", human_bytes(p * 2).c_str());
  std::printf("  weights q8       : %s\n", human_bytes(p * 1.06).c_str());
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
    std::printf("resumed from %s at step %lld\n", a.str("resume").c_str(),
                static_cast<long long>(meta.step));
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
  const int64_t eval_every = a.integer("eval-every", std::max<int64_t>(20, steps / 10));
  const int64_t save_every = a.integer("save-every", 0);
  Dtype dt = Dtype::F32;
  parse_dtype(a.str("dtype", "f32"), &dt);
  const std::string out = a.str("out", "model.slm");

  double t_start = now_seconds();
  int64_t tokens_done = 0;
  float best_val = 1e30f;
  for (int64_t step = 0; step < steps; ++step) {
    model.zero_grad();
    float loss_sum = 0.0f;
    for (int64_t k = 0; k < accum; ++k) {
      Batch b = ds.sample_batch(B, T, rng);
      float lv = 0.0f;
      Tensor l = model.loss(b.ids, b.targets, B, T, &lv, nullptr);
      // scale so that accumulated gradients average
      Tensor scaled = l.scale(1.0f / static_cast<float>(accum));
      scaled.backward();
      loss_sum += lv;
      tokens_done += B * T;
    }
    const float lr = lr_schedule(step, oc.lr, warmup, steps, 0.1f);
    const float gn = opt.step(lr);
    const float loss = loss_sum / static_cast<float>(accum);
    if (step % std::max<int64_t>(1, a.integer("log-every", 10)) == 0 || step + 1 == steps) {
      const double dt_s = now_seconds() - t_start;
      std::printf("step %5lld | loss %.4f | ppl %8.2f | lr %.2e | gnorm %.3f | %6.0f tok/s | mem %s\n",
                  static_cast<long long>(step), loss, std::exp(std::min(20.0f, loss)),
                  lr, gn, static_cast<double>(tokens_done) / std::max(1e-6, dt_s),
                  human_bytes(static_cast<double>(backend_allocated_bytes())).c_str());
      std::fflush(stdout);
    }
    if (!val.empty() && ((step + 1) % eval_every == 0 || step + 1 == steps)) {
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
      g.write_to(m.extra);
      m.extra.set("tokenizer", a.str("tokenizer"));
      save_checkpoint(out, *model.snapshot(), m, dt);
    }
  }
  CheckpointMeta m;
  m.step = steps;
  m.tokens_seen = tokens_done;
  g.write_to(m.extra);
  m.extra.set("tokenizer", a.str("tokenizer"));
  m.extra.set("train.holdout_loss", std::to_string(best_val));
  m.extra.set("train.mixture", a.has("mix") ? a.str("mix") : a.str("data"));
  if (!save_checkpoint(out, *model.snapshot(), m, dt)) {
    std::fprintf(stderr, "pretrain: cannot write %s\n", out.c_str());
    return 1;
  }
  std::printf("\nsaved %s (%s, %s)\n", out.c_str(), dtype_name(dt),
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
    const std::string prompt = "<|user|>" + line + "<|assistant|>";
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
    const std::vector<int32_t>& toks = ds.data(si).tokens();
    const int64_t limit = ds.data(si).train_tokens();
    std::vector<std::vector<int32_t>> prompts;
    for (int64_t i = 0; i < limit && static_cast<int>(prompts.size()) < 256; ++i) {
      if (toks[static_cast<size_t>(i)] != Tokenizer::kUser) continue;
      int64_t j = i + 1;
      while (j < limit && toks[static_cast<size_t>(j)] != Tokenizer::kAssistant && j - i < 64) ++j;
      if (j >= limit || toks[static_cast<size_t>(j)] != Tokenizer::kAssistant) continue;
      prompts.emplace_back(toks.begin() + i, toks.begin() + j + 1);
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
  std::printf("  int8 (group 64)  : %s\n", human(P * 1.0625).c_str());
  std::printf("  int4 (group 64)  : %s\n", human(P * 0.5625).c_str());

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
  std::printf("  weights int8     : %s\n", human(P * 1.0625).c_str());
  std::printf("  KV cache fp16    : %s\n", human(kv).c_str());
  std::printf("  total int8       : %s\n", human(P * 1.0625 + kv).c_str());

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
    if (cmd == "info") return cmd_info(a);
    if (cmd == "tokenizer") return cmd_tokenizer(a);
    if (cmd == "pretrain") return cmd_pretrain(a);
    if (cmd == "chat") return cmd_chat(a);
    if (cmd == "eval") return cmd_eval(a);
    if (cmd == "langcheck") return cmd_langcheck(a);
    if (cmd == "quantize") return cmd_quantize(a);
    if (cmd == "bench") return cmd_bench(a);
    if (cmd == "plan") return cmd_plan(a);
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
