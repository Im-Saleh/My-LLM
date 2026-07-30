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

void print_usage() {
  std::puts(
      "slm - a small language model with a hybrid self-training pipeline\n"
      "\n"
      "usage: slm <command> [options]\n"
      "\n"
      "  info          [--config F]                       environment + memory budget\n"
      "  tokenizer     --input F --out F [--vocab 4096]   train a byte level BPE\n"
      "  pretrain      --data F --tokenizer F --out F     base training\n"
      "                [--config F] [--steps N] [--batch N] [--ctx N] [--lr X]\n"
      "                [--accum N] [--eval-every N] [--save-every N] [--dtype f16]\n"
      "                [--resume F] [--threads N] [--seed N]\n"
      "  chat          --ckpt F --tokenizer F [--prompt S] [--max-new N] [--temp X]\n"
      "  eval          --ckpt F --tokenizer F --data F [--batch N] [--ctx N]\n"
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
  const std::string in = a.str("input");
  const std::string out = a.str("out", "tokenizer.slmtok");
  const int32_t vocab = static_cast<int32_t>(a.integer("vocab", 4096));
  if (in.empty()) {
    std::fprintf(stderr, "tokenizer: --input is required\n");
    return 2;
  }
  bool ok = false;
  const std::string text = read_file(in, &ok);
  if (!ok || text.empty()) {
    std::fprintf(stderr, "tokenizer: cannot read %s\n", in.c_str());
    return 1;
  }
  Tokenizer tok;
  const double t0 = now_seconds();
  tok.train(text, vocab, static_cast<int>(a.integer("min-freq", 2)),
            [](int32_t done, int32_t total) {
              if (total > 0 && done % 256 == 0)
                std::printf("\r  merges %d/%d", done, total), std::fflush(stdout);
            });
  std::printf("\rtrained %zu merges, vocab=%d in %.1fs\n", tok.num_merges(),
              tok.vocab_size(), now_seconds() - t0);
  if (!tok.save(out)) {
    std::fprintf(stderr, "tokenizer: cannot write %s\n", out.c_str());
    return 1;
  }
  const std::vector<int32_t> ids = tok.encode(text.substr(0, 4000));
  std::printf("saved %s   (compression on sample: %.2f bytes/token)\n", out.c_str(),
              ids.empty() ? 0.0 : 4000.0 / static_cast<double>(ids.size()));
  const std::string sample = "Hello world! سلام دنیا.";
  const std::vector<int32_t> rt = tok.encode(sample);
  std::printf("roundtrip: \"%s\" -> %zu tokens -> \"%s\"  [%s]\n", sample.c_str(),
              rt.size(), tok.decode(rt).c_str(),
              tok.decode(rt) == sample ? "exact" : "MISMATCH");
  return tok.decode(rt) == sample ? 0 : 1;
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

  TokenDataset ds;
  std::string err;
  if (!ds.load_text_file(a.str("data"), tok, &err)) {
    std::fprintf(stderr, "pretrain: %s\n", err.c_str());
    return 1;
  }
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
  std::printf("corpus: %zu tokens (train %lld / holdout %lld)\n", ds.num_tokens(),
              static_cast<long long>(ds.train_tokens()),
              static_cast<long long>(ds.holdout_tokens()));
  std::printf("optimiser state: %s\n", human_bytes(static_cast<double>(opt.state_bytes())).c_str());
  std::printf("training %lld steps, batch %lld x %lld tokens, accum %lld\n\n",
              static_cast<long long>(steps), static_cast<long long>(B),
              static_cast<long long>(T), static_cast<long long>(accum));

  const std::vector<Batch> val = ds.holdout_batches(B, T, 2);
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
      float vl = 0.0f;
      for (const Batch& vb : val) vl += model.eval_loss(vb.ids, vb.targets, vb.B, vb.T);
      vl /= static_cast<float>(val.size());
      best_val = std::min(best_val, vl);
      std::printf("        >> holdout loss %.4f  (ppl %.2f)%s\n", vl, std::exp(std::min(20.0f, vl)),
                  vl <= best_val ? "  *best" : "");
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
  TokenDataset ds;
  std::string err;
  if (!ds.load_text_file(a.str("data"), tok, &err)) {
    std::fprintf(stderr, "eval: %s\n", err.c_str());
    return 1;
  }
  const int64_t B = a.integer("batch", 4);
  const int64_t T = std::min<int64_t>(a.integer("ctx", model->config().block_size),
                                      model->config().block_size);
  const std::vector<Batch> vb = ds.holdout_batches(B, T, static_cast<int>(a.integer("batches", 8)));
  if (vb.empty()) {
    std::fprintf(stderr, "eval: hold-out too small\n");
    return 1;
  }
  double sum = 0.0;
  for (const Batch& b : vb) sum += model->eval_loss(b.ids, b.targets, b.B, b.T);
  const double loss = sum / static_cast<double>(vb.size());
  std::printf("batches %zu  loss %.4f  perplexity %.2f\n", vb.size(), loss, std::exp(loss));
  return 0;
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
    if (cmd == "quantize") return cmd_quantize(a);
    if (cmd == "bench") return cmd_bench(a);
    if (cmd == "live" || cmd == "dashboard") {
      AppOptions o;
      if (a.has("config")) o.cfg.load(a.str("config"));
      a.apply_sets(&o.cfg);
      o.ckpt = a.str("ckpt");
      o.tokenizer = a.str("tokenizer");
      o.data = a.str("data");
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
