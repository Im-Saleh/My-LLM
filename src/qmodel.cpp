// SPDX-License-Identifier: Apache-2.0
#include "qmodel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "core/rng.h"
#include "core/serialize.h"

#if defined(__unix__) || defined(__APPLE__)
#define SLM_HAVE_MMAP 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace slm {
namespace {

constexpr char kQMagic[8] = {'S', 'L', 'M', 'Q', '0', '0', '1', '\0'};
constexpr int64_t kAlign = 64;

int64_t align_up(int64_t x, int64_t a) { return ((x + a - 1) / a) * a; }

// A quantised type needs whole groups along the reduction axis; anything else
// falls back to f16, which has no alignment requirement.
QType pick_type(QType want, int64_t K) {
  if (qtype_is_quantised(want) && K % kQBlock != 0) return QType::F16;
  return want;
}

enum class Init : uint8_t { kRandn = 0, kOnes = 1, kZeros = 2 };

struct PlanEntry {
  std::string name;
  QType type = QType::F32;
  int64_t rows = 0, cols = 0;   // stored shape: rows of `cols` values
  bool transposed = false;      // training layout is [cols, rows]
  int64_t row_bytes = 0, bytes = 0, offset = 0;
  Init init = Init::kRandn;     // only used by the synthesiser
  float std = 0.02f;
};

struct Plan {
  std::vector<PlanEntry> e;
  int64_t payload_offset = 0;
  int64_t total_bytes = 0;
  int64_t params = 0;
  int64_t quant_params = 0;  // parameters stored in q4/q8
};

void add_1d(Plan* p, const std::string& name, int64_t n, Init init) {
  PlanEntry e;
  e.name = name;
  e.type = QType::F32;  // norms and biases are tiny and precision-critical
  e.rows = 1;
  e.cols = n;
  e.init = init;
  p->e.push_back(e);
}

void add_2d(Plan* p, const std::string& name, int64_t out, int64_t in, QType want,
            bool transposed, Init init, float std) {
  PlanEntry e;
  e.name = name;
  e.type = pick_type(want, in);
  e.rows = out;
  e.cols = in;
  e.transposed = transposed;
  e.init = init;
  e.std = std;
  p->e.push_back(e);
}

// Mirrors the parameter registration of GPT's constructor exactly; the reader
// rebuilds the same plan and checks the file against it, so a mismatch between
// the two is a load error rather than silent garbage.
Plan build_plan(const GPTConfig& cfg, const QPackOptions& o) {
  Plan p;
  const int64_t C = cfg.n_embd;
  const int64_t Dh = cfg.head_dim();
  const int64_t KV = cfg.kv_heads() * Dh;
  const int64_t Hf = cfg.hidden_dim();
  const bool rms = cfg.norm == NormKind::kRMSNorm;
  const float resid = cfg.init_std / std::sqrt(2.0f * static_cast<float>(cfg.n_layer));

  // [vocab, C]: rows are token vectors *and* the tied output projection, so the
  // stored orientation is the same as the training one.
  add_2d(&p, "tok_emb", cfg.vocab_size, C, o.embed_type, false, Init::kRandn,
         cfg.init_std);
  if (cfg.pos == PosKind::kLearned)
    add_2d(&p, "pos_emb", cfg.block_size, C, QType::F32, false, Init::kRandn,
           cfg.init_std * 0.5f);

  for (int64_t l = 0; l < cfg.n_layer; ++l) {
    const std::string b = "h." + std::to_string(l) + ".";
    add_1d(&p, b + "n1.g", C, Init::kOnes);
    if (!rms) add_1d(&p, b + "n1.b", C, Init::kZeros);
    add_2d(&p, b + "attn.qkv.w", C + 2 * KV, C, o.type, true, Init::kRandn,
           cfg.init_std);
    if (cfg.linear_bias) add_1d(&p, b + "attn.qkv.b", C + 2 * KV, Init::kZeros);
    if (cfg.qk_norm) {
      add_1d(&p, b + "attn.qnorm.g", Dh, Init::kOnes);
      add_1d(&p, b + "attn.knorm.g", Dh, Init::kOnes);
    }
    add_2d(&p, b + "attn.proj.w", C, C, o.type, true, Init::kRandn, resid);
    if (cfg.linear_bias) add_1d(&p, b + "attn.proj.b", C, Init::kZeros);
    add_1d(&p, b + "n2.g", C, Init::kOnes);
    if (!rms) add_1d(&p, b + "n2.b", C, Init::kZeros);
    if (cfg.ffn == FFNKind::kSwiGLU) {
      add_2d(&p, b + "mlp.gate.w", Hf, C, o.type, true, Init::kRandn, cfg.init_std);
      add_2d(&p, b + "mlp.up.w", Hf, C, o.type, true, Init::kRandn, cfg.init_std);
      add_2d(&p, b + "mlp.down.w", C, Hf, o.type, true, Init::kRandn, resid);
    } else {
      add_2d(&p, b + "mlp.up.w", Hf, C, o.type, true, Init::kRandn, cfg.init_std);
      if (cfg.linear_bias) add_1d(&p, b + "mlp.up.b", Hf, Init::kZeros);
      add_2d(&p, b + "mlp.down.w", C, Hf, o.type, true, Init::kRandn, resid);
      if (cfg.linear_bias) add_1d(&p, b + "mlp.down.b", C, Init::kZeros);
    }
  }
  add_1d(&p, "norm_f.g", C, Init::kOnes);
  if (!rms) add_1d(&p, "norm_f.b", C, Init::kZeros);
  if (!cfg.tie_weights)
    add_2d(&p, "lm_head.w", cfg.vocab_size, C, o.embed_type, true, Init::kRandn,
           cfg.init_std);

  for (PlanEntry& e : p.e) {
    e.row_bytes = qrow_bytes(e.type, e.cols);
    e.bytes = e.rows * e.row_bytes;
    p.params += e.rows * e.cols;
    if (qtype_is_quantised(e.type)) p.quant_params += e.rows * e.cols;
  }
  return p;
}

int64_t directory_bytes(const Plan& p) {
  int64_t n = 0;
  for (const PlanEntry& e : p.e) {
    const int ndim = (e.rows == 1) ? 1 : 2;
    n += 2 + static_cast<int64_t>(e.name.size()) + 1 + 1 + 8LL * ndim + 8 + 8;
  }
  return n;
}

void finalise_offsets(Plan* p, int64_t header_len) {
  const int64_t fixed = 8 + 4 + 4 + 8;
  p->payload_offset = align_up(fixed + header_len + directory_bytes(*p), kAlign);
  int64_t off = p->payload_offset;
  for (PlanEntry& e : p->e) {
    e.offset = off;
    off = align_up(off + e.bytes, kAlign);
  }
  p->total_bytes = off;
}

template <typename T>
void put(std::ostream& f, T v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

std::string header_text(const GPTConfig& cfg, const QPackOptions& o,
                        const Plan& plan, const Config& extra) {
  Config c = extra;
  cfg.write_to(c);
  c.set("quant.body", qtype_name(o.type));
  c.set("quant.embed", qtype_name(o.embed_type));
  c.set("quant.group", std::to_string(kQBlock));
  c.set("quant.params", std::to_string(plan.params));
  c.set("quant.tensors", std::to_string(plan.e.size()));
  if (!o.note.empty()) c.set("quant.note", o.note);
  return c.to_string();
}

// Row producer: fills `dst` with `cols` float values of logical row `r`.
using RowFn = std::function<void(const PlanEntry&, int64_t r, float* dst)>;

bool write_plan(const std::string& path, const Plan& plan, const std::string& header,
                const RowFn& rows, bool progress, std::string* err) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    if (err) *err = "cannot open " + path + " for writing";
    return false;
  }
  f.write(kQMagic, sizeof(kQMagic));
  put<uint32_t>(f, static_cast<uint32_t>(header.size()));
  put<uint32_t>(f, static_cast<uint32_t>(plan.e.size()));
  put<uint64_t>(f, static_cast<uint64_t>(plan.payload_offset));
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const PlanEntry& e : plan.e) {
    put<uint16_t>(f, static_cast<uint16_t>(e.name.size()));
    f.write(e.name.data(), static_cast<std::streamsize>(e.name.size()));
    put<uint8_t>(f, static_cast<uint8_t>(e.type));
    const uint8_t ndim = (e.rows == 1) ? 1 : 2;
    put<uint8_t>(f, ndim);
    if (ndim == 2) put<int64_t>(f, e.rows);
    put<int64_t>(f, e.cols);
    put<uint64_t>(f, static_cast<uint64_t>(e.offset));
    put<uint64_t>(f, static_cast<uint64_t>(e.bytes));
  }
  const std::vector<char> pad(static_cast<size_t>(kAlign), 0);
  auto pad_to = [&](int64_t target) {
    int64_t here = static_cast<int64_t>(f.tellp());
    while (here < target) {
      const int64_t n = std::min<int64_t>(target - here, kAlign);
      f.write(pad.data(), static_cast<std::streamsize>(n));
      here += n;
    }
  };
  pad_to(plan.payload_offset);

  const int64_t nthreads = std::max<int64_t>(1,
#ifdef _OPENMP
                                             omp_get_max_threads()
#else
                                             1
#endif
  );
  int64_t done = 0;
  double last_report = -1.0;
  for (size_t i = 0; i < plan.e.size(); ++i) {
    const PlanEntry& e = plan.e[i];
    pad_to(e.offset);
    // Rows are produced and quantised in parallel chunks, then written in order.
    const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(e.rows, 8 * nthreads));
    std::vector<uint8_t> out(static_cast<size_t>(chunk * e.row_bytes));
    std::vector<std::vector<float>> scratch(static_cast<size_t>(nthreads));
    for (auto& s : scratch) s.resize(static_cast<size_t>(e.cols));
    for (int64_t r0 = 0; r0 < e.rows; r0 += chunk) {
      const int64_t n = std::min<int64_t>(chunk, e.rows - r0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 1)
#endif
      for (int64_t j = 0; j < n; ++j) {
        const int tid =
#ifdef _OPENMP
            omp_get_thread_num();
#else
            0;
#endif
        float* buf = scratch[static_cast<size_t>(tid)].data();
        rows(e, r0 + j, buf);
        quantize_row(e.type, buf, e.cols, out.data() + j * e.row_bytes);
      }
      f.write(reinterpret_cast<const char*>(out.data()),
              static_cast<std::streamsize>(n * e.row_bytes));
      if (!f) {
        if (err) *err = "write failed (disk full?)";
        return false;
      }
      done += n * e.cols;
      if (progress) {
        const double frac = static_cast<double>(done) / static_cast<double>(plan.params);
        if (frac - last_report > 0.02) {
          last_report = frac;
          std::fprintf(stderr, "\r  packing %5.1f%%  (%s)", 100.0 * frac,
                       e.name.c_str());
          std::fflush(stderr);
        }
      }
    }
  }
  pad_to(plan.total_bytes);
  if (progress) std::fprintf(stderr, "\r  packing 100.0%%%30s\n", "");
  f.flush();
  if (!f) {
    if (err) *err = "write failed";
    return false;
  }
  return true;
}

// ------------------------------------------------------------- small f32 maths
void rms_norm(const float* x, const float* g, int64_t C, float eps, float* y) {
  double ss = 0.0;
  for (int64_t j = 0; j < C; ++j) ss += static_cast<double>(x[j]) * x[j];
  const float rs =
      1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(C)) + eps);
  for (int64_t j = 0; j < C; ++j) y[j] = x[j] * rs * g[j];
}

void layer_norm(const float* x, const float* g, const float* b, int64_t C, float eps,
                float* y) {
  float m = 0.0f;
  for (int64_t j = 0; j < C; ++j) m += x[j];
  m /= static_cast<float>(C);
  float v = 0.0f;
  for (int64_t j = 0; j < C; ++j) {
    const float d = x[j] - m;
    v += d * d;
  }
  const float rs = 1.0f / std::sqrt(v / static_cast<float>(C) + eps);
  for (int64_t j = 0; j < C; ++j) y[j] = (x[j] - m) * rs * g[j] + b[j];
}

void rope_inplace(float* v, int64_t D, int64_t pos, float theta) {
  const int64_t half = D / 2;
  for (int64_t i = 0; i < half; ++i) {
    const float freq =
        1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(D));
    const float ang = static_cast<float>(pos) * freq;
    const float c = std::cos(ang), s = std::sin(ang);
    const float a = v[2 * i], b = v[2 * i + 1];
    v[2 * i] = a * c - b * s;
    v[2 * i + 1] = a * s + b * c;
  }
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

inline float gelu_tanh(float x) {
  // matches the tensor backend's GELU (tanh approximation)
  const float x3 = x * x * x;
  return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
}

void axpy_f16(float* dst, const uint16_t* src, int64_t n, float a) {
  for (int64_t i = 0; i < n; ++i) dst[i] += a * half_to_float(src[i]);
}

void softmax_inplace(float* s, int64_t n) {
  float mx = s[0];
  for (int64_t i = 1; i < n; ++i) mx = std::max(mx, s[i]);
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    s[i] = std::exp(s[i] - mx);
    sum += s[i];
  }
  const float inv = static_cast<float>(1.0 / std::max(1e-30, sum));
  for (int64_t i = 0; i < n; ++i) s[i] *= inv;
}

}  // namespace

// ================================================================== packing
int64_t qpack_estimate_bytes(const GPTConfig& cfg, const QPackOptions& o) {
  Plan p = build_plan(cfg, o);
  finalise_offsets(&p, 2048);
  return p.total_bytes;
}

bool qpack_from_params(const ParamStore& ps, const GPTConfig& cfg,
                       const QPackOptions& o, const std::string& path,
                       std::string* err) {
  Plan plan = build_plan(cfg, o);
  // Validate against the snapshot before writing anything.
  for (const PlanEntry& e : plan.e) {
    const int idx = ps.find(e.name);
    if (idx < 0) {
      if (err) *err = "checkpoint is missing parameter '" + e.name + "'";
      return false;
    }
    const int64_t n = static_cast<int64_t>(ps.data[static_cast<size_t>(idx)].size());
    if (n != e.rows * e.cols) {
      if (err)
        *err = "parameter '" + e.name + "' has " + std::to_string(n) +
               " values, expected " + std::to_string(e.rows * e.cols);
      return false;
    }
  }
  Config extra;
  extra.set("quant.source", "checkpoint");
  const std::string header = header_text(cfg, o, plan, extra);
  finalise_offsets(&plan, static_cast<int64_t>(header.size()));

  RowFn rows = [&ps](const PlanEntry& e, int64_t r, float* dst) {
    const int idx = ps.find(e.name);
    const float* src = ps.data[static_cast<size_t>(idx)].data();
    if (!e.transposed) {
      std::memcpy(dst, src + r * e.cols, sizeof(float) * static_cast<size_t>(e.cols));
    } else {
      // training layout [cols, rows] -> stored row r is a column of the source
      const int64_t stride = e.rows;
      for (int64_t k = 0; k < e.cols; ++k) dst[k] = src[k * stride + r];
    }
  };
  return write_plan(path, plan, header, rows, o.progress, err);
}

bool qpack_from_checkpoint(const std::string& ckpt, const QPackOptions& o,
                           const std::string& out, std::string* err) {
  ParamStore ps;
  CheckpointMeta meta;
  if (!load_checkpoint(ckpt, &ps, &meta)) {
    if (err) *err = "cannot read checkpoint " + ckpt;
    return false;
  }
  GPTConfig cfg = GPTConfig::from_config(meta.extra);
  // The checkpoint is authoritative about depth: progressive growth adds blocks
  // that the config file does not know about.
  int64_t layers = 0;
  for (const std::string& n : ps.names) {
    if (n.rfind("h.", 0) != 0) continue;
    const size_t dot = n.find('.', 2);
    layers = std::max<int64_t>(layers, std::stoll(n.substr(2, dot - 2)) + 1);
  }
  if (layers > 0 && layers != cfg.n_layer) cfg.n_layer = layers;
  const int idx = ps.find("tok_emb");
  if (idx >= 0 && ps.shapes[static_cast<size_t>(idx)].size() == 2) {
    cfg.vocab_size = static_cast<int32_t>(ps.shapes[static_cast<size_t>(idx)][0]);
    cfg.n_embd = ps.shapes[static_cast<size_t>(idx)][1];
  }
  QPackOptions oo = o;
  if (oo.note.empty()) oo.note = "packed from " + ckpt;
  return qpack_from_params(ps, cfg, oo, out, err);
}

bool qpack_synthesise(const GPTConfig& cfg, const QPackOptions& o,
                      const std::string& path, uint64_t seed, std::string* err) {
  Plan plan = build_plan(cfg, o);
  Config extra;
  extra.set("quant.source", "synthetic");
  extra.set("quant.seed", std::to_string(seed));
  const std::string header = header_text(cfg, o, plan, extra);
  finalise_offsets(&plan, static_cast<int64_t>(header.size()));

  RowFn rows = [seed](const PlanEntry& e, int64_t r, float* dst) {
    if (e.init == Init::kOnes) {
      std::fill(dst, dst + e.cols, 1.0f);
      return;
    }
    if (e.init == Init::kZeros) {
      std::fill(dst, dst + e.cols, 0.0f);
      return;
    }
    uint64_t h = 1469598103934665603ULL ^ seed;
    for (char c : e.name) {
      h ^= static_cast<unsigned char>(c);
      h *= 1099511628211ULL;
    }
    h ^= static_cast<uint64_t>(r) * 0x9E3779B97F4A7C15ULL;
    Rng rng(h);
    for (int64_t k = 0; k < e.cols; ++k) dst[k] = rng.normal() * e.std;
  };
  return write_plan(path, plan, header, rows, o.progress, err);
}

// ================================================================== runtime
size_t QGenState::cache_bytes() const {
  size_t n = 0;
  for (const auto& t : k) n += t.size() * sizeof(uint16_t);
  for (const auto& t : v) n += t.size() * sizeof(uint16_t);
  return n;
}

QModel::QModel() = default;
QModel::~QModel() { close(); }

void QModel::close() {
#ifdef SLM_HAVE_MMAP
  if (base_ && !owns_heap_) ::munmap(const_cast<uint8_t*>(base_), static_cast<size_t>(size_));
  if (fd_ >= 0) ::close(fd_);
#endif
  if (base_ && owns_heap_) delete[] const_cast<uint8_t*>(base_);
  base_ = nullptr;
  fd_ = -1;
  size_ = 0;
  owns_heap_ = false;
  params_ = 0;
  quant_bytes_ = 0;
  tensors_.clear();
  order_.clear();
}

namespace {
template <typename T>
bool take(const uint8_t*& p, const uint8_t* end, T* out) {
  if (p + sizeof(T) > end) return false;
  std::memcpy(out, p, sizeof(T));
  p += sizeof(T);
  return true;
}
}  // namespace

bool QModel::open(const std::string& path, std::string* err) {
  close();
  path_ = path;
#ifdef SLM_HAVE_MMAP
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    if (err) *err = "cannot open " + path;
    return false;
  }
  struct stat st{};
  if (::fstat(fd_, &st) != 0 || st.st_size < 32) {
    if (err) *err = "cannot stat " + path;
    close();
    return false;
  }
  size_ = static_cast<int64_t>(st.st_size);
  void* m = ::mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_SHARED, fd_, 0);
  if (m == MAP_FAILED) {
    if (err) *err = "mmap failed for " + path;
    close();
    return false;
  }
  base_ = static_cast<const uint8_t*>(m);
#else
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    if (err) *err = "cannot open " + path;
    return false;
  }
  size_ = static_cast<int64_t>(f.tellg());
  f.seekg(0);
  uint8_t* buf = new uint8_t[static_cast<size_t>(size_)];
  f.read(reinterpret_cast<char*>(buf), size_);
  base_ = buf;
  owns_heap_ = true;
#endif

  const uint8_t* p = base_;
  const uint8_t* end = base_ + size_;
  if (std::memcmp(p, kQMagic, sizeof(kQMagic)) != 0) {
    if (err) *err = path + " is not a .slmq file";
    close();
    return false;
  }
  p += sizeof(kQMagic);
  uint32_t header_len = 0, ntensors = 0;
  uint64_t payload_off = 0;
  if (!take(p, end, &header_len) || !take(p, end, &ntensors) ||
      !take(p, end, &payload_off)) {
    if (err) *err = "truncated header";
    close();
    return false;
  }
  if (p + header_len > end) {
    if (err) *err = "truncated meta";
    close();
    return false;
  }
  meta_ = Config();
  {
    const std::string text(reinterpret_cast<const char*>(p), header_len);
    size_t pos = 0;
    while (pos < text.size()) {
      const size_t nl = text.find('\n', pos);
      meta_.set_kv(text.substr(pos, nl == std::string::npos ? nl : nl - pos));
      if (nl == std::string::npos) break;
      pos = nl + 1;
    }
  }
  p += header_len;

  for (uint32_t i = 0; i < ntensors; ++i) {
    uint16_t nlen = 0;
    if (!take(p, end, &nlen) || p + nlen > end) {
      if (err) *err = "truncated directory";
      close();
      return false;
    }
    const std::string name(reinterpret_cast<const char*>(p), nlen);
    p += nlen;
    uint8_t ty = 0, ndim = 0;
    if (!take(p, end, &ty) || !take(p, end, &ndim)) {
      if (err) *err = "truncated directory entry";
      close();
      return false;
    }
    int64_t rows = 1, cols = 0;
    if (ndim == 2 && !take(p, end, &rows)) ndim = 0;
    if (!take(p, end, &cols)) ndim = 0;
    uint64_t off = 0, bytes = 0;
    if (ndim == 0 || !take(p, end, &off) || !take(p, end, &bytes)) {
      if (err) *err = "truncated directory entry for '" + name + "'";
      close();
      return false;
    }
    if (static_cast<int64_t>(off + bytes) > size_) {
      if (err) *err = "tensor '" + name + "' runs past the end of the file";
      close();
      return false;
    }
    QTensorView v;
    v.type = static_cast<QType>(ty);
    v.rows = rows;
    v.cols = cols;
    v.row_bytes = qrow_bytes(v.type, cols);
    v.data = base_ + off;
    tensors_[name] = v;
    order_.push_back(name);
  }

  cfg_ = GPTConfig::from_config(meta_);
  QPackOptions o;
  if (!parse_qtype(meta_.get_str("quant.body", "q4"), &o.type) ||
      !parse_qtype(meta_.get_str("quant.embed", "q8"), &o.embed_type)) {
    if (err) *err = "unknown quantisation type in header";
    close();
    return false;
  }
  body_type_ = o.type;
  // Structural validation: the file must contain exactly the tensors this
  // architecture needs, with the shapes and encodings the plan predicts.
  Plan plan = build_plan(cfg_, o);
  params_ = 0;
  quant_bytes_ = 0;
  for (const PlanEntry& e : plan.e) {
    auto it = tensors_.find(e.name);
    if (it == tensors_.end()) {
      if (err) *err = "file is missing tensor '" + e.name + "'";
      close();
      return false;
    }
    const QTensorView& v = it->second;
    if (v.rows != e.rows || v.cols != e.cols || v.type != e.type) {
      if (err)
        *err = "tensor '" + e.name + "' is [" + std::to_string(v.rows) + "," +
               std::to_string(v.cols) + "] " + qtype_name(v.type) + ", expected [" +
               std::to_string(e.rows) + "," + std::to_string(e.cols) + "] " +
               qtype_name(e.type);
      close();
      return false;
    }
    params_ += e.rows * e.cols;
    quant_bytes_ += e.bytes;
  }
  if (static_cast<size_t>(plan.e.size()) != tensors_.size()) {
    if (err) *err = "file has extra tensors this architecture does not use";
    close();
    return false;
  }
  return true;
}

double QModel::bits_per_weight() const {
  if (params_ <= 0) return 0.0;
  return 8.0 * static_cast<double>(quant_bytes_) / static_cast<double>(params_);
}

void QModel::prefetch() {
  if (!base_) return;
#ifdef SLM_HAVE_MMAP
  if (!owns_heap_) {
    ::madvise(const_cast<uint8_t*>(base_), static_cast<size_t>(size_), MADV_WILLNEED);
    ::posix_fadvise(fd_, 0, 0, POSIX_FADV_WILLNEED);
  }
#endif
  // madvise is only a hint and returns immediately, so touch every page in file
  // order as well: that turns the several hundred thousand random 4 KiB faults
  // the first forward pass would otherwise take into one sequential read.
  volatile uint64_t sink = 0;
  const int64_t page = 4096;
  for (int64_t off = 0; off < size_; off += page) sink += base_[off];
  (void)sink;
}

void QModel::drop_page_cache() {
#ifdef SLM_HAVE_MMAP
  if (fd_ >= 0) ::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
#endif
}

size_t QModel::resident_bytes() const {
#ifdef SLM_HAVE_MMAP
  if (!base_ || owns_heap_) return static_cast<size_t>(size_);
  const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  const size_t pages = (static_cast<size_t>(size_) + page - 1) / page;
  std::vector<unsigned char> vec(pages, 0);
  if (::mincore(const_cast<uint8_t*>(base_), static_cast<size_t>(size_), vec.data()) != 0)
    return 0;
  size_t n = 0;
  for (unsigned char c : vec)
    if (c & 1) ++n;
  return n * page;
#else
  return static_cast<size_t>(size_);
#endif
}

std::string QModel::describe() const {
  std::ostringstream os;
  os << "QModel(" << cfg_.n_layer << "L x " << cfg_.n_embd << "d, vocab "
     << cfg_.vocab_size << ", ctx " << cfg_.block_size << ", "
     << cfg_.arch_summary() << ") " << (params_ / 1e6) << "M params, "
     << qtype_name(body_type_) << ", " << (bits_per_weight()) << " bits/weight, "
     << (size_ / (1024.0 * 1024.0)) << " MiB file";
  return os.str();
}

const QTensorView* QModel::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const QTensorView& QModel::need(const std::string& name) const {
  const QTensorView* v = find(name);
  if (!v) throw std::runtime_error("qmodel: missing tensor " + name);
  return *v;
}

// ------------------------------------------------------------------ forward
void QModel::reset(QGenState* st, int64_t max_ctx) const {
  int64_t T = max_ctx > 0 ? max_ctx : cfg_.block_size;
  if (cfg_.pos == PosKind::kLearned) T = std::min(T, cfg_.block_size);
  st->max_ctx = T;
  st->pos = 0;
  st->history.clear();
  const int64_t Hkv = cfg_.kv_heads(), Dh = cfg_.head_dim();
  st->k.assign(static_cast<size_t>(cfg_.n_layer),
               std::vector<uint16_t>(static_cast<size_t>(Hkv * T * Dh), 0));
  st->v = st->k;
  st->scores.assign(static_cast<size_t>(cfg_.n_head * T), 0.0f);
}

void QModel::block(QGenState* st, int64_t layer, int64_t T, int64_t pos0) const {
  const int64_t C = cfg_.n_embd, H = cfg_.n_head, Hkv = cfg_.kv_heads();
  const int64_t Dh = cfg_.head_dim(), KV = Hkv * Dh, QKV = C + 2 * KV;
  const int64_t Hf = cfg_.hidden_dim(), Tmax = st->max_ctx;
  const int64_t rep = H / Hkv;
  const bool rms = cfg_.norm == NormKind::kRMSNorm;
  const std::string b = "h." + std::to_string(layer) + ".";

  const QTensorView& n1g = need(b + "n1.g");
  const QTensorView* n1b = find(b + "n1.b");
  const QTensorView& qkvw = need(b + "attn.qkv.w");
  const QTensorView* qkvb = find(b + "attn.qkv.b");
  const QTensorView* qn = find(b + "attn.qnorm.g");
  const QTensorView* kn = find(b + "attn.knorm.g");
  const QTensorView& pw = need(b + "attn.proj.w");
  const QTensorView* pb = find(b + "attn.proj.b");
  const QTensorView& n2g = need(b + "n2.g");
  const QTensorView* n2b = find(b + "n2.b");
  const QTensorView& upw = need(b + "mlp.up.w");
  const QTensorView* upb = find(b + "mlp.up.b");
  const QTensorView* gatew = find(b + "mlp.gate.w");
  const QTensorView& downw = need(b + "mlp.down.w");
  const QTensorView* downb = find(b + "mlp.down.b");

  float* x = st->x.data();
  float* h = st->h.data();
  float* qkv = st->qkv.data();
  float* att = st->att.data();

  // ---- attention norm
  for (int64_t t = 0; t < T; ++t) {
    if (rms)
      rms_norm(x + t * C, n1g.f32(), C, cfg_.ln_eps, h + t * C);
    else
      layer_norm(x + t * C, n1g.f32(), n1b->f32(), C, cfg_.ln_eps, h + t * C);
  }

  // ---- fused q,k,v
  qlinear(qkvw.type, qkvw.data, QKV, C, h, T, qkvb ? qkvb->f32() : nullptr, qkv, QKV,
          &st->act);

  // ---- per-head QK norm, RoPE, cache append
  uint16_t* kc = st->k[static_cast<size_t>(layer)].data();
  uint16_t* vc = st->v[static_cast<size_t>(layer)].data();
  for (int64_t t = 0; t < T; ++t) {
    float* row = qkv + t * QKV;
    const int64_t pos = pos0 + t;
    for (int64_t hq = 0; hq < H; ++hq) {
      float* q = row + hq * Dh;
      if (cfg_.qk_norm && qn) {
        std::vector<float>& tmp = st->ffn;  // scratch, large enough
        rms_norm(q, qn->f32(), Dh, cfg_.ln_eps, tmp.data());
        std::memcpy(q, tmp.data(), sizeof(float) * static_cast<size_t>(Dh));
      }
      if (cfg_.pos == PosKind::kRoPE) rope_inplace(q, Dh, pos, cfg_.rope_theta);
    }
    for (int64_t hk = 0; hk < Hkv; ++hk) {
      float* k = row + C + hk * Dh;
      if (cfg_.qk_norm && kn) {
        std::vector<float>& tmp = st->ffn;
        rms_norm(k, kn->f32(), Dh, cfg_.ln_eps, tmp.data());
        std::memcpy(k, tmp.data(), sizeof(float) * static_cast<size_t>(Dh));
      }
      if (cfg_.pos == PosKind::kRoPE) rope_inplace(k, Dh, pos, cfg_.rope_theta);
      uint16_t* dstk = kc + (hk * Tmax + pos) * Dh;
      uint16_t* dstv = vc + (hk * Tmax + pos) * Dh;
      const float* v = row + C + KV + hk * Dh;
      for (int64_t d = 0; d < Dh; ++d) {
        dstk[d] = float_to_half(k[d]);
        dstv[d] = float_to_half(v[d]);
      }
    }
  }

  // ---- causal attention against the f16 cache
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (H > 1)
#endif
  for (int64_t hq = 0; hq < H; ++hq) {
    const int64_t hk = hq / rep;
    float* s = st->scores.data() + hq * Tmax;
    const uint16_t* kh = kc + hk * Tmax * Dh;
    const uint16_t* vh = vc + hk * Tmax * Dh;
    for (int64_t t = 0; t < T; ++t) {
      const float* q = qkv + t * QKV + hq * Dh;
      const int64_t Tk = pos0 + t + 1;
      for (int64_t j = 0; j < Tk; ++j)
        s[j] = qdot_f16(reinterpret_cast<const uint8_t*>(kh + j * Dh), q, Dh) * scale;
      softmax_inplace(s, Tk);
      float* o = att + t * C + hq * Dh;
      std::fill(o, o + Dh, 0.0f);
      for (int64_t j = 0; j < Tk; ++j) {
        if (s[j] < 1e-7f) continue;  // 90% of a long context contributes nothing
        axpy_f16(o, vh + j * Dh, Dh, s[j]);
      }
    }
  }

  // ---- output projection + residual
  qlinear(pw.type, pw.data, C, C, att, T, pb ? pb->f32() : nullptr, h, C, &st->act);
  for (int64_t i = 0; i < T * C; ++i) x[i] += h[i];

  // ---- feed forward
  for (int64_t t = 0; t < T; ++t) {
    if (rms)
      rms_norm(x + t * C, n2g.f32(), C, cfg_.ln_eps, h + t * C);
    else
      layer_norm(x + t * C, n2g.f32(), n2b->f32(), C, cfg_.ln_eps, h + t * C);
  }
  float* g = st->ffn.data();
  float* u = st->ffn.data() + T * Hf;
  if (cfg_.ffn == FFNKind::kSwiGLU) {
    qlinear(gatew->type, gatew->data, Hf, C, h, T, nullptr, g, Hf, &st->act);
    qlinear(upw.type, upw.data, Hf, C, h, T, nullptr, u, Hf, &st->act);
    for (int64_t i = 0; i < T * Hf; ++i) g[i] = silu(g[i]) * u[i];
  } else {
    qlinear(upw.type, upw.data, Hf, C, h, T, upb ? upb->f32() : nullptr, g, Hf,
            &st->act);
    for (int64_t i = 0; i < T * Hf; ++i) g[i] = gelu_tanh(g[i]);
  }
  qlinear(downw.type, downw.data, C, Hf, g, T, downb ? downb->f32() : nullptr, att, C,
          &st->act);
  for (int64_t i = 0; i < T * C; ++i) x[i] += att[i];
}

void QModel::head_logits(QGenState* st, const float* x, int64_t T, bool last_only,
                         std::vector<float>* out) const {
  const int64_t C = cfg_.n_embd, V = cfg_.vocab_size;
  const QTensorView& head = cfg_.tie_weights ? need("tok_emb") : need("lm_head.w");
  const int64_t rows = last_only ? 1 : T;
  out->assign(static_cast<size_t>(rows * V), 0.0f);
  const float* src = last_only ? x + (T - 1) * C : x;
  qlinear(head.type, head.data, V, C, src, rows, nullptr, out->data(), V, &st->act);
}

void QModel::forward(QGenState* st, const std::vector<int32_t>& ids,
                     std::vector<float>* logits) const {
  if (!is_open()) throw std::runtime_error("qmodel: no file open");
  const int64_t T = static_cast<int64_t>(ids.size());
  if (T == 0) throw std::runtime_error("qmodel: empty input");
  if (st->max_ctx == 0) reset(st);
  const int64_t pos0 = st->pos;
  if (pos0 + T > st->max_ctx)
    throw std::runtime_error("qmodel: context overflow (" + std::to_string(pos0 + T) +
                             " > " + std::to_string(st->max_ctx) + ")");
  const int64_t C = cfg_.n_embd, KV = cfg_.kv_heads() * cfg_.head_dim();
  const int64_t Hf = cfg_.hidden_dim();
  st->x.assign(static_cast<size_t>(T * C), 0.0f);
  st->h.resize(static_cast<size_t>(T * C));
  st->qkv.resize(static_cast<size_t>(T * (C + 2 * KV)));
  st->att.resize(static_cast<size_t>(T * C));
  // The FFN scratch doubles as the per-head norm scratch, so keep a floor.
  st->ffn.resize(static_cast<size_t>(std::max<int64_t>(2 * T * Hf, C)));

  // ---- token (and optional learned position) embedding
  const QTensorView& emb = need("tok_emb");
  for (int64_t t = 0; t < T; ++t) {
    const int32_t id = ids[static_cast<size_t>(t)];
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::runtime_error("qmodel: token id out of range");
    dequantize_row(emb.type, emb.data + id * emb.row_bytes, C, st->x.data() + t * C);
  }
  if (cfg_.pos == PosKind::kLearned) {
    const QTensorView& pe = need("pos_emb");
    for (int64_t t = 0; t < T; ++t) {
      const float* p = pe.f32() + (pos0 + t) * C;
      float* xr = st->x.data() + t * C;
      for (int64_t j = 0; j < C; ++j) xr[j] += p[j];
    }
  }

  for (int64_t l = 0; l < cfg_.n_layer; ++l) block(st, l, T, pos0);

  const QTensorView& nfg = need("norm_f.g");
  const QTensorView* nfb = find("norm_f.b");
  for (int64_t t = 0; t < T; ++t) {
    if (cfg_.norm == NormKind::kRMSNorm)
      rms_norm(st->x.data() + t * C, nfg.f32(), C, cfg_.ln_eps, st->h.data() + t * C);
    else
      layer_norm(st->x.data() + t * C, nfg.f32(), nfb->f32(), C, cfg_.ln_eps,
                 st->h.data() + t * C);
  }
  head_logits(st, st->h.data(), T, true, logits);
  st->pos += T;
  for (int32_t id : ids) st->history.push_back(id);
}

void QModel::forward_token(QGenState* st, int32_t id, std::vector<float>* logits) const {
  const std::vector<int32_t> one{id};
  forward(st, one, logits);
}

void QModel::logits_for(const std::vector<int32_t>& ids,
                        std::vector<float>* logits) const {
  QGenState st;
  reset(&st, static_cast<int64_t>(ids.size()) + 1);
  forward(&st, ids, logits);
}

// ------------------------------------------------------------------ sampling
std::vector<int32_t> QModel::generate(
    const std::vector<int32_t>& prompt, const GenOptions& opt,
    const std::function<bool(const GenStep&)>& on_step) const {
  std::vector<int32_t> out;
  if (prompt.empty()) return out;
  const int32_t V = cfg_.vocab_size;
  const int64_t want =
      static_cast<int64_t>(prompt.size()) + std::max(1, opt.max_new_tokens) + 1;
  const int64_t ctx = (cfg_.pos == PosKind::kLearned)
                          ? std::min(want, cfg_.block_size)
                          : std::max(want, cfg_.block_size);
  QGenState st;
  reset(&st, ctx);

  const int64_t keep = std::min<int64_t>(static_cast<int64_t>(prompt.size()),
                                         ctx - std::max(1, opt.max_new_tokens));
  std::vector<int32_t> ids(prompt.end() - std::max<int64_t>(1, keep), prompt.end());

  std::vector<float> logits;
  forward(&st, ids, &logits);

  Rng rng(opt.seed ? opt.seed : 0x5eed1234u);
  std::vector<float> row(static_cast<size_t>(V));
  std::vector<std::pair<float, int32_t>> cand;

  for (int step = 0; step < opt.max_new_tokens; ++step) {
    std::copy(logits.begin(), logits.begin() + V, row.begin());
    if (opt.repetition_penalty > 1.0f) {
      const size_t look = std::min<size_t>(st.history.size(), 128);
      for (size_t i = st.history.size() - look; i < st.history.size(); ++i) {
        const int32_t t = st.history[i];
        if (t < 0 || t >= V) continue;
        float& r = row[static_cast<size_t>(t)];
        r = (r > 0.0f) ? r / opt.repetition_penalty : r * opt.repetition_penalty;
      }
    }
    const float temp = std::max(1e-4f, opt.temperature);
    for (float& v : row) v /= temp;

    cand.clear();
    cand.reserve(static_cast<size_t>(V));
    for (int32_t i = 0; i < V; ++i) cand.emplace_back(row[static_cast<size_t>(i)], i);
    const int k = (opt.top_k > 0) ? std::min<int>(opt.top_k, V) : V;
    std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                      [](const std::pair<float, int32_t>& a,
                         const std::pair<float, int32_t>& b) { return a.first > b.first; });
    cand.resize(static_cast<size_t>(k));
    const float mx = cand[0].first;
    double sum = 0.0;
    for (auto& c : cand) {
      c.first = std::exp(c.first - mx);
      sum += c.first;
    }
    for (auto& c : cand) c.first = static_cast<float>(c.first / sum);
    if (opt.min_p > 0.0f && cand.size() > 1) {
      // Relative floor, so it scales with how peaked this step happens to be.
      const float floor_p = opt.min_p * cand[0].first;
      size_t keep_n = 1;
      while (keep_n < cand.size() && cand[keep_n].first >= floor_p) ++keep_n;
      if (keep_n < cand.size()) {
        cand.resize(keep_n);
        double s2 = 0.0;
        for (const auto& c : cand) s2 += c.first;
        for (auto& c : cand) c.first = static_cast<float>(c.first / s2);
      }
    }
    if (opt.top_p > 0.0f && opt.top_p < 1.0f) {
      double acc = 0.0;
      size_t keep_n = 0;
      for (; keep_n < cand.size(); ++keep_n) {
        acc += cand[keep_n].first;
        if (acc >= opt.top_p) {
          ++keep_n;
          break;
        }
      }
      cand.resize(std::max<size_t>(1, keep_n));
      double s2 = 0.0;
      for (const auto& c : cand) s2 += c.first;
      for (auto& c : cand) c.first = static_cast<float>(c.first / s2);
    }
    const float u = rng.uniform();
    double acc = 0.0;
    int32_t chosen = cand.back().second;
    float chosen_p = cand.back().first;
    for (const auto& c : cand) {
      acc += c.first;
      if (u <= acc) {
        chosen = c.second;
        chosen_p = c.first;
        break;
      }
    }
    GenStep gs;
    gs.token = chosen;
    gs.logprob = std::log(std::max(1e-12f, chosen_p));
    const size_t ntop = std::min<size_t>(cand.size(), 12);
    for (size_t i = 0; i < ntop; ++i) gs.top.emplace_back(cand[i].second, cand[i].first);
    out.push_back(chosen);
    if (on_step && !on_step(gs)) break;
    if (opt.stop_on_eot && chosen == 0) break;
    if (st.pos + 1 > st.max_ctx) break;
    forward_token(&st, chosen, &logits);
  }
  return out;
}

double QModel::eval_nats(const std::vector<int32_t>& ids, int64_t* ntok,
                         int64_t chunk_in) const {
  if (ids.size() < 2) {
    if (ntok) *ntok = 0;
    return 0.0;
  }
  const int64_t V = cfg_.vocab_size;
  const int64_t chunk =
      chunk_in > 0 ? chunk_in : std::min<int64_t>(cfg_.block_size, 512);
  double total = 0.0;
  int64_t n = 0;
  QGenState st;
  std::vector<float> logits;
  for (size_t start = 0; start + 1 < ids.size(); start += static_cast<size_t>(chunk)) {
    const size_t stop = std::min(ids.size() - 1, start + static_cast<size_t>(chunk));
    reset(&st, chunk + 1);
    // Teacher forcing one token at a time: the KV cache makes each step cheap and
    // this reuses exactly the path that generation uses.
    for (size_t i = start; i < stop; ++i) {
      forward_token(&st, ids[i], &logits);
      const int32_t tgt = ids[i + 1];
      float mx = logits[0];
      for (int64_t j = 1; j < V; ++j) mx = std::max(mx, logits[static_cast<size_t>(j)]);
      double sum = 0.0;
      for (int64_t j = 0; j < V; ++j)
        sum += std::exp(static_cast<double>(logits[static_cast<size_t>(j)]) - mx);
      total += std::log(sum) + mx - static_cast<double>(logits[static_cast<size_t>(tgt)]);
      ++n;
    }
  }
  if (ntok) *ntok = n;
  return n ? total / static_cast<double>(n) : 0.0;
}

}  // namespace slm
