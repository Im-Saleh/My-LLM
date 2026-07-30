// SPDX-License-Identifier: Apache-2.0
#include "core/serialize.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace slm {
namespace {

constexpr char kMagic[7] = {'S', 'L', 'M', 'C', 'K', 'P', 'T'};
constexpr uint8_t kVersion = 1;

template <typename T>
void put(std::ostream& os, const T& v) {
  os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
bool get(std::istream& is, T* v) {
  is.read(reinterpret_cast<char*>(v), sizeof(T));
  return static_cast<bool>(is);
}

}  // namespace

const char* dtype_name(Dtype d) {
  switch (d) {
    case Dtype::F32:
      return "f32";
    case Dtype::F16:
      return "f16";
    case Dtype::Q8:
      return "q8";
  }
  return "?";
}

bool parse_dtype(const std::string& s, Dtype* out) {
  if (s == "f32" || s == "fp32" || s == "float") *out = Dtype::F32;
  else if (s == "f16" || s == "fp16" || s == "half") *out = Dtype::F16;
  else if (s == "q8" || s == "int8" || s == "i8") *out = Dtype::Q8;
  else return false;
  return true;
}

// ------------------------------------------------------------------ fp16
uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) {  // inf / nan
    return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
  }
  if (exp >= 0x1f) return static_cast<uint16_t>(sign | 0x7bffu);  // clamp to max
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t sub = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) ++sub;  // round to nearest
    return static_cast<uint16_t>(sign | sub);
  }
  uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                     (mant >> 13));
  if ((mant >> 12) & 1u) ++h;  // round to nearest
  return h;
}

float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t mant = h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      // subnormal half -> normalised float
      int e = -1;
      uint32_t m = mant;
      do {
        ++e;
        m <<= 1;
      } while (!(m & 0x400u));
      out = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) |
            ((m & 0x3ffu) << 13);
    }
  } else if (exp == 0x1f) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

// -------------------------------------------------------------------- int8
void quantize_q8(const float* src, int64_t n, std::vector<int8_t>* q,
                 std::vector<float>* scales) {
  const int64_t groups = (n + kQuantGroup - 1) / kQuantGroup;
  q->assign(static_cast<size_t>(n), 0);
  scales->assign(static_cast<size_t>(groups), 0.0f);
  for (int64_t g = 0; g < groups; ++g) {
    const int64_t b = g * kQuantGroup;
    const int64_t e = std::min<int64_t>(n, b + kQuantGroup);
    float amax = 0.0f;
    for (int64_t i = b; i < e; ++i) amax = std::max(amax, std::fabs(src[i]));
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    (*scales)[static_cast<size_t>(g)] = scale;
    const float inv = 1.0f / scale;
    for (int64_t i = b; i < e; ++i) {
      float v = src[i] * inv;
      v = std::max(-127.0f, std::min(127.0f, v));
      (*q)[static_cast<size_t>(i)] = static_cast<int8_t>(std::lround(v));
    }
  }
}

void dequantize_q8(const int8_t* q, const float* scales, int64_t n, float* dst) {
  for (int64_t i = 0; i < n; ++i)
    dst[i] = static_cast<float>(q[i]) * scales[i / kQuantGroup];
}

int64_t encoded_bytes(int64_t n, Dtype d) {
  switch (d) {
    case Dtype::F32:
      return n * 4;
    case Dtype::F16:
      return n * 2;
    case Dtype::Q8:
      return n + ((n + kQuantGroup - 1) / kQuantGroup) * 4;
  }
  return n * 4;
}

// -------------------------------------------------------------- checkpoints
bool save_checkpoint(const std::string& path, const ParamStore& ps,
                     const CheckpointMeta& meta, Dtype dtype) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  f.write(kMagic, sizeof(kMagic));
  put<uint8_t>(f, kVersion);
  put<uint8_t>(f, static_cast<uint8_t>(dtype));
  put<int64_t>(f, meta.step);
  put<int64_t>(f, meta.tokens_seen);
  const std::string metatext = meta.extra.to_string();
  put<uint32_t>(f, static_cast<uint32_t>(metatext.size()));
  f.write(metatext.data(), static_cast<std::streamsize>(metatext.size()));
  put<uint32_t>(f, static_cast<uint32_t>(ps.names.size()));

  std::vector<int8_t> q;
  std::vector<float> scales;
  std::vector<uint16_t> halves;
  for (size_t i = 0; i < ps.names.size(); ++i) {
    const std::string& name = ps.names[i];
    const Shape& shape = ps.shapes[i];
    const std::vector<float>& d = ps.data[i];
    put<uint16_t>(f, static_cast<uint16_t>(name.size()));
    f.write(name.data(), static_cast<std::streamsize>(name.size()));
    put<uint8_t>(f, static_cast<uint8_t>(shape.size()));
    for (int64_t dim : shape) put<int64_t>(f, dim);
    const Dtype dt = (shape.size() >= 2) ? dtype : Dtype::F32;
    put<uint8_t>(f, static_cast<uint8_t>(dt));
    const int64_t n = static_cast<int64_t>(d.size());
    put<uint64_t>(f, static_cast<uint64_t>(encoded_bytes(n, dt)));
    switch (dt) {
      case Dtype::F32:
        f.write(reinterpret_cast<const char*>(d.data()),
                static_cast<std::streamsize>(n * 4));
        break;
      case Dtype::F16:
        halves.resize(static_cast<size_t>(n));
        for (int64_t k = 0; k < n; ++k) halves[static_cast<size_t>(k)] = float_to_half(d[static_cast<size_t>(k)]);
        f.write(reinterpret_cast<const char*>(halves.data()),
                static_cast<std::streamsize>(n * 2));
        break;
      case Dtype::Q8:
        quantize_q8(d.data(), n, &q, &scales);
        f.write(reinterpret_cast<const char*>(scales.data()),
                static_cast<std::streamsize>(scales.size() * 4));
        f.write(reinterpret_cast<const char*>(q.data()),
                static_cast<std::streamsize>(n));
        break;
    }
  }
  return f.good();
}

namespace {

bool read_header(std::istream& f, CheckpointMeta* meta, Dtype* dtype,
                 uint32_t* nparams) {
  char magic[7];
  f.read(magic, sizeof(magic));
  if (!f || std::memcmp(magic, kMagic, sizeof(magic)) != 0) return false;
  uint8_t version = 0, dt = 0;
  if (!get(f, &version) || version != kVersion) return false;
  if (!get(f, &dt)) return false;
  if (dtype) *dtype = static_cast<Dtype>(dt);
  int64_t step = 0, tokens = 0;
  if (!get(f, &step) || !get(f, &tokens)) return false;
  uint32_t metalen = 0;
  if (!get(f, &metalen)) return false;
  std::string metatext(metalen, '\0');
  if (metalen) f.read(&metatext[0], metalen);
  if (!f) return false;
  if (meta) {
    meta->step = step;
    meta->tokens_seen = tokens;
    meta->extra = Config();
    size_t pos = 0;
    while (pos < metatext.size()) {
      const size_t nl = metatext.find('\n', pos);
      const std::string line = metatext.substr(pos, nl == std::string::npos ? nl : nl - pos);
      meta->extra.set_kv(line);
      if (nl == std::string::npos) break;
      pos = nl + 1;
    }
  }
  uint32_t np = 0;
  if (!get(f, &np)) return false;
  if (nparams) *nparams = np;
  return true;
}

}  // namespace

bool peek_checkpoint(const std::string& path, CheckpointMeta* meta, Dtype* dtype) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  uint32_t np = 0;
  return read_header(f, meta, dtype, &np);
}

bool load_checkpoint(const std::string& path, ParamStore* ps, CheckpointMeta* meta) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  uint32_t nparams = 0;
  Dtype file_dtype = Dtype::F32;
  if (!read_header(f, meta, &file_dtype, &nparams)) return false;
  ps->names.clear();
  ps->shapes.clear();
  ps->data.clear();
  ps->index.clear();
  std::vector<int8_t> q;
  std::vector<float> scales;
  std::vector<uint16_t> halves;
  for (uint32_t i = 0; i < nparams; ++i) {
    uint16_t nlen = 0;
    if (!get(f, &nlen)) return false;
    std::string name(nlen, '\0');
    f.read(&name[0], nlen);
    uint8_t ndim = 0;
    if (!get(f, &ndim)) return false;
    Shape shape(ndim);
    for (uint8_t d = 0; d < ndim; ++d)
      if (!get(f, &shape[d])) return false;
    uint8_t dt8 = 0;
    uint64_t nbytes = 0;
    if (!get(f, &dt8) || !get(f, &nbytes)) return false;
    const Dtype dt = static_cast<Dtype>(dt8);
    const int64_t n = numel_of(shape);
    if (encoded_bytes(n, dt) != static_cast<int64_t>(nbytes)) return false;
    std::vector<float> data(static_cast<size_t>(n));
    switch (dt) {
      case Dtype::F32:
        f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * 4));
        break;
      case Dtype::F16:
        halves.resize(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(halves.data()), static_cast<std::streamsize>(n * 2));
        for (int64_t k = 0; k < n; ++k)
          data[static_cast<size_t>(k)] = half_to_float(halves[static_cast<size_t>(k)]);
        break;
      case Dtype::Q8: {
        const int64_t groups = (n + kQuantGroup - 1) / kQuantGroup;
        scales.resize(static_cast<size_t>(groups));
        q.resize(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(scales.data()),
               static_cast<std::streamsize>(groups * 4));
        f.read(reinterpret_cast<char*>(q.data()), static_cast<std::streamsize>(n));
        dequantize_q8(q.data(), scales.data(), n, data.data());
        break;
      }
      default:
        return false;
    }
    if (!f) return false;
    ps->add(name, shape, std::move(data));
  }
  return true;
}

}  // namespace slm
