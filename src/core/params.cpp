// SPDX-License-Identifier: Apache-2.0
#include "core/params.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slm {

void ParamStore::add(const std::string& name, const Shape& s, std::vector<float> v) {
  index[name] = static_cast<int>(names.size());
  names.push_back(name);
  shapes.push_back(s);
  data.push_back(std::move(v));
}

int ParamStore::find(const std::string& name) const {
  auto it = index.find(name);
  return it == index.end() ? -1 : it->second;
}

int64_t ParamStore::total_elements() const {
  int64_t n = 0;
  for (const auto& d : data) n += static_cast<int64_t>(d.size());
  return n;
}

std::shared_ptr<ParamStore> ParamStore::clone() const {
  return std::make_shared<ParamStore>(*this);
}

bool ParamStore::compatible_with(const ParamStore& other) const {
  if (names.size() != other.names.size()) return false;
  for (size_t i = 0; i < names.size(); ++i) {
    const int j = other.find(names[i]);
    if (j < 0) return false;
    if (shapes[i] != other.shapes[static_cast<size_t>(j)]) return false;
  }
  return true;
}

FlatSpec FlatSpec::build(const ParamStore& store, const std::vector<std::string>& names) {
  FlatSpec s;
  s.names = names;
  s.offsets.reserve(names.size() + 1);
  int64_t off = 0;
  for (const std::string& n : names) {
    const int i = store.find(n);
    if (i < 0) throw std::runtime_error("FlatSpec: unknown parameter " + n);
    s.offsets.push_back(off);
    off += static_cast<int64_t>(store.data[static_cast<size_t>(i)].size());
  }
  s.offsets.push_back(off);
  s.total = off;
  return s;
}

void FlatSpec::gather(const ParamStore& src, std::vector<float>& out) const {
  out.assign(static_cast<size_t>(total), 0.0f);
  for (size_t k = 0; k < names.size(); ++k) {
    const int i = src.find(names[k]);
    if (i < 0) throw std::runtime_error("FlatSpec::gather: missing " + names[k]);
    const std::vector<float>& d = src.data[static_cast<size_t>(i)];
    std::copy(d.begin(), d.end(), out.begin() + static_cast<long>(offsets[k]));
  }
}

void FlatSpec::scatter(const std::vector<float>& in, ParamStore& dst) const {
  if (static_cast<int64_t>(in.size()) != total)
    throw std::runtime_error("FlatSpec::scatter: size mismatch");
  for (size_t k = 0; k < names.size(); ++k) {
    const int i = dst.find(names[k]);
    if (i < 0) throw std::runtime_error("FlatSpec::scatter: missing " + names[k]);
    std::vector<float>& d = dst.data[static_cast<size_t>(i)];
    std::copy(in.begin() + static_cast<long>(offsets[k]),
              in.begin() + static_cast<long>(offsets[k + 1]), d.begin());
  }
}

void FlatSpec::add_scaled(const std::vector<float>& delta, float alpha,
                          ParamStore& dst) const {
  if (static_cast<int64_t>(delta.size()) != total)
    throw std::runtime_error("FlatSpec::add_scaled: size mismatch");
  for (size_t k = 0; k < names.size(); ++k) {
    const int i = dst.find(names[k]);
    if (i < 0) throw std::runtime_error("FlatSpec::add_scaled: missing " + names[k]);
    std::vector<float>& d = dst.data[static_cast<size_t>(i)];
    const int64_t base = offsets[k];
    for (size_t j = 0; j < d.size(); ++j)
      d[j] += alpha * delta[static_cast<size_t>(base) + j];
  }
}

double vec_dot(const std::vector<float>& a, const std::vector<float>& b) {
  const size_t n = std::min(a.size(), b.size());
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return s;
}

double vec_norm(const std::vector<float>& a) {
  double s = 0.0;
  for (float v : a) s += static_cast<double>(v) * static_cast<double>(v);
  return std::sqrt(s);
}

void vec_axpy(float a, const std::vector<float>& x, std::vector<float>& y) {
  const size_t n = std::min(x.size(), y.size());
  for (size_t i = 0; i < n; ++i) y[i] += a * x[i];
}

void vec_scale(std::vector<float>& x, float a) {
  for (float& v : x) v *= a;
}

}  // namespace slm
