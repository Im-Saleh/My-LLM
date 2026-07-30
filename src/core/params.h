// SPDX-License-Identifier: Apache-2.0
//
// ParamStore: an immutable, device independent snapshot of every model weight.
//
// It is the unit of exchange between the model replicas, the coordinator, the
// checkpoint files and the GUI.  Snapshots are shared through
// std::shared_ptr<const ParamStore>, which gives us RCU semantics: publishing a
// new set of weights is a single atomic pointer swap and readers that are still
// running keep the version they started with alive.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"

namespace slm {

struct ParamStore {
  std::vector<std::string> names;
  std::vector<Shape> shapes;
  std::vector<std::vector<float>> data;
  std::unordered_map<std::string, int> index;

  void add(const std::string& name, const Shape& s, std::vector<float> v);
  int find(const std::string& name) const;
  int64_t total_elements() const;
  size_t size() const { return names.size(); }
  std::shared_ptr<ParamStore> clone() const;
  // Exact structural match (names + shapes, order independent).
  bool compatible_with(const ParamStore& other) const;
};

using ParamStorePtr = std::shared_ptr<const ParamStore>;

// A flat view over a *subset* of parameters (the trainable ones).  The
// coordinator does all of its linear algebra in this flat space.
struct FlatSpec {
  std::vector<std::string> names;
  std::vector<int64_t> offsets;  // size == names.size() + 1
  int64_t total = 0;

  static FlatSpec build(const ParamStore& store, const std::vector<std::string>& names);
  void gather(const ParamStore& src, std::vector<float>& out) const;
  void scatter(const std::vector<float>& in, ParamStore& dst) const;
  // dst += alpha * delta
  void add_scaled(const std::vector<float>& delta, float alpha, ParamStore& dst) const;
};

// -------------------------------------------------------------- vector helpers
double vec_dot(const std::vector<float>& a, const std::vector<float>& b);
double vec_norm(const std::vector<float>& a);
void vec_axpy(float a, const std::vector<float>& x, std::vector<float>& y);  // y += a*x
void vec_scale(std::vector<float>& x, float a);

}  // namespace slm
