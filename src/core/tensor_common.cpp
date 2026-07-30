// SPDX-License-Identifier: Apache-2.0
#include "core/tensor.h"

namespace slm {

int64_t numel_of(const Shape& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

std::string shape_str(const Shape& s) {
  std::string r = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) r += ",";
    r += std::to_string(s[i]);
  }
  return r + "]";
}

}  // namespace slm
