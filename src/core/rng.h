// SPDX-License-Identifier: Apache-2.0
// Deterministic, header-only RNG (xoshiro256++) so every run is reproducible.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace slm {

class Rng {
 public:
  explicit Rng(uint64_t seed = 1234567891011ULL) { reseed(seed); }

  void reseed(uint64_t seed) {
    // SplitMix64 to spread the seed over the state.
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; ++i) {
      z += 0x9E3779B97F4A7C15ULL;
      uint64_t x = z;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      s_[i] = x ^ (x >> 31);
    }
    has_spare_ = false;
  }

  uint64_t next_u64() {
    const uint64_t r = rotl(s_[0] + s_[3], 23) + s_[0];
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return r;
  }

  // Uniform in [0,1)
  float uniform() {
    return static_cast<float>((next_u64() >> 40) * (1.0 / 16777216.0));
  }

  uint64_t below(uint64_t n) { return n ? next_u64() % n : 0; }

  float normal() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    float u, v, s;
    do {
      u = 2.0f * uniform() - 1.0f;
      v = 2.0f * uniform() - 1.0f;
      s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    const float f = std::sqrt(-2.0f * std::log(s) / s);
    spare_ = v * f;
    has_spare_ = true;
    return u * f;
  }

 private:
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t s_[4]{};
  bool has_spare_ = false;
  float spare_ = 0.0f;
};

}  // namespace slm
