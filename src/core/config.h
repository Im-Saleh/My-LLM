// SPDX-License-Identifier: Apache-2.0
//
// Tiny "key = value" configuration store (no external JSON dependency).
//
//   # comment
//   model.n_layer = 6
//   train.lr = 3e-4
//
// Unknown keys are preserved, so a config file round-trips losslessly and the
// GUI can dump the live configuration back to disk.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace slm {

class Config {
 public:
  bool load(const std::string& path);
  bool save(const std::string& path) const;

  // "key=value" (used for CLI --set overrides)
  bool set_kv(const std::string& assignment);
  void set(const std::string& key, const std::string& value);

  bool has(const std::string& key) const;
  std::string get_str(const std::string& key, const std::string& def) const;
  double get_num(const std::string& key, double def) const;
  int64_t get_int(const std::string& key, int64_t def) const;
  bool get_bool(const std::string& key, bool def) const;

  const std::vector<std::pair<std::string, std::string>>& entries() const { return kv_; }
  std::string to_string() const;

 private:
  int find(const std::string& key) const;
  std::vector<std::pair<std::string, std::string>> kv_;
};

}  // namespace slm
