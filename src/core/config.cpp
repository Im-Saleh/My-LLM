// SPDX-License-Identifier: Apache-2.0
#include "core/config.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace slm {
namespace {

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n'))
    --b;
  return s.substr(a, b - a);
}

}  // namespace

int Config::find(const std::string& key) const {
  for (size_t i = 0; i < kv_.size(); ++i)
    if (kv_[i].first == key) return static_cast<int>(i);
  return -1;
}

bool Config::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) {
    const size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    line = trim(line);
    if (line.empty()) continue;
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    set(trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
  }
  return true;
}

bool Config::save(const std::string& path) const {
  std::ofstream f(path);
  if (!f) return false;
  f << to_string();
  return f.good();
}

std::string Config::to_string() const {
  std::ostringstream os;
  for (const auto& e : kv_) os << e.first << " = " << e.second << "\n";
  return os.str();
}

void Config::set(const std::string& key, const std::string& value) {
  const int i = find(key);
  if (i >= 0)
    kv_[static_cast<size_t>(i)].second = value;
  else
    kv_.emplace_back(key, value);
}

bool Config::set_kv(const std::string& assignment) {
  const size_t eq = assignment.find('=');
  if (eq == std::string::npos) return false;
  set(trim(assignment.substr(0, eq)), trim(assignment.substr(eq + 1)));
  return true;
}

bool Config::has(const std::string& key) const { return find(key) >= 0; }

std::string Config::get_str(const std::string& key, const std::string& def) const {
  const int i = find(key);
  return i < 0 ? def : kv_[static_cast<size_t>(i)].second;
}

double Config::get_num(const std::string& key, double def) const {
  const int i = find(key);
  if (i < 0) return def;
  try {
    return std::stod(kv_[static_cast<size_t>(i)].second);
  } catch (...) {
    return def;
  }
}

int64_t Config::get_int(const std::string& key, int64_t def) const {
  const int i = find(key);
  if (i < 0) return def;
  try {
    return static_cast<int64_t>(std::stod(kv_[static_cast<size_t>(i)].second));
  } catch (...) {
    return def;
  }
}

bool Config::get_bool(const std::string& key, bool def) const {
  const int i = find(key);
  if (i < 0) return def;
  const std::string v = kv_[static_cast<size_t>(i)].second;
  if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
  if (v == "0" || v == "false" || v == "no" || v == "off") return false;
  return def;
}

}  // namespace slm
