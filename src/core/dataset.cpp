// SPDX-License-Identifier: Apache-2.0
#include "core/dataset.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "tokenizer.h"

namespace slm {

bool TokenDataset::load_text_file(const std::string& path, const Tokenizer& tok,
                                  std::string* err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (err) *err = "cannot open " + path;
    return false;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  if (text.empty()) {
    if (err) *err = "empty corpus " + path;
    return false;
  }
  set_tokens(tok.encode(text), holdout_frac_);
  if (tokens_.size() < 32) {
    if (err) *err = "corpus too small after tokenisation";
    return false;
  }
  return true;
}

void TokenDataset::set_tokens(std::vector<int32_t> tokens, float holdout_frac) {
  tokens_ = std::move(tokens);
  holdout_frac_ = holdout_frac;
  const int64_t n = static_cast<int64_t>(tokens_.size());
  const int64_t hold = std::min<int64_t>(
      std::max<int64_t>(0, static_cast<int64_t>(static_cast<double>(n) * holdout_frac)),
      std::max<int64_t>(0, n / 2));
  train_end_ = n - hold;
}

void TokenDataset::append_text(const std::string& text, const Tokenizer& tok) {
  std::vector<int32_t> extra = tok.encode(text);
  std::vector<int32_t> all;
  all.reserve(tokens_.size() + extra.size());
  all.insert(all.end(), tokens_.begin(), tokens_.begin() + train_end_);
  all.insert(all.end(), extra.begin(), extra.end());
  all.insert(all.end(), tokens_.begin() + train_end_, tokens_.end());
  const int64_t hold = static_cast<int64_t>(tokens_.size()) - train_end_;
  tokens_ = std::move(all);
  train_end_ = static_cast<int64_t>(tokens_.size()) - hold;
}

Batch TokenDataset::sample_batch(int64_t B, int64_t T, Rng& rng) const {
  Batch b;
  b.B = B;
  b.T = T;
  b.ids.resize(static_cast<size_t>(B * T));
  b.targets.resize(static_cast<size_t>(B * T));
  const int64_t limit = train_end_ - T - 1;
  for (int64_t i = 0; i < B; ++i) {
    const int64_t off =
        limit > 0 ? static_cast<int64_t>(rng.below(static_cast<uint64_t>(limit))) : 0;
    for (int64_t t = 0; t < T; ++t) {
      const size_t src = static_cast<size_t>(off + t);
      b.ids[static_cast<size_t>(i * T + t)] = src < tokens_.size() ? tokens_[src] : 0;
      b.targets[static_cast<size_t>(i * T + t)] =
          src + 1 < tokens_.size() ? tokens_[src + 1] : -100;
    }
  }
  return b;
}

std::vector<Batch> TokenDataset::holdout_batches(int64_t B, int64_t T, int count) const {
  std::vector<Batch> out;
  const int64_t begin = train_end_;
  const int64_t avail = static_cast<int64_t>(tokens_.size()) - begin - 1;
  if (avail < T + 1) return out;
  const int64_t per_batch = B * T;
  int64_t cursor = begin;
  for (int c = 0; c < count; ++c) {
    Batch b;
    b.B = B;
    b.T = T;
    b.ids.resize(static_cast<size_t>(per_batch));
    b.targets.resize(static_cast<size_t>(per_batch));
    for (int64_t i = 0; i < B; ++i) {
      int64_t off = cursor + i * T;
      if (off + T + 1 > static_cast<int64_t>(tokens_.size()))
        off = begin + ((off - begin) % std::max<int64_t>(1, avail - T));
      for (int64_t t = 0; t < T; ++t) {
        const size_t src = static_cast<size_t>(off + t);
        b.ids[static_cast<size_t>(i * T + t)] = src < tokens_.size() ? tokens_[src] : 0;
        b.targets[static_cast<size_t>(i * T + t)] =
            src + 1 < tokens_.size() ? tokens_[src + 1] : -100;
      }
    }
    out.push_back(std::move(b));
    cursor += per_batch;
    if (cursor + T + 1 > static_cast<int64_t>(tokens_.size())) cursor = begin;
  }
  return out;
}

std::vector<Batch> TokenDataset::batches_from_sequences(
    const std::vector<std::vector<int32_t>>& seqs, int64_t B, int64_t T) {
  std::vector<Batch> out;
  for (size_t s = 0; s < seqs.size(); s += static_cast<size_t>(B)) {
    Batch b;
    b.B = B;
    b.T = T;
    b.ids.assign(static_cast<size_t>(B * T), 0);
    b.targets.assign(static_cast<size_t>(B * T), -100);
    bool any = false;
    for (int64_t i = 0; i < B; ++i) {
      const size_t si = s + static_cast<size_t>(i);
      if (si >= seqs.size()) break;
      const std::vector<int32_t>& seq = seqs[si];
      const int64_t len = std::min<int64_t>(static_cast<int64_t>(seq.size()), T + 1);
      for (int64_t t = 0; t + 1 < len; ++t) {
        b.ids[static_cast<size_t>(i * T + t)] = seq[static_cast<size_t>(t)];
        b.targets[static_cast<size_t>(i * T + t)] = seq[static_cast<size_t>(t + 1)];
        any = true;
      }
    }
    if (any) out.push_back(std::move(b));
  }
  return out;
}

// ============================================================ MixtureDataset
namespace {

std::string read_all(const std::string& path, bool* ok) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    *ok = false;
    return {};
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  *ok = true;
  return ss.str();
}

}  // namespace

bool MixtureDataset::add_source(const std::string& name, const std::string& path, Lang lang,
                                double weight, const Tokenizer& tok, std::string* err) {
  bool ok = false;
  const std::string text = read_all(path, &ok);
  if (!ok || text.empty()) {
    if (err) *err = "cannot read " + path;
    return false;
  }
  Entry e;
  e.info.name = name;
  e.info.path = path;
  e.info.weight = weight;
  e.info.bytes = text.size();
  e.info.lang = (lang == Lang::kUnknown) ? detect_language(text) : lang;
  e.data = std::make_unique<TokenDataset>();
  std::vector<int32_t> ids = tok.encode(text);
  if (ids.size() < 64) {
    if (err) *err = path + " is too small after tokenisation";
    return false;
  }
  e.info.tokens = ids.size();
  e.info.chars_per_token =
      static_cast<double>(utf8_length(tok.preprocess(text))) / static_cast<double>(ids.size());
  e.data->set_tokens(std::move(ids));
  sources_.push_back(std::move(e));
  return true;
}

bool MixtureDataset::add_spec(const std::string& spec, const Tokenizer& tok,
                             std::string* err) {
  size_t pos = 0;
  int added = 0;
  while (pos <= spec.size()) {
    const size_t comma = spec.find(',', pos);
    const std::string item =
        spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (!item.empty()) {
      // name=path[:weight]   |   path
      std::string name, path, wtxt;
      const size_t eq = item.find('=');
      if (eq == std::string::npos) {
        path = item;
      } else {
        name = item.substr(0, eq);
        path = item.substr(eq + 1);
      }
      const size_t colon = path.rfind(':');
      if (colon != std::string::npos && colon + 1 < path.size() &&
          path.find_first_not_of("0123456789.eE-+", colon + 1) == std::string::npos) {
        wtxt = path.substr(colon + 1);
        path = path.substr(0, colon);
      }
      double weight = 0.0;
      if (!wtxt.empty()) {
        try {
          weight = std::stod(wtxt);
        } catch (...) {
          weight = 0.0;
        }
      }
      Lang lang = Lang::kUnknown;
      if (!name.empty()) parse_lang(name, &lang);
      if (name.empty()) {
        const size_t slash = path.find_last_of('/');
        name = slash == std::string::npos ? path : path.substr(slash + 1);
      }
      if (!add_source(name, path, lang, weight > 0 ? weight : 1.0, tok, err)) return false;
      if (weight <= 0.0) sources_.back().info.weight = 0.0;  // means "equal share"
      ++added;
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (!added) {
    if (err) *err = "empty mixture specification";
    return false;
  }
  // Sources without an explicit weight share what is left of the probability
  // mass (or everything, when nobody specified a weight).
  double explicit_sum = 0.0;
  int implicit = 0;
  for (const Entry& e : sources_) {
    if (e.info.weight > 0.0) explicit_sum += e.info.weight;
    else ++implicit;
  }
  if (implicit) {
    const double rest = std::max(0.0, 1.0 - explicit_sum) / static_cast<double>(implicit);
    for (Entry& e : sources_)
      if (e.info.weight <= 0.0) e.info.weight = rest > 0 ? rest : 1.0 / static_cast<double>(sources_.size());
  }
  return true;
}

int MixtureDataset::find_source(const std::string& name) const {
  for (size_t i = 0; i < sources_.size(); ++i)
    if (sources_[i].info.name == name) return static_cast<int>(i);
  return -1;
}

int MixtureDataset::find_lang(Lang l) const {
  for (size_t i = 0; i < sources_.size(); ++i)
    if (sources_[i].info.lang == l) return static_cast<int>(i);
  return -1;
}

std::vector<double> MixtureDataset::normalised_weights() const {
  std::vector<double> w;
  double sum = 0.0;
  for (const Entry& e : sources_) sum += std::max(0.0, e.info.weight);
  for (const Entry& e : sources_)
    w.push_back(sum > 0 ? std::max(0.0, e.info.weight) / sum
                        : 1.0 / static_cast<double>(std::max<size_t>(1, sources_.size())));
  return w;
}

Batch MixtureDataset::sample_batch(int64_t B, int64_t T, Rng& rng) const {
  if (sources_.empty()) return Batch{};
  const std::vector<double> w = normalised_weights();
  const double u = static_cast<double>(rng.uniform());
  double acc = 0.0;
  int pick = static_cast<int>(sources_.size()) - 1;
  for (size_t i = 0; i < w.size(); ++i) {
    acc += w[i];
    if (u <= acc) {
      pick = static_cast<int>(i);
      break;
    }
  }
  return sample_batch_from(pick, B, T, rng);
}

Batch MixtureDataset::sample_batch_from(int source, int64_t B, int64_t T, Rng& rng) const {
  Batch b = sources_[static_cast<size_t>(source)].data->sample_batch(B, T, rng);
  b.lang = sources_[static_cast<size_t>(source)].info.lang;
  b.source = source;
  return b;
}

std::vector<Batch> MixtureDataset::sample_round_robin(int64_t B, int64_t T, Rng& rng,
                                                      int per_source) const {
  std::vector<Batch> out;
  for (int s = 0; s < num_sources(); ++s)
    for (int k = 0; k < per_source; ++k) out.push_back(sample_batch_from(s, B, T, rng));
  return out;
}

std::vector<Batch> MixtureDataset::holdout_batches(int source, int64_t B, int64_t T,
                                                    int count) const {
  std::vector<Batch> out =
      sources_[static_cast<size_t>(source)].data->holdout_batches(B, T, count);
  for (Batch& b : out) {
    b.lang = sources_[static_cast<size_t>(source)].info.lang;
    b.source = source;
  }
  return out;
}

size_t MixtureDataset::total_tokens() const {
  size_t n = 0;
  for (const Entry& e : sources_) n += e.data->num_tokens();
  return n;
}

std::string MixtureDataset::describe() const {
  const std::vector<double> w = normalised_weights();
  std::ostringstream os;
  for (size_t i = 0; i < sources_.size(); ++i) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "  %-4s %-3s w=%.2f  %8zu tokens (holdout %6lld)  %.2f chars/token  %s\n",
                  sources_[i].info.name.c_str(), lang_code(sources_[i].info.lang), w[i],
                  sources_[i].info.tokens,
                  static_cast<long long>(sources_[i].data->holdout_tokens()),
                  sources_[i].info.chars_per_token, sources_[i].info.path.c_str());
    os << buf;
  }
  return os.str();
}

}  // namespace slm
