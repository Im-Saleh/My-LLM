// SPDX-License-Identifier: Apache-2.0
#include "backend/model_backend.h"

#include <algorithm>

namespace slm {

// The three presets exist so the debate engine does not scatter magic numbers.
// The schedule is the standard one for self-consistency style ensembles: diverge
// first, converge last.  Sampling the *drafts* at a high temperature is what
// makes the votes informative - identical answers cannot outvote each other -
// while the final synthesis has to be near-deterministic or the whole debate is
// thrown away by one unlucky sample.
SamplingParams SamplingParams::creative(uint64_t seed) {
  SamplingParams p;
  p.temperature = 1.0f;
  p.top_k = 64;
  p.top_p = 0.97f;
  p.min_p = 0.03f;
  p.repetition_penalty = 1.05f;
  p.seed = seed;
  return p;
}

SamplingParams SamplingParams::critical(uint64_t seed) {
  SamplingParams p;
  p.temperature = 0.7f;
  p.top_k = 40;
  p.top_p = 0.92f;
  p.min_p = 0.05f;
  p.repetition_penalty = 1.10f;
  p.seed = seed;
  return p;
}

SamplingParams SamplingParams::decisive(uint64_t seed) {
  SamplingParams p;
  p.temperature = 0.35f;
  p.top_k = 20;
  p.top_p = 0.85f;
  p.min_p = 0.10f;
  p.repetition_penalty = 1.05f;
  p.seed = seed;
  return p;
}

std::vector<GenResponse> IModelBackend::generate_many(const std::vector<GenRequest>& reqs,
                                                     const ChunkFn& on_chunk,
                                                     std::atomic<bool>* cancel) {
  // Correct for every backend; overridden only where the runtime can genuinely
  // decode several sequences in one pass.
  std::vector<GenResponse> out;
  out.reserve(reqs.size());
  for (const GenRequest& r : reqs) {
    if (cancel && cancel->load()) {
      GenResponse g;
      g.tag = r.tag;
      g.cancelled = true;
      out.push_back(std::move(g));
      continue;
    }
    out.push_back(generate(r, on_chunk, cancel));
  }
  return out;
}

// ------------------------------------------------------------------ registry
void BackendRegistry::add(BackendPtr b) {
  if (!b) return;
  std::lock_guard<std::mutex> g(m_);
  for (BackendPtr& e : v_)
    if (e->id() == b->id()) {
      e = b;  // replacing a backend by id keeps GUI selections valid
      return;
    }
  v_.push_back(std::move(b));
}

BackendPtr BackendRegistry::get(const std::string& id) const {
  std::lock_guard<std::mutex> g(m_);
  for (const BackendPtr& e : v_)
    if (e->id() == id) return e;
  return nullptr;
}

std::vector<BackendPtr> BackendRegistry::all() const {
  std::lock_guard<std::mutex> g(m_);
  return v_;
}

std::vector<std::string> BackendRegistry::ids() const {
  std::lock_guard<std::mutex> g(m_);
  std::vector<std::string> out;
  out.reserve(v_.size());
  for (const BackendPtr& e : v_) out.push_back(e->id());
  return out;
}

size_t BackendRegistry::size() const {
  std::lock_guard<std::mutex> g(m_);
  return v_.size();
}

size_t BackendRegistry::weight_bytes() const {
  std::lock_guard<std::mutex> g(m_);
  size_t n = 0;
  for (const BackendPtr& e : v_)
    if (e->loaded()) n += e->status().weight_bytes;
  return n;
}

size_t BackendRegistry::kv_bytes() const {
  std::lock_guard<std::mutex> g(m_);
  size_t n = 0;
  for (const BackendPtr& e : v_)
    if (e->loaded()) n += e->status().kv_bytes;
  return n;
}

}  // namespace slm
