// SPDX-License-Identifier: Apache-2.0
//
// Finding (or creating) the one model, so that no command ever needs arguments.
//
// The rule: there is exactly one SPT model, it is called `spt`, and every command
// finds it by itself.  `slm up` with no arguments must open a working dashboard
// on a freshly installed machine - the previous behaviour, printing
// "cannot load tokenizer 'run/tok.slmtok'" and exiting 1, is the bug this file
// exists to remove.
//
// Search order (first hit wins), each looking for `spt.slmtok` plus `spt.slm`
// and optionally `spt-q4.slmq`:
//
//   1. $SLM_HOME                      explicit override
//   2. ./models, ./run                a git checkout, or a previous session
//   3. $XDG_DATA_HOME/slm  (~/.local/share/slm)
//   4. /usr/share/slm/models, /usr/local/share/slm/models    a package install
//   5. the directory of the running binary, and ../share/slm/models
//
// If nothing is found and bootstrapping is allowed, a starter model is created in
// the user data directory from a small built-in corpus.  It is genuinely tiny and
// says so loudly, but it means the GUI opens, the self-training threads have
// something to improve, and the user can point the training panel at a real
// dataset instead of being stuck at a command line error.
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace slm {

struct SptAssets {
  std::string tokenizer;   // spt.slmtok
  std::string ckpt;        // spt.slm      (float weights, trainable)
  std::string quant;       // spt-q4.slmq  (int4, inference only; may be empty)
  std::string dir;         // where they were found
  std::string workdir;     // writable scratch for this session
  bool bootstrapped = false;
  std::string note;        // human readable summary of what was found
};

// Directories searched, in order.  Exposed so `slm up --where` can print them.
std::vector<std::string> spt_search_dirs();

// Writable per-user data directory (created on demand).
std::string spt_user_dir();

// Locates the model.  Never throws.  With allow_bootstrap the function creates a
// starter model rather than failing; without it, returns false and explains.
bool resolve_spt(SptAssets* out, bool allow_bootstrap, std::string* err,
                 const std::function<void(const std::string&)>& log = nullptr);

// Trains a small tokenizer and initialises a small model into `dir`.
// `corpus` may be empty, in which case the built-in seed text is used.
bool bootstrap_spt(const std::string& dir, const std::string& corpus_path,
                   SptAssets* out, std::string* err,
                   const std::function<void(const std::string&)>& log = nullptr);

// The built-in multilingual seed corpus (Persian + English + Python).  Small on
// purpose: it exists to make a tokenizer and a runnable model, not to teach.
const char* spt_seed_corpus();

}  // namespace slm
