#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Optional: fetch a real public-domain corpus for a more interesting model.
# The repository itself ships only the synthetic corpus produced by
# scripts/make_sample_data.py, so nothing third-party is vendored here.
#
#   ./scripts/fetch_data.sh              -> data/tinyshakespeare.txt (~1.1 MB)
#   ./scripts/fetch_data.sh --mixed      -> also concatenates the synthetic corpus
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data"
OUT="$ROOT/data/tinyshakespeare.txt"
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

echo "==> downloading tiny shakespeare (public domain)"
if command -v curl >/dev/null; then
  curl -fsSL "$URL" -o "$OUT"
else
  wget -qO "$OUT" "$URL"
fi
wc -c "$OUT"

if [[ "${1:-}" == "--mixed" ]]; then
  python3 "$ROOT/scripts/make_sample_data.py" "$ROOT/data/sample_corpus.txt"
  cat "$ROOT/data/sample_corpus.txt" "$OUT" > "$ROOT/data/mixed_corpus.txt"
  wc -c "$ROOT/data/mixed_corpus.txt"
  echo "use --data data/mixed_corpus.txt"
fi
