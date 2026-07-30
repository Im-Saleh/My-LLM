#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One command to (re)build the whole data side of the project:
#
#   fetch real corpora  ->  clean/dedup  ->  train the tokenizer  ->  tokenise
#   into memory-mappable binaries  ->  print fertility and token statistics
#
# Everything is incremental: already downloaded shards and already produced
# binaries are reused unless --force is given.
#
#   ./scripts/update_data.sh                        # ~1 GB of real data
#   ./scripts/update_data.sh --budget-mb 4000       # a lot more Persian
#   ./scripts/update_data.sh --vocab 16384 --force
#   ./scripts/update_data.sh --synthetic-only       # no download, templates only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build/slm}"
DATA="$ROOT/data"
TOKENS="$DATA/tokens"
VOCAB=8192
BUDGET=1000
SHARE="fa=0.55,en=0.15,py=0.30"
TOK_BUDGET=48
FORCE=0
SYNTH_ONLY=0
THREADS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vocab) VOCAB="${2:?}"; shift 2 ;;
    --budget-mb) BUDGET="${2:?}"; shift 2 ;;
    --share) SHARE="${2:?}"; shift 2 ;;
    --tok-budget-mb) TOK_BUDGET="${2:?}"; shift 2 ;;
    --threads) THREADS="${2:?}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --synthetic-only) SYNTH_ONLY=1; shift ;;
    -h|--help) sed -n '3,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$BIN" ]] || { echo "build first: ./build.sh"; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
mkdir -p "$DATA" "$TOKENS"

echo "==> 1/4  synthetic instruction corpus (fa / en / py, deterministic)"
if [[ $FORCE == 1 || ! -f "$DATA/synth/fa.txt" ]]; then
  python3 "$ROOT/scripts/make_trilingual_data.py" --out-dir "$DATA/synth" --turns 12000
else
  echo "    reusing $DATA/synth"
fi

if [[ $SYNTH_ONLY == 0 ]]; then
  echo "==> 2/4  real corpora (Persian Wikipedia, Python, English) + mined bridge pairs"
  if ! python3 -c "import pyarrow" 2>/dev/null; then
    echo "    installing pyarrow (needed to read the HuggingFace parquet files)"
    python3 -m pip install --quiet pyarrow || {
      echo "    pip failed - run with --synthetic-only or install pyarrow manually" >&2
      exit 1; }
  fi
  if [[ $FORCE == 1 || ! -f "$DATA/fa.txt" ]]; then
    python3 "$ROOT/scripts/fetch_data.py" --all --budget-mb "$BUDGET" --share "$SHARE" \
      --out-dir "$DATA" --cache-dir "$DATA/raw"
  else
    echo "    reusing $DATA/{fa,en,py,bridge}.txt (use --force to refetch)"
  fi
else
  echo "==> 2/4  skipped (--synthetic-only)"
fi

# Which text files exist, and with which weight in the tokenizer mixture.
TOK_MIX=""
add_tok() { [[ -s "$2" ]] && TOK_MIX="${TOK_MIX:+$TOK_MIX,}$1=$2:$3"; }
add_tok fa     "$DATA/fa.txt"        0.45
add_tok py     "$DATA/py.txt"        0.25
add_tok en     "$DATA/en.txt"        0.12
add_tok bridge "$DATA/bridge.txt"    0.08
add_tok sfa    "$DATA/synth/fa.txt"  0.06
add_tok spy    "$DATA/synth/py.txt"  0.04
[[ -n "$TOK_MIX" ]] || { echo "no corpora found"; exit 1; }

echo "==> 3/4  tokenizer (vocab $VOCAB, weighted over every source)"
if [[ $FORCE == 1 || ! -f "$DATA/tok.slmtok" ]]; then
  "$BIN" tokenizer --mix "$TOK_MIX" --out "$DATA/tok.slmtok" \
    --vocab "$VOCAB" --budget-mb "$TOK_BUDGET"
else
  echo "    reusing $DATA/tok.slmtok"
fi

echo "==> 4/4  tokenise everything into memory-mappable uint16 binaries"
TOKENIZE_IN=""
for pair in "fa:$DATA/fa.txt" "en:$DATA/en.txt" "py:$DATA/py.txt" \
            "bridge:$DATA/bridge.txt" "sfa:$DATA/synth/fa.txt" \
            "sen:$DATA/synth/en.txt" "spy:$DATA/synth/py.txt"; do
  name="${pair%%:*}"; path="${pair#*:}"
  [[ -s "$path" ]] && TOKENIZE_IN="${TOKENIZE_IN:+$TOKENIZE_IN,}$name=$path"
done
"$BIN" tokenize --tokenizer "$DATA/tok.slmtok" --in "$TOKENIZE_IN" \
  --out-dir "$TOKENS" --threads "$THREADS"

cat <<EOF

data is ready
  tokenizer : $DATA/tok.slmtok
  binaries  : $TOKENS/*.bin   (memory mapped, no RAM cost)

pre-training mixture (Persian first):
  --mix fa=$TOKENS/fa.bin:0.50,py=$TOKENS/py.bin:0.25,en=$TOKENS/en.bin:0.15,bridge=$TOKENS/bridge.bin:0.10

instruction (SFT) mixture:
  --mix sfa=$TOKENS/sfa.bin:0.40,spy=$TOKENS/spy.bin:0.20,sen=$TOKENS/sen.bin:0.12,bridge=$TOKENS/bridge.bin:0.18,fa=$TOKENS/fa.bin:0.07,py=$TOKENS/py.bin:0.03

next: ./scripts/train_stages.sh --steps 4000
EOF
