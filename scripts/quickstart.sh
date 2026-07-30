#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# End to end: corpus -> tokenizer -> base model -> live self-training session.
#
#   ./scripts/quickstart.sh                    # tiny config, ~3 minutes
#   ./scripts/quickstart.sh --config configs/slm-demo.conf --steps 500
#   ./scripts/quickstart.sh --gui              # open the ImGui dashboard at the end
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build/slm}"
RUN="${RUN:-$ROOT/run}"
CONFIG="$ROOT/configs/slm-tiny.conf"
STEPS=200
VOCAB=2048
SECONDS_LIVE=90
GUI=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:?}"; shift 2 ;;
    --steps) STEPS="${2:?}"; shift 2 ;;
    --vocab) VOCAB="${2:?}"; shift 2 ;;
    --seconds) SECONDS_LIVE="${2:?}"; shift 2 ;;
    --gui) GUI=1; shift ;;
    -h|--help) sed -n '3,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$BIN" ]] || { echo "build first: ./build.sh"; exit 1; }
mkdir -p "$RUN"

CORPUS="$ROOT/data/sample_corpus.txt"
if [[ ! -f "$CORPUS" ]]; then
  echo "==> generating the sample corpus"
  python3 "$ROOT/scripts/make_sample_data.py" "$CORPUS"
fi

echo "==> 1/4 tokenizer"
"$BIN" tokenizer --input "$CORPUS" --out "$RUN/tok.slmtok" --vocab "$VOCAB"

echo "==> 2/4 base model ($STEPS steps, config $(basename "$CONFIG"))"
"$BIN" pretrain --data "$CORPUS" --tokenizer "$RUN/tok.slmtok" --out "$RUN/base.slm" \
  --config "$CONFIG" --steps "$STEPS" --save-every 100

echo "==> 3/4 sanity check"
"$BIN" eval --ckpt "$RUN/base.slm" --tokenizer "$RUN/tok.slmtok" --data "$CORPUS"
"$BIN" chat --ckpt "$RUN/base.slm" --tokenizer "$RUN/tok.slmtok" \
  --prompt '<|user|>what is the capital of France?<|assistant|>' --max-new 32

echo "==> 4/4 live self-training"
if [[ "$GUI" == 1 ]]; then
  "$BIN" dashboard --ckpt "$RUN/base.slm" --tokenizer "$RUN/tok.slmtok" --data "$CORPUS" \
    --config "$CONFIG" --workdir "$RUN/session"
else
  "$BIN" live --ckpt "$RUN/base.slm" --tokenizer "$RUN/tok.slmtok" --data "$CORPUS" \
    --config "$CONFIG" --workdir "$RUN/session" --autopilot --seconds "$SECONDS_LIVE" --threads 2
  echo
  echo "audit trail: $RUN/session/audit.jsonl"
  echo "checkpoints: $(ls "$RUN/session" | tr '\n' ' ')"
fi
