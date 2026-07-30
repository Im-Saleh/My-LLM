#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The staged training recipe from docs/MULTILINGUAL.fa.md, as a script.
#
#   stage 1  general pre-training   fa 50 / py 25 / en 15 / bridge 10
#   stage 2  code anneal            py 55 / fa 20 / en 15 / bridge 10   (lower LR)
#   stage 3  instruction SFT        synthetic instructions + bridge + retention
#
# Every stage resumes from the previous checkpoint and keeps the same step
# counter, so the warmup-stable-decay schedule and the progressive growth
# schedule stay meaningful.  You can stop and continue at any time: re-running
# the script picks up from the checkpoint on disk.
#
#   ./scripts/train_stages.sh --steps 4000                 # all three stages
#   ./scripts/train_stages.sh --steps 4000 --only 1        # just pre-training
#   ./scripts/train_stages.sh --resume run/fa.slm --only 3 # just the SFT
#   LIBTORCH=/opt/libtorch ./scripts/train_stages.sh       # 4x faster on CPU
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build/slm}"
T="${TOKENS:-$ROOT/data/tokens}"
TOK="${TOK:-$ROOT/data/tok.slmtok}"
RUN="${RUN:-$ROOT/run}"
CONFIG="$ROOT/configs/slm-fa.conf"
STEPS=4000
BATCH=16
ONLY=""
CKPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS="${2:?}"; shift 2 ;;
    --batch) BATCH="${2:?}"; shift 2 ;;
    --config) CONFIG="${2:?}"; shift 2 ;;
    --only) ONLY="${2:?}"; shift 2 ;;
    --resume) CKPT="${2:?}"; shift 2 ;;
    -h|--help) sed -n '3,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$BIN" ]] || { echo "build first: ./build.sh"; exit 1; }
[[ -f "$TOK" ]] || { echo "run ./scripts/update_data.sh first"; exit 1; }
mkdir -p "$RUN"
[[ -n "${LIBTORCH:-}" ]] && export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"

bin() { echo "$T/$1.bin"; }
have() { [[ -s "$T/$1.bin" ]]; }
mix() {  # mix name=weight ...
  local out=""
  for kv in "$@"; do
    local n="${kv%%=*}" w="${kv#*=}"
    have "$n" && out="${out:+$out,}$n=$(bin "$n"):$w"
  done
  echo "$out"
}

S1=$(mix fa=0.50 py=0.25 en=0.15 bridge=0.10)
S2=$(mix py=0.55 fa=0.20 en=0.15 bridge=0.10)
S3=$(mix sfa=0.40 spy=0.20 sen=0.12 bridge=0.18 fa=0.07 py=0.03)

# 70% pre-training, 15% code anneal, 15% instruction tuning
S1_STEPS=$(( STEPS * 70 / 100 ))
S2_STEPS=$(( STEPS * 15 / 100 ))
S3_STEPS=$(( STEPS - S1_STEPS - S2_STEPS ))
TOTAL=$STEPS
OUT="$RUN/fa.slm"

run_stage() {
  local n="$1" mixspec="$2" steps="$3" lr="$4" extra="${5:-}"
  [[ -n "$ONLY" && "$ONLY" != "$n" ]] && return 0
  [[ -z "$mixspec" ]] && { echo "stage $n: no data, skipped"; return 0; }
  echo
  echo "=============== stage $n : $steps steps, lr $lr ==============="
  local resume=()
  if [[ -n "$CKPT" ]]; then
    resume=(--resume "$CKPT")
  elif [[ -f "$OUT" ]]; then
    resume=(--resume "$OUT")
  fi
  # shellcheck disable=SC2086
  "$BIN" pretrain --mix "$mixspec" --tokenizer "$TOK" --out "$OUT" "${resume[@]}" \
    --config "$CONFIG" --batch "$BATCH" --steps "$steps" --total-steps "$TOTAL" \
    --lr "$lr" --log-every 100 --eval-every 500 --save-every 250 $extra
  CKPT=""
  cp -f "$OUT" "$RUN/fa-stage$n.slm"
}

run_stage 1 "$S1" "$S1_STEPS" 1.5e-3
run_stage 2 "$S2" "$S2_STEPS" 5e-4
run_stage 3 "$S3" "$S3_STEPS" 6e-4 "--set train.decay_frac=0.5"

echo
echo "=============== evaluation ==============="
EVAL=$(mix fa=1 en=1 py=1 sfa=1)
"$BIN" eval --ckpt "$OUT" --tokenizer "$TOK" --mix "$EVAL" --batches 8 || true
"$BIN" langcheck --ckpt "$OUT" --tokenizer "$TOK" --mix "$EVAL" --samples 8 || true

cat <<EOF

checkpoints
  $OUT                 (final)
  $RUN/fa-stage{1,2,3}.slm

try it
  $BIN chat --ckpt $OUT --tokenizer $TOK
  $BIN dashboard --ckpt $OUT --tokenizer $TOK --config $CONFIG \\
       --mix $(mix fa=0.5 py=0.25 en=0.15 bridge=0.10) --autopilot
EOF
