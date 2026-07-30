#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One shot build script.
#
#   ./build.sh                     release build with the ImGui dashboard
#   ./build.sh --no-gui            headless build (no OpenGL/GLFW needed)
#   ./build.sh --libtorch /opt/libtorch
#   ./build.sh --deb               build, then produce dist/slm_<ver>_amd64.deb
#   ./build.sh --run-tests         build, then run the gradient checks
#   ./build.sh --native            add -march=native (do not use for packaging)
#   ./build.sh --jobs 4
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
GUI=ON
DEB=0
TESTS=0
NATIVE=OFF
TORCH_DIR=""
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-gui) GUI=OFF; shift ;;
    --gui) GUI=ON; shift ;;
    --deb) DEB=1; shift ;;
    --run-tests|--test) TESTS=1; shift ;;
    --native) NATIVE=ON; shift ;;
    --libtorch) TORCH_DIR="${2:?--libtorch needs a path}"; shift 2 ;;
    --jobs|-j) JOBS="${2:?}"; shift 2 ;;
    --build-dir) BUILD_DIR="${2:?}"; shift 2 ;;
    -h|--help) sed -n '3,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

echo "==> checking the toolchain"
command -v cmake >/dev/null || { echo "cmake is required (see scripts/install_deps.sh)"; exit 1; }
command -v c++   >/dev/null || { echo "a C++17 compiler is required"; exit 1; }
cmake --version | head -1
c++ --version | head -1

CMAKE_ARGS=(
  -S "$ROOT" -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DSLM_WITH_GUI="$GUI"
  -DSLM_NATIVE_ARCH="$NATIVE"
)
if [[ -n "$TORCH_DIR" ]]; then
  CMAKE_ARGS+=(-DSLM_WITH_LIBTORCH=ON "-DCMAKE_PREFIX_PATH=$TORCH_DIR")
fi
# Allow fully offline builds: point these at local clones of the deps.
[[ -n "${IMGUI_DIR:-}" ]] && CMAKE_ARGS+=("-DFETCHCONTENT_SOURCE_DIR_IMGUI=$IMGUI_DIR")
[[ -n "${GLFW_DIR:-}"  ]] && CMAKE_ARGS+=("-DFETCHCONTENT_SOURCE_DIR_GLFW=$GLFW_DIR")

echo "==> configuring (gui=$GUI, native=$NATIVE, jobs=$JOBS)"
cmake "${CMAKE_ARGS[@]}"

echo "==> building"
cmake --build "$BUILD_DIR" -j "$JOBS"

if [[ "$TESTS" == 1 ]]; then
  echo "==> gradient checks"
  "$BUILD_DIR/slm_gradcheck"
  echo "==> text / tokenizer checks"
  "$BUILD_DIR/slm_texttest"
  echo "==> quantised inference checks"
  "$BUILD_DIR/slm_qtest"
fi

echo
echo "built: $BUILD_DIR/slm"
"$BUILD_DIR/slm" info | sed 's/^/    /'

if [[ "$DEB" == 1 ]]; then
  echo "==> packaging"
  "$ROOT/scripts/make_deb.sh" --build-dir "$BUILD_DIR"
fi

cat <<EOF

next steps
  1) sample corpus     python3 scripts/make_sample_data.py data/sample_corpus.txt
  2) tokenizer         $BUILD_DIR/slm tokenizer --input data/sample_corpus.txt --out run/tok.slmtok --vocab 2048
  3) base model        $BUILD_DIR/slm pretrain --data data/sample_corpus.txt --tokenizer run/tok.slmtok \\
                            --out run/base.slm --config configs/slm-demo.conf --steps 500
  4) live dashboard    $BUILD_DIR/slm dashboard --ckpt run/base.slm --tokenizer run/tok.slmtok \\
                            --data data/sample_corpus.txt --config configs/slm-demo.conf
     (no display?)     $BUILD_DIR/slm live      --ckpt run/base.slm --tokenizer run/tok.slmtok \\
                            --data data/sample_corpus.txt --config configs/slm-demo.conf --autopilot
  5) deploy int4       $BUILD_DIR/slm pack --in run/base.slm --out run/base-q4.slmq --bits 4
                       $BUILD_DIR/slm qrun --model run/base-q4.slmq --tokenizer run/tok.slmtok --prompt "..."
EOF
