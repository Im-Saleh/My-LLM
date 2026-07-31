#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One command install:  ./install.sh
#
# Builds slm with the GUI and llama.cpp, installs the binary, both models and a
# desktop launcher, and then `slm` on its own opens the dashboard.
#
#   ./install.sh                    system wide (/usr/local, uses sudo if needed)
#   ./install.sh --prefix ~/.local  no root required
#   ./install.sh --with-olmo        also download OLMo 3 7B (4.5 GB)
#   ./install.sh --no-llama         skip llama.cpp (then only SPT works)
#   ./install.sh --deps             install the build dependencies first
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="/usr/local"
WITH_LLAMA=1
WITH_OLMO=0
DO_DEPS=0
JOBS="$(nproc 2>/dev/null || echo 4)"
BUILD_DIR="$ROOT/build"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="${2:?}"; shift 2 ;;
    --with-olmo) WITH_OLMO=1; shift ;;
    --no-llama) WITH_LLAMA=0; shift ;;
    --deps) DO_DEPS=1; shift ;;
    --jobs|-j) JOBS="${2:?}"; shift 2 ;;
    --build-dir) BUILD_DIR="${2:?}"; shift 2 ;;
    -h|--help) sed -n '3,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn\033[0m %s\n' "$*"; }

# sudo only when the prefix is not writable; a --prefix in $HOME needs none.
SUDO=""
if [[ ! -w "$(dirname "$PREFIX")" || ( -d "$PREFIX" && ! -w "$PREFIX" ) ]]; then
  if [[ $EUID -ne 0 ]]; then
    if command -v sudo >/dev/null; then SUDO="sudo"; else
      echo "need root to write $PREFIX (or pass --prefix ~/.local)" >&2; exit 1
    fi
  fi
fi

# ------------------------------------------------------------------- 1. deps
if [[ $DO_DEPS == 1 ]]; then
  say "installing build dependencies"
  if command -v apt-get >/dev/null; then
    $SUDO apt-get update -qq
    $SUDO apt-get install -y build-essential cmake git pkg-config \
      libglfw3-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxrandr-dev \
      libxinerama-dev libxi-dev libfreetype6-dev libharfbuzz-dev libcurl4-openssl-dev \
      fonts-dejavu-core fonts-noto-core
  elif command -v dnf >/dev/null; then
    $SUDO dnf install -y gcc-c++ cmake git pkgconfig glfw-devel mesa-libGL-devel \
      libX11-devel libXcursor-devel libXrandr-devel libXinerama-devel libXi-devel \
      freetype-devel harfbuzz-devel libcurl-devel dejavu-sans-fonts \
      google-noto-sans-arabic-fonts
  elif command -v pacman >/dev/null; then
    $SUDO pacman -Sy --needed --noconfirm base-devel cmake git glfw mesa \
      freetype2 harfbuzz curl ttf-dejavu noto-fonts
  else
    warn "unknown package manager - install cmake, a C++17 compiler, glfw, OpenGL, freetype, harfbuzz and libcurl yourself"
  fi
fi

# ------------------------------------------------------------------ 2. build
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DSLM_WITH_GUI=ON -DSLM_BUILD_TESTS=OFF
            -DCMAKE_INSTALL_PREFIX="$PREFIX")
if [[ $WITH_LLAMA == 1 ]]; then
  CMAKE_ARGS+=(-DSLM_WITH_LLAMA=ON)
  [[ -d "${SLM_LLAMA_DIR:-}" ]] && CMAKE_ARGS+=(-DSLM_LLAMA_DIR="$SLM_LLAMA_DIR")
  say "building with the GUI and llama.cpp (this downloads llama.cpp once)"
else
  say "building with the GUI, without llama.cpp (SPT only)"
fi
cmake -S "$ROOT" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j "$JOBS"

# ---------------------------------------------------------------- 3. install
say "installing to $PREFIX"
$SUDO cmake --install "$BUILD_DIR"

# The install rules mark the models OPTIONAL so a source tree without them still
# installs; copy them explicitly here and say so if they are missing.
MODELDIR="$PREFIX/share/slm/models"
$SUDO mkdir -p "$MODELDIR"
INSTALLED_MODEL=0
for f in spt.slmtok spt.slm spt-q4.slmq; do
  if [[ -f "$ROOT/models/$f" ]]; then
    $SUDO cp -f "$ROOT/models/$f" "$MODELDIR/"
    INSTALLED_MODEL=1
  fi
done
if [[ $INSTALLED_MODEL == 1 ]]; then
  say "SPT installed into $MODELDIR"
else
  warn "no model found in $ROOT/models - `slm` will create a starter model on first run"
fi

if command -v update-desktop-database >/dev/null; then
  $SUDO update-desktop-database "$PREFIX/share/applications" 2>/dev/null || true
fi

# ------------------------------------------------------------------- 4. OLMo
if [[ $WITH_OLMO == 1 ]]; then
  say "downloading OLMo 3 7B Instruct Q4_K_M (4.5 GB)"
  # Into the shared model directory when writable, otherwise the user's own.
  if [[ -w "$MODELDIR" ]]; then
    "$PREFIX/bin/slm" fetch-model olmo --yes --dir "$MODELDIR"
  else
    "$PREFIX/bin/slm" fetch-model olmo --yes
  fi
fi

# ------------------------------------------------------------------ 5. report
echo
say "installed"
"$PREFIX/bin/slm" up --where | sed 's/^/    /'
echo
printf '  start it:      \033[1;32m%s\033[0m\n' "slm"
printf '  or explicitly: %s\n' "slm up"
printf '  terminal only: %s\n' "slm up --terminal"
if [[ $WITH_OLMO == 0 ]]; then
  printf '  add OLMo:      %s\n' "slm fetch-model olmo"
fi
if [[ ":$PATH:" != *":$PREFIX/bin:"* ]]; then
  echo
  warn "$PREFIX/bin is not on your PATH; add it with:"
  printf '    echo '\''export PATH="%s/bin:$PATH"'\'' >> ~/.profile\n' "$PREFIX"
fi
