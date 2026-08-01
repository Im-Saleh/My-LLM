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
# A cache from an older configure still carries BUILD_SHARED_LIBS=ON, which is
# exactly what produced the libllama.so failure; start clean when the setting
# that matters differs.
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] && [[ $WITH_LLAMA == 1 ]]; then
  if grep -q '^BUILD_SHARED_LIBS:BOOL=ON' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    say "discarding a stale build directory (it was configured for shared libraries)"
    rm -rf "$BUILD_DIR"
  fi
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

# ------------------------------------------------------- 4. does it actually run
# Checked here, not at the end: every step below uses the binary, and a dynamic
# linker failure at this point is far easier to read than the same failure three
# steps later.
say "verifying the installed binary"
if ! VERIFY_OUT="$("$PREFIX/bin/slm" --version 2>&1)"; then
  echo >&2
  echo "the installed binary does not start:" >&2
  echo "$VERIFY_OUT" | sed 's/^/    /' >&2
  echo >&2
  if grep -q "libllama\|libggml" <<<"$VERIFY_OUT"; then
    cat >&2 <<'EOT'
This is the shared-library problem fixed in this version: llama.cpp was linked
dynamically and its .so files are not on the loader path.  Rebuild from a clean
build directory so the static link takes effect:

    rm -rf build && ./install.sh --prefix PREFIX

If you must keep a shared llama.cpp, configure with -DSLM_LLAMA_SHARED=ON, which
installs the libraries and sets an RPATH.
EOT
  elif grep -q "libtorch\|libc10" <<<"$VERIFY_OUT"; then
    echo "libtorch was linked dynamically; export LD_LIBRARY_PATH=<libtorch>/lib" >&2
  fi
  exit 1
fi
printf '    %s\n' "$VERIFY_OUT"

# ------------------------------------------------------------------- 5. OLMo
if [[ $WITH_OLMO == 1 ]]; then
  # Prefer the shared model directory so every user of the machine sees it.
  OLMO_DIR="$MODELDIR"
  [[ -w "$OLMO_DIR" ]] || OLMO_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/slm/models"
  OLMO_FILE="$OLMO_DIR/allenai_Olmo-3-7B-Instruct-Q4_K_M.gguf"
  # Look for an existing GGUF *everywhere* the program can find one, not just in
  # the directory we would download into: a 4.5 GB re-download because the file
  # sits one directory over is unacceptable.
  EXISTING=""
  for d in "$OLMO_DIR" "$MODELDIR" \
           "${XDG_DATA_HOME:-$HOME/.local/share}/slm/models" \
           "${XDG_DATA_HOME:-$HOME/.local/share}/slm" \
           /usr/share/slm/models /usr/local/share/slm/models \
           "$HOME/models" "$ROOT/models"; do
    [[ -d "$d" ]] || continue
    while IFS= read -r -d '' f; do
      # Anything over 1 GB starting with the GGUF magic is a real model file.
      if [[ "$(stat -c%s "$f")" -gt 1000000000 ]] && [[ "$(head -c4 "$f")" == "GGUF" ]]; then
        EXISTING="$f"
        break 2
      fi
    done < <(find "$d" -maxdepth 1 -name '*.gguf' -print0 2>/dev/null)
  done
  if [[ -n "$EXISTING" ]]; then
    say "OLMo already present, not downloading again:"
    printf '    %s (%s)\n' "$EXISTING" "$(du -h "$EXISTING" | cut -f1)"
  else
    say "downloading OLMo 3 7B Instruct Q4_K_M (4.5 GB) into $OLMO_DIR"
    # The download needs nothing from slm, so do it with curl directly: that way
    # a broken binary cannot stop the model from arriving, and -C - resumes.
    URL="https://huggingface.co/bartowski/allenai_Olmo-3-7B-Instruct-GGUF/resolve/main/allenai_Olmo-3-7B-Instruct-Q4_K_M.gguf?download=true"
    mkdir -p "$OLMO_DIR"
    if ! curl -L --fail --proto '=https' -C - --retry 5 --retry-delay 3 \
              --retry-all-errors --progress-bar -o "$OLMO_FILE" "$URL"; then
      warn "the OLMo download did not finish.  Resume it any time with:"
      printf '    slm fetch-model olmo\n'
    else
      # A truncated download or an HTML error page is worse than none: check the
      # GGUF magic before declaring success.
      if [[ "$(head -c4 "$OLMO_FILE")" != "GGUF" ]]; then
        warn "$OLMO_FILE does not start with the GGUF magic - removing it"
        rm -f "$OLMO_FILE"
      else
        say "OLMo installed ($(du -h "$OLMO_FILE" | cut -f1))"
      fi
    fi
  fi
fi

# ------------------------------------------------------------------ 6. report
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
printf '  update later:  %s\n' "./update.sh"
if [[ ":$PATH:" != *":$PREFIX/bin:"* ]]; then
  echo
  warn "$PREFIX/bin is not on your PATH; add it with:"
  printf '    echo '\''export PATH="%s/bin:$PATH"'\'' >> ~/.profile\n' "$PREFIX"
fi
