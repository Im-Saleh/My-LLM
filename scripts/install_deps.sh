#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Installs the build dependencies on the common Linux distributions.
#
#   sudo ./scripts/install_deps.sh          # everything, including the GUI
#   sudo ./scripts/install_deps.sh --no-gui # compiler + cmake only
set -euo pipefail
GUI=1
[[ "${1:-}" == "--no-gui" ]] && GUI=0

if command -v apt-get >/dev/null; then
  apt-get update
  apt-get install -y build-essential cmake git python3 libgomp1
  [[ $GUI == 1 ]] && apt-get install -y libglfw3-dev libgl1-mesa-dev xorg-dev
elif command -v dnf >/dev/null; then
  dnf install -y gcc-c++ cmake git python3 libgomp
  [[ $GUI == 1 ]] && dnf install -y mesa-libGL-devel libX11-devel libXrandr-devel \
      libXinerama-devel libXcursor-devel libXi-devel
elif command -v pacman >/dev/null; then
  pacman -Sy --noconfirm base-devel cmake git python glfw mesa
elif command -v zypper >/dev/null; then
  zypper install -y gcc-c++ cmake git python3 Mesa-libGL-devel libX11-devel
else
  echo "unknown package manager - install: a C++17 compiler, cmake >= 3.16, git," >&2
  echo "python3, OpenMP, and (for the GUI) OpenGL + X11 development headers." >&2
  exit 1
fi
echo "dependencies installed. Next: ./build.sh"
