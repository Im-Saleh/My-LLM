#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Builds dist/slm_<version>_<arch>.deb.
#
# Uses dpkg-deb when it is available and falls back to assembling the archive
# with `ar` + `tar`, which is exactly what a .deb is:
#     ar archive { debian-binary, control.tar.gz, data.tar.gz }
# so the package can be produced on a non-Debian machine (or in a container
# without dpkg installed) and still installs with `dpkg -i` / `apt install`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build"
VERSION="0.2.0"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="${2:?}"; shift 2 ;;
    --version) VERSION="${2:?}"; shift 2 ;;
    --arch) ARCH="${2:?}"; shift 2 ;;
    -h|--help) sed -n '3,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Relative build dirs would break the `cd` + tar steps below.
BUILD_DIR="$(cd "$BUILD_DIR" 2>/dev/null && pwd || echo "$BUILD_DIR")"
BIN="$BUILD_DIR/slm"
[[ -x "$BIN" ]] || { echo "no binary at $BIN - run ./build.sh first" >&2; exit 1; }

PKG="slm"
STAGE="$BUILD_DIR/deb/$PKG"
DIST="$ROOT/dist"
rm -rf "$STAGE"
mkdir -p "$STAGE/DEBIAN" \
         "$STAGE/usr/bin" \
         "$STAGE/usr/share/$PKG/configs" \
         "$STAGE/usr/share/$PKG/scripts" \
         "$STAGE/usr/share/$PKG/data" \
         "$STAGE/usr/share/doc/$PKG" \
         "$DIST"

install -m 0755 "$BIN" "$STAGE/usr/bin/slm"
[[ -x "$BUILD_DIR/slm_gradcheck" ]] && install -m 0755 "$BUILD_DIR/slm_gradcheck" "$STAGE/usr/bin/slm-gradcheck"
install -m 0644 "$ROOT"/configs/*.conf "$STAGE/usr/share/$PKG/configs/"
install -m 0755 "$ROOT"/scripts/make_sample_data.py "$STAGE/usr/share/$PKG/scripts/"
install -m 0755 "$ROOT"/scripts/fetch_data.sh "$STAGE/usr/share/$PKG/scripts/"
install -m 0755 "$ROOT"/scripts/quickstart.sh "$STAGE/usr/share/$PKG/scripts/"
install -m 0644 "$ROOT"/README.md "$STAGE/usr/share/doc/$PKG/"
install -m 0644 "$ROOT"/docs/*.md "$STAGE/usr/share/doc/$PKG/"
[[ -f "$ROOT/LICENSE" ]] && install -m 0644 "$ROOT/LICENSE" "$STAGE/usr/share/doc/$PKG/copyright"
if [[ -f "$ROOT/data/sample_corpus.txt" ]]; then
  gzip -9 -c "$ROOT/data/sample_corpus.txt" > "$STAGE/usr/share/$PKG/data/sample_corpus.txt.gz"
fi

SIZE_KB="$(du -sk "$STAGE" | cut -f1)"

# Derive Depends from what the binary actually links against, so a --no-gui
# build does not drag in OpenGL/X11 and a GUI build does not miss them.
LIBS="$(ldd "$BIN" 2>/dev/null || true)"
DEPS="libc6 (>= 2.34), libstdc++6 (>= 11)"
case "$LIBS" in *libgomp*)    DEPS="$DEPS, libgomp1" ;; esac
case "$LIBS" in *libGLX*)     DEPS="$DEPS, libglx0" ;; esac
case "$LIBS" in *libOpenGL*)  DEPS="$DEPS, libopengl0" ;; esac
case "$LIBS" in *libGL.so*)   DEPS="$DEPS, libgl1" ;; esac
case "$LIBS" in *libX11*)     DEPS="$DEPS, libx11-6" ;; esac
case "$LIBS" in *libXext*)    DEPS="$DEPS, libxext6" ;; esac
case "$LIBS" in *libtorch*)   DEPS="$DEPS, libtorch (>= 2.0)" ;; esac
echo "==> Depends: $DEPS"
cat > "$STAGE/DEBIAN/control" <<EOF
Package: $PKG
Version: $VERSION
Section: science
Priority: optional
Architecture: $ARCH
Maintainer: slm maintainers <slm@example.org>
Depends: $DEPS
Recommends: python3
Installed-Size: $SIZE_KB
Homepage: https://github.com/Im-Saleh/My-LLM
Description: Small language model with a hybrid self-training pipeline
 A decoder-only transformer written in C++ with its own tensor/autograd engine,
 a live Dear ImGui monitoring dashboard and three concurrent self-training
 mechanisms (continual learning, self-generated data and preference feedback)
 whose updates are merged by a coordinator that validates every change on a
 held-out set before it is allowed to land.
 .
 Installed helpers live in /usr/share/slm (configs, sample corpus, scripts).
EOF

cat > "$STAGE/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if [ "$1" = configure ]; then
  echo "slm installed. Quick start:"
  echo "  mkdir -p ~/slm-run && cd ~/slm-run"
  echo "  zcat /usr/share/slm/data/sample_corpus.txt.gz > corpus.txt"
  echo "  slm tokenizer --input corpus.txt --out tok.slmtok --vocab 2048"
  echo "  slm pretrain  --data corpus.txt --tokenizer tok.slmtok --out base.slm \\"
  echo "                --config /usr/share/slm/configs/slm-demo.conf --steps 500"
  echo "  slm dashboard --ckpt base.slm --tokenizer tok.slmtok --data corpus.txt \\"
  echo "                --config /usr/share/slm/configs/slm-demo.conf"
fi
exit 0
EOF
chmod 0755 "$STAGE/DEBIAN/postinst"

# md5sums over the data tree
( cd "$STAGE" && find usr -type f -print0 | sort -z | xargs -0 md5sum > DEBIAN/md5sums )

OUT="$DIST/${PKG}_${VERSION}_${ARCH}.deb"
if command -v dpkg-deb >/dev/null 2>&1; then
  echo "==> dpkg-deb --build"
  dpkg-deb --root-owner-group --build "$STAGE" "$OUT" >/dev/null
else
  echo "==> assembling the archive manually (no dpkg-deb on this system)"
  WORK="$BUILD_DIR/deb/work"
  rm -rf "$WORK"; mkdir -p "$WORK"
  printf '2.0\n' > "$WORK/debian-binary"
  TAR_OPTS="--owner=root --group=root --numeric-owner --mtime=@0"
  # shellcheck disable=SC2086
  ( cd "$STAGE/DEBIAN" && tar $TAR_OPTS -czf "$WORK/control.tar.gz" ./* )
  # shellcheck disable=SC2086
  ( cd "$STAGE" && tar $TAR_OPTS --exclude=./DEBIAN -czf "$WORK/data.tar.gz" ./usr )
  rm -f "$OUT"
  if command -v python3 >/dev/null 2>&1; then
    # Write the ar container ourselves: GNU ar appends a '/' terminator to
    # member names, which some strict readers dislike, and we want a fully
    # deterministic archive.
    python3 - "$OUT" "$WORK" <<'PYAR'
import sys, os
out, work = sys.argv[1], sys.argv[2]
with open(out, 'wb') as f:
    f.write(b'!<arch>\n')
    for name in ('debian-binary', 'control.tar.gz', 'data.tar.gz'):
        path = os.path.join(work, name)
        body = open(path, 'rb').read()
        f.write(name.ljust(16).encode())      # name
        f.write(b'0'.ljust(12))               # mtime
        f.write(b'0'.ljust(6))                # uid
        f.write(b'0'.ljust(6))                # gid
        f.write(b'100644'.ljust(8))           # mode
        f.write(str(len(body)).ljust(10).encode())
        f.write(b'`\n')
        f.write(body)
        if len(body) % 2:
            f.write(b'\n')
PYAR
  else
    ( cd "$WORK" && ar rcD "$OUT" debian-binary control.tar.gz data.tar.gz )
  fi
fi

echo
echo "package: $OUT"
ls -la "$OUT"
if command -v dpkg-deb >/dev/null 2>&1; then
  dpkg-deb --info "$OUT" | sed 's/^/    /'
else
  echo "    contents:"
  ar p "$OUT" data.tar.gz | tar tzf - | sed 's/^/      /'
fi
cat <<EOF

install with
    sudo apt install ./$(basename "$OUT")      # or: sudo dpkg -i <file>.deb
remove with
    sudo apt remove slm
EOF
