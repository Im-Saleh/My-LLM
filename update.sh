#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Update an existing installation:  ./update.sh
#
# Pulls, rebuilds, reinstalls, and leaves your data alone.  What is preserved:
#   * the OLMo GGUF (never re-downloaded)
#   * your memory store, audit logs and codebase index under the work directory
#   * a locally trained SPT, unless the repository ships a newer one
#
#   ./update.sh                      pull + rebuild + reinstall to the same prefix
#   ./update.sh --prefix ~/.local    when you installed somewhere else
#   ./update.sh --no-pull            rebuild what is already checked out
#   ./update.sh --keep-model         never overwrite models/spt.slm from the repo
#   ./update.sh --with-olmo          also fetch OLMo if it is missing
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX=""
DO_PULL=1
KEEP_MODEL=0
WITH_OLMO=0
EXTRA=()
JOBS="$(nproc 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="${2:?}"; shift 2 ;;
    --no-pull) DO_PULL=0; shift ;;
    --keep-model) KEEP_MODEL=1; shift ;;
    --with-olmo) WITH_OLMO=1; shift ;;
    --no-llama) EXTRA+=(--no-llama); shift ;;
    --jobs|-j) JOBS="${2:?}"; shift 2 ;;
    -h|--help) sed -n '3,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn\033[0m %s\n' "$*"; }

# ------------------------------------------------- where is it installed now
if [[ -z "$PREFIX" ]]; then
  if command -v slm >/dev/null; then
    BIN="$(command -v slm)"
    PREFIX="$(cd "$(dirname "$BIN")/.." && pwd)"
    say "found an existing install at $PREFIX"
  else
    PREFIX="/usr/local"
    warn "no slm on PATH; assuming $PREFIX"
  fi
fi

OLD_VERSION="$("$PREFIX/bin/slm" --version 2>/dev/null || echo "unknown")"

# --------------------------------------------------------- preserve local state
# A model the user trained themselves must not be clobbered by a git checkout.
STASHED_MODEL=""
if [[ $KEEP_MODEL == 1 && -f "$ROOT/models/spt.slm" ]]; then
  STASHED_MODEL="$(mktemp -d)/spt.slm"
  cp "$ROOT/models/spt.slm" "$STASHED_MODEL"
  say "keeping your models/spt.slm"
fi

# ------------------------------------------------------------------- 1. pull
if [[ $DO_PULL == 1 ]]; then
  if [[ ! -d "$ROOT/.git" ]]; then
    warn "$ROOT is not a git checkout - skipping the pull"
  else
    say "pulling"
    BEFORE="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo none)"
    # Local edits to tracked files would make the pull fail; say so clearly
    # instead of leaving a half-updated tree.
    if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
      warn "you have local changes; stashing them (git stash list to recover)"
      git -C "$ROOT" stash push -q -m "update.sh $(date -Iseconds)" || true
    fi
    git -C "$ROOT" pull --ff-only
    AFTER="$(git -C "$ROOT" rev-parse --short HEAD)"
    if [[ "$BEFORE" == "$AFTER" ]]; then
      say "already up to date ($AFTER)"
    else
      say "$BEFORE -> $AFTER"
      git -C "$ROOT" log --oneline "$BEFORE..$AFTER" | sed 's/^/    /' | head -20
    fi
  fi
fi

if [[ -n "$STASHED_MODEL" ]]; then
  cp "$STASHED_MODEL" "$ROOT/models/spt.slm"
  rm -rf "$(dirname "$STASHED_MODEL")"
fi

# ------------------------------------------------- 2. rebuild and reinstall
# install.sh already handles a stale cache, the static llama link, the models and
# the desktop launcher, so this is a thin wrapper over it rather than a copy.
ARGS=(--prefix "$PREFIX" --jobs "$JOBS")
[[ ${#EXTRA[@]} -gt 0 ]] && ARGS+=("${EXTRA[@]}")
# --with-olmo is only forwarded when asked for, and install.sh skips the download
# when any GGUF is already present, so an update never re-downloads 4.5 GB.
[[ $WITH_OLMO == 1 ]] && ARGS+=(--with-olmo)
say "rebuilding and reinstalling"
"$ROOT/install.sh" "${ARGS[@]}"

# ------------------------------------------------------------------ 3. report
NEW_VERSION="$("$PREFIX/bin/slm" --version 2>/dev/null || echo unknown)"
echo
say "updated"
printf '    %s\n    %s\n' "before: $OLD_VERSION" "after:  $NEW_VERSION"
# Report the ready-made model by asking the binary what it can see, so this
# cannot disagree with what the program actually finds at run time.
OLMO_SEEN="$("$PREFIX/bin/slm" --gguf-path 2>/dev/null || true)"
if [[ -n "$OLMO_SEEN" ]]; then
  printf '    OLMo:   %s (kept)\n' "$OLMO_SEEN"
else
  printf '    OLMo:   not installed (%s)\n' "slm fetch-model olmo"
fi
echo
printf '  start it: \033[1;32m%s\033[0m\n' "slm"
