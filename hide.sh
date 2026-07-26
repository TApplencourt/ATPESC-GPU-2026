#!/usr/bin/env bash
# hide.sh --- sync the private Overleaf repo into the public GitHub mirror while
# keeping the hands-on SOLUTIONS out of public history.
#
# The solutions are the three *_sol.cpp files. They are copied into the mirror's
# working tree (so you can build/run them locally) but never staged, because the
# mirror's .gitignore lists `*_sol.cpp`. Public history stays at a SINGLE commit
# that has never contained a solution. Run this whenever you edit slides/code.
#
# Reveal them live during the talk with reveal.sh (git add -f).
#
# Usage:  ./hide.sh
set -euo pipefail

SRC="/home/applenco/project/p26.07/6a5683c47d60b2796290f5e6"
MIRROR="/home/applenco/project/p26.07/ATPESC-GPU-2026"

# --- sanity checks -----------------------------------------------------------
[ -d "$SRC/.git" ]    || { echo "ERROR: source repo not found at $SRC"; exit 1; }
[ -d "$MIRROR/.git" ] || { echo "ERROR: mirror repo not found at $MIRROR"; exit 1; }

# --- 1. mirror the working tree (exclude git metadata and old demo code) ------
# --delete so files removed in SRC also vanish from the mirror. We never copy
# .git/ (each repo keeps its own history) or demo/ (retired code). out.pdf is the
# built slides: it lives ONLY in the mirror (too big for Overleaf), so exclude it
# from the sync --- otherwise --delete would wipe it every run.
echo ">> rsync $SRC -> $MIRROR"
rsync -a --delete \
      --exclude='.git/' \
      --exclude='demo/' \
      --exclude='out.pdf' \
      --exclude='hide.sh' \
      "$SRC/" "$MIRROR/"

# --- 2. make sure the mirror ignores every solution file ---------------------
# `*_sol.cpp` catches peak_sol.cpp, exp1_sol.cpp, exp2_sol.cpp (and any future
# _sol solution). Append only if not already present.
if ! grep -qxF '*_sol.cpp' "$MIRROR/.gitignore" 2>/dev/null; then
  echo ">> adding '*_sol.cpp' to mirror .gitignore"
  printf '\n# hands-on solutions --- revealed live via reveal.sh\n*_sol.cpp\n' >> "$MIRROR/.gitignore"
fi

# --- 3. stage everything (solutions are excluded by .gitignore) --------------
cd "$MIRROR"
git add -A

# --- 4. anti-leak guard: refuse to push if any *_sol.cpp got staged ----------
if git diff --cached --name-only | grep -q '_sol\.cpp$'; then
  echo "ABORT: a *_sol.cpp file is staged --- it would leak. Not pushing." >&2
  git diff --cached --name-only | grep '_sol\.cpp$' >&2
  exit 1
fi

# --- 5. amend the single mirror commit and force-push ------------------------
# Keeping one commit means the pre-reveal public history never grows a solution.
git commit --amend --no-edit --allow-empty
echo ">> force-pushing mirror to GitHub"
git push --force origin main

echo
echo "DONE. Public mirror updated; solutions withheld."
echo "Tracked *_sol files (should be EMPTY): $(git ls-files | grep _sol || echo none)"
