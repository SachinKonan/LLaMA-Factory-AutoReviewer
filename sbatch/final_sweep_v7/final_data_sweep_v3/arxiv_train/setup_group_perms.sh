#!/bin/bash
# Setup group (zhuangl) permissions on the LLaMA-Factory-AutoReviewer repo so a
# collaborator (group member) can train under their account, while sk7524's
# files remain group-readable / -writable in the relevant locations.
#
# Permissions semantics:
#   - Repo-wide: `g+rX` -- read + traverse-on-dirs only. Capital X grants
#     execute ONLY on dirs and on files that already have execute set
#     (e.g. .venv/bin/python). Source/data files (.py, .yaml, .json, .md)
#     stay non-executable.
#   - saves/, logs/, results/, data/: `g+rwX` so the collaborator's training
#     can write outputs.
#   - sbatch/: `g+rx` (lowercase, scoped) so .sbatch and .sh scripts are
#     group-executable.
#   - All directories: setgid (`g+s`) so files created by the collaborator
#     inherit the `zhuangl` group.
#
# Idempotent. Run once on the login node from the project root, or from
# anywhere -- $ROOT is hardcoded.

set -e
ROOT=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer

if [ ! -d "$ROOT" ]; then
    echo "ERROR: $ROOT does not exist"
    exit 1
fi

# IMPORTANT: parent dir /scratch/gpfs/ZHUANGL/sk7524 also needs group=zhuangl
# (default is the user's personal group, e.g. `cs`) so collaborators can
# `cd` through it. We don't change perms there (--x for group is already
# enough), just the group ownership of that single dir.
PARENT=/scratch/gpfs/ZHUANGL/sk7524
echo "[0/4] Re-group parent $PARENT to zhuangl (so group can traverse) ..."
chgrp zhuangl "$PARENT" 2>/dev/null || echo "    (skip: not owner of $PARENT)"

echo "[1/4] Repo-wide read + traverse (g+rX) on $ROOT ..."
chmod -R g+rX "$ROOT"

echo "[2/4] Group-write on saves/logs/results/data ..."
for d in saves logs results data; do
    if [ -d "$ROOT/$d" ]; then
        chmod -R g+rwX "$ROOT/$d"
        echo "    set g+rwX on $d/"
    fi
done

echo "[3/4] Explicit g+x on sbatch/ tree (so .sbatch and .sh are runnable by group) ..."
chmod -R g+rx "$ROOT/sbatch"

echo "[4/4] Setgid (g+s) on every directory so new files inherit zhuangl group ..."
find "$ROOT" -type d -exec chmod g+s {} +

echo ""
echo "Done. Sample perms:"
ls -ld "$ROOT" "$ROOT/saves" "$ROOT/data" "$ROOT/results" "$ROOT/logs" "$ROOT/sbatch"
