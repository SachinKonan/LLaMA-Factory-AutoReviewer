#!/bin/bash
# Submit all 3 keepalive smoke tests on gpu-test.
#
# Usage:
#   bash scripts/run_keepalive_smoke_tests.sh
#
# Each test runs in its own scratch directory (printed below) so they
# don't interfere with each other.

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer

STAMP=$(date +%Y%m%d_%H%M%S)_$$
ROOT=/scratch/gpfs/ZHUANGL/sk7524/keepalive_smoke/$STAMP
mkdir -p "$ROOT"

DIR1="$ROOT/test1_success"
DIR2="$ROOT/test2_fail_loop"
DIR3="$ROOT/test3_fail_release"
mkdir -p "$DIR1" "$DIR2" "$DIR3"

echo "Smoke test root: $ROOT"
echo

# Test 1: success path. Should COMPLETE in <2 min, never enter keepalive.
echo "=== Submitting TEST 1 (success) ==="
JID1=$(sbatch --parsable \
    --export=ALL,MODE=success,MODEL_DIR=$DIR1 \
    sbatch/test_keepalive.sbatch)
echo "  job ID: $JID1   model_dir: $DIR1"

# Test 2: fail enters keepalive. No self-release; will run until walltime
# or manual scancel. Verifies that an exception triggers keepalive entry.
echo "=== Submitting TEST 2 (fail enters loop, walltime kill) ==="
JID2=$(sbatch --parsable \
    --export=ALL,MODE=fail,MODEL_DIR=$DIR2,SELF_RELEASE_AFTER=0 \
    sbatch/test_keepalive.sbatch)
echo "  job ID: $JID2   model_dir: $DIR2"

# Test 3: fail enters keepalive, self-release after 90s. Should COMPLETE
# in ~3-5 min (vLLM startup ~60-90s + 90s keepalive runtime + cleanup).
echo "=== Submitting TEST 3 (fail enters loop, self-release after 90s) ==="
JID3=$(sbatch --parsable \
    --export=ALL,MODE=fail-release,MODEL_DIR=$DIR3,SELF_RELEASE_AFTER=90 \
    sbatch/test_keepalive.sbatch)
echo "  job ID: $JID3   model_dir: $DIR3"

cat <<EOF

================================================================
Submitted: $JID1 (success), $JID2 (fail-loop), $JID3 (fail-release)

Expected outcomes:
  TEST 1 ($JID1): COMPLETED, exit 0, runs <2 min, no [KEEPALIVE] in log
                  expect file: $DIR1/.test.success

  TEST 2 ($JID2): walltime TIMEOUT after 30 min (we don't release it)
                  log will show [TEST] caught -> [KEEPALIVE] active
                  expect files: $DIR2/.test.crashed, $DIR2/.error.txt
                  TO ABORT EARLY: scancel $JID2

  TEST 3 ($JID3): COMPLETED, exit 0, runs ~3-5 min total
                  log shows [TEST] caught -> [KEEPALIVE] -> release after 90s
                  expect files: $DIR3/.test.crashed, $DIR3/.test.released

Monitor:
  squeue -j $JID1,$JID2,$JID3 -o "%.10i %.15j %.8T %.10M"
  watch tail -n5 logs/keepalive_smoke_${JID1}.out logs/keepalive_smoke_${JID2}.out logs/keepalive_smoke_${JID3}.out

When everything is done, verify:
  bash scripts/check_keepalive_smoke.sh $JID1 $JID2 $JID3 $ROOT
================================================================
EOF
