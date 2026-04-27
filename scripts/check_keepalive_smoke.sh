#!/bin/bash
# Verify the 3 keepalive smoke tests passed.
#
# Usage:
#   bash scripts/check_keepalive_smoke.sh <jid1> <jid2> <jid3> <root_dir>

set -u
JID1=$1
JID2=$2
JID3=$3
ROOT=$4

PASS=0
FAIL=0

check() {
    local name=$1 condition=$2 detail=$3
    if [ "$condition" = "1" ]; then
        echo "  ✓ $name"
        PASS=$((PASS+1))
    else
        echo "  ✗ $name -- $detail"
        FAIL=$((FAIL+1))
    fi
}

state_of() {
    sacct -j "$1" --format=State -X --parsable2 -n 2>/dev/null | head -1
}
exit_of() {
    sacct -j "$1" --format=ExitCode -X --parsable2 -n 2>/dev/null | head -1
}

echo "================================================================"
echo "TEST 1: success path (job $JID1)"
echo "----------------------------------------------------------------"
S1=$(state_of $JID1); E1=$(exit_of $JID1)
echo "  state=$S1 exit=$E1"
[ "$S1" = "COMPLETED" ] && c=1 || c=0
check "TEST 1 state==COMPLETED" $c "got $S1"
[ "$E1" = "0:0" ] && c=1 || c=0
check "TEST 1 exit==0:0" $c "got $E1"
[ -f "$ROOT/test1_success/.test.success" ] && c=1 || c=0
check "TEST 1 .test.success file present" $c ""
[ -f "$ROOT/test1_success/.test.crashed" ] && c=0 || c=1
check "TEST 1 .test.crashed NOT present" $c "(should not have entered keepalive)"
if [ -f "logs/keepalive_smoke_${JID1}.out" ]; then
    grep -q "\[KEEPALIVE\]" "logs/keepalive_smoke_${JID1}.out" && c=0 || c=1
    check "TEST 1 log has no [KEEPALIVE] markers" $c ""
fi

echo
echo "================================================================"
echo "TEST 2: fail enters keepalive, walltime kill (job $JID2)"
echo "----------------------------------------------------------------"
S2=$(state_of $JID2); E2=$(exit_of $JID2)
echo "  state=$S2 exit=$E2"
[ -f "$ROOT/test2_fail_loop/.test.crashed" ] && c=1 || c=0
check "TEST 2 .test.crashed present" $c ""
[ -f "$ROOT/test2_fail_loop/.error.txt" ] && c=1 || c=0
check "TEST 2 .error.txt present" $c ""
if [ -f "logs/keepalive_smoke_${JID2}.out" ]; then
    grep -q "\[KEEPALIVE\] GPU KEEPALIVE active" "logs/keepalive_smoke_${JID2}.out" && c=1 || c=0
    check "TEST 2 log shows keepalive entered" $c ""
fi
case "$S2" in
    TIMEOUT|RUNNING|CANCELLED*) c=1 ;;
    *) c=0 ;;
esac
check "TEST 2 state ∈ {RUNNING,TIMEOUT,CANCELLED} (kept node alive past failure)" $c "got $S2"
[ -f "$ROOT/test2_fail_loop/.test.released" ] && c=0 || c=1
check "TEST 2 .test.released absent (no release file written)" $c ""

echo
echo "================================================================"
echo "TEST 3: fail enters keepalive, self-release (job $JID3)"
echo "----------------------------------------------------------------"
S3=$(state_of $JID3); E3=$(exit_of $JID3)
echo "  state=$S3 exit=$E3"
[ "$S3" = "COMPLETED" ] && c=1 || c=0
check "TEST 3 state==COMPLETED" $c "got $S3"
[ "$E3" = "0:0" ] && c=1 || c=0
check "TEST 3 exit==0:0" $c "got $E3"
[ -f "$ROOT/test3_fail_release/.test.crashed" ] && c=1 || c=0
check "TEST 3 .test.crashed present" $c ""
[ -f "$ROOT/test3_fail_release/.test.released" ] && c=1 || c=0
check "TEST 3 .test.released present" $c ""
if [ -f "logs/keepalive_smoke_${JID3}.out" ]; then
    grep -q "\[KEEPALIVE\] GPU KEEPALIVE active" "logs/keepalive_smoke_${JID3}.out" && c=1 || c=0
    check "TEST 3 log shows keepalive entered" $c ""
    grep -q "release file detected, exiting cleanly" "logs/keepalive_smoke_${JID3}.out" && c=1 || c=0
    check "TEST 3 log shows clean release exit" $c ""
fi

echo
echo "================================================================"
echo "Summary: $PASS passed, $FAIL failed"
echo "================================================================"
[ $FAIL -eq 0 ]
