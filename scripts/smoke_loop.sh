#!/usr/bin/env bash
# Iterative 2K smoke test for complete_pulverization.
# Runs sim, verifies, exits when verifier passes.

set -e

PY="/c/Users/ok429/anaconda3/envs/crack_py11/python.exe"
PROMPT="soda-lime glass object completely pulverized into hundreds of tiny shards"

ITER=${1:-1}
TS=$(date +%Y%m%d_%H%M%S)
OUT="output/smoke_2k_iter${ITER}_${TS}"
mkdir -p "$OUT"
echo "$OUT" > /tmp/last_smoke_out.txt

echo "=== smoke iter $ITER -> $OUT ==="
"$PY" -u scripts/inspect_gravity_crack_progression.py \
    --prompt "$PROMPT" \
    --gravity-particles 2000 \
    --gravity-frames 80 \
    --physics-substeps 3 \
    --gravity-grids 32 \
    --drop-center-z 0.25 \
    --gravity-z -5000 \
    --skip-png \
    --snapshot-stride 2 \
    --out "$OUT" >"$OUT/run.log" 2>&1
tail -10 "$OUT/run.log"
echo
echo "--- force-promote messages ---"
grep -i "force-promote\|force_promote" "$OUT/run.log" | head -5
echo "exit=$?"

echo
echo "=== verifier ==="
"$PY" scripts/verify_smoke.py "$OUT/houdini_export"
VERIFY_EXIT=$?
echo "verify_exit=$VERIFY_EXIT"
exit $VERIFY_EXIT
