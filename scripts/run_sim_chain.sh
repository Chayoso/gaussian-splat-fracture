#!/bin/bash
# Sequential sim chain:
#  1. wait for v13 (100K radial) to finish
#  2. rubber 100K
#  3. radial 150K (sqrt(15)x scaling)
#  4. rubber 150K
# Locks each output dir read-only when complete.
set -euo pipefail

cd /home/chayo/Desktop/gaussian_phase_field

CONDA_ENV=diffmpm_v2.3.0
RUN_PY="conda run -n ${CONDA_ENV} --no-capture-output python scripts/inspect_gravity_crack_progression.py"

wait_for_done() {
    local outdir="$1"
    until [ -f "${outdir}/progression_sweep_report.md" ]; do
        sleep 30
    done
    echo "[chain] $(date '+%H:%M:%S')  done: ${outdir}"
    chmod -R a-w "${outdir}" 2>/dev/null || true
}

run_sim() {
    local label="$1"
    local n="$2"
    local prompt="$3"
    local override="$4"
    local outdir="$5"
    echo "[chain] $(date '+%H:%M:%S')  start: ${label} (${n} particles) -> ${outdir}"
    if [ -n "${override}" ]; then
        ${RUN_PY} \
            --gravity-particles "${n}" \
            --gravity-frames 120 \
            --fragment-every 1 \
            --snapshot-stride 1 \
            --skip-png \
            --override-json "${override}" \
            --prompt "${prompt}" \
            --out "${outdir}" \
            > "/tmp/sim_chain_${label}.log" 2>&1
    else
        ${RUN_PY} \
            --gravity-particles "${n}" \
            --gravity-frames 120 \
            --fragment-every 1 \
            --snapshot-stride 1 \
            --skip-png \
            --prompt "${prompt}" \
            --out "${outdir}" \
            > "/tmp/sim_chain_${label}.log" 2>&1
    fi
    chmod -R a-w "${outdir}" 2>/dev/null || true
    echo "[chain] $(date '+%H:%M:%S')  done: ${label}"
}

# ----- step 1: wait for v13 (already running) -----
echo "[chain] $(date '+%H:%M:%S')  waiting for v13 (100K radial)..."
wait_for_done output/local_100k_radial_v13

# ----- step 2: rubber 100K -----
RUBBER_PROMPT='rubber ball bouncing on the floor'
RUBBER_100K_OVERRIDE='{"manifold.post_impact_gravity_z": -2000.0, "manifold.post_impact_damping": 0.985}'
run_sim "rubber_100k" 100000 "${RUBBER_PROMPT}" "${RUBBER_100K_OVERRIDE}" \
    "output/local_100k_rubber_v13b"

# ----- step 3: radial 150K (sqrt(15) x scaling, ~3.87x from 10K base) -----
RADIAL_PROMPT='soda-lime glass object shattering into many sharp radial cracks'
RADIAL_150K_OVERRIDE='{"manifold.fragment_persistent_min_size": 155, "manifold.fragment_physical_min_size": 46, "manifold.fragment_render_min_size": 12, "manifold.shape_match_fragment_strength": 0.40}'
run_sim "radial_150k" 150000 "${RADIAL_PROMPT}" "${RADIAL_150K_OVERRIDE}" \
    "output/local_150k_radial_v14"

# ----- step 4: rubber 150K -----
RUBBER_150K_OVERRIDE='{"manifold.post_impact_gravity_z": -2000.0, "manifold.post_impact_damping": 0.985}'
run_sim "rubber_150k" 150000 "${RUBBER_PROMPT}" "${RUBBER_150K_OVERRIDE}" \
    "output/local_150k_rubber_v14b"

echo "[chain] $(date '+%H:%M:%S')  ALL DONE"
