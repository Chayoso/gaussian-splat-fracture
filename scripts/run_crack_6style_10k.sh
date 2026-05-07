#!/bin/bash
# Re-run the 6-style crack progression at 10K particles.
# Matches the original crack_smoke_* parameters (gravity_validation_config.yaml):
#   particles=10000, frames=100, grids=32, substeps=3,
#   drop_center_z=0.8, gravity_z=-3500, fragment_every=2.
set -euo pipefail

cd /home/chayo/Desktop/gaussian_phase_field

CONDA_ENV=diffmpm_v2.3.0
RUN_PY="conda run -n ${CONDA_ENV} --no-capture-output python -u scripts/inspect_gravity_crack_progression.py"

declare -A STYLES=(
    [smooth]="soda-lime glass cleanly split in half by a single fracture line"
    [diffuse]="soda-lime glass with diffuse microcracks across the surface"
    [radial]="soda-lime glass shattering into many sharp radial shards"
    [chunky]="soda-lime glass crumbling into chunky irregular pieces"
    [pulverize]="soda-lime glass completely pulverized into hundreds of tiny shards"
    [ultra]="soda-lime glass exploding into thousands of fine glittering shards"
)

ORDER=(smooth diffuse radial chunky pulverize ultra)

run_one() {
    local style="$1"
    local prompt="$2"
    local outdir="output/crack_smoke_${style}"
    echo "[6style] $(date '+%H:%M:%S')  start: ${style}  ->  ${outdir}"
    ${RUN_PY} \
        --gravity-particles 10000 \
        --gravity-frames 100 \
        --gravity-grids 32 \
        --physics-substeps 3 \
        --drop-center-z 0.8 \
        --gravity-z -3500 \
        --fragment-every 2 \
        --skip-png \
        --prompt "${prompt}" \
        --out "${outdir}" \
        > "/tmp/crack6_${style}.log" 2>&1
    echo "[6style] $(date '+%H:%M:%S')  done:  ${style}"
}

for style in "${ORDER[@]}"; do
    run_one "${style}" "${STYLES[$style]}"
done

echo "[6style] $(date '+%H:%M:%S')  ALL DONE"
