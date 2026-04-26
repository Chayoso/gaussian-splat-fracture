# Next Experiments

Date: 2026-04-26

## Current retained outputs

Only the useful comparison outputs are kept under `output/`:

- `secondary_shatter_gravity_50k_probe`: latest gravity-drop validation with staged secondary shatter.
- `secondary_shatter_surface_50k_probe`: latest 50K surface sentence probe.
- `material_sentence_validation_50k_catastrophic_v1`: pre-secondary-shatter 50K baseline.

Older smoke/probe/archive outputs were removed to keep the workspace manageable.

## Current read

The surface-first pipeline is now showing the target control signal:

- radial glass: high fragment count and broad surface damage
- smooth crack: low fragment count and narrow crack path
- ceramic: limited fracture
- concrete/chunky: medium fragmentation
- rubber/diffuse: no fracture release

The latest 50K gravity probe moved glass from the previous tens-of-fragments range to roughly 160 fragments while keeping rubber at 1 fragment.

## Next implementation work

1. Stabilize secondary shatter labels.
   - Remove tiny one-node artifacts from sector/band splitting.
   - Add a minimum visible fragment area rule separate from internal labels.
   - Keep radial glass high-fragment, but make fragment size distribution less uniform.

2. Improve fragment motion after release.
   - Increase visual separation for fully shattered glass.
   - Add per-fragment angular scatter and slight spin-like displacement.
   - Keep ceramic/concrete motion less explosive than glass.

3. Connect fracture surfaces to rendering.
   - Verify internal cut-surface normals.
   - Ensure newly exposed internal faces/shell particles render when fragments separate.
   - Add a one-PNG render smoke after matplotlib validation, not during every sweep.

4. Add a prompt sweep preset file.
   - Small smoke: 2K, 3 prompts.
   - Surface validation: 50K, sentence-style prompts.
   - Gravity validation: 50K, material prompts.
   - Render smoke: one selected final frame per mode.

## Next validation work

1. Regression table:
   - Compare baseline catastrophic vs secondary shatter.
   - Track `max_n_fragments`, `max_secondary_shatter_nodes`, `max_physical_fragment_drop`, and `cracked_count`.

2. Visual morphology check:
   - Inspect final matplotlib PNGs for radial, smooth, chunky, diffuse.
   - Flag if radial becomes too grid/sector-looking.

3. Gravity robustness:
   - Run 3 seeds for glass/ceramic/concrete/rubber at 10K first.
   - If ordering is stable, run 50K once.

4. Render readiness:
   - Use matplotlib as the gate.
   - Only run Gaussian rendering for the best final frame after metrics pass.

## Practical next command targets

```powershell
# 10K seed robustness before expensive 50K runs
C:\Users\ok429\anaconda3\envs\crack_py11\python.exe scripts\validate_material_sentence_gravity.py `
  --surface-particles 10000 --gravity-particles 10000 `
  --gravity-frames 56 --physics-substeps 3 `
  --out output\seed_robustness_10k_v1

# Targeted 50K surface morphology probe
C:\Users\ok429\anaconda3\envs\crack_py11\python.exe scripts\validate_sentence_materials.py `
  --particles 50000 --frames 24 --fragment-every 4 `
  --out output\surface_morphology_50k_next `
  --prompts "glass bottle shattering into many sharp radial cracks" `
            "glass bottle with one long smooth crack" `
            "concrete block crumbling into rough granular chunks" `
            "vulcanized rubber ball deforming without visible fracture"
```
