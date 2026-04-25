# Surface-Manifold Gaussian Crack Implementation

## 1. Direction

The project should move toward a surface-first fracture simulator:

```text
text prompt
-> material / fracture-family prior
-> surface Gaussian graph
-> crack-front propagation on the graph
-> edge cut and connected components
-> matplotlib validation
-> Gaussian / 3DGS rendering
```

This is not a strict volumetric AT2 phase-field solver. It is a phase-field-inspired, sentence-conditioned fracture model whose state lives on the Gaussian surface manifold.

The current approach is preferred over a traditional volumetric method because the final representation is Gaussian/surface-centric, and the desired output is visually convincing material-specific fracture rather than exact bulk fracture mechanics.

## 2. Current Confirmed Base

The existing `GaussianFractureField` already stores fracture state on surface Gaussian nodes:

- `c`: surface damage
- `H`: irreversible history / accumulated drive
- `n`: crack normal
- `a`: crack opening
- `f`: fragment label

The current `phase_field` config values are passed into `ManifoldSimulator`, then used by `GaussianFractureField`. The field is initialized with `N_surf = surface_mask.sum()`, not with a volumetric grid.

So the current implementation is already close to the desired core:

```text
surface graph + tip-based propagation + phase-field-like damage band
```

The next step is to stop treating volume MPM as mandatory for fracture and make it an optional driver.

## 3. Target Architecture

### Required Core

- `MaterialPriorAdapter`
  - converts text/CLIP retrieval into fracture family and behavior parameters.

- `GaussianGraph`
  - stores surface adjacency and edge weights.

- `GaussianFractureField`
  - evolves crack tips, damage, history, normals, and opening on the surface graph.

- `GraphFragmentManager`
  - turns damaged/cut graph edges into components and fragment labels.

- `SurfaceCrackDriver` (new)
  - produces surface drive without requiring volume particles.

- `smoke_test.py`
  - validates crack propagation, cut edges, fragments, and opening using matplotlib first.

### Optional Backend

- MPM remains useful for impact/deformation experiments, but it should not be required for the default fracture loop.

```text
default: text -> surface driver -> surface fracture -> matplotlib
optional: MPM -> projected drive -> surface fracture -> render
```

## 4. Surface-Only Fracture Rule

Fracture should be decided on the surface graph:

```text
node damage c_i >= threshold -> cracked node
edge cut score d_ij >= threshold -> broken edge
connected components after broken edges -> fragments
```

Important: a visible crack does not always imply a fragment.

Correct interpretation:

| Surface crack topology | Result |
|---|---|
| short open crack | crack only |
| boundary-to-boundary crack | split candidate |
| closed loop crack | detachable island |
| long high-confidence open crack in brittle material | local support-loss flake / chip |
| broad damaged region | chunk / shard candidate |

This is good. It allows ceramic to remain crack-only while glass/concrete can split when topology supports it.
For the surface-only backend, stiffness loss does not automatically collapse geometry because there is no volumetric mass solve. Ring-free collapse is therefore modeled explicitly as a material-gated support-loss release near a strong crack corridor.

## 5. Material-Family Targets

| Family | Target |
|---|---|
| `sharp_brittle` | narrow crack, strong cross-edge cut, clean split |
| `brittle_moderate` | visible crack/opening, split optional or delayed |
| `rough_quasi_brittle` | rough network, chunk release, stable fragments |
| `diffuse_damage` | no explicit crack path, no fragment detach |

The text prompt should control style through family parameters:

- `tau_init`
- `growth_gain`
- `band_width`
- `open_gain`
- `branching_bias`
- `anisotropy_strength`
- `cut_vote_strength`
- `fragment_threshold`
- `detach_tendency`

## 6. Implementation Plan

### Phase 1. Consolidate Surface Driver

Add a surface-only driver that produces:

```python
init_score: Tensor[N]
growth_drive: Tensor[N]
growth_dir: Tensor[N, 3]
opening_hint: Tensor[N]
```

Initial driver can be procedural:

- impact center on surface
- radial propagation
- normal/tangent-aware direction
- material-family noise / branching
- optional user prompt style modifiers

This lets smoke tests run without full volume MPM.

### Phase 2. Keep `GaussianFractureField`

Do not rewrite the current crack model yet. Reuse it as the canonical surface phase-field-like model.

Change only the source of `init_score`, `growth_drive`, and `growth_dir`.

### Phase 3. Fragment by Surface Graph

Use `GraphFragmentManager` as the topology layer:

- cut core
- side-aware cross-edge cuts
- authoritative cut memory
- closure candidate detection
- connected components

Fragment creation should come from graph separation, not volume particles.

### Phase 4. Matplotlib Validation First

Before 3DGS rendering, every material run should save:

- damage scatter
- crack-front scatter
- cut-edge overlay
- fragment label scatter
- opening map
- closure candidate overlay

This is the primary development loop.

### Phase 5. Rendering Only After Debug Pass

Only run Gaussian/3DGS rendering after matplotlib confirms:

- crack path is plausible
- cut edges exist where expected
- fragment labels are stable
- rubber remains no-split

## 7. Validation Particle Budget

Use tiered particle budgets. Do not use one particle count for every decision.

Recommended schedule:

```text
L0 prior check:       no simulation
L1 smoke:            1k-2k surface nodes, no render
L2 default validate: 10k surface nodes, matplotlib / CSV
L3 final validate:   50k surface nodes, selected prompts only
L4 pre-render:       50k-150k, selected successful cases only
```

Interpretation:

- 1k-2k is only a directional smoke test. Small shards may be under-counted.
- 10k is the default material/sentence behavior validation budget.
- 50k is the first serious topology and fragment-count check.
- 150k is not an iteration budget. Use it only after 50k passes.

For surface-first mode, the important budget is surface node count, not volume particle count. If the run uses `surface_ratio: 1.0`, then particle count is effectively surface graph resolution.

## 8. Validation Protocol

Validation should answer four separate questions:

```text
1. Did CLIP / text select the intended material family?
2. Did sentence wording change crack morphology?
3. Did gravity impact produce the expected damage and fragment release?
4. Are the released fragments renderable and visually stable?
```

Do not move to rendering until questions 1-3 pass by CSV/MD metrics.

### L0. Material Prior Check

Purpose:

- verify CLIP top-k retrieval
- verify material family assignment
- verify sentence style classification
- catch obvious prompt/material mismatch before running simulation

Metrics:

- `family`
- `sentence_style`
- `top1`
- `top1_weight`
- `family_scores`
- `E_raw`, `Gc_raw`, `nu_raw`, `density_raw`
- `tau_init`, `growth_gain`, `band_width`, `open_gain`, `edge_break_rate`, `branching_bias`

Pass criteria:

- glass / ice / crystal -> `sharp_brittle`
- porcelain / ceramic -> `brittle_moderate`
- concrete / stone / mortar -> `rough_quasi_brittle`
- rubber / elastomer -> `diffuse_damage`
- radial / shatter wording -> `radial_shatter`
- one long / clean split wording -> `single_smooth`
- scratch / shallow / diffuse wording -> `diffuse_microcrack`
- crumble / granular / chunks wording -> `chunky_crumble`

### L1. Surface Morphology Smoke

Purpose:

- validate crack-front behavior without gravity or rendering
- compare sentence styles cheaply
- catch broken graph/front/fragment code before MPM

Budget:

```text
particles:       2k-10k
frames:          24-40
fragment_every:  2-4
render:          off
```

Metrics:

- `c_max`, `c_mean`
- `cracked_count`, `weak_count`
- `visited_count`, `tip_count`
- `branchiness`
- `cracked_span_*`, `visited_span_*`
- `max_cut_edges`
- `max_closure_candidates`
- `max_n_frags`

Expected behavior:

| Prompt type | Expected surface result |
|---|---|
| glass radial shatter | many visited nodes, many tips, high cut edges |
| glass single smooth | low tip count, narrow crack path, optional 1-4 fragments |
| glass diffuse scratch | low damage, no fragment release |
| concrete crumble | broad damage, many tips, rough crack network |
| rubber no fracture | near-zero explicit crack, no fragments |

### L2. Gravity Impact Validation

Purpose:

- validate material + gravity + sentence style coupling
- check whether damage turns into physical fragment release
- avoid rendering while tuning fracture behavior

Budget:

```text
smoke:    1k-2k particles, 44-56 frames, grids 32-40
default:  10k particles, 64-80 frames, grids 64
final:    50k particles, selected prompts, grids tuned by memory
```

Metrics:

- `impact_frame`
- `max_n_fragments`, `final_n_fragments`
- `max_n_cracked`, `final_n_cracked`
- `final_visited_count`, `final_tip_count`, `final_branchiness`
- `max_c_max`, `final_c_mean`
- `max_cut_edges`, `max_broken_edges`
- `max_release_candidates`
- `max_open_release_patches`, `max_open_release_nodes`
- `max_physical_fragment_drop`
- `max_physical_detached_distance`
- `elapsed_sec`

Pass criteria by prompt:

| Prompt | Required outcome |
|---|---|
| glass bottle shattering into many sharp radial cracks | high crack coverage, high tip count, clear fragment release |
| glass bottle with one long smooth crack | narrow split, low tip count, 1-4 large fragments or crack-only at low resolution |
| glass bottle with diffuse tiny surface scratches | low damage, `final_n_fragments == 1`, no open release |
| concrete block crumbling into rough granular chunks | broad damage and rough tips; chunk release should appear at 2k+ and improve at 10k/50k |
| rubber ball deforming without visible fracture | `final_n_fragments == 1`, no explicit crack, no open release |

Known resolution caveat:

- 1k smoke can under-count fragments because shard sizes fall below graph/physical fragment minimums.
- A 1k result is allowed to fail fragment-count targets if crack morphology separates correctly.
- 2k and above should start showing brittle/radial release.
- 50k is the first budget where fragment count should be treated seriously.

### L3. Render Readiness Gate

A case is render-ready only if:

- `impact_frame` is non-null for gravity tests
- `c_max` and crack coverage match the prompt
- fragment count matches the target material/style
- `max_physical_fragment_drop` or `max_physical_detached_distance` is non-zero for released-fragment prompts
- diffuse/rubber controls remain no-split
- runtime is acceptable at the intended particle budget

### L4. Gaussian Render Validation

Rendering should verify only visual quality, not core fracture semantics.

Check:

- all expected Gaussians are visible
- detached fragments are visible after release
- interior/cut surfaces are visible when exposed
- interior normals face the camera correctly enough for splat rendering
- rubber/diffuse prompts do not show false fracture
- no severe opacity holes, flicker, or fragment overlap artifacts

## 9. Canonical Prompt Suites

Use three prompt files per validation run:

```text
material_prompts.txt
sentence_prompts.txt
gravity_prompts.txt
```

Material prompts should test material retrieval:

```text
thin soda-lime glass bottle shattering into sharp clean cracks
porcelain ceramic mug cracking into a few brittle splits
rough concrete block crumbling into irregular chunks
vulcanized rubber ball deforming without visible fracture
clear ice sphere cracking with radial brittle fractures
hardwood oak block splitting along the grain
structural steel block denting without brittle fracture
```

Sentence prompts should hold material mostly fixed and vary crack style:

```text
glass bottle with one long smooth crack
glass bottle with spiderweb branching cracks
glass bottle shattering into many sharp radial cracks
glass bottle with diffuse tiny surface scratches
concrete block splitting with one clean fracture line
concrete block crumbling into rough granular chunks
concrete block with wide branching cracks
concrete block with shallow diffuse microcracks
rubber ball deforming without visible fracture
```

Gravity prompts should be a smaller, high-signal subset:

```text
glass bottle shattering into many sharp radial cracks
glass bottle with one long smooth crack
glass bottle with diffuse tiny surface scratches
concrete block crumbling into rough granular chunks
rubber ball deforming without visible fracture
```

## 10. Recommended Commands

Prior-only check:

```text
python scripts/validate_sentence_materials.py --no-sim --prompts-file output/material_sentence_validation_20260425/sentence_prompts.txt --out output/prior_check
```

Full material/sentence validation at default scale:

```text
conda run -n crack_py11 --no-capture-output python scripts/validate_material_sentence_gravity.py --out output/material_sentence_validation_YYYYMMDD --surface-particles 10000 --surface-frames 24 --gravity-particles 10000 --gravity-frames 64 --gravity-grids 64 --physics-substeps 3
```

Fast gravity smoke:

```text
conda run -n crack_py11 --no-capture-output python scripts/validate_material_sentence_gravity.py --skip-surface --gravity-prompts-file output/material_sentence_validation_20260425/gravity_prompts.txt --out output/gravity_smoke_2k --gravity-particles 2000 --gravity-frames 56 --gravity-grids 40 --physics-substeps 2 --drop-center-z 0.31 --gravity-z -4200
```

Final 50k selected validation:

```text
conda run -n crack_py11 --no-capture-output python scripts/validate_material_sentence_gravity.py --skip-surface --gravity-prompts-file output/material_sentence_validation_20260425/gravity_prompts.txt --out output/gravity_final_50k --gravity-particles 50000 --gravity-frames 80 --gravity-grids 80 --physics-substeps 2 --drop-center-z 0.31 --gravity-z -4200
```

## 11. Output Review Checklist

For every run, inspect:

```text
gravity_material_validation.md
gravity_material_validation.csv
surface_material_validation.csv
surface_sentence_validation.csv
manifest.json
```

Do not accept a run based on a single metric. Required checks:

1. Material family matches prompt.
2. Sentence style matches wording.
3. Crack morphology differs across sentence variants.
4. Fragment release differs across material/style variants.
5. Rubber/diffuse controls remain stable.
6. Runtime is recorded and acceptable.
7. The output directory name records budget and purpose.

## 12. First Experiments

Run these first:

```text
glass / 50k / matplotlib
concrete / 50k / matplotlib
ceramic / 50k / matplotlib
rubber / 50k / matplotlib
```

Success criteria:

- glass: narrow crack and cut-edge split candidate
- concrete: rough crack network and stable components
- ceramic: visible crack/opening, split optional
- rubber: no explicit crack path, no fragments

Then repeat selected successful cases at 100k or 150k.

## 13. Immediate Code Tasks

1. `src/fracture/surface_crack_driver.py` provides procedural, material-family-conditioned surface drive.
2. `smoke_test.py --surface-only` runs fracture directly on surface graph nodes and skips MPM.
3. In surface-only mode, the particle budget is treated as surface node budget, not surface+volume budget.
4. `GaussianGraph` uses compact KD-tree kNN for large surfaces, avoiding dense all-pairs distance matrices at 50k+ nodes.
5. Fragment connected-components detection runs at a configurable cadence so crack propagation can be checked cheaply between topology passes.
6. Ring-free fragment creation is allowed only for brittle/rough families through local crack-corridor support loss. `neutral_reference` remains crack-only by default.
7. Matplotlib damage/opening/cut/closure overlays are the validation gate before Gaussian rendering.
8. Keep MPM path working as optional regression path.

Useful commands:

```text
python smoke_test.py --surface-only --frames 80 --particles 50000 --out output/surface_50k
python smoke_test.py --surface-only --family rough_quasi_brittle --frames 30 --particles 50000 --plot-every 10 --fragment-every 2 --out output/surface_fragment_rough_50k
python smoke_test.py --surface-only --frames 80 --particles 100000 --plot-every 10 --out output/surface_100k
python smoke_test.py --surface-only --frames 80 --particles 150000 --plot-every 10 --fragment-every 4 --out output/surface_150k
```

## 14. Final Positioning

The system should be described as:

```text
sentence-conditioned surface-manifold fracture for Gaussian splats
```

not as:

```text
full volumetric physical fracture simulation
```

This is the better technical and research framing for the current project.
