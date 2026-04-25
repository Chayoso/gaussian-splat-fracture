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

## 7. Smoke Test Particle Budget

Use 50k as the default smoke-test particle budget.

Reason:

- faster iteration
- lower memory pressure
- enough surface nodes for crack topology debugging
- matplotlib output remains readable

Use 150k only for pre-render validation after behavior is already correct at 50k.

Recommended schedule:

```text
dev smoke:        50k particles, matplotlib only
behavior check:  80k-100k particles, matplotlib only
pre-render:      150k max, selected materials only
beauty render:   only after debug metrics pass
```

For surface-only mode, the more important budget is surface node count, not volume particle count. The target should be enough graph resolution to represent crack paths without making graph operations slow.

## 8. First Experiments

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

## 9. Immediate Code Tasks

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

## 10. Final Positioning

The system should be described as:

```text
sentence-conditioned surface-manifold fracture for Gaussian splats
```

not as:

```text
full volumetric physical fracture simulation
```

This is the better technical and research framing for the current project.
