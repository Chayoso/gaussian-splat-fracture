# Problem A — Pipeline Implementation Spec (LOCKED 2026-05-09)

This is the **implementation contract** for the pipeline rewrite.
Sister doc: `algorithm_diagnosis_2026-05-09.md` (the diagnosis that motivated this spec).

Branches:
- `backup-pre-pipeline-rewrite-20260509-1841` — pre-rewrite snapshot
- `pipeline-rewrite-from-spec` — active working branch

## 1. Role separation

| Role | System | Frequency | Output |
|---|---|---|---|
| Decide where breaks occur | Phase-field (continuously evolves) | every fracture tick | `c_visual`, `c_topology` |
| Decide what actually separated | Voronoi (bond eval/commit) | every fracture tick | `_physical_fragment_labels` |
| Move separated bodies | MPM | every physics substep | `x_mpm`, `v_mpm` |
| Auxiliary stabilization (NOT mechanics driver) | Shape match | base body strong, fragment demoted | x/v micro-correction |

Single source of truth for fragment identity: **`_physical_fragment_labels`**.
`fragment_manager.fragment_ids` is read-only display adapter.

## 2. Three-way `c` split

```python
c_visual   = phase_field.c                   # render/crack viz, [0, 1]
c_topology = phase_field.c                   # voronoi bond evidence, [0, 1]
c_mech     = degrade_for_mech(c_visual,      # MPM stress
                              fragment_state,
                              birth_registry)
```

Policy:
```python
g_min = manifold.mechanical_stiffness_floor   # 0.0 = disabled, > 0 active

# Unbroken / process zone
if g_min > 0.0:
    c_cap  = 1.0 - math.sqrt(g_min)           # e.g. g_min=0.05 -> c_cap=0.776
    c_mech = c_visual.clamp(max=c_cap)
else:
    c_mech = c_visual                         # current behavior preserved

# Inside detached fragments (Step 1b active)
if manifold.c_mech_anneal_after_break:
    age_seconds = mpm.time - birth_time[label]
    decay       = math.exp(-age_seconds / tau_seconds)
    c_mech[idx_in_fragment] = c_visual[idx] * decay   # decay->0 -> stiffness restored
```

`c_mech` is recomputed **only on fracture ticks**; intermediate substeps reuse `_last_c_mech` cache.

## 3. Per-frame pipeline (9 steps, projection LAST)

```
for each frame:
  for each physics substep:

    [fracture tick subset — every fracture_tick_interval_substeps]
      1. drive (current MPM state)
      2. phase-field evolve (impact-window burst multiplier optional)
      3. c_topology + opening + tip -> bond_score
      4. energy-budgeted irreversible bond break commit
      5. _physical_fragment_labels update + birth registry update
      6. c_mech generation (cap + detached anneal) -> _last_c_mech cache

    [every substep]
      7. MPM p2g2p with c_mech (cached or fresh)
         (NOTE: p2g2p internal grid floor BC unchanged in this rollout)
      8. shape match / handoff
           - base body: strong (current behavior preserved)
           - detached fragment: position pull off + velocity blend scale -> 0
      9. single simulator-level post projection
           - simulator floor clamp (sub-dx penetration cleanup)
           - simulator-level contact response
           - velocity cap (CFL-safe v_limit)
           - speed cap (per-particle |v|)

  render/export:
       fragment_id source = _physical_fragment_labels
       fragment_manager   = read-only display/stat adapter
```

Core principles:
- `c_visual != c_mech`: visual claim preserved, mechanical singularity prevented
- Voronoi is sole fragment authority; all mechanics consume it
- Shape match injects no force into fragments (after Step 4b)
- Floor / contact / cap consolidated into one simulator-level projection (grid BC stays inside p2g2p)

## 4. State / registry

```python
class ManifoldSimulator:
    # Existing
    self._physical_fragment_labels: torch.LongTensor

    # New (Step 2)
    self._physical_fragment_birth_time:        dict[int, float] = {}  # mpm.time
    self._physical_fragment_birth_physics_step: dict[int, int]  = {}  # global substep counter
    self._physics_step: int = 0

    # New (Step 1a onward)
    self._last_c_mech: torch.Tensor | None = None
```

```python
def _has_physical_fragments(self) -> bool:
    labels = self._physical_fragment_labels
    return labels is not None and bool((labels > 0).any())

def _update_physical_fragment_birth_registry(self):
    if self._physical_fragment_labels is None:
        return
    cur_t = float(self.mpm.time)
    cur_s = int(self._physics_step)
    for lab in torch.unique(self._physical_fragment_labels).tolist():
        if lab <= 0:
            continue
        if lab not in self._physical_fragment_birth_physics_step:
            self._physical_fragment_birth_time[lab] = cur_t
            self._physical_fragment_birth_physics_step[lab] = cur_s
```

## 5. Implementation order

```
1a  mechanical_stiffness_floor               [+ _last_c_mech cache scaffold]
 ↓
2   _physical_fragment_labels authority      [+ birth registry, helper, gate redirect]
 ↓
4a  fragment position pull off               [rest-pose drag off]
 ↓
3a  Voronoi bond eval/commit per fracture tick  [phase-field timing preserved]
     -- requires splitting current _step_voronoi_fracture() into:
        - _voronoi_bond_eval_commit()
        - _voronoi_spatial_split_and_refine()
 ↓
3b  phase-field evolve per fracture tick     [fully synced]
 ↓
4b  fragment velocity blend scale 1.0 -> 0.0 [rigid_v re-injection removed]
 ↓
1b  detached internal c_mech damage -> 0 anneal [stiffness restore]
```

## 6. Per-step contracts

### Step 1a — mechanical_stiffness_floor + cache scaffold

| | |
|---|---|
| **What** | (a) When `g_min > 0`: `c_mech = c_visual.clamp(max=c_cap)`, `c_cap = 1 - sqrt(g_min)`. When `g_min = 0`: no-op pass-through. (b) Introduce `_last_c_mech` cache (always written, read by future steps). |
| **Files** | `src/core/manifold_simulator.py` (around stress computation in `_step_physics`) |
| **Flag** | `manifold.mechanical_stiffness_floor: 0.0` |
| **Dep** | none |
| **Validation** | 5-style 10K smoke. peak CFL change ≤5%. Default (0.0) behavior identical to baseline. With g_min=0.02 / 0.05, smoke must remain stable. |
| **Risk** | very low |

### Step 2 — Authority migration + birth registry

| | |
|---|---|
| **What** | (a) `_has_physical_fragments()` helper. (b) Physics gate switches from `n_fragments > 1` to `_has_physical_fragments()`. (c) Shape match / render / export label source unified to `_physical_fragment_labels`. (d) `fragment_manager.fragment_ids` becomes read-only display adapter. (e) Birth registry (`_birth_time`, `_birth_physics_step`) updated per fracture tick via `_update_physical_fragment_birth_registry()`. (f) `_physics_step` global counter introduced. |
| **Files** | `manifold_simulator.py`, `simulator_mixins/fragment_physics.py`, `simulator_mixins/render_fragments.py`, `simulator_mixins/surface_binding.py` |
| **Flag** | `manifold.use_physical_fragment_authority: false` |
| **Dep** | none |
| **Validation** | per-frame label flip-flop count -> 0 or single-digit. Visual: floating-fragment / re-attach artifact eliminated. |
| **Risk** | medium (many consumers) |

### Step 4a — Fragment position pull off

| | |
|---|---|
| **What** | In `_shape_match_component`, when fragment label > 0, skip the position pull (`x_mpm[idx] = old.lerp(target, strength)`). Velocity injection retained. |
| **Files** | `simulator_mixins/fragment_physics.py` |
| **Flag** | `manifold.shape_match_fragment_position_pull: true` (set false to skip pull) |
| **Dep** | Step 2 |
| **Validation** | Rest-pose drag motion eliminated. Fragments no longer pulled toward ancient drop pose. |
| **Risk** | low |

### Step 3a — Voronoi bond eval/commit per fracture tick

| | |
|---|---|
| **What** | (a) Split current `_step_voronoi_fracture()` into `_voronoi_bond_eval_commit()` (substep-tick subset) and `_voronoi_spatial_split_and_refine()` (frame-end). (b) Move bond_score/break commit/Mode-I kick into substep tick. (c) Spatial split / render label refinement stays at frame end. (d) Phase-field evolve still at frame end. (e) Bond score computed via swappable function. (f) Energy budget enforced (joule, no dt). |
| **Files** | `fracture/voronoi_pipeline.py`, `manifold_simulator.py` |
| **Flags** | `manifold.voronoi_eval_per_fracture_tick: false`<br>`manifold.fracture_tick_interval_substeps: 1`<br>`manifold.voronoi_bond_score_mode: "mean"`<br>`manifold.fracture_energy_budget_fraction: 0.0`<br>`manifold.fracture_release_window_ticks: 30` |
| **Dep** | Step 2, 4a |
| **Validation** | Inter-frame label jump flattens. Crack propagation temporally smooth. Spatial split compute isolated to frame end. |
| **Risk** | medium (1-frame-stale phase-field damage; cost throttled by `fracture_tick_interval_substeps`) |

### Step 3b — Phase-field evolve per fracture tick

| | |
|---|---|
| **What** | Move `_step_fracture()` from frame end into the fracture tick. Shares `fracture_tick_interval_substeps` with Voronoi. |
| **Files** | `manifold_simulator.py` (`step_rendering` reorganization) |
| **Flag** | `manifold.phase_field_evolve_per_fracture_tick: false` |
| **Dep** | Step 3a validated |
| **Validation** | No more stale damage. Visual stability matches or exceeds 3a. If indistinguishable, 3a alone is sufficient. |
| **Risk** | medium (per-tick cost, interaction with `front_substeps`) |

### Step 4b — Velocity blend scale ramp + handoff smoothing

| | |
|---|---|
| **What** | (a) In `_shape_match_component` fragment branch: `v[idx] = (1 - s·blend)·v + s·blend·rigid_v + 0.15·s·velocity_blend·correction_v`, `s = manifold.shape_match_fragment_velocity_blend_scale`. (b) On detach (lookup birth_physics_step), apply v_com smoothing only for `N_smoothing_substeps`:<br>`age_substeps = _physics_step - birth_physics_step[lab]`<br>`if age_substeps < N: apply v_com smoothing` |
| **Files** | `simulator_mixins/fragment_physics.py` |
| **Flags** | `manifold.shape_match_fragment_velocity_blend_scale: 1.0`<br>`manifold.fragment_handoff_smoothing_substeps: 0` |
| **Dep** | Step 2 (birth registry), 4a |
| **Validation** | Post-detach omega magnitude (frame +5 onward) reduced ≥5x vs baseline. In-place spin permanently fixed. |
| **Risk** | medium (too-fast ramp -> post-detach explosion) |

### Step 1b — c_mech anneal inside detached

| | |
|---|---|
| **What** | Inside detached fragments, `c_mech = c_visual * exp(-age_seconds / tau)` where `age_seconds = mpm.time - birth_time[label]`. Visual crack preserved via `c_visual`. |
| **Files** | `manifold_simulator.py` (`_compute_c_mech`) |
| **Flags** | `manifold.c_mech_anneal_after_break: false`<br>`manifold.c_mech_anneal_tau_seconds: 0.05` |
| **Dep** | Step 2 (birth registry), 3a/3b (fracture tick timing) |
| **Validation** | Internal stress max in broken fragments unchanged vs Step 4b (regression check). Visual: fragments behave as sharp shards, no internal jitter. |
| **Risk** | low |

## 7. Bond score policy (Step 3a)

```python
def compute_bond_score(damage, bond_idx, opening_traction=None, mode="mean"):
    if mode == "mean":
        return float(damage[bond_idx].mean())             # current baseline
    elif mode == "weighted_q90":
        d = damage[bond_idx]
        q90 = float(np.quantile(d, 0.90))
        opening = float(opening_traction.norm()) if opening_traction is not None else 0.0
        return q90 + 0.3 * opening + hysteresis_term      # noise-resistant
    elif mode == "max":
        return float(damage[bond_idx].max())              # debug only
    else:
        raise ValueError(f"unknown mode: {mode}")
```

## 8. Energy budget (Step 3a)

```python
# At impact (once)
total_budget = manifold.fracture_energy_budget_fraction * impact_KE   # joule
release_window = manifold.fracture_release_window_ticks
per_tick_cap = total_budget / release_window                          # joule per tick

# Every fracture tick
candidates = bonds_above_score_threshold sorted by score desc
released = 0.0
for bond in candidates:
    cost = Gc * bond.estimated_area                                   # joule (no dt)
    if released + cost > per_tick_cap:
        break
    bond.commit_break()
    released += cost
total_budget -= released                                              # cumulative
# total_budget <= 0 -> break stop
```

## 9. `_step_physics` pseudocode (3a/3b active)

```python
def _step_physics(self, dt):
    # 1. drive
    is_fracture_tick = (self._physics_step
                        % manifold.fracture_tick_interval_substeps) == 0

    if is_fracture_tick:
        # 2. phase-field evolve (3b)
        if manifold.phase_field_evolve_per_fracture_tick:
            self._step_fracture()
        # 3-5. voronoi bond eval/commit + label/birth update (3a)
        if manifold.voronoi_eval_per_fracture_tick:
            self._voronoi_bond_eval_commit()
            self._update_physical_fragment_birth_registry()
        # 6. c_mech recompute + cache
        self._last_c_mech = self._compute_c_mech(...)

    # 7. MPM
    c_mech = self._last_c_mech if self._last_c_mech is not None \
             else self._get_volumetric_damage()
    stress = self.elasticity(self.F, c=c_mech)
    self.x_mpm, self.v_mpm, self.C, self.F = self.mpm.p2g2p(
        self.x_mpm, self.v_mpm, self.C, self.F, stress)

    # 8. shape match (base / fragment-handoff)
    self._apply_shape_matching(dt)

    # 9. single simulator-level post projection
    self._post_projection(dt)

    self._physics_step += 1


# step_rendering frame end:
def step_rendering(self):
    for _ in range(self.substeps):
        self._step_physics(dt)
    # frame-end-only:
    if not manifold.phase_field_evolve_per_fracture_tick:
        self._step_fracture()                       # legacy frame-end path
    if not manifold.voronoi_eval_per_fracture_tick:
        self._voronoi_bond_eval_commit()            # legacy frame-end path
        self._update_physical_fragment_birth_registry()
    # always frame-end:
    self._voronoi_spatial_split_and_refine()
    # ... fragment detection, gaussian update ...
```

## 10. Validation matrix

| Step | Metric | How measured | Pass |
|---|---|---|---|
| 1a | peak CFL on 5-style smoke | log `CFL~` max | ≤ baseline |
| 2 | per-frame label flip count | `_physical_fragment_labels` diff | 0 or monotonic decrease |
| 4a | rest-pose drag motion | fragment trajectory replay | absent |
| 3a | inter-frame label jump | per-frame max position delta | flattened |
| 3b | staleness artifact | 3a vs 3b visual diff | improvement or no change |
| 4b | post-detach omega @ frame +5 | `fragment_omega` tracking | 5x reduction |
| 1b | broken fragment internal stress | per-fragment stress max | unchanged (regression) |

## 11. Feature flag namespace (`manifold.*`)

```yaml
# Step 1a
manifold.mechanical_stiffness_floor: 0.0          # 0 = disabled

# Step 2
manifold.use_physical_fragment_authority: false   # implies birth registry update

# Step 4a
manifold.shape_match_fragment_position_pull: true # default = current

# Step 3a
manifold.voronoi_eval_per_fracture_tick: false
manifold.fracture_tick_interval_substeps: 1
manifold.voronoi_bond_score_mode: "mean"
manifold.fracture_energy_budget_fraction: 0.0
manifold.fracture_release_window_ticks: 30

# Step 3b
manifold.phase_field_evolve_per_fracture_tick: false   # shares interval

# Step 4b
manifold.shape_match_fragment_velocity_blend_scale: 1.0
manifold.fragment_handoff_smoothing_substeps: 0

# Step 1b
manifold.c_mech_anneal_after_break: false
manifold.c_mech_anneal_tau_seconds: 0.05
```

All flag defaults preserve current behavior. Each step is gated; toggle to roll out, untoggle to rollback.

---

## Implementation note for Step 3a

Current `voronoi_pipeline.py` does NOT yet split bond eval/commit from spatial split / refine.
Step 3a must first refactor `_step_voronoi_fracture()` into:

```python
def _voronoi_bond_eval_commit(self):
    # damage -> bond breakage update -> Mode-I kicks -> connected components
    # (cheap, runs per fracture tick)

def _voronoi_spatial_split_and_refine(self):
    # cKDTree spatial split + render label refinement
    # (expensive, runs once at frame end)
```

Without this split, moving the whole `_step_voronoi_fracture()` into the substep loop will blow up compute cost.

## Step 1a kickoff checklist

1. Default `g_min = 0.0` -> verify simulator runs with current behavior unchanged (5-style 10K smoke).
2. `g_min = 0.02` -> verify smoke still stable, peak CFL not worse.
3. `g_min = 0.05` -> verify smoke still stable, peak CFL not worse.
4. Visual sanity check on diffuse / chunky / radial / pulverize / ultra.
5. Commit Step 1a if validations pass.
