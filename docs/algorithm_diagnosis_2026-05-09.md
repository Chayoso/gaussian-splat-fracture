# Algorithm Diagnosis — 5/6 commit (`92d6cd3`) baseline

Date: 2026-05-09
HEAD: `92d6cd3 Multi-mesh + mesh-to-mesh collision scaffold`

This document captures the **structural problems** in the simulation algorithm
that cannot be solved by parameter tuning (dt reduction, substep increase, damping
schedules, etc.). Each is observed by reading the code path of `_step_physics` /
`step_rendering` / `_step_voronoi_fracture` / `_shape_match_component`.

The numbered fixes (F1–F7) below the diagnosis are listed in priority order
based on minimum-change-for-maximum-impact.

> **Revision history**
> - First pass (2026-05-09 morning): initial diagnosis (8 problems, 7 fixes)
> - Reviewed (2026-05-09 afternoon): wording precision pass; D weakened from
>   primary cause to candidate; A, B, C, F precise mechanism corrections;
>   E, G, H confirmed unchanged.

---

## Problem A — Three fracture systems coupled across one frame (asynchronously)

A single `step_rendering()` call runs three fracture-related systems at
**different points in the frame**, not in the same substep:

- **Phase-field damage stress** — applied **inside every substep**:
  `stress = elasticity(F, c=c_vol)`
  ([manifold_simulator.py:846](../src/core/manifold_simulator.py#L846))
- **Müller shape matching** — also applied **inside every substep**:
  ([fragment_physics.py:619](../src/core/simulator_mixins/fragment_physics.py#L619),
  invoked from [manifold_simulator.py:883](../src/core/manifold_simulator.py#L883))
- **Voronoi bond breakage + Mode-I kicks** — applied **once per frame, after
  the substep loop**:
  ([manifold_simulator.py:769](../src/core/manifold_simulator.py#L769),
  [voronoi_pipeline.py:208–218](../src/fracture/voronoi_pipeline.py#L208-L218))

The three systems impose conflicting forces on the same particles and the
ordering changes the answer (a particle may still be a graph-fragment but
already voronoi-broken; phase-field damage may still be growing inside a
voronoi fragment). The discontinuity is not within a substep; it is between
the per-substep state and the frame-end voronoi update.

Original wording was "same timestep"; the precise statement is **"same frame,
async coupling between substep-level systems and the frame-end voronoi
update"**.

## Problem B — Fragment shape matching re-injects rigid rotation every substep

[fragment_physics.py:502–508](../src/core/simulator_mixins/fragment_physics.py#L502-L508):

```python
cov = rest_centered.T @ current_centered
u, _, vh = svd(cov)
rot = u @ vh
```

This is *not* a closed-form conservation of angular momentum. The function
re-measures `omega` from the current velocity field every call
([fragment_physics.py:553–555](../src/core/simulator_mixins/fragment_physics.py#L553-L555)):

```python
velocity = self.v_mpm[idx]
v_com = velocity.mean(dim=0)
omega = self._component_angular_velocity(current_centered, velocity - v_com)
```

then applies a contact impulse, two-regime angular damping, blend, and
correction-velocity injection. The reason a separated fragment keeps
spinning in place is **not** mathematical preservation but **rigid rotation
re-injection**: every substep extracts the small residual omega from
numerical noise and re-applies it as a rigid rotation through the shape
match target

```python
target = rest_centered @ rot + current_com
target = target + dt * cross(omega, target - current_com)
self.x_mpm[idx] = old.lerp(target, strength)
```

so the spin is regenerated even when the underlying physics would naturally
damp it. Earlier wording overclaimed this as a math-level invariant; the
correct claim is structural (per-substep re-injection) and equally
parameter-independent in effect.

## Problem C — Post-MPM sequential overrides break solver consistency

Within one `_step_physics()` call, `x_mpm` and `v_mpm` are written in a
sequential pipeline. For sharp-brittle materials (glass-class) most of the
soft-body branches are no-ops; the live path is:

1. `mpm.p2g2p`
2. floor clamp (`v_z = 0` below floor)
   ([manifold_simulator.py:861–880](../src/core/manifold_simulator.py#L861-L880))
3. `_apply_shape_matching`
   ([manifold_simulator.py:883](../src/core/manifold_simulator.py#L883))
4. velocity clamp ±`v_limit`
   ([manifold_simulator.py:1007–1019](../src/core/manifold_simulator.py#L1007-L1019))
5. per-particle speed cap

For diffuse-damage / soft-elastic families the pipeline additionally fires
`_apply_soft_elastic_contact_response`, `_apply_soft_elastic_squash`, the
second floor clamp, and the kinematic lock; for brittle these are gated off.
The earlier wording "9 overrides per substep" was inflated; the actual count
is around 5 for the sharp_brittle smoke runs we have been visualising. The
**structural problem** — sequential corrections each with their own
assumption, breaking the post-P2G2P grid-particle consistency — is real
regardless of count. Small numerical noise compounds non-linearly across
these stages and is impossible to attribute to a single offender during
debugging.

## Problem D (weakened) — Stress discontinuity at damage boundaries (candidate, not primary CFL cause)

[manifold_simulator.py:846](../src/core/manifold_simulator.py#L846):

```python
stress = self.elasticity(self.F, c=c_vol)
```

Important caveats that downgrade the earlier claim:

- The constitutive model is **corotated_phase_field**, which applies
  **tension-only degradation**. Compression response is preserved.
  ([physical_constitutive_models.py:557](../src/elasticity/physical_constitutive_models.py#L557))
- A residual stiffness `k=1e-6` floor is applied even at fully damaged
  points, so the "stress = 0 → grid singularity" mechanism never fully
  triggers.
- The local CFL metric used in `_step_physics`
  ([manifold_simulator.py:1021–1025](../src/core/manifold_simulator.py#L1021-L1025))
  is `c_wave * dt / dx` with `c_wave = sqrt(s_max / density_eff)`. It is
  **driven by `s_max`, not by velocity spike**, so degraded particles
  *lower* the local CFL contribution, not raise it.

The earlier write-up claimed that damage → stress=0 → CFL spike was the
"real cause" of the CFL ≈ 1.0 we observed on `ultra` / `wood` / `rubber`.
That causal chain was overstated:

- The CFL increase comes from genuine high stress at impact (compression
  component, undamaged-region wave speed), not from damage zeroing the
  stress.
- A real concern remains: **stress discontinuity** at the boundary between
  highly-damaged and undamaged regions can produce visible artifacts,
  particularly when the velocity field is also being overridden by shape
  matching at the same boundary. But this is an *artifact candidate*, not
  the primary CFL driver.

Net: keep this problem in the list as a smaller item ("stress discontinuity
at damage boundaries") and remove the CFL-as-primary-symptom framing.

## Problem E — Voronoi update is asynchronous with the MPM substep loop

[manifold_simulator.py:769–772](../src/core/manifold_simulator.py#L769-L772):

```python
if self.voronoi is not None and self._gravity_drop_contacted:
    self._step_voronoi_fracture()  # once per frame, after all substeps
```

[voronoi_pipeline.py:180](../src/fracture/voronoi_pipeline.py#L180),
[voronoi_pipeline.py:216](../src/fracture/voronoi_pipeline.py#L216):

Within a single frame, voronoi labels and broken-bond state are frozen
across all `substeps × physics` steps. At frame end voronoi runs once,
discovers new broken bonds, applies Mode-I kicks, updates labels. The next
frame begins with new velocity impulses and possibly different fragment
labels.

This is a strong structural problem: dynamics become frame-rate-dependent
and there is a visible discontinuity at frame boundaries that no amount of
substep tuning hides.

## Problem F — Shape-match rest pose is older than fragment detachment

The earlier diagnosis said the rest pose was frozen "at the moment of
fragment detachment". That is wrong; the rest pose is set even earlier:

- [manifold_simulator.py:682](../src/core/manifold_simulator.py#L682) in
  `enable_gravity_drop()`:

  ```python
  if self.x_mpm is not None:
      self._shape_match_rest_positions = self.x_mpm.detach().clone()
  ```

- [manifold_simulator.py:696](../src/core/manifold_simulator.py#L696) in
  `initialize()`:

  ```python
  self._shape_match_rest_positions = init_positions.clone()
  ```

So the rest pose is the **pre-drop** geometry (or the very initial loaded
state), well before any contact, deformation, or damage has occurred. After
detachment:

- Phase-field damage continues to grow inside the fragment
- Voronoi may break further bonds inside the fragment
- Particles deform under residual stress

But shape matching keeps pulling the deformed particles back toward this
much earlier static pose. The mismatch is larger and grows for longer than
the previous wording implied; the resulting correction force grows with it
and divergence follows.

## Problem G — Fragment IDs are managed by two independent systems

- `fragment_manager.fragment_ids` — graph-based, derived from the surface
  manifold cut field
- `_physical_fragment_labels` — voronoi-based, derived from cell connectivity

They are reconciled by `_map_surface_labels_to_particles` and
`_update_physical_fragment_registry`, but different code branches consume
different sources:

- [fragment_physics.py:290](../src/core/simulator_mixins/fragment_physics.py#L290) reads `_physical_fragment_labels`
- [surface_binding.py:87](../src/core/simulator_mixins/surface_binding.py#L87) consumes the graph-side ids
- [manifold_simulator.py:1649](../src/core/manifold_simulator.py#L1649) branches on `fragment_manager.n_fragments`

When the two systems disagree (graph still treats two cells as connected;
voronoi has already broken the bond between them), the same particle may be
classified as base body in one substep and as a fragment in the next. This
is a strong candidate for the "fragment that floats / detaches and
re-attaches" artifact.

## Problem H — Floor handling is duplicated four times

1. MPM grid slip BC: `grid_v_z = 0`
   ([mpm_model.py:494](../src/mpm_core/mpm_model.py#L494))
2. `_step_physics` floor clamp #1 after p2g2p
   ([manifold_simulator.py:861](../src/core/manifold_simulator.py#L861))
3. `_shape_match_component` floor penetration push
   ([fragment_physics.py:594](../src/core/simulator_mixins/fragment_physics.py#L594))
4. `_step_physics` floor clamp #2 after shape matching

Some material families add a soft contact / squash response on top
([manifold_simulator.py:882–884](../src/core/manifold_simulator.py#L882-L884)).
The same boundary is enforced four (or more) times with slightly different
rules. Contact momentum is silently absorbed at unpredictable points in the
pipeline.

---

## Confidence ranking after review

| # | Problem | Confidence | Notes |
|---|---|---|---|
| E | Voronoi async with substep loop | strong | confirmed |
| G | Fragment ID double management | strong | confirmed; strong floating-fragment candidate |
| H | Floor handling duplicated | strong | confirmed |
| C | Sequential override pipeline | strong | "9 → 5 for brittle"; structural pattern remains |
| F | Static rest pose (older than detachment) | strong | timing corrected; conflict actually larger than initial wording |
| B | Spin re-injection per substep | medium-strong | mechanism corrected (re-injection, not math-conservation) |
| A | Three-system coupling within a frame | medium-strong | wording corrected (frame-async, not same-timestep) |
| D | Stress discontinuity at damage boundaries | weak | demoted from "primary CFL cause" to artifact candidate |

---

## Fix priority

The fixes are not a parameter sweep. Each is a structural change.

| # | Fix | Solves | Refactor scale |
|---|---|---|---|
| **F1** | Use **only one** of phase-field or voronoi (recommended: voronoi-only; intrinsic stiffness preserved, fragment count directly controllable) | A, partially D | Medium — disable one subsystem |
| **F2** | **Damage cap = 0.95** (one-line clamp on `c_vol`) | partial D mitigation | 1 line |
| **F3** | **Disable fragment shape matching**; detached fragments use ballistic (COM + gravity) only, no rest-pose pull | B, F | Medium — fragment branch in `_apply_shape_matching` |
| **F4** | Consolidate post-MPM corrections into a single explicit projection step, fixed ordering | C | Medium–large |
| **F5** | Move voronoi update **inside** the substep loop, not at frame end | E | Medium |
| **F6** | Single fragment-ID source of truth (drop graph-fragment OR voronoi-fragment, keep one) | G | Large |
| **F7** | Single floor handler (use only the grid BC + an explicit projection at the end of `_step_physics`) | H | Medium |

### Recommended starting order (revised)

After the review, the priority of F2 drops slightly (it no longer claims to
fix a CFL primary cause), but the structural fixes E/G/H/F2/F3 line up
nearly the same as before:

1. **F1** — pick voronoi-only as the single fracture system; eliminates the
   A/D coupling concern at the structural level.
2. **F3** — eliminates the rotation re-injection regardless of dt; addresses
   B and the worst symptom of F (fragment dragged to ancient rest pose).
3. **F2** — keep as a small safety cap (defensive); not the headline fix.
4. F5 / F4 / F7 / F6 — incremental cleanup once the algorithmic core is
   stable.

Parameters (dt, substeps, damping, etc.) should not be touched until **F1
+ F3** are in place. Tuning before structural fixes is what made the past
sweeps converge to local-minimum hacks rather than a robust algorithm.
