# Scatter Decision Log — `complete_pulverization`

This file captures the current state of the brittle-pulverization
animation pipeline, the open question about how to push lateral
scatter further, and the reproducibility checklist needed to compare
runs across machines.

---

## Current state (commit `c8ea061`)

### What works

- **Base remnant elimination** for `complete_pulverization`: AT2 halt
  + force-promote partitions any cohesive base into spatial-grid
  chunks, each graduating with a Griffith + unified release impulse.
  Final `fragment_id == 0` count is zero (all particles are part of
  some fragment).  See `_maybe_force_promote_base_remnant` in
  `src/core/simulator_mixins/fragment_physics.py`.

- **Per-particle floor bounce**: MPM's "slip" BC at the ground plane
  is bypassed for fragment-labeled particles by saving pre-p2g2p
  `v_z`, then writing `v_z = -fragment_floor_restitution * v_z_pre`
  for fragments inside the floor band after p2g2p.  Without this,
  the rigid-contact-impulse path saw `vn = 0` after the slip BC and
  silently returned, so glass shards stuck on contact instead of
  pinging off.

- **NET COM lateral scatter**: per-fragment unified rigid-body release
  at graduation moment, decomposed into XY-radial direction (full
  `v_com_gain` magnitude) + small upward boost
  (`upward_fraction × v_com_gain`).  Centered drops put the impact
  center directly below the body so the raw 3D radial vector was
  nearly pure +Z; the XY decomposition was the key fix that turned
  "주저앉으면서 부서지는" collapses into actual lateral spread.

- **Stochastic-jolt suppression**: the random-±-sign Griffith impulse
  has its COM net subtracted before applying to `v_mpm`, so per-
  particle dispersion (paper claim 3) is preserved while the NET
  COM kick is exactly zero by construction.  Without this,
  finite-N variance in the random signs leaked ~`max_speed/sqrt(N)`
  m/s into the fragment's COM, and CUDA non-determinism in the
  per-fragment seed key made one chunk occasionally fly wild ("튕김").

- **Phantom-Gaussian fid=-1**: densified Gaussian slots without a
  surface-particle mapping get `fid = -1` in Houdini export instead
  of `fid = 0`, so the viewer's B-key (hide base) doesn't conflate
  them with genuine label-0 base particles.

### Visible result (latest 10K × 120f, prompt = "soda-lime glass
object completely pulverized into hundreds of tiny shards"):

| metric | value |
| --- | ---: |
| total particles | 10,072 |
| final base (`fid == 0`) | 0 |
| unique fragment labels | 36 |
| max upward dz / frame | 0.023 |
| max lateral (XY) dxy / frame | 0.131 |
| **XY bbox spread (peak)** | **1.75x** |
| sim wall-clock | 754s |

PASSES `scripts/verify_smoke.py` (all four criteria: base==0, bounce,
lateral scatter, bbox spread > 1.3x).

### What still feels off

User feedback: "scatter 솔직히 아직도 크지는 않아."  Although the
verifier passes and the motion is clearly lateral (not collapsing in
place), the visual energy of the explosion still reads as moderate
rather than dramatic glass-shatter.  Open question is which direction
to take the fix.

---

## Three options for stronger scatter

### Option A — Explosion shockwave at impact (external force)

Add a new mechanism: at the moment the body contacts the ground,
apply an outward radial impulse to ALL particles within some radius
of `impact_center`, scaled with `impact_speed`.

**Pros**
- Physically motivated: impact-induced explosion is a real
  phenomenon for brittle bodies (the elastic strain energy at
  impact propagates outward as a shock).
- Affects every particle uniformly so no fragment is "lucky" or
  "unlucky".

**Cons**
- New mechanism → new supporting-contribution paragraph in the
  paper.  More for reviewers to scrutinize.
- Risk of double-counting with the existing unified rigid-body
  impact response (which already gives the whole body a horizontal
  v_com slide + tumble).

### Option B — Bump per-fragment `v_com_gain` (fragment motion)

Just turn up the existing knobs: `v_com_gain` 8 → 16-24, `max_speed`
18 → 30-40, scale up `tumble_gain` correspondingly.

**Pros**
- Smallest change.  Knob values only, no new mechanism.
- Paper framework unchanged.

**Cons**
- Fragments can hit `clip_bound` walls and stop visibly mid-flight.
- Inherited downward MPM velocity at graduation moment (~15-25 m/s
  from free-fall + post-impact damping path) competes with the
  lateral kick: the resultant velocity vector is still
  mostly-downward + slightly-lateral even at v_com_gain = 24.
- May saturate visual benefit before the knob value is large enough
  to be physically reasonable.

### Option C — Cancel inherited downward velocity at graduation, then apply unified release (recommended)

When a fragment graduates, scale down its inherited `v_z` (e.g.
multiply by 0.4-0.5) before adding the unified release impulse.  The
chunk effectively "starts fresh" at graduation moment, so the
lateral kick dominates the resultant velocity vector and the visual
reads as outward explosion rather than diagonal-downward drift.

**Physical motivation**
- Brittle fracture releases stored elastic energy in a NEW direction
  (along the principal tensile eigenvector / opening direction).
  The pre-fracture COM velocity (gravitational fall) is part of
  what the elastic strain energy is being released *from*: as a
  fragment graduates, the elastic energy goes from COM-translation +
  internal strain to fragment-COM-translation + per-particle
  dispersion, with the NEW translation direction set by the fracture
  mechanics, not by the pre-fracture velocity.

**Pros**
- Paper framework unchanged: this is a refinement of how
  `_apply_unified_fragment_release` interacts with the inherited
  velocity, not a new mechanism.
- Preserves Griffith claim 3 cleanly (per-particle dispersion
  untouched).
- Maximizes the visual effect of v_com_gain values that are already
  in a physically-reasonable range, so we don't need to overshoot
  knob values.

**Cons**
- Cancelling inherited v_z is locally non-conservative
  (energy disappears).  Reviewer may ask "where does the kinetic
  energy go?"  Answer: into the per-particle Griffith dispersion
  (which can carry significant per-particle KE without producing
  net COM motion).  Need to be precise about this in the paper.

**Recommended:** Option C with v_com_gain in the 8-12 range.

---

## Run-to-run variation

Even with `--seed 1234` and `torch.use_deterministic_algorithms(True,
warn_only=True)`, the same code on the same machine produces visually
different sims across runs.  Three observed sources:

1. **MPM p2g uses GPU atomic adds.**  Floating-point addition is
   commutative but not associative; the order in which thousands of
   particles deposit mass + velocity into a grid node is non-
   deterministic, so node values differ at the bit level between
   runs.

2. **Fragment ID-keyed RNG seeds.**  Several places use
   `_next_physical_fragment_id` as the seed for deterministic random
   choices (force-promote leftover direction fallback, spin axis
   perturbation, force-promote chunk seeding).  If the order in
   which fragments graduate changes by even one tick due to (1), the
   IDs shift and seeds change, producing different scatter patterns.

3. **SVD sign indeterminacy.**  Shape-match's SVD can flip the
   rotation sign for near-degenerate input, so tumble axis can
   reverse between runs.

Empirically, three runs of `2cdb9ac` on the same machine produced:

| run | bbox spread (peak) | lateral dxy / frame | visible artefacts |
| --- | ---: | ---: | --- |
| 1 | 1.61x | 0.077 | clean |
| 2 | 1.49x | 0.022 | one fragment "tweng" |
| 3 | 1.35x | 0.024 | clean |

The variation is intrinsic to running this codebase on any CUDA
device with deterministic_algorithms in `warn_only` mode.  It will
NOT vanish by re-running on a different machine.

### Reproducibility checklist (for cross-machine comparison)

To eliminate code/config differences (so the only remaining
variation is the CUDA-atomic kind described above):

- [ ] Code commit: `c8ea061` (this repo's
  `surface-manifold-fracture-wip-20260425` branch).
- [ ] Conda env: `crack_py11` (Python 3.11 + torch + CUDA + open3d +
  trimesh).  Other envs in the same machine (`crack`,
  `gsplat_crack`, `omniphysgs`, `omniphysgs2`, etc.) are unrelated.
- [ ] Base YAML: `configs/gravity_drop_manifold.yaml` (default of
  `scripts/inspect_gravity_crack_progression.py`).
- [ ] CLI invocation:
  ```sh
  python -u scripts/inspect_gravity_crack_progression.py \
      --prompt "soda-lime glass object completely pulverized into hundreds of tiny shards" \
      --gravity-particles 10000 \
      --gravity-frames 120 \
      --skip-png \
      --snapshot-stride 2 \
      --out output/<RUN_TAG>
  ```
- [ ] Effective config the script writes: every run dir contains
  `gravity_validation_config.yaml` which is the merged output of the
  base YAML + CLI overrides + (CLIP-derived material params later
  applied via runtime overrides).  For an EXACT replay use that
  file's contents as the source of truth, not the base YAML alone —
  the CLI args (`--gravity-particles`, `--drop-center-z`,
  `--gravity-z`, etc.) modify it in-place.
- [ ] CLIP material prediction: deterministic given the same prompt
  + same `clip ViT-B/32` weights + same material DB
  (`data/material_db.json`).  Re-deriving the material prior on a
  different machine should produce the same `family`, `physics`, and
  `runtime` overrides.

After all of the above are pinned, residual run-to-run variation is
purely CUDA atomic-add non-determinism, and is bounded by the spread
in the table above.  For paper figures, the standard mitigation is to
report the median (or median + IQR) over N=5-10 runs at the same
seed, OR to fully disable non-deterministic ops via
`torch.use_deterministic_algorithms(True, warn_only=False)` (which
will throw on a few CUDA kernels and force a CPU fallback for those —
slower, but bit-exact).

---

## Decision pending

Awaiting user confirmation: proceed with **Option C** (cancel
inherited `v_z` at graduation, then apply unified release)?
