# Tip-Based Gaussian-Manifold Fracture Pipeline
## Graphics-Oriented Implementation Specification
### MPM Physics + Surface-Aware Gaussian Graph + Crack-Tip Propagation + Delayed Feedback

---

## 1. Document Goal

This document specifies a practical implementation plan for converting the current Gaussian-manifold fracture pipeline from a **damage-diffusion-style model** into a **tip-based crack growth model**.

The goal is not merely to tune parameters, but to change the algorithmic structure so that:

- cracks **initiate** near physically plausible impact/tension hotspots,
- crack tips **advance upward along the body** rather than staying trapped at the bottom,
- fracture propagation is handled as **front advancement on a Gaussian graph**,
- rendering remains **3DGS-native**,
- and MPM feedback is scheduled in a way that does not kill wave transmission too early.

This document explains:
1. why the current pipeline fails,
2. why the proposed pipeline works,
3. what each file should be changed into,
4. and what exact implementation flow is recommended.

---

## 2. Core Diagnosis

### 2.1 What is happening right now
The current behavior is:

- damage does appear near the lower contact region,
- but it does not propagate upward through the torso,
- and by ~80 frames the damaged region still remains concentrated near the bottom.

Observed pattern:
- `c > 0.3` remains near the lower area,
- `z_max` for damaged points stays around a low band,
- upper body / ears remain almost untouched.

So the issue is **not** “damage fails to appear”.

The issue is:

> **there is no effective crack-front propagation mechanism that moves damage upward through the object.**

---

### 2.2 Why the current algorithm fails
The current pipeline fails for four structural reasons.

#### (A) Initiation and propagation are not separated
The current system uses local tensile drive (`psi+`-like signal) both to:
- start crack damage,
- and continue crack growth.

That is not sufficient.

In fracture mechanics terms:
- **initiation** is where a crack first appears,
- **propagation** is where an already existing crack tip moves.

These are different decisions and must use different logic.

---

#### (B) Propagation is still treated as diffusion / neighbor spreading
The current Gaussian fracture update behaves too much like:

```text
local drive + neighbor smoothing -> damage field spreading
```

This tends to produce:
- a damaged bottom patch,
- or a broad stain-like region,
- rather than a narrow, advancing crack band.

A crack is not just a scalar field becoming larger.
A crack is a **front** that moves.

---

#### (C) Damage feeds back into MPM too early
If damage immediately reduces stiffness in the contact region, then:

- the bottom gets softer,
- the object deforms locally,
- wave transmission upward becomes weaker,
- and the crack front never has a chance to travel upward.

So the system self-traps:
- lower part gets damaged,
- lower part softens,
- upper part never receives enough propagation-supporting signal.

---

#### (D) Euclidean kNN is not a good enough growth graph
If the Gaussian graph is built only from Euclidean proximity, crack growth can:
- spread sideways near the bottom,
- wrap around a bottom ring,
- or choose shortcuts through local 3D proximity rather than surface continuity.

To move upward along the visible object surface, propagation needs:
- surface tangential continuity,
- geodesic-like neighborhood bias,
- and directional preference consistent with a growing crack path.

---

## 3. Graphics Expert View: What the Model Should Be

From a graphics + simulation perspective, the right abstraction is:

> **Do not treat fracture as a globally evolving scalar damage field.**
> Treat fracture as a **tip-guided path growth process** with a supporting local damage band.

This means the fracture representation should be a hybrid:

### 3.1 Field component
A local damage field still exists and is useful for:
- crack band thickness,
- appearance modulation,
- opacity drop,
- rough local fracture texture,
- and neighborhood weakening.

### 3.2 Front component
A crack-tip/front state is additionally required for:
- deciding where crack growth goes next,
- determining path continuity,
- selecting only a few valid successor nodes,
- triggering opening / split,
- and producing believable directional fracture.

So the correct representation is not:
- **damage only**

but:
- **damage field + active crack tip set + growth direction + delayed feedback**

---

## 4. Why the Proposed Pipeline Works

The proposed pipeline works because it resolves the exact structural failures above.

### 4.1 Initiation becomes local and physically grounded
The initial crack seeds still come from MPM-side physics:
- tensile energy,
- impact response,
- local stress concentration.

So the model stays physics-informed.

### 4.2 Propagation becomes path-based instead of diffusion-based
Once a crack exists, the next step is no longer “increase damage where drive is positive”.
Instead it becomes:
- select crack tips,
- evaluate candidate neighboring nodes,
- choose a small number of forward successors,
- advance the front.

That is much closer to actual visible crack behavior.

### 4.3 Surface-aware graph prevents bottom-ring spreading
By adding tangent continuity / geodesic-like bias, propagation follows plausible visible surface paths instead of short Euclidean neighborhoods.

### 4.4 Delayed feedback prevents premature bottom collapse
If stiffness degradation is delayed during the early post-impact window, the wave / signal can travel upward before the lower contact patch dies.
That makes upward front propagation much more likely.

### 4.5 Damage field remains useful but becomes subordinate
Damage is still maintained, but it is no longer the sole carrier of fracture logic.
Instead:
- crack tips drive the path,
- damage fills in the band around that path,
- rendering uses both.

That is the right hierarchy.

---

## 5. Recommended New Pipeline

```text
Text / Config / Material Prior
    -> Geometry + Gaussian initialization
    -> MPM physical simulation
    -> Physics projector:
         initiation score
         growth drive
         growth direction cue
    -> Crack seed extraction
    -> Tip-based graph growth
    -> Local damage band update around active tips
    -> Delayed coupling of damage back into MPM
    -> Gaussian opening / split / fragment update
    -> 3DGS rendering
```

---

## 6. High-Level File Map

Assume the relevant current files are:

```text
src/
├── fracture/
│   ├── gaussian_fracture_field.py
│   ├── physics_projector.py
│   ├── gaussian_graph.py          # recommended new / expanded role
│   └── crack_front.py             # recommended new
├── constitutive_models/
│   └── physical_constitutive_models.py
├── core/
│   └── manifold_simulator.py
├── visualization/
│   └── gaussian_updater.py
└── pipeline/
    └── fracture_pipeline.py
```

Below is the recommended role of each file.

---

## 7. File-by-File Specification

# 7.1 `src/fracture/physics_projector.py`

## Current problem
Right now the projector likely passes a local physics signal (e.g. tensile energy) directly into the Gaussian fracture update, often after a neighborhood averaging step.

That creates two problems:
- initiation and propagation cues are mixed together,
- local bottom-drive dominates everything.

## New role
This file should become a **feature projector for crack growth**, not just a raw scalar transfer.

It should project **three different quantities** into Gaussian space:

1. `init_score` — used only for crack initiation,
2. `growth_drive` — used only for tip propagation scoring,
3. `growth_dir` — used only for directional alignment during propagation.

## Recommended output interface
```python
projected = {
    "init_score": init_score_g,     # shape [N]
    "growth_drive": growth_drive_g, # shape [N]
    "growth_dir": growth_dir_g,     # shape [N, 3]
    "raw_energy": raw_energy_g,     # optional debug
}
```

## How to compute these
### `init_score`
Use a sparse hotspot-like measure:
- percentile-thresholded tensile energy,
- top-k pooled from particles to Gaussian,
- optionally masked near impact/contact region.

This is only for **seed formation**.

### `growth_drive`
Do **not** use raw `psi+` directly.
Use a smoother but propagation-specific quantity:
- thresholded or normalized tension support,
- possibly local contrast rather than absolute magnitude,
- so that higher body regions can still receive nonzero support after the initial event.

### `growth_dir`
Project a direction cue such as:
- principal tensile direction,
- dominant local stress direction,
- or another physically derived propagation hint.

Important:
- this is a **growth direction cue**,
- not the crack opening normal.

## Why this change matters
It decouples:
- “where a crack begins”
from
- “where an existing crack should go next”.

That separation is essential.

---

# 7.2 `src/fracture/gaussian_fracture_field.py`

## Current problem
This file currently behaves too much like a damage diffusion field:
- positive drive raises damage,
- neighbors spread damage,
- but there is no explicit crack tip state.

That is why damage remains near the bottom instead of forming an advancing crack band.

## New role
This file should no longer be the sole fracture logic container.
It should maintain the **damage band state**, while the actual **front advancement logic** is driven by an explicit tip set.

So this file becomes:
- a local fracture-band state updater,
- not the master crack path planner.

## State to maintain
Recommended per-Gaussian state:

```python
state = {
    "damage": c,              # [N]
    "history": H,             # [N]
    "active_front": tip_mask, # [N] bool
    "visited": visited_mask,  # [N] bool
    "growth_dir": g,          # [N,3] optional cached
}
```

## New behavior
### damage update should be local to active crack front
Instead of:
- raising damage everywhere with positive drive,

do:
- only increase damage near active front nodes,
- or near newly activated successor nodes,
- and optionally apply a small band-thickening step around them.

### example conceptual update
```text
if node is on active front:
    damage += local_front_gain
elif node is immediate neighbor of front:
    damage += weak_band_fill
else:
    no update
```

## Why this change matters
This ensures the damage field becomes a **consequence of front movement**, not the cause of global propagation.

That is exactly the desired inversion.

---

# 7.3 `src/fracture/crack_front.py`  *(recommended new file)*

## Why this file is needed
The current architecture is missing an explicit crack tip/front abstraction.

This file should own:
- the current tip set,
- successor candidate evaluation,
- next-tip selection,
- and path continuity bookkeeping.

This is the most important architectural addition.

## Core state
```python
front_state = {
    "tip_mask": tip_mask,            # active crack tips
    "parent_index": parent_idx,      # predecessor for each activated node
    "front_age": front_age,          # optional
    "path_confidence": confidence,   # optional
}
```

## Core operation
For each active tip node `i`:
1. get candidate neighbors `j`,
2. compute tip-advance score `S(i->j)`,
3. pick top-1 or top-k successors,
4. activate them as the next front.

## Tip advance score

S(i->j) = alpha * D_j + beta * A_ij + gamma * G_ij + delta * C_ij

where:

- `D_j`: growth drive at candidate node,
- `A_ij`: alignment with growth direction,
- `G_ij`: geodesic / surface continuity bias,
- `C_ij`: crack path continuity / low-curvature preference.

## Important design rule
Only a **small number** of successors should be activated.
Otherwise the front becomes a broad wave and turns back into diffusion.

## Why this file matters
This is the file that finally introduces:
- explicit crack-tip mechanics,
- selective forward growth,
- and narrow advancing bands.

Without it, the pipeline remains a damage field model.

---

# 7.4 `src/fracture/gaussian_graph.py`

## Current problem
Euclidean kNN by itself does not encode visible-surface continuity.
Cracks can spread sideways or wrap around the base.

## New role
This file should define a **surface-aware growth graph**.

It can still start from Euclidean kNN, but the edge weights used for crack growth must be enriched.

## Recommended edge terms

For neighbor edge `(i, j)`, define:

w_advance_ij = w_dist_ij * w_tangent_ij * w_growth_align_ij * w_geo_ij

### terms
- `w_dist`: local distance weight
- `w_tangent`: favors motion along surface tangent continuity
- `w_growth_align`: favors edges aligned with current growth direction
- `w_geo`: optional geodesic-like bias or surface continuity heuristic

## Practical simplified version
Even if full geodesics are unavailable, a strong practical approximation is enough:

- compute local Gaussian surface tangent basis,
- penalize jumps that point off local tangent continuation,
- penalize edges that sharply turn relative to the current crack path.

## Why this change matters
The graph should represent:
- **where a crack is allowed to travel visually**
not merely
- **what is Euclidean-nearby in 3D**.

That difference is huge.

---

# 7.5 `src/core/manifold_simulator.py`

## Current problem
Damage currently feeds back into MPM stiffness too early.

That makes the bottom soften before the upward signal has propagated.

## New role
This file should manage **coupling schedule** between fracture state and MPM material degradation.

## Required modification
Introduce **delayed feedback** for early post-impact frames.

### recommended schedule
```text
frames 0 ~ T0:     no fracture -> MPM stiffness degradation
frames T0 ~ T1:    partial / ramped feedback
frames T1+:        full feedback
```

Example:
- `T0 = 5`
- `T1 = 10`

### coupling scalar
```python
feedback_scale = schedule(frame_idx)
effective_damage_for_mpm = feedback_scale * damage
```

This means:
- rendering can already show damage/front evolution,
- but constitutive weakening is intentionally delayed.

## Why this change matters
It prevents the lower contact patch from dying too early and allows wave / crack-driving information to travel upward before local collapse.

This single change can dramatically alter the resulting fracture path.

---

# 7.6 `src/constitutive_models/physical_constitutive_models.py`

## Current problem
The local tensile quantity is currently useful for initiation, but insufficient by itself for propagation.

## New role
This file should continue to provide **physics-side descriptors**, but should no longer implicitly define the visible crack evolution.

It should output:
- tensile energy / `psi+`,
- optional principal direction,
- optional stress orientation cue,
- optional impact/contact proximity signal.

These are **inputs to fracture reasoning**, not the fracture update itself.

## Recommended interface additions
Make sure the simulator can query:
- local tensile drive,
- local principal direction,
- and any other propagation-relevant physical feature.

## Why this change matters
This keeps the model physics-informed while avoiding the mistake of equating local instantaneous tension with the full crack growth rule.

---

# 7.7 `src/visualization/gaussian_updater.py`

## Current problem
If rendering uses only scalar damage, cracks look like stains or soft damaged regions.

## New role
This file should consume both:
- damage field,
- and front/growth states.

## Inputs
```python
render_state = {
    "damage": c,
    "tip_mask": tip_mask,
    "growth_dir": growth_dir,
    "opening_dir": opening_dir,   # if separated
    "fragment_id": fragment_id,   # optional
}
```

## Suggested behavior
### Use `damage` for:
- opacity attenuation,
- local fracture appearance,
- darkening,
- covariance roughening.

### Use `tip/front` for:
- opening trigger,
- Gaussian splitting,
- path-focused crack visibility,
- narrow advancing fracture highlight.

## Important note
Opening direction and growth direction should be separate concepts:
- `growth_dir` = where the crack goes next,
- `opening_dir` = how the crack separates visually.

Do not conflate them.

## Why this change matters
It lets rendering express:
- a visible advancing crack path,
not just
- an area with high damage value.

---

# 7.8 `src/pipeline/fracture_pipeline.py`

## New role
This file should orchestrate the new order of operations.

## Recommended per-frame order
```text
1. Run MPM substeps
2. Extract physics descriptors
3. Project initiation / growth cues to Gaussian nodes
4. If no crack yet or new crack allowed: update seed tips
5. Advance crack front on Gaussian graph
6. Update local damage band around active front
7. Update delayed-feedback damage for MPM
8. Update Gaussian opening / split / appearance
9. Render frame
```

## Why this order matters
The crack path is now decided before damage band fill and before full constitutive feedback.
That is the correct causal order.

---

## 8. Recommended Data Structures

### 8.1 Gaussian node attributes
```python
N = num_gaussians

damage:        float[N]
history:       float[N]
tip_mask:      bool[N]
visited:       bool[N]
growth_dir:    float[N,3]
opening_dir:   float[N,3]   # optional
fragment_id:   int[N]       # optional
```

### 8.2 Graph storage
```python
neighbors: list[list[int]]
edge_weight_dist: float[E]
edge_weight_tangent: float[E]
edge_weight_growth: float[E]
edge_weight_geo: float[E]
```

### 8.3 Projected physics descriptors
```python
init_score:   float[N]
growth_drive: float[N]
growth_dir:   float[N,3]
raw_energy:   float[N]   # debug only
```

---

## 9. Crack-Tip Propagation Specification

## 9.1 Seed generation
Seed nodes should be created only from strong `init_score`.

Example rule:
```text
seed if init_score > threshold_init
```

Only a small number of seed nodes should be activated.

Optional:
- restrict to impact-near region,
- or restrict to local maxima of `init_score`.

---

## 9.2 Front advancement
For each active tip node `i`, evaluate successors `j` in neighbors.

Compute:
- drive term,
- growth alignment term,
- surface continuity term,
- curvature / continuity term.

Then:
- pick top-1 or top-k,
- activate them as next tips,
- mark old tip as visited/non-tip.

This produces a discrete crack path.

---

## 9.3 Band fill
Once a new tip is activated:
- increase damage at the tip strongly,
- increase damage in nearby nodes weakly,
- optionally thicken the band by one local neighborhood ring.

This gives the crack a visible width without turning the whole process into diffusion.

---

## 10. Delayed Feedback Specification

## 10.1 Principle
Do not immediately use visible damage as full constitutive damage.

Instead define:
```python
mpm_damage = feedback_scale(frame_idx) * visible_damage
```

## 10.2 Recommended schedule
Example:
```python
def feedback_scale(frame_idx):
    if frame_idx < 5:
        return 0.0
    elif frame_idx < 10:
        return (frame_idx - 5) / 5.0
    else:
        return 1.0
```

This is only a baseline and can be tuned later.

## 10.3 Why it works
It lets the simulation separate:
- crack detection and growth logic,
from
- constitutive weakening.

That is essential for upward crack propagation in impact-driven scenarios.

---

## 11. Why This Pipeline Is More Correct for Graphics

From a graphics expert’s standpoint, the new pipeline is better for five reasons.

### 11.1 It matches what the renderer actually needs
A renderer does not need an abstract global damage field.
It needs:
- a visible path,
- local crack width,
- opening cues,
- and fragment separation.

The new model gives exactly that.

### 11.2 It matches visible fracture phenomenology
Many visible cracks behave like:
- local initiation,
- narrow advancing front,
- band thickening behind the tip,
- later opening and fragmentation.

That is much closer to a tip-growth representation than to global diffusion.

### 11.3 It preserves physics where physics is useful
MPM still provides:
- wave response,
- strain/tension cues,
- impact-driven activation.

So the model remains physically grounded.

### 11.4 It avoids over-trusting scalar damage
A scalar field is useful, but by itself it cannot encode:
- path direction,
- tip position,
- successor choice,
- or continuity.

Adding explicit front state solves that.

### 11.5 It is much easier to debug
If crack growth fails, we can now inspect:
- seed selection,
- tip locations,
- successor scores,
- graph bias,
- and feedback schedule.

That is far more interpretable than debugging a single damage diffusion equation.

---

## 12. Recommended Implementation Order

### Phase 1 — minimal structural upgrade
1. add `crack_front.py`,
2. modify `physics_projector.py` to output `init_score`, `growth_drive`, `growth_dir`,
3. modify `gaussian_fracture_field.py` so damage is updated only around active front,
4. add delayed feedback in `manifold_simulator.py`.

### Phase 2 — graph quality upgrade
5. enrich `gaussian_graph.py` with tangent-aware / continuity-aware edge bias,
6. use top-k successor selection instead of broad activation.

### Phase 3 — rendering refinement
7. modify `gaussian_updater.py` to use both damage and tip/front state,
8. separate `growth_dir` and `opening_dir`,
9. connect opening / split / fragment logic.

### Phase 4 — parameter sweeps
10. only after the above, sweep `E`, `Gc`, thresholds, feedback delay, and graph scoring weights.

---

## 13. What Not to Do

Before the above structural changes are made, do **not** rely on:

- only increasing `E`,
- only changing damage threshold,
- only increasing simulation length,
- only changing diffusion strength,
- only tuning scalar damage feedback.

Those are secondary adjustments.
They cannot replace a missing crack-front algorithm.

---

## 14. Final Summary

### Main issue
Damage is generated at the bottom, but there is no mechanism for a crack tip to advance upward.

### Structural fix
Replace damage-diffusion-style propagation with:
- explicit crack tip/front tracking,
- selective graph-based successor activation,
- surface-aware graph bias,
- delayed damage feedback to MPM.

### File-level summary
- `physics_projector.py`: produce initiation, growth, and direction cues separately
- `gaussian_fracture_field.py`: maintain local damage band around active front
- `crack_front.py`: new file for tip advancement logic
- `gaussian_graph.py`: surface-aware growth graph
- `manifold_simulator.py`: delayed coupling of damage back into MPM
- `physical_constitutive_models.py`: provide physics descriptors, not full crack logic
- `gaussian_updater.py`: render damage + front together
- `fracture_pipeline.py`: orchestrate the new causal order

### One-sentence conclusion
The correct next step is not to make the current damage field stronger, but to redesign the Gaussian fracture model as a **tip-based crack growth system with surface-aware graph propagation and delayed constitutive feedback**.

---

## 15. Compact Research Framing

A concise way to describe the resulting method is:

> We use MPM to provide impact-driven fracture cues, project those cues onto a surface-aware Gaussian graph, explicitly propagate crack tips as a front advancement process, update a local damage band around the active front, and couple damage back to MPM through a delayed schedule so that visible crack growth can emerge before early local collapse suppresses upward transmission.
