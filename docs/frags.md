# Graphics-Oriented Implementation Specification
## Step-by-Step Roadmap for Fragment Separation, Fracture Event Quality, and Render Validation
### CLIP Semantic Prior -> Fracture Family -> Crack Growth -> Fragment Separation -> Render Quality

---

## 1. Purpose

This document defines a **practical implementation roadmap** for the next stage of the current fracture system from a **graphics-oriented expert perspective**.

The key point is that the system has already reached a meaningful milestone:

- crack generation works,
- crack propagation works,
- material-family separation has begun to emerge,
- and the result is no longer just a bottom damage stain.

However, the current system is still at an intermediate stage:

> **crack generation/propagation is now largely working, but true fragment separation and final fracture-event quality are not yet complete.**

So the next implementation direction should be structured in a staged way.

This document explains:
1. what should be implemented first,
2. how each stage should be validated,
3. when it is safe to move to the next stage,
4. and how render quality should be integrated into the process.

The philosophy is:

> **Do not optimize everything at once.**
> Build the fracture system in ordered layers:
> crack -> separation -> event quality -> render quality.

---

## 2. Current State: What Is Already Good

From a graphics perspective, the current system can already be evaluated positively in one important way:

- it is no longer stuck in the old “damage stain near the bottom” failure mode,
- the crack path now enters the object in the correct qualitative direction,
- and material families are beginning to separate:
  - sharp path,
  - rough network,
  - diffuse response.

That means the current crack structure is **now moving in the right direction**.

This is important because it means the system is no longer in the “does fracture even work?” stage.

It is now in the more advanced stage of:

> **how to make fracture events visibly convincing and materially specific.**

---

## 3. The Most Important Principle Going Forward

The current temptation is to improve:
- glass saliency,
- ceramic arrest/restart,
- concrete patchiness,
- and fragment separation,

all at once.

That is **not recommended**.

From a graphics systems perspective, the correct order is:

```text
Stage 1: true fragment separation
Stage 2: family-specific fracture event quality
Stage 3: final render quality refinement
```

Why?

Because if fragment separation is not yet physically/structurally visible, then it becomes unclear whether a weak result is caused by:
- poor split threshold,
- weak edge break,
- weak opening,
- poor render saliency,
- or material-family tuning.

So before polishing fracture style, the system should first reliably answer:

> **Can a crack path actually detach into multiple visible fragments?**

Only once that is true should the system push material-specific quality aggressively.

---

## 4. Global Roadmap

The recommended roadmap is:

```text
Step A. Fragment computation
Step B. Fragment validation
Step C. Visible detach validation
Step D. Family-aware fracture event quality
Step E. Family-aware render quality
Step F. Final comparison experiments
```

The rest of this document expands each step in detail.

---

## 5. Step A — Make True Fragment Separation Happen

## Goal
The first next-stage goal is:

> **make true fragment separation actually occur and become visible**

At the moment, the system already has:
- crack band formation,
- edge weakening,
- graph fragment detection,
- and a per-fragment transition pathway.

But recent runs suggest that in practice:
- crack generation happens,
- but graph connectivity is still not often broken strongly enough,
- so the system rarely reaches a clear visible detached-fragment regime.

So Step A is not about beauty.
It is about structural success.

---

## 6. What Step A Should Implement

### 6.1 Strengthen crack opening and split conditions
The crack should not only exist as a band.
It must produce:
- larger opening,
- stronger local split conditions,
- and higher chance of actual graph disconnection.

What to adjust:
- opening gain,
- split threshold,
- local separation geometry offset,
- post-split velocity divergence.

---

### 6.2 Make fragment detection thresholds family-aware
The same graph disconnection criterion should not necessarily apply to all families.

Recommended direction:
- `sharp_brittle`: easier clean edge break
- `brittle_moderate`: moderate break threshold
- `rough_quasi_brittle`: rougher but still breakable
- `diffuse_damage`: very difficult or nearly impossible to break

This is especially important because fragment separation is not only about crack existence.
It is about connectivity failure.

---

### 6.3 Strengthen visible detach immediately after split
Even if connected components are correctly detected, the result may still not look detached enough.

So after graph disconnection:
- geometry offset should be made visible,
- velocity difference between fragments should be emphasized,
- and fragment identities should not collapse visually back together.

This is extremely important for graphics.
A mathematically separated fragment that still looks visually glued is not yet a success.

---

## 7. Step A — Files to Modify

### `graph_fragment_manager.py` or equivalent
Focus here first.

Key responsibilities:
- edge integrity update,
- graph disconnection,
- connected-component computation,
- fragment labeling,
- split condition logic.

Add or adjust:
- family-aware break threshold,
- family-aware edge break rate,
- stronger disconnection after active split events.

---

### `manifold_simulator.py`
This file should manage:
- when split is allowed,
- when per-fragment MPM starts,
- and how strong initial fragment separation impulse is.

Adjust:
- post-split impulse,
- per-fragment velocity divergence,
- post-separation geometry offset,
- delayed / gated transition into per-fragment motion.

---

### `gaussian_updater.py`
In this stage, it should support:
- stronger visible local opening,
- fragment-aware visual separation,
- and temporary exaggeration of split visibility for debugging.

At this stage, graphics debugging is more important than photorealistic subtlety.

---

## 8. Step A — Validation Experiments

At this stage, do **not** jump to beauty render evaluation only.
First validate the structure.

### Experiment A1 — Graph disconnection test
For fixed geometry / impact:
- run concrete
- run glass
- run ceramic
- run rubber

Measure:
- first frame where `n_frags > 1`
- first frame where edge break exceeds threshold
- number of connected components over time
- broken edge count over time

Expected:
- concrete: should separate first or most easily
- glass: should eventually separate cleanly
- ceramic: moderate / delayed separation
- rubber: no real separation

---

### Experiment A2 — Visible detach test
Same experiment, but now inspect render/debug frames.

Save:
- beauty frame
- fragment label visualization
- opening heatmap
- broken-edge overlay
- per-fragment motion overlay

Goal:
- verify that graph disconnection is actually visible in image space

This matters because purely graph-level separation is not enough.

---

### Step A Completion Criteria
You may move to the next stage only if:

1. At least one brittle or quasi-brittle family reliably reaches `n_frags > 1`
2. The detach is visually recognizable in render/debug frames
3. Rubber still does **not** visibly split
4. Fragment labels remain stable for multiple frames after split

If these are not satisfied, do **not** proceed to fracture style refinement yet.

---

## 9. Expert Tip for Step A

### Tip 1
When debugging fragment separation, temporarily exaggerate split visibility.
This is not “cheating.”
It is a diagnostic tool.

For example:
- temporarily scale up fragment offset,
- temporarily scale up post-split velocity divergence,
- temporarily increase shell separation around broken edges.

If the result becomes visible only under exaggeration, then the fragment logic may be correct but underpowered visually.

That is valuable information.

### Tip 2
Always inspect graph disconnection and render detachment together.
If graph disconnection occurs but visible split is weak, the issue is rendering.
If no graph disconnection occurs, the issue is connectivity / fracture coupling.

Do not mix those two failure modes.

---

## 10. Step B — Stabilize Fragment Separation Across Families

Once fragment separation is visible at least in one family, the next task is to stabilize it.

## Goal
Turn fragment separation from a lucky event into a material-aware structural feature.

At this stage:
- the graph split exists,
- but the family distinction may still be weak.

So now we ask:

> **Does each material family separate in the correct way?**

---

## 11. What Step B Should Implement

### 11.1 Family-specific split style
Now use fracture family to change:
- split timing,
- split aggressiveness,
- fragment count tendency,
- and separation cleanliness.

Examples:

#### `sharp_brittle`
- cleaner split
- fewer but more decisive detached pieces
- higher opening and stronger local edge break

#### `brittle_moderate`
- moderate split
- sometimes partial detach
- not as dramatic as glass

#### `rough_quasi_brittle`
- rougher split
- more broken edges
- more irregular fragment shapes

#### `diffuse_damage`
- almost no split

---

### 11.2 Separate “split existence” from “split style”
This is crucial.

First confirm:
- split exists

Then refine:
- how cleanly it exists
- how many fragments form
- how visually sharp the split is

Do not try to solve both at the same time.

---

## 12. Step B — Validation Experiments

### Experiment B1 — Family separation comparison
Same geometry, same impact, same camera.

Compare:
- glass-like / `sharp_brittle`
- ceramic-like / `brittle_moderate`
- concrete-like / `rough_quasi_brittle`
- rubber-like / `diffuse_damage`
- neutral baseline

Measure:
- split onset frame
- number of fragments
- fragment size distribution
- broken edge count
- visual split cleanliness

Expected:
- glass: earlier and cleaner split
- ceramic: moderate split, less aggressive
- concrete: rougher fragmentation
- rubber: nearly no split

---

### Experiment B2 — Family ablation
For each family, turn off family-specific split parameters and compare.

Goal:
- verify that family-aware split priors actually matter

This is critical for debugging and future paper argumentation.

---

### Step B Completion Criteria
Move forward only if:
1. Family separation behavior is visually distinguishable
2. Glass-like, concrete-like, and rubber-like no longer collapse into the same split mode
3. Neutral baseline stays moderate

---

## 13. Expert Tip for Step B

### Tip 3
Do not judge glass by fragment count alone.
For glass-like fracture, the main markers are often:
- clean opening,
- strong split saliency,
- and dominant primary fracture event.

More fragments is not always more glass-like.

### Tip 4
Do not judge concrete by opening alone.
Concrete-like fracture often wins by:
- roughness,
- patchiness,
- and fractured zone texture,
not by maximum clean opening.

This distinction is very important.

---

## 14. Step C — Improve Fracture Event Quality

Once true fragment separation works, then it becomes correct to improve family-specific fracture event quality.

This is where the current partially successful results should be refined.

## Goal
Make each family not only structurally distinct, but also **qualitatively correct in event character**.

At this stage, the key targets are:

- glass -> sharper, more open, more dominant crack event
- ceramic -> interrupted brittle event (arrest/restart)
- concrete -> rougher, patchier fracture texture
- rubber -> stable diffuse mode

---

## 15. Glass Implementation Direction

## Current issue
The current glass main crack exists, but still looks too much like a clean graph trajectory.

## Desired result
The crack should feel like:
- a true fracture event,
- not just a path.

## What to implement
### 15.1 Suppress visited trail saliency
Reduce the contribution of:
- old visited path
- weak trail shell
- broad halo

### 15.2 Increase core and opening saliency
Increase:
- core crack contrast,
- opening magnitude near active segments,
- split emphasis near broken edges.

### 15.3 Make split more event-like
Use stronger local split only where:
- core damage is high,
- opening is high,
- edge break is active,
- and recent front motion exists.

### 15.4 Add slight micro-irregularity
Do not keep the crack too perfectly graph-clean.
Add mild thickness or opacity irregularity around the core.

Not rough like concrete.
Just enough to remove synthetic line appearance.

---

## 16. Ceramic Implementation Direction

## Current issue
Ceramic is too topologically similar to glass.

## Desired result
Ceramic should feel like:
- brittle,
- but less cleanly continuous than glass,
- with pauses, restarts, and small secondary events.

## What to implement
### 16.1 Arrest probability
Introduce local crack-front pausing.

### 16.2 Restart behavior
Allow paused cracks to restart if nearby drive remains sufficient.

### 16.3 Short-lived secondary branches
Permit small side branches that die early.

### 16.4 Moderate opening
Keep opening lower than glass.

This combination creates a clearly different material signature.

---

## 17. Concrete Implementation Direction

## Current issue
Concrete is already convincing, but still reveals graph regularity.

## Desired result
Concrete should feel:
- rough,
- patchy,
- and locally irregular.

## What to implement
### 17.1 Correlated patchy roughness
Do not use only per-node random variation.
Use locally correlated variation across neighborhoods.

### 17.2 Patchy band fill
Some local zones should fill strongly, others weakly.

### 17.3 Rough shell overlay
Render a fractured rough shell around the main path rather than only a clean thin line.

### 17.4 Chunk-wise edge weakening
Allow irregular local break patterns rather than perfectly smooth propagation.

This is the correct graphics direction for concrete-like fracture.

---

## 18. Rubber Implementation Direction

## Current issue
The current direction is already correct.

## Desired result
Maintain:
- no explicit crack path,
- diffuse response,
- little to no split.

## What to implement
Mostly preserve current mode.

Optional improvement:
- make diffuse damage or deformation visually softer and more plausible,
- but do not reintroduce front-like visibility.

---

## 19. Step C — Validation Experiments

### Experiment C1 — Event quality comparison
For each material family, inspect:
- main crack saliency
- opening
- trail visibility
- patchiness
- arrest/restart behavior
- split cleanliness

Use fixed:
- geometry
- impact
- camera
- frame checkpoints

Recommended checkpoints:
- 20
- 40
- 60
- 80 frames

---

### Experiment C2 — Render ablation
For the same solver output, vary render composition:

- with / without trail
- low / high opening emphasis
- patchy shell off / on
- split saliency off / on

Goal:
- determine whether a material issue is actually solver-side or render-side.

This is especially important for glass.

---

### Experiment C3 — Event ablation
Individually disable:
- glass split focus
- ceramic arrest/restart
- concrete patchiness

Goal:
- verify which event module is responsible for family identity.

---

### Step C Completion Criteria
This stage is successful when:
1. Glass no longer looks like just a graph trajectory
2. Ceramic is clearly distinguishable from glass without relying only on weaker opening
3. Concrete looks rougher and patchier, not just more branched
4. Rubber remains diffuse and non-cracking

---

## 20. Expert Tip for Step C

### Tip 5
At this stage, path count is no longer the main metric.
The main metrics become:
- saliency,
- opening,
- event timing,
- roughness character,
- and split visibility.

This is the correct shift in evaluation logic.

### Tip 6
Keep solver evaluation and render evaluation separate.
A family may already be correct structurally but still look wrong because of render composition.
Do not over-tune the solver to fix a rendering problem.

---

## 21. Step D — Final Render Quality Pass

Only after the structural behavior and event behavior are correct should the final render-quality pass begin.

## Goal
Make the final fracture output visually convincing enough for presentation, figures, demos, or paper videos.

At this stage:
- the fracture should already be correct,
- now the job is to make it look polished.

---

## 22. What Step D Should Refine

### 22.1 Beauty render tuning
- shell blending
- opacity curves
- split visibility polish
- edge shading / contrast
- temporal smoothness of visible crack growth

### 22.2 Family-specific beauty polish
- glass: cleaner and sharper highlight
- ceramic: brittle but interrupted
- concrete: rough, fragmented shell
- rubber: soft diffuse damage only

### 22.3 Temporal polish
Ensure the fracture progression over time does not flicker or jump visually in an unnatural way.

---

## 23. Step D — Validation Experiments

### Experiment D1 — Beauty render comparison
Generate a curated side-by-side comparison:
- same object
- same impact
- all family prompts
- same camera
- same frames

This is the final graphics-facing validation.

### Experiment D2 — Short animation validation
Render short animations for each family and inspect:
- temporal coherence,
- opening visibility,
- split timing,
- and family distinction over time.

This is often more informative than single frames.

---

## 24. Recommended Master Evaluation Table

A recommended internal evaluation table for each run:

| Prompt | Family | Crack Exists | `n_frags>1` | Visible Detach | Main Crack Saliency | Opening | Roughness | Trail Visibility | Fragment Style | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| glass bunny | sharp_brittle | yes/no | yes/no | yes/no | low-mid-high | low-mid-high | low | low | clean split / none | pass/fail |
| ceramic bunny | brittle_moderate | yes/no | yes/no | yes/no | low-mid-high | low-mid-high | low-mid | low-mid | moderate / interrupted | pass/fail |
| concrete bunny | rough_quasi_brittle | yes/no | yes/no | yes/no | low-mid-high | low-mid | mid-high | mid | rough / patchy | pass/fail |
| rubber bunny | diffuse_damage | yes/no | yes/no | yes/no | very low | very low | diffuse | none | none | pass/fail |
| default | neutral_reference | yes/no | yes/no | yes/no | mid-low | mid-low | mid | mid-low | neutral | pass/fail |

This should be used throughout development.

---

## 25. Recommended Order in One Line

The correct implementation order is:

```text
fragment separation -> fragment validation -> visible detach validation -> family-specific fracture event quality -> family-specific render quality -> final comparison experiments
```

This is the safest and most interpretable development path.

---

## 26. What Not to Do

At this stage, avoid:
- polishing glass saliency before true split exists,
- tuning concrete roughness before visible detach is validated,
- treating fragment graph disconnection as sufficient without render confirmation,
- mixing rubber back into explicit crack-front mode,
- judging success only from internal state without beauty render.

These lead to confusion and unstable iteration.

---

## 27. Final Summary

### Main current status
- crack generation / propagation is successful,
- material-family mode split is emerging,
- true fragment separation is still incomplete.

### Correct next move
First make true fragment separation visible and stable.
Only after that, improve family-specific fracture event quality.
Finally, refine render quality.

### Why this is correct
Because graphics validation depends on:
1. structural split,
2. event-level distinction,
3. visible render saliency,
in that order.

### Practical sequence
1. strengthen split / edge break / detach
2. verify graph split and visible detach
3. refine glass / ceramic / concrete event quality
4. polish final family-aware rendering

### One-sentence conclusion
From a graphics expert’s perspective, the next implementation should proceed in ordered layers: first ensure that crack paths truly detach into visible fragments, then shape those fracture events so each material family has the correct character, and only after that finalize the beauty of the rendered result.
