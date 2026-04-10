# Final Night-Run Graphics Specification
## Material-Aware Crack, Limited Detach, and the Last Implementation Branch
### CLIP Semantic Prior -> Fracture Family -> Crack Event -> Cut Surface -> Fragment Detach -> Shard / Debris / Render Polish

---

## 1. Purpose

This document is the **final implementation specification for the current project stage**, written from a **graphics expert / graphics research perspective**, and intended to guide an overnight implementation-and-validation pass.

At this point, the system has already achieved a meaningful milestone:

- material-aware crack behavior exists,
- family separation exists,
- fragment logic exists,
- concrete already shows stable split,
- glass already reaches late split,
- rubber correctly remains no-split,
- and ceramic is now the only major interpretive ambiguity.

This means the system is no longer asking:

> “Can fracture happen?”

The system is now asking:

> “What is the correct material-specific success criterion, and how should the final fracture event be turned into a convincing detach / shard / debris visual event?”

This document formalizes the next and likely final implementation branch.

---

## 2. Core Redefinition of Success

The most important conceptual change is:

> **We should now abandon the criterion that every material must necessarily reach full fracture/detach.**

That is no longer the correct standard.

Instead, the fracture target should now be **material-specific**.

### Recommended material-specific goals

| Family / Material Interpretation | Correct target |
|---|---|
| glass / `sharp_brittle` | narrow crack + cut-surface split expected |
| concrete / `rough_quasi_brittle` | rough crack network + stable fragment separation expected |
| ceramic / `brittle_moderate` | crack/opening must be clearly visible; detach is optional or delayed |
| rubber / `diffuse_damage` | explicit crack path and detach should be absent or nearly absent |

This should now be treated as the project’s working criterion.

This redefinition is not a compromise.
It is the correct graphics-oriented interpretation.

---

## 3. Current Implemented Structure

The following are already implemented and should now be treated as part of the stable base system.

### 3.1 CLIP semantic prior -> fracture family
Material prior is already mapped from pretrained CLIP retrieval into fracture family behavior.

Core file:
- `material_prior_adapter.py`

This is the family interpretation layer.

---

### 3.2 Crack generation and propagation
Crack generation is no longer handled by direct volumetric phase-field propagation.
Instead, it is handled by a **tip-based crack front**.

This means the following concepts are already structurally separated:
- initiation
- growth
- opening
- visited front

Main integration:
- `manifold_simulator.py`
- `graph_fragment_manager.py`

This is already a strong graphics-friendly formulation.

---

### 3.3 Fragment detection and memory
Fragment detection is active and includes:
- broken-edge memory
- detached-node hysteresis
- family-aware thresholds
- recent crack-front conditioning
- cut-core-based edge cutting

Main file:
- `graph_fragment_manager.py`

This means fragment detection is not hypothetical anymore.

---

### 3.4 Sharp-brittle cut support
The current sharp-brittle path already uses:

```text
cut_core -> support_field -> cross-edge vote -> hard cut
```

This is already a major structural improvement.

However, the system still does **not** yet include:
- explicit signed-plane side classification
- direct left/right cut-side separation logic

That missing piece is the current reason ceramic remains ambiguous as a split family.

---

### 3.5 Split event visibility
When split is detected:
- fragment impulse is applied
- visual offset is applied

This means the system already expresses weak detach events visually, not just graph-level separation.

Main file:
- `manifold_simulator.py`

---

### 3.6 Crack visualization
Crack visualization already reflects:
- crack core
- visited front
- tips
- opening

Main file:
- `gaussian_updater.py`

This means the fracture is already readable, not just internally computed.

---

### 3.7 Delayed damage feedback
Damage feedback to MPM is delayed.
So the old failure mode—where early bottom softening ruined the simulation—has already been escaped.

Configured in:
- `gravity_drop_manifold.yaml`

---

## 4. Current Verified Results

### 4.1 Glass
Run:
- `stepA5_glass_v8`

Observed:
- `first_split_frame = 78`
- `longest_split_run = 2`
- `final_n_frags = 2`
- `max_detached_distance = 0.1297`

Interpretation:
- late split exists
- split is real but still weak / short-lived
- narrow crack can already become detach-capable

Graphics interpretation:
> Glass is already no longer only a crack line; it is beginning to behave like a split family.

But it is not yet visually strong enough.

---

### 4.2 Concrete
Run:
- `stepA5_concrete_v1`

Observed:
- `first_split_frame = 46`
- `final_n_frags = 3`

Interpretation:
- stable split exists
- fragment separation is sustained

Graphics interpretation:
> Concrete is already a confirmed split-success family.

---

### 4.3 Rubber
Run:
- `stepA5_rubber_v1`

Observed:
- `split_frame_count = 0`
- `cracked(c > 0.3) = 0`

Interpretation:
- diffuse no-split mode is correct

Graphics interpretation:
> Rubber is behaving correctly and should remain this way.

---

### 4.4 Ceramic
Run:
- `stepA5_ceramic_v4`

Observed:
- crack and cut-edge events are clearly present
- but no detach occurs within 80 frames

Interpretation:
- ceramic is currently ambiguous but not necessarily wrong

Graphics interpretation:
> Ceramic can currently be treated as a valid crack-only family, unless the project explicitly requires delayed split.

This is now the key design branch.

---

## 5. What Is Still Missing

### 5.1 Debris scattering is not yet present
Right now the system has:
- fragment impulse
- visual offset

But not yet:
- visible debris burst
- shard scattering
- strong post-split breakup event

This means the current state is:
- detach-capable
but not yet
- cinematic fracture aftermath

---

### 5.2 Gaussian shard duplication is still disabled
Current state:
- `splitting_enabled: false`

So topology-changing shard duplication is still off.

That means the system is still at:
- split visualization
not yet
- explicit shard generation

---

### 5.3 Fragment rendering is weak
Current fragment rendering is still mild:
- slight shrink
- slight darkening

This is not enough for a strong fracture event.

---

### 5.4 Ceramic side-aware cut is missing
Current cut logic uses directionality, but not explicit side-aware separation.
This is why ceramic may crack strongly yet still fail to become a robust cut surface.

---

### 5.5 CLIP prior is semantic, not exact material calibration
The CLIP material prior already works well as a **family separator**.
But it should not be overclaimed as exact physical fracture parameter identification.

This is important both for interpretation and for debugging expectations.

---

## 6. The Main Strategic Decision: Ceramic

Ceramic is now the only major unresolved branch.

### Option A — Accept ceramic as crack-only family
Under this interpretation:
- ceramic must crack clearly
- ceramic must open clearly
- ceramic does **not** need to detach within 80 frames
- delayed split remains optional

This is currently the **recommended default**.

Why this is good:
- it preserves family distinction
- it avoids collapsing ceramic into “just weaker glass”
- it matches a plausible brittle-but-not-clean-split material interpretation
- it allows immediate progress toward shard/debris work using glass and concrete

---

### Option B — Require ceramic delayed split
This is only recommended if:
- the demo/paper explicitly needs ceramic to detach,
- ceramic looks too incomplete next to other brittle families,
- or delayed split is judged necessary for the final story.

If this option is chosen, then the next structural upgrade is:

> **local signed cut-plane / side-aware cross-edge cut**

This is the correct structural solution.
Threshold tuning alone is not enough.

This is a larger task and should not be attempted unless ceramic split is genuinely required.

---

## 7. Recommended Strategy

From a graphics expert perspective, the best strategy now is:

> **Approve ceramic as crack-only for now, and move immediately toward visible shard/debris event quality using glass and concrete as the main split-success families.**

This is the most productive path because:
1. glass already splits
2. concrete already splits stably
3. rubber already behaves correctly
4. ceramic can still be valid without detach
5. the next true visual milestone is not “more split,” but **better fracture aftermath**

This is the correct project-level prioritization.

---

## 8. Final Implementation Roadmap

The next implementation should follow this order:

```text
Phase 1. Lock material-specific success criteria
Phase 2. Push glass and concrete into shard/debris stage
Phase 3. Strengthen fracture event render quality
Phase 4. Run long-horizon delayed-split validation
Phase 5. Decide ceramic branch
Phase 6. If needed, implement side-aware ceramic cut-plane
Phase 7. Final beauty polish
```

This is the recommended path for the overnight implementation run.

---

## 9. Phase 1 — Lock Success Criteria by Family

### Goal
Make future runs unambiguous to evaluate.

### Required rule
Use the following as the active criterion:

| Family | Required for success |
|---|---|
| glass / `sharp_brittle` | crack + split + visible detach |
| concrete / `rough_quasi_brittle` | crack network + stable split |
| ceramic / `brittle_moderate` | crack + opening required; split optional/delayed |
| rubber / `diffuse_damage` | no explicit crack path / no detach |

### Why
Without locking this, all overnight runs become difficult to interpret.

### Implementation
No code needed, but this criterion should be written into:
- experiment notes
- validation checklist
- output naming / comparison workflow

---

## 10. Phase 2 — Move Glass and Concrete to Shard / Debris Stage

### Goal
Turn split-success families into visible fracture-event families.

At this stage, split should no longer merely be:
- a graph event
- a weak detach offset

It should become:
- a visible break
- a visible split gap
- a visible fragment event
- and eventually a shard/debris event

---

## 11. Phase 2 — What to Implement

### 11.1 Enable family-aware Gaussian splitting
Currently disabled:
- `splitting_enabled: false`

Recommended next action:
- enable for `sharp_brittle`
- enable for `rough_quasi_brittle`
- keep disabled for `diffuse_damage`
- optional disabled for `brittle_moderate`

This should be family-controlled.

### 11.2 Increase split gap visibility
A split that exists but looks glued is not yet good enough.

Increase:
- split gap magnitude
- local shell separation
- detached-surface contrast

Especially for:
- glass
- concrete

### 11.3 Add minimal shard logic
This does not have to be full debris explosion.
At this stage, the goal is:

- a detached fragment surface
- one or more visible shard-like substructures
- slight family-aware divergence after split

### 11.4 Keep family distinction
Shard behavior should remain family-aware:

#### `sharp_brittle`
- cleaner, fewer, sharper shards
- more decisive split gap
- less roughness

#### `rough_quasi_brittle`
- chunkier, rougher detached fragments
- broader shell damage around split
- more local irregularity

---

## 12. Phase 2 — Files to Modify

### `gravity_drop_manifold.yaml`
Add/adjust:
- `splitting_enabled` by family
- split-gap scale
- shard activation flags
- family-specific shard parameters

### `material_prior_adapter.py`
Add:
- `shard_enable`
- `shard_count_scale`
- `split_gap_gain`
- `fragment_shell_gain`
- `fragment_offset_gain`
- `debris_motion_gain`
- `debris_darkening`
- `fragment_contrast_gain`

### `manifold_simulator.py`
Add:
- family-aware shard activation
- split-event timing window
- family-aware post-split boost
- stronger detach event for split-success families

### `gaussian_updater.py`
Add:
- detached surface rendering
- split gap visibility
- family-aware fragment shell styling
- shard-like visibility logic

---

## 13. Phase 2 — Overnight Validation Experiments

### Batch A — Split-success family shard activation
Run:
- glass / 80 frame / shard on
- concrete / 80 frame / shard on

Measure:
- split onset
- visible split gap
- visible shard count
- max detached distance
- fragment shell contrast

Success:
- glass and concrete visually read as fracture events, not just crack states

---

### Batch B — Regression
Run:
- ceramic / 80 frame / shard off
- rubber / 80 frame / shard off

Goal:
- ceramic remains crack-only intermediate family
- rubber remains diffuse no-split

---

## 14. Phase 3 — Strengthen Eventness in Rendering

### Goal
Move from debug split to strong visible fracture event.

Current weakness:
- split exists but feels mild
- fragment shell is weak
- debris is only slightly smaller/darker
- eventness is not yet cinematic

### What to strengthen

#### 14.1 Fragment shell
Enhance:
- shell contrast
- detached surface visibility
- family-aware shell shape

#### 14.2 Split gap
Enhance:
- gap readability
- edge shading around gap
- local opacity falloff near split

#### 14.3 Shard motion
Add:
- slightly stronger divergence
- family-aware shard drift
- avoid explosion-like instability

#### 14.4 Temporal eventness
Ensure the event reads across time:
- onset
- detach
- sustain

This is more important than a single strong frame.

---

## 15. Phase 3 — Files to Modify

### `gaussian_updater.py`
This becomes the main eventness file.

Add:
- fragment shell polish
- split-gap emphasis
- shard visibility composition
- detached fragment shading

### `manifold_simulator.py`
Add:
- family-aware event boost window
- split-event timer
- brief fragment motion amplification

### Optional helper
If the updater becomes crowded, introduce:
- `fragment_render_style.py`

This can isolate family-aware event styling from geometry logic.

---

## 16. Phase 3 — Validation Experiments

### Batch C — Beauty-render detach comparison
For:
- glass
- concrete
- ceramic
- rubber

Save:
- split onset frames
- detached event windows
- shard emphasis frames

Goal:
- verify that split-success families now read as fracture events in final render

### Batch D — Short animation comparison
Render 20~40 frame clips around split onset.

Goal:
- verify temporal eventness
- ensure the event is readable, not just the endpoint

---

## 17. Phase 4 — Long-Horizon Validation

### Goal
Check delayed split / delayed detach beyond 80 frames.

### Why
Now that split-success families exist, the next question becomes:
- does detach persist?
- does delayed split appear?
- does ceramic deserve split-family promotion?

### Recommended horizons
- 80
- 120
- 160
- 200 frames

### Priority targets
1. glass
2. ceramic
3. concrete regression
4. rubber regression

---

## 18. Phase 4 — Validation Experiments

### Batch E — Glass delayed detach
Run:
- glass / 120
- glass / 160
- glass / 200

Measure:
- `longest_split_run`
- `final_n_frags`
- `max_detached_distance`
- visible shard persistence

Success:
- late split becomes sustained and visually strong

---

### Batch F — Ceramic delayed split interpretation
Run:
- ceramic / 120
- ceramic / 160
- ceramic / 200

Two acceptable outcomes:
1. ceramic remains crack-only -> approve crack-only family
2. ceramic shows delayed partial split -> candidate for upgrade

This batch is not just validation.
It is the ceramic decision batch.

---

### Batch G — Impact-energy sweep
Run energy variants for:
- glass
- ceramic
- concrete

Goal:
- verify that family interpretation is not only an artifact of one impact regime

This is especially useful before finalizing ceramic as crack-only vs delayed-split.

---

## 19. Ceramic Decision Tree

### If ceramic remains crack-only through long horizon
Then:
- approve ceramic as intermediate crack-only family
- do not implement signed cut-plane yet
- proceed to final beauty/render polish

This is the recommended path unless ceramic visibly demands more.

---

### If ceramic shows near-split but unstable delayed behavior
Then:
- ceramic becomes a candidate for split-family upgrade
- next implementation is:

> **local signed cut-plane / side-aware cross-edge cut**

This should only be done if long-horizon evidence justifies it.

---

## 20. If Ceramic Signed Cut-Plane Becomes Necessary

### Required upgrade
Current:
```text
cut_core -> support_field -> cross-edge vote -> hard cut
```

Next:
```text
cut_core -> local signed cut-plane -> side classification -> side-aware cross-edge cut
```

### Why
The current logic knows directionality but not explicit side separation.
That is exactly what thin ceramic cracks need if they are to become reliable cut surfaces.

### Files
- `graph_fragment_manager.py`
- optional local plane helper
- `material_prior_adapter.py`
- `smoke_test.py`

### Important rule
Do not start this task unless ceramic split is genuinely needed.

---

## 21. Recommended Overnight Run Schedule

### Run Group 1 — Reconfirm current family status
- glass / 80 / shard off
- concrete / 80 / shard off
- ceramic / 80 / shard off
- rubber / 80 / shard off

Goal:
- lock the baseline state

---

### Run Group 2 — Shard/debris activation
- glass / 80 / shard on
- concrete / 80 / shard on
- ceramic / 80 / shard off
- rubber / 80 / shard off

Goal:
- push split-success families into visible event stage

---

### Run Group 3 — Long-horizon delayed split
- glass / 120 / 160 / 200
- ceramic / 120 / 160 / 200
- concrete / 120 regression
- rubber / 120 regression

Goal:
- determine whether ceramic remains crack-only or needs promotion

---

### Run Group 4 — Energy sweep (optional if time allows)
- glass energy sweep
- ceramic energy sweep
- concrete energy sweep

Goal:
- test family robustness across impact conditions

---

## 22. Metrics to Log

### Structural
- `first_split_frame`
- `longest_split_run`
- `final_n_frags`
- `max_n_frags`
- `split_frame_count`
- `broken_edge_count`
- `cross_edge_break_count`

### Detach
- `max_detached_distance`
- `mean_detached_distance`
- `split_event_duration`

### Render/Eventness
- split gap visibility
- fragment shell contrast
- visible shard count
- shard persistence

### Interpretation
- family
- crack-only vs split-success
- delayed split yes/no

---

## 23. Expert Tips

### Tip 1
Do not force ceramic into split unless the project really needs it.
An intermediate crack-only brittle family is fully valid in graphics.

### Tip 2
Glass and concrete should now carry the debris/shard burden.
They are the correct families to push into visible fracture aftermath.

### Tip 3
The next goal is not more path count.
It is:
- crack surface readability
- detach readability
- shard readability
- family-specific eventness

### Tip 4
When shard rendering is first enabled, exaggeration is fine.
The first goal is readability, not subtlety.

### Tip 5
Do not polish beauty too early.
First make sure the event is structurally clear and temporally sustained.

### Tip 6
Ceramic should be treated as a deliberate design choice, not as an unresolved accident.
Decide explicitly whether it is:
- crack-only
or
- delayed-split

and then commit.

---

## 24. Final Recommended Priority Order

From a graphics expert perspective, the correct next priorities are:

1. **Lock material-specific success criteria**
2. **Approve ceramic as crack-only by default**
3. **Enable shard/debris stage for glass and concrete**
4. **Strengthen fragment shell / split gap / detach render eventness**
5. **Run 120~200 frame delayed-split validation**
6. **Only if needed, implement ceramic signed cut-plane**
7. **Finish with shard/debris motion and beauty polish**

This is the recommended final branch.

---

## 25. Final Conclusion

The current system has already succeeded at:
- material-aware crack generation
- limited but real fragment separation
- stable concrete split
- late glass split
- correct rubber no-split behavior

So the project is no longer in a fracture-finding phase.
It is in a **fracture event design phase**.

The correct next move is not to force all materials into the same detach behavior.
Instead, the system should:
- lock success criteria by family,
- treat ceramic as crack-only unless delayed split is truly required,
- push glass and concrete into shard/debris event quality,
- validate delayed split over longer horizons,
- and only then decide whether ceramic needs a larger signed-cut upgrade.

### One-sentence summary
From a graphics expert perspective, the project is now ready to move from “material-aware crack with limited detach” into “family-specific fracture event and shard/debris rendering,” using glass and concrete as the main split-success families, keeping rubber diffuse, and treating ceramic as an explicitly chosen intermediate family unless long-horizon evidence proves otherwise.
