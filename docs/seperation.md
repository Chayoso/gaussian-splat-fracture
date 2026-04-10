# Final Graphics Wrap-Up Roadmap
## Cut-Surface Promotion, Support-Loss Separation, and the Last Steps to a Convincing Fracture Event
### Material-Aware Crack -> Structural Cut -> Support Loss -> Detach -> Shard / Debris Polish

---

## 1. Purpose of This Document

This document summarizes the **final implementation direction** of the current system from a **graphics expert perspective**.

At this stage, the project is no longer about:
- whether cracks can appear,
- whether material families can be distinguished,
- or whether fragment logic exists in principle.

Those milestones have already been reached.

The current bottleneck is now more specific:

> **visible cracks exist, but they are not yet consistently promoted into structural cuts that make the body lose support and collapse in a visually convincing way.**

So the purpose of this document is to define:

1. how to interpret the current state correctly,
2. what the final missing implementation layer is,
3. how to finish the project in the right order,
4. and how to avoid wasting effort on the wrong final polish too early.

This document is written as a practical graphics-oriented wrap-up plan.

---

## 2. Current System Status: Correct Interpretation

The current system can already be considered successful in several important ways.

### 2.1 Material-aware fracture family behavior exists
The system already translates CLIP semantic priors into fracture-family behavior.
This is a major design success.

### 2.2 Crack generation is no longer the main problem
Crack initiation, tip-based propagation, local crack band formation, and opening are all already functioning.

### 2.3 Fragment logic exists and is not hypothetical
Graph-based fragment detection, broken-edge memory, detached-node hysteresis, family-aware thresholding, and weak detach events are all already in the system.

### 2.4 Family separation is real
- glass-like behavior can reach late split,
- concrete-like behavior can reach stable split,
- rubber-like behavior correctly avoids split,
- ceramic remains the only ambiguous family.

That means the project has already crossed the stage of “does fracture even work?”

---

## 3. The New Bottleneck

From a graphics perspective, the current bottleneck is now:

> **not crack rendering, but structural fracture topology.**

More specifically, the main missing layer is:

> **cut-surface formation + support-loss separation**

This is the most important statement in the current phase.

Crack-rich results are already possible.
But many of those cracks are still only:
- visually plausible crack lines,
not yet
- actual separating surfaces that make the body lose support and collapse.

This is especially visible in concrete-like results:
- there are many cracks,
- but the object still reads as under-fractured compared to how broken it appears.

This means the project is now at the transition between:

- **crack visualization**
and
- **true structural fracture event**

---

## 4. Why This Matters in Graphics

In graphics, fracture only becomes convincing when three layers align:

### Layer 1 — Visible crack
A crack path exists and can be seen.

### Layer 2 — Structural cut
That crack is not just a line; it becomes a real separating surface.

### Layer 3 — Support loss / collapse
Because of that cut, part of the object loses support and falls, splits, or collapses.

At the moment:
- Layer 1 is already good,
- Layer 2 is only partial,
- Layer 3 is still too weak.

So the final missing implementation should focus on Layers 2 and 3.

---

## 5. Updated Material-Specific Success Criteria

It is now correct to **abandon the idea that all materials must necessarily fully detach**.

The proper graphics-oriented success criteria are:

| Family / Material Interpretation | Correct target |
|---|---|
| glass / `sharp_brittle` | narrow crack + cut-surface split |
| concrete / `rough_quasi_brittle` | rough crack network + stable fragment separation + visible support-loss collapse |
| ceramic / `brittle_moderate` | crack/opening clearly visible; detach optional or delayed |
| rubber / `diffuse_damage` | no explicit crack path / no detach |

This criterion should now be treated as fixed.

This is important because the final implementation should optimize toward these targets, not toward a universal split requirement.

---

## 6. Correct Interpretation of Each Family

### 6.1 Glass
Glass should read as:
- narrow crack,
- sharp cut,
- clean split,
- clear detach.

The key visual cue is not necessarily many cracks.
It is:
- a small number of strong cracks that truly separate.

### 6.2 Concrete
Concrete should read as:
- rough crack network,
- broad fracture zone,
- support loss,
- chunk-like release,
- progressive collapse.

The key visual cue is not just crack count.
It is:
- **crack-rich but structurally consequential fracture**

### 6.3 Ceramic
Ceramic should read as:
- brittle crack,
- visible opening,
- but not necessarily full detach in short horizon.

At this stage, it is perfectly acceptable to keep ceramic as an intermediate crack-only family unless delayed split becomes clearly necessary.

### 6.4 Rubber
Rubber should continue to read as:
- diffuse damage / deformation,
- no explicit crack topology,
- no fragment detach.

This family should not be pulled back into brittle fracture logic.

---

## 7. What Is Still Missing

The remaining missing pieces are now clear.

### 7.1 Crack-to-cut-surface promotion is still incomplete
The system already has:
- crack core,
- opening,
- cross-edge vote,
- hard cut.

But not every visible crack is yet promoted into a true structural cut.

This is now the first critical missing layer.

### 7.2 Support-aware fragment promotion is still weak
Current fragment promotion is still too conservative and too geometry-centric.
It does not yet strongly answer:
- did this piece lose real support from the body or ground?

This is the second critical missing layer.

### 7.3 Falling / collapse readability is still weak
Even when fragments exist, the current render does not yet strongly communicate:
- support loss,
- body separation,
- or collapse direction.

This is the third missing layer.

### 7.4 Debris / shards are still intentionally disabled
This is fine.
At the current stage, debris should remain off until cut-surface and support-loss logic become stronger.

---

## 8. The Final Missing Structural Layer

The final missing structural layer should be defined as:

> **Crack-to-cut-surface promotion + support-loss-based fragment release**

This should now be the central implementation goal.

That means:
1. not all cracks are equal,
2. only some cracks become **authoritative cuts**,
3. those authoritative cuts reduce support connectivity,
4. and once support is lost, a fragment is promoted and released.

This is the correct abstraction.

---

## 9. Final Recommended Implementation Order

The correct final implementation order is:

```text
Phase 1. Add authoritative cut-surface promotion
Phase 2. Add support-aware fragment promotion
Phase 3. Strengthen falling / collapse readability
Phase 4. Revalidate long-horizon split behavior
Phase 5. Decide ceramic final status
Phase 6. Only then enable shard/debris stage
Phase 7. Final beauty render polish
```

This is the recommended wrap-up path.

---

## 10. Phase 1 — Add Authoritative Cut-Surface Promotion

### Goal
Convert a subset of visible cracks into true structural cuts.

### Why
Currently many visible cracks remain only visual or local graph events.
They are not yet granted enough structural authority.

### Correct concept
Not every crack should become a cut.
Instead:
- visible cracks exist in many places,
- but only some become **authoritative cut surfaces**.

This avoids noisy over-fragmentation.

### What should define an authoritative cut
A crack segment should be promoted when it has:
- high crack-core confidence,
- persistent opening,
- sufficient cross-edge break density,
- continuity along a meaningful crack segment,
- and enough support-field evidence.

### Main file
- `graph_fragment_manager.py`

### Recommended outputs
Introduce explicit state such as:
- `authoritative_cut_mask`
- `authoritative_cut_score`
- `cut_surface_id`
- `cut_surface_persistence`

### Graphics interpretation
This layer answers:
> which cracks are actually allowed to cut the object?

That is the correct next question.

---

## 11. Phase 2 — Add Support-Aware Fragment Promotion

### Goal
Make fragments detach because they lose support, not only because they are a disconnected component of sufficient size.

### Why
At the moment, the system is too conservative and too component-centric.
That is why it can still look crack-rich but under-fractured.

A proper fracture topology event needs:
- a structural cut
- followed by a support-loss release.

### Correct concept
Fragment release should consider:
- support path to the ground or main body,
- connectivity to lower supporting regions,
- whether the component is now suspended only by weak cut edges,
- and whether gravity should now dominate.

### Main files
- `graph_fragment_manager.py`
- `manifold_simulator.py`

### Recommended new logic
Introduce:
- `support_connectivity_score`
- `support_lost_mask`
- `collapse_candidate_score`
- `release_candidate_id`

### Practical recommendation
Do not globally relax fragment size thresholds.
Instead:
- keep them conservative globally,
- but locally relax promotion near authoritative cut surfaces and support-loss zones.

This is much more stable.

### Graphics interpretation
This phase answers:
> which pieces are no longer truly supported and should now fall?

That is what makes fracture read as collapse.

---

## 12. Phase 3 — Strengthen Falling / Collapse Readability

### Goal
Make structural separation visibly read as:
- cut
- release
- falling
- collapse

### Why
Even if topology is correct, the render can still feel weak if the released fragment does not look physically liberated.

### What to strengthen
- detached-body separation from the parent body,
- gravity-direction movement readability,
- contrast between still-supported body and released body,
- support-loss zone visualization,
- fragment silhouette separation.

### Main files
- `manifold_simulator.py`
- `gaussian_updater.py`
- `smoke_test.py`

### Important note
At this stage, the PNG/debug render should shift from:
- crack heatmap dominant
to
- fracture-topology / detached-body dominant

That is an important conceptual shift.

### Graphics interpretation
This phase answers:
> does the object actually look like it lost support and broke apart?

---

## 13. Phase 4 — Revalidate Long-Horizon Behavior

### Goal
Move beyond short-horizon runs and determine delayed behavior correctly.

### Why
The current 80-frame interpretation is enough for:
- concrete
- rubber

but not fully enough for:
- glass
- ceramic

### Required frame horizons
- 80
- 120
- 160
- 200

### Priority materials
1. glass
2. ceramic
3. concrete regression
4. rubber regression

### Key question
- Does glass sustain detach?
- Does ceramic stay crack-only?
- Or does ceramic become delayed-split material?

This is the phase that resolves the ceramic ambiguity.

---

## 14. Phase 5 — Decide Ceramic’s Final Role

### Recommended default
Keep ceramic as:
- crack/opening success family
- delayed split optional
- detach not required

This is currently the most graphics-consistent interpretation.

### Only upgrade ceramic if necessary
Ceramic should only be promoted to split-family if:
- long-horizon tests show consistent delayed split,
or
- the final paper/demo explicitly needs ceramic detach.

### If ceramic split is required
Then the next structural upgrade is:

> **local signed cut-plane / side-aware cross-edge cut**

This is correct because the current directional cut logic is not enough to reliably classify crack sides.

This should only be implemented if ceramic truly needs promotion.

---

## 15. Phase 6 — Only Then Enable Shards / Debris

### Goal
Add shards and debris only after structural fracture topology is convincing.

### Why
If shards are enabled too early, the system risks going back to:
- “it looks broken”
instead of:
- “it is structurally broken”

This is the wrong order.

### Correct condition for enabling shards
Only after:
- authoritative cuts exist,
- support-loss release exists,
- falling detached bodies are readable,
- and family targets are stable.

### Recommended shard policy
- enable for glass
- enable for concrete
- keep off for rubber
- optional for ceramic

### Graphics interpretation
Shards are **after-fracture aftermath**, not a substitute for structural fracture.

---

## 16. Phase 7 — Final Beauty Render Polish

### Goal
Once topology and detach are correct, polish the visual event.

### What should be polished
- detached fragment shell contrast
- split gap readability
- gravity-driven fragment motion clarity
- family-specific fragment appearance
- shard silhouette
- temporal coherence

### Important rule
Do not do this too early.
Beauty polish should come last.

---

## 17. Concrete-Specific Guidance

Concrete is now the family that most clearly demonstrates the current gap.

### Current concrete status
- cracks are many,
- some fragment bodies detach,
- but the object still does not fully read as sufficiently collapsed.

### Correct next move for concrete
Do **not** just increase crack count.
Instead:
- promote some visible cracks into authoritative cuts,
- use support-loss logic to identify which regions should fall,
- allow chunk-like release,
- keep global fragmentation conservative,
- but let structural cut zones become stronger.

### Concrete should move toward
- rough split,
- support loss,
- chunk release,
- progressive collapse.

This is more important than more line density.

---

## 18. Glass-Specific Guidance

Glass is already close to a split-success family.

### Current glass status
- late split exists,
- but still feels weak and short-lived.

### Correct next move for glass
- strengthen authoritative cut promotion,
- strengthen split persistence,
- strengthen clean release,
- keep path count low,
- make the event read as:
  - thin cut
  - clean detach
  - clear split gap

Glass should not become rougher.
It should become more decisive.

---

## 19. Ceramic-Specific Guidance

Ceramic is the most ambiguous family.

### Current ceramic status
- crack/opening are clearly present,
- but split is not yet mandatory.

### Recommended interpretation
Ceramic can currently be approved as:
- crack-only
- brittle but non-detached
- intermediate family

This is not a failure.

### Only if later necessary
Upgrade ceramic to delayed-split family using:
- local signed cut-plane
- side-aware cut logic

But do not force this too early.

---

## 20. Rubber-Specific Guidance

Rubber is already behaving correctly.

### Correct action
- keep explicit crack and split disabled
- maintain diffuse damage interpretation
- regression-check after major fragment-body updates

Do not try to align rubber with the split-success families.

---

## 21. Recommended Overnight Run Plan

The recommended overnight run plan is now:

### Group A — Authoritative cut / support-loss baseline
- concrete / 80
- glass / 80
- ceramic / 80
- rubber / 80

Goal:
- validate current structure before new support-loss modifications

### Group B — Support-loss promotion enabled
- concrete / 80 / support-loss on
- glass / 80 / support-loss on
- ceramic / 80 / support-loss on
- rubber / 80 / support-loss on

Goal:
- test whether concrete/glass read more structurally broken

### Group C — Long horizon delayed behavior
- glass / 120 / 160 / 200
- ceramic / 120 / 160 / 200
- concrete / 120 regression
- rubber / 120 regression

Goal:
- determine final ceramic status

### Group D — Shard stage (only if A/B/C succeed)
- glass / shard on
- concrete / shard on

Goal:
- move from detach to visible aftermath

---

## 22. Metrics to Log

### Structural
- `first_split_frame`
- `longest_split_run`
- `final_n_frags`
- `max_n_frags`
- `broken_edge_count`
- `cross_edge_break_count`

### Support-loss
- `support_lost_component_count`
- `support_loss_score_max`
- `promoted_fragment_count`
- `release_candidate_count`

### Detach / Collapse
- `max_detached_distance`
- `mean_detached_distance`
- `fall_distance_after_release`
- `collapse_duration`

### Render/Eventness
- split gap visibility
- detached body visibility
- fragment shell contrast
- visible shard count (if enabled)

### Interpretation
- family
- crack-only / split-success / delayed-split
- final verdict

---

## 23. Graphics Expert Tips

### Tip 1
At this stage, the most important metric is not crack count.
It is:
- how many cracks become authoritative cuts,
- and whether those cuts actually cause support loss.

### Tip 2
Do not globally relax fragment thresholds.
Only relax them near authoritative cut surfaces or support-loss zones.

### Tip 3
Concrete should feel like:
- rough fracture
- support loss
- chunk release
not just:
- many lines

### Tip 4
Glass should feel like:
- narrow cut
- decisive split
- clean detach
not:
- rough fragmentation

### Tip 5
Ceramic should be treated as a deliberate design choice.
Do not let it remain ambiguous by accident.

### Tip 6
Do not enable debris before support-loss separation is visually convincing.
Otherwise the system will regress into cosmetic fracture.

---

## 24. Final Recommended Priority Order

From a graphics expert perspective, the correct final priorities are:

1. **Add authoritative cut-surface promotion**
2. **Add support-aware fragment promotion**
3. **Strengthen falling / collapse readability**
4. **Run long-horizon validation**
5. **Decide final ceramic status**
6. **Only then enable shard/debris**
7. **Finish with render polish**

This is the right path to finish the system convincingly.

---

## 25. Final Conclusion

The project is now at a strong but incomplete graphics milestone.

It has already succeeded in:
- material-aware crack generation
- family separation
- graph-based fragment logic
- stable concrete split
- late glass split
- correct rubber no-split behavior

So the remaining work is no longer about making cracks exist.
It is about finishing the missing structural fracture layer.

The correct final implementation direction is therefore:

> **move from crack visualization to authoritative cut-surface formation, then to support-loss-driven fragment release, then to visible falling/collapse, and only after that to shards/debris and beauty polish.**

### One-sentence summary
From a graphics expert perspective, the right way to finish the project is to treat the remaining bottleneck as a structural fracture-topology problem—specifically cut-surface promotion plus support-loss separation—then validate long-horizon behavior, decide ceramic explicitly, and only afterward turn on shard/debris and final render polish.
ㅌ