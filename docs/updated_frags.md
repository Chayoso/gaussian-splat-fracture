# Updated Graphics-Oriented Implementation Specification
## Current Status Review + Next Step for Sharp-Brittle Cut-Surface Logic
### CLIP Semantic Prior -> Fracture Family -> Crack Growth -> Fragment Separation -> Event Quality -> Render Quality

---

## 1. Purpose of This Update

This document updates the previous implementation roadmap based on the **current actual system status**.

The previous roadmap argued that the next correct development order should be:

```text
fragment separation -> fragment validation -> visible detach validation -> family-specific fracture event quality -> family-specific render quality
```

At this point, the system has advanced enough that the first major checkpoint has been partially passed.

This update therefore focuses on:

1. what has already been successfully validated,
2. what is still incomplete,
3. why the current bottleneck is now specifically a **sharp-brittle cut problem**,
4. and what the next implementation step should be from a **graphics expert perspective**.

---

## 2. Current Graphics-Level Assessment

From a graphics perspective, the overall state of the system is now clearly better than before.

### What is already good
- The system is no longer trapped in the old bottom-stain failure mode.
- Crack generation and propagation are working.
- Material-family modes are separating:
  - sharp path,
  - rough network,
  - diffuse response.
- Fragment logic is no longer merely theoretical in code; it has produced visible detach at least in one family.

So the current evaluation is:

> **The fracture structure is now moving in the correct direction.**

That is a meaningful success.

However, the system is still in an intermediate state:

> **true fragment separation is validated for some families, but not yet for all materially important families.**

In particular, sharp-brittle behavior is still incomplete.

---

## 3. What Has Been Successfully Implemented

The following Stage A fragment logic has already been implemented:

- persistent broken-edge memory,
- detached-node hysteresis,
- family-aware fragment presets,
- visited-front-aware fragment detection,
- diffuse-family fallback,
- split stability metrics,
- split-driven impulse / visual detach handling.

The relevant implementation has been reflected in:
- `graph_fragment_manager.py`
- `material_prior_adapter.py`
- `manifold_simulator.py`
- `smoke_test.py`

This is important because it means the fragment pipeline is no longer hypothetical.

It exists and works in practice for at least part of the family space.

---

## 4. Current Verified Results

### 4.1 Concrete bunny sculpture
**Status:** Stage A passed

Observed:
- `first_split_frame = 40`
- `longest_split_run = 40`
- `final_n_frags = 4`
- `max_n_frags = 4`

Interpretation:
- fragment separation is not merely transient,
- the split remains stable,
- detach remains visible,
- and graph disconnection is sustained.

Graphics interpretation:
> concrete currently proves that the Stage A fragment architecture is structurally valid.

This is a major milestone.

---

### 4.2 Rubber bunny toy
**Status:** behaves correctly

Observed:
- `split_frame_count = 0`
- `cracked(c > 0.3) = 0`
- no explicit crack / fragment event

Interpretation:
- the diffuse-family fallback works,
- rubber is not incorrectly entering brittle-fracture behavior.

Graphics interpretation:
> the no-split diffuse response for rubber is working as intended.

This is also an important success because it confirms that family separation is meaningful.

---

### 4.3 Soda-lime glass bunny statue
**Status:** still incomplete

Observed:
- multiple corrected runs,
- clean crack path appears,
- but detach still does not occur within 80 frames.

Interpretation:
- the visible narrow crack path is present,
- but that path still does not act strongly enough as a graph-disconnecting surface.

Graphics interpretation:
> the glass family currently succeeds as a visible crack line, but not yet as a true cut-surface fracture event.

This is now the main bottleneck.

---

## 5. What This Means Structurally

This is a very important turning point.

At this stage, the current diagnosis is no longer:

- fragment pipeline missing,
- fragment labeling not implemented,
- or fragment stability not working.

Instead, the new diagnosis is:

> **the fragment detector is no longer the main bottleneck; the bottleneck is that sharp-brittle narrow cracks do not yet function strongly enough as cut surfaces on the graph.**

This changes the implementation priority significantly.

The current problem is not:
- “make split thresholds more sensitive”
in a generic way.

The current problem is:
- **how to translate a narrow sharp crack core into an actual connectivity-cutting event.**

That is a much more specific and graphics-relevant problem.

---

## 6. Updated Development Logic

The previous roadmap said:

```text
Step A. Fragment computation
Step B. Fragment validation
Step C. Visible detach validation
Step D. Family-aware fracture event quality
Step E. Family-aware render quality
```

Based on the current verified state, that should now be updated to:

```text
Step A. Fragment computation              [concrete/rubber validated]
Step A.5 Sharp-brittle cut-surface logic [new immediate priority]
Step B. Family-specific split stabilization
Step C. Family-specific fracture event quality
Step D. Family-specific render quality
```

This is the correct updated ordering.

---

## 7. Why a New Step A.5 Is Needed

A new intermediate step is needed because the system has passed Stage A for:
- `rough_quasi_brittle`
- and `diffuse_damage`

but not yet for:
- `sharp_brittle`
- and not yet revalidated for `brittle_moderate`.

So now the question is not:

> can the fragment system work at all?

That has already been answered by concrete.

The new question is:

> can a **thin, narrow, visually sharp brittle crack** act as a real graph-cutting surface rather than just a visible trajectory?

This is exactly what Step A.5 should solve.

---

## 8. Graphics Expert Diagnosis of the Glass Failure

Why does concrete split but glass not split?

### Concrete
Concrete has:
- broader crack band,
- rougher fractured region,
- wider local weakening,
- more graph-edge exposure.

That means standard edge weakening is often enough to disconnect the graph.

### Glass
Glass has:
- narrow crack core,
- fewer paths,
- thinner band,
- more line-like fracture structure.

That means ordinary broad edge weakening often leaves the graph still connected.

So glass currently behaves like:
- **a visible fracture line**
rather than
- **a true connectivity-cutting surface**

This is the key issue.

From a graphics perspective, that is perfectly plausible:
the crack *looks* correct, but the topology still behaves as if the object is not truly cut.

That is why an explicit cut-surface mechanism is now needed.

---

## 9. The Correct Next Step: Sharp-Brittle Cut-Surface Logic

The next implementation should introduce **family-aware cut-surface logic** for:

- `sharp_brittle`
- `brittle_moderate`

The key principle is:

> **Do not rely only on diffuse edge weakening for narrow brittle cracks.**
> Instead, explicitly detect and cut the edges that cross the crack core.

This is the correct next move.

---

## 10. What Cut-Surface Logic Should Mean

The system already has:
- crack path,
- crack core,
- opening,
- edge memory,
- fragment detection.

What is missing is:

> **explicit cross-edge cutting around the crack core**

In other words:
- not all edges near the crack should weaken equally,
- the edges that *cross* the crack should receive a strong cut vote,
- while edges that run *along* the crack should not be treated the same way.

This is the core implementation insight.

---

## 11. Updated Implementation Plan

Below is the recommended next implementation sequence.

### Step A.5.1 — Define a true cut-core mask

#### Goal
Extract only the part of the crack that should act as a cutting surface.

#### Why
Using the full visited trail is too broad and includes weak historical crack traces.

#### Recommended definition
A node belongs to the cut-core if it satisfies:
- family is `sharp_brittle` or `brittle_moderate`
- damage is above a strong core threshold
- opening is above an opening threshold
- node is recent-tip or recent-visited

This should create a much narrower, more event-centered cut source.

#### Expected effect
Instead of using the full path history, the system only uses:
- active brittle crack core,
- strong local opening region,
- and recent fracture event zones.

That is much more correct.

---

### Step A.5.2 — Compute local cut direction

#### Goal
For each cut-core node, determine which direction corresponds to “cutting across” the crack.

#### Why
To disconnect the graph properly, we must cut the edges that cross the crack, not the edges that continue along the crack path.

#### Important distinction
Do not confuse:
- crack growth direction
with
- cut direction / split direction.

The growth direction tells where the crack advances.
The cut direction tells which local cross-edges should be removed.

#### Expected effect
This provides a local notion of:
- along-crack direction
- across-crack direction

which is essential for selective edge breaking.

---

### Step A.5.3 — Detect cross-edges explicitly

#### Goal
Score graph edges based on whether they cross the crack core.

#### Why
Current logic weakens edges broadly.
Glass needs edge cutting that is directional and selective.

#### Recommended concept
For each cut-core node:
- inspect neighboring edges,
- measure alignment with cut direction,
- reject edges aligned with crack tangent,
- apply a strong cut vote to cross-edges.

#### Expected effect
The graph begins to see the glass crack not just as a line, but as a real dividing surface.

---

### Step A.5.4 — Add family-aware cut votes

#### Goal
Add explicit integrity reduction or hard break votes for cross-edges.

#### Why
For glass/ceramic, weakening alone is often not enough.
A narrow brittle crack needs a more discrete topological effect.

#### Family-specific behavior
##### `sharp_brittle`
- strong cut vote
- easier edge break
- more decisive split

##### `brittle_moderate`
- weaker cut vote
- delayed or partial split
- intermediate behavior

##### `rough_quasi_brittle`
- mostly keep current edge weakening logic

##### `diffuse_damage`
- no cut-surface logic

#### Expected effect
Glass and ceramic finally become topologically capable of producing a split from a narrow crack.

---

### Step A.5.5 — Add split-event emphasis right after cut

#### Goal
Make newly separated fragments visibly detach right after the cut event.

#### Why
Even if graph splitting succeeds, the result may still look visually glued without enough local separation emphasis.

#### What to strengthen temporarily
- local geometry offset
- local opening boost
- fragment velocity divergence
- shell separation visibility

#### Important note
This can be temporary and debug-friendly.
At this stage, visual confirmation matters more than perfect subtlety.

#### Expected effect
The moment of detachment becomes visible as an actual event.

---

## 12. Files to Update Next

### `graph_fragment_manager.py`
This becomes the primary file for Step A.5.

Add:
- cut-core mask logic
- local cut direction
- cross-edge scoring
- family-aware cut vote application

This is the most important next file.

---

### `material_prior_adapter.py`
Add new family-specific cut priors such as:
- `cut_surface_enable`
- `cut_vote_strength`
- `tau_cross`
- `tau_tangent`
- `split_event_boost`

This file should now decide not only fracture style, but also graph-cut mode.

---

### `manifold_simulator.py`
Add:
- split-event timer
- temporary detach boost window
- family-aware post-split motion emphasis

This file should make successful split events visible in time.

---

### `smoke_test.py`
Expand split validation to include:
- cut-edge count
- cross-edge break count
- first split frame under cut logic
- detached distance
- split event duration

This will let the system distinguish:
- visible crack line only
from
- true cut-surface split.

---

## 13. Validation Plan for Step A.5

### Experiment A.5.1 — Glass cut activation test
Target:
- soda-lime glass bunny statue

Goal:
- `n_frags > 1` within 80 frames
- visible detach at least once
- longest split run > 1

Save:
- beauty frame
- broken-edge overlay
- cross-edge overlay
- fragment label render
- opening map

Success means:
- glass is no longer only a visible path
- it becomes a graph-cutting fracture event

---

### Experiment A.5.2 — Ceramic moderated cut test
Target:
- ceramic bunny

Goal:
- split should be possible
- but weaker / later than glass
- should preserve intermediate brittle character

Success means:
- ceramic no longer collapses into “glass without opening”
- but still remains distinct from concrete.

---

### Experiment A.5.3 — Regression test
Targets:
- concrete bunny
- rubber bunny

Goal:
- concrete Stage A behavior must remain stable
- rubber must remain no-split

This is essential because Step A.5 is meant to extend family coverage, not to break existing correctness.

---

## 14. Completion Criteria for the Updated Stage

The next stage may be considered complete if:

1. glass reaches `n_frags > 1`
2. glass shows visible detach in beauty/debug frames
3. ceramic shows weaker but plausible brittle split behavior
4. concrete still passes Stage A
5. rubber still remains no-split

Only after these five conditions are satisfied should the system move on to:
- family-specific fracture event quality polishing
- and final beauty render refinement.

---

## 15. What Comes After Step A.5

Once sharp-brittle cut logic works, then the next stage becomes meaningful:

### glass
- sharper / more open / cleaner split

### ceramic
- arrest / restart + partial split

### concrete
- more patchy / rough local fracture texture

### render
- family-aware beauty polish

But **not before** Step A.5 succeeds.

---

## 16. Graphics Expert Tips

### Tip 1
At this stage, glass needs **eventness**, not more path count.

The key question is:
- does the crack actually cut the object?

not:
- does the object have more crack lines?

### Tip 2
Cut only cross-edges.
Do not broadly weaken tangent-aligned path edges.
Otherwise the object fragments incorrectly along the crack line itself.

### Tip 3
It is okay to over-cut in the first debugging pass.
Once split appears, scale the cut logic back to a cleaner final form.

### Tip 4
Ceramic should not just be weaker glass.
It should be:
- same mechanism,
- lower confidence,
- more interruption,
- more moderate split.

### Tip 5
Always compare debug split and beauty split.
Graph disconnection without visible detach is still incomplete from a graphics point of view.

---

## 17. Updated Graphics-Oriented Conclusion

The current system has now reached an important milestone:
- Stage A has been validated for concrete,
- diffuse no-split has been validated for rubber,
- and the fragment architecture itself has been proven to work.

Therefore, the next graphics-relevant bottleneck is no longer generic fragment logic.
It is specifically:

> **how to make sharp-brittle narrow cracks act as true cut surfaces on the graph.**

So the correct next step is to add:
- cut-core extraction,
- directional cross-edge cut detection,
- family-aware cut votes,
- and split-event emphasis for glass and ceramic.

Only after that should the system move to:
- event-quality refinement,
- and final beauty-render polishing.
