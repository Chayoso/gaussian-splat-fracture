"""Sentence-conditioned Voronoi pre-fracture decomposition.

Tessellates a body's particle cloud into N Voronoi cells whose seed
distribution is conditioned on the input sentence's fracture style.
Each MPM particle is assigned to its nearest seed cell.  Cells are
connected by bonds (Delaunay-adjacent seed pairs); bonds break when
the AT2 damage averaged over their shared particle boundary exceeds a
threshold.  Connected components of unbroken bonds = fragments.

The visible result is a brittle-fracture pattern that follows the
mesh geometry rather than an axis-aligned grid: cells inherit the
local body curvature, bonds break along genuine crack surfaces, and
the resulting fragment cloud is anisotropic (no perfect-circle
shockwave from uniform partitioning).
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
from torch import Tensor

try:
    from scipy.spatial import Delaunay, cKDTree
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


SeedDistribution = str  # 'uniform' | 'impact_biased' | 'curvature_biased' | 'axis_aligned'


class VoronoiDecomposer:
    """Pre-fracture Voronoi cell decomposition with bond network.

    Parameters
    ----------
    n_cells:
        Target number of Voronoi cells.  Actual count may be smaller if
        rejection-sampling drops out-of-mesh seeds.
    distribution:
        Seed-distribution policy:
          'uniform'          : random in mesh bbox (accept if inside)
          'impact_biased'    : density ~ 1 / (eps + |p - impact_center|)
          'curvature_biased' : density ~ surface_curvature (highest at
                               edges/corners -> chunkier where mesh is
                               thicker, smaller pieces at thin features)
          'axis_aligned'     : seeds clustered along anisotropy_axis
                               (produces directional split pattern)
    bond_break_threshold:
        Average damage along the bond's shared-particle ring above which
        the bond is considered broken.
    impact_center, anisotropy_axis:
        Geometry hints used by impact_biased / axis_aligned.
    """

    def __init__(
        self,
        n_cells: int = 200,
        distribution: SeedDistribution = "impact_biased",
        bond_break_threshold: float = 0.40,
        impact_center: Optional[Tensor] = None,
        anisotropy_axis: Optional[Tensor] = None,
        seed: int = 1234,
        force_shrink_max_frac: float = 0.0,
        impact_bias_scale: float = 0.25,
    ):
        if not _SCIPY_OK:
            raise RuntimeError("scipy is required for VoronoiDecomposer")
        self.n_cells = int(max(2, n_cells))
        self.distribution = str(distribution)
        self.bond_break_threshold = float(bond_break_threshold)
        self.force_shrink_max_frac = float(force_shrink_max_frac)
        self.impact_bias_scale = float(impact_bias_scale)
        self.impact_center = impact_center
        self.anisotropy_axis = anisotropy_axis
        self.rng = np.random.default_rng(int(seed))

        # Tessellation state, populated by `tessellate`.
        self.cell_centers: Optional[np.ndarray] = None       # (n_cells, 3)
        self.cell_assignment: Optional[np.ndarray] = None    # (n_particles,)
        self.cell_adjacency: List[Tuple[int, int]] = []
        # Map (cell_a, cell_b) -> bond particles' indices (used for damage avg)
        self._bond_boundary_idx: Dict[Tuple[int, int], np.ndarray] = {}
        self.bond_broken: Dict[Tuple[int, int], bool] = {}
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    # ------------------------------------------------------------------
    # Seed sampling
    # ------------------------------------------------------------------
    def _sample_seeds(
        self,
        particle_pos_np: np.ndarray,
    ) -> np.ndarray:
        """Sample `n_cells` seed points using the configured distribution.

        We use rejection sampling against the particle cloud: a candidate
        is kept if at least one particle is within `accept_radius` of it.
        That guarantees every seed has particles assigned to it (no
        empty Voronoi cells) and the seed cloud follows the mesh shape.
        """
        bbox_min = particle_pos_np.min(axis=0)
        bbox_max = particle_pos_np.max(axis=0)
        bbox_size = (bbox_max - bbox_min).clip(min=1e-6)
        accept_radius = float(np.linalg.norm(bbox_size)) * 0.05

        kdt = cKDTree(particle_pos_np)
        seeds: List[np.ndarray] = []
        max_iters = self.n_cells * 25
        it = 0
        while len(seeds) < self.n_cells and it < max_iters:
            it += 1
            cand = self._propose_candidate(bbox_min, bbox_max, bbox_size)
            d, _ = kdt.query(cand, k=1)
            if float(d) > accept_radius:
                continue
            seeds.append(cand)
        if len(seeds) < min(2, self.n_cells):
            raise RuntimeError(
                f"VoronoiDecomposer: only {len(seeds)} seeds sampled "
                f"(target {self.n_cells}); check distribution params"
            )
        return np.stack(seeds, axis=0).astype(np.float32)

    def _propose_candidate(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        bbox_size: np.ndarray,
    ) -> np.ndarray:
        if self.distribution == "uniform":
            return bbox_min + self.rng.random(3) * bbox_size

        if self.distribution == "impact_biased":
            ic = self._impact_center_np(bbox_min, bbox_max)
            # Sample radius from exponential: more density near impact.
            # ``impact_bias_scale`` (default 0.25) controls how tightly
            # cells cluster around the impact point.  Larger values
            # (0.5–0.7) approach a uniform spread while preserving
            # the impact-centered density profile.
            scale_frac = float(getattr(self, "impact_bias_scale", 0.25))
            scale = float(np.linalg.norm(bbox_size)) * scale_frac
            r = self.rng.exponential(scale=scale)
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            phi = np.arccos(1.0 - 2.0 * self.rng.random())
            offset = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ], dtype=np.float64)
            return ic + offset

        if self.distribution == "axis_aligned":
            ax = self._anisotropy_axis_np()
            ic = self._impact_center_np(bbox_min, bbox_max)
            # Sample t along axis, then small jitter perpendicular.
            t_extent = float(np.linalg.norm(bbox_size)) * 0.6
            t = (self.rng.random() - 0.5) * t_extent
            perp = self.rng.normal(scale=t_extent * 0.10, size=3)
            perp -= ax * float(perp.dot(ax))
            return ic + ax * t + perp

        # 'curvature_biased' falls back to uniform here; the simulator
        # passes pre-computed seed bias via `seed_density_field` if
        # curvature-aware sampling is desired (TODO).
        return bbox_min + self.rng.random(3) * bbox_size

    def _impact_center_np(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
    ) -> np.ndarray:
        if self.impact_center is not None:
            ic = self.impact_center
            if isinstance(ic, Tensor):
                ic = ic.detach().cpu().numpy()
            return np.asarray(ic, dtype=np.float64)
        # Fallback: bottom-center of bbox.
        cx = 0.5 * (bbox_min[0] + bbox_max[0])
        cy = 0.5 * (bbox_min[1] + bbox_max[1])
        cz = bbox_min[2]
        return np.array([cx, cy, cz], dtype=np.float64)

    def _anisotropy_axis_np(self) -> np.ndarray:
        if self.anisotropy_axis is not None:
            ax = self.anisotropy_axis
            if isinstance(ax, Tensor):
                ax = ax.detach().cpu().numpy()
            ax = np.asarray(ax, dtype=np.float64)
            n = float(np.linalg.norm(ax))
            if n > 1e-8:
                return ax / n
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Tessellation
    # ------------------------------------------------------------------
    def tessellate(self, particles: Tensor) -> None:
        """Build cell assignment + bond network for the given particle cloud."""
        self._device = particles.device
        self._dtype = particles.dtype
        pos_np = particles.detach().cpu().numpy().astype(np.float32)

        seeds = self._sample_seeds(pos_np)
        self.cell_centers = seeds
        self.n_cells = int(seeds.shape[0])

        # Per-particle nearest-seed assignment (Voronoi cell id).
        kdt = cKDTree(seeds)
        _, assign = kdt.query(pos_np, k=1)
        self.cell_assignment = assign.astype(np.int64)

        # Adjacency from Delaunay tetrahedralization of seeds.  Each
        # simplex edge -> adjacent cell pair.
        try:
            tri = Delaunay(seeds)
            adj_set = set()
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        a, b = int(simplex[i]), int(simplex[j])
                        if a == b:
                            continue
                        adj_set.add((min(a, b), max(a, b)))
            self.cell_adjacency = sorted(adj_set)
        except Exception as exc:
            # Degenerate seeds (collinear).  Fallback: all-pair adjacency.
            self.cell_adjacency = [
                (i, j) for i in range(self.n_cells)
                for j in range(i + 1, self.n_cells)
            ]
            print(f"[Voronoi] Delaunay failed ({exc}); using all-pair adjacency")

        # Bond boundary indices: for each (a, b), the particles in cell a
        # whose nearest neighbor in cell b is within the bond-band radius
        # (or vice versa) form the bond's "shared boundary".  We use
        # 5th-percentile-distance as a per-bond threshold so even small
        # bonds capture the closest particles.
        self._bond_boundary_idx.clear()
        self.bond_broken.clear()
        bbox_max_extent = float(
            (pos_np.max(axis=0) - pos_np.min(axis=0)).max()
        )
        bond_band = max(bbox_max_extent * 0.04, 1e-3)
        # Pre-build per-cell particle-index lists for fast bond queries.
        cell_particle_idx: Dict[int, np.ndarray] = {}
        for c in range(self.n_cells):
            cell_particle_idx[c] = np.where(self.cell_assignment == c)[0]

        for (a, b) in self.cell_adjacency:
            idx_a = cell_particle_idx.get(a, np.empty(0, dtype=np.int64))
            idx_b = cell_particle_idx.get(b, np.empty(0, dtype=np.int64))
            if idx_a.size == 0 or idx_b.size == 0:
                # Empty cell -> bond meaningless.  Mark broken.
                self.bond_broken[(a, b)] = True
                self._bond_boundary_idx[(a, b)] = np.empty(0, dtype=np.int64)
                continue
            kdt_b = cKDTree(pos_np[idx_b])
            d_a, _ = kdt_b.query(pos_np[idx_a], k=1)
            close_in_a = idx_a[d_a < bond_band]
            kdt_a = cKDTree(pos_np[idx_a])
            d_b, _ = kdt_a.query(pos_np[idx_b], k=1)
            close_in_b = idx_b[d_b < bond_band]
            bond_idx = np.concatenate([close_in_a, close_in_b], axis=0)
            self._bond_boundary_idx[(a, b)] = bond_idx
            self.bond_broken[(a, b)] = bool(bond_idx.size == 0)

    # ------------------------------------------------------------------
    # Bond breakage + connected components
    # ------------------------------------------------------------------
    def update_bond_breakage(
        self,
        particle_damage: Tensor,
        bond_aging: float = 0.0,
        impact_center: Optional[Tensor] = None,
        impact_radius: float = 0.0,
        cascade_radius: float = 0.0,
        wave_speed_per_frame: float = 0.0,
    ) -> List[Tuple[int, int, float]]:
        """Mark bonds broken when avg damage on their boundary exceeds
        the threshold.  Three break criteria, OR'd together:

          1. damage-based: avg AT2 damage along the bond boundary >= threshold
          2. aging-based:  threshold decays by `bond_aging` per call;
             weak/marginal bonds eventually break with time.
          3. proximity-based: any bond whose midpoint is within
             `impact_radius` of `impact_center` breaks immediately
             (emulates impact shock front).

        Returns a list ``[(cell_a, cell_b, stress_at_break), ...]`` of
        bonds that just broke this call -- the caller uses this for
        Mode-I bond-opening kicks (paired Newton-3rd kicks on adjacent
        cells along the bond direction).
        """
        if self.cell_assignment is None:
            return []
        damage_np = particle_damage.detach().cpu().numpy().astype(np.float32)
        # Effective threshold decays over calls (bonds weaken with time).
        if bond_aging > 0.0:
            self.bond_break_threshold = max(
                0.01, float(self.bond_break_threshold) - float(bond_aging))
        thr = self.bond_break_threshold
        newly_broken_bonds: List[Tuple[int, int, float]] = []

        ic_np = None
        if impact_center is not None:
            if isinstance(impact_center, Tensor):
                ic_np = impact_center.detach().cpu().numpy().astype(np.float32)
            else:
                ic_np = np.asarray(impact_center, dtype=np.float32)

        # Stress-wave fracture propagation: at each call, grow the
        # `_wave_radius` from impact_center by `wave_speed_per_frame`.
        # ONLY bonds whose midpoint is within `_wave_radius` are allowed
        # to break this frame.  Bonds outside the wave stay intact even
        # if their damage/aging would otherwise let them break.  This
        # produces a fracture FRONT propagating outward from the impact
        # zone over time -- "shatter from the bottom up" -- rather than
        # uniform simultaneous breakage everywhere.
        if wave_speed_per_frame > 0.0:
            self._wave_radius = float(getattr(
                self, "_wave_radius", float(impact_radius))) + float(
                wave_speed_per_frame)
        else:
            # Wave gating disabled: any bond can break this frame.
            self._wave_radius = float("inf")

        for bond_key, bond_idx in self._bond_boundary_idx.items():
            if self.bond_broken.get(bond_key, False):
                continue
            # Stress-wave gate: bonds outside the propagating wave-front
            # cannot break this frame, regardless of damage/aging.
            in_wave = True
            if (ic_np is not None and self.cell_centers is not None
                    and self._wave_radius != float("inf")):
                a, b = bond_key
                mid = 0.5 * (self.cell_centers[a] + self.cell_centers[b])
                in_wave = (float(np.linalg.norm(mid - ic_np))
                           < float(self._wave_radius))
            if not in_wave:
                continue
            if bond_idx.size == 0:
                self.bond_broken[bond_key] = True
                newly_broken_bonds.append((bond_key[0], bond_key[1], 0.0))
                continue
            avg_dmg = float(damage_np[bond_idx].mean())
            if avg_dmg >= thr:
                self.bond_broken[bond_key] = True
                newly_broken_bonds.append((bond_key[0], bond_key[1], avg_dmg))
                continue
            if (ic_np is not None and impact_radius > 0.0
                    and self.cell_centers is not None):
                a, b = bond_key
                mid = 0.5 * (self.cell_centers[a] + self.cell_centers[b])
                if float(np.linalg.norm(mid - ic_np)) < impact_radius:
                    self.bond_broken[bond_key] = True
                    newly_broken_bonds.append((a, b, avg_dmg))

        # Force-shrink the largest component: if one connected component
        # of unbroken bonds owns more than `force_shrink_max_frac` of all
        # cells, snap the bonds incident to its boundary cells until it
        # falls below the cap.  Pulverization-style prompts need this
        # because purely damage-driven breakage tends to leave a single
        # large cohesive component (the body's deep interior, which
        # never receives strong AT2 damage).
        force_shrink_max_frac = float(getattr(
            self, "force_shrink_max_frac", 0.0))
        if force_shrink_max_frac > 0.0:
            # Compute current connected components.
            parent = list(range(self.n_cells))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry

            for (a, b), broken in self.bond_broken.items():
                if not broken:
                    union(a, b)
            comp = [find(i) for i in range(self.n_cells)]
            from collections import Counter
            counter = Counter(comp)
            if counter:
                largest_comp, largest_size = counter.most_common(1)[0]
                cap = int(force_shrink_max_frac * self.n_cells)
                # Iteratively break boundary bonds of the largest comp
                # until size <= cap or no more progress.
                if not hasattr(self, "_cell_incident_bonds"):
                    self._cell_incident_bonds = {
                        c: [] for c in range(self.n_cells)
                    }
                    for (a, b) in self.cell_adjacency:
                        self._cell_incident_bonds[a].append((a, b))
                        self._cell_incident_bonds[b].append((a, b))
                guard = 0
                while largest_size > cap and guard < self.n_cells:
                    guard += 1
                    # Pick a cell on the boundary of the largest comp
                    # (i.e. one with at least one incident broken bond,
                    # so it's already next to the "outside") and
                    # ISOLATE it by breaking ALL its remaining bonds.
                    # This is guaranteed to shrink the largest comp.
                    cells_in_largest = [
                        c for c in range(self.n_cells)
                        if find(c) == largest_comp
                    ]
                    target = None
                    for c in cells_in_largest:
                        incident = self._cell_incident_bonds.get(c, [])
                        any_broken = any(
                            self.bond_broken.get(nb, False) for nb in incident
                        )
                        any_unbroken = any(
                            not self.bond_broken.get(nb, False) for nb in incident
                        )
                        if any_broken and any_unbroken:
                            target = c
                            break
                    if target is None:
                        # No boundary cell -- pick any cell with an
                        # unbroken bond so we make progress anyway.
                        for c in cells_in_largest:
                            for nb in self._cell_incident_bonds.get(c, []):
                                if not self.bond_broken.get(nb, False):
                                    target = c
                                    break
                            if target is not None:
                                break
                    if target is None:
                        break
                    # Break all of target's incident bonds.
                    progress = False
                    for nb in self._cell_incident_bonds.get(target, []):
                        if not self.bond_broken.get(nb, False):
                            self.bond_broken[nb] = True
                            newly_broken_bonds.append((nb[0], nb[1], 0.0))
                            progress = True
                    if not progress:
                        break
                    # Recompute components after isolating one cell.
                    parent = list(range(self.n_cells))
                    for (a, b), broken in self.bond_broken.items():
                        if not broken:
                            union(a, b)
                    comp = [find(i) for i in range(self.n_cells)]
                    counter = Counter(comp)
                    if counter:
                        largest_comp, largest_size = counter.most_common(1)[0]

        # Cascade break: any bond adjacent (sharing a cell) to an
        # already-broken bond breaks as well, up to `cascade_radius`
        # graph hops.  Emulates how a brittle crack propagates: once a
        # surface is broken the neighboring surface is suddenly
        # unsupported and crack runs through it.  cascade_radius=1.0
        # means "one hop"; >=2.0 means "two hops"; etc.
        if cascade_radius > 0.0:
            # Build cell -> incident-bond map (cached after first build).
            if not hasattr(self, "_cell_incident_bonds"):
                self._cell_incident_bonds = {c: [] for c in range(self.n_cells)}
                for (a, b) in self.cell_adjacency:
                    self._cell_incident_bonds[a].append((a, b))
                    self._cell_incident_bonds[b].append((a, b))
            hops = int(max(1, round(cascade_radius)))
            frontier = {k for k, v in self.bond_broken.items() if v}
            for _ in range(hops):
                next_frontier: set = set()
                for (a, b) in frontier:
                    for cell in (a, b):
                        for nb in self._cell_incident_bonds.get(cell, []):
                            if not self.bond_broken.get(nb, False):
                                self.bond_broken[nb] = True
                                next_frontier.add(nb)
                                newly_broken_bonds.append((nb[0], nb[1], 0.0))
                if not next_frontier:
                    break
                frontier = next_frontier
        return newly_broken_bonds

    def connected_components(self) -> np.ndarray:
        """Union-find over unbroken bonds.  Returns per-cell component id."""
        if self.cell_assignment is None:
            return np.zeros(0, dtype=np.int64)
        parent = list(range(self.n_cells))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for (a, b), broken in self.bond_broken.items():
            if not broken:
                union(a, b)
        comp = np.array([find(i) for i in range(self.n_cells)],
                        dtype=np.int64)
        # Re-label compactly (0..K-1).
        unique = sorted(set(comp.tolist()))
        relabel = {old: new for new, old in enumerate(unique)}
        return np.array([relabel[c] for c in comp], dtype=np.int64)

    def particle_fragment_ids(self) -> np.ndarray:
        """Map each particle to its current connected-component id.

        Component 0 is reserved as "still-cohesive base body" (the
        largest component); all others are individual fragments
        (component id >= 1 after re-labeling).
        """
        comp_per_cell = self.connected_components()
        if self.cell_assignment is None:
            return np.zeros(0, dtype=np.int64)
        per_particle = comp_per_cell[self.cell_assignment]
        # Identify largest component (= base) and remap to label 0;
        # everything else gets sequential labels 1..K.
        if per_particle.size > 0:
            counts = np.bincount(per_particle)
            largest = int(np.argmax(counts))
            # Only relabel the largest component as "base" (id 0) if it
            # genuinely owns a majority of particles; otherwise the body
            # has already pulverized into many small chunks and "base"
            # is a misleading concept (we'd be calling whichever chunk
            # happens to have the most particles "base", artificially
            # inflating base_frac).  Threshold: 0.30 of total particles.
            n_total = int(per_particle.size)
            if int(counts[largest]) > int(0.30 * n_total):
                mask_base = per_particle == largest
                unique_non_base = sorted(
                    set(per_particle[~mask_base].tolist()))
                remap = {largest: 0}
                for new_id, old_id in enumerate(unique_non_base, start=1):
                    remap[old_id] = new_id
            else:
                # No dominant base; sequential labels 1..K, no zero.
                unique_all = sorted(set(per_particle.tolist()))
                remap = {old_id: new_id
                         for new_id, old_id in enumerate(unique_all, start=1)}
            per_particle = np.array(
                [remap[c] for c in per_particle.tolist()],
                dtype=np.int64,
            )
        return per_particle

    # ------------------------------------------------------------------
    # Helpers for per-cell kick computation
    # ------------------------------------------------------------------
    def cell_pca_thin_axis(
        self,
        cell_id: int,
        particles: Tensor,
    ) -> Optional[Tensor]:
        """PCA thin axis (smallest-eigvec) of a cell's particles.  Used
        as the natural fracture-surface normal for kick direction."""
        if self.cell_assignment is None:
            return None
        idx = np.where(self.cell_assignment == cell_id)[0]
        if idx.size < 4:
            return None
        pos = particles[idx]
        centered = pos - pos.mean(dim=0, keepdim=True)
        cov = centered.t() @ centered / max(idx.size, 1)
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
        except RuntimeError:
            return None
        return eigvecs[:, 0].clone()
