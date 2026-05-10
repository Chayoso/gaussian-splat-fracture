"""Voronoi pre-fracture pipeline mixin for ``ManifoldSimulator``.

This module hosts the impact-time tessellation, per-frame stress-wave
bond breakage, and Mode-I bond opening kick — the three methods that
together form the v12 architecture's brittle-fracture substrate.

Design notes:
    * The pipeline lives on top of a *shared* global MPM grid.  Cells
      are particle-label sets, not separate grids.
    * One impact-energy factor ``k_e`` (computed in
      ``_init_voronoi_tessellation``) drives every scale knob in the
      cascade: cell count (``k_e^3``), bond threshold
      (``sqrt(1-k_e)``), wave / shock radius (``k_e^2``), and the
      Mode-I kick magnitude (``k_e^2``).  This is the "single energy
      budget" thesis of the paper.
    * Mode-I opening velocity is bounded by Griffith's critical
      energy release rate: ``v_open = sqrt(G_c * A_bond / m_cell)``,
      so the released KE never exceeds physical fracture-surface
      energy.

The mixin reads / writes the following attributes on ``self``:

    Read:
        ``self.x_mpm``, ``self.v_mpm``, ``self.fracture_cfg``,
        ``self.mpm``, ``self.elasticity``, ``self._impact_center``,
        ``self._soft_impact_speed``, ``self._voronoi_anisotropy_axis``,
        ``self.fragment_release_v_com_gain``,
        ``self.fragment_release_min_open_kick``,
        ``self.fragment_release_energy_fraction``,
        ``self.fragment_physical_max_speed``,
        ``self.fragment_release_position_offset``.
        ``self._get_volumetric_damage`` (method from
        ``GaussianPhysicsProjectionMixin``).

    Write:
        ``self.voronoi`` (``VoronoiDecomposer`` instance),
        ``self._voronoi_*`` (cached scaling factors and per-frame
        component state),
        ``self._physical_fragment_labels``,
        ``self._next_physical_fragment_id``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class VoronoiPipelineMixin:
    """Voronoi cell + bond-network fracture substrate (v12 architecture).

    Provides three methods on ``ManifoldSimulator``:

        ``_init_voronoi_tessellation`` — at impact time, seed cells
            and build the bond graph; cache impact-KE-derived scale
            factors for the rest of the cascade.
        ``_step_voronoi_fracture`` — per-frame: damage → wave-gated
            bond breakage → Mode-I kicks → component re-label.
        ``_apply_bond_opening_kick`` — Newton-3rd, Griffith-bounded
            opening kick at a single broken bond.

    See the module docstring for the full read / write attribute set.
    """

    # ------------------------------------------------------------------
    # Impact-time tessellation
    # ------------------------------------------------------------------

    def _init_voronoi_tessellation(self):
        """Tessellate the body into Voronoi cells at the impact moment.

        Voronoi parameters are scaled by impact KE so a slow drop
        produces partial fracture (few large chunks, base remnant) and
        a fast drop produces full pulverization (many small chunks, no
        base).  See module docstring for the exponent derivation.
        """
        from src.fracture.voronoi_decomposer import VoronoiDecomposer
        n_cells_full = int(self.fracture_cfg.get('voronoi_n_cells', 200))
        # n_cells < 5 means a "single-body" style (diffuse_microcrack
        # uses n_cells=2 to express "no fragmentation, just surface
        # microcracks").  Qhull Delaunay needs >=5 input points; below
        # that we skip the voronoi pipeline entirely and let the
        # particle stay on the base body label (renderer shows no
        # detached fragments — exactly what the style wants).
        if n_cells_full < 5:
            print(f"  [Voronoi] n_cells={n_cells_full} < 5 — single-body mode, skipping tessellation.")
            self.voronoi = None
            return
        distribution = str(self.fracture_cfg.get(
            'voronoi_seed_distribution', 'impact_biased'))
        bond_thr = float(self.fracture_cfg.get(
            'voronoi_bond_break_threshold', 0.40))
        impact_center = getattr(self, '_impact_center', None)
        anisotropy_axis = getattr(self, '_voronoi_anisotropy_axis', None)
        seed = int(self.fracture_cfg.get('voronoi_seed', 1234))
        force_shrink_full = float(self.fracture_cfg.get(
            'voronoi_force_shrink_max_frac', 0.0))

        # Impact-energy scaling: factor in [min_factor, 1.0].
        # ref_speed=70 maps z=0.8+ drops to ke=1.0 (full pulverization);
        # smaller drops fall on a steep cubic curve so z=0.22 (ke~0.26)
        # produces only a handful of cells.
        ref_speed = float(self.fracture_cfg.get(
            'voronoi_impact_speed_ref', 70.0))
        min_factor = float(self.fracture_cfg.get(
            'voronoi_impact_min_factor', 0.01))
        impact_speed = float(getattr(self, '_soft_impact_speed', ref_speed))
        ke_factor = max(min_factor, min(1.0, impact_speed / max(ref_speed, 1e-3)))
        # Cell-count scaling.  The physical fracture-area argument gives a
        # cubic relation (A ∝ KE ∝ v^2 and A ∝ n^(2/3) -> n ∝ v^3).  The
        # previous v^6 falloff collapsed glass-tier 15K impacts
        # (v≈42, ref=70) to two cells, so sentence styles could not express
        # radial/chunky/pulverized topology.  Keep the cubic default and
        # expose the exponent for future sweeps.
        cell_exp = float(self.fracture_cfg.get(
            'voronoi_impact_cell_exponent', 3.0))
        cell_exp = max(1.0, min(cell_exp, 6.0))
        n_cells = max(2, int(n_cells_full * (ke_factor ** cell_exp)))

        # force_shrink scales inversely with impact energy, but gently.
        # The old +2*ke_inv term saturated to 0.95 for ordinary glass drops,
        # leaving a single giant base component even when many bonds broke.
        ke_inv = max(0.0, 1.0 - ke_factor)
        force_shrink = force_shrink_full + ke_inv * 0.40 * max(
            1.0 - force_shrink_full, 0.0)
        force_shrink = min(0.95, max(0.0, force_shrink))

        # ke^2 family: kick magnitude, wave speed, shock radius, bond
        # aging — soft drops don't propagate the fracture front
        # through the whole body.
        self._voronoi_kick_scale = float(ke_factor) ** 2.0
        wave_speed_full = float(self.fracture_cfg.get(
            'voronoi_wave_speed_per_frame', 0.0))
        shock_radius_full = float(self.fracture_cfg.get(
            'voronoi_impact_shock_radius', 0.0))
        bond_aging_full = float(self.fracture_cfg.get(
            'voronoi_bond_aging_per_frame', 0.0))
        ke2 = float(ke_factor) ** 2.0
        self._voronoi_wave_speed_effective = wave_speed_full * ke2
        self._voronoi_shock_radius_effective = shock_radius_full * ke2
        self._voronoi_bond_aging_effective = bond_aging_full * ke2

        # bond_break_threshold scales INVERSELY with ke; sqrt(1-ke)
        # gate makes ke=0.05 effectively unbreakable (thr ≈ 0.98).
        bond_thr_full = float(self.fracture_cfg.get(
            'voronoi_bond_break_threshold', 0.15))
        self._voronoi_bond_thr_effective = (
            bond_thr_full
            + ((1.0 - ke_factor) ** 0.5) * (1.0 - bond_thr_full))

        # Cached effective values for unified-impact scaling and
        # position-offset (consumed by _apply_bond_opening_kick).
        self._voronoi_unified_impulse_scale_eff = (
            float(self.fracture_cfg.get(
                'unified_impact_impulse_scale', 0.05)) * ke2)
        self._voronoi_unified_tumble_scale_eff = (
            float(self.fracture_cfg.get(
                'unified_impact_tumble_scale', 0.20)) * ke2)
        self._voronoi_position_offset_eff = (
            float(self.fracture_cfg.get(
                'fragment_release_position_offset', 0.05)) * ke2)

        print(f"  [Voronoi] impact_speed={impact_speed:.2f} ke_factor={ke_factor:.2f} "
              f"n_cells={n_cells} (full={n_cells_full}) "
              f"force_shrink={force_shrink:.2f} (full={force_shrink_full:.2f}) "
              f"wave={self._voronoi_wave_speed_effective:.4f} "
              f"shock={self._voronoi_shock_radius_effective:.3f}")
        impact_bias_scale = float(self.fracture_cfg.get(
            'voronoi_impact_bias_scale', 0.25))
        self.voronoi = VoronoiDecomposer(
            n_cells=n_cells,
            distribution=distribution,
            bond_break_threshold=self._voronoi_bond_thr_effective,
            impact_center=impact_center,
            anisotropy_axis=anisotropy_axis,
            seed=seed,
            force_shrink_max_frac=force_shrink,
            impact_bias_scale=impact_bias_scale,
        )
        self.voronoi.tessellate(self.x_mpm)
        self._voronoi_cell_graduated = [False] * self.voronoi.n_cells
        self._voronoi_prev_components = None
        self._spatial_split_prev_labels = None
        self._spatial_split_next_id = 1
        print(f"  [Voronoi] Tessellated into {self.voronoi.n_cells} cells "
              f"(distribution={distribution}, n_bonds={len(self.voronoi.cell_adjacency)})")

    # ------------------------------------------------------------------
    # Per-frame stress-wave bond breakage + component re-label
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _step_voronoi_fracture(self):
        """Wrapper that runs both halves of the voronoi update in order.

        Pipeline rewrite (2026-05-09): the body has been factored into
        :meth:`_voronoi_bond_eval_commit` (cheap; bond-breakage + Mode-I
        kicks + cell-level connected components) and
        :meth:`_voronoi_spatial_split_and_refine` (expensive;
        particle-level spatial split + label write-back).  Step 3a
        moves the cheap half into the per-substep fracture tick while
        keeping the expensive half at frame end.  Until that flag is
        enabled, this wrapper preserves the legacy single-call
        behavior.
        """
        if self.voronoi is None or self.x_mpm is None:
            return
        components_state = self._voronoi_bond_eval_commit()
        if components_state is None:
            return
        self._voronoi_spatial_split_and_refine(components_state)

    @torch.no_grad()
    def _voronoi_bond_eval_commit(self):
        """Cheap half: bond-breakage check, Mode-I kicks, cell-level CC.

        Returns the cell-level connected component state needed by
        :meth:`_voronoi_spatial_split_and_refine`, or ``None`` if the
        precondition is not met (no voronoi, no particle field).
        """
        if self.voronoi is None or self.x_mpm is None:
            return None
        if bool(self.fracture_cfg.get('voronoi_use_topology_damage', True)):
            damage = self._get_voronoi_topology_damage()
        else:
            damage = self._get_volumetric_damage()
        if damage.numel() != self.x_mpm.shape[0]:
            return None
        bond_aging = float(getattr(
            self, '_voronoi_bond_aging_effective',
            self.fracture_cfg.get('voronoi_bond_aging_per_frame', 0.0)))
        impact_radius = float(getattr(
            self, '_voronoi_shock_radius_effective',
            self.fracture_cfg.get('voronoi_impact_shock_radius', 0.0)))
        cascade_radius = float(self.fracture_cfg.get(
            'voronoi_cascade_radius', 0.0))
        wave_speed = float(getattr(
            self, '_voronoi_wave_speed_effective',
            self.fracture_cfg.get('voronoi_wave_speed_per_frame', 0.0)))
        newly_broken = self.voronoi.update_bond_breakage(
            damage,
            bond_aging=bond_aging,
            impact_center=getattr(self, '_impact_center', None),
            impact_radius=impact_radius,
            cascade_radius=cascade_radius,
            wave_speed_per_frame=wave_speed,
            impact_damage_floor=float(self.fracture_cfg.get(
                'voronoi_impact_damage_floor', 0.0)),
            cascade_damage_floor=float(self.fracture_cfg.get(
                'voronoi_cascade_damage_floor', 0.0)),
            cascade_from_new_bonds_only=bool(self.fracture_cfg.get(
                'voronoi_cascade_from_new_bonds_only', True)),
            # Default damage floor for force-shrink: 0.30 (was 0.0).
            # Matches the AT2 damage_threshold the splat renderer uses
            # to draw "visible damage" (gaussian_updater._apply_damage_
            # visualization).  Without this gate force_shrink iteratively
            # breaks bonds in the largest CC down to
            # `voronoi_force_shrink_max_frac` * n_cells regardless of
            # phase-field damage, which fragments still-undamaged
            # regions and visually contradicts the c_visual map (the
            # renderer paints them as undamaged but the labels say
            # detached fragment).  Threshold 0.30 keeps physical and
            # visual narratives aligned.
            force_shrink_damage_floor=float(self.fracture_cfg.get(
                'voronoi_force_shrink_damage_floor', 0.30)),
        )
        # Mode-I bond opening.  Accumulate per-cell impulses first, then
        # write them once with a cap; otherwise one cell that loses several
        # bonds in the same fracture tick receives the same rigid-body kick
        # repeatedly and the result reads as an explosion rather than a crack.
        self._apply_bond_opening_kicks(newly_broken)
        comp_per_cell = self.voronoi.connected_components()
        per_particle = self.voronoi.particle_fragment_ids()
        # Telemetry: count cells whose component flipped this frame.
        prev = self._voronoi_prev_components
        n_grad = 0
        for cell_id in range(self.voronoi.n_cells):
            cur_comp = int(comp_per_cell[cell_id]) if cell_id < len(comp_per_cell) else -1
            prev_comp = int(prev[cell_id]) if (
                prev is not None and cell_id < len(prev)) else -1
            if cur_comp != prev_comp and not self._voronoi_cell_graduated[cell_id]:
                self._voronoi_cell_graduated[cell_id] = True
                n_grad += 1
        self._voronoi_prev_components = comp_per_cell.tolist()
        return {"per_particle": per_particle, "n_graduated": n_grad}

    @torch.no_grad()
    def _voronoi_spatial_split_and_refine(self, components_state: dict) -> None:
        """Expensive half: particle-level spatial split + label write-back.

        Refines cell-based fragment ids with a particle-level
        connected-component pass in *current* world space so chunks
        that drift apart (after the cell partition was fixed) get
        their own fragment id, then writes the result back to
        ``_physical_fragment_labels`` and updates the persistent
        ``_next_physical_fragment_id`` counter.

        Stays at frame end (Step 3a) because the cKDTree spatial split
        is the dominant cost in this pipeline.
        """
        per_particle = components_state["per_particle"]
        per_particle = self._spatial_split_fragments(per_particle)

        labels = torch.from_numpy(per_particle).to(
            self.x_mpm.device, dtype=torch.long)
        if labels.shape[0] == self.x_mpm.shape[0]:
            self._physical_fragment_labels = labels
            self._next_physical_fragment_id = int(labels.max().item()) + 1
            has_physical_fragment = bool((labels > 0).any())
            if (has_physical_fragment
                    and not bool(getattr(self, "fragmentation_active", False))):
                self.fragmentation_active = True
                self._fragment_activation_frame = self.frame_count
                # The birth registry also synchronizes rigid handoff state
                # and may rewrite tiny/no-base residue labels in place.
                # Surface/render labels must be written after that sync.
                self._update_physical_fragment_birth_registry()
                labels = self._physical_fragment_labels
                if (getattr(self, "_surface_indices", None) is not None
                        and getattr(self, "fracture_field", None) is not None
                        and self.fracture_field.c is not None
                        and labels is not None):
                    surf_labels = labels[self._surface_indices]
                    n_field = int(self.fracture_field.c.shape[0])
                    if surf_labels.shape[0] != n_field:
                        f_phys = torch.zeros(
                            n_field,
                            dtype=torch.long,
                            device=self.fracture_field.c.device,
                        )
                        n_assign = min(n_field, int(surf_labels.shape[0]))
                        if n_assign > 0:
                            f_phys[:n_assign] = surf_labels[:n_assign]
                        surf_labels = f_phys
                    self.fracture_field.f = surf_labels.to(
                        device=self.fracture_field.c.device,
                        dtype=torch.long,
                    ).clone()
            else:
                # Pipeline rewrite Step 2: keep the registry warm for
                # diagnostics even when physical authority is disabled.
                self._update_physical_fragment_birth_registry()

        n_grad = int(components_state.get("n_graduated", 0))
        if n_grad > 0:
            n_components = int(per_particle.max() + 1) if per_particle.size else 1
            print(f"  [Voronoi] {n_grad} cells graduated this frame "
                  f"(components={n_components})")

    @torch.no_grad()
    def _spatial_split_fragments(self, per_particle: np.ndarray) -> np.ndarray:
        """Refine cell-based fragment ids with a particle-level CC pass.

        A Voronoi cell is fixed at tessellation time, but its
        particles may drift apart afterwards (e.g. a chunk detaches
        and flies up while the rest of its cell stays on the floor).
        Without refinement they keep the same fragment id and render
        with the same colour, which reads as "a small piece floating
        is the same fragment as the main body".

        For each unique fragment id we build a kd-tree on the
        particles' current positions and union-find groups within
        ``eps``; each spatially-disjoint sub-component gets a fresh
        fragment id.  Cluster ids are stabilised across frames via
        majority-overlap inheritance with the previous frame's labels
        (so a fragment keeps its render colour as long as ≥ 50% of
        its particles came from the same prior id).  Cost is
        O(N log N) per frame; tolerable at ≤ 50K particles.
        """
        from scipy.spatial import cKDTree
        if per_particle.size == 0 or self.x_mpm is None:
            return per_particle
        positions = self.x_mpm.detach().cpu().numpy()
        if positions.shape[0] != per_particle.size:
            return per_particle
        bbox_diag = float(np.linalg.norm(
            positions.max(axis=0) - positions.min(axis=0)))
        eps = float(self.fracture_cfg.get(
            'voronoi_spatial_split_eps_frac', 0.025)) * bbox_diag
        if eps <= 0.0:
            return per_particle

        # Pass 1: spatial-CC produces a *temporary* sub-id per
        # particle, unique within this frame but not yet stabilised.
        #
        # Label convention is part of the mechanics contract:
        # 0 = still-cohesive base body, >0 = detached fragment.  The
        # incoming Voronoi labels already follow that convention when a
        # dominant base exists.  Preserve the largest spatial component
        # of label 0 as 0; any smaller pieces that split away from the
        # base graduate to positive fragment ids.
        has_cohesive_base = bool(np.any(per_particle == 0))
        temp = np.zeros_like(per_particle)
        next_temp = 1
        for old_fid in np.unique(per_particle):
            mask = per_particle == old_fid
            idxs = np.where(mask)[0]
            if idxs.size <= 1:
                if int(old_fid) == 0:
                    temp[idxs] = 0
                else:
                    temp[idxs] = next_temp
                    next_temp += 1
                continue
            pts = positions[idxs]
            kdt = cKDTree(pts)
            parent = np.arange(idxs.size, dtype=np.int64)

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b in kdt.query_pairs(eps, output_type='ndarray'):
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[ra] = rb
            roots = np.array([find(i) for i in range(idxs.size)],
                             dtype=np.int64)
            unique_roots, root_counts = np.unique(roots, return_counts=True)
            base_root = None
            if int(old_fid) == 0 and unique_roots.size > 0:
                base_root = int(unique_roots[int(np.argmax(root_counts))])
            for r in np.unique(roots):
                rmask = roots == r
                if rmask.sum() <= 0:
                    continue
                if base_root is not None and int(r) == base_root:
                    temp[idxs[rmask]] = 0
                else:
                    temp[idxs[rmask]] = next_temp
                    next_temp += 1

        min_physical_size = max(
            1,
            int(getattr(self, "fragment_physical_min_size", 1)),
        )
        if min_physical_size > 1:
            cluster_ids, cluster_sizes = np.unique(temp, return_counts=True)
            for cid, size in zip(cluster_ids, cluster_sizes):
                cid_int = int(cid)
                if cid_int <= 0:
                    continue
                if int(size) < min_physical_size and has_cohesive_base:
                    temp[temp == cid_int] = 0

        # A tiny label-0 residue is not a coherent base body.  If it is
        # left as label 0, base shape matching treats disconnected crumbs
        # as one rigid component and injects artificial angular velocity.
        # Promote it to a normal fragment so the rigid handoff path can
        # absorb it into a real nearby shard.
        base_count = int(np.count_nonzero(temp == 0))
        if base_count > 0:
            base_min_particles = max(
                int(getattr(self, "shape_match_min_particles", 12)) * 2,
                int(getattr(self, "fragment_render_min_size", 6)) * 4,
                int(getattr(self, "rigid_handoff_min_particles", 1)),
            )
            if base_count < base_min_particles:
                temp[temp == 0] = next_temp
                next_temp += 1

        # Pass 2: stabilise.  Inherit each new cluster's id from its
        # majority overlap with the previous frame's labels; clusters
        # without a sufficient overlap get fresh ids.
        prev = getattr(self, '_spatial_split_prev_labels', None)
        if prev is None or prev.shape != temp.shape:
            self._spatial_split_prev_labels = temp.copy()
            self._spatial_split_next_id = int(temp.max()) + 1
            return temp

        next_id = int(getattr(self, '_spatial_split_next_id',
                              int(prev.max()) + 1))
        used_ids: set = set()
        cluster_to_final: Dict[int, int] = {}
        # Process new clusters in size-descending order so the largest
        # gets first claim on its preferred id (avoids two large new
        # clusters fighting over the same prev-id).
        cluster_ids, cluster_sizes = np.unique(temp, return_counts=True)
        order = np.argsort(-cluster_sizes)
        for ci in order:
            cid = int(cluster_ids[ci])
            if cid == 0:
                cluster_to_final[cid] = 0
                used_ids.add(0)
                continue
            mask = temp == cid
            n_in = int(mask.sum())
            if n_in == 0:
                continue
            prev_in = prev[mask]
            uniq_prev, counts = np.unique(prev_in, return_counts=True)
            best_idx = int(np.argmax(counts))
            best_prev = int(uniq_prev[best_idx])
            best_share = float(counts[best_idx] / max(n_in, 1))
            inherit = (best_share >= 0.5
                       and best_prev > 0
                       and best_prev not in used_ids)
            if inherit:
                cluster_to_final[cid] = best_prev
                used_ids.add(best_prev)
            else:
                cluster_to_final[cid] = next_id
                used_ids.add(next_id)
                next_id += 1

        refined = np.zeros_like(temp)
        for cid, fin_id in cluster_to_final.items():
            refined[temp == cid] = fin_id
        self._spatial_split_prev_labels = refined.copy()
        self._spatial_split_next_id = next_id
        return refined

    # ------------------------------------------------------------------
    # Mode-I bond opening (Griffith-bounded)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_bond_opening_kicks(self, newly_broken: list[tuple[int, int, float]]):
        """Apply Griffith-bounded Mode-I opening for a batch of broken bonds.

        Each adjacent cell accumulates equal-and-opposite Newton-3rd
        kicks along the bond normal ``(b - a)``.  The opening velocity
        is bounded by Griffith's critical energy release rate

            v_open = sqrt(G_c * A_bond / m_cell)

        and the final cell-level delta is capped once per fracture tick.
        This prevents multi-bond graduation from injecting N copies of the
        same rigid-body velocity into a single fragment.

        Pipeline rewrite Option A (2026-05-10): when
        ``manifold.voronoi_kick_scope`` is ``'bond_boundary'`` the kick
        applies only to particles on the bond's boundary band (the
        actual interface region between the two cells), not to every
        particle assigned to either cell.  This eliminates the
        legacy artifact where breaking a bond at the impact zone
        ejected every particle of an adjacent un-damaged cell (e.g.
        the bunny's legs) as floating dust.  Default scope remains
        ``'cell'`` so existing pipelines are unchanged unless the
        flag is enabled.
        """
        if not newly_broken:
            return
        if self.voronoi is None or self.voronoi.cell_centers is None:
            return
        release_gain = float(getattr(self, "fragment_release_v_com_gain", 0.0))
        if release_gain <= 0.0:
            return
        scope = str(self.fracture_cfg.get('voronoi_kick_scope', 'cell'))
        if scope == 'bond_boundary':
            self._apply_bond_opening_kicks_bond_scope(newly_broken)
            return
        max_speed = max(float(getattr(
            self, "fragment_physical_max_speed", 30.0)), 0.05)
        cell_velocity: dict[int, torch.Tensor] = {}
        cell_offset: dict[int, torch.Tensor] = {}

        # Griffith-bounded opening velocity.  Bond-area heuristic uses
        # cube-face of the cells' combined volume.
        Gc = float(getattr(self.elasticity, 'Gc', 9.0))
        energy_frac = float(getattr(
            self, "fragment_release_energy_fraction", 1.0))
        kick_scale = float(getattr(self, "_voronoi_kick_scale", 1.0))
        min_open_kick = float(getattr(
            self, "fragment_release_min_open_kick", 0.1))
        cell_speed_cap = max(0.05, min(max_speed, release_gain))
        offset_scale = float(getattr(
            self, "_voronoi_position_offset_eff",
            getattr(self, "fragment_release_position_offset", 0.0)))
        offset_cap = max(0.0, float(getattr(
            self, "fragment_release_cell_offset_cap", offset_scale)))

        for (cell_a, cell_b, _stress) in newly_broken:
            ca_np = self.voronoi.cell_centers[cell_a]
            cb_np = self.voronoi.cell_centers[cell_b]
            direction_np = cb_np - ca_np
            n = float(np.linalg.norm(direction_np))
            if n < 1e-6:
                continue
            direction_np = direction_np / n
            direction = torch.from_numpy(direction_np).to(
                self.x_mpm.device, self.x_mpm.dtype)

            cell_count_a = int(np.sum(self.voronoi.cell_assignment == cell_a))
            cell_count_b = int(np.sum(self.voronoi.cell_assignment == cell_b))
            cell_count_min = max(1, min(cell_count_a, cell_count_b))
            m_cell = float(self.mpm.p_mass) * cell_count_min
            cell_vol = float(self.mpm.vol) * cell_count_min
            bond_area = max(cell_vol ** (2.0 / 3.0), 1e-8)
            v_open_griffith = float((Gc * bond_area / max(m_cell, 1e-8)) ** 0.5)
            v_open = v_open_griffith * (max(energy_frac, 0.0) ** 0.5)
            v_open = v_open * kick_scale
            # ``fragment_release_v_com_gain`` is a material/style cap, not a
            # lower bound.  A high sentence-style value should allow energetic
            # cracks when the Griffith budget supports them, not force every
            # broken bond to inject that much velocity.
            v_open = max(min_open_kick * kick_scale, v_open)
            v_open = min(v_open, cell_speed_cap)

            if cell_a not in cell_velocity:
                cell_velocity[cell_a] = torch.zeros(
                    3, device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            if cell_b not in cell_velocity:
                cell_velocity[cell_b] = torch.zeros(
                    3, device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            cell_velocity[cell_a] = cell_velocity[cell_a] - direction * v_open
            cell_velocity[cell_b] = cell_velocity[cell_b] + direction * v_open
            if offset_scale > 0.0:
                if cell_a not in cell_offset:
                    cell_offset[cell_a] = torch.zeros(
                        3, device=self.x_mpm.device, dtype=self.x_mpm.dtype)
                if cell_b not in cell_offset:
                    cell_offset[cell_b] = torch.zeros(
                        3, device=self.x_mpm.device, dtype=self.x_mpm.dtype)
                cell_offset[cell_a] = cell_offset[cell_a] - direction * offset_scale
                cell_offset[cell_b] = cell_offset[cell_b] + direction * offset_scale

        if not cell_velocity and not cell_offset:
            return
        assignment = torch.from_numpy(self.voronoi.cell_assignment).to(
            self.x_mpm.device)
        for cell_id, dv in cell_velocity.items():
            idx = torch.where(assignment == int(cell_id))[0]
            if idx.numel() == 0:
                continue
            speed = float(dv.norm().item())
            if speed > cell_speed_cap:
                dv = dv * (cell_speed_cap / max(speed, 1e-8))
            if offset_cap > 0.0:
                dx = cell_offset.get(int(cell_id))
                if dx is None:
                    dx = torch.zeros_like(dv)
                else:
                    dist = float(dx.norm().item())
                    if dist > offset_cap:
                        dx = dx * (offset_cap / max(dist, 1e-8))
            else:
                dx = None
            self._queue_voronoi_cell_release(int(cell_id), dv, dx)
        for cell_id, dx in cell_offset.items():
            if int(cell_id) in cell_velocity:
                continue
            dist = float(dx.norm().item())
            if offset_cap > 0.0 and dist > offset_cap:
                dx = dx * (offset_cap / max(dist, 1e-8))
            self._queue_voronoi_cell_release(
                int(cell_id),
                torch.zeros(3, device=self.x_mpm.device, dtype=self.x_mpm.dtype),
                dx,
            )

    def _queue_voronoi_cell_release(
        self,
        cell_id: int,
        dv: torch.Tensor,
        dx: torch.Tensor | None = None,
    ) -> None:
        ramp = max(1, int(getattr(
            self, "fragment_release_impulse_ramp_substeps", 1)))
        if ramp <= 1:
            self._apply_voronoi_cell_release_now(cell_id, dv, dx)
            return
        pending = getattr(self, "_pending_voronoi_cell_releases", None)
        if pending is None:
            pending = []
            self._pending_voronoi_cell_releases = pending
        pending.append({
            "cell": int(cell_id),
            "dv": dv.detach().clone(),
            "dx": None if dx is None else dx.detach().clone(),
            "remaining": ramp,
        })

    @torch.no_grad()
    def _apply_pending_voronoi_cell_releases(self) -> None:
        pending = getattr(self, "_pending_voronoi_cell_releases", None)
        if not pending:
            return
        keep = []
        for item in pending:
            remaining = max(int(item.get("remaining", 1)), 1)
            dv = item["dv"].to(device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            dx_raw = item.get("dx")
            dx = None if dx_raw is None else dx_raw.to(
                device=self.x_mpm.device, dtype=self.x_mpm.dtype)
            step_dv = dv / float(remaining)
            step_dx = None if dx is None else dx / float(remaining)
            self._apply_voronoi_cell_release_now(
                int(item["cell"]), step_dv, step_dx)
            remaining -= 1
            if remaining > 0:
                item["dv"] = (dv - step_dv).detach().clone()
                item["dx"] = None if dx is None else (dx - step_dx).detach().clone()
                item["remaining"] = remaining
                keep.append(item)
        self._pending_voronoi_cell_releases = keep

    @torch.no_grad()
    def _apply_voronoi_cell_release_now(
        self,
        cell_id: int,
        dv: torch.Tensor,
        dx: torch.Tensor | None = None,
    ) -> None:
        if self.voronoi is None or self.x_mpm is None or self.v_mpm is None:
            return
        assignment = torch.from_numpy(self.voronoi.cell_assignment).to(
            self.x_mpm.device)
        idx = torch.where(assignment == int(cell_id))[0]
        if idx.numel() == 0:
            return
        if dv is not None and float(dv.norm().item()) > 0.0:
            self.v_mpm[idx] = self.v_mpm[idx] + dv.unsqueeze(0)
        if dx is not None and float(dx.norm().item()) > 0.0:
            self.x_mpm[idx] = self.x_mpm[idx] + dx.unsqueeze(0)

    @torch.no_grad()
    def _apply_bond_opening_kicks_bond_scope(
        self,
        newly_broken: list[tuple[int, int, float]],
    ) -> None:
        """Bond-boundary scoped Mode-I kicks (Pipeline rewrite Option A).

        Apply Newton-3rd kicks only to particles inside the bond's
        boundary band (``voronoi._bond_boundary_idx[(a, b)]``), split
        by which cell they belong to.  This localises the impulse to
        the actual interface region instead of every particle
        assigned to either cell, eliminating the artifact where
        breaking a bond at the impact zone ejected entire un-damaged
        adjacent cells (e.g. the bunny's legs) as floating dust.

        v_open / offset are computed per bond identically to the
        cell-scope path, then applied once to the boundary particles.
        Multiple bonds breaking on the same cell pair contribute
        independently (each bond touches a small particle subset).
        """
        if not newly_broken:
            return
        if self.voronoi is None or self.voronoi.cell_centers is None:
            return
        if self.x_mpm is None or self.v_mpm is None:
            return
        max_speed = max(float(getattr(
            self, "fragment_physical_max_speed", 30.0)), 0.05)
        Gc = float(getattr(self.elasticity, 'Gc', 9.0))
        energy_frac = float(getattr(
            self, "fragment_release_energy_fraction", 1.0))
        kick_scale = float(getattr(self, "_voronoi_kick_scale", 1.0))
        min_open_kick = float(getattr(
            self, "fragment_release_min_open_kick", 0.1))
        release_gain = float(getattr(self, "fragment_release_v_com_gain", 0.0))
        cell_speed_cap = max(0.05, min(max_speed, release_gain))
        offset_scale = float(getattr(
            self, "_voronoi_position_offset_eff",
            getattr(self, "fragment_release_position_offset", 0.0)))

        assignment_np = self.voronoi.cell_assignment
        device = self.x_mpm.device

        for (cell_a, cell_b, _stress) in newly_broken:
            ca_np = self.voronoi.cell_centers[int(cell_a)]
            cb_np = self.voronoi.cell_centers[int(cell_b)]
            direction_np = cb_np - ca_np
            n = float(np.linalg.norm(direction_np))
            if n < 1e-6:
                continue
            direction_np = direction_np / n
            direction = torch.from_numpy(direction_np).to(
                device, self.x_mpm.dtype)

            cell_count_a = int(np.sum(assignment_np == int(cell_a)))
            cell_count_b = int(np.sum(assignment_np == int(cell_b)))
            cell_count_min = max(1, min(cell_count_a, cell_count_b))
            m_cell = float(self.mpm.p_mass) * cell_count_min
            cell_vol = float(self.mpm.vol) * cell_count_min
            bond_area = max(cell_vol ** (2.0 / 3.0), 1e-8)
            v_open = float((Gc * bond_area / max(m_cell, 1e-8)) ** 0.5)
            v_open = v_open * (max(energy_frac, 0.0) ** 0.5)
            v_open = v_open * kick_scale
            v_open = max(min_open_kick * kick_scale, v_open)
            v_open = min(v_open, cell_speed_cap)

            boundary = self.voronoi._bond_boundary_idx.get(
                (int(cell_a), int(cell_b)))
            if boundary is None or boundary.size == 0:
                continue

            sides = assignment_np[boundary]
            a_side = boundary[sides == int(cell_a)]
            b_side = boundary[sides == int(cell_b)]

            if a_side.size > 0:
                a_idx = torch.from_numpy(a_side).to(device).long()
                self.v_mpm[a_idx] = (
                    self.v_mpm[a_idx]
                    - direction.unsqueeze(0) * v_open
                )
                if offset_scale > 0.0:
                    self.x_mpm[a_idx] = (
                        self.x_mpm[a_idx]
                        - direction.unsqueeze(0) * offset_scale
                    )
            if b_side.size > 0:
                b_idx = torch.from_numpy(b_side).to(device).long()
                self.v_mpm[b_idx] = (
                    self.v_mpm[b_idx]
                    + direction.unsqueeze(0) * v_open
                )
                if offset_scale > 0.0:
                    self.x_mpm[b_idx] = (
                        self.x_mpm[b_idx]
                        + direction.unsqueeze(0) * offset_scale
                    )
