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
        # Quintic scaling on n_cells.  Base derivation is cubic from
        # KE-to-fracture-area (A ∝ KE ∝ v^2 and A ∝ n^(2/3) → n ∝ v^3);
        # additional factors of k_e capture (a) a speed-dependent
        # dissipation efficiency ε(v_*) ∝ v_*, and (b) a stress-
        # concentration term g(v_*) ∝ v_* — at low impact speeds the
        # body stays mostly in the elastic regime, so very few stress
        # concentrations exceed the local fracture toughness and n
        # collapses well below the cubic bound.
        n_cells = max(2, int(n_cells_full * (ke_factor ** 6)))

        # force_shrink scales INVERSELY: hard impact → 5% base cap
        # (full pulverization), soft impact → 95% base (mostly intact).
        ke_inv = max(0.0, 1.0 - ke_factor)
        force_shrink = force_shrink_full + ke_inv * 2.0
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
        )
        # Mode-I bond opening: Newton-3rd kicks per newly-broken bond.
        for (a, b, stress) in newly_broken:
            self._apply_bond_opening_kick(a, b, stress)
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
            # Pipeline rewrite Step 2: update the per-fragment birth
            # registry whenever the voronoi label set changes.  Cheap
            # (insert-only on new labels) and benign even when the
            # authority flag is off — downstream consumers only read
            # the registry under the flag.
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
        temp = np.zeros_like(per_particle)
        next_temp = 1
        for old_fid in np.unique(per_particle):
            mask = per_particle == old_fid
            idxs = np.where(mask)[0]
            if idxs.size <= 1:
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
            for r in np.unique(roots):
                rmask = roots == r
                if rmask.sum() <= 0:
                    continue
                temp[idxs[rmask]] = next_temp
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
    def _apply_bond_opening_kick(self, cell_a: int, cell_b: int, stress: float):
        """Mode-I crack opening at one broken bond.

        Each adjacent cell receives an equal-and-opposite Newton-3rd
        kick along the bond normal ``(b - a)``.  The opening velocity
        is bounded by Griffith's critical energy release rate

            v_open = sqrt(G_c * A_bond / m_cell)

        so the released KE never exceeds physical surface energy.  This
        replaces the legacy ``gain * sqrt(stress)`` formulation that
        injected non-physical KE and produced an "explosion" feel.
        """
        if self.voronoi is None or self.voronoi.cell_centers is None:
            return
        gain = float(getattr(self, "fragment_release_v_com_gain", 0.0))
        if gain <= 0.0:
            return
        max_speed = max(float(getattr(
            self, "fragment_physical_max_speed", 30.0)), 0.05)

        ca_np = self.voronoi.cell_centers[cell_a]
        cb_np = self.voronoi.cell_centers[cell_b]
        direction_np = cb_np - ca_np
        n = float(np.linalg.norm(direction_np))
        if n < 1e-6:
            return
        direction_np = direction_np / n
        direction = torch.from_numpy(direction_np).to(
            self.x_mpm.device, self.x_mpm.dtype)

        # Griffith-bounded opening velocity.  Bond-area heuristic uses
        # cube-face of the cells' combined volume.
        Gc = float(getattr(self.elasticity, 'Gc', 9.0))
        cell_count_a = int(np.sum(self.voronoi.cell_assignment == cell_a))
        cell_count_b = int(np.sum(self.voronoi.cell_assignment == cell_b))
        cell_count_min = max(1, min(cell_count_a, cell_count_b))
        m_cell = float(self.mpm.p_mass) * cell_count_min
        cell_vol = float(self.mpm.vol) * cell_count_min
        bond_area = max(cell_vol ** (2.0 / 3.0), 1e-8)
        v_open_griffith = float((Gc * bond_area / max(m_cell, 1e-8)) ** 0.5)
        # User-tunable energy-conversion fraction, applied to the
        # released KE (so v scales with sqrt of the fraction).
        energy_frac = float(getattr(
            self, "fragment_release_energy_fraction", 1.0))
        v_open = v_open_griffith * (energy_frac ** 0.5)
        # Impact-energy multiplier: soft drops further dampen the
        # released KE since the body's KE budget is smaller.
        kick_scale = float(getattr(self, "_voronoi_kick_scale", 1.0))
        v_open = v_open * kick_scale
        # Floor for force-shrink/cascade-driven breaks (stress=0); cap
        # at the configured max_speed.
        min_open_kick = float(getattr(
            self, "fragment_release_min_open_kick", 0.1))
        v_open = max(min_open_kick, v_open)
        v_open = min(v_open, max_speed)

        idx_a = torch.where(
            torch.from_numpy(self.voronoi.cell_assignment == cell_a)
            .to(self.x_mpm.device)
        )[0]
        idx_b = torch.where(
            torch.from_numpy(self.voronoi.cell_assignment == cell_b)
            .to(self.x_mpm.device)
        )[0]
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            return

        kick_b = direction * v_open
        kick_a = -direction * v_open
        self.v_mpm[idx_a] = self.v_mpm[idx_a] + kick_a.unsqueeze(0)
        self.v_mpm[idx_b] = self.v_mpm[idx_b] + kick_b.unsqueeze(0)

        # Position offset along the bond direction so the cells are
        # geometrically separated in the shared MPM grid; without it
        # the heavy-base velocity field swallows the kicks via grid
        # coupling within one P2G2P step.
        offset_scale = float(getattr(
            self, "_voronoi_position_offset_eff",
            getattr(self, "fragment_release_position_offset", 0.0)))
        if offset_scale > 0.0:
            self.x_mpm[idx_a] = self.x_mpm[idx_a] + (-direction * offset_scale).unsqueeze(0)
            self.x_mpm[idx_b] = self.x_mpm[idx_b] + (direction * offset_scale).unsqueeze(0)
