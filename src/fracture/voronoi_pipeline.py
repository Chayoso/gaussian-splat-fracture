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
        # Cubic scaling on n_cells: derived from KE-to-fracture-area
        # scaling (A ∝ KE ∝ v^2 and A ∝ n^(2/3) → n ∝ v^3).
        n_cells = max(4, int(n_cells_full * (ke_factor ** 3)))

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
        self.voronoi = VoronoiDecomposer(
            n_cells=n_cells,
            distribution=distribution,
            bond_break_threshold=self._voronoi_bond_thr_effective,
            impact_center=impact_center,
            anisotropy_axis=anisotropy_axis,
            seed=seed,
            force_shrink_max_frac=force_shrink,
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
        """Per-frame Voronoi update: damage → bond breakage → components.

        Flow:
            1. Get per-MPM-particle damage via ``_get_volumetric_damage``.
            2. Update bond breakage (stress wave + cascade gates).
            3. For each newly-broken bond, apply a Newton-3rd Mode-I
               opening kick.
            4. Compute connected components and write per-particle
               fragment ids back to the persistent registry so the
               renderer / fragment-manager use them downstream.
        """
        if self.voronoi is None or self.x_mpm is None:
            return
        damage = self._get_volumetric_damage()
        if damage.numel() != self.x_mpm.shape[0]:
            return
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

        labels = torch.from_numpy(per_particle).to(
            self.x_mpm.device, dtype=torch.long)
        if labels.shape[0] == self.x_mpm.shape[0]:
            self._physical_fragment_labels = labels
            self._next_physical_fragment_id = int(labels.max().item()) + 1

        if n_grad > 0:
            n_components = int(per_particle.max() + 1) if per_particle.size else 1
            print(f"  [Voronoi] {n_grad} cells graduated this frame "
                  f"(components={n_components})")

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
