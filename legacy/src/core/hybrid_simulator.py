"""
Hybrid Crack Simulator

MPM physics + Phase field crack propagation + Gaussian Splatting visualization.
"""

import torch
from torch import Tensor
from typing import Dict, Optional
import time
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
# Also ensure project root is in path for shared modules
_project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, os.path.abspath(_project_root))

from src.mpm_core.mpm_model import MPMModel
from legacy.src.constitutive_models.phase_field import update_phase_field
from legacy.src.constitutive_models.damage_mapper import VolumetricToSurfaceDamageMapper
from src.visualization.gaussian_updater import GaussianCrackVisualizer
from src.core.coordinate_mapper import CoordinateMapper
from legacy.src.core.fragment_manager import FragmentManager


class HybridCrackSimulator:
    """
    Unified MPM + Gaussian Splats simulator for crack visualization.

    Per-timestep flow:
    1. MPM physics → updated (x, v, F)
    2. Coordinate map: x_mpm → x_world
    3. Damage project: c_vol → c_surf
    4. Gaussian update → render
    """

    def __init__(
        self,
        mpm_model: MPMModel,
        gaussians,
        elasticity_module,
        coord_mapper: CoordinateMapper,
        damage_mapper: VolumetricToSurfaceDamageMapper,
        visualizer: GaussianCrackVisualizer,
        surface_mask: Tensor,
        physics_substeps: int = 10,
        phase_field_params: Optional[Dict] = None,
        simulation_mode: str = "crack_only",
        seismic_params: Optional[Dict] = None
    ):
        self.mpm = mpm_model
        self.gaussians = gaussians
        self.elasticity = elasticity_module
        self.mapper = coord_mapper
        self.damage_mapper = damage_mapper
        self.visualizer = visualizer
        self.surface_mask = surface_mask
        self.substeps = physics_substeps
        self.pf_params = phase_field_params or {}
        self.simulation_mode = simulation_mode

        self.seismic = seismic_params or {}
        self.seismic_enabled = self.seismic.get("enabled", False)

        # Simulation state
        self.x_mpm = None
        self.v_mpm = None
        self.F = None
        self.C = None
        self.c_vol = None

        self.crack_only = (simulation_mode == "crack_only")
        self.psi_static = None

        # Gravity drop: Phase 1 = free fall, Phase 2 = MPM impact
        self._gravity_drop = False
        self._gravity_drop_ground_z = 0.1
        self._gravity_drop_contacted = False
        self._v_com = None

        self.frame_count = 0
        self.last_render_time = time.time()
        self.init_positions = None

        device = next(mpm_model.parameters()).device if hasattr(mpm_model, 'parameters') else torch.device('cuda')

        # Fragment tracking
        frag_enabled = self.pf_params.get('fragmentation_enabled', False)
        self.fragment_manager = FragmentManager(
            damage_threshold=self.pf_params.get('fragment_damage_threshold', 0.5),
            min_fragment_particles=self.pf_params.get('min_fragment_particles', 50),
            device=str(device),
        ) if frag_enabled else None
        self.fragmentation_active = False

        print(f"\n{'='*60}")
        print(f"HybridCrackSimulator Initialized")
        print(f"{'='*60}")
        print(f"  - Simulation mode: {simulation_mode}")
        print(f"  - MPM particles: {self.surface_mask.shape[0]}")
        print(f"  - Surface particles: {self.surface_mask.sum().item()}")
        print(f"  - Physics substeps: {physics_substeps}")
        if self.seismic_enabled:
            print(f"  - Seismic loading: ON")
            print(f"    amplitude={self.seismic.get('amplitude')}, "
                  f"freq={self.seismic.get('frequency')}Hz, "
                  f"dir={self.seismic.get('direction')}")
        print(f"  - Device: {device}")
        print(f"{'='*60}\n")

    def enable_gravity_drop(self, ground_z: float = 0.1):
        """Enable 2-phase gravity drop mode."""
        self._gravity_drop = True
        self._gravity_drop_ground_z = ground_z
        self._gravity_drop_contacted = False
        self._v_com = torch.zeros(3, device=self.mpm.gravity.device)
        print(f"[GravityDrop] Enabled. Ground at z={ground_z}")
        print(f"  - Phase 1: free-fall translation (no grid stress)")
        print(f"  - Phase 2: MPM impact physics after contact")

    def initialize(self, init_positions: Tensor):
        """Initialize simulation state from particle positions in [0,1]^3."""
        device = init_positions.device
        N = init_positions.shape[0]

        print(f"[HybridSimulator] Initializing state...")
        print(f"  - Particles: {N}")
        print(f"  - Device: {device}")

        self.x_mpm = init_positions.clone()
        self.v_mpm = torch.zeros((N, 3), device=device)
        self.F = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3).clone()
        self.C = torch.zeros((N, 3, 3), device=device)
        self.c_vol = torch.zeros(N, device=device)
        self.init_positions = init_positions.clone()

        x_surf_mpm = self.x_mpm[self.surface_mask]
        x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)

        # Direct PLY mode: track displacement relative to surface particles
        self._ply_direct = getattr(self.gaussians, '_ply_direct_mode', False)
        if self._ply_direct:
            ply_to_surf = self.gaussians._ply_to_surface  # (K,) numpy
            self._ply_to_surface = torch.from_numpy(ply_to_surf).long().to(device)
            self._ply_init_xyz = self.gaussians._xyz.data.clone()
            self._surf_init_world = x_surf_world.clone()
            # Apply initial alignment: shift PLY Gaussians to match surface positions
            surf_pos_for_ply = x_surf_world[self._ply_to_surface]
            offset = surf_pos_for_ply - self._ply_init_xyz
            # Keep PLY positions (they're already well-placed), just store reference
            print(f"  - Direct PLY mode: {self._ply_init_xyz.shape[0]} Gaussians")
        else:
            self.gaussians._xyz.data = x_surf_world

        self.frame_count = 0
        self.last_render_time = time.time()

        print(f"  - State initialized")
        print(f"  - Surface Gaussians: {x_surf_world.shape[0]}")

    @torch.no_grad()
    def apply_pre_notch(self, notches: list):
        """Seed initial cracks from pre-defined notch lines."""
        device = self.x_mpm.device
        n = self.mpm.num_grids
        dx = self.mpm.dx
        crack_width = self.pf_params.get('crack_width', 0.03)

        if not hasattr(self, 'c_grid'):
            self._init_grid_infrastructure()
        if not hasattr(self, 'crack_paths'):
            self.crack_paths = []
            self.crack_dirs = []

        H_ref = getattr(self.elasticity, 'Gc', 30.0) / (2.0 * getattr(self.elasticity, 'l0', 0.025))

        for notch in notches:
            start = torch.tensor(notch['start'], device=device, dtype=torch.float32)
            end = torch.tensor(notch['end'], device=device, dtype=torch.float32)
            damage = notch.get('damage', 0.9)

            center = (start + end) * 0.5
            direction = (end - start)
            length = direction.norm().item()
            direction = direction / (direction.norm() + 1e-8)

            # Full notch damage seeding
            n_pts_full = max(2, int(length / dx) + 1)
            t_full = torch.linspace(0.0, 1.0, n_pts_full, device=device)
            full_path = start.unsqueeze(0) + t_full.unsqueeze(1) * (end - start).unsqueeze(0)

            min_dist = self._point_to_polyline_dist(self.x_mpm, full_path)
            c_notch = (1.0 - (min_dist / crack_width)).clamp(0.0, 1.0) * damage
            self.c_vol = torch.maximum(self.c_vol, c_notch)

            # H seed along notch
            if not hasattr(self, '_history_H'):
                self._history_H = torch.zeros(self.x_mpm.shape[0], device=device)
            H_seed = H_ref * 3.0
            notch_influence = (1.0 - (min_dist / (crack_width * 2.0))).clamp(0.0, 1.0)
            self._history_H = torch.maximum(self._history_H, notch_influence * H_seed)

            if hasattr(self, 'H_grid'):
                self.H_grid = self._bin_particles_to_grid(self._history_H)

            # Short seed paths from center
            seed_len = 3 * dx
            n_seed = max(2, int(seed_len / dx) + 1)
            t_fwd = torch.linspace(0.0, 1.0, n_seed, device=device)

            path_fwd = center.unsqueeze(0) + t_fwd.unsqueeze(1) * (seed_len * direction).unsqueeze(0)
            self.crack_paths.append(path_fwd)
            self.crack_dirs.append(direction.clone())

            path_bwd = center.unsqueeze(0) + t_fwd.unsqueeze(1) * (seed_len * (-direction)).unsqueeze(0)
            self.crack_paths.append(path_bwd)
            self.crack_dirs.append(-direction.clone())

            n_damaged = (c_notch > 0.01).sum().item()
            print(f"  [Pre-notch] {notch['start']} → {notch['end']}")
            print(f"    center=[{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}]")
            print(f"    seed: 2 paths x {n_seed}pts, damaged particles: {n_damaged}")

    def initialize_crack_energy(self, impact_center_mpm: Tensor,
                                impact_energy: float = 1.0,
                                impact_radius: float = 0.03):
        """Create grid-based damage seed for crack-only mode."""
        n = self.mpm.num_grids
        dx = self.mpm.dx
        device = self.x_mpm.device
        P = self.x_mpm.shape[0]
        chunk = self.mpm.particle_chunk or P

        coords_1d = torch.arange(n, device=device).float() * dx
        gi, gj, gk = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        grid_pos = torch.stack([gi, gj, gk], dim=-1)

        dists = (grid_pos - impact_center_mpm.view(1, 1, 1, 3)).norm(dim=-1)
        self.c_grid = torch.exp(-dists ** 2 / (2 * impact_radius ** 2)).clamp(0.0, 1.0)

        # Occupancy mask from particles (dilated by 1 cell)
        grid_w = torch.zeros(n ** 3, device=device)
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self.mpm._weight_and_index(self.x_mpm[i:j])
            grid_w.index_add_(0, index.reshape(-1), weight.reshape(-1))
        self.grid_occupied = (grid_w.view(n, n, n) > 1e-6)
        occ = self.grid_occupied.float().unsqueeze(0).unsqueeze(0)
        occ_dilated = torch.nn.functional.max_pool3d(occ, kernel_size=3, stride=1, padding=1)
        self.grid_occupied = (occ_dilated[0, 0] > 0)

        self.c_grid = self.c_grid * self.grid_occupied.float()
        self.c_grid_seed = self.c_grid.clone()
        self._gather_grid_to_particles()

        n_seeded = (self.c_grid > 0.01).sum().item()
        n_high = (self.c_grid > 0.5).sum().item()
        n_occupied = self.grid_occupied.sum().item()
        print(f"\n[CrackOnly] Grid-based crack initialized:")
        print(f"  - Impact center (MPM): {impact_center_mpm.detach().cpu().numpy()}")
        print(f"  - Impact radius: {impact_radius}")
        print(f"  - Grid seeded cells (c>0.01): {n_seeded}")
        print(f"  - Grid high cells (c>0.5): {n_high}")
        print(f"  - Grid occupied cells: {n_occupied}/{n ** 3}")
        print(f"  - c_grid max: {self.c_grid.max():.4f}")
        print(f"  - c_vol max (particles): {self.c_vol.max():.4f}")
        print(f"  - Crack-only mode ENABLED (grid-based Fisher-KPP)")

    def initialize_deformation_impact(self, impact_center_mpm: Tensor,
                                      impact_energy: float = 1.0,
                                      impact_radius: float = 0.03,
                                      impact_direction: Optional[Tensor] = None):
        """Apply impact for deformation mode: velocity impulse + damage seed."""
        device = self.x_mpm.device

        dists = (self.x_mpm - impact_center_mpm.unsqueeze(0)).norm(dim=1)
        influence = torch.exp(-dists ** 2 / (2 * impact_radius ** 2))

        # Velocity impulse
        if impact_direction is not None:
            imp_dir = impact_direction.to(device)
            imp_dir = imp_dir / (imp_dir.norm() + 1e-12)
            self.v_mpm = self.v_mpm + imp_dir.unsqueeze(0) * influence.unsqueeze(1) * impact_energy
        else:
            directions = impact_center_mpm.unsqueeze(0) - self.x_mpm
            directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-12)
            self.v_mpm = self.v_mpm + directions * influence.unsqueeze(1) * impact_energy

        self.c_vol = torch.maximum(self.c_vol, influence * 0.8)

        # Initialize grid
        n = self.mpm.num_grids
        dx = self.mpm.dx
        P = self.x_mpm.shape[0]
        chunk = self.mpm.particle_chunk or P

        coords_1d = torch.arange(n, device=device).float() * dx
        gi, gj, gk = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        grid_pos = torch.stack([gi, gj, gk], dim=-1)
        grid_dists = (grid_pos - impact_center_mpm.view(1, 1, 1, 3)).norm(dim=-1)
        self.c_grid = torch.exp(-grid_dists ** 2 / (2 * impact_radius ** 2)).clamp(0.0, 1.0)

        grid_w = torch.zeros(n ** 3, device=device)
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self.mpm._weight_and_index(self.x_mpm[i:j])
            grid_w.index_add_(0, index.reshape(-1), weight.reshape(-1))
        self.grid_occupied = (grid_w.view(n, n, n) > 1e-6)
        occ = self.grid_occupied.float().unsqueeze(0).unsqueeze(0)
        occ_dilated = torch.nn.functional.max_pool3d(occ, kernel_size=3, stride=1, padding=1)
        self.grid_occupied = (occ_dilated[0, 0] > 0)

        self.c_grid = self.c_grid * self.grid_occupied.float()
        self.c_grid_seed = self.c_grid.clone()
        self.H_grid = torch.zeros(n, n, n, device=device)

        n_impulse = (influence > 0.01).sum().item()
        n_total_damaged = (self.c_vol > 0.001).sum().item()
        n_grid_seeded = (self.c_grid > 0.01).sum().item()
        n_occupied = self.grid_occupied.sum().item()
        v_max = self.v_mpm.abs().max().item()
        print(f"\n[Deformation+Hybrid] Impact applied:")
        print(f"  - Impact center (MPM): {impact_center_mpm.detach().cpu().numpy()}")
        print(f"  - Impact direction: {'camera→object' if impact_direction is not None else 'radial inward'}")
        print(f"  - Impact energy: {impact_energy}, radius: {impact_radius}")
        print(f"  - Particles with impulse (>0.01): {n_impulse}")
        print(f"  - Total particles with damage: {n_total_damaged}")
        print(f"  - Grid seeded cells: {n_grid_seeded}/{n**3}")
        print(f"  - Grid occupied cells: {n_occupied}/{n**3}")
        print(f"  - Max velocity: {v_max:.4f}")
        print(f"  - Hybrid mode: MPM physics + grid Fisher-KPP + H modulation")

    # ================================================================
    # Fisher-KPP crack-only mode
    # ================================================================

    @torch.no_grad()
    def step_crack_only(self, dt: float):
        """Fisher-KPP reaction-diffusion on persistent c_grid (no deformation)."""
        if not hasattr(self, '_crack_step'):
            self._crack_step = 0

        diff_coeff = self.pf_params.get('crack_diff_coeff', 0.1)
        alpha = self.pf_params.get('crack_alpha', 10.0)
        n_iters = self.pf_params.get('crack_grid_iters', 10)
        n = self.mpm.num_grids
        dx = self.mpm.dx

        dt_rd = min(0.8 * dx * dx / (2.0 * diff_coeff * 3.0 + 1e-12), 0.01)

        occ_float = self.grid_occupied.float()
        for _ in range(n_iters):
            g5 = self.c_grid.unsqueeze(0).unsqueeze(0)
            gp = torch.nn.functional.pad(g5, (1,1,1,1,1,1), mode='replicate')[0, 0]
            lap = (
                gp[2:, 1:-1, 1:-1] + gp[:-2, 1:-1, 1:-1] +
                gp[1:-1, 2:, 1:-1] + gp[1:-1, :-2, 1:-1] +
                gp[1:-1, 1:-1, 2:] + gp[1:-1, 1:-1, :-2] - 6.0 * self.c_grid
            ) / (dx * dx)

            reaction = alpha * self.c_grid * (1.0 - self.c_grid)
            self.c_grid = (self.c_grid + dt_rd * (diff_coeff * lap + reaction)).clamp(0.0, 1.0)
            self.c_grid = self.c_grid * occ_float

        c_old = self.c_vol.clone()
        self._gather_grid_to_particles()
        self.c_vol = torch.maximum(self.c_vol, c_old)

        step = self._crack_step
        if step < 20 or step % 10 == 0:
            dc = self.c_vol - c_old
            gc_max = self.c_grid.max().item()
            gc_cells = (self.c_grid > 0.3).sum().item()
            print(f"  [crack {step:3d}] c_max={self.c_vol.max():.4f} "
                  f"gc_max={gc_max:.4f} gc_cells(>0.3)={gc_cells} "
                  f"dc_max={dc.max():.6f} "
                  f"cracked(>0.01)={(self.c_vol > 0.01).sum().item()} "
                  f"cracked(>0.3)={(self.c_vol > 0.3).sum().item()} "
                  f"cracked(>0.8)={(self.c_vol > 0.8).sum().item()}", flush=True)

        self._crack_step += 1

    # ================================================================
    # Grid ↔ Particle transfers
    # ================================================================

    @torch.no_grad()
    def _gather_grid_to_particles(self):
        """Interpolate c_grid → particle damage using MPM B-spline weights."""
        grid_flat = self.c_grid.reshape(-1)
        P = self.x_mpm.shape[0]
        chunk = self.mpm.particle_chunk or P
        device = self.x_mpm.device

        c_new = torch.empty(P, device=device)
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self.mpm._weight_and_index(self.x_mpm[i:j])
            c_new[i:j] = (weight * grid_flat[index].view(-1, 27)).sum(dim=1)

        self.c_vol = c_new.clamp(0.0, 1.0)

    @torch.no_grad()
    def _bin_particles_to_grid(self, values: Tensor) -> Tensor:
        """Bin particle scalars to grid (weighted average)."""
        n = self.mpm.num_grids
        P = self.x_mpm.shape[0]
        chunk = self.mpm.particle_chunk or P
        device = self.x_mpm.device

        grid_num = torch.zeros(n ** 3, device=device)
        grid_den = torch.zeros(n ** 3, device=device)

        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self.mpm._weight_and_index(self.x_mpm[i:j])
            wv = weight * values[i:j].unsqueeze(1)
            grid_num.index_add_(0, index.reshape(-1), wv.reshape(-1))
            grid_den.index_add_(0, index.reshape(-1), weight.reshape(-1))

        return (grid_num / (grid_den + 1e-12)).view(n, n, n)

    # ================================================================
    # Seismic loading
    # ================================================================

    @torch.no_grad()
    def _apply_seismic_loading(self, dt: float):
        """Apply oscillating sinusoidal body acceleration to all particles."""
        if not self.seismic_enabled:
            return

        t = self.mpm.time
        amp = self.seismic.get("amplitude", 1000.0)
        freq = self.seismic.get("frequency", 80.0)
        direction = self.seismic.get("direction", [1.0, 0.0, 0.0])
        ramp_time = self.seismic.get("ramp_time", 0.005)

        device = self.v_mpm.device
        dir_tensor = torch.tensor(direction, device=device, dtype=self.v_mpm.dtype)
        dir_tensor = dir_tensor / (dir_tensor.norm() + 1e-12)

        envelope = min(t / ramp_time, 1.0) if ramp_time > 0 else 1.0
        accel = amp * math.sin(2.0 * math.pi * freq * t) * envelope
        self.v_mpm += dir_tensor.unsqueeze(0) * (accel * dt)

    # ================================================================
    # Grid infrastructure
    # ================================================================

    @torch.no_grad()
    def _init_grid_infrastructure(self):
        """Initialize c_grid, occupancy mask, and H_grid from current particles."""
        n = self.mpm.num_grids
        device = self.x_mpm.device
        P = self.x_mpm.shape[0]
        chunk = self.mpm.particle_chunk or P

        self.c_grid = torch.zeros(n, n, n, device=device)

        grid_w = torch.zeros(n ** 3, device=device)
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self.mpm._weight_and_index(self.x_mpm[i:j])
            grid_w.index_add_(0, index.reshape(-1), weight.reshape(-1))
        self.grid_occupied = (grid_w.view(n, n, n) > 1e-6)

        occ = self.grid_occupied.float().unsqueeze(0).unsqueeze(0)
        occ_dilated = torch.nn.functional.max_pool3d(occ, kernel_size=3, stride=1, padding=1)
        self.grid_occupied = (occ_dilated[0, 0] > 0)

        self.c_grid_seed = self.c_grid.clone()
        self.H_grid = torch.zeros(n, n, n, device=device)

        n_occupied = self.grid_occupied.sum().item()
        print(f"\n[Grid Init] Auto-initialized grid infrastructure (seismic-only mode)")
        print(f"  - Grid occupied cells: {n_occupied}/{n**3}")
        print(f"  - c_grid: all zeros (nucleation will seed)")

    # ================================================================
    # Anisotropic diffusion (stress-directed)
    # ================================================================

    @torch.no_grad()
    def _compute_aniso_diffusion(self) -> Tensor:
        """Compute anisotropic diffusion tensor D from stress eigenvectors."""
        n = self.mpm.num_grids
        device = self.x_mpm.device

        D_base = self.pf_params.get('crack_diff_coeff', 0.0005)
        aniso_ratio = self.pf_params.get('crack_aniso_ratio', 10.0)
        D_max = D_base
        D_min = D_base / aniso_ratio

        D_tensor = torch.zeros(n, n, n, 6, device=device)
        D_tensor[:, :, :, 0] = D_base
        D_tensor[:, :, :, 1] = D_base
        D_tensor[:, :, :, 2] = D_base

        if not hasattr(self, '_last_stress'):
            return D_tensor

        stress = self._last_stress
        components = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
        S_comps = [self._bin_particles_to_grid(stress[:, i, j]) for (i, j) in components]

        occ_idx = self.grid_occupied.nonzero()
        M = occ_idx.shape[0]
        if M == 0:
            return D_tensor

        S_occ = torch.zeros(M, 3, 3, device=device)
        for k, (i, j) in enumerate(components):
            vals = S_comps[k][occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]]
            S_occ[:, i, j] = vals
            if i != j:
                S_occ[:, j, i] = vals

        S_occ = torch.nan_to_num(S_occ, 0.0, 0.0, 0.0)
        S_norm = S_occ.norm(dim=(1, 2))
        sig_mask = (S_norm > 1e-3) & torch.isfinite(S_norm)
        if sig_mask.sum() == 0:
            return D_tensor

        sig_idx = occ_idx[sig_mask]
        S_sig = S_occ[sig_mask]
        S_sig = 0.5 * (S_sig + S_sig.transpose(1, 2))

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(S_sig)
        except Exception:
            return D_tensor
        n1 = eigenvectors[:, :, -1]

        if not hasattr(self, '_n1_grid'):
            self._n1_grid = torch.zeros(n, n, n, 3, device=device)
        self._n1_grid.zero_()
        self._n1_grid[sig_idx[:, 0], sig_idx[:, 1], sig_idx[:, 2]] = n1

        dD = D_min - D_max
        ix, iy, iz = sig_idx[:, 0], sig_idx[:, 1], sig_idx[:, 2]
        D_tensor[ix, iy, iz, 0] = D_max + dD * n1[:, 0] ** 2
        D_tensor[ix, iy, iz, 1] = D_max + dD * n1[:, 1] ** 2
        D_tensor[ix, iy, iz, 2] = D_max + dD * n1[:, 2] ** 2
        D_tensor[ix, iy, iz, 3] = dD * n1[:, 0] * n1[:, 1]
        D_tensor[ix, iy, iz, 4] = dD * n1[:, 0] * n1[:, 2]
        D_tensor[ix, iy, iz, 5] = dD * n1[:, 1] * n1[:, 2]

        return D_tensor

    @torch.no_grad()
    def _anisotropic_laplacian(self, c: Tensor, D_tensor: Tensor) -> Tensor:
        """Compute L = sum_ij D_ij * d²c/dxi dxj using central finite differences."""
        dx = self.mpm.dx
        dx2 = dx * dx

        g5 = c.unsqueeze(0).unsqueeze(0)
        gp = torch.nn.functional.pad(g5, (1, 1, 1, 1, 1, 1), mode='replicate')[0, 0]

        c_xx = (gp[2:, 1:-1, 1:-1] - 2 * c + gp[:-2, 1:-1, 1:-1]) / dx2
        c_yy = (gp[1:-1, 2:, 1:-1] - 2 * c + gp[1:-1, :-2, 1:-1]) / dx2
        c_zz = (gp[1:-1, 1:-1, 2:] - 2 * c + gp[1:-1, 1:-1, :-2]) / dx2

        c_xy = (gp[2:, 2:, 1:-1] - gp[2:, :-2, 1:-1]
                - gp[:-2, 2:, 1:-1] + gp[:-2, :-2, 1:-1]) / (4 * dx2)
        c_xz = (gp[2:, 1:-1, 2:] - gp[2:, 1:-1, :-2]
                - gp[:-2, 1:-1, 2:] + gp[:-2, 1:-1, :-2]) / (4 * dx2)
        c_yz = (gp[1:-1, 2:, 2:] - gp[1:-1, 2:, :-2]
                - gp[1:-1, :-2, 2:] + gp[1:-1, :-2, :-2]) / (4 * dx2)

        return (D_tensor[:,:,:,0] * c_xx + D_tensor[:,:,:,1] * c_yy +
                D_tensor[:,:,:,2] * c_zz + 2.0 * D_tensor[:,:,:,3] * c_xy +
                2.0 * D_tensor[:,:,:,4] * c_xz + 2.0 * D_tensor[:,:,:,5] * c_yz)

    # ================================================================
    # Hybrid crack propagation (AT2 + geometric)
    # ================================================================

    @torch.no_grad()
    def step_hybrid_crack(self, dt: float):
        """Hybrid crack: nucleation + tip advance + AT2 PDE + geometric damage."""
        if not hasattr(self, 'c_grid'):
            self._init_grid_infrastructure()
        if not hasattr(self, '_hybrid_step'):
            self._hybrid_step = 0

        n = self.mpm.num_grids
        dx = self.mpm.dx
        device = self.x_mpm.device
        step = self._hybrid_step

        Gc = getattr(self.elasticity, 'Gc', 30.0)
        l0 = getattr(self.elasticity, 'l0', 0.025)
        H_ref = Gc / (2.0 * l0)
        max_nuc = self.pf_params.get('max_nucleation_per_frame', 1)
        nucleation_frac = self.pf_params.get('nucleation_fraction', 0.3)
        min_spacing = self.pf_params.get('nucleation_min_spacing', 8)
        crack_tip_speed = self.pf_params.get('crack_tip_speed', 1.5)
        crack_width = self.pf_params.get('crack_width', 0.025)
        max_total_cracks = self.pf_params.get('max_total_cracks', 5)

        if not hasattr(self, 'crack_paths'):
            self.crack_paths = []
            self.crack_dirs = []

        # 1) Bin particle H to grid + crack-tip H boost
        if hasattr(self, '_history_H'):
            self.H_grid = self._bin_particles_to_grid(self._history_H)
        else:
            self.H_grid = torch.zeros(n, n, n, device=device)

        self._H_grid_physics = self.H_grid.clone()

        H_crack = H_ref * 1.2
        if self.crack_paths:
            for path in self.crack_paths:
                gi = (path * n).long().clamp(0, n - 1)
                current_H = self.H_grid[gi[:, 0], gi[:, 1], gi[:, 2]]
                self.H_grid[gi[:, 0], gi[:, 1], gi[:, 2]] = torch.maximum(
                    current_H, torch.full_like(current_H, H_crack))

        # 2) Stress eigenvectors for propagation direction
        if hasattr(self, '_last_stress') and self._last_stress is not None:
            self._compute_aniso_diffusion()

        # 3) Nucleation
        n_new = 0
        if len(self.crack_paths) < max_total_cracks:
            crack_mask = torch.zeros(n, n, n, device=device, dtype=torch.bool)
            for path in self.crack_paths:
                for pt in path:
                    gi = (pt * n).long().clamp(0, n - 1)
                    lo = (gi - min_spacing).clamp(min=0)
                    hi = (gi + min_spacing + 1).clamp(max=n)
                    crack_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True

            occ = self.grid_occupied
            interior = (occ[1:-1, 1:-1, 1:-1] &
                        occ[2:, 1:-1, 1:-1] & occ[:-2, 1:-1, 1:-1] &
                        occ[1:-1, 2:, 1:-1] & occ[1:-1, :-2, 1:-1] &
                        occ[1:-1, 1:-1, 2:] & occ[1:-1, 1:-1, :-2])
            interior_full = torch.zeros_like(occ)
            interior_full[1:-1, 1:-1, 1:-1] = interior

            # Impact zone override for gravity_drop
            if (getattr(self, '_gravity_drop', False) and
                    getattr(self, '_gravity_drop_contacted', False) and
                    hasattr(self, '_impact_center')):
                ic = self._impact_center
                ir = self._impact_radius
                iz, iy, ix = torch.meshgrid(
                    torch.arange(n, device=device),
                    torch.arange(n, device=device),
                    torch.arange(n, device=device), indexing='ij')
                gc = torch.stack([
                    (iz.float() + 0.5) / n,
                    (iy.float() + 0.5) / n,
                    (ix.float() + 0.5) / n], dim=-1)
                dist_grid = (gc - ic.view(1, 1, 1, 3)).norm(dim=-1)
                impact_zone = (dist_grid < ir * 1.5) & occ
                interior_full = interior_full | impact_zone

            candidate_mask = ((self.H_grid > nucleation_frac * H_ref) &
                              interior_full & ~crack_mask)
            n_candidates = candidate_mask.sum().item()

            if n_candidates > 0 and max_nuc > 0:
                H_score = self.H_grid.clone()
                H_score[~candidate_mask] = 0.0
                K = min(max_nuc, n_candidates)
                _, topk_flat = H_score.view(-1).topk(K)

                for flat_idx in topk_flat:
                    fi = flat_idx.item()
                    i0 = fi // (n * n)
                    i1 = (fi % (n * n)) // n
                    i2 = fi % n
                    pos = torch.tensor(
                        [(i0 + 0.5) / n, (i1 + 0.5) / n, (i2 + 0.5) / n],
                        device=device, dtype=torch.float32)
                    self.crack_paths.append(pos.unsqueeze(0))
                    if (getattr(self, '_gravity_drop', False) and
                            getattr(self, '_gravity_drop_contacted', False)):
                        init_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
                        init_dir[:2] = torch.randn(2, device=device) * 0.3
                        init_dir = init_dir / init_dir.norm()
                        self.crack_dirs.append(init_dir)
                    else:
                        self.crack_dirs.append(None)
                    n_new += 1
                    if step < 50:
                        print(f"  [NUC] New crack at grid=({i0},{i1},{i2}) "
                              f"pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] "
                              f"H={self.H_grid[i0,i1,i2]:.1f}", flush=True)

        # 4) Advance crack tips
        self._advance_crack_tips(dx, n, device, crack_tip_speed, H_ref, step)

        # 5) AT2 phase field + geometric crack damage
        c_old = self.c_vol.clone()

        if step == 0 and self.c_vol.max() > 0:
            c_from_vol = self._bin_particles_to_grid(self.c_vol)
            self.c_grid = torch.maximum(self.c_grid, c_from_vol)
        at2_iters = 50 if step < 3 else 30
        self._solve_at2_phase_field(n_iters=at2_iters, crack_paths=None)
        self._gather_grid_to_particles()

        self._assign_crack_damage(crack_width=self.pf_params.get('crack_width', 0.004))
        self.c_vol = torch.maximum(self.c_vol, c_old)

        # 6) Diagnostics
        if step < 5 or step % 20 == 0:
            n_paths = len(self.crack_paths)
            total_pts = sum(p.shape[0] for p in self.crack_paths)
            max_len = max((p.shape[0] for p in self.crack_paths), default=0)
            n_cracked = (self.c_vol > 0.3).sum().item()
            dc = self.c_vol - c_old
            print(f"  [AT2 {step:3d}] paths={n_paths} pts={total_pts} "
                  f"max_seg={max_len} c_vol={self.c_vol.max():.4f} "
                  f"c_grid={self.c_grid.max():.4f}({(self.c_grid > 0.3).sum().item()}cells) "
                  f"cracked(>0.3)={n_cracked} dc_max={dc.max():.6f} "
                  f"H_max={self.H_grid.max():.2e} H_ref={H_ref:.2e} "
                  f"nuc_new={n_new}", flush=True)

        self._hybrid_step += 1

    @torch.no_grad()
    def _advance_crack_tips(self, dx, n, device, crack_tip_speed, H_ref, step):
        """Advance all crack tips with EMA direction smoothing and branching."""
        ema_alpha = 0.20
        min_step_dist = 0.3 * dx

        branch_angle = self.pf_params.get('branch_angle', 35.0)
        branch_min_len = self.pf_params.get('branch_min_length', 6)
        branch_prob = self.pf_params.get('branch_probability', 0.3)
        max_branches = self.pf_params.get('max_branches_per_path', 1)
        max_total_cracks = self.pf_params.get('max_total_cracks', 5)
        if not hasattr(self, '_branch_count'):
            self._branch_count = {}

        pending_branches = []

        for path_idx in range(len(self.crack_paths)):
            path = self.crack_paths[path_idx]
            tip = path[-1]

            gi = (tip * n).long().clamp(1, n - 2)
            i, j, k = gi[0].item(), gi[1].item(), gi[2].item()
            H_local = self.H_grid[i, j, k].item()

            # Direction from -grad(H) projected perpendicular to n1
            H_phys = self._H_grid_physics
            grad_H = torch.zeros(3, device=device)
            grad_H[0] = (H_phys[min(i+1, n-1), j, k] - H_phys[max(i-1, 0), j, k]) / (2 * dx)
            grad_H[1] = (H_phys[i, min(j+1, n-1), k] - H_phys[i, max(j-1, 0), k]) / (2 * dx)
            grad_H[2] = (H_phys[i, j, min(k+1, n-1)] - H_phys[i, j, max(k-1, 0)]) / (2 * dx)

            raw_dir = -grad_H
            if hasattr(self, '_n1_grid') and self._n1_grid is not None:
                n1 = self._n1_grid[i, j, k]
                n1_mag = n1.norm()
                if n1_mag > 1e-6:
                    n1 = n1 / n1_mag
                    raw_dir = grad_H - (grad_H * n1).sum() * n1

            raw_mag = raw_dir.norm()
            if raw_mag > 1e-8:
                raw_dir = raw_dir / raw_mag
            else:
                if self.crack_dirs[path_idx] is not None:
                    raw_dir = self.crack_dirs[path_idx]
                elif path.shape[0] >= 2:
                    raw_dir = path[-1] - path[-2]
                    if raw_dir.norm() < 1e-8:
                        continue
                    raw_dir = raw_dir / raw_dir.norm()
                else:
                    continue

            # EMA smoothing
            if self.crack_dirs[path_idx] is None:
                smooth_dir = raw_dir
            else:
                smooth_dir = (1.0 - ema_alpha) * self.crack_dirs[path_idx] + ema_alpha * raw_dir
                sm = smooth_dir.norm()
                smooth_dir = raw_dir if sm < 1e-8 else smooth_dir / sm
            self.crack_dirs[path_idx] = smooth_dir

            # Energy-proportional speed
            speed_scale = H_local / (H_ref + 1e-12)
            if speed_scale < 0.1:
                continue
            speed = min(crack_tip_speed * dx * min(speed_scale, 5.0), 4.0 * dx)
            new_tip = (tip + speed * smooth_dir).clamp(dx, 1.0 - dx)

            # Stay inside occupied grid
            ngi = (new_tip * n).long().clamp(0, n - 1)
            if not self.grid_occupied[ngi[0], ngi[1], ngi[2]]:
                found = False
                best_tip = None
                best_dot = -1.0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            alt_dir = torch.tensor([float(di), float(dj), float(dk)], device=device)
                            alt_dir = alt_dir / alt_dir.norm()
                            alt_tip = (tip + speed * alt_dir).clamp(dx, 1.0 - dx)
                            agi = (alt_tip * n).long().clamp(0, n - 1)
                            if self.grid_occupied[agi[0], agi[1], agi[2]]:
                                dot = (alt_dir * smooth_dir).sum()
                                if dot > best_dot:
                                    best_dot = dot
                                    best_tip = alt_tip
                                    found = True
                if found:
                    new_tip = best_tip
                    self.crack_dirs[path_idx] = (new_tip - tip) / ((new_tip - tip).norm() + 1e-8)
                else:
                    continue

            if (new_tip - tip).norm() > min_step_dist:
                self.crack_paths[path_idx] = torch.cat([path, new_tip.unsqueeze(0)], dim=0)

            # Branching
            n_branches_so_far = self._branch_count.get(path_idx, 0)
            path_len = self.crack_paths[path_idx].shape[0]
            can_branch = (path_len >= branch_min_len and
                          n_branches_so_far < max_branches and
                          len(self.crack_paths) + len(pending_branches) * 2 < max_total_cracks and
                          H_local > 0.3 * H_ref)
            if can_branch and torch.rand(1).item() < branch_prob:
                parent_dir = self.crack_dirs[path_idx]
                if parent_dir is not None:
                    dir1 = self._rotate_direction(parent_dir, branch_angle)
                    dir2 = self._rotate_direction(parent_dir, -branch_angle)
                    pending_branches.append((new_tip.clone(), dir1, dir2))
                    self._branch_count[path_idx] = n_branches_so_far + 1

        # Add branched paths
        for tip_pos, dir1, dir2 in pending_branches:
            self.crack_paths.append(tip_pos.unsqueeze(0))
            self.crack_dirs.append(dir1)
            self.crack_paths.append(tip_pos.unsqueeze(0))
            self.crack_dirs.append(dir2)
            if step < 50:
                print(f"  [BRANCH] New fork at [{tip_pos[0]:.3f},{tip_pos[1]:.3f},{tip_pos[2]:.3f}] "
                      f"angle=±{branch_angle}°", flush=True)

    # ================================================================
    # AT2 Phase Field PDE Solver
    # ================================================================

    @torch.no_grad()
    def _solve_at2_phase_field(self, n_iters: int = 30, crack_paths: list = None):
        """
        Solve AT2 phase field via Jacobi iteration.
        Euler-Lagrange: (2H + Gc/l0) c - Gc l0 Laplacian(c) = 2H
        """
        n = self.mpm.num_grids
        dx = self.mpm.dx
        device = self.c_grid.device

        Gc = getattr(self.elasticity, 'Gc', 30.0)
        l0 = getattr(self.elasticity, 'l0', 0.025)

        Gc_over_l0 = Gc / l0
        Gc_l0_over_dx2 = Gc * l0 / (dx * dx)
        H2 = 2.0 * self.H_grid
        diag = H2 + Gc_over_l0 + 6.0 * Gc_l0_over_dx2

        # Optional Dirichlet BC on crack paths
        crack_bc = None
        if crack_paths:
            crack_raw = torch.zeros(n, n, n, device=device)
            for path in crack_paths:
                gi = (path * n).long().clamp(0, n - 1)
                crack_raw[gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0
            crack_bc = torch.nn.functional.max_pool3d(
                crack_raw.unsqueeze(0).unsqueeze(0),
                kernel_size=3, stride=1, padding=1
            )[0, 0] > 0.5

        c_old = self.c_grid.clone()
        c = self.c_grid.clone()
        occ = self.grid_occupied.float()

        for _ in range(n_iters):
            cp = torch.nn.functional.pad(
                c.unsqueeze(0).unsqueeze(0),
                (1, 1, 1, 1, 1, 1), mode='replicate'
            )[0, 0]

            nbr_sum = (cp[2:, 1:-1, 1:-1] + cp[:-2, 1:-1, 1:-1] +
                       cp[1:-1, 2:, 1:-1] + cp[1:-1, :-2, 1:-1] +
                       cp[1:-1, 1:-1, 2:] + cp[1:-1, 1:-1, :-2])

            c = (H2 + Gc_l0_over_dx2 * nbr_sum) / (diag + 1e-12)
            c = c.clamp(0.0, 1.0) * occ
            c = torch.maximum(c, c_old)
            if crack_bc is not None:
                c[crack_bc] = 1.0

        self.c_grid = c

    @torch.no_grad()
    def _assign_crack_damage(self, crack_width: float = 0.025):
        """Assign geometric c=1 damage along crack paths (linear falloff)."""
        if not self.crack_paths:
            return

        positions = self.x_mpm
        min_dist = torch.full((positions.shape[0],), float('inf'), device=positions.device)

        for path in self.crack_paths:
            if path.shape[0] == 1:
                dist = (positions - path[0]).norm(dim=1)
            else:
                dist = self._point_to_polyline_dist(positions, path)
            min_dist = torch.minimum(min_dist, dist)

        c_new = (1.0 - (min_dist / crack_width)).clamp(0.0, 1.0)
        self.c_vol = torch.maximum(self.c_vol, c_new)

    # ================================================================
    # Geometry helpers
    # ================================================================

    @torch.no_grad()
    def _point_to_polyline_dist(self, points: Tensor, polyline: Tensor) -> Tensor:
        """Minimum distance from each point to any segment of a polyline."""
        N = points.shape[0]
        M = polyline.shape[0]
        if M < 2:
            return (points - polyline[0]).norm(dim=1)

        min_dist = torch.full((N,), float('inf'), device=points.device)
        a = polyline[:-1]
        b = polyline[1:]
        ab = b - a
        ab_len = ab.norm(dim=1)
        valid = ab_len > 1e-10
        if not valid.any():
            return (points - polyline[0]).norm(dim=1)

        a_v, ab_v, ab_len_v = a[valid], ab[valid], ab_len[valid]
        ab_norm_v = ab_v / ab_len_v.unsqueeze(1)

        for seg_idx in range(a_v.shape[0]):
            ap = points - a_v[seg_idx]
            t = (ap * ab_norm_v[seg_idx]).sum(dim=1).clamp(0.0, ab_len_v[seg_idx].item())
            closest = a_v[seg_idx] + t.unsqueeze(1) * ab_norm_v[seg_idx]
            dist = (points - closest).norm(dim=1)
            min_dist = torch.minimum(min_dist, dist)

        return min_dist

    @torch.no_grad()
    def _rotate_direction(self, direction: Tensor, angle_deg: float) -> Tensor:
        """Rotate direction by angle_deg using Rodrigues' formula."""
        device = direction.device
        if abs(direction[1].item()) < 0.9:
            up = torch.tensor([0.0, 1.0, 0.0], device=device)
        else:
            up = torch.tensor([1.0, 0.0, 0.0], device=device)
        axis = torch.linalg.cross(direction, up)
        axis = axis / (axis.norm() + 1e-8)

        angle = angle_deg * math.pi / 180.0
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = (direction * cos_a +
                   torch.linalg.cross(axis, direction) * sin_a +
                   axis * (axis @ direction) * (1.0 - cos_a))
        return rotated / (rotated.norm() + 1e-8)

    @torch.no_grad()
    # ================================================================
    # Physics step
    # ================================================================

    @torch.no_grad()
    def step_physics(self, dt: float):
        """Single MPM physics timestep with gravity drop handling."""
        if not hasattr(self, '_physics_step'):
            self._physics_step = 0

        # Phase 1: free-fall translation (no grid physics)
        if self._gravity_drop and not self._gravity_drop_contacted:
            g = self.mpm.gravity
            self._v_com = self._v_com + dt * g
            self.x_mpm = self.x_mpm + dt * self._v_com.unsqueeze(0)

            z_min = self.x_mpm[:, 2].min().item()
            step = self._physics_step
            if step < 50 or step % 10 == 0:
                z_max = self.x_mpm[:, 2].max().item()
                print(f"  [fall {step:3d}] v_com_z={self._v_com[2].item():.4f} "
                      f"z=[{z_min:.4f},{z_max:.4f}]", flush=True)

            if z_min <= self._gravity_drop_ground_z + 0.01:
                self._handle_ground_impact()

            self._physics_step += 1
            return

        # Phase 2: full MPM physics
        # Graduated damping: aggressive right after impact, taper to steady-state.
        # Simulates fracture energy absorption that geometric cracks don't capture.
        if self._gravity_drop and self._gravity_drop_contacted:
            frames_since = getattr(self, '_impact_frame_count', 0)
            if frames_since < 5:
                self.mpm.damping = 0.93 + 0.012 * frames_since
            else:
                self.mpm.damping = 0.999

        self._apply_seismic_loading(dt)

        stress = self.elasticity(self.F, c=self.c_vol)
        E = torch.exp(self.elasticity.log_E).item() if hasattr(self.elasticity, 'log_E') else 1e6
        stress = stress.clamp(-5.0 * E, 5.0 * E)

        self._last_stress = stress.detach()

        x_old = self.x_mpm.clone()

        if self.fragmentation_active and self.fragment_manager.n_fragments > 1:
            # Per-fragment p2g2p: each fragment runs physics independently
            for frag_idx in self.fragment_manager.fragment_particle_indices:
                if len(frag_idx) < self.fragment_manager.min_fragment_particles:
                    self.v_mpm[frag_idx] += dt * self.mpm.gravity.unsqueeze(0)
                    debris_v_limit = 5.0
                    self.v_mpm[frag_idx] = self.v_mpm[frag_idx].clamp(
                        -debris_v_limit, debris_v_limit)
                    self.x_mpm[frag_idx] += self.v_mpm[frag_idx] * dt
                    self.x_mpm[frag_idx] = self.x_mpm[frag_idx].clamp(
                        self.mpm.clip_bound, 1.0 - self.mpm.clip_bound)
                    continue
                self.x_mpm, self.v_mpm, self.C, self.F = self.mpm.p2g2p_subset(
                    self.x_mpm, self.v_mpm, self.C, self.F, stress, frag_idx)

            self.mpm.time += dt
        else:
            self.x_mpm, self.v_mpm, self.C, self.F = self.mpm.p2g2p(
                self.x_mpm, self.v_mpm, self.C, self.F, stress)

        # Velocity & F clamping
        v_limit = 0.4 * self.mpm.dx / dt
        if self._gravity_drop and self._gravity_drop_contacted:
            v_limit = min(v_limit, 80.0)
        self.v_mpm = self.v_mpm.clamp(-v_limit, v_limit)

        F_limit = 1.5 if (self._gravity_drop and self._gravity_drop_contacted) else 2.0
        self.F = self.F.clamp(-F_limit, F_limit)

        # Accumulate tension energy H
        if hasattr(self.elasticity, 'tension_energy_density'):
            psi = self.elasticity.tension_energy_density(self.F)
        else:
            I = torch.eye(3, device=self.F.device, dtype=self.F.dtype).unsqueeze(0)
            Cmat = self.F.transpose(1, 2) @ self.F
            Egl = 0.5 * (Cmat - I)
            psi = (Egl * Egl).sum(dim=(1, 2))

        Gc = getattr(self.elasticity, 'Gc', 30.0)
        l0 = getattr(self.elasticity, 'l0', 0.025)
        psi = torch.clamp(psi, min=0.0, max=50.0 * Gc / (2.0 * l0))
        self._last_psi = psi

        if not hasattr(self, '_history_H'):
            self._history_H = psi.clone()
        else:
            self._history_H = torch.maximum(self._history_H, psi)

        # CFL computation (always, for adaptive dt)
        s_max = stress.abs().max().item()
        density_eff = self.mpm.p_mass / self.mpm.vol + 1e-12
        c_wave = (s_max / density_eff) ** 0.5
        cfl = c_wave * dt / self.mpm.dx if c_wave > 0 else 0
        self._last_cfl = cfl

        # Clamp extreme velocities to prevent particle ejection
        v_max_limit = 10.0  # max allowed speed in normalized space
        v_mag = self.v_mpm.norm(dim=1)
        too_fast = v_mag > v_max_limit
        if too_fast.any():
            scale = v_max_limit / v_mag[too_fast].clamp(min=1e-8)
            self.v_mpm[too_fast] *= scale.unsqueeze(-1)

        # Diagnostics
        step = self._physics_step
        if step < 50 or step % 10 == 0:
            v_max = self.v_mpm.abs().max().item()
            F_min, F_max = self.F.min().item(), self.F.max().item()
            disp = (self.x_mpm - x_old).norm(dim=1).max().item()
            v_mean = self.v_mpm.mean(dim=0)
            KE = 0.5 * (self.v_mpm ** 2).sum().item() * self.mpm.p_mass
            print(f"  [step {step:3d}] |v|={v_max:.4f} F=[{F_min:.4f},{F_max:.4f}] "
                  f"|stress|={s_max:.2e} disp={disp:.6f} x=[{self.x_mpm.min().item():.4f},{self.x_mpm.max().item():.4f}] "
                  f"CFL~{cfl:.3f} v_com=[{v_mean[0]:.4f},{v_mean[1]:.4f},{v_mean[2]:.4f}] "
                  f"KE={KE:.4f}", flush=True)

        self._physics_step += 1

    def _handle_ground_impact(self):
        """Handle the moment of ground contact in gravity drop mode."""
        step = self._physics_step
        self._gravity_drop_contacted = True
        self.v_mpm[:] = self._v_com.unsqueeze(0)
        v_impact = self._v_com[2].item()
        z_min = self.x_mpm[:, 2].min().item()
        print(f"  [IMPACT] Ground contact at step {step}! "
              f"v_impact={v_impact:.3f} z_min={z_min:.4f}")

        # Reset deformation state
        N = self.F.shape[0]
        self.F = torch.eye(3, device=self.F.device).unsqueeze(0).expand(N, 3, 3).clone()
        self.C = torch.zeros_like(self.C)
        if hasattr(self, 'c_grid'):
            delattr(self, 'c_grid')
        if hasattr(self, 'grid_occupied'):
            delattr(self, 'grid_occupied')
        self.frame_count = 0
        self._hybrid_step = 0
        self._impact_frame_count = 0

        # Spherical H seed at impact zone
        z_vals = self.x_mpm[:, 2]
        z_min_val = z_vals.min().item()
        z_max_val = z_vals.max().item()
        obj_height = z_max_val - z_min_val

        z_threshold = z_min_val + obj_height * 0.05
        bottom_mask = z_vals < z_threshold
        impact_center = self.x_mpm[bottom_mask].mean(dim=0)

        self._impact_center = impact_center
        self._impact_radius = obj_height * 0.25

        Gc = getattr(self.elasticity, 'Gc', 200.0)
        l0 = getattr(self.elasticity, 'l0', 0.035)
        H_ref = Gc / (2.0 * l0)

        dist_from_impact = (self.x_mpm - impact_center.unsqueeze(0)).norm(dim=1)
        tight_radius = obj_height * 0.30
        H_tight = torch.exp(-0.5 * (dist_from_impact / tight_radius) ** 2) * (3.0 * H_ref)
        nuc_frac = self.pf_params.get('nucleation_fraction', 0.3)
        H_tight[H_tight < nuc_frac * H_ref] = 0.0

        if not hasattr(self, '_history_H'):
            self._history_H = H_tight
        else:
            self._history_H = torch.maximum(self._history_H, H_tight)

        print(f"  [IMPACT-H] Spherical seed: {(H_tight > 0.0).sum().item()} particles")
        print(f"    center=[{impact_center[0]:.3f},{impact_center[1]:.3f},{impact_center[2]:.3f}] "
              f"H_ref={H_ref:.1f}")

        # Create radial crack seeds
        dev = self.x_mpm.device
        n_radial = 8
        if not hasattr(self, 'crack_paths'):
            self.crack_paths = []
            self.crack_dirs = []
            self._branch_count = {}
        for _ in range(n_radial):
            self.crack_paths.append(impact_center.unsqueeze(0).clone())
            self.crack_dirs.append(None)
        print(f"  [IMPACT-CRACK] Created {n_radial} crack seeds at impact "
              f"(direction from physics -∇H)")

        # Post-impact adjustments
        post_g = -400.0
        g_vec = self.mpm.gravity.clone()
        g_vec[:] = 0.0
        g_vec[2] = post_g
        self.mpm.gravity = g_vec
        self.mpm.damping = 0.975
        self._post_impact_gravity_restored = False
        print(f"  [POST-IMPACT] gravity→[0,0,{post_g}] damping→0.975")

    # ================================================================
    # Rendering step
    # ================================================================

    def step_rendering(self) -> bool:
        """Full frame update: physics substeps + crack + damage projection + Gaussian update."""
        if self.crack_only:
            for _ in range(self.substeps):
                self.step_crack_only(self.mpm.dt)
        else:
            # Adaptive dt: subdivide timestep when CFL > threshold
            dt_base = self.mpm.dt  # original dt from config
            cfl_target = 0.4       # target CFL ceiling
            dt_min = dt_base / 8   # minimum dt (max 8x subdivision)
            for _ in range(self.substeps):
                dt_current = dt_base
                # Check if previous step had high CFL → preemptively subdivide
                last_cfl = getattr(self, '_last_cfl', 0.0)
                if last_cfl > cfl_target and self._gravity_drop_contacted:
                    # Scale dt to bring CFL to ~cfl_target
                    scale = cfl_target / (last_cfl + 1e-8)
                    dt_current = max(dt_base * scale, dt_min)
                    n_sub = max(1, int(math.ceil(dt_base / dt_current)))
                    dt_current = dt_base / n_sub
                    if n_sub > 1:
                        if not hasattr(self, '_adaptive_dt_logged') or self._physics_step % 20 == 0:
                            print(f"  [ADAPTIVE-DT] CFL={last_cfl:.3f} → {n_sub} sub-steps "
                                  f"(dt={dt_current:.2e})", flush=True)
                            self._adaptive_dt_logged = True
                    # Temporarily override mpm.dt and run sub-steps
                    orig_dt = self.mpm.dt
                    self.mpm.dt = dt_current
                    for _s in range(n_sub):
                        self.step_physics(dt_current)
                    self.mpm.dt = orig_dt  # restore
                else:
                    self.step_physics(dt_current)

            # ---- Rigid Body Velocity Projection (once per frame) ----
            # Suppress internal elastic wobble while preserving ballistic motion
            if self.fragmentation_active and self.fragment_manager.n_fragments > 1:
                rigid_alpha = 0.5
                p_mass = self.mpm.p_mass
                for frag_idx in self.fragment_manager.fragment_particle_indices:
                    if len(frag_idx) < self.fragment_manager.min_fragment_particles:
                        continue
                    x_frag = self.x_mpm[frag_idx]
                    v_frag = self.v_mpm[frag_idx]
                    com = x_frag.mean(dim=0)
                    v_com = v_frag.mean(dim=0)
                    r = x_frag - com.unsqueeze(0)
                    v_rel = v_frag - v_com.unsqueeze(0)
                    L = torch.cross(r, v_rel, dim=1).sum(dim=0) * p_mass
                    r_sq = (r * r).sum(dim=1)
                    I_tensor = torch.zeros(3, 3, device=r.device)
                    I_tensor[0, 0] = (r_sq - r[:, 0] ** 2).sum()
                    I_tensor[1, 1] = (r_sq - r[:, 1] ** 2).sum()
                    I_tensor[2, 2] = (r_sq - r[:, 2] ** 2).sum()
                    I_tensor[0, 1] = I_tensor[1, 0] = -(r[:, 0] * r[:, 1]).sum()
                    I_tensor[0, 2] = I_tensor[2, 0] = -(r[:, 0] * r[:, 2]).sum()
                    I_tensor[1, 2] = I_tensor[2, 1] = -(r[:, 1] * r[:, 2]).sum()
                    I_tensor *= p_mass
                    try:
                        omega = torch.linalg.solve(I_tensor, L)
                    except Exception:
                        omega = torch.zeros(3, device=r.device)
                    omega_exp = omega.unsqueeze(0).expand_as(r)
                    v_rigid = v_com.unsqueeze(0) + torch.cross(omega_exp, r, dim=1)
                    self.v_mpm[frag_idx] = rigid_alpha * v_rigid + (1.0 - rigid_alpha) * v_frag

            # Burst mode at impact for near-instantaneous brittle fracture
            if (self._gravity_drop and self._gravity_drop_contacted and
                    hasattr(self, '_impact_frame_count')):
                frames_since = self._impact_frame_count
                burst_schedule = {0: 50, 1: 20, 2: 10, 3: 5}
                if frames_since in burst_schedule:
                    burst_iters = burst_schedule[frames_since]
                    print(f"  [BURST] Impact frame+{frames_since}: "
                          f"running {burst_iters} crack iterations", flush=True)
                    for _ in range(burst_iters):
                        self.step_hybrid_crack(self.mpm.dt)
            else:
                self.step_hybrid_crack(self.mpm.dt)
        torch.cuda.empty_cache()

        # Restore gravity gradually after burst mode
        if (self._gravity_drop and self._gravity_drop_contacted
                and hasattr(self, '_impact_frame_count')
                and not getattr(self, '_post_impact_gravity_restored', True)):
            if self._impact_frame_count >= 10:
                self._post_impact_gravity_restored = True
                print(f"  [GRAVITY] Maintaining -300.0 (adaptive dt handles CFL)")

        # Fragment detection with upward impulse (Run 14 style)
        if (self.fragment_manager is not None
                and self._gravity_drop and self._gravity_drop_contacted
                and hasattr(self, '_impact_frame_count')
                and self._impact_frame_count in (1, 2, 3, 5, 10, 20, 40)
                and hasattr(self, 'grid_occupied')
                and hasattr(self, 'crack_paths') and len(self.crack_paths) > 0):
            n_frags = self._detect_fragments_from_crack_planes()
            if n_frags > 1:
                first_detection = not self.fragmentation_active
                self.fragmentation_active = True

                # Only apply separation impulse on FIRST detection (avoid repeated flinch)
                if first_detection and hasattr(self, '_impact_center'):
                    ic = self._impact_center.unsqueeze(0)  # [1, 3]
                    total_particles = sum(len(fi) for fi in self.fragment_manager.fragment_particle_indices)
                    for frag_idx in self.fragment_manager.fragment_particle_indices:
                        n_p = len(frag_idx)
                        if n_p < 10:
                            continue
                        com = self.x_mpm[frag_idx].mean(dim=0, keepdim=True)
                        direction = com - ic
                        dist = direction.norm() + 1e-8
                        direction = direction / dist
                        # Add upward component (+z) for bounce effect
                        direction[0, 2] += 0.6
                        direction = direction / (direction.norm() + 1e-8)
                        # Scale impulse inversely with fragment size
                        size_ratio = n_p / (total_particles + 1e-8)
                        impulse_strength = 2.0 * max(0.3, min(1.0, size_ratio * 5.0))
                        self.v_mpm[frag_idx] += impulse_strength * direction
                    print(f"  [SEPARATION] Applied impulse (w/ upward) to {n_frags} fragments")

        # Project damage to surface and update Gaussians
        x_surf_mpm = self.x_mpm[self.surface_mask]
        c_surf = self.damage_mapper.project_damage(
            self.c_vol, self.x_mpm, x_surf_mpm, self.surface_mask)
        x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)

        # Extract deformation gradient for surface particles
        F_surf = self.F[self.surface_mask]  # (N_surf, 3, 3)

        # Direct PLY mode: compute displacement-based positions
        if self._ply_direct:
            # Displacement of each surface particle from its initial position
            disp = x_surf_world - self._surf_init_world
            # Apply displacement to each PLY Gaussian via its mapped surface particle
            ply_disp = disp[self._ply_to_surface]
            x_ply_world = self._ply_init_xyz + ply_disp
            # Map damage to PLY Gaussians via surface mapping
            c_ply = c_surf[self._ply_to_surface]
            # Map deformation gradient to PLY Gaussians
            F_ply = F_surf[self._ply_to_surface]
        else:
            x_ply_world = x_surf_world
            c_ply = c_surf
            F_ply = F_surf

        # Build debris mask: small fragments hidden before visualization
        debris_mask = None
        if (self.fragmentation_active
                and self.fragment_manager is not None
                and self.fragment_manager.n_fragments > 1):
            frag_ids = self.fragment_manager.fragment_ids
            surf_frag_ids = frag_ids[self.surface_mask]
            if self._ply_direct:
                # Map debris mask from surface to PLY Gaussians
                surf_debris = torch.zeros(x_surf_world.shape[0],
                                          dtype=torch.bool, device=x_surf_world.device)
                for frag_idx in self.fragment_manager.fragment_particle_indices:
                    if (len(frag_idx) == 0
                            or len(frag_idx) >= self.fragment_manager.min_fragment_particles):
                        continue
                    frag_label = frag_ids[frag_idx[0]].item()
                    surf_debris |= (surf_frag_ids == frag_label)
                debris_mask = surf_debris[self._ply_to_surface]
            else:
                debris_mask = torch.zeros(x_surf_world.shape[0],
                                          dtype=torch.bool, device=x_surf_world.device)
                for frag_idx in self.fragment_manager.fragment_particle_indices:
                    if (len(frag_idx) == 0
                            or len(frag_idx) >= self.fragment_manager.min_fragment_particles):
                        continue
                    frag_label = frag_ids[frag_idx[0]].item()
                    debris_mask |= (surf_frag_ids == frag_label)

        self.visualizer.update_gaussians(
            self.gaussians, c_ply, x_ply_world,
            preserve_original=True,
            debris_mask=debris_mask,
            F_per_gaussian=F_ply,
            camera_pos=getattr(self, '_camera_pos', None))

        self._save_diagnostics_if_needed()

        self.frame_count += 1
        if hasattr(self, '_impact_frame_count'):
            self._impact_frame_count += 1
        return True

    # ================================================================
    # Fragment detection
    # ================================================================

    @torch.no_grad()
    def _detect_fragments_from_crack_planes(self) -> int:
        """Build planar crack surfaces from polylines and run CC.

        Each crack path defines a cutting plane:
        - The plane passes through the crack polyline
        - It extends vertically (in Z) and radially from the impact center
        - Grid cells near the plane are marked as cracked
        - CC on the remaining material gives fragments
        """
        n = self.mpm.num_grids
        dx = 1.0 / n
        device = self.x_mpm.device

        # Impact center for radial direction computation
        impact_center = getattr(self, '_impact_center',
                                self.x_mpm.mean(dim=0))
        ic_xy = impact_center[:2]  # XY components

        # Grid cell centers
        coords = (torch.arange(n, device=device).float() + 0.5) / n

        damage_grid = torch.zeros(n, n, n, device=device)
        plane_thickness = 2.5 * dx  # ~2.5 cells wide barrier

        # Group crack paths by unique radial angle to avoid duplicate planes
        seen_angles = []
        unique_planes = []

        for path in self.crack_paths:
            if path.shape[0] < 2:
                continue
            # Compute average radial direction in XY from impact center
            path_center = path.mean(dim=0)
            radial = path_center[:2] - ic_xy
            if radial.norm() < 1e-6:
                continue
            radial = radial / radial.norm()
            angle = torch.atan2(radial[1], radial[0]).item()

            # Skip near-duplicate angles (within 5 degrees)
            is_dup = False
            for sa in seen_angles:
                diff = abs(angle - sa)
                diff = min(diff, 2 * 3.14159 - diff)
                if diff < 0.087:  # ~5 degrees
                    is_dup = True
                    break
            if is_dup:
                continue
            seen_angles.append(angle)

            # Plane normal = perpendicular to radial in XY
            # normal = (-radial_y, radial_x, 0)
            normal_xy = torch.tensor([-radial[1].item(), radial[0].item()],
                                     device=device)

            # Mark grid cells close to this plane
            # Signed distance of cell (i,j,k) to the plane passing through
            # impact_center with normal (normal_xy, 0):
            # d = (cell_xy - ic_xy) · normal_xy
            # Mark if |d| < plane_thickness/2
            cell_x = coords  # (n,)
            cell_y = coords  # (n,)
            # Compute signed distance for all XY positions
            dist_x = cell_x.unsqueeze(1) - ic_xy[0]  # (n, 1)
            dist_y = cell_y.unsqueeze(0) - ic_xy[1]  # (1, n)
            signed_dist = dist_x * normal_xy[0] + dist_y * normal_xy[1]  # (n, n)
            plane_mask_2d = signed_dist.abs() < (plane_thickness / 2)  # (n, n)

            # Limit Z-extent to crack path range + margin (not all Z)
            z_vals = path[:, 2]
            z_lo = max(0.0, z_vals.min().item() - 0.08)
            z_hi = min(1.0, z_vals.max().item() + 0.15)
            z_lo_idx = max(0, int(z_lo * n))
            z_hi_idx = min(n, int(z_hi * n) + 1)
            plane_mask_3d = plane_mask_2d.unsqueeze(2).expand(n, n, n).clone()
            plane_mask_3d[:, :, :z_lo_idx] = False
            plane_mask_3d[:, :, z_hi_idx:] = False
            damage_grid[plane_mask_3d] = 1.0

            unique_planes.append(angle)

        # Restrict crack to material region only
        damage_grid = damage_grid * self.grid_occupied.float()

        n_crack_cells = (damage_grid > 0.5).sum().item()
        n_occupied = self.grid_occupied.sum().item()
        print(f"  [FRAGMENT] Built {len(unique_planes)} crack planes, "
              f"{n_crack_cells}/{n_occupied} cells marked as crack")

        n_frags = self.fragment_manager.detect_fragments(
            damage_grid, self.grid_occupied, self.x_mpm, self.mpm)

        print(f"  [FRAGMENT] CC detected {n_frags} fragments")
        for k in range(min(n_frags, 20)):
            nk = len(self.fragment_manager.fragment_particle_indices[k])
            print(f"    Fragment {k}: {nk} particles")

        return n_frags

    # ================================================================
    # Diagnostics & utilities
    # ================================================================

    def _save_diagnostics_if_needed(self):
        """Save particle physics data at key frames."""
        if not getattr(self, '_save_diagnostics', False):
            return
        diag_frames = getattr(self, '_diag_frames', set())
        frame = getattr(self, '_render_frame', self.frame_count)
        if frame not in diag_frames:
            return

        import numpy as np
        diag_dir = os.path.join(getattr(self, '_output_dir', 'output'), 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)

        data = {
            'positions': self.x_mpm.detach().cpu().numpy(),
            'frame': frame,
            'c': self.c_vol.detach().cpu().numpy(),
        }
        if hasattr(self, '_history_H'):
            data['H'] = self._history_H.detach().cpu().numpy()
        if hasattr(self, '_last_psi'):
            data['psi_plus'] = self._last_psi.detach().cpu().numpy()
        if hasattr(self, '_last_stress') and self._last_stress is not None:
            s = self._last_stress.detach()
            tr = s[:, 0, 0] + s[:, 1, 1] + s[:, 2, 2]
            s_dev = s - (tr / 3.0).unsqueeze(1).unsqueeze(2) * torch.eye(3, device=s.device)
            vm = torch.sqrt(1.5 * (s_dev * s_dev).sum(dim=(1, 2)).clamp(min=0))
            data['von_mises'] = vm.cpu().numpy()

        path = os.path.join(diag_dir, f'diag_frame_{frame:04d}.npz')
        np.savez_compressed(path, **data)
        print(f"  [DIAG] Saved diagnostics at frame {frame}: {path}", flush=True)

    def apply_external_force(self, force_center: Tensor, force_magnitude: float,
                             force_radius: float, force_direction: Optional[Tensor] = None,
                             surface_only: bool = True):
        """Apply localized external force impulse to MPM particles."""
        force_center_mpm = self.mapper.world_to_mpm(force_center.unsqueeze(0)).squeeze(0)
        force_radius_mpm = self.mapper.scale_world_to_mpm(force_radius)
        dists = (self.x_mpm - force_center_mpm).norm(dim=1)
        influence = torch.exp(-dists**2 / (2 * force_radius_mpm**2))

        if surface_only:
            influence = influence * self.surface_mask.float()

        if force_direction is None:
            directions = self.x_mpm - force_center_mpm
            directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-12)
        else:
            force_direction_mpm = force_direction / (force_direction.norm() + 1e-12)
            directions = force_direction_mpm.unsqueeze(0).expand_as(self.x_mpm)

        self.v_mpm += directions * influence.unsqueeze(1) * force_magnitude

        n_affected = (influence > 0.01).sum().item()
        print(f"[HybridSimulator] Applied external force:")
        print(f"  - Center (MPM): {force_center_mpm.detach().cpu().numpy()}")
        print(f"  - Magnitude: {force_magnitude}, Affected: {n_affected}/{self.x_mpm.shape[0]}")

    def detect_large_deformation(self, threshold: float = 0.3) -> bool:
        """Check if particles have moved beyond threshold from initial positions."""
        if self.init_positions is None:
            return False
        return (self.x_mpm - self.init_positions).norm(dim=1).max().item() > threshold

    def get_statistics(self) -> Dict:
        """Return current simulation statistics."""
        current_time = time.time()
        elapsed = current_time - self.last_render_time
        fps = 1.0 / (elapsed + 1e-6)
        self.last_render_time = current_time

        return {
            "frame": self.frame_count,
            "time": self.mpm.time,
            "c_max": self.c_vol.max().item(),
            "c_mean": self.c_vol.mean().item(),
            "c_surface_max": self.c_vol[self.surface_mask].max().item(),
            "c_surface_mean": self.c_vol[self.surface_mask].mean().item(),
            "n_cracked": (self.c_vol > 0.3).sum().item(),
            "n_particles": self.x_mpm.shape[0],
            "fps": fps,
        }

    def save_state(self, path: str):
        """Save simulation state to disk."""
        torch.save({
            "frame": self.frame_count,
            "x_mpm": self.x_mpm, "v_mpm": self.v_mpm,
            "F": self.F, "C": self.C, "c_vol": self.c_vol,
            "gaussian_xyz": self.gaussians._xyz,
            "gaussian_opacity": self.gaussians._opacity,
            "gaussian_features_dc": self.gaussians._features_dc,
        }, path)
        print(f"[HybridSimulator] State saved to {path}")

    def load_state(self, path: str):
        """Load simulation state from disk."""
        checkpoint = torch.load(path)
        self.frame_count = checkpoint["frame"]
        self.x_mpm = checkpoint["x_mpm"]
        self.v_mpm = checkpoint["v_mpm"]
        self.F = checkpoint["F"]
        self.C = checkpoint["C"]
        self.c_vol = checkpoint["c_vol"]
        self.gaussians._xyz.data = checkpoint["gaussian_xyz"]
        self.gaussians._opacity.data = checkpoint["gaussian_opacity"]
        self.gaussians._features_dc.data = checkpoint["gaussian_features_dc"]
        print(f"[HybridSimulator] State loaded from {path} (frame {self.frame_count})")

    def __repr__(self) -> str:
        return (f"HybridCrackSimulator(particles={self.x_mpm.shape[0] if self.x_mpm is not None else 'N/A'}, "
                f"surface={self.surface_mask.sum().item()}, frame={self.frame_count})")
