"""
Manifold Simulator

Gaussian-manifold fracture pipeline replacing HybridCrackSimulator.

Architecture:
    MPM physics → PhysicsProjector → GaussianFractureField → GaussianUpdater
                                    → GaussianSplitter
                                    → GraphFragmentManager

Key difference from HybridCrackSimulator:
    - Fracture state lives ON Gaussians, not on a volumetric grid
    - No damage_mapper (volume→surface projection) needed
    - Graph-based Laplacian instead of grid Laplacian
    - Crack normals and opening are first-class state
"""

import torch
from torch import Tensor
from typing import Dict, Optional
import time
import math

from src.mpm_core.mpm_model import MPMModel
from src.core.coordinate_mapper import CoordinateMapper
from src.fracture.graph_builder import GaussianGraph
from src.fracture.physics_projector import PhysicsProjector
from src.fracture.tip_based_fracture_field import GaussianFractureField
from src.fracture.gaussian_splitter import GaussianSplitter
from src.fracture.graph_fragment_manager import GraphFragmentManager


class ManifoldSimulator:
    """
    Gaussian-manifold fracture simulator.

    Per-frame flow:
        1. MPM physics → updated (x, v, F)
        2. PhysicsProjector: ψ⁺, n₁, F → Gaussians
        3. GaussianFractureField: damage evolution on graph
        4. GaussianSplitter: crack opening + split
        5. GraphFragmentManager: fragment detection
        6. GaussianUpdater: positions, lighting, deformation
    """

    def __init__(
        self,
        mpm_model: MPMModel,
        gaussians,
        elasticity_module,
        coord_mapper: CoordinateMapper,
        visualizer,
        surface_mask: Tensor,
        physics_substeps: int = 10,
        fracture_params: Optional[Dict] = None,
        simulation_mode: str = "deformation",
        seismic_params: Optional[Dict] = None,
    ):
        self.mpm = mpm_model
        self.gaussians = gaussians
        self.elasticity = elasticity_module
        self.mapper = coord_mapper
        self.visualizer = visualizer
        self.surface_mask = surface_mask
        self.substeps = physics_substeps
        self.simulation_mode = simulation_mode

        self.seismic = seismic_params or {}
        self.seismic_enabled = self.seismic.get("enabled", False)

        fp = fracture_params or {}
        self.fracture_cfg = fp
        device_str = str(next(iter(mpm_model.parameters())).device
                         if hasattr(mpm_model, 'parameters')
                         else 'cuda')

        # --- Fracture components ---
        Gc = getattr(elasticity_module, 'Gc', fp.get('Gc', 60000.0))
        l0 = getattr(elasticity_module, 'l0', fp.get('l0', 0.025))

        self.graph = GaussianGraph(
            k=fp.get('graph_k', 12),
            sigma=fp.get('graph_sigma', 0.03),
            rebuild_every=fp.get('graph_rebuild_every', 5),
            device=device_str,
        )

        self.physics_projector = PhysicsProjector(
            k_neighbors=fp.get('projector_k', 8),
            influence_radius=fp.get('projector_sigma', 0.05),
            device=device_str,
        )

        self.fracture_field = GaussianFractureField(
            Gc=Gc,
            l0=l0,
            dC_max=fp.get('dC_max', 0.015),
            warmup_frames=fp.get('warmup_frames', 5),
            aniso_ratio=fp.get('aniso_ratio', 3.0),
            opening_scale=fp.get('opening_scale', 0.02),
            damage_source_scale=fp.get('damage_source_scale', 0.35),
            damage_spread=fp.get('damage_spread', 0.18),
            drive_quantile=fp.get('drive_quantile', 0.90),
            front_threshold=fp.get('front_threshold', 0.05),
            radial_bias=fp.get('radial_bias', 2.5),
            tip_propagation_scale=fp.get('tip_propagation_scale', 0.75),
            graph=self.graph,
            device=device_str,
        )

        self.splitter = GaussianSplitter(
            split_threshold=fp.get('split_threshold', 0.8),
            opacity_threshold=fp.get('opacity_threshold', 0.6),
            flatten_threshold=fp.get('flatten_threshold', 0.3),
            max_flatten=fp.get('max_flatten', 0.5),
            max_opacity_reduction=fp.get('max_opacity_reduction', 0.8),
            split_offset_scale=fp.get('split_offset_scale', 1.5),
            device=device_str,
        )

        frag_enabled = fp.get('fragmentation_enabled', False)
        self.fragment_manager = GraphFragmentManager(
            damage_threshold=fp.get('fragment_damage_threshold', 0.5),
            min_fragment_size=fp.get('min_fragment_particles', 20),
            device=device_str,
        ) if frag_enabled else None
        self.fragmentation_active = False

        self.damage_feedback_delay_frames = int(
            fp.get('damage_feedback_delay_frames', 8))
        self.damage_feedback_ramp_frames = int(
            fp.get('damage_feedback_ramp_frames', 6))
        self.interior_damage_scale = float(
            fp.get('interior_damage_scale', 0.2))
        self.drive_tension_weight = float(
            fp.get('drive_tension_weight', 1.0))
        self.drive_shear_weight = float(
            fp.get('drive_shear_weight', 0.35))
        self.drive_principal_weight = float(
            fp.get('drive_principal_weight', 0.25))
        self.drive_kinetic_weight = float(
            fp.get('drive_kinetic_weight', 0.15))

        # Enable Gaussian splitting
        self.splitting_enabled = fp.get('splitting_enabled', False)

        # --- MPM state ---
        self.x_mpm = None
        self.v_mpm = None
        self.F = None
        self.C = None

        # --- Gravity drop ---
        self._gravity_drop = False
        self._gravity_drop_ground_z = 0.1
        self._gravity_drop_contacted = False
        self._v_com = None

        self.frame_count = 0
        self._physics_step = 0
        self._last_cfl = 0.0
        self._last_stress = None
        self._render_frame = 0
        self.init_positions = None

        print(f"\n{'='*60}")
        print(f"ManifoldSimulator Initialized")
        print(f"{'='*60}")
        print(f"  Mode: {simulation_mode}")
        print(f"  Particles: {surface_mask.shape[0]}")
        print(f"  Surface: {surface_mask.sum().item()}")
        print(f"  Substeps: {physics_substeps}")
        print(f"  Fracture: Gc={Gc}, l0={l0}")
        print(f"  Graph: k={self.graph.k}, sigma={self.graph.sigma}")
        print(f"  Damage feedback: delay={self.damage_feedback_delay_frames}, "
              f"ramp={self.damage_feedback_ramp_frames}, "
              f"interior_scale={self.interior_damage_scale:.2f}")
        print(f"  Splitting: {'ON' if self.splitting_enabled else 'OFF'}")
        print(f"  Fragmentation: {'ON' if frag_enabled else 'OFF'}")
        if self.seismic_enabled:
            print(f"  Seismic: amp={self.seismic.get('amplitude')}, "
                  f"freq={self.seismic.get('frequency')}Hz")
        print(f"{'='*60}\n")

    # ================================================================
    # Initialization
    # ================================================================

    def enable_gravity_drop(self, ground_z: float = 0.1):
        """Enable 2-phase gravity drop mode."""
        self._gravity_drop = True
        self._gravity_drop_ground_z = ground_z
        self._gravity_drop_contacted = False
        self._v_com = torch.zeros(3, device=self.mpm.gravity.device)
        print(f"[GravityDrop] Enabled. Ground at z={ground_z}")

    def initialize(self, init_positions: Tensor):
        """Initialize simulation state from particle positions in [0,1]^3."""
        device = init_positions.device
        N = init_positions.shape[0]

        self.x_mpm = init_positions.clone()
        self.v_mpm = torch.zeros((N, 3), device=device)
        self.F = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3).clone()
        self.C = torch.zeros((N, 3, 3), device=device)
        self.init_positions = init_positions.clone()

        # Initialize fracture field for surface Gaussians
        N_surf = self.surface_mask.sum().item()
        self.fracture_field.initialize(N_surf)

        # Set initial Gaussian positions
        x_surf_mpm = self.x_mpm[self.surface_mask]
        x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)

        self._ply_direct = getattr(self.gaussians, '_ply_direct_mode', False)
        if self._ply_direct:
            ply_to_surf = self.gaussians._ply_to_surface
            self._ply_to_surface = (torch.from_numpy(ply_to_surf).long().to(device)
                                    if not isinstance(ply_to_surf, Tensor)
                                    else ply_to_surf.long().to(device))
            self._ply_init_xyz = self.gaussians._xyz.data.clone()
            self._surf_init_world = x_surf_world.clone()
        else:
            self.gaussians._xyz.data = x_surf_world

        self.frame_count = 0
        self._physics_step = 0
        self._surface_normals = None  # set via set_surface_normals()
        print(f"[ManifoldSim] Initialized: {N} particles, {N_surf} surface")

    def set_surface_normals(self, all_normals: Tensor):
        """Store surface normals and pass to the graph builder.

        Args:
            all_normals: (N_total, 3) normals for ALL particles
                         (surface subset is extracted via surface_mask)
        """
        surf_normals = all_normals[self.surface_mask]
        surf_normals = torch.nn.functional.normalize(surf_normals, dim=-1)
        self._surface_normals = surf_normals
        self.graph.set_normals(surf_normals)
        print(f"[ManifoldSim] Surface normals set: {surf_normals.shape[0]} vectors")

    # ================================================================
    # Main frame update
    # ================================================================

    @torch.no_grad()
    def step_rendering(self) -> bool:
        """Full frame: physics → fracture → rendering."""

        # --- Physics substeps ---
        dt_base = self.mpm.dt
        cfl_target = 0.4
        dt_min = dt_base / 8

        for _ in range(self.substeps):
            dt_current = dt_base
            last_cfl = self._last_cfl

            if (last_cfl > cfl_target
                    and self._gravity_drop_contacted):
                scale = cfl_target / (last_cfl + 1e-8)
                dt_current = max(dt_base * scale, dt_min)
                n_sub = max(1, int(math.ceil(dt_base / dt_current)))
                dt_current = dt_base / n_sub
                orig_dt = self.mpm.dt
                self.mpm.dt = dt_current
                for _ in range(n_sub):
                    self._step_physics(dt_current)
                self.mpm.dt = orig_dt
            else:
                self._step_physics(dt_current)

        torch.cuda.empty_cache()

        # --- Fracture update on Gaussian manifold ---
        self._step_fracture()

        # --- Fragment detection ---
        if (self.fragment_manager is not None
                and self._gravity_drop_contacted
                and self.frame_count > 0
                and self.frame_count % 5 == 0):
            self._detect_fragments()

        # --- Update Gaussians for rendering ---
        self._update_gaussians()

        self.frame_count += 1
        if hasattr(self, '_impact_frame_count'):
            self._impact_frame_count += 1
        return True

    # ================================================================
    # Physics step (MPM)
    # ================================================================

    @torch.no_grad()
    def _step_physics(self, dt: float):
        """Single MPM physics timestep."""
        # Phase 1: free-fall (no grid physics)
        if self._gravity_drop and not self._gravity_drop_contacted:
            g = self.mpm.gravity
            self._v_com = self._v_com + dt * g
            self.x_mpm = self.x_mpm + dt * self._v_com.unsqueeze(0)

            z_min = self.x_mpm[:, 2].min().item()
            step = self._physics_step
            if step < 50 or step % 10 == 0:
                print(f"  [fall {step:3d}] v_com_z={self._v_com[2].item():.4f} "
                      f"z_min={z_min:.4f}")

            if z_min <= self._gravity_drop_ground_z + 0.01:
                self._handle_ground_impact()

            self._physics_step += 1
            return

        # Phase 2: full MPM physics
        if self._gravity_drop and self._gravity_drop_contacted:
            frames_since = getattr(self, '_impact_frame_count', 0)
            if frames_since < 5:
                self.mpm.damping = 0.93 + 0.012 * frames_since
            else:
                self.mpm.damping = 0.999

        self._apply_seismic_loading(dt)

        # Compute stress with damage degradation
        c_vol = self._get_volumetric_damage()
        stress = self.elasticity(self.F, c=c_vol)
        E = (torch.exp(self.elasticity.log_E).item()
             if hasattr(self.elasticity, 'log_E') else 1e6)
        stress = stress.clamp(-5.0 * E, 5.0 * E)
        self._last_stress = stress.detach()

        # MPM P2G2P
        if (self.fragmentation_active
                and self.fragment_manager is not None
                and self.fragment_manager.n_fragments > 1):
            self._step_fragmented_physics(stress, dt)
        else:
            self.x_mpm, self.v_mpm, self.C, self.F = self.mpm.p2g2p(
                self.x_mpm, self.v_mpm, self.C, self.F, stress)

        # Velocity & F clamping
        v_limit = 0.4 * self.mpm.dx / dt
        if self._gravity_drop and self._gravity_drop_contacted:
            v_limit = min(v_limit, 80.0)
        self.v_mpm = self.v_mpm.clamp(-v_limit, v_limit)
        self.F = self.F.clamp(-1.5 if self._gravity_drop_contacted else -2.0,
                               1.5 if self._gravity_drop_contacted else 2.0)

        # CFL
        s_max = stress.abs().max().item()
        density_eff = self.mpm.p_mass / self.mpm.vol + 1e-12
        c_wave = (s_max / density_eff) ** 0.5
        self._last_cfl = c_wave * dt / self.mpm.dx if c_wave > 0 else 0

        # Speed limit
        v_mag = self.v_mpm.norm(dim=1)
        too_fast = v_mag > 10.0
        if too_fast.any():
            scale = 10.0 / v_mag[too_fast].clamp(min=1e-8)
            self.v_mpm[too_fast] *= scale.unsqueeze(-1)

        step = self._physics_step
        if step < 50 or step % 10 == 0:
            print(f"  [phys {step:3d}] |v|={self.v_mpm.abs().max():.4f} "
                  f"|stress|={s_max:.2e} CFL~{self._last_cfl:.3f}")

        self._physics_step += 1

    def _get_volumetric_damage(self) -> Tensor:
        """
        Project Gaussian fracture damage back to volumetric particles
        for stress degradation in the MPM elasticity model.

        This is the "reverse" projection: Gaussian damage → MPM particles.
        Uses surface_mask to map surface Gaussians back.
        """
        N = self.x_mpm.shape[0]
        c_vol = torch.zeros(N, device=self.x_mpm.device)

        if self.fracture_field.c is None:
            return c_vol

        feedback_scale = 1.0
        if self._gravity_drop_contacted:
            frames_since = getattr(self, '_impact_frame_count', 0)
            if frames_since < self.damage_feedback_delay_frames:
                feedback_scale = 0.0
            elif self.damage_feedback_ramp_frames > 0:
                feedback_scale = min(
                    (frames_since - self.damage_feedback_delay_frames + 1)
                    / float(self.damage_feedback_ramp_frames),
                    1.0,
                )

        # Surface particles get direct damage from their Gaussian
        c_surf = self.fracture_field.c * feedback_scale
        N_surf = c_surf.shape[0]
        surf_indices = torch.where(self.surface_mask)[0]

        # Handle size mismatch (from Gaussian splitting)
        n_assign = min(N_surf, surf_indices.shape[0])
        c_vol[surf_indices[:n_assign]] = c_surf[:n_assign]

        # Interior particles: KNN from surface damage (light-weight)
        interior_mask = ~self.surface_mask
        if interior_mask.any() and c_surf.max() > 0.01:
            x_int = self.x_mpm[interior_mask]
            x_surf = self.x_mpm[self.surface_mask]
            # Simple nearest-neighbor (not full KNN)
            if x_int.shape[0] > 0 and x_surf.shape[0] > 0:
                dists = torch.cdist(x_int, x_surf[:n_assign])
                nearest = dists.argmin(dim=1)
                c_vol[interior_mask] = (
                    c_surf[:n_assign][nearest] * self.interior_damage_scale
                )

        return c_vol

    def _step_fragmented_physics(self, stress: Tensor, dt: float):
        """Per-fragment MPM physics."""
        # Map Gaussian fragments back to MPM particles
        surf_frag_ids = self.fragment_manager.fragment_ids
        mpm_frag_ids = torch.zeros(self.x_mpm.shape[0], dtype=torch.long,
                                    device=self.x_mpm.device)
        surf_indices = torch.where(self.surface_mask)[0]
        n_assign = min(surf_frag_ids.shape[0], surf_indices.shape[0])
        mpm_frag_ids[surf_indices[:n_assign]] = surf_frag_ids[:n_assign]

        # Interior particles: assign to nearest surface fragment
        interior_mask = ~self.surface_mask
        if interior_mask.any():
            x_int = self.x_mpm[interior_mask]
            x_surf = self.x_mpm[self.surface_mask][:n_assign]
            dists = torch.cdist(x_int, x_surf)
            nearest = dists.argmin(dim=1)
            mpm_frag_ids[interior_mask] = surf_frag_ids[:n_assign][nearest]

        # Per-fragment P2G2P
        for frag_id in range(self.fragment_manager.n_fragments):
            frag_mask = mpm_frag_ids == frag_id
            frag_idx = torch.where(frag_mask)[0]
            if len(frag_idx) < 10:
                self.v_mpm[frag_idx] += dt * self.mpm.gravity.unsqueeze(0)
                self.x_mpm[frag_idx] += self.v_mpm[frag_idx] * dt
                self.x_mpm[frag_idx] = self.x_mpm[frag_idx].clamp(
                    self.mpm.clip_bound, 1.0 - self.mpm.clip_bound)
                continue
            self.x_mpm, self.v_mpm, self.C, self.F = self.mpm.p2g2p_subset(
                self.x_mpm, self.v_mpm, self.C, self.F, stress, frag_idx)

        self.mpm.time += dt

    # ================================================================
    # Fracture step (Gaussian manifold)
    # ================================================================

    @staticmethod
    def _robust_normalize(values: Tensor, quantile: float = 0.95) -> Tensor:
        """Normalize a nonnegative field by a robust upper quantile."""
        q = float(min(max(quantile, 0.5), 0.999))
        scale = torch.quantile(values.detach(), q).clamp(min=1e-8)
        return (values / scale).clamp(0.0, 2.0)

    def _build_effective_fracture_drive(
        self,
        psi_mpm: Tensor,
        stress: Optional[Tensor],
    ) -> Tensor:
        """
        Build a fracture drive that is not locked to pure contact tension.

        Mix tensile energy with deviatoric stress, principal tensile stress,
        and relative kinetic activity so impact waves can continue driving
        a narrow crack band after the initial contact patch.
        """
        drive = self.drive_tension_weight * self._robust_normalize(psi_mpm, 0.90)

        if stress is not None:
            S = 0.5 * (stress + stress.transpose(1, 2))
            S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e8, 1e8)
            tr = torch.diagonal(S, dim1=1, dim2=2).sum(dim=1) / 3.0
            I = torch.eye(3, device=S.device).unsqueeze(0)
            dev = S - tr.view(-1, 1, 1) * I
            von_mises = torch.sqrt(
                torch.clamp(1.5 * (dev ** 2).sum(dim=(1, 2)), min=0.0))
            try:
                sigma1 = torch.linalg.eigvalsh(S)[:, -1].clamp(min=0.0)
            except RuntimeError:
                sigma1 = torch.diagonal(S, dim1=1, dim2=2).max(dim=1).values.clamp(min=0.0)

            drive = drive + (
                self.drive_shear_weight
                * self._robust_normalize(von_mises, 0.92)
            )
            drive = drive + (
                self.drive_principal_weight
                * self._robust_normalize(sigma1, 0.92)
            )

        if self.v_mpm is not None:
            v_rel = self.v_mpm - self.v_mpm.mean(dim=0, keepdim=True)
            speed_rel = v_rel.norm(dim=1)
            drive = drive + (
                self.drive_kinetic_weight
                * self._robust_normalize(speed_rel, 0.95)
            )

        return drive

    @staticmethod
    def _safe_vector_normalize(vectors: Tensor) -> Tensor:
        norm = vectors.norm(dim=1, keepdim=True)
        return torch.where(
            norm > 1e-8,
            vectors / norm.clamp(min=1e-8),
            torch.zeros_like(vectors),
        )

    def _build_growth_direction(
        self,
        stress_dir: Optional[Tensor],
        x_surf_world: Tensor,
        impact_center_world: Optional[Tensor],
    ) -> Tensor:
        if stress_dir is None:
            growth_dir = torch.zeros_like(x_surf_world)
        else:
            growth_dir = self._safe_vector_normalize(stress_dir)

        if impact_center_world is not None:
            radial = self._safe_vector_normalize(
                x_surf_world - impact_center_world.unsqueeze(0))
            if stress_dir is None:
                growth_dir = radial
            else:
                growth_dir = self._safe_vector_normalize(0.35 * growth_dir + 0.65 * radial)

        return growth_dir

    @torch.no_grad()
    def _step_fracture(self):
        """
        Fracture evolution on the Gaussian manifold.

        1. Compute tensile energy from current F
        2. Project physics signals to Gaussians
        3. Update fracture field (damage, normal, opening)
        """
        if not self._gravity_drop_contacted and self._gravity_drop:
            return  # No fracture during free-fall

        # Compute tensile energy on MPM particles
        if hasattr(self.elasticity, 'tension_energy_density'):
            psi_mpm = self.elasticity.tension_energy_density(self.F)
        else:
            I = torch.eye(3, device=self.F.device).unsqueeze(0)
            Cmat = self.F.transpose(1, 2) @ self.F
            Egl = 0.5 * (Cmat - I)
            psi_mpm = (Egl * Egl).sum(dim=(1, 2))

        Gc = getattr(self.elasticity, 'Gc', 60000.0)
        l0 = getattr(self.elasticity, 'l0', 0.025)
        psi_mpm = psi_mpm.clamp(0.0, 50.0 * Gc / (2.0 * l0))
        drive_mpm = self._build_effective_fracture_drive(
            psi_mpm, self._last_stress)

        # Project to surface Gaussians
        x_surf_mpm = self.x_mpm[self.surface_mask]
        x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)

        raw_energy_gauss = self.physics_projector.project_scalar(
            psi_mpm, self.x_mpm, x_surf_mpm, frame=self.frame_count)
        drive_gauss = self.physics_projector.project_scalar(
            drive_mpm, self.x_mpm, x_surf_mpm, frame=self.frame_count)
        growth_drive = self._robust_normalize(drive_gauss, 0.90).clamp(0.0, 1.0)

        # Principal stress direction
        stress_dir = None
        sigma1_gauss = torch.zeros_like(growth_drive)
        if self._last_stress is not None:
            stress_dir, sigma1_gauss = self.physics_projector.project_principal_stress_direction(
                self._last_stress, self.x_mpm, x_surf_mpm, frame=self.frame_count)
            sigma1_gauss = self._robust_normalize(sigma1_gauss, 0.94).clamp(0.0, 1.0)

        raw_energy_gauss = self._robust_normalize(raw_energy_gauss, 0.97).clamp(0.0, 1.0)
        init_score = (raw_energy_gauss ** 2) * (
            0.35 + 0.65 * torch.maximum(sigma1_gauss, growth_drive)
        )
        init_score = init_score * (growth_drive > 0.10).float()

        impact_center_world = None
        if hasattr(self, '_impact_center'):
            impact_center_world = self.mapper.mpm_to_world(
                self._impact_center.unsqueeze(0)).squeeze(0)
        growth_dir = self._build_growth_direction(
            stress_dir,
            x_surf_world,
            impact_center_world,
        )

        # Deformation gradient on Gaussians
        F_gauss = self.physics_projector.project_matrix(
            self.F, self.x_mpm, x_surf_mpm, frame=self.frame_count)

        # Update fracture field
        self.fracture_field.update(
            positions=x_surf_world,
            init_score=init_score,
            growth_drive=growth_drive,
            growth_dir=growth_dir,
            F_gaussian=F_gauss,
            impact_center=impact_center_world,
        )

    # ================================================================
    # Fragment detection
    # ================================================================

    @torch.no_grad()
    def _detect_fragments(self):
        """Detect fragments using graph connectivity."""
        if self.fragment_manager is None:
            return
        if self.fracture_field.c is None:
            return
        if self.fracture_field.c.max() < 0.3:
            return

        n_frags = self.fragment_manager.detect_fragments(
            self.graph, self.fracture_field.c)

        if n_frags > 1 and not self.fragmentation_active:
            self.fragmentation_active = True
            self.fracture_field.f = self.fragment_manager.fragment_ids.clone()

            # Apply separation impulse
            if hasattr(self, '_impact_center'):
                x_surf_mpm = self.x_mpm[self.surface_mask]
                x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)
                ic_world = self.mapper.mpm_to_world(
                    self._impact_center.unsqueeze(0)).squeeze(0)

                # Map fragment impulse back to MPM velocities
                N_surf = min(self.fracture_field.c.shape[0],
                             self.surface_mask.sum().item())
                v_surf = torch.zeros(N_surf, 3, device=x_surf_world.device)
                v_surf = self.fragment_manager.apply_fragment_impulse(
                    x_surf_world[:N_surf], v_surf, ic_world)

                surf_indices = torch.where(self.surface_mask)[0]
                n_assign = min(N_surf, surf_indices.shape[0])
                self.v_mpm[surf_indices[:n_assign]] += v_surf[:n_assign]

                print(f"[ManifoldSim] Fragment separation impulse applied")

    # ================================================================
    # Gaussian update for rendering
    # ================================================================

    @torch.no_grad()
    def _update_gaussians(self):
        """Project MPM state to Gaussians and apply fracture visualization."""
        x_surf_mpm = self.x_mpm[self.surface_mask]
        x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)
        F_surf = self.F[self.surface_mask]

        # Handle PLY direct mode
        if self._ply_direct:
            disp = x_surf_world - self._surf_init_world
            ply_disp = disp[self._ply_to_surface]
            x_final = self._ply_init_xyz + ply_disp
            c_final = self.fracture_field.c[self._ply_to_surface] if self.fracture_field.c is not None else None
            F_final = F_surf[self._ply_to_surface]
        else:
            x_final = x_surf_world
            c_final = self.fracture_field.c
            F_final = F_surf

        # Apply fracture-induced deformation BEFORE visualizer
        if (c_final is not None
                and self.fracture_field.n is not None
                and c_final.max() > self.splitter.flatten_threshold):
            n_final = self.fracture_field.n
            a_final = self.fracture_field.a
            if self._ply_direct:
                n_final = n_final[self._ply_to_surface]
                a_final = a_final[self._ply_to_surface]

            # Restore originals before applying effects
            # (visualizer.update_gaussians handles this internally)

        # Build debris mask
        debris_mask = None
        if (self.fragmentation_active
                and self.fragment_manager is not None
                and self.fragment_manager.n_fragments > 1):
            frag_ids = self.fragment_manager.fragment_ids
            if self._ply_direct:
                surf_frag = frag_ids[self._ply_to_surface]
            else:
                N_gauss = x_final.shape[0]
                surf_frag = frag_ids[:N_gauss] if frag_ids is not None else None

            if surf_frag is not None:
                debris_mask = torch.zeros(x_final.shape[0], dtype=torch.bool,
                                          device=x_final.device)
                for i, frag_idx in enumerate(self.fragment_manager.fragment_indices):
                    if len(frag_idx) < self.fragment_manager.min_fragment_size:
                        frag_label = i
                        debris_mask |= (surf_frag == frag_label)

        # Update visualizer
        self.visualizer.update_gaussians(
            self.gaussians, c_final, x_final,
            preserve_original=True,
            debris_mask=debris_mask,
            F_per_gaussian=F_final,
            camera_pos=getattr(self, '_camera_pos', None),
            crack_normals=self.fracture_field.n if not self._ply_direct else (
                self.fracture_field.n[self._ply_to_surface]
                if self.fracture_field.n is not None else None),
            crack_opening=self.fracture_field.a if not self._ply_direct else (
                self.fracture_field.a[self._ply_to_surface]
                if self.fracture_field.a is not None else None),
            crack_tips=(
                self.fracture_field.crack_front.tip_mask
                if (hasattr(self.fracture_field, "crack_front")
                    and not self._ply_direct)
                else (
                    self.fracture_field.crack_front.tip_mask[self._ply_to_surface]
                    if (hasattr(self.fracture_field, "crack_front")
                        and self.fracture_field.crack_front.tip_mask is not None
                        and self._ply_direct)
                    else None
                )
            ),
            crack_visited=(
                self.fracture_field.crack_front.visited_mask
                if (hasattr(self.fracture_field, "crack_front")
                    and not self._ply_direct)
                else (
                    self.fracture_field.crack_front.visited_mask[self._ply_to_surface]
                    if (hasattr(self.fracture_field, "crack_front")
                        and self.fracture_field.crack_front.visited_mask is not None
                        and self._ply_direct)
                    else None
                )
            ),
        )

        # Optional: Gaussian splitting
        if (self.splitting_enabled
                and c_final is not None
                and c_final.max() > self.splitter.split_threshold):
            n_vis = self.fracture_field.n
            a_vis = self.fracture_field.a
            if self._ply_direct:
                n_vis = n_vis[self._ply_to_surface]
                a_vis = a_vis[self._ply_to_surface]

            split_info = self.splitter.split_gaussians(
                self.gaussians, c_final, n_vis, a_vis)
            if split_info['n_split'] > 0:
                self.splitter.extend_fracture_state(
                    self.fracture_field, split_info)

    # ================================================================
    # Impact handling
    # ================================================================

    def _handle_ground_impact(self):
        """Handle ground contact in gravity drop mode."""
        self._gravity_drop_contacted = True
        self.v_mpm[:] = self._v_com.unsqueeze(0)
        v_impact = self._v_com[2].item()
        print(f"  [IMPACT] Ground contact! v_impact={v_impact:.3f}")

        N = self.F.shape[0]
        self.F = torch.eye(3, device=self.F.device).unsqueeze(0).expand(N, 3, 3).clone()
        self.C = torch.zeros_like(self.C)
        self.frame_count = 0
        self._impact_frame_count = 0

        # Impact zone damage seed on Gaussians
        z_vals = self.x_mpm[:, 2]
        z_min = z_vals.min().item()
        z_max = z_vals.max().item()
        obj_height = z_max - z_min

        z_threshold = z_min + obj_height * 0.03
        bottom_mask = z_vals < z_threshold
        impact_center = self.x_mpm[bottom_mask].mean(dim=0)
        self._impact_center = impact_center

        # Seed damage on Gaussians via projector
        x_surf_mpm = self.x_mpm[self.surface_mask]
        x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)
        ic_world = self.mapper.mpm_to_world(impact_center.unsqueeze(0)).squeeze(0)

        seed_radius = max(
            obj_height * self.fracture_cfg.get('impact_seed_radius_factor', 0.08)
            * self.mapper.world_scale,
            2.0 * self.graph.sigma,
        )
        seed_magnitude = self.fracture_cfg.get('impact_seed_magnitude', 0.18)
        seed_H_multiplier = self.fracture_cfg.get('impact_seed_H_multiplier', 0.0)

        self.fracture_field.seed_damage(
            positions=x_surf_world,
            center=ic_world,
            radius=seed_radius,
            magnitude=seed_magnitude,
            H_multiplier=seed_H_multiplier,
        )
        print(f"  [IMPACT] seed_radius={seed_radius:.4f} "
              f"seed_mag={seed_magnitude:.3f} Hx={seed_H_multiplier:.2f}")

        # Post-impact physics adjustments
        g_vec = self.mpm.gravity.clone()
        g_vec[:] = 0.0
        g_vec[2] = -400.0
        self.mpm.gravity = g_vec
        self.mpm.damping = 0.975
        print(f"  [POST-IMPACT] gravity→[0,0,-400] damping→0.975")

    # ================================================================
    # Seismic loading
    # ================================================================

    @torch.no_grad()
    def _apply_seismic_loading(self, dt: float):
        """Apply oscillating body acceleration."""
        if not self.seismic_enabled:
            return
        t = self.mpm.time
        amp = self.seismic.get("amplitude", 1000.0)
        freq = self.seismic.get("frequency", 80.0)
        direction = self.seismic.get("direction", [1.0, 0.0, 0.0])
        ramp_time = self.seismic.get("ramp_time", 0.005)

        device = self.v_mpm.device
        dir_t = torch.tensor(direction, device=device, dtype=self.v_mpm.dtype)
        dir_t = dir_t / (dir_t.norm() + 1e-12)

        envelope = min(t / ramp_time, 1.0) if ramp_time > 0 else 1.0
        accel = amp * math.sin(2.0 * math.pi * freq * t) * envelope
        self.v_mpm += dir_t.unsqueeze(0) * (accel * dt)

    # ================================================================
    # Pre-notch / external force
    # ================================================================

    def apply_pre_notch(self, notches: list):
        """Seed initial cracks from pre-defined notch lines."""
        if not notches:
            return

        x_surf_mpm = self.x_mpm[self.surface_mask]
        x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)

        for notch in notches:
            start = torch.tensor(notch['start'], device=self.x_mpm.device)
            end = torch.tensor(notch['end'], device=self.x_mpm.device)
            damage = notch.get('damage', 0.9)

            center = (start + end) * 0.5
            center_world = self.mapper.mpm_to_world(center.unsqueeze(0)).squeeze(0)
            length = (end - start).norm().item()
            radius = length * 0.5 * self.mapper.world_scale

            self.fracture_field.seed_damage(
                x_surf_world, center_world, radius,
                magnitude=damage, H_multiplier=3.0)

    def initialize_deformation_impact(
        self,
        impact_center_mpm: Tensor,
        impact_energy: float = 1.0,
        impact_radius: float = 0.03,
        impact_direction: Optional[Tensor] = None,
    ):
        """Apply impact: velocity impulse + damage seed."""
        device = self.x_mpm.device
        dists = (self.x_mpm - impact_center_mpm.unsqueeze(0)).norm(dim=1)
        influence = torch.exp(-dists ** 2 / (2 * impact_radius ** 2))

        if impact_direction is not None:
            imp_dir = impact_direction.to(device)
            imp_dir = imp_dir / (imp_dir.norm() + 1e-12)
            self.v_mpm += imp_dir.unsqueeze(0) * influence.unsqueeze(1) * impact_energy
        else:
            directions = impact_center_mpm.unsqueeze(0) - self.x_mpm
            directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-12)
            self.v_mpm += directions * influence.unsqueeze(1) * impact_energy

        # Seed damage on Gaussians
        ic_world = self.mapper.mpm_to_world(impact_center_mpm.unsqueeze(0)).squeeze(0)
        x_surf_world = self.mapper.mpm_to_world(self.x_mpm[self.surface_mask])
        self.fracture_field.seed_damage(
            x_surf_world, ic_world,
            radius=impact_radius * self.mapper.world_scale,
            magnitude=0.8, H_multiplier=4.5)

        self._impact_center = impact_center_mpm
        self._impact_radius = impact_radius

    # ================================================================
    # Utilities
    # ================================================================

    def get_statistics(self) -> Dict:
        """Return simulation statistics."""
        c = self.fracture_field.c
        c_max = c.max().item() if c is not None else 0.0
        c_mean = c.mean().item() if c is not None else 0.0
        n_cracked = (c > 0.3).sum().item() if c is not None else 0

        return {
            "frame": self.frame_count,
            "time": self.mpm.time,
            "c_max": c_max,
            "c_mean": c_mean,
            "n_cracked": n_cracked,
            "n_fragments": (self.fragment_manager.n_fragments
                            if self.fragment_manager else 0),
        }

    def save_state(self, path: str):
        """Save simulation state."""
        state = {
            "frame": self.frame_count,
            "x_mpm": self.x_mpm, "v_mpm": self.v_mpm,
            "F": self.F, "C": self.C,
            "fracture": self.fracture_field.get_state(),
            "gaussian_xyz": self.gaussians._xyz.data,
            "gaussian_opacity": self.gaussians._opacity.data,
            "gaussian_features_dc": self.gaussians._features_dc.data,
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """Load simulation state."""
        ckpt = torch.load(path)
        self.frame_count = ckpt["frame"]
        self.x_mpm = ckpt["x_mpm"]
        self.v_mpm = ckpt["v_mpm"]
        self.F = ckpt["F"]
        self.C = ckpt["C"]
        if "fracture" in ckpt:
            self.fracture_field.load_state(ckpt["fracture"])
        self.gaussians._xyz.data = ckpt["gaussian_xyz"]
        self.gaussians._opacity.data = ckpt["gaussian_opacity"]
        self.gaussians._features_dc.data = ckpt["gaussian_features_dc"]
        print(f"[ManifoldSim] Loaded state from {path} (frame {self.frame_count})")

    def __repr__(self) -> str:
        return (f"ManifoldSimulator(particles={self.x_mpm.shape[0] if self.x_mpm is not None else 'N/A'}, "
                f"frame={self.frame_count})")
