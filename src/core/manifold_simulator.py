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
from src.fracture.crack_front import CrackFront
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

        crack_front = CrackFront(
            seed_quantile=fp.get('seed_quantile', 0.995),
            max_seed_points=fp.get('max_seed_points', 2),
            min_seed_spacing=fp.get('min_seed_spacing', 0.04),
            successor_topk=fp.get('successor_topk', 2),
            min_successor_score=fp.get('min_successor_score', 0.25),
            drive_weight=fp.get('drive_weight', 0.40),
            distance_weight=fp.get('distance_weight', 0.12),
            align_weight=fp.get('align_weight', 0.24),
            tangent_weight=fp.get('tangent_weight', 0.16),
            continuity_weight=fp.get('continuity_weight', 0.10),
            radial_weight=fp.get('radial_weight', 0.16),
            lift_weight=fp.get('lift_weight', 0.40),
            max_tip_age=fp.get('max_tip_age', 2),
            revisit_drive_threshold=fp.get('revisit_drive_threshold', 0.8),
            branch_score_ratio=fp.get('branch_score_ratio', 0.97),
            branch_drive_threshold=fp.get('branch_drive_threshold', 0.70),
            max_branching_tips=fp.get('max_branching_tips', 12),
            tau_init=fp.get('tau_init', 0.30),
            growth_gain=fp.get('growth_gain', 1.0),
            branching_bias=fp.get('branching_bias', 0.20),
            anisotropy_strength=fp.get('anisotropy_strength', 0.10),
            crack_style=fp.get('sentence_style', fp.get('crack_style', 'material_default')),
            material_family=fp.get('material_family', 'neutral_reference'),
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
            front_substeps=fp.get('front_substeps', 2),
            tau_init=fp.get('tau_init', 0.30),
            growth_gain=fp.get('growth_gain', 1.0),
            band_width=fp.get('band_width', 1.5),
            band_fill_gain=fp.get('band_fill_gain', 0.30),
            open_gain=fp.get('open_gain', 1.0),
            material_family=fp.get('material_family', 'neutral_reference'),
            enable_front_propagation=fp.get('enable_front_propagation', True),
            material_drive_floor=fp.get('material_drive_floor', None),
            diffuse_damage_gain=fp.get('diffuse_damage_gain', 0.16),
            diffuse_neighborhood_steps=fp.get('diffuse_neighborhood_steps', 2),
            graph=self.graph,
            crack_front=crack_front,
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
            edge_break_rate=fp.get('edge_break_rate', 1.0),
            opening_weight=fp.get('fragment_opening_weight', 0.35),
            active_tip_weight=fp.get('fragment_active_tip_weight', 0.18),
            recent_front_weight=fp.get('fragment_recent_front_weight', 0.12),
            pair_break_weight=fp.get('fragment_pair_break_weight', 0.10),
            edge_memory_decay=fp.get('fragment_edge_memory_decay', 0.97),
            edge_memory_weight=fp.get('fragment_edge_memory_weight', 0.72),
            cut_diffusion_alpha=fp.get('fragment_cut_diffusion_alpha', 0.0),
            cut_diffusion_iters=fp.get('fragment_cut_diffusion_iters', 0),
            cut_cos_gate_tangent=fp.get('fragment_cut_cos_gate_tangent', 0.5),
            cut_cos_gate_normal=fp.get('fragment_cut_cos_gate_normal', 0.4),
            primary_cut_ratio=fp.get('fragment_primary_cut_ratio', 0.75),
            fallback_cut_ratio=fp.get('fragment_fallback_cut_ratio', 0.55),
            min_boundary_edges=fp.get('fragment_min_boundary_edges', 12),
            detached_node_decay=fp.get('fragment_detached_node_decay', 0.95),
            persistent_min_fragment_size=fp.get('fragment_persistent_min_size', 8),
            component_hysteresis=fp.get('fragment_component_hysteresis', 0.35),
            post_split_threshold_scale=fp.get('fragment_post_split_threshold_scale', 0.92),
            cut_surface_enable=fp.get('cut_surface_enable', False),
            cut_vote_strength=fp.get('cut_vote_strength', 0.0),
            tau_cross=fp.get('tau_cross', 0.60),
            tau_tangent=fp.get('tau_tangent', 0.45),
            cut_core_damage_threshold=fp.get('cut_core_damage_threshold', 0.18),
            cut_core_opening_threshold=fp.get('cut_core_opening_threshold', 0.16),
            cut_hard_break_threshold=fp.get('cut_hard_break_threshold', 0.42),
            authoritative_cut_decay=fp.get('authoritative_cut_decay', 0.96),
            authoritative_cut_threshold=fp.get('authoritative_cut_threshold', 0.20),
            support_loss_enable=fp.get('support_loss_enable', True),
            support_anchor_quantile=fp.get('support_anchor_quantile', 0.10),
            support_release_threshold=fp.get('support_release_threshold', 0.56),
            support_promote_min_size=fp.get('support_promote_min_size', 6),
            support_overlap_threshold=fp.get('support_overlap_threshold', 0.10),
            open_crack_release_enable=fp.get('open_crack_release_enable', True),
            open_crack_release_threshold=fp.get('open_crack_release_threshold', 0.0),
            open_crack_release_max_patches=fp.get('open_crack_release_max_patches', 2),
            crack_style=fp.get('sentence_style', fp.get('crack_style', 'material_default')),
            brittle_release_intensity=fp.get('brittle_release_intensity', 1.0),
            catastrophic_release_enable=fp.get('catastrophic_release_enable', False),
            catastrophic_release_fragility=fp.get('catastrophic_release_fragility', 0.0),
            catastrophic_release_threshold=fp.get('catastrophic_release_threshold', 0.36),
            catastrophic_release_min_threshold=fp.get('catastrophic_release_min_threshold', 0.08),
            catastrophic_release_threshold_decay=fp.get('catastrophic_release_threshold_decay', 0.010),
            catastrophic_release_patches_per_step=fp.get('catastrophic_release_patches_per_step', 0),
            catastrophic_release_patch_radius=fp.get('catastrophic_release_patch_radius', 0.060),
            catastrophic_release_core_radius=fp.get('catastrophic_release_core_radius', 0.024),
            catastrophic_release_min_size=fp.get('catastrophic_release_min_size', 16),
            catastrophic_release_max_size_ratio=fp.get('catastrophic_release_max_size_ratio', 0.040),
            catastrophic_release_max_released_ratio=fp.get('catastrophic_release_max_released_ratio', 0.55),
            secondary_shatter_enable=fp.get('secondary_shatter_enable', False),
            secondary_shatter_start_step=fp.get('secondary_shatter_start_step', 4),
            secondary_shatter_threshold=fp.get('secondary_shatter_threshold', 0.18),
            secondary_shatter_floor=fp.get('secondary_shatter_floor', 0.0),
            secondary_shatter_max_patches=fp.get('secondary_shatter_max_patches', 0),
            secondary_shatter_sector_count=fp.get('secondary_shatter_sector_count', 24),
            secondary_shatter_band_count=fp.get('secondary_shatter_band_count', 3),
            secondary_shatter_height_count=fp.get('secondary_shatter_height_count', 2),
            secondary_shatter_min_size=fp.get('secondary_shatter_min_size', 8),
            secondary_shatter_max_size_ratio=fp.get('secondary_shatter_max_size_ratio', 0.012),
            secondary_shatter_max_released_ratio=fp.get('secondary_shatter_max_released_ratio', 0.70),
            material_family=fp.get('material_family', 'neutral_reference'),
            device=device_str,
        ) if frag_enabled else None
        self.fragmentation_active = False
        self.fragment_detect_every = max(int(fp.get('fragment_detect_every', 2)), 1)
        self.fragment_impulse_strength = float(fp.get('fragment_impulse_strength', 2.8))
        self.fragment_upward_bias = float(fp.get('fragment_upward_bias', 0.45))
        self.fragment_visual_offset_scale = float(fp.get('fragment_visual_offset_scale', 0.012))
        self.fragment_visual_ramp_frames = max(int(fp.get('fragment_visual_ramp_frames', 6)), 1)
        self.fragment_impulse_boost_frames = max(int(fp.get('fragment_impulse_boost_frames', 5)), 0)
        self.fragment_impulse_decay = float(fp.get('fragment_impulse_decay', 0.75))
        self.fragment_event_boost = float(fp.get('fragment_event_boost', 1.0))
        self._fragment_activation_frame = -1
        self.material_family = str(fp.get('material_family', 'neutral_reference'))
        self.crack_style = str(fp.get('sentence_style', fp.get('crack_style', 'material_default')))
        self.shard_enable = bool(fp.get('shard_enable', False))
        self.shard_count_scale = float(fp.get('shard_count_scale', 0.0))
        self.fragment_offset_gain = float(fp.get('fragment_offset_gain', 1.0))
        self.debris_motion_gain = float(fp.get('debris_motion_gain', 0.0))
        self.split_gap_gain = float(fp.get('split_gap_gain', 1.0))
        self.fragment_shell_gain = float(fp.get('fragment_shell_gain', 1.0))
        self.fragment_contrast_gain = float(fp.get('fragment_contrast_gain', 1.0))
        self.debris_darkening = float(fp.get('debris_darkening', 0.20))
        self.shard_scale_gain = float(fp.get('shard_scale_gain', 1.0))
        self.shard_opacity_gain = float(fp.get('shard_opacity_gain', 1.0))
        self._last_render_state: Optional[Dict] = None
        self.fragment_render_gravity = float(fp.get('fragment_render_gravity', 0.0026))
        self.fragment_render_damping = float(fp.get('fragment_render_damping', 0.985))
        self.fragment_render_lateral_damping = float(fp.get('fragment_render_lateral_damping', 0.992))
        self.fragment_render_gap_scale = float(fp.get('fragment_render_gap_scale', 1.55))
        self.fragment_render_velocity_scale = float(fp.get('fragment_render_velocity_scale', 0.22))
        self.fragment_render_min_size = max(int(fp.get('fragment_render_min_size', 6)), 1)
        self.fragment_render_overlap_threshold = float(
            fp.get('fragment_render_overlap_threshold', 0.24)
        )
        self.fragment_physical_min_size = max(
            int(fp.get('fragment_physical_min_size', max(self.fragment_render_min_size * 4, 24))),
            1,
        )
        self.fragment_physical_overlap_threshold = float(
            fp.get('fragment_physical_overlap_threshold', 0.18)
        )
        self._render_fragment_labels: Optional[Tensor] = None
        self._render_fragment_states = {}
        self._next_render_fragment_id = 1
        self._physical_fragment_labels: Optional[Tensor] = None
        self._physical_fragment_states = {}
        self._next_physical_fragment_id = 1

        self.damage_feedback_delay_frames = int(
            fp.get('damage_feedback_delay_frames', 8))
        self.damage_feedback_ramp_frames = int(
            fp.get('damage_feedback_ramp_frames', 6))
        self.interior_damage_scale = float(
            fp.get('interior_damage_scale', 0.2))
        self.volumetric_cut_damage_scale = float(
            fp.get('volumetric_cut_damage_scale', 0.58))
        self.volumetric_auth_damage_floor = float(
            fp.get('volumetric_auth_damage_floor', 0.72))
        self.volumetric_detached_damage_floor = float(
            fp.get('volumetric_detached_damage_floor', 0.90))
        self.crack_volume_feedback_gain = float(
            fp.get('crack_volume_feedback_gain', 0.42))
        self.crack_volume_opening_gain = float(
            fp.get('crack_volume_opening_gain', 0.38))
        self.crack_volume_visited_floor = float(
            fp.get('crack_volume_visited_floor', 0.50))
        self.crack_volume_tip_floor = float(
            fp.get('crack_volume_tip_floor', 0.66))
        self.crack_volume_interior_scale = float(
            fp.get('crack_volume_interior_scale', 0.85))
        self.fragment_physical_gap_scale = float(
            fp.get('fragment_physical_gap_scale', 0.0))
        self.fragment_physical_release_velocity = float(
            fp.get('fragment_physical_release_velocity', 0.0))
        self.fragment_physical_downward_bias = float(
            fp.get('fragment_physical_downward_bias', 0.32))
        self.fragment_physical_release_frames = max(
            int(fp.get('fragment_physical_release_frames', 20)), 0)
        self.drive_tension_weight = float(
            fp.get('drive_tension_weight', 1.0))
        self.drive_shear_weight = float(
            fp.get('drive_shear_weight', 0.35))
        self.drive_principal_weight = float(
            fp.get('drive_principal_weight', 0.25))
        self.drive_kinetic_weight = float(
            fp.get('drive_kinetic_weight', 0.15))

        # Render-only shards are disabled in the fragment-separation phase.
        self.splitting_enabled = False

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
        self._surface_indices: Optional[Tensor] = None
        self._interior_indices: Optional[Tensor] = None
        self._particle_to_surface_local: Optional[Tensor] = None
        self._surface_local_index: Optional[Tensor] = None
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
        print(f"  Family: {fp.get('material_family', 'neutral_reference')}")
        print(
            f"  Material-aware: tau={fp.get('tau_init', 0.30):.2f}, "
            f"growth={fp.get('growth_gain', 1.0):.2f}, "
            f"band={fp.get('band_width', 1.5):.2f}, "
            f"open={fp.get('open_gain', 1.0):.2f}, "
            f"branch={fp.get('branching_bias', 0.20):.2f}, "
            f"edge_break={fp.get('edge_break_rate', 1.0):.2f}"
        )
        print(f"  Damage feedback: delay={self.damage_feedback_delay_frames}, "
              f"ramp={self.damage_feedback_ramp_frames}, "
              f"interior_scale={self.interior_damage_scale:.2f}")
        print(
            f"  Volumetric cut feedback: scale={self.volumetric_cut_damage_scale:.2f}, "
            f"auth_floor={self.volumetric_auth_damage_floor:.2f}, "
            f"detach_floor={self.volumetric_detached_damage_floor:.2f}"
        )
        print(
            f"  Crack-volume coupling: gain={self.crack_volume_feedback_gain:.2f}, "
            f"opening={self.crack_volume_opening_gain:.2f}, "
            f"interior={self.crack_volume_interior_scale:.2f}"
        )
        print(f"  Splitting: {'ON' if self.splitting_enabled else 'OFF'}")
        print(
            f"  Shards: {'ON' if self.shard_enable else 'OFF'} "
            f"(count_scale={self.shard_count_scale:.2f}, motion={self.debris_motion_gain:.2f})"
        )
        print(f"  Fragmentation: {'ON' if frag_enabled else 'OFF'}")
        if frag_enabled:
            print(
                f"  Fragment detach: detect_every={self.fragment_detect_every}, "
                f"impulse={self.fragment_impulse_strength:.2f}, "
                f"visual_offset={self.fragment_visual_offset_scale:.4f}, "
                f"memory={fp.get('fragment_edge_memory_decay', 0.97):.2f}, "
                f"hysteresis={fp.get('fragment_component_hysteresis', 0.35):.2f}"
            )
            if fp.get('cut_surface_enable', False):
                print(
                    f"  Cut-surface: vote={fp.get('cut_vote_strength', 0.0):.2f}, "
                    f"tau_cross={fp.get('tau_cross', 0.60):.2f}, "
                    f"tau_tangent={fp.get('tau_tangent', 0.45):.2f}, "
                    f"diffusion={fp.get('fragment_cut_diffusion_alpha', 0.0):.2f}x"
                    f"{int(fp.get('fragment_cut_diffusion_iters', 0))}, "
                    f"primary={fp.get('fragment_primary_cut_ratio', 0.75):.2f}, "
                    f"event_boost={self.fragment_event_boost:.2f}"
                )
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
        self._last_render_state = None
        self._render_fragment_labels = None
        self._render_fragment_states = {}
        self._next_render_fragment_id = 1
        self._physical_fragment_labels = None
        self._physical_fragment_states = {}
        self._next_physical_fragment_id = 1
        self._surface_indices = torch.where(self.surface_mask)[0]
        self._interior_indices = torch.where(~self.surface_mask)[0]
        self._build_rest_surface_binding()
        print(f"[ManifoldSim] Initialized: {N} particles, {N_surf} surface")

    def _build_rest_surface_binding(self, chunk_size: int = 2048) -> None:
        """Bind each particle to a rest-state nearest surface Gaussian."""
        if self.init_positions is None:
            return
        if self._surface_indices is None:
            self._surface_indices = torch.where(self.surface_mask)[0]
        if self._interior_indices is None:
            self._interior_indices = torch.where(~self.surface_mask)[0]

        surface_indices = self._surface_indices
        interior_indices = self._interior_indices
        device = self.init_positions.device
        n_total = self.init_positions.shape[0]
        n_surf = int(surface_indices.shape[0])
        if n_surf <= 0:
            self._particle_to_surface_local = None
            self._surface_local_index = None
            return

        particle_to_surface = torch.full(
            (n_total,), -1, dtype=torch.long, device=device
        )
        surface_local_index = torch.full(
            (n_total,), -1, dtype=torch.long, device=device
        )
        surface_local = torch.arange(n_surf, device=device, dtype=torch.long)
        particle_to_surface[surface_indices] = surface_local
        surface_local_index[surface_indices] = surface_local

        if interior_indices.numel() > 0:
            x_surf = self.init_positions[surface_indices]
            nearest_chunks = []
            for start in range(0, interior_indices.shape[0], chunk_size):
                end = min(start + chunk_size, interior_indices.shape[0])
                x_chunk = self.init_positions[interior_indices[start:end]]
                dists = torch.cdist(x_chunk, x_surf)
                nearest_chunks.append(dists.argmin(dim=1))
            nearest = torch.cat(nearest_chunks, dim=0)
            particle_to_surface[interior_indices] = nearest

        self._particle_to_surface_local = particle_to_surface
        self._surface_local_index = surface_local_index
        print(
            f"[ManifoldSim] Rest surface binding built: "
            f"{n_total} particles -> {n_surf} surface anchors"
        )

    def _project_surface_scalar_to_particles(
        self,
        surface_values: Tensor,
        interior_scale: float = 1.0,
    ) -> Tensor:
        """Project a per-surface scalar field to all particles via rest binding."""
        out = torch.zeros(
            self.x_mpm.shape[0],
            dtype=surface_values.dtype,
            device=surface_values.device,
        )
        if self._surface_indices is None or self._particle_to_surface_local is None:
            return out

        n_assign = min(surface_values.shape[0], self._surface_indices.shape[0])
        if n_assign <= 0:
            return out

        surface_indices = self._surface_indices[:n_assign]
        out[surface_indices] = surface_values[:n_assign]
        if self._interior_indices is not None and self._interior_indices.numel() > 0:
            mapped = self._particle_to_surface_local[self._interior_indices]
            valid = (mapped >= 0) & (mapped < n_assign)
            if bool(valid.any()):
                out[self._interior_indices[valid]] = (
                    surface_values[mapped[valid]] * interior_scale
                )
        return out

    def _map_surface_labels_to_particles(
        self,
        surface_labels: Tensor,
    ) -> Tensor:
        """Map per-surface fragment labels to all particles via rest binding."""
        out = torch.zeros(
            self.x_mpm.shape[0],
            dtype=torch.long,
            device=surface_labels.device,
        )
        if self._surface_indices is None or self._particle_to_surface_local is None:
            return out

        n_assign = min(surface_labels.shape[0], self._surface_indices.shape[0])
        if n_assign <= 0:
            return out

        surface_indices = self._surface_indices[:n_assign]
        out[surface_indices] = surface_labels[:n_assign]
        if self._interior_indices is not None and self._interior_indices.numel() > 0:
            mapped = self._particle_to_surface_local[self._interior_indices]
            valid = (mapped >= 0) & (mapped < n_assign)
            if bool(valid.any()):
                out[self._interior_indices[valid]] = surface_labels[mapped[valid]]
        return out

    def _ensure_physical_fragment_registry(self, count: int, device) -> None:
        if (self._physical_fragment_labels is not None
                and self._physical_fragment_labels.shape[0] == count
                and self._physical_fragment_labels.device == device):
            return
        self._physical_fragment_labels = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self._physical_fragment_states = {}
        self._next_physical_fragment_id = 1

    def _update_physical_fragment_registry(self, raw_particle_labels: Tensor) -> Tensor:
        """Persist released particle chunks even if graph labels later merge back."""
        self._ensure_physical_fragment_registry(raw_particle_labels.shape[0], raw_particle_labels.device)
        persistent = self._physical_fragment_labels.clone()
        active_raw = raw_particle_labels.unique(sorted=True)

        for raw_id in active_raw.tolist():
            if raw_id <= 0:
                continue
            mask = raw_particle_labels == raw_id
            raw_size = int(mask.sum().item())
            if raw_size < self.fragment_physical_min_size:
                continue

            overlap_vals, overlap_counts = persistent[mask].unique(return_counts=True)
            persistent_label = None
            if overlap_vals.numel() > 0:
                overlap_mask = overlap_vals > 0
                if bool(overlap_mask.any()):
                    overlap_vals = overlap_vals[overlap_mask]
                    overlap_counts = overlap_counts[overlap_mask]
                    best_idx = int(torch.argmax(overlap_counts).item())
                    best_count = int(overlap_counts[best_idx].item())
                    if best_count / max(raw_size, 1) >= self.fragment_physical_overlap_threshold:
                        persistent_label = int(overlap_vals[best_idx].item())

            if persistent_label is None:
                persistent_label = self._next_physical_fragment_id
                self._next_physical_fragment_id += 1
                self._physical_fragment_states[persistent_label] = {
                    "age": 0,
                    "release_score": 0.0,
                    "support_lost": False,
                }

            persistent[mask] = persistent_label
            state = self._physical_fragment_states.get(
                persistent_label,
                {"age": 0, "release_score": 0.0, "support_lost": False},
            )
            release_score = 0.0
            support_lost = False
            if (self.fragment_manager is not None
                    and raw_id < len(self.fragment_manager.fragment_release_scores)):
                release_score = float(self.fragment_manager.fragment_release_scores[raw_id])
            if (self.fragment_manager is not None
                    and raw_id < len(self.fragment_manager.fragment_support_lost)):
                support_lost = bool(self.fragment_manager.fragment_support_lost[raw_id])
            state["release_score"] = max(float(state.get("release_score", 0.0)), release_score)
            state["support_lost"] = bool(state.get("support_lost", False) or support_lost)
            state["age"] = int(state.get("age", 0)) + 1
            self._physical_fragment_states[persistent_label] = state

        self._physical_fragment_labels = persistent
        if bool((persistent > 0).any()):
            return persistent
        return raw_particle_labels

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
                and self.frame_count % self.fragment_detect_every == 0):
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

        # Surface particles get direct damage from their Gaussian.
        c_surf = self.fracture_field.c * feedback_scale
        if self._surface_indices is None:
            self._surface_indices = torch.where(self.surface_mask)[0]
        N_surf = c_surf.shape[0]
        n_assign = min(N_surf, self._surface_indices.shape[0])
        if n_assign <= 0:
            return c_vol

        surf_indices = self._surface_indices[:n_assign]
        c_vol[surf_indices] = c_surf[:n_assign]

        # Interior particles use a persistent rest-state surface binding so
        # crack-through thickness remains stable instead of reattaching each step.
        if (self._interior_indices is not None
                and self._interior_indices.numel() > 0
                and c_surf.max() > 0.01
                and self._particle_to_surface_local is not None):
            mapped = self._particle_to_surface_local[self._interior_indices]
            valid = (mapped >= 0) & (mapped < n_assign)
            if bool(valid.any()):
                c_vol[self._interior_indices[valid]] = (
                    c_surf[:n_assign][mapped[valid]] * self.interior_damage_scale
                )

        # Couple visible surface crack state back into the volumetric stress
        # solve. This makes crack-front history and opening weaken a through-
        # thickness corridor instead of leaving the MPM body glued together.
        crack_feedback = torch.zeros_like(c_surf[:n_assign])
        crack_front = getattr(self.fracture_field, "crack_front", None)
        visited_mask = None
        tip_mask = None
        if crack_front is not None:
            visited_mask = getattr(crack_front, "visited_mask", None)
            tip_mask = getattr(crack_front, "tip_mask", None)

        if visited_mask is not None:
            visited_mask = visited_mask[:n_assign]
            crack_feedback = torch.maximum(
                crack_feedback,
                visited_mask.float() * self.crack_volume_visited_floor,
            )
        if tip_mask is not None:
            tip_mask = tip_mask[:n_assign]
            crack_feedback = torch.maximum(
                crack_feedback,
                tip_mask.float() * self.crack_volume_tip_floor,
            )
        if self.fracture_field.a is not None:
            opening = self.fracture_field.a[:n_assign].clamp(min=0.0)
            if bool(opening.max() > 0.0):
                opening_scale = torch.quantile(opening.detach(), 0.85).clamp(min=1e-6)
                opening_norm = (opening / opening_scale).clamp(0.0, 1.0)
                crack_feedback = torch.maximum(
                    crack_feedback,
                    (c_surf[:n_assign] + self.crack_volume_opening_gain * opening_norm)
                    .clamp(0.0, 1.0),
                )
        elif bool(c_surf[:n_assign].max() > 0.0):
            crack_feedback = torch.maximum(crack_feedback, c_surf[:n_assign])

        if bool(crack_feedback.max() > 0.0):
            crack_feedback = (
                crack_feedback
                * self.crack_volume_feedback_gain
                * feedback_scale
            ).clamp(0.0, 1.0)
            c_vol = torch.maximum(
                c_vol,
                self._project_surface_scalar_to_particles(
                    crack_feedback,
                    interior_scale=self.crack_volume_interior_scale,
                ),
            )

        # Promote structural crack corridors into a volumetric stiffness loss.
        if self.fragment_manager is not None:
            structural_floor = torch.zeros_like(c_surf[:n_assign])
            structural_mask = torch.zeros_like(c_surf[:n_assign], dtype=torch.bool)

            auth_mask = getattr(self.fragment_manager, "last_authoritative_cut_mask", None)
            if auth_mask is not None:
                auth_mask = auth_mask[:n_assign]
                structural_mask |= auth_mask
                structural_floor = torch.maximum(
                    structural_floor,
                    auth_mask.float() * self.volumetric_auth_damage_floor,
                )

            cut_core_mask = getattr(self.fragment_manager, "last_cut_core_mask", None)
            if cut_core_mask is not None:
                cut_core_mask = cut_core_mask[:n_assign]
                structural_mask |= cut_core_mask
                structural_floor = torch.maximum(
                    structural_floor,
                    cut_core_mask.float() * (0.78 * self.volumetric_auth_damage_floor),
                )

            support_mask = getattr(self.fragment_manager, "last_support_lost_mask", None)
            if support_mask is not None:
                support_mask = support_mask[:n_assign]
                structural_mask |= support_mask
                structural_floor = torch.maximum(
                    structural_floor,
                    support_mask.float() * self.volumetric_detached_damage_floor,
                )

            closure_mask = getattr(self.fragment_manager, "last_closure_candidate_mask", None)
            if closure_mask is not None:
                closure_mask = closure_mask[:n_assign]
                structural_mask |= closure_mask
                structural_floor = torch.maximum(
                    structural_floor,
                    closure_mask.float() * (0.86 * self.volumetric_detached_damage_floor),
                )

            frag_ids = getattr(self.fragment_manager, "fragment_ids", None)
            if frag_ids is not None:
                detached_mask = frag_ids[:n_assign] > 0
                structural_mask |= detached_mask
                structural_floor = torch.maximum(
                    structural_floor,
                    detached_mask.float() * (0.65 * self.volumetric_detached_damage_floor),
                )

            if self.fracture_field.a is not None and bool(structural_mask.any()):
                opening = self.fracture_field.a[:n_assign].clamp(min=0.0)
                opening_scale = torch.quantile(opening.detach(), 0.85).clamp(min=1e-6)
                opening_norm = (opening / opening_scale).clamp(0.0, 1.0)
                structural_floor = torch.maximum(
                    structural_floor,
                    structural_mask.float() * opening_norm * self.volumetric_cut_damage_scale,
                )

            if bool(structural_floor.max() > 0.0):
                c_vol = torch.maximum(
                    c_vol,
                    self._project_surface_scalar_to_particles(
                        structural_floor * feedback_scale,
                        interior_scale=1.0,
                    ),
                )

        return c_vol

    def _step_fragmented_physics(self, stress: Tensor, dt: float):
        """Per-fragment MPM physics."""
        # Map Gaussian fragments back to MPM particles
        surf_frag_ids = self.fragment_manager.fragment_ids
        raw_mpm_frag_ids = self._map_surface_labels_to_particles(surf_frag_ids)
        mpm_frag_ids = self._update_physical_fragment_registry(raw_mpm_frag_ids)

        # Per-fragment P2G2P
        for frag_id in mpm_frag_ids.unique(sorted=True).tolist():
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

        self._apply_physical_fragment_release_drift(mpm_frag_ids, dt)
        self.mpm.time += dt

    def _apply_physical_fragment_release_drift(self, mpm_frag_ids: Tensor, dt: float) -> None:
        """Apply a small physical gap / release drift to support-lost fragments."""
        if self.fragment_manager is None or self.fragment_manager.n_fragments <= 1:
            return
        if self.fragment_physical_release_frames <= 0:
            return
        if self.fragment_physical_gap_scale <= 0.0 and self.fragment_physical_release_velocity <= 0.0:
            return
        if self._fragment_activation_frame < 0:
            return

        frames_since = max(self.frame_count - self._fragment_activation_frame, 0)
        if frames_since > self.fragment_physical_release_frames:
            return
        taper = 1.0 - (frames_since / max(float(self.fragment_physical_release_frames), 1.0))
        base_mask = mpm_frag_ids == 0
        if not bool(base_mask.any()):
            return
        base_com = self.x_mpm[base_mask].mean(dim=0)

        for frag_id in mpm_frag_ids.unique(sorted=True).tolist():
            if frag_id <= 0:
                continue
            mask = mpm_frag_ids == frag_id
            if int(mask.sum().item()) < 8:
                continue
            state = self._physical_fragment_states.get(frag_id, {})
            release_score = float(state.get("release_score", 0.0))
            support_lost = bool(state.get("support_lost", False))
            if not support_lost and release_score < 0.72:
                continue

            frag_com = self.x_mpm[mask].mean(dim=0)
            direction = frag_com - base_com
            direction[2] -= self.fragment_physical_downward_bias * (0.65 + 0.85 * release_score)
            direction = self._safe_vector_normalize(direction.unsqueeze(0)).squeeze(0)
            if float(direction.norm().item()) < 1e-8:
                direction = torch.tensor([0.0, 0.0, -1.0], device=self.x_mpm.device)

            gap_step = self.fragment_physical_gap_scale * taper * (1.0 + 1.25 * release_score)
            vel_step = self.fragment_physical_release_velocity * taper * (0.55 + 0.95 * release_score)
            if gap_step > 0.0:
                self.x_mpm[mask] = self.x_mpm[mask] + direction.unsqueeze(0) * gap_step
            if vel_step > 0.0:
                self.v_mpm[mask] = self.v_mpm[mask] + direction.unsqueeze(0) * vel_step

        self.x_mpm = self.x_mpm.clamp(self.mpm.clip_bound, 1.0 - self.mpm.clip_bound)

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

    def _apply_sentence_crack_style_drive(
        self,
        x_surf_world: Tensor,
        init_score: Tensor,
        growth_drive: Tensor,
        growth_dir: Tensor,
        impact_center_world: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Shape impact drive into sentence-level crack motifs.

        This is a graphics-side controller layered on top of physical stress:
        it only biases where the surface crack front can run. Material priors
        still control whether those cracks can detach fragments.
        """
        style = self.crack_style
        if (
            impact_center_world is None
            or style in {"material_default", "diffuse_microcrack"}
            or x_surf_world.numel() == 0
        ):
            if style == "diffuse_microcrack":
                return 0.45 * init_score, 0.38 * growth_drive, growth_dir
            return init_score, growth_drive, growth_dir

        rel = x_surf_world - impact_center_world.unsqueeze(0)
        extent = x_surf_world.max(dim=0).values - x_surf_world.min(dim=0).values
        diag = extent.norm().clamp(min=1e-6)
        planar = rel[:, :2]
        planar_r = planar.norm(dim=1).clamp(min=1e-8)
        r_norm = (planar_r / (0.45 * diag)).clamp(0.0, 1.0)
        theta = torch.atan2(planar[:, 1], planar[:, 0])
        radial = torch.zeros_like(rel)
        radial[:, :2] = planar / planar_r.unsqueeze(1).clamp(min=1e-8)
        upward = torch.zeros_like(rel)
        upward[:, 2] = 1.0
        tangent = torch.zeros_like(rel)
        tangent[:, 0] = -radial[:, 1]
        tangent[:, 1] = radial[:, 0]

        if style == "radial_shatter":
            ray_count = 14.0
            phase = 0.45
            ray = (0.5 + 0.5 * torch.cos(ray_count * theta + phase)).clamp(0.0, 1.0)
            ray = ray.pow(2.3)
            near = torch.exp(-0.5 * (planar_r / (0.15 * diag)).pow(2.0))
            far_gain = (0.16 + 0.84 * r_norm).clamp(0.0, 1.0)
            ray_drive = (ray * far_gain + 0.34 * near).clamp(0.0, 1.0)
            growth_drive = torch.maximum(growth_drive, 0.94 * ray_drive)
            init_score = torch.maximum(init_score, 0.74 * near * (0.30 + 0.70 * ray))
            growth_dir = self._safe_vector_normalize(1.70 * radial + 0.18 * upward)
        elif style == "spiderweb_branching":
            ray = (0.5 + 0.5 * torch.cos(9.0 * theta + 0.25)).clamp(0.0, 1.0).pow(2.0)
            rings = (0.5 + 0.5 * torch.cos(30.0 * r_norm + 0.35)).clamp(0.0, 1.0).pow(2.2)
            web = torch.maximum(0.88 * ray, 0.82 * rings) * (0.16 + 0.84 * r_norm)
            near = torch.exp(-0.5 * (planar_r / (0.16 * diag)).pow(2.0))
            growth_drive = torch.maximum(growth_drive, 0.76 * web.clamp(0.0, 1.0))
            init_score = torch.maximum(init_score, 0.48 * near * (0.45 + 0.55 * ray))
            tangent_sign = torch.sign(torch.sin(9.0 * theta + 0.25))
            tangent_sign = torch.where(
                tangent_sign.abs() > 0,
                tangent_sign,
                torch.ones_like(tangent_sign),
            )
            tangent = tangent * tangent_sign.unsqueeze(1)
            ring_mix = (0.25 + 0.75 * rings).unsqueeze(1)
            ray_mix = (0.35 + 0.65 * ray).unsqueeze(1)
            growth_dir = self._safe_vector_normalize(
                0.78 * ray_mix * radial
                + 0.66 * ring_mix * tangent
                + 0.18 * upward
            )
        elif style == "single_smooth":
            angle = torch.tensor(0.35, dtype=x_surf_world.dtype, device=x_surf_world.device)
            axis = torch.stack([torch.cos(angle), torch.sin(angle)])
            tangent = planar @ axis
            cross = planar[:, 0] * axis[1] - planar[:, 1] * axis[0]
            width = 0.045 * diag
            line = torch.exp(-0.5 * (cross / width).pow(2.0))
            forward = (tangent > (-0.10 * diag)).float()
            line_drive = line * forward * (0.25 + 0.75 * r_norm)
            growth_drive = torch.maximum(0.38 * growth_drive, 0.70 * line_drive)
            near = torch.exp(-0.5 * (planar_r / (0.12 * diag)).pow(2.0))
            init_score = torch.maximum(init_score, 0.36 * near * line)
            axis3 = torch.zeros_like(rel)
            axis3[:, 0] = axis[0]
            axis3[:, 1] = axis[1]
            growth_dir = self._safe_vector_normalize(axis3 + 0.16 * upward)
        elif style == "chunky_crumble":
            noise = torch.sin(
                31.0 * x_surf_world[:, 0]
                - 17.0 * x_surf_world[:, 1]
                + 23.0 * x_surf_world[:, 2]
            )
            grain = (0.5 + 0.5 * noise).clamp(0.0, 1.0)
            broad = torch.exp(-0.5 * (planar_r / (0.26 * diag)).pow(2.0))
            crumble = (0.45 * broad + 0.55 * grain * (0.25 + 0.75 * r_norm)).clamp(0.0, 1.0)
            growth_drive = torch.maximum(growth_drive, 0.58 * crumble)
            init_score = torch.maximum(init_score, 0.28 * broad * (0.55 + 0.45 * grain))
            growth_dir = self._safe_vector_normalize(0.60 * radial + 0.38 * tangent + 0.18 * upward)

        return init_score.clamp(0.0, 1.0), growth_drive.clamp(0.0, 1.0), growth_dir

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
        init_score, growth_drive, growth_dir = self._apply_sentence_crack_style_drive(
            x_surf_world,
            init_score,
            growth_drive,
            growth_dir,
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
        if getattr(self.fragment_manager, 'material_family', '') == 'diffuse_damage':
            N = self.fracture_field.c.shape[0]
            self.fragment_manager.n_fragments = 1
            self.fragment_manager.fragment_ids = torch.zeros(
                N, dtype=torch.long, device=self.fracture_field.c.device
            )
            self.fragment_manager.fragment_sizes = [N]
            self.fragment_manager.fragment_indices = [
                torch.arange(N, device=self.fracture_field.c.device)
            ]
            return
        opening_max = (
            float(self.fracture_field.a.max().item())
            if self.fracture_field.a is not None else 0.0
        )
        if self.fracture_field.c.max() < 0.3 and opening_max < 1e-4:
            return

        crack_front = getattr(self.fracture_field, "crack_front", None)
        tip_mask = crack_front.tip_mask if crack_front is not None else None
        recent_front_mask = None
        family = getattr(self.fragment_manager, 'material_family', 'neutral_reference')
        crack_tangent = crack_front.growth_dir if crack_front is not None else None
        if crack_front is not None and crack_front.visited_mask is not None:
            recent_threshold = 0.22
            if family == 'sharp_brittle':
                recent_threshold = 0.08
            elif family == 'brittle_moderate':
                recent_threshold = 0.18
            recent_front_mask = crack_front.visited_mask & (self.fracture_field.c > recent_threshold)
            if family == 'sharp_brittle' and tip_mask is not None:
                recent_front_mask = recent_front_mask | tip_mask
        opening = self.fracture_field.a if self.fracture_field.a is not None else None
        crack_normal = self.fracture_field.n if self.fracture_field.n is not None else None
        x_surf_world = self.mapper.mpm_to_world(self.x_mpm[self.surface_mask])
        N_surf = min(self.fracture_field.c.shape[0], x_surf_world.shape[0])
        n_frags = self.fragment_manager.detect_fragments(
            self.graph,
            self.fracture_field.c,
            positions=x_surf_world[:N_surf],
            opening=opening,
            active_tip_mask=tip_mask,
            recent_front_mask=recent_front_mask,
            crack_normal=crack_normal,
            crack_tangent=crack_tangent,
        )

        if n_frags > 1:
            self.fracture_field.f = self.fragment_manager.fragment_ids.clone()
            apply_impulse = False
            if not self.fragmentation_active:
                self.fragmentation_active = True
                self._fragment_activation_frame = self.frame_count
                apply_impulse = True
            elif (
                self._fragment_activation_frame >= 0
                and (self.frame_count - self._fragment_activation_frame) < self.fragment_impulse_boost_frames
            ):
                apply_impulse = True

            # Apply separation impulse
            if apply_impulse and hasattr(self, '_impact_center'):
                x_surf_mpm = self.x_mpm[self.surface_mask]
                x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)
                ic_world = self.mapper.mpm_to_world(
                    self._impact_center.unsqueeze(0)).squeeze(0)

                # Map fragment impulse back to MPM velocities
                N_surf = min(self.fracture_field.c.shape[0],
                             self.surface_mask.sum().item())
                v_surf = torch.zeros(N_surf, 3, device=x_surf_world.device)
                v_surf = self.fragment_manager.apply_fragment_impulse(
                    x_surf_world[:N_surf],
                    v_surf,
                    ic_world,
                    impulse_strength=self._current_fragment_impulse_strength(),
                    upward_bias=self.fragment_upward_bias,
                )

                surf_indices = torch.where(self.surface_mask)[0]
                n_assign = min(N_surf, surf_indices.shape[0])
                self.v_mpm[surf_indices[:n_assign]] += v_surf[:n_assign]

                print(
                    f"[ManifoldSim] Fragment separation impulse applied "
                    f"(n_frags={n_frags}, broken_edges={self.fragment_manager.last_broken_edges})"
                )

    # ================================================================
    # Gaussian update for rendering
    # ================================================================

    def _current_fragment_impulse_strength(self) -> float:
        if not self.fragmentation_active or self._fragment_activation_frame < 0:
            return self.fragment_impulse_strength
        frames_since = max(self.frame_count - self._fragment_activation_frame, 0)
        if frames_since >= self.fragment_impulse_boost_frames:
            return self.fragment_impulse_strength
        decay = self.fragment_impulse_decay ** frames_since
        return self.fragment_impulse_strength * self.fragment_event_boost * max(decay, 0.45)

    def _current_fragment_visual_boost(self) -> float:
        if not self.fragmentation_active or self._fragment_activation_frame < 0:
            return 1.0
        frames_since = max(self.frame_count - self._fragment_activation_frame, 0)
        if frames_since >= self.fragment_impulse_boost_frames:
            return 1.0
        decay = self.fragment_impulse_decay ** frames_since
        return 1.0 + (self.fragment_event_boost - 1.0) * max(decay, 0.50)

    def _ensure_render_fragment_registry(self, count: int, device) -> None:
        if (self._render_fragment_labels is not None
                and self._render_fragment_labels.shape[0] == count
                and self._render_fragment_labels.device == device):
            return
        self._render_fragment_labels = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self._render_fragment_states = {}
        self._next_render_fragment_id = 1

    def _advance_render_fragment_states(self) -> None:
        if not self._render_fragment_states:
            return
        gravity_step = float(self.fragment_render_gravity)
        damp_z = float(self.fragment_render_damping)
        damp_xy = float(self.fragment_render_lateral_damping)
        max_detach = 0.45
        if self.material_family == "rough_quasi_brittle":
            max_detach = 0.55
        elif self.material_family == "brittle_moderate":
            max_detach = 0.40
        for state in self._render_fragment_states.values():
            vel = state["velocity"]
            vel = vel.clone()
            release_score = float(state.get("release_score", 0.0))
            support_lost = bool(state.get("support_lost", False))
            local_gravity = gravity_step * (1.0 + (0.85 * release_score if support_lost else 0.0))
            local_damp_z = max(0.90, damp_z - (0.05 * release_score if support_lost else 0.0))
            vel[:2] *= damp_xy
            vel[2] = vel[2] * local_damp_z - local_gravity
            state["velocity"] = vel
            new_offset = state["offset"] + vel
            offset_norm = float(new_offset.norm().item())
            if offset_norm > max_detach:
                new_offset = new_offset * (max_detach / max(offset_norm, 1e-8))
                vel = vel * 0.55
                state["velocity"] = vel
            state["offset"] = new_offset
            state["age"] = int(state["age"]) + 1

    def _register_render_fragments(
        self,
        positions: Tensor,
        fragment_ids: Optional[Tensor],
        opening: Optional[Tensor] = None,
    ) -> None:
        if fragment_ids is None:
            return
        if self._render_fragment_labels is None:
            self._ensure_render_fragment_registry(positions.shape[0], positions.device)

        active_labels = fragment_ids.unique(sorted=True)
        impact_center_world = None
        if hasattr(self, '_impact_center'):
            impact_center_world = self.mapper.mpm_to_world(
                self._impact_center.unsqueeze(0)
            ).squeeze(0)
        else:
            impact_center_world = positions.mean(dim=0)
        base_mask = fragment_ids == 0
        base_com = positions[base_mask].mean(dim=0) if bool(base_mask.any()) else impact_center_world

        for frag_id in active_labels.tolist():
            if frag_id <= 0:
                continue
            mask = fragment_ids == frag_id
            frag_size = int(mask.sum().item())
            if frag_size < self.fragment_render_min_size:
                continue

            overlap_vals, overlap_counts = self._render_fragment_labels[mask].unique(
                return_counts=True
            )
            persistent_label = None
            if overlap_vals.numel() > 0:
                overlap_mask = overlap_vals > 0
                if bool(overlap_mask.any()):
                    overlap_vals = overlap_vals[overlap_mask]
                    overlap_counts = overlap_counts[overlap_mask]
                    best_idx = int(torch.argmax(overlap_counts).item())
                    best_count = int(overlap_counts[best_idx].item())
                    if best_count / max(frag_size, 1) >= self.fragment_render_overlap_threshold:
                        persistent_label = int(overlap_vals[best_idx].item())

            frag_pos = positions[mask]
            frag_com = frag_pos.mean(dim=0)
            release_score = 0.0
            support_lost = False
            if (self.fragment_manager is not None
                    and frag_id < len(self.fragment_manager.fragment_release_scores)):
                release_score = float(self.fragment_manager.fragment_release_scores[frag_id])
            if (self.fragment_manager is not None
                    and frag_id < len(self.fragment_manager.fragment_support_lost)):
                support_lost = bool(self.fragment_manager.fragment_support_lost[frag_id])

            direction = frag_com - impact_center_world
            if support_lost:
                direction = frag_com - base_com
                direction[2] -= 0.28 + 0.48 * release_score
            else:
                direction[2] += 0.22 * self.fragment_upward_bias
            direction = self._safe_vector_normalize(direction.unsqueeze(0)).squeeze(0)
            if float(direction.norm().item()) < 1e-8:
                direction = torch.tensor([0.0, 0.0, -1.0 if support_lost else 1.0], device=positions.device)

            size_ratio = frag_size / max(float(positions.shape[0]), 1.0)
            size_gain = max(0.85, 1.70 - 10.0 * size_ratio)
            opening_mag = 0.0
            if opening is not None:
                opening_mag = float(opening[mask].mean().item())
            event_gain = self._current_fragment_visual_boost()
            base_gap = (
                self.fragment_visual_offset_scale
                * self.fragment_offset_gain
                * self.fragment_render_gap_scale
                * size_gain
                * max(event_gain, 1.0)
                * (1.0 + 4.0 * opening_mag)
            )
            vel_mag = (
                self.fragment_visual_offset_scale
                * self.fragment_render_velocity_scale
                * size_gain
                * (0.55 + 0.35 * max(event_gain - 1.0, 0.0) + 6.0 * opening_mag)
            )
            if support_lost:
                base_gap *= 1.0 + 0.95 * release_score
                vel_mag *= 1.0 + 1.35 * release_score

            if persistent_label is None:
                persistent_label = self._next_render_fragment_id
                self._next_render_fragment_id += 1
                self._render_fragment_states[persistent_label] = {
                    "offset": direction * base_gap,
                    "velocity": direction * vel_mag,
                    "age": 0,
                    "release_score": release_score,
                    "support_lost": support_lost,
                }
            else:
                state = self._render_fragment_states.get(persistent_label)
                if state is not None:
                    target_vel = direction * vel_mag
                    state["velocity"] = 0.72 * state["velocity"] + 0.28 * target_vel
                    state["offset"] = state["offset"] + direction * (0.20 * base_gap)
                    state["release_score"] = max(float(state.get("release_score", 0.0)), release_score)
                    state["support_lost"] = bool(state.get("support_lost", False) or support_lost)

            self._render_fragment_labels[mask] = persistent_label

    def _apply_persistent_fragment_separation(
        self,
        positions: Tensor,
        fragment_ids: Optional[Tensor],
        opening: Optional[Tensor] = None,
    ) -> Tensor:
        self._ensure_render_fragment_registry(positions.shape[0], positions.device)
        if fragment_ids is not None and bool((fragment_ids > 0).any()):
            self._register_render_fragments(positions, fragment_ids, opening)

        render_ids = self._render_fragment_labels.clone()
        out = positions.clone()
        if self._render_fragment_states:
            for persistent_label, state in self._render_fragment_states.items():
                mask = render_ids == persistent_label
                if not bool(mask.any()):
                    continue
                out[mask] = out[mask] + state["offset"].unsqueeze(0)
        self._advance_render_fragment_states()
        return out, render_ids

    def _apply_fragment_visual_offset(
        self,
        positions: Tensor,
        frag_ids: Tensor,
    ) -> Tensor:
        if self.fragment_visual_offset_scale <= 0.0:
            return positions
        if self._fragment_activation_frame < 0:
            return positions
        if frag_ids is None or frag_ids.numel() == 0:
            return positions

        frames_since = max(self.frame_count - self._fragment_activation_frame, 0)
        ramp = min((frames_since + 1) / float(self.fragment_visual_ramp_frames), 1.0)
        offset_scale = self.fragment_visual_offset_scale * self.fragment_offset_gain * ramp
        if frames_since < self.fragment_impulse_boost_frames:
            offset_scale *= self._current_fragment_visual_boost()
        if offset_scale <= 0.0:
            return positions

        out = positions.clone()
        impact_center_world = None
        if hasattr(self, '_impact_center'):
            impact_center_world = self.mapper.mpm_to_world(
                self._impact_center.unsqueeze(0)).squeeze(0)
        else:
            impact_center_world = positions.mean(dim=0)

        unique_ids = frag_ids.unique(sorted=True)
        total = float(max(frag_ids.shape[0], 1))
        for frag_id in unique_ids.tolist():
            if frag_id == 0:
                continue
            mask = frag_ids == frag_id
            if not bool(mask.any()):
                continue
            frag_pos = positions[mask]
            com = frag_pos.mean(dim=0)
            direction = com - impact_center_world
            direction[2] += 0.35 * self.fragment_upward_bias
            norm = direction.norm().clamp(min=1e-8)
            direction = direction / norm
            strength = min(mask.sum().item() / total, 0.35)
            out[mask] = out[mask] + direction.unsqueeze(0) * offset_scale * max(0.6, 1.4 - 2.0 * strength)
        return out

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

        # Build persistent fragment render state.
        debris_mask = None
        surf_frag = None
        if self.fragmentation_active and self.fragment_manager is not None:
            frag_ids = None
            if (self._physical_fragment_labels is not None
                    and self._surface_indices is not None
                    and bool((self._physical_fragment_labels > 0).any())):
                frag_ids = self._physical_fragment_labels[self._surface_indices]
            elif self.fragment_manager.n_fragments > 1:
                frag_ids = self.fragment_manager.fragment_ids
            if frag_ids is not None:
                if self._ply_direct:
                    surf_frag = frag_ids[self._ply_to_surface]
                else:
                    N_gauss = x_final.shape[0]
                    surf_frag = frag_ids[:N_gauss]

        if self.fragmentation_active:
            opening_vis = (
                self.fracture_field.a if not self._ply_direct else (
                    self.fracture_field.a[self._ply_to_surface]
                    if self.fracture_field.a is not None else None
                )
            )
            x_final, surf_frag = self._apply_persistent_fragment_separation(
                x_final,
                surf_frag,
                opening=opening_vis,
            )

        c_visual = c_final

        crack_tips = (
            self.fracture_field.crack_front.tip_mask
            if (hasattr(self.fracture_field, "crack_front") and not self._ply_direct)
            else (
                self.fracture_field.crack_front.tip_mask[self._ply_to_surface]
                if (
                    hasattr(self.fracture_field, "crack_front")
                    and self.fracture_field.crack_front.tip_mask is not None
                    and self._ply_direct
                )
                else None
            )
        )
        crack_visited = (
            self.fracture_field.crack_front.visited_mask
            if (hasattr(self.fracture_field, "crack_front") and not self._ply_direct)
            else (
                self.fracture_field.crack_front.visited_mask[self._ply_to_surface]
                if (
                    hasattr(self.fracture_field, "crack_front")
                    and self.fracture_field.crack_front.visited_mask is not None
                    and self._ply_direct
                )
                else None
            )
        )

        # Update visualizer
        self.visualizer.update_gaussians(
            self.gaussians, c_visual, x_final,
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
            crack_tips=crack_tips,
            crack_visited=crack_visited,
            fragment_ids=surf_frag,
        )

        shard_mask = None
        self.splitter._last_append_parent_idx = None
        self.splitter._last_metrics = {
            "visible_shard_count": 0,
            "split_gap_visibility": 0.0,
            "fragment_shell_contrast": 0.0,
            "shard_persistence": 0.0,
        }

        render_positions = self.gaussians._xyz.data.detach()
        base_render_count = int(c_visual.shape[0]) if c_visual is not None else int(x_final.shape[0])
        if render_positions.shape[0] != base_render_count:
            render_positions = render_positions[:base_render_count]

        self._last_render_state = self.splitter.compose_render_state(
            render_positions,
            c_visual,
            opening=(
                self.fracture_field.a if not self._ply_direct else (
                    self.fracture_field.a[self._ply_to_surface]
                    if self.fracture_field.a is not None else None
                )
            ),
            fragment_ids=surf_frag,
            crack_tips=crack_tips,
            crack_visited=crack_visited,
            debris_mask=debris_mask,
        )
        self._last_render_state["interior_surface_count"] = int(
            getattr(self.visualizer, "_last_interior_count", 0)
        )
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
        detached_distance = 0.0
        mean_detached_distance = 0.0
        physical_detached_distance = 0.0
        physical_mean_detached_distance = 0.0
        physical_fragment_drop = 0.0
        split_event_age = -1
        z_min = 0.0
        z_com = 0.0
        v_com_z = 0.0
        if self.x_mpm is not None:
            z_min = float(self.x_mpm[:, 2].min().item())
            z_com = float(self.x_mpm[:, 2].mean().item())
        if self._v_com is not None:
            v_com_z = float(self._v_com[2].item())
        render_state = self._last_render_state or {}
        if self.fragmentation_active and self._fragment_activation_frame >= 0:
            split_event_age = max(self.frame_count - self._fragment_activation_frame, 0)
        render_positions = render_state.get("positions", None)
        render_fragment_ids = render_state.get("fragment_ids", None)
        render_n_frags = 0
        if render_fragment_ids is not None and render_fragment_ids.numel() > 0:
            if bool((render_fragment_ids > 0).any()):
                render_n_frags = int(render_fragment_ids.max().item()) + 1
            if render_n_frags > 1 and render_positions is not None:
                coms = []
                for frag_id in range(render_n_frags):
                    mask = render_fragment_ids == frag_id
                    if not bool(mask.any()):
                        continue
                    coms.append(render_positions[mask].mean(dim=0))
                if len(coms) > 1:
                    base = coms[0]
                    distances = [
                        float((com - base).norm().item())
                        for com in coms[1:]
                    ]
                    detached_distance = max(distances)
                    mean_detached_distance = sum(distances) / max(len(distances), 1)
        physical_n_frags = 0
        frag_ids_phys = None
        if (self._physical_fragment_labels is not None
                and self._surface_indices is not None
                and bool((self._physical_fragment_labels > 0).any())):
            frag_ids_phys = self._physical_fragment_labels[self._surface_indices]
            physical_n_frags = int(frag_ids_phys.max().item()) + 1
        elif self.fragment_manager is not None and self.fragment_manager.fragment_ids is not None:
            frag_ids_phys = self.fragment_manager.fragment_ids
            physical_n_frags = int(frag_ids_phys.max().item()) + 1 if bool((frag_ids_phys > 0).any()) else 1

        if frag_ids_phys is not None and physical_n_frags > 1:
            x_surf_world = self.mapper.mpm_to_world(self.x_mpm[self.surface_mask])
            n_assign = min(x_surf_world.shape[0], frag_ids_phys.shape[0])
            if n_assign > 0:
                x_phys = x_surf_world[:n_assign]
                frag_ids = frag_ids_phys[:n_assign]
                coms = []
                for frag_id in frag_ids.unique(sorted=True).tolist():
                    mask = frag_ids == frag_id
                    if not bool(mask.any()):
                        continue
                    coms.append((frag_id, x_phys[mask].mean(dim=0)))
                if len(coms) > 1:
                    base = coms[0][1]
                    distances = [
                        float((com - base).norm().item())
                        for frag_id, com in coms[1:]
                    ]
                    drops = [
                        max(float(base[2].item() - com[2].item()), 0.0)
                        for frag_id, com in coms[1:]
                    ]
                    physical_detached_distance = max(distances)
                    physical_mean_detached_distance = sum(distances) / max(len(distances), 1)
                    physical_fragment_drop = max(drops) if drops else 0.0
        n_frags_out = max(physical_n_frags, render_n_frags)

        return {
            "frame": self.frame_count,
            "time": self.mpm.time,
            "z_min": z_min,
            "z_com": z_com,
            "v_com_z": v_com_z,
            "gravity_drop": bool(self._gravity_drop),
            "gravity_contacted": bool(self._gravity_drop_contacted),
            "gravity_ground_z": float(self._gravity_drop_ground_z),
            "c_max": c_max,
            "c_mean": c_mean,
            "n_cracked": n_cracked,
            "n_fragments": n_frags_out,
            "broken_edges": (self.fragment_manager.last_broken_edges
                              if self.fragment_manager else 0),
            "raw_components": (self.fragment_manager.last_raw_components
                                if self.fragment_manager else 1),
            "promoted_components": (self.fragment_manager.last_promoted_components
                                     if self.fragment_manager else 0),
            "primary_promoted_components": (self.fragment_manager.last_primary_promoted_components
                                             if self.fragment_manager else 0),
            "fallback_promoted_components": (self.fragment_manager.last_fallback_promoted_components
                                              if self.fragment_manager else 0),
            "cut_core_nodes": (self.fragment_manager.last_cut_core_nodes
                                 if self.fragment_manager else 0),
            "cut_edges": (self.fragment_manager.last_cut_edges
                            if self.fragment_manager else 0),
            "cut_corridor_edges": (self.fragment_manager.last_cut_corridor_edges
                                    if self.fragment_manager else 0),
            "cross_edge_breaks": (self.fragment_manager.last_cross_edge_breaks
                                    if self.fragment_manager else 0),
            "authoritative_cut_nodes": (self.fragment_manager.last_authoritative_cut_nodes
                                          if self.fragment_manager else 0),
            "authoritative_cut_score_max": (self.fragment_manager.last_authoritative_cut_score_max
                                             if self.fragment_manager else 0.0),
            "support_lost_components": (self.fragment_manager.last_support_lost_components
                                         if self.fragment_manager else 0),
            "support_loss_score_max": (self.fragment_manager.last_support_loss_score_max
                                        if self.fragment_manager else 0.0),
            "release_candidate_count": (self.fragment_manager.last_release_candidate_count
                                         if self.fragment_manager else 0),
            "support_anchor_nodes": (self.fragment_manager.last_support_anchor_nodes
                                       if self.fragment_manager else 0),
            "boundary_cut_ratio_mean": (self.fragment_manager.last_boundary_cut_ratio_mean
                                         if self.fragment_manager else 0.0),
            "boundary_cut_ratio_q50": (self.fragment_manager.last_boundary_cut_ratio_q50
                                        if self.fragment_manager else 0.0),
            "boundary_cut_ratio_q90": (self.fragment_manager.last_boundary_cut_ratio_q90
                                        if self.fragment_manager else 0.0),
            "boundary_cut_ratio_max": (self.fragment_manager.last_boundary_cut_ratio_max
                                        if self.fragment_manager else 0.0),
            "components_above_primary": (self.fragment_manager.last_components_above_primary
                                          if self.fragment_manager else 0),
            "components_above_fallback": (self.fragment_manager.last_components_above_fallback
                                           if self.fragment_manager else 0),
            "absorbed_components": (self.fragment_manager.last_absorbed_components
                                     if self.fragment_manager else 0),
            "closure_candidate_count": (self.fragment_manager.last_closure_candidate_count
                                         if self.fragment_manager else 0),
            "closure_candidate_nodes": (self.fragment_manager.last_closure_candidate_nodes
                                         if self.fragment_manager else 0),
            "closure_score_max": (self.fragment_manager.last_closure_score_max
                                   if self.fragment_manager else 0.0),
            "closure_candidate_sizes": (self.fragment_manager.last_closure_candidate_sizes
                                         if self.fragment_manager else []),
            "open_release_patches": (self.fragment_manager.last_open_release_patches
                                      if self.fragment_manager else 0),
            "open_release_nodes": (self.fragment_manager.last_open_release_nodes
                                   if self.fragment_manager else 0),
            "open_release_score_max": (self.fragment_manager.last_open_release_score_max
                                       if self.fragment_manager else 0.0),
            "catastrophic_release_patches": (self.fragment_manager.last_catastrophic_release_patches
                                      if self.fragment_manager else 0),
            "catastrophic_release_nodes": (self.fragment_manager.last_catastrophic_release_nodes
                                   if self.fragment_manager else 0),
            "catastrophic_release_score_max": (self.fragment_manager.last_catastrophic_release_score_max
                                       if self.fragment_manager else 0.0),
            "secondary_shatter_patches": (self.fragment_manager.last_secondary_shatter_patches
                                      if self.fragment_manager else 0),
            "secondary_shatter_nodes": (self.fragment_manager.last_secondary_shatter_nodes
                                   if self.fragment_manager else 0),
            "secondary_shatter_score_max": (self.fragment_manager.last_secondary_shatter_score_max
                                       if self.fragment_manager else 0.0),
            "top_component_sizes": (self.fragment_manager.last_top_component_sizes
                                       if self.fragment_manager else []),
            "detached_distance": detached_distance,
            "mean_detached_distance": mean_detached_distance,
            "physical_detached_distance": physical_detached_distance,
            "physical_mean_detached_distance": physical_mean_detached_distance,
            "physical_fragment_drop": physical_fragment_drop,
            "split_event_age": split_event_age,
            "visible_shard_count": int(render_state.get("visible_shard_count", 0)),
            "interior_surface_count": int(render_state.get("interior_surface_count", 0)),
            "split_gap_visibility": float(render_state.get("split_gap_visibility", 0.0)),
            "fragment_shell_contrast": float(render_state.get("fragment_shell_contrast", 0.0)),
            "shard_persistence": float(render_state.get("shard_persistence", 0.0)),
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
