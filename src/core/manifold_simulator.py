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

import numpy as np
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
from src.fracture.voronoi_pipeline import VoronoiPipelineMixin
from src.core.simulator_mixins import (
    SurfaceBindingMixin,
    FragmentEventStatsMixin,
    FragmentPhysicsMixin,
    RuntimeProfilesMixin,
    FractureDriveMixin,
    RenderFragmentMixin,
)


class ManifoldSimulator(
    SurfaceBindingMixin,
    FragmentEventStatsMixin,
    FragmentPhysicsMixin,
    RuntimeProfilesMixin,
    FractureDriveMixin,
    RenderFragmentMixin,
    VoronoiPipelineMixin,
):
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
            growth_griffith_threshold=fp.get('growth_griffith_threshold', 0.50),
            branch_direction_mode=fp.get('branch_direction_mode', 'energy'),
            branch_angle_prior_floor=fp.get('branch_angle_prior_floor', 0.50),
            branch_event_topk=fp.get('branch_event_topk', 2),
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
            at2_jacobi_enable=fp.get('at2_jacobi_enable', True),
            at2_drive_gain=fp.get('at2_drive_gain', 1.0),
            at2_reg_gain=fp.get('at2_reg_gain', 1.0),
            at2_dc_fraction=fp.get('at2_dc_fraction', 0.5),
            curvature_weight=fp.get('curvature_weight', 0.0),
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

        self.material_family = str(fp.get('material_family', 'neutral_reference'))
        self.crack_connected_release_only = bool(
            fp.get(
                'crack_connected_release_only',
                self.material_family != 'diffuse_damage',
            )
        )
        burst_defaults = self._default_impact_fracture_burst(self.material_family)
        self.impact_fracture_burst_frames = max(
            int(fp.get('impact_fracture_burst_frames', burst_defaults["frames"])), 0)
        self.impact_fracture_burst_steps = max(
            int(fp.get('impact_fracture_burst_steps', burst_defaults["steps"])), 1)
        self.impact_fracture_burst_front_substeps = max(
            int(fp.get('impact_fracture_burst_front_substeps', burst_defaults["front_substeps"])), 1)
        self._fracture_burst_active = False
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
            persistent_min_fragment_size_ratio=fp.get('fragment_persistent_min_size_ratio', 0.0),
            persistent_min_fragment_reference_nodes=fp.get(
                'fragment_persistent_reference_nodes', 10000),
            persistent_min_fragment_resolution_exponent=fp.get(
                'fragment_persistent_resolution_exponent', 0.5),
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
            fragment_boundary_cut_min_ratio=fp.get('fragment_boundary_cut_min_ratio', 0.0),
            open_crack_release_enable=fp.get('open_crack_release_enable', True),
            open_crack_release_threshold=fp.get('open_crack_release_threshold', 0.0),
            open_crack_release_max_patches=fp.get('open_crack_release_max_patches', 2),
            crack_connected_release_only=self.crack_connected_release_only,
            strict_closure_max_released_ratio=fp.get('strict_closure_max_released_ratio', 0.54),
            crack_style=fp.get('sentence_style', fp.get('crack_style', 'material_default')),
            brittle_release_intensity=fp.get('brittle_release_intensity', 1.0),
            impact_release_gain=fp.get('impact_release_gain', 1.0),
            impact_closure_target_ratio=fp.get('impact_closure_target_ratio', -1.0),
            impact_closure_max_patches=fp.get('impact_closure_max_patches', -1),
            impact_closure_active_frames=fp.get('impact_closure_active_frames', 2),
            impact_closure_sector_count=fp.get('impact_closure_sector_count', 0),
            impact_closure_band_count=fp.get('impact_closure_band_count', 0),
            impact_closure_layer_count=fp.get('impact_closure_layer_count', 0),
            impact_closure_min_size_ratio=fp.get('impact_closure_min_size_ratio', 0.0),
            impact_closure_max_size_ratio=fp.get('impact_closure_max_size_ratio', 0.0),
            impact_closure_adaptive_extra_frames=fp.get(
                'impact_closure_adaptive_extra_frames', 6),
            impact_closure_completion_ratio=fp.get(
                'impact_closure_completion_ratio', 0.92),
            phase_approval_enable=fp.get('phase_approval_enable', True),
            phase_approval_threshold_scale=fp.get('phase_approval_threshold_scale', 1.0),
            phase_approval_threshold_offset=fp.get('phase_approval_threshold_offset', 0.0),
            phase_cc_modulation_enable=fp.get('phase_cc_modulation_enable', True),
            material_family=fp.get('material_family', 'neutral_reference'),
            device=device_str,
        ) if frag_enabled else None
        self.fragmentation_active = False
        self.fragment_detect_every = max(int(fp.get('fragment_detect_every', 2)), 1)
        phase_defaults = self._default_phase_y_thresholds(self.material_family)
        self.phase_y_coupling_enabled = bool(
            fp.get('phase_y_coupling_enabled', True))
        self.phase_y_seed_threshold = float(
            fp.get('phase_y_seed_threshold', phase_defaults["seed"]))
        self.phase_y_advance_threshold = float(
            fp.get('phase_y_advance_threshold', phase_defaults["advance"]))
        self.phase_y_cut_threshold = float(
            fp.get('phase_y_cut_threshold', phase_defaults["cut"]))
        self.phase_y_gate_width = float(
            fp.get('phase_y_gate_width', phase_defaults["width"]))
        self._last_phase_y_gauss: Optional[Tensor] = None
        self._last_phase_seed_gate: Optional[Tensor] = None
        self._last_phase_advance_gate: Optional[Tensor] = None
        self._last_phase_cut_gate: Optional[Tensor] = None
        self.fragment_impulse_strength = float(fp.get('fragment_impulse_strength', 2.8))
        self.fragment_upward_bias = float(fp.get('fragment_upward_bias', 0.45))
        self.fragment_visual_offset_scale = float(fp.get('fragment_visual_offset_scale', 0.012))
        self.fragment_visual_ramp_frames = max(int(fp.get('fragment_visual_ramp_frames', 6)), 1)
        self.fragment_impulse_boost_frames = max(int(fp.get('fragment_impulse_boost_frames', 5)), 0)
        self.fragment_impulse_decay = float(fp.get('fragment_impulse_decay', 0.75))
        self.fragment_event_boost = float(fp.get('fragment_event_boost', 1.0))
        self._fragment_activation_frame = -1
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
        # Render-fragment label tracking only.  All offset/velocity
        # parameters (fragment_render_gravity / damping / max_detach /
        # gravity_scale / strict_*) were removed when the render
        # offset path was deleted — fragments now render at raw MPM
        # positions.  fragment_render_min_size + overlap_threshold
        # are kept since they gate label persistence.
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
        self._next_render_fragment_id = 1
        self._physical_fragment_labels: Optional[Tensor] = None
        self._physical_fragment_states = {}
        self._rigid_fragment_states: Dict[int, Dict] = {}
        self._next_physical_fragment_id = 1
        self._spatial_split_prev_labels = None
        self._spatial_split_next_id = 1
        self._reset_frame_fragment_event_stats()

        # Post-impact damage→stress feedback timing.  Original baseline used
        # 8-frame delay + 6-frame ramp = 14 decoupled frames, leaving
        # shape-matching as the dominant cohesion in the most informative
        # window.  Conservative tightening (4+2 = 6 frames) keeps stability
        # while letting damage influence stress earlier.
        self.damage_feedback_delay_frames = int(
            fp.get('damage_feedback_delay_frames', 4))
        self.damage_feedback_ramp_frames = int(
            fp.get('damage_feedback_ramp_frames', 2))
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
        immediate_defaults = {
            "sharp_brittle": 0.72,
            "brittle_moderate": 0.48,
            "rough_quasi_brittle": 0.42,
            "neutral_reference": 0.18,
            "diffuse_damage": 0.0,
        }
        self.crack_volume_immediate_feedback = float(
            fp.get(
                'crack_volume_immediate_feedback',
                immediate_defaults.get(self.material_family, 0.18),
            )
        )
        self.fragment_physical_gap_scale = float(
            fp.get('fragment_physical_gap_scale', 0.0))
        self.fragment_physical_release_velocity = float(
            fp.get('fragment_physical_release_velocity', 0.0))
        self.fragment_physical_downward_bias = float(
            fp.get('fragment_physical_downward_bias', 0.32))
        self.fragment_physical_release_frames = max(
            int(fp.get('fragment_physical_release_frames', 20)), 0)
        shape_defaults = self._default_shape_matching_params(self.material_family)
        self.fragment_physical_lateral_bias = float(
            fp.get('fragment_physical_lateral_bias', shape_defaults.get("fragment_lateral_bias", 0.0)))
        self.fragment_physical_spin_gain = float(
            fp.get('fragment_physical_spin_gain', shape_defaults.get("fragment_spin_gain", 0.0)))
        self.fragment_release_jitter = max(0.0, float(
            fp.get('fragment_release_jitter', 0.0)))
        # Voronoi Mode-I bond opening kick magnitude (m/s).  Each broken
        # bond gives equal-and-opposite kicks of magnitude
        # `v_com_gain * sqrt(bond_stress_at_break)` to its two cells
        # along the bond direction.  Clamped to
        # `fragment_physical_max_speed`.  0.0 disables (cells separate
        # purely from inherited body velocity + gravity).
        self.fragment_release_v_com_gain = max(0.0, float(
            fp.get('fragment_release_v_com_gain', 0.0)))
        # Position offset (MPM-space) along kick direction at fragment
        # graduation; decouples chunk from cohesive base body in the
        # shared MPM grid so the kick survives grid gather/scatter.
        self.fragment_release_position_offset = max(0.0, float(
            fp.get('fragment_release_position_offset', 0.0)))
        # Spread fracture release over a few physics ticks.  A crack event
        # can break many Voronoi bonds at once; applying the whole opening
        # velocity and geometric separation in the same tick reads as an
        # explosion.  The default preserves legacy behavior.
        self.fragment_release_impulse_ramp_substeps = max(
            int(fp.get('fragment_release_impulse_ramp_substeps', 1)), 1)
        # Velocity-clamp scale: 1.0 = standard CFL-safe cap; >1.0 lets
        # fragments fall faster than the dx/dt limit (use with caution
        # under aggressive gravity, may cause numerical jitter).
        self.velocity_cap_scale = max(0.1, float(
            fp.get('velocity_cap_scale', 1.0)))
        # Per-particle speed magnitude cap (m/s).  Default 10 was too
        # conservative for brittle pulverization under strong gravity
        # (-4500+); fragments stall at 10 m/s instead of falling
        # freely.  Increase to 60-100 for fast brittle fall.
        self.particle_speed_cap = max(1.0, float(
            fp.get('particle_speed_cap', 10.0)))
        # Extra per-substep downward acceleration on fragment-labeled
        # particles only; additive on top of the global post-impact
        # gravity (`post_impact_gravity_z`).  Compensates for the fact
        # that strong post-impact damping tuned to suppress base-body
        # elastic vibration also bleeds the gravity-induced velocity of
        # small detached fragments.  Negative = downward.  0.0 disables.
        self.fragment_extra_gravity_z = float(
            fp.get('fragment_extra_gravity_z', 0.0))
        # Per-particle floor bounce on fragment-labeled particles.  The
        # MPM grid uses a "slip" BC at the floor which zeros the normal
        # velocity component as part of p2g2p, so any per-fragment
        # rigid_contact_impulse runs after slip and sees v_z=0 (the
        # restitution kick has nothing to bounce off of).  This knob
        # restores the bounce by recovering pre-p2g2p v_z and applying
        # `v_z = -restitution * v_z_pre` for fragment particles inside
        # the floor band.  0.0 disables.
        self.fragment_floor_restitution = max(0.0, min(0.95, float(
            fp.get('fragment_floor_restitution', 0.0))))
        self.fragment_physical_max_speed = float(
            fp.get('fragment_physical_max_speed', shape_defaults.get("fragment_max_speed", 0.35)))
        self.shape_matching_enabled = bool(
            fp.get('shape_matching_enabled', True))
        self.shape_match_strength = float(
            fp.get('shape_match_strength', shape_defaults["body"]))
        self.shape_match_fragment_strength = float(
            fp.get('shape_match_fragment_strength', shape_defaults["fragment"]))
        self.shape_match_damaged_strength = float(
            fp.get('shape_match_damaged_strength', shape_defaults["damaged"]))
        self.shape_match_velocity_blend = float(
            fp.get('shape_match_velocity_blend', 0.35))
        # Affine shape matching (Müller et al. 2005, "Meshless
        # Deformations Based on Shape Matching").  When > 0, the
        # closest *rigid* rotation R is interpolated with the optimal
        # *affine* transform A computed via least-squares fit:
        #     A = (sum p_i q_i^T) (sum q_i q_i^T)^-1
        # T = (1 - blend) * R + blend * A.  blend=0 reproduces the
        # rigid Procrustes shape match (legacy).  blend=1 enables full
        # affine deformation modes (uniform compression / shear /
        # scaling) which lets a body squash isotropically without
        # creating a head/body deformation band — necessary for
        # rubber-like soft impact.  Brittle families keep blend=0
        # so fragments stay rigid in flight.
        self.shape_match_affine_blend = float(
            fp.get('shape_match_affine_blend', 0.0))
        # SV clamp for affine deformation: per-axis stretch limits.
        # Defaults [0.78, 1.0] = 22% max squash, no vertical run-away.
        self.shape_match_sv_min = float(
            fp.get('shape_match_sv_min', 0.78))
        self.shape_match_sv_max = float(
            fp.get('shape_match_sv_max', 1.00))
        # Post-impact kinematic lock: for ``post_impact_kinematic_lock_frames``
        # frames immediately after the first ground contact, every
        # particle's velocity is forced to the body COM velocity.
        # Used by non-fragment-capable families (rubber, wood, metal)
        # to suppress the 2-3 frame residual MPM grid-induced cluster
        # split at high impact velocity that shape match alone cannot
        # fully prevent.  Default 0 = no lock (legacy behaviour).
        self.post_impact_kinematic_lock_frames = int(
            fp.get('post_impact_kinematic_lock_frames', 0))
        self._kinematic_lock_remaining = 0
        # Per-substep multiplicative decay applied to the shape-matched
        # angular velocity once the body has hit the ground.  Acts as a
        # surrogate for kinetic ground friction; free-fall rotation is
        # preserved.  `1.0` disables (legacy behavior, body spins
        # indefinitely).
        self.shape_match_angular_damping = float(
            fp.get('shape_match_angular_damping', 0.985))
        # Static-friction surrogate: when omega.norm() falls below this
        # threshold (rad/s), apply `shape_match_static_damping` per
        # substep instead of the kinetic-friction value.  Without a
        # static cutoff the contact impulse loop sustains tiny rotations
        # from numerical noise indefinitely (body never reaches rest).
        self.shape_match_static_omega = float(
            fp.get('shape_match_static_omega', 8.0))
        self.shape_match_static_damping = float(
            fp.get('shape_match_static_damping', 0.5))
        self.shape_match_min_particles = max(
            int(fp.get('shape_match_min_particles', 12)), 1)
        self.rigid_contact_enabled = bool(
            fp.get('rigid_contact_enabled', True))
        self.rigid_contact_restitution = float(
            fp.get('rigid_contact_restitution', shape_defaults["restitution"]))
        self.rigid_contact_angular_gain = float(
            fp.get('rigid_contact_angular_gain', shape_defaults["angular_gain"]))
        self.rigid_contact_friction = float(
            fp.get('rigid_contact_friction', shape_defaults["friction"]))
        self.rigid_contact_band = float(
            fp.get('rigid_contact_band', 1.5))
        self.detached_fragment_dynamics = str(
            fp.get('detached_fragment_dynamics', 'mpm_shape_match'))
        rigid_handoff_min_default = max(
            self.fragment_physical_min_size,
            self.fragment_render_min_size * 4,
            24,
        )
        self.rigid_handoff_min_particles = max(
            int(fp.get('rigid_handoff_min_particles', rigid_handoff_min_default)),
            1,
        )
        self.rigid_handoff_attach_tiny_to_nearest_fragment = bool(
            fp.get('rigid_handoff_attach_tiny_to_nearest_fragment', True))
        self.rigid_handoff_merge_tiny_fragments = bool(
            fp.get('rigid_handoff_merge_tiny_fragments', False))
        self.rigid_handoff_restitution = float(
            fp.get('rigid_handoff_restitution', self.rigid_contact_restitution))
        self.rigid_handoff_floor_friction = float(
            fp.get('rigid_handoff_floor_friction',
                   max(self.rigid_contact_friction, 0.45)))
        self.rigid_handoff_floor_angular_friction = max(0.0, float(
            fp.get('rigid_handoff_floor_angular_friction', 0.0)))
        # Rigid handoff birth controls.  These are initial-condition
        # corrections at the moment a Voronoi fragment becomes mechanically
        # independent: cap lateral COM energy from accumulated crack-opening
        # kicks, and optionally transfer some release energy into downward
        # motion so shards fall instead of hovering near the parent body.
        # 0.0 disables each knob.
        self.rigid_handoff_birth_lateral_velocity_cap = max(0.0, float(
            fp.get('rigid_handoff_birth_lateral_velocity_cap', 0.0)))
        self.rigid_handoff_birth_inherited_lateral_velocity_scale = max(
            0.0,
            min(float(fp.get(
                'rigid_handoff_birth_inherited_lateral_velocity_scale', 1.0)),
                1.0),
        )
        self.rigid_handoff_birth_downward_velocity = max(0.0, float(
            fp.get('rigid_handoff_birth_downward_velocity', 0.0)))
        self.rigid_handoff_birth_angular_velocity_scale = max(0.0, float(
            fp.get('rigid_handoff_birth_angular_velocity_scale', 0.0)))
        self.rigid_handoff_birth_body_angular_velocity_scale = max(0.0, float(
            fp.get('rigid_handoff_birth_body_angular_velocity_scale', 0.0)))
        self.rigid_handoff_birth_tumble_gain = max(0.0, float(
            fp.get('rigid_handoff_birth_tumble_gain', 0.0)))
        self.rigid_handoff_birth_angular_velocity_cap = max(0.0, float(
            fp.get('rigid_handoff_birth_angular_velocity_cap', 0.0)))
        self.rigid_handoff_pair_contact_birth_slop = max(0.0, float(
            fp.get('rigid_handoff_pair_contact_birth_slop', 2.0)))
        self.rigid_handoff_floor_contact_torque = bool(
            fp.get('rigid_handoff_floor_contact_torque', False))
        self.rigid_handoff_floor_contact_angular_gain = max(0.0, float(
            fp.get('rigid_handoff_floor_contact_angular_gain',
                   self.rigid_contact_angular_gain)))
        self.soft_contact_rebound = float(
            fp.get('soft_contact_rebound', shape_defaults.get("soft_rebound", 0.0)))
        self.soft_contact_rebound_frames = max(
            int(fp.get('soft_contact_rebound_frames', shape_defaults.get("soft_rebound_frames", 0))), 0)
        self.soft_contact_rebound_body_fraction = float(
            fp.get('soft_contact_rebound_body_fraction', shape_defaults.get("soft_rebound_body_fraction", 0.0)))
        self.soft_contact_squash = float(
            fp.get('soft_contact_squash', shape_defaults.get("soft_squash", 0.0)))
        self.soft_contact_squash_frames = max(
            int(fp.get('soft_contact_squash_frames', shape_defaults.get("soft_squash_frames", 0))), 0)
        self._soft_impact_speed = 0.0
        self._last_rigid_angular_speed_max = 0.0
        self._last_rigid_angular_speed_mean = 0.0
        self._shape_match_rest_positions: Optional[Tensor] = None
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

        # --- Voronoi pre-fracture (lazy-init at impact) ---
        self.voronoi = None
        self._voronoi_cell_graduated = []
        self._voronoi_prev_components = None
        self._voronoi_anisotropy_axis = None

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
        self._impact_release_gain = float(fp.get('impact_release_gain', 1.0))

        self.frame_count = 0
        self._physics_step = 0
        self._last_cfl = 0.0
        self._last_stress = None
        self._last_volumetric_damage_max = 0.0
        self._last_volumetric_damage_mean = 0.0
        self._last_topology_damage_max = 0.0
        self._last_topology_damage_mean = 0.0
        # Pipeline rewrite Step 1a: c_mech cache scaffold.  Set per
        # fracture tick by _step_physics; reused by future steps when
        # fracture tick interval > 1 substep.
        self._last_c_mech: Optional[Tensor] = None
        # Pipeline rewrite Step 3a: cached voronoi components-state
        # produced by _voronoi_bond_eval_commit() inside the substep
        # loop.  step_rendering consumes it at frame end via
        # _voronoi_spatial_split_and_refine() when the flag is on.
        self._last_voronoi_components_state: Optional[Dict] = None
        self._pending_voronoi_cell_releases = []
        self._phase_tick_budget_frame: int = -1
        self._phase_tick_budget_total: int = 0
        self._phase_tick_budget_ticks: int = 1
        self._phase_tick_seen: int = 0
        self._phase_tick_done: int = 0
        # Pipeline rewrite Step 2 prep: per-fragment birth-time registries.
        # Populated when manifold.use_physical_fragment_authority is True
        # (also requires _physical_fragment_labels to be the active
        # source of truth).  Step 4b consumes birth_physics_step for
        # substep-counted handoff smoothing; Step 1b consumes birth_time
        # for time-based c_mech anneal.
        self._physical_fragment_birth_time: Dict[int, float] = {}
        self._physical_fragment_birth_physics_step: Dict[int, int] = {}
        self._last_surface_volume_damage_proxy_max = 0.0
        self._last_surface_volume_damage_proxy_mean = 0.0
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
        # Propagate the ground level to the MPM core so its slip-BC
        # floor matches the simulator's collision floor (otherwise
        # the viewer's auto-detected floor and the physics floor
        # disagree, and particles can tunnel below the visible
        # ground before being stopped at the domain bottom).
        self.mpm.ground_z = float(ground_z)
        # Optional grid-level restitution: when > 0, the slip BC
        # reflects v_z (-e * v_z) instead of zeroing it, so soft
        # elastomers actually bounce off the floor at the grid level.
        self.mpm.floor_restitution_grid = float(
            self.fracture_cfg.get('floor_restitution_grid', 0.0))
        self._gravity_drop_contacted = False
        device = self.mpm.gravity.device
        self._v_com = torch.zeros(3, device=device)
        # Initial angular velocity of the body during free-fall (rad / sec).
        # Default zero; can be set non-zero from config to give a tumbling
        # drop, in which case impact velocities reflect the rigid rotation.
        drop_omega = self.fracture_cfg.get('drop_omega', None)
        if drop_omega is None:
            self._omega_com = torch.zeros(3, device=device)
        else:
            self._omega_com = torch.tensor(
                list(drop_omega), device=device, dtype=torch.float32
            )
        if self.x_mpm is not None:
            self._shape_match_rest_positions = self.x_mpm.detach().clone()
        print(f"[GravityDrop] Enabled. Ground at z={ground_z}, "
              f"omega={self._omega_com.tolist()}")

    def initialize(self, init_positions: Tensor):
        """Initialize simulation state from particle positions in [0,1]^3."""
        device = init_positions.device
        N = init_positions.shape[0]

        self.x_mpm = init_positions.clone()
        self.v_mpm = torch.zeros((N, 3), device=device)
        self.F = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3).clone()
        self.C = torch.zeros((N, 3, 3), device=device)
        self.init_positions = init_positions.clone()
        self._shape_match_rest_positions = init_positions.clone()

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
        self._next_render_fragment_id = 1
        self._physical_fragment_labels = None
        self._physical_fragment_states = {}
        self._rigid_fragment_states = {}
        self._next_physical_fragment_id = 1
        self._spatial_split_prev_labels = None
        self._spatial_split_next_id = 1
        # Reset birth registry (Step 2 prep).
        self._physical_fragment_birth_time = {}
        self._physical_fragment_birth_physics_step = {}
        self._reset_frame_fragment_event_stats()
        self._last_rigid_angular_speed_max = 0.0
        self._last_rigid_angular_speed_mean = 0.0
        self._impact_release_gain = float(self.fracture_cfg.get('impact_release_gain', 1.0))
        if self.fragment_manager is not None:
            self.fragment_manager.impact_release_gain = self._impact_release_gain
        self._surface_indices = torch.where(self.surface_mask)[0]
        self._interior_indices = torch.where(~self.surface_mask)[0]
        self._build_rest_surface_binding()
        print(f"[ManifoldSim] Initialized: {N} particles, {N_surf} surface")

    # ----------------------------------------------------------------
    # Physical fragment authority helpers (Pipeline rewrite Step 2)
    # ----------------------------------------------------------------

    def _compute_c_mech(self, c_visual: Tensor) -> Tensor:
        """Map raw phase-field damage ``c_visual`` to MPM-side ``c_mech``.

        Step 1a: cap ``c_mech`` at ``c_cap = 1 - sqrt(g_min)`` so the
        constitutive degradation function ``g(c) = (1-c)^2 + k_residual``
        cannot drop below a floor stiffness ``g_min`` even when the
        phase-field damage saturates.  ``g_min = 0`` (default) leaves
        ``c_mech == c_visual`` and reproduces legacy behaviour.

        Step 1b: anneal ``c_mech`` toward 0 inside detached fragments
        when ``manifold.c_mech_anneal_after_break`` is True.  Each
        physical fragment's age is looked up from
        ``self._physical_fragment_birth_time``; particles in label
        ``L`` get ``c_mech *= exp(-age_seconds / tau)``.  This restores
        mechanical stiffness inside a detached fragment so it behaves
        as a cohesive elastic body, while ``c_visual`` keeps the crack
        appearance for the renderer.  No-op unless the flag is
        enabled and the physical-fragment-authority pipeline is
        active.
        """
        g_min = float(self.fracture_cfg.get('mechanical_stiffness_floor', 0.0))
        if g_min > 0.0:
            c_cap = 1.0 - math.sqrt(max(g_min, 1e-12))
            c_mech = c_visual.clamp(max=c_cap)
        else:
            c_mech = c_visual.clone() if isinstance(c_visual, torch.Tensor) else c_visual

        # Step 1b — fragment-internal damage anneal.
        if (bool(self.fracture_cfg.get('use_physical_fragment_authority', False))
                and bool(self.fracture_cfg.get('c_mech_anneal_after_break', False))
                and self._physical_fragment_labels is not None
                and isinstance(c_mech, torch.Tensor)
                and self._physical_fragment_labels.shape[0] == c_mech.shape[0]
                and len(self._physical_fragment_birth_time) > 0):
            tau = float(self.fracture_cfg.get(
                'c_mech_anneal_tau_seconds', 0.05))
            tau = max(tau, 1e-6)
            cur_t = float(getattr(self.mpm, "time", 0.0))
            decay_scale = torch.ones_like(c_mech)
            for lab, birth_t in self._physical_fragment_birth_time.items():
                lab_int = int(lab)
                if lab_int <= 0:
                    continue
                mask = (self._physical_fragment_labels == lab_int)
                if not bool(mask.any()):
                    continue
                age = max(cur_t - float(birth_t), 0.0)
                decay = math.exp(-age / tau)
                decay_scale = torch.where(
                    mask,
                    torch.full_like(decay_scale, decay),
                    decay_scale,
                )
            c_mech = c_mech * decay_scale

        return c_mech

    def _has_physical_fragments(self) -> bool:
        """Return True iff the voronoi-driven label tensor has any fragment.

        Source of truth for fragment existence once
        ``manifold.use_physical_fragment_authority`` is enabled.  Until
        then this is a passive helper that may be called for
        diagnostics; the legacy gate ``fragment_manager.n_fragments > 1``
        is still authoritative.
        """
        labels = self._physical_fragment_labels
        return labels is not None and bool((labels > 0).any())

    def _fragmented_physics_gate(self) -> bool:
        """Decide whether to take the fragmented MPM path.

        When ``manifold.use_physical_fragment_authority`` is enabled,
        defer to ``_has_physical_fragments()`` (voronoi-driven).
        Otherwise fall back to the legacy graph-fragment counter on
        ``fragment_manager``.  Both paths still require
        ``self.fragmentation_active`` to be True so the upstream
        impact-detection gating is respected.
        """
        if not self.fragmentation_active:
            return False
        if bool(self.fracture_cfg.get('use_physical_fragment_authority', False)):
            return self._has_physical_fragments()
        return (self.fragment_manager is not None
                and self.fragment_manager.n_fragments > 1)

    def _update_physical_fragment_birth_registry(self) -> None:
        """Record (mpm.time, _physics_step) for any new physical fragment label.

        Called from a fracture-tick path once
        ``manifold.use_physical_fragment_authority`` is enabled.  Labels
        are insert-only here; eviction (if any) happens elsewhere when a
        fragment is fully merged or removed.

        The registry also seeds ``_physical_fragment_states`` for the
        label.  Fragment mechanics and diagnostics read ``birth_com`` /
        ``birth_base_com`` from that dict; when the physical-authority
        migration made Voronoi labels the source of truth, the old graph
        path stopped populating those fields, so release-motion metrics
        stayed at zero and the physical release-drift gate had no useful
        birth state.
        """
        labels = self._physical_fragment_labels
        if labels is None:
            return
        cur_t = float(getattr(self.mpm, "time", 0.0))
        cur_s = int(getattr(self, "_physics_step", 0))
        unique = torch.unique(labels)
        base_mask = labels == 0
        base_com = None
        if (self.x_mpm is not None
                and labels.shape[0] == self.x_mpm.shape[0]
                and bool(base_mask.any())):
            base_com = self.x_mpm[base_mask].mean(dim=0).detach().clone()
        elif self.x_mpm is not None and labels.shape[0] == self.x_mpm.shape[0]:
            base_com = self.x_mpm.mean(dim=0).detach().clone()
        for lab in unique.tolist():
            if lab <= 0:
                continue
            if lab not in self._physical_fragment_birth_physics_step:
                self._physical_fragment_birth_time[lab] = cur_t
                self._physical_fragment_birth_physics_step[lab] = cur_s
            if self.x_mpm is None or labels.shape[0] != self.x_mpm.shape[0]:
                continue
            lab_int = int(lab)
            state = self._physical_fragment_states.get(lab_int, {})
            if "birth_com" not in state:
                frag_mask = labels == lab_int
                if not bool(frag_mask.any()):
                    continue
                frag_com = self.x_mpm[frag_mask].mean(dim=0).detach().clone()
                state["birth_com"] = frag_com
                state["birth_base_com"] = (
                    base_com.detach().clone() if base_com is not None else frag_com.clone()
                )
                state["birth_time"] = cur_t
                state["birth_physics_step"] = cur_s
                state["birth_size"] = int(frag_mask.sum().item())
                # Physical fragments are already detached by definition in
                # this pipeline.  Marking support_lost lets the existing
                # optional release-drift path work when its velocity/gap
                # knobs are explicitly enabled; with the current zero
                # defaults this remains diagnostics-only.
                state.setdefault("support_lost", True)
                state.setdefault("release_score", 1.0)
                state.setdefault("pseudo_thickness_mass", max(int(frag_mask.sum().item()), 1))
                self._physical_fragment_states[lab_int] = state
        self._sync_rigid_fragment_states(labels)

    def _fracture_tick_interval(self) -> int:
        return max(1, int(self.fracture_cfg.get(
            'fracture_tick_interval_substeps', 1)))

    def _prepare_phase_fracture_tick_budget(self) -> None:
        """Distribute legacy frame-end fracture work across fracture ticks.

        Step 3b keeps the total number of `_step_fracture()` calls per
        visible frame equal to the legacy burst path, but spreads those
        calls over substep fracture ticks.  This preserves impact burst
        work while letting each call read fresh post-MPM stress.
        """
        interval = self._fracture_tick_interval()
        self._phase_tick_budget_frame = int(self.frame_count)
        self._phase_tick_budget_total = max(1, int(
            self._fracture_iterations_this_frame()))
        self._phase_tick_budget_ticks = max(
            1, int(math.ceil(float(self.substeps) / float(interval))))
        self._phase_tick_seen = 0
        self._phase_tick_done = 0

    def _phase_fracture_iterations_this_tick(self) -> int:
        if int(getattr(self, '_phase_tick_budget_frame', -1)) != int(self.frame_count):
            self._prepare_phase_fracture_tick_budget()
        self._phase_tick_seen += 1
        total = max(0, int(self._phase_tick_budget_total))
        ticks = max(1, int(self._phase_tick_budget_ticks))
        seen = min(max(1, int(self._phase_tick_seen)), ticks)
        target_done = int(math.ceil(float(total) * float(seen) / float(ticks)))
        todo = max(0, target_done - int(self._phase_tick_done))
        self._phase_tick_done += todo
        return todo

    def _fracture_projection_frame_key(self) -> int:
        if bool(self.fracture_cfg.get('phase_field_evolve_per_fracture_tick', False)):
            return int(self.frame_count) * 1_000_000 + int(self._physics_step)
        return int(self.frame_count)

    def _run_phase_field_fracture_tick(self) -> bool:
        """Run Step 3b phase-field work after fresh stress is available."""
        iters = self._phase_fracture_iterations_this_tick()
        if iters <= 0:
            return False
        prev_burst_active = bool(getattr(self, '_fracture_burst_active', False))
        self._fracture_burst_active = int(self._phase_tick_budget_total) > 1
        try:
            for _ in range(iters):
                self._step_fracture()
        finally:
            self._fracture_burst_active = prev_burst_active

        c_visual = self._get_volumetric_damage()
        self._last_volumetric_damage_max = (
            float(c_visual.max().item()) if c_visual.numel() else 0.0)
        self._last_volumetric_damage_mean = (
            float(c_visual.mean().item()) if c_visual.numel() else 0.0)
        self._last_c_mech = self._compute_c_mech(c_visual)
        return True

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

    @torch.no_grad()
    def step_rendering(self) -> bool:
        """Full frame: physics → fracture → rendering."""
        self._reset_frame_fragment_event_stats()
        phase_per_tick = bool(self.fracture_cfg.get(
            'phase_field_evolve_per_fracture_tick', False))
        use_physical_authority = bool(self.fracture_cfg.get(
            'use_physical_fragment_authority', False))
        if bool(self.fracture_cfg.get('voronoi_eval_per_fracture_tick', False)):
            self._last_voronoi_components_state = None
        if phase_per_tick:
            self._prepare_phase_fracture_tick_budget()

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

        fragment_detected_in_burst = False
        if not phase_per_tick:
            # --- Fracture update on Gaussian manifold ---
            burst_iters = self._fracture_iterations_this_frame()
            self._fracture_burst_active = burst_iters > 1
            try:
                for burst_idx in range(burst_iters):
                    self._step_fracture()
                    if (
                        burst_iters > 1
                        and burst_idx >= 1
                        and self.fragment_manager is not None
                        and self._gravity_drop_contacted
                    ):
                        self._detect_fragments()
                        fragment_detected_in_burst = True
            finally:
                self._fracture_burst_active = False

        # --- Voronoi pre-fracture update (if enabled) ---
        # Pipeline rewrite Step 3a: when
        # ``manifold.voronoi_eval_per_fracture_tick`` is True the
        # cheap half (bond eval / commit / Mode-I kicks) runs inside
        # the substep loop (see ``_step_physics``); only the
        # expensive spatial-split / label-refinement half runs at
        # frame end.  Otherwise we fall back to the legacy single-
        # call wrapper which executes both halves serially here.
        if (getattr(self, 'voronoi', None) is not None
                and self._gravity_drop_contacted):
            if bool(self.fracture_cfg.get(
                    'voronoi_eval_per_fracture_tick', False)):
                state = getattr(self, '_last_voronoi_components_state', None)
                if state is not None:
                    self._voronoi_spatial_split_and_refine(state)
            else:
                self._step_voronoi_fracture()
        # --- Fragment detection ---
        if (((not fragment_detected_in_burst) or use_physical_authority)
                and self.fragment_manager is not None
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
        # Phase 1: free-fall (no grid physics).  COM follows gravity; if a
        # non-zero drop_omega was configured, every particle additionally
        # gets a rigid-body rotational velocity v = omega x (x - com), so
        # the body tumbles in flight and arrives at the ground with
        # spatially-varying impact velocities (no torque, omega is
        # conserved in free-fall).
        if self._gravity_drop and not self._gravity_drop_contacted:
            g = self.mpm.gravity
            self._v_com = self._v_com + dt * g
            omega = getattr(self, '_omega_com', None)
            if omega is not None and float(omega.norm().item()) > 1e-8:
                com = self.x_mpm.mean(dim=0)
                rel = self.x_mpm - com.unsqueeze(0)
                v_rot = torch.cross(
                    omega.unsqueeze(0).expand_as(rel), rel, dim=1)
                v_total = self._v_com.unsqueeze(0) + v_rot
                self.x_mpm = self.x_mpm + dt * v_total
            else:
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
            if self.material_family == "diffuse_damage":
                # Soft elastomers should keep post-impact kinetic energy so
                # the MPM elasticity can visibly squash and recover.
                self.mpm.damping = 0.997 if frames_since < 5 else 0.9995
            elif frames_since < 5:
                self.mpm.damping = 0.93 + 0.012 * frames_since
            else:
                self.mpm.damping = 0.999

        self._apply_seismic_loading(dt)
        fracture_interval = self._fracture_tick_interval()
        is_fracture_tick = (
            self._gravity_drop_contacted
            and (int(self._physics_step) % fracture_interval) == 0
        )
        phase_per_tick = bool(self.fracture_cfg.get(
            'phase_field_evolve_per_fracture_tick', False))
        voronoi_per_tick = bool(self.fracture_cfg.get(
            'voronoi_eval_per_fracture_tick', False))

        if hasattr(self, "_apply_pending_voronoi_cell_releases"):
            self._apply_pending_voronoi_cell_releases()

        # Compute stress with damage degradation.
        #
        # Step 1a (pipeline rewrite, 2026-05-09): introduce c_visual / c_mech
        # split.  c_visual is the raw phase-field damage (used for crack
        # rendering and as Voronoi bond evidence; range [0, 1] preserved).
        # c_mech is the damage value actually fed into the constitutive
        # model.  When manifold.mechanical_stiffness_floor > 0, we cap
        # c_mech so the residual stiffness never collapses to the
        # constitutive k_residual floor at fully-damaged points (which
        # produces a discontinuous stress field at damage boundaries).
        # The cap value c_cap = 1 - sqrt(g_min) so the post-cap stiffness
        # g(c_cap) = (1 - c_cap)^2 + k_residual ~= g_min.
        # Default g_min = 0 -> c_mech == c_visual -> identical to legacy
        # behavior.
        c_visual = self._get_volumetric_damage()
        self._last_volumetric_damage_max = float(c_visual.max().item()) if c_visual.numel() else 0.0
        self._last_volumetric_damage_mean = float(c_visual.mean().item()) if c_visual.numel() else 0.0

        # Pipeline rewrite Step 3a: per-fracture-tick voronoi bond
        # eval / commit + Mode-I kicks.  Runs every
        # ``fracture_tick_interval_substeps`` physics substeps when
        # the flag is enabled.  The expensive spatial-split + label
        # refinement still happens at frame end (see step_rendering).
        if (getattr(self, 'voronoi', None) is not None
                and voronoi_per_tick
                and not phase_per_tick
                and is_fracture_tick):
            state = self._voronoi_bond_eval_commit()
            if state is not None:
                self._last_voronoi_components_state = state
            self._apply_pending_voronoi_cell_releases()

        c_mech = self._compute_c_mech(c_visual)

        # Cache c_mech for future fracture-tick consumers (Step 3a/3b/1b).
        # In Step 1a this is recomputed every substep; later steps will
        # use the cache when fracture tick interval > 1 substep.
        self._last_c_mech = c_mech

        stress = self.elasticity(self.F, c=c_mech)
        E = (torch.exp(self.elasticity.log_E).item()
             if hasattr(self.elasticity, 'log_E') else 1e6)
        stress = stress.clamp(-5.0 * E, 5.0 * E)
        self._last_stress = stress.detach()

        # MPM P2G2P.  Gate selects between the global p2g2p and the
        # fragmented variant.  Pipeline rewrite Step 2: when
        # manifold.use_physical_fragment_authority is enabled, the
        # source of truth is _physical_fragment_labels (voronoi-driven).
        # Otherwise the legacy graph-fragment counter is consulted.
        if self._fragmented_physics_gate():
            self._step_fragmented_physics(stress, dt)
        else:
            self.x_mpm, self.v_mpm, self.C, self.F = self.mpm.p2g2p(
                self.x_mpm, self.v_mpm, self.C, self.F, stress)

        if phase_per_tick and is_fracture_tick:
            self._run_phase_field_fracture_tick()
            if (getattr(self, 'voronoi', None) is not None
                    and voronoi_per_tick):
                state = self._voronoi_bond_eval_commit()
                if state is not None:
                    self._last_voronoi_components_state = state
                self._apply_pending_voronoi_cell_releases()

        # Floor position clamp: the slip BC only zeros grid v_z, it
        # does not stop particles from drifting below ``ground_z``
        # within a single MPM step at high impact velocity.  Without
        # this clamp the body bisects across the floor (top half
        # above, bottom half below) and reads as a 50/50 horizontal
        # split.  We push every particle back to ``ground_z`` and zero
        # its downward velocity component to be consistent with the
        # slip BC.
        if (self._gravity_drop_contacted
                and self.x_mpm is not None
                and self.v_mpm is not None):
            ground_z = float(getattr(self, '_gravity_drop_ground_z', 0.0))
            if ground_z > 0.0:
                below = self.x_mpm[:, 2] < ground_z
                if bool(below.any()):
                    self.x_mpm[below, 2] = ground_z
                    # Kill any lingering downward velocity (slip).
                    v_z = self.v_mpm[below, 2]
                    self.v_mpm[below, 2] = torch.where(
                        v_z < 0.0, torch.zeros_like(v_z), v_z)

        self._apply_soft_elastic_contact_response(dt)
        self._apply_shape_matching(dt)
        self._apply_soft_elastic_squash(dt)

        # Re-apply floor clamp after shape matching: the position lerp
        # toward (cur_COM + rest_offset) can push particles below the
        # ground when cur_COM has descended near the floor — exactly
        # the case for non-fragment-capable bodies (rubber, wood)
        # whose COM sits at floor level after impact.  Without this
        # second clamp, the body bisects visually across the floor
        # plane (top half above, bottom half below as a mirror).
        if (self._gravity_drop_contacted
                and self.x_mpm is not None
                and self.v_mpm is not None):
            ground_z = float(getattr(self, '_gravity_drop_ground_z', 0.0))
            if ground_z > 0.0:
                below = self.x_mpm[:, 2] < ground_z
                if bool(below.any()):
                    self.x_mpm[below, 2] = ground_z
                    v_z = self.v_mpm[below, 2]
                    self.v_mpm[below, 2] = torch.where(
                        v_z < 0.0, torch.zeros_like(v_z), v_z)

        # Post-impact kinematic lock: for the first K post-impact
        # *frames*, blend every particle's velocity toward the body
        # COM velocity and pull positions toward the rest-pose offset
        # from COM.  Both blend strengths decay linearly from 1.0 at
        # frame 0 to 0.0 at frame K, so the lock fades out smoothly
        # rather than freezing the body for a hard window.  This
        # suppresses the 2-3 frame transient MPM grid-induced cluster
        # split AND lets the body retain natural compress/recover
        # bounce dynamics that an outright v_mpm[:] = v_com lock would
        # block.  Used by non-fragment-capable families (rubber, wood,
        # metal); brittle families have K=0 by default and are not
        # affected.
        frames_since_impact = getattr(self, '_impact_frame_count', 0)
        K = int(self.post_impact_kinematic_lock_frames)
        if (self._gravity_drop_contacted
                and K > 0
                and frames_since_impact < K
                and self.v_mpm is not None
                and self.x_mpm is not None):
            # Blend strength: hold w=1.0 for first M frames, then
            # linearly decay to 0.0 by frame K.  The hold suppresses
            # the impact-transient deformation band that forms in
            # the first ~5 frames after contact when bottom particles
            # decelerate while top particles still fall.
            hold_frames = int(self.fracture_cfg.get(
                'post_impact_kinematic_lock_hold_frames', 0))
            if frames_since_impact < hold_frames:
                w = 1.0
            else:
                decay_progress = (
                    (frames_since_impact - hold_frames)
                    / max(float(K - hold_frames), 1.0))
                w = max(0.0, 1.0 - decay_progress)
            v_com = self._v_com if self._v_com is not None else torch.zeros(
                3, device=self.v_mpm.device, dtype=self.v_mpm.dtype)
            omega = getattr(self, '_omega_com', None)
            if omega is not None and float(omega.norm().item()) > 1e-8:
                com = self.x_mpm.mean(dim=0)
                rel = self.x_mpm - com.unsqueeze(0)
                v_rot = torch.cross(
                    omega.unsqueeze(0).expand_as(rel), rel, dim=1)
                v_target = v_com.unsqueeze(0) + v_rot
            else:
                v_target = v_com.unsqueeze(0).expand_as(self.v_mpm)
            self.v_mpm[:] = (1.0 - w) * self.v_mpm + w * v_target
            # Position blend toward (current_COM + rest_offset),
            # gated by ``post_impact_position_lerp_strength`` so that
            # rubber-like materials (where visible compression is the
            # whole point) can disable the position lock entirely
            # while still benefiting from the velocity lock that
            # suppresses MPM grid-split.  Default 1.0 = full position
            # lerp (rigid wood/metal); set to 0.0 in runtime for
            # diffuse_damage so the body can squash freely.
            pos_lerp = float(self.fracture_cfg.get(
                'post_impact_position_lerp_strength', 1.0))
            if pos_lerp > 0.0:
                rest = getattr(self, '_shape_match_rest_positions', None)
                if rest is not None and rest.shape == self.x_mpm.shape:
                    rest_com = rest.mean(dim=0)
                    rest_rel = rest - rest_com.unsqueeze(0)
                    cur_com = self.x_mpm.mean(dim=0)
                    # When affine shape match is active, the lock
                    # target must use the SAME affine transform A as
                    # _shape_match_component (otherwise lock pulls
                    # toward rigid rest pose and overrides affine).
                    affine_blend = float(getattr(
                        self, 'shape_match_affine_blend', 0.0))
                    if affine_blend > 0.0:
                        cur_centered = self.x_mpm - cur_com.unsqueeze(0)
                        try:
                            P_mat = cur_centered.transpose(0, 1) @ rest_rel
                            Q_mat = rest_rel.transpose(0, 1) @ rest_rel
                            eye3 = torch.eye(
                                3, device=Q_mat.device, dtype=Q_mat.dtype)
                            Q_reg = Q_mat + 1e-6 * eye3 * float(
                                Q_mat.diag().abs().max())
                            A_col = P_mat @ torch.linalg.inv(Q_reg)
                            u_a, s_a, vh_a = torch.linalg.svd(A_col)
                            sv_min = float(getattr(
                                self, 'shape_match_sv_min', 0.78))
                            sv_max = float(getattr(
                                self, 'shape_match_sv_max', 1.00))
                            s_a = torch.clamp(s_a, min=sv_min, max=sv_max)
                            A_col = u_a @ torch.diag(s_a) @ vh_a
                            cov = rest_rel.transpose(0, 1) @ cur_centered
                            u_r, _, vh_r = torch.linalg.svd(cov)
                            R_rigid = u_r @ vh_r
                            if torch.det(R_rigid) < 0.0:
                                u_r = u_r.clone()
                                u_r[:, -1] *= -1.0
                                R_rigid = u_r @ vh_r
                            A_row = A_col.transpose(0, 1)
                            T = ((1.0 - affine_blend) * R_rigid
                                 + affine_blend * A_row)
                            target = rest_rel @ T + cur_com.unsqueeze(0)
                        except RuntimeError:
                            target = cur_com.unsqueeze(0) + rest_rel
                    else:
                        target = cur_com.unsqueeze(0) + rest_rel
                    w_pos = w * pos_lerp
                    self.x_mpm[:] = (1.0 - w_pos) * self.x_mpm + w_pos * target

        # Velocity & F clamping.  Standard MPM safety cap = 0.4 * dx/dt
        # (CFL-safe).  Override with `velocity_cap_scale` to relax it
        # when fragments need to fall under strong gravity faster than
        # the default cap (e.g. brittle pulverization).
        velocity_cap_scale = float(getattr(
            self, "velocity_cap_scale", 1.0))
        v_limit = velocity_cap_scale * 0.4 * self.mpm.dx / dt
        if self._gravity_drop and self._gravity_drop_contacted:
            # Allow up to 8x the post-impact cap (was 80 m/s hard cap).
            v_limit = min(v_limit, 8.0 * 80.0 * velocity_cap_scale)
        self.v_mpm = self.v_mpm.clamp(-v_limit, v_limit)
        self.F = self.F.clamp(-1.5 if self._gravity_drop_contacted else -2.0,
                               1.5 if self._gravity_drop_contacted else 2.0)

        # CFL
        s_max = stress.abs().max().item()
        density_eff = self.mpm.p_mass / self.mpm.vol + 1e-12
        c_wave = (s_max / density_eff) ** 0.5
        self._last_cfl = c_wave * dt / self.mpm.dx if c_wave > 0 else 0

        # Speed limit (per-particle magnitude cap).  Default 10 m/s
        # is conservative for soft elastic body; brittle pulverization
        # under strong gravity needs higher to avoid stalled fragments.
        speed_cap = float(getattr(self, "particle_speed_cap", 10.0))
        v_mag = self.v_mpm.norm(dim=1)
        too_fast = v_mag > speed_cap
        if too_fast.any():
            scale = speed_cap / v_mag[too_fast].clamp(min=1e-8)
            self.v_mpm[too_fast] *= scale.unsqueeze(-1)

        if (hasattr(self, "_rigid_handoff_enabled")
                and self._rigid_handoff_enabled()):
            self._write_rigid_fragment_particles_from_state()

        step = self._physics_step
        if step < 50 or step % 10 == 0:
            print(f"  [phys {step:3d}] |v|={self.v_mpm.abs().max():.4f} "
                  f"|stress|={s_max:.2e} CFL~{self._last_cfl:.3f}")

        self._physics_step += 1

    def _get_voronoi_topology_damage(self) -> Tensor:
        """Project phase-field topology evidence to MPM particles.

        This is the Voronoi breakage signal (`c_topology` in the
        pipeline spec).  It intentionally excludes mechanical feedback
        timing, graph-fragment structural floors, and
        `_physical_fragment_labels` feedback.  Otherwise the topology
        solver can use its own previous labels as new "damage" evidence
        and create fragments before the AT2/front field has actually
        reached that region.
        """
        N = self.x_mpm.shape[0]
        c_topology = torch.zeros(N, device=self.x_mpm.device)
        field = getattr(self, "fracture_field", None)
        if field is None or field.c is None:
            self._last_topology_damage_max = 0.0
            self._last_topology_damage_mean = 0.0
            return c_topology

        if self._surface_indices is None:
            self._surface_indices = torch.where(self.surface_mask)[0]
        n_assign = min(int(field.c.shape[0]), int(self._surface_indices.shape[0]))
        if n_assign <= 0:
            self._last_topology_damage_max = 0.0
            self._last_topology_damage_mean = 0.0
            return c_topology

        surf_topology = field.c[:n_assign].clamp(0.0, 1.0).clone()

        crack_front = getattr(field, "crack_front", None)
        if crack_front is not None:
            visited = getattr(crack_front, "visited_mask", None)
            tips = getattr(crack_front, "tip_mask", None)
            if visited is not None:
                visited = visited[:n_assign]
                surf_topology = torch.maximum(
                    surf_topology,
                    visited.float() * self.crack_volume_visited_floor,
                )
            if tips is not None:
                tips = tips[:n_assign]
                surf_topology = torch.maximum(
                    surf_topology,
                    tips.float() * self.crack_volume_tip_floor,
                )

        if field.a is not None:
            opening = field.a[:n_assign].clamp(min=0.0)
            if bool(opening.max() > 0.0):
                opening_scale = torch.quantile(
                    opening.detach(), 0.85).clamp(min=1e-6)
                opening_norm = (opening / opening_scale).clamp(0.0, 1.0)
                surf_topology = torch.maximum(
                    surf_topology,
                    (
                        field.c[:n_assign].clamp(0.0, 1.0)
                        + self.crack_volume_opening_gain * opening_norm
                    ).clamp(0.0, 1.0),
                )

        interior_scale = float(self.fracture_cfg.get(
            "voronoi_topology_interior_scale",
            max(float(self.crack_volume_interior_scale), 0.85),
        ))
        c_topology = self._project_surface_scalar_to_particles(
            surf_topology.clamp(0.0, 1.0),
            interior_scale=interior_scale,
        ).clamp(0.0, 1.0)
        self._last_topology_damage_max = (
            float(c_topology.max().item()) if c_topology.numel() else 0.0)
        self._last_topology_damage_mean = (
            float(c_topology.mean().item()) if c_topology.numel() else 0.0)
        return c_topology

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
        immediate_feedback = (
            max(0.0, min(float(self.crack_volume_immediate_feedback), 1.0))
            if self._gravity_drop_contacted else 0.0
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
            crack_feedback_scale = max(feedback_scale, 0.45 * immediate_feedback)
            crack_feedback = (
                crack_feedback
                * self.crack_volume_feedback_gain
                * crack_feedback_scale
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
            use_physical_authority = bool(self.fracture_cfg.get(
                'use_physical_fragment_authority', False))

            if not use_physical_authority:
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

            if use_physical_authority:
                if (
                    self._physical_fragment_labels is not None
                    and self._surface_indices is not None
                    and self._physical_fragment_labels.shape[0] == self.x_mpm.shape[0]
                ):
                    phys_frag_ids = self._physical_fragment_labels[
                        self._surface_indices[:n_assign]]
                    detached_mask = phys_frag_ids > 0
                    structural_mask |= detached_mask
                    structural_floor = torch.maximum(
                        structural_floor,
                        detached_mask.float() * (0.65 * self.volumetric_detached_damage_floor),
                    )
            else:
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
                structural_feedback_scale = max(feedback_scale, immediate_feedback)
                c_vol = torch.maximum(
                    c_vol,
                    self._project_surface_scalar_to_particles(
                        structural_floor * structural_feedback_scale,
                        interior_scale=1.0,
                    ),
                )

        return c_vol

    def _get_surface_volume_damage_proxy(self, n_surf: int) -> Optional[Tensor]:
        """Current narrow-band damage proxy used to approve surface fragment birth."""
        if self.fracture_field.c is None or n_surf <= 0:
            self._last_surface_volume_damage_proxy_max = 0.0
            self._last_surface_volume_damage_proxy_mean = 0.0
            return None

        n = min(int(n_surf), int(self.fracture_field.c.shape[0]))
        if n <= 0:
            self._last_surface_volume_damage_proxy_max = 0.0
            self._last_surface_volume_damage_proxy_mean = 0.0
            return None

        proxy = self.fracture_field.c[:n].clamp(0.0, 1.0).clone()
        if self._last_phase_cut_gate is not None:
            proxy = torch.maximum(
                proxy,
                0.62 * self._last_phase_cut_gate[:n].clamp(0.0, 1.0),
            )

        crack_front = getattr(self.fracture_field, "crack_front", None)
        if crack_front is not None:
            visited = getattr(crack_front, "visited_mask", None)
            if visited is not None:
                proxy = torch.maximum(
                    proxy,
                    visited[:n].float() * self.crack_volume_visited_floor,
                )
            tips = getattr(crack_front, "tip_mask", None)
            if tips is not None:
                proxy = torch.maximum(
                    proxy,
                    tips[:n].float() * self.crack_volume_tip_floor,
                )

        if self.fracture_field.a is not None:
            opening = self.fracture_field.a[:n].clamp(min=0.0)
            if bool(opening.max() > 0.0):
                opening_scale = torch.quantile(opening.detach(), 0.85).clamp(min=1e-6)
                opening_norm = (opening / opening_scale).clamp(0.0, 1.0)
                proxy = torch.maximum(
                    proxy,
                    (
                        self.fracture_field.c[:n].clamp(0.0, 1.0)
                        + self.crack_volume_opening_gain * opening_norm
                    ).clamp(0.0, 1.0),
                )

        if self.fragment_manager is not None:
            for attr, scale in (
                ("last_authoritative_cut_mask", self.volumetric_auth_damage_floor),
                ("last_cut_core_mask", 0.78 * self.volumetric_auth_damage_floor),
                ("last_closure_candidate_mask", 0.86 * self.volumetric_detached_damage_floor),
            ):
                mask = getattr(self.fragment_manager, attr, None)
                if mask is not None and mask.shape[0] >= n:
                    proxy = torch.maximum(proxy, mask[:n].float() * float(scale))

        proxy = proxy.clamp(0.0, 1.0)
        self._last_surface_volume_damage_proxy_max = float(proxy.max().item()) if proxy.numel() else 0.0
        self._last_surface_volume_damage_proxy_mean = float(proxy.mean().item()) if proxy.numel() else 0.0
        return proxy

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
        phase_y_mpm = (
            2.0 * float(l0) * psi_mpm / max(float(Gc), 1e-8)
        ).clamp(0.0, 50.0)
        drive_mpm = self._build_effective_fracture_drive(
            psi_mpm, self._last_stress)

        # Project to surface Gaussians
        x_surf_mpm = self.x_mpm[self.surface_mask]
        x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)
        projection_frame = self._fracture_projection_frame_key()

        raw_energy_gauss = self.physics_projector.project_scalar(
            psi_mpm, self.x_mpm, x_surf_mpm, frame=projection_frame)
        phase_y_gauss = self.physics_projector.project_scalar(
            phase_y_mpm, self.x_mpm, x_surf_mpm, frame=projection_frame)
        phase_y_gauss = phase_y_gauss.clamp(0.0, 50.0)
        phase_seed_gate = self._phase_y_gate(
            phase_y_gauss, self.phase_y_seed_threshold)
        phase_advance_gate = self._phase_y_gate(
            phase_y_gauss, self.phase_y_advance_threshold)
        phase_cut_gate = self._phase_y_gate(
            phase_y_gauss, self.phase_y_cut_threshold)
        damage_memory_gate = self._damage_memory_gate(
            self.fracture_field.c if self.fracture_field is not None else None)
        if damage_memory_gate.numel() > 0:
            phase_seed_gate = torch.maximum(phase_seed_gate, damage_memory_gate)
            phase_advance_gate = torch.maximum(phase_advance_gate, damage_memory_gate)
            phase_cut_gate = torch.maximum(
                phase_cut_gate,
                (0.65 * damage_memory_gate + 0.35 * phase_advance_gate).clamp(0.0, 1.0),
            )
        self._last_phase_y_gauss = phase_y_gauss.detach()
        self._last_phase_seed_gate = phase_seed_gate.detach()
        self._last_phase_advance_gate = phase_advance_gate.detach()
        self._last_phase_cut_gate = phase_cut_gate.detach()
        drive_gauss = self.physics_projector.project_scalar(
            drive_mpm, self.x_mpm, x_surf_mpm, frame=projection_frame)
        growth_drive = self._robust_normalize(drive_gauss, 0.90).clamp(0.0, 1.0)

        # Principal stress direction
        stress_dir = None
        sigma1_gauss = torch.zeros_like(growth_drive)
        if self._last_stress is not None:
            stress_dir, sigma1_gauss = self.physics_projector.project_principal_stress_direction(
                self._last_stress, self.x_mpm, x_surf_mpm, frame=projection_frame)
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
            self.F, self.x_mpm, x_surf_mpm, frame=projection_frame)

        # Update fracture field.  Brittle impact burst uses more internal front
        # advances per visible frame without changing the external frame count.
        # Skip the AT2 update entirely once fragmentation has saturated:
        # post-saturation iteration just keeps redistributing the damage
        # field into already-fragmented regions, which propagates noise
        # into the stress tensor that the per-fragment shape match then
        # tries (and fails) to damp out, producing visible base-body
        # particle oscillation.  Saturation = c_max already at 1.0 AND
        # the physical fragment registry is non-empty AND the registry
        # count is stable (frame-to-frame change <= 1).
        skip_at2 = False
        c_max_now = (
            float(self.fracture_field.c.max().item())
            if self.fracture_field.c is not None else 0.0
        )
        # AT2 halt threshold for the registry-stable base damage clamp.
        # At higher particle counts c_max plateaus below 1.0 (graph
        # spreads thinner), so 0.999 would be too strict; 0.40 covers
        # the plateau range observed across 2K-150K runs.
        c_max_threshold = 0.40
        if (bool(self.fracture_cfg.get('use_physical_fragment_authority', False))
                and self.fracture_field.c is not None
                and c_max_now >= c_max_threshold
                and self._physical_fragment_labels is not None
                and bool((self._physical_fragment_labels > 0).any())):
            detached_labels = self._physical_fragment_labels[
                self._physical_fragment_labels > 0]
            cur_count = int(torch.unique(detached_labels).numel())
            prev_count = int(getattr(self, "_at2_halt_prev_count", -1))
            if prev_count >= 0 and abs(cur_count - prev_count) <= 1:
                skip_at2 = True
            self._at2_halt_prev_count = cur_count

        # AT2 halt + stable registry -> clamp damage on the still-cohesive
        # base body to zero.  Without this, the residual damage field on
        # label-0 particles keeps degrading their MPM stiffness, so under
        # grid coupling base particles slowly drift / leak (visible as
        # individual splats popping out of the cohesive remnant).  Once
        # the registry is stable the visible cracks are baked into the
        # fragment labels; any residual damage on label-0 is post-
        # saturation noise that should not weaken the base body.
        if (skip_at2
                and self._surface_indices is not None
                and self.fracture_field.c is not None
                and self.fracture_field.c.shape[0] == self._surface_indices.shape[0]):
            base_surf_mask = (
                self._physical_fragment_labels[self._surface_indices] == 0
            )
            if bool(base_surf_mask.any()):
                self.fracture_field.c[base_surf_mask] = 0.0

        if not skip_at2:
            old_front_substeps = int(getattr(self.fracture_field, "front_substeps", 1))
            if bool(getattr(self, "_fracture_burst_active", False)):
                self.fracture_field.front_substeps = max(
                    old_front_substeps,
                    int(self.impact_fracture_burst_front_substeps),
                )
            try:
                self.fracture_field.update(
                    positions=x_surf_world,
                    init_score=init_score,
                    growth_drive=growth_drive,
                    growth_dir=growth_dir,
                    F_gaussian=F_gauss,
                    impact_center=impact_center_world,
                    phase_gate=phase_advance_gate,
                    seed_phase_gate=phase_seed_gate,
                )
            finally:
                if bool(getattr(self, "_fracture_burst_active", False)):
                    self.fracture_field.front_substeps = old_front_substeps

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
        use_physical_authority = bool(self.fracture_cfg.get(
            'use_physical_fragment_authority', False))
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
            if use_physical_authority:
                self.fracture_field.f = torch.zeros(
                    N, dtype=torch.long, device=self.fracture_field.c.device
                )
            return
        opening_max = (
            float(self.fracture_field.a.max().item())
            if self.fracture_field.a is not None else 0.0
        )
        if self.fracture_field.c.max() < 0.3 and opening_max < 1e-4:
            return

        crack_front = getattr(self.fracture_field, "crack_front", None)
        self.fragment_manager.impact_release_gain = float(getattr(self, '_impact_release_gain', 1.0))
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
        phase_cut_gate = (
            self._last_phase_cut_gate[:N_surf]
            if self._last_phase_cut_gate is not None
            else None
        )
        surface_volume_damage = self._get_surface_volume_damage_proxy(N_surf)
        frames_since_impact = int(getattr(self, "_impact_frame_count", 10**6))
        default_target = 0.64 if self.crack_style == "radial_shatter" else 0.50
        configured_target = float(
            getattr(self.fragment_manager, "configured_impact_closure_target_ratio", -1.0)
        )
        target = configured_target if configured_target >= 0.0 else default_target
        target = min(
            max(target, 0.0),
            float(self.fragment_manager._strict_closure_release_cap()),
        )
        configured_active_frames = int(
            getattr(self.fragment_manager, "configured_impact_closure_active_frames", 2)
        )
        active_window = configured_active_frames
        released_ratio = (
            float(getattr(self.fragment_manager, "last_hard_detached_nodes", 0))
            / max(float(N_surf), 1.0)
        )
        completion_ratio = float(
            getattr(self.fragment_manager, "impact_closure_completion_ratio", 0.92)
        )
        if released_ratio < completion_ratio * max(target, 0.0):
            active_window += int(
                getattr(self.fragment_manager, "impact_closure_adaptive_extra_frames", 0)
            )
        impact_closure_active = bool(
            self.crack_connected_release_only
            and self.material_family == "sharp_brittle"
            and self.crack_style in {
                "radial_shatter",
                "spiderweb",
                "spiderweb_branching",
            }
            and self._gravity_drop_contacted
            and frames_since_impact <= active_window
        )
        self.fragment_manager.impact_closure_active = impact_closure_active
        if impact_closure_active:
            default_max_patches = 96 if self.crack_style == "radial_shatter" else 56
            configured_max_patches = int(
                getattr(self.fragment_manager, "configured_impact_closure_max_patches", -1)
            )
            self.fragment_manager.impact_closure_target_ratio = target
            self.fragment_manager.impact_closure_max_patches = (
                configured_max_patches
                if configured_max_patches >= 0
                else default_max_patches
            )
        else:
            self.fragment_manager.impact_closure_target_ratio = 0.0
            self.fragment_manager.impact_closure_max_patches = 0
        n_frags = self.fragment_manager.detect_fragments(
            self.graph,
            self.fracture_field.c,
            positions=x_surf_world[:N_surf],
            opening=opening,
            active_tip_mask=tip_mask,
            recent_front_mask=recent_front_mask,
            crack_normal=crack_normal,
            crack_tangent=crack_tangent,
            phase_gate=phase_cut_gate,
            volume_damage=surface_volume_damage,
        )
        self._accumulate_frame_fragment_event_stats()

        if use_physical_authority:
            physical_surface_labels = None
            has_physical_surface_fragment = False
            if (self._physical_fragment_labels is not None
                    and self._surface_indices is not None
                    and self.fracture_field.c is not None):
                physical_surface_labels = self._physical_fragment_labels[
                    self._surface_indices]
                n_field = int(self.fracture_field.c.shape[0])
                if physical_surface_labels.shape[0] != n_field:
                    f_phys = torch.zeros(
                        n_field,
                        dtype=torch.long,
                        device=self.fracture_field.c.device,
                    )
                    n_assign = min(n_field, int(physical_surface_labels.shape[0]))
                    if n_assign > 0:
                        f_phys[:n_assign] = physical_surface_labels[:n_assign]
                    physical_surface_labels = f_phys
                else:
                    physical_surface_labels = physical_surface_labels.to(
                        device=self.fracture_field.c.device,
                        dtype=torch.long,
                    )
                has_physical_surface_fragment = bool(
                    (physical_surface_labels > 0).any())

            if physical_surface_labels is not None:
                self.fracture_field.f = physical_surface_labels.clone()
            else:
                self.fracture_field.f = torch.zeros_like(
                    self.fracture_field.c, dtype=torch.long)

            if has_physical_surface_fragment and not self.fragmentation_active:
                self.fragmentation_active = True
                self._fragment_activation_frame = self.frame_count
            return

        if n_frags > 1:
            self.fracture_field.f = self.fragment_manager.fragment_ids.clone()
            apply_impulse = False
            if not self.fragmentation_active:
                self.fragmentation_active = True
                self._fragment_activation_frame = self.frame_count
                apply_impulse = not self.crack_connected_release_only
            elif (
                self._fragment_activation_frame >= 0
                and (self.frame_count - self._fragment_activation_frame) < self.fragment_impulse_boost_frames
            ):
                apply_impulse = not self.crack_connected_release_only

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
            use_physical = bool(self.fracture_cfg.get(
                'use_physical_fragment_authority', False))
            if (use_physical
                    and self._physical_fragment_labels is not None
                    and self._surface_indices is not None
                    and bool((self._physical_fragment_labels > 0).any())):
                frag_ids = self._physical_fragment_labels[self._surface_indices]
            elif (not use_physical) and self.fragment_manager.n_fragments > 1:
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

        use_physical_render = bool(self.fracture_cfg.get(
            'use_physical_fragment_authority', False))
        append_interior_faces = not (
            use_physical_render
            and surf_frag is not None
            and bool((surf_frag > 0).any())
        )
        apply_crack_opening_offsets = append_interior_faces

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
            graph_knn_idx=(
                self.graph.knn_idx
                if (self.graph is not None
                    and getattr(self.graph, "knn_idx", None) is not None
                    and surf_frag is not None
                    and self.graph.knn_idx.shape[0] == surf_frag.shape[0])
                else None
            ),
            append_interior_faces=append_interior_faces,
            apply_crack_opening_offsets=apply_crack_opening_offsets,
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

        # Clamp final Gaussian positions so the visualizer's auto-detected
        # floor matches the simulation's ground_z and particles don't
        # appear to tunnel through the rendered floor mesh.  The
        # visualizer applies crack-opening offsets that can push
        # Gaussians slightly past the MPM domain bounds, which the MPM
        # core's `clip_bound` doesn't catch (it operates only on x_mpm,
        # not on Gaussian render positions).
        if (self._gravity_drop_contacted
                and self.gaussians is not None
                and self.gaussians._xyz.data.shape[0] > 0):
            xyz = self.gaussians._xyz.data
            ground_world = self.mapper.mpm_to_world(
                torch.tensor([0.0, 0.0, float(self._gravity_drop_ground_z)],
                             device=xyz.device, dtype=xyz.dtype))[2].item()
            # Account for the Gaussian splat ellipsoid extent: each
            # splat is rendered as a 3D ellipsoid whose lower extent
            # can dip below the particle center by up to half its
            # max scale value.  Clamp the centre to be at least
            # `ground_world + splat_z_buffer` so the splat doesn't
            # visually tunnel through the floor mesh in the viewer.
            splat_z_buffer = 0.0
            try:
                if self.gaussians._scaling is not None:
                    scaling = torch.exp(self.gaussians._scaling.data)
                    splat_z_buffer = float(0.5 * scaling[:, 2].max().item())
                    splat_z_buffer = min(splat_z_buffer, 0.02)
            except Exception:
                splat_z_buffer = 0.0
            floor_visual = ground_world + splat_z_buffer
            xyz[:, 2] = xyz[:, 2].clamp(min=floor_visual)

    # ================================================================
    # Impact handling
    # ================================================================

    def _handle_ground_impact(self):
        """Handle ground contact in gravity drop mode."""
        self._gravity_drop_contacted = True
        # Per-particle impact velocity: COM translation plus rigid rotation
        # contribution if the body is tumbling.  This preserves angular
        # momentum across the impact event (B1+B2 SIGGRAPH defense).
        omega = getattr(self, '_omega_com', None)
        if omega is not None and float(omega.norm().item()) > 1e-8:
            com = self.x_mpm.mean(dim=0)
            rel = self.x_mpm - com.unsqueeze(0)
            v_rot = torch.cross(
                omega.unsqueeze(0).expand_as(rel), rel, dim=1)
            self.v_mpm[:] = self._v_com.unsqueeze(0) + v_rot
        else:
            self.v_mpm[:] = self._v_com.unsqueeze(0)
        v_impact = self._v_com[2].item()
        impact_speed = abs(float(v_impact))
        self._soft_impact_speed = impact_speed
        self._impact_release_gain = max(
            0.75,
            min(2.65, float(self.fracture_cfg.get('impact_release_gain', 1.0)) * (1.0 + 0.18 * impact_speed)),
        )
        if self.fragment_manager is not None:
            self.fragment_manager.impact_release_gain = self._impact_release_gain
        print(f"  [IMPACT] Ground contact! v_impact={v_impact:.3f}")

        # Unified impact impulse (rigid-body form): convert part of the
        # pre-impact kinetic energy into a horizontal v_com kick plus a
        # tumble omega around the orthogonal horizontal axis.  Per-
        # particle radial impulses sum to ~zero in v_com (outward
        # vectors cancel by symmetry), so shape matching damps them out
        # within a few substeps and the base body looks frozen.  A v_com
        # + omega kick is a NET rigid-body motion -- shape matching can
        # damp internal jitter but the COM still translates and the body
        # still rolls sideways, exactly the "옆으로 뒹군다" the impact
        # would physically produce when the bunny lands off-center.
        # Scale the unified impact rigid kick by the impact-energy
        # factor so soft drops produce minimal whole-body kick + tumble.
        # ke_factor here mirrors the Voronoi pipeline's scaling so the
        # body, the bond breakage, and the Mode-I kicks are all
        # gravity-energy-budgeted together.
        ref_speed_eff = float(self.fracture_cfg.get(
            'voronoi_impact_speed_ref', 70.0))
        min_factor_eff = float(self.fracture_cfg.get(
            'voronoi_impact_min_factor', 0.01))
        ke_factor_eff = max(min_factor_eff, min(
            1.0, impact_speed / max(ref_speed_eff, 1e-3)))
        # ke^5 for unified rigid kick + tumble: at ke=0.68 -> 0.145
        # (~7x reduction vs full), at ke=0.30 -> 0.0024 (~410x).
        # Soft drops produce a tumble-free, kick-free landing -- the
        # body simply rests on the floor without spinning or sliding.
        # Hard impacts (ke≈0.94) keep ~73% of full kick magnitude.
        ke4_eff = ke_factor_eff ** 6.0
        impulse_scale = float(self.fracture_cfg.get(
            'unified_impact_impulse_scale', 0.0)) * ke4_eff
        if impulse_scale > 0.0 and impact_speed > 1e-3:
            # Direction of horizontal kick: from impact_center toward the
            # body COM (in XY).  Off-center landings naturally lean the
            # body in that direction.  For perfectly centered drops we
            # fall back to a deterministic seeded vector so the run is
            # reproducible.
            z_vals_pre = self.x_mpm[:, 2]
            z_min_pre = float(z_vals_pre.min().item())
            z_max_pre = float(z_vals_pre.max().item())
            obj_h = max(z_max_pre - z_min_pre, 1e-6)
            z_thresh_pre = z_min_pre + obj_h * 0.03
            bottom_pre = z_vals_pre < z_thresh_pre
            if bool(bottom_pre.any()):
                center_pre = self.x_mpm[bottom_pre].mean(dim=0)
            else:
                center_pre = self.x_mpm.mean(dim=0)
            body_com = self.x_mpm.mean(dim=0)
            offset_xy = body_com[:2] - center_pre[:2]
            offset_norm = float(offset_xy.norm().item())
            if offset_norm < 1e-4:
                # Symmetric drop -> pick a deterministic seed direction.
                offset_xy = torch.tensor([1.0, 0.0],
                                         device=offset_xy.device,
                                         dtype=offset_xy.dtype)
                offset_norm = 1.0
            horiz_dir = offset_xy / offset_norm
            horiz_mag = impulse_scale * impact_speed

            # NOTE: Earlier we also applied Griffith stress-driven
            # release to ALL particles at the impact moment (including
            # base body) to fix the "frozen base / scattering fragments"
            # asymmetry.  But Griffith uses a random +/- sign per
            # particle along the principal stress direction (correct for
            # cracks: the bond opens to BOTH sides), and applying that
            # to a still-cohesive base body just scattered the splats
            # randomly.  Reverted: Griffith release is only applied at
            # fragment graduation now (cohesive base body keeps the
            # rigid v_com + omega kick alone).

            # v_com horizontal kick: applied uniformly to v_mpm (translation).
            self.v_mpm[:, 0] = self.v_mpm[:, 0] + horiz_mag * horiz_dir[0]
            self.v_mpm[:, 1] = self.v_mpm[:, 1] + horiz_mag * horiz_dir[1]
            self._v_com[0] = self._v_com[0] + horiz_mag * horiz_dir[0]
            self._v_com[1] = self._v_com[1] + horiz_mag * horiz_dir[1]

            # Tumble axis: perpendicular to horiz_dir in XY plane (so the
            # body rolls forward in the slide direction).  axis = z x horiz
            # -> (-h_y, h_x, 0) which curls the top of the body forward.
            tumble_scale = float(self.fracture_cfg.get(
                'unified_impact_tumble_scale', 0.0)) * ke4_eff
            if tumble_scale > 0.0:
                omega_axis = torch.tensor(
                    [-float(horiz_dir[1]), float(horiz_dir[0]), 0.0],
                    device=self.v_mpm.device, dtype=self.v_mpm.dtype,
                )
                # rad/s scaled by impact speed / object size.
                omega_mag = tumble_scale * impact_speed / obj_h
                domega = omega_axis * omega_mag
                rel = self.x_mpm - body_com.unsqueeze(0)
                v_rot_kick = torch.cross(
                    domega.unsqueeze(0).expand_as(rel), rel, dim=1)
                self.v_mpm = self.v_mpm + v_rot_kick
                if hasattr(self, '_omega_com') and self._omega_com is not None:
                    self._omega_com = self._omega_com + domega.to(
                        self._omega_com.device)
                print(
                    f"  [IMPACT] unified rigid kick: horiz=[{horiz_dir[0]:.2f},"
                    f"{horiz_dir[1]:.2f}] mag={horiz_mag:.3f}  "
                    f"omega_axis=[{omega_axis[0]:.2f},{omega_axis[1]:.2f},0] "
                    f"omega_mag={omega_mag:.2f}rad/s"
                )
            else:
                print(
                    f"  [IMPACT] unified horiz kick: dir=[{horiz_dir[0]:.2f},"
                    f"{horiz_dir[1]:.2f}] mag={horiz_mag:.3f}"
                )

        # Soft F reset: blend F toward identity rather than wipe.  alpha=0
        # preserves all pre-impact deformation (paper-defensible: physics is
        # continuous through impact); alpha=1 is the legacy hard reset.
        # Default alpha is 0.0 because in pure free-fall F doesn't accumulate
        # significant deviation from I, so the soft path is numerically safe.
        f_reset_alpha = float(
            self.fracture_cfg.get('impact_F_reset_alpha', 0.0)
        )
        f_reset_alpha = min(max(f_reset_alpha, 0.0), 1.0)
        N = self.F.shape[0]
        if f_reset_alpha > 0.0:
            eye = torch.eye(
                3, device=self.F.device, dtype=self.F.dtype
            ).unsqueeze(0).expand(N, 3, 3)
            self.F = ((1.0 - f_reset_alpha) * self.F
                      + f_reset_alpha * eye).clone()
            self.C = (1.0 - f_reset_alpha) * self.C
        # else: leave F and C as-is — physics is continuous.
        self.frame_count = 0
        self._impact_frame_count = 0
        # Arm the post-impact kinematic lock if configured.  Counter
        # decrements each MPM step in ``_step_physics`` and forces
        # ``v_mpm = v_com`` while > 0, suppressing residual MPM
        # grid-induced transient split for non-fragment-capable
        # families.  Re-read every impact so a runtime override that
        # changed the value mid-session is honoured.
        self._kinematic_lock_remaining = int(
            self.fracture_cfg.get(
                'post_impact_kinematic_lock_frames',
                self.post_impact_kinematic_lock_frames))

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
              f"seed_mag={seed_magnitude:.3f} Hx={seed_H_multiplier:.2f} "
              f"release_gain={self._impact_release_gain:.2f}")

        # Post-impact physics adjustments.  Default values (-220 / -400)
        # are intentionally weakened relative to the free-fall gravity
        # (-2000) for diffuse-damage / soft materials so fragments don't
        # tunnel through the floor.  For brittle materials we want
        # uniformly accelerating fall before AND after impact, so allow
        # a runtime override to keep gravity at its free-fall magnitude.
        g_vec = self.mpm.gravity.clone()
        g_vec[:] = 0.0
        default_gz = -220.0 if self.material_family == "diffuse_damage" else -400.0
        g_vec[2] = float(self.fracture_cfg.get('post_impact_gravity_z', default_gz))
        self.mpm.gravity = g_vec
        default_damping = 0.995 if self.material_family == "diffuse_damage" else 0.975
        self.mpm.damping = float(self.fracture_cfg.get('post_impact_damping', default_damping))
        print(f"  [POST-IMPACT] gravity→[0,0,{g_vec[2].item():.0f}] damping→{self.mpm.damping:.3f}")

        # Voronoi pre-fracture tessellation: at impact, partition the
        # body into N seed-conditioned cells.  AT2 damage propagates
        # through the cell-bond network; cells whose bonds break end up
        # as fragments (PCA-thin-axis kick + inherited body velocity).
        # Replaces the spatial-grid force-promote partition with a
        # geometry-aware decomposition.
        if bool(self.fracture_cfg.get('voronoi_enable', False)):
            self._init_voronoi_tessellation()


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
        self._impact_release_gain = max(
            0.75,
            min(2.65, float(self.fracture_cfg.get('impact_release_gain', 1.0)) * (0.85 + 0.18 * abs(float(impact_energy)))),
        )
        if self.fragment_manager is not None:
            self.fragment_manager.impact_release_gain = self._impact_release_gain
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
        phase_y_max = (
            float(self._last_phase_y_gauss.max().item())
            if self._last_phase_y_gauss is not None else 0.0
        )
        phase_y_mean = (
            float(self._last_phase_y_gauss.mean().item())
            if self._last_phase_y_gauss is not None else 0.0
        )
        phase_seed_gate_max = (
            float(self._last_phase_seed_gate.max().item())
            if self._last_phase_seed_gate is not None else 0.0
        )
        phase_advance_gate_max = (
            float(self._last_phase_advance_gate.max().item())
            if self._last_phase_advance_gate is not None else 0.0
        )
        phase_advance_gate_mean = (
            float(self._last_phase_advance_gate.mean().item())
            if self._last_phase_advance_gate is not None else 0.0
        )
        phase_cut_gate_max = (
            float(self._last_phase_cut_gate.max().item())
            if self._last_phase_cut_gate is not None else 0.0
        )
        detached_distance = 0.0
        mean_detached_distance = 0.0
        render_offset_distance = 0.0
        render_mean_offset_distance = 0.0
        physical_detached_distance = 0.0
        physical_mean_detached_distance = 0.0
        physical_fragment_drop = 0.0
        physical_release_displacement = 0.0
        physical_mean_release_displacement = 0.0
        physical_lateral_release_displacement = 0.0
        physical_mean_lateral_release_displacement = 0.0
        physical_fragment_lateral_spread = 0.0
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
        graph_n_frags = (
            int(self.fragment_manager.n_fragments)
            if self.fragment_manager is not None else 0
        )
        if render_fragment_ids is not None and render_fragment_ids.numel() > 0:
            if bool((render_fragment_ids > 0).any()):
                render_n_frags = int(render_fragment_ids.unique().numel())
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
        # Render offset metrics removed: render position equals MPM
        # position so per-fragment offset distance is always 0.
        physical_n_frags = 0
        frag_ids_phys = None
        use_physical = bool(self.fracture_cfg.get(
            'use_physical_fragment_authority', False))
        if (use_physical
                and self._physical_fragment_labels is not None
                and self._surface_indices is not None
                and bool((self._physical_fragment_labels > 0).any())):
            frag_ids_phys = self._physical_fragment_labels[self._surface_indices]
            physical_n_frags = int(frag_ids_phys.unique().numel())
        elif ((not use_physical)
              and self.fragment_manager is not None
              and self.fragment_manager.fragment_ids is not None):
            frag_ids_phys = self.fragment_manager.fragment_ids
            physical_n_frags = int(frag_ids_phys.unique().numel()) if bool((frag_ids_phys > 0).any()) else 1

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
                    lateral_distances = [
                        float((com[:2] - base[:2]).norm().item())
                        for frag_id, com in coms[1:]
                    ]
                    drops = [
                        max(float(base[2].item() - com[2].item()), 0.0)
                        for frag_id, com in coms[1:]
                    ]
                    physical_detached_distance = max(distances)
                    physical_mean_detached_distance = sum(distances) / max(len(distances), 1)
                    physical_fragment_lateral_spread = (
                        max(lateral_distances) if lateral_distances else 0.0
                    )
                    physical_fragment_drop = max(drops) if drops else 0.0
                    release_distances = []
                    lateral_release_distances = []
                    for frag_id, com in coms[1:]:
                        state = self._physical_fragment_states.get(int(frag_id), {})
                        birth_com = state.get("birth_com")
                        birth_base_com = state.get("birth_base_com")
                        if birth_com is None or birth_base_com is None:
                            continue
                        birth_rel = birth_com.to(com.device) - birth_base_com.to(com.device)
                        current_rel = com - base
                        release_distances.append(
                            float((current_rel - birth_rel).norm().item())
                        )
                        lateral_release_distances.append(
                            float((current_rel[:2] - birth_rel[:2]).norm().item())
                        )
                    if release_distances:
                        physical_release_displacement = max(release_distances)
                        physical_mean_release_displacement = (
                            sum(release_distances) / max(len(release_distances), 1)
                        )
                    if lateral_release_distances:
                        physical_lateral_release_displacement = max(lateral_release_distances)
                        physical_mean_lateral_release_displacement = (
                            sum(lateral_release_distances)
                            / max(len(lateral_release_distances), 1)
                        )
        if use_physical:
            n_frags_out = physical_n_frags
        else:
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
            "impact_release_gain": float(getattr(self, '_impact_release_gain', 1.0)),
            "c_max": c_max,
            "c_mean": c_mean,
            "n_cracked": n_cracked,
            "phase_y_max": phase_y_max,
            "phase_y_mean": phase_y_mean,
            "phase_seed_gate_max": phase_seed_gate_max,
            "phase_advance_gate_max": phase_advance_gate_max,
            "phase_advance_gate_mean": phase_advance_gate_mean,
            "phase_cut_gate_max": phase_cut_gate_max,
            "phase_y_seed_threshold": float(self.phase_y_seed_threshold),
            "phase_y_advance_threshold": float(self.phase_y_advance_threshold),
            "phase_y_cut_threshold": float(self.phase_y_cut_threshold),
            "volumetric_damage_max": float(self._last_volumetric_damage_max),
            "volumetric_damage_mean": float(self._last_volumetric_damage_mean),
            "topology_damage_max": float(self._last_topology_damage_max),
            "topology_damage_mean": float(self._last_topology_damage_mean),
            "surface_volume_damage_proxy_max": float(self._last_surface_volume_damage_proxy_max),
            "surface_volume_damage_proxy_mean": float(self._last_surface_volume_damage_proxy_mean),
            "rigid_angular_speed_max": float(self._last_rigid_angular_speed_max),
            "rigid_angular_speed_mean": float(self._last_rigid_angular_speed_mean),
            "n_fragments": n_frags_out,
            "physical_n_fragments": physical_n_frags,
            "graph_n_fragments": graph_n_frags,
            "render_n_fragments": render_n_frags,
            "broken_edges": (self.fragment_manager.last_broken_edges
                              if self.fragment_manager else 0),
            "raw_components": (self.fragment_manager.last_raw_components
                                if self.fragment_manager else 1),
            "hard_detached_nodes": (self.fragment_manager.last_hard_detached_nodes
                                     if self.fragment_manager else 0),
            "new_detached_nodes": int(self._frame_fragment_stat("new_detached_nodes", 0)),
            "detached_boundary_edges": (self.fragment_manager.last_detached_boundary_edges
                                        if self.fragment_manager else 0),
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
            "phase_cut_edges": (self.fragment_manager.last_phase_cut_edges
                                  if self.fragment_manager else 0),
            "phase_fragment_gate_max": (self.fragment_manager.last_phase_gate_max
                                         if self.fragment_manager else 0.0),
            "phase_fragment_gate_mean": (self.fragment_manager.last_phase_gate_mean
                                          if self.fragment_manager else 0.0),
            "phase_candidate_patches": int(self._frame_fragment_stat("phase_candidate_patches", 0)),
            "phase_approved_patches": int(self._frame_fragment_stat("phase_approved_patches", 0)),
            "phase_rejected_patches": int(self._frame_fragment_stat("phase_rejected_patches", 0)),
            "phase_approved_nodes": int(self._frame_fragment_stat("phase_approved_nodes", 0)),
            "phase_approval_score_max": float(self._frame_fragment_stat("phase_approval_score_max", 0.0)),
            "phase_approval_score_mean": float(
                self._frame_fragment_stat("phase_approval_score_weighted_sum", 0.0)
                / max(float(self._frame_fragment_stat("phase_approved_patches", 0)), 1.0)
            ),
            "phase_approval_cvol_max": float(self._frame_fragment_stat("phase_approval_cvol_max", 0.0)),
            "phase_approval_gate_max": float(self._frame_fragment_stat("phase_approval_gate_max", 0.0)),
            "pseudo_thickness_mass": float(self._frame_fragment_stat("pseudo_thickness_mass", 0.0)),
            "birth_phase_score": float(self._frame_fragment_stat("birth_phase_score", 0.0)),
            "birth_cvol_max": float(self._frame_fragment_stat("birth_cvol_max", 0.0)),
            "birth_phase_gate_max": float(self._frame_fragment_stat("birth_phase_gate_max", 0.0)),
            "birth_phase_approved": bool(self._frame_fragment_stat("birth_phase_approved", False)),
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
            "open_release_patches": int(self._frame_fragment_stat("open_release_patches", 0)),
            "open_release_nodes": int(self._frame_fragment_stat("open_release_nodes", 0)),
            "open_release_score_max": float(self._frame_fragment_stat("open_release_score_max", 0.0)),
            "impact_closure_patches": int(self._frame_fragment_stat("impact_closure_patches", 0)),
            "impact_closure_nodes": int(self._frame_fragment_stat("impact_closure_nodes", 0)),
            "impact_closure_score_max": float(self._frame_fragment_stat("impact_closure_score_max", 0.0)),
            "top_component_sizes": (self.fragment_manager.last_top_component_sizes
                                       if self.fragment_manager else []),
            "detached_distance": detached_distance,
            "mean_detached_distance": mean_detached_distance,
            "render_offset_distance": render_offset_distance,
            "render_mean_offset_distance": render_mean_offset_distance,
            "physical_detached_distance": physical_detached_distance,
            "physical_mean_detached_distance": physical_mean_detached_distance,
            "physical_fragment_drop": physical_fragment_drop,
            "physical_release_displacement": physical_release_displacement,
            "physical_mean_release_displacement": physical_mean_release_displacement,
            "physical_lateral_release_displacement": physical_lateral_release_displacement,
            "physical_mean_lateral_release_displacement": physical_mean_lateral_release_displacement,
            "physical_fragment_lateral_spread": physical_fragment_lateral_spread,
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
