"""
Material-prior adapter for CLIP-driven fracture behavior.

This layer translates semantic CLIP retrieval results into:
  1. physical priors   : E, Gc, nu, density
  2. fracture priors   : crack-growth style parameters
  3. runtime overrides : manifold + visualization controls
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from src.ml.material_db import MaterialEntry


DEFAULT_STYLE = {
    "tau_init": 0.32,
    "growth_gain": 1.00,
    "band_width": 1.50,
    "band_fill_gain": 0.30,
    "open_gain": 1.00,
    "split_threshold": 0.52,
    "edge_break_rate": 1.00,
    "branching_bias": 0.20,
    "anisotropy_strength": 0.10,
}


CATEGORY_STYLE_DB = {
    "glass": {
        "tau_init": 0.18,
        "growth_gain": 1.45,
        "band_width": 1.00,
        "band_fill_gain": 0.10,
        "open_gain": 1.65,
        "split_threshold": 0.30,
        "edge_break_rate": 1.70,
        "branching_bias": 0.08,
        "anisotropy_strength": 0.18,
    },
    "ceramic": {
        "tau_init": 0.22,
        "growth_gain": 1.25,
        "band_width": 1.10,
        "band_fill_gain": 0.18,
        "open_gain": 1.35,
        "split_threshold": 0.34,
        "edge_break_rate": 1.45,
        "branching_bias": 0.18,
        "anisotropy_strength": 0.20,
    },
    "concrete": {
        "tau_init": 0.30,
        "growth_gain": 0.95,
        "band_width": 2.20,
        "band_fill_gain": 0.55,
        "open_gain": 0.90,
        "split_threshold": 0.46,
        "edge_break_rate": 1.05,
        "branching_bias": 0.60,
        "anisotropy_strength": 0.08,
    },
    "stone": {
        "tau_init": 0.28,
        "growth_gain": 1.00,
        "band_width": 1.80,
        "band_fill_gain": 0.35,
        "open_gain": 1.00,
        "split_threshold": 0.42,
        "edge_break_rate": 1.15,
        "branching_bias": 0.34,
        "anisotropy_strength": 0.16,
    },
    "metal": {
        "tau_init": 0.60,
        "growth_gain": 0.35,
        "band_width": 2.50,
        "band_fill_gain": 0.20,
        "open_gain": 0.30,
        "split_threshold": 0.80,
        "edge_break_rate": 0.30,
        "branching_bias": 0.05,
        "anisotropy_strength": 0.06,
    },
    "wood": {
        "tau_init": 0.36,
        "growth_gain": 0.90,
        "band_width": 1.70,
        "band_fill_gain": 0.26,
        "open_gain": 0.80,
        "split_threshold": 0.55,
        "edge_break_rate": 0.85,
        "branching_bias": 0.12,
        "anisotropy_strength": 0.55,
    },
    "ice": {
        "tau_init": 0.16,
        "growth_gain": 1.55,
        "band_width": 1.00,
        "band_fill_gain": 0.12,
        "open_gain": 1.70,
        "split_threshold": 0.28,
        "edge_break_rate": 1.80,
        "branching_bias": 0.18,
        "anisotropy_strength": 0.12,
    },
    "polymer": {
        "tau_init": 0.52,
        "growth_gain": 0.48,
        "band_width": 2.80,
        "band_fill_gain": 0.75,
        "open_gain": 0.28,
        "split_threshold": 0.78,
        "edge_break_rate": 0.28,
        "branching_bias": 0.05,
        "anisotropy_strength": 0.04,
    },
    "biological": {
        "tau_init": 0.30,
        "growth_gain": 0.88,
        "band_width": 1.60,
        "band_fill_gain": 0.30,
        "open_gain": 0.85,
        "split_threshold": 0.48,
        "edge_break_rate": 0.95,
        "branching_bias": 0.16,
        "anisotropy_strength": 0.22,
    },
    "food": {
        "tau_init": 0.22,
        "growth_gain": 1.10,
        "band_width": 1.60,
        "band_fill_gain": 0.28,
        "open_gain": 1.05,
        "split_threshold": 0.36,
        "edge_break_rate": 1.35,
        "branching_bias": 0.25,
        "anisotropy_strength": 0.08,
    },
    "composite": {
        "tau_init": 0.34,
        "growth_gain": 0.92,
        "band_width": 1.40,
        "band_fill_gain": 0.22,
        "open_gain": 0.85,
        "split_threshold": 0.50,
        "edge_break_rate": 0.90,
        "branching_bias": 0.10,
        "anisotropy_strength": 0.65,
    },
    "other": {
        "tau_init": 0.24,
        "growth_gain": 1.20,
        "band_width": 1.30,
        "band_fill_gain": 0.18,
        "open_gain": 1.20,
        "split_threshold": 0.34,
        "edge_break_rate": 1.45,
        "branching_bias": 0.20,
        "anisotropy_strength": 0.18,
    },
}


NAME_STYLE_OVERRIDES = {
    "rubber": {
        "tau_init": 0.70,
        "growth_gain": 0.25,
        "band_width": 3.20,
        "band_fill_gain": 0.95,
        "open_gain": 0.12,
        "split_threshold": 0.92,
        "edge_break_rate": 0.12,
        "branching_bias": 0.00,
        "anisotropy_strength": 0.02,
    },
    "polycarbonate": {
        "tau_init": 0.58,
        "growth_gain": 0.35,
        "band_width": 2.60,
        "band_fill_gain": 0.70,
        "open_gain": 0.20,
        "split_threshold": 0.85,
        "edge_break_rate": 0.18,
        "branching_bias": 0.02,
        "anisotropy_strength": 0.04,
    },
    "polystyrene": {
        "tau_init": 0.26,
        "growth_gain": 1.05,
        "band_width": 1.40,
        "band_fill_gain": 0.28,
        "open_gain": 1.10,
        "split_threshold": 0.40,
        "edge_break_rate": 1.15,
        "branching_bias": 0.22,
        "anisotropy_strength": 0.06,
    },
    "acrylic": {
        "tau_init": 0.24,
        "growth_gain": 1.08,
        "band_width": 1.30,
        "band_fill_gain": 0.20,
        "open_gain": 1.20,
        "split_threshold": 0.38,
        "edge_break_rate": 1.20,
        "branching_bias": 0.16,
        "anisotropy_strength": 0.08,
    },
    "cast iron": {
        "tau_init": 0.30,
        "growth_gain": 1.00,
        "band_width": 1.50,
        "band_fill_gain": 0.20,
        "open_gain": 1.00,
        "split_threshold": 0.42,
        "edge_break_rate": 1.05,
        "branching_bias": 0.12,
        "anisotropy_strength": 0.08,
    },
    "slate": {
        "tau_init": 0.26,
        "growth_gain": 1.05,
        "band_width": 1.30,
        "band_fill_gain": 0.18,
        "open_gain": 1.00,
        "split_threshold": 0.34,
        "edge_break_rate": 1.20,
        "branching_bias": 0.10,
        "anisotropy_strength": 0.75,
    },
    "bamboo": {
        "tau_init": 0.32,
        "growth_gain": 0.92,
        "band_width": 1.50,
        "band_fill_gain": 0.20,
        "open_gain": 0.72,
        "split_threshold": 0.52,
        "edge_break_rate": 0.92,
        "branching_bias": 0.08,
        "anisotropy_strength": 0.85,
    },
    "carbon fiber": {
        "tau_init": 0.38,
        "growth_gain": 0.86,
        "band_width": 1.20,
        "band_fill_gain": 0.16,
        "open_gain": 0.68,
        "split_threshold": 0.58,
        "edge_break_rate": 0.82,
        "branching_bias": 0.04,
        "anisotropy_strength": 0.90,
    },
    "fiberglass": {
        "tau_init": 0.32,
        "growth_gain": 0.88,
        "band_width": 1.50,
        "band_fill_gain": 0.24,
        "open_gain": 0.74,
        "split_threshold": 0.56,
        "edge_break_rate": 0.88,
        "branching_bias": 0.12,
        "anisotropy_strength": 0.45,
    },
}


STYLE_KEYS = tuple(DEFAULT_STYLE.keys())
FAMILY_NAMES = (
    "sharp_brittle",
    "brittle_moderate",
    "rough_quasi_brittle",
    "diffuse_damage",
    "neutral_reference",
)

NAME_FAMILY_OVERRIDES = {
    "rubber": "diffuse_damage",
    "latex": "diffuse_damage",
    "neoprene": "diffuse_damage",
    "elastomer": "diffuse_damage",
    "silicone": "diffuse_damage",
    "foam": "diffuse_damage",
    "gel": "diffuse_damage",
    "glass": "sharp_brittle",
    "ice": "sharp_brittle",
    "crystal": "sharp_brittle",
    "porcelain": "brittle_moderate",
    "ceramic": "brittle_moderate",
    "china": "brittle_moderate",
    "eggshell": "brittle_moderate",
    "acrylic": "brittle_moderate",
    "pmma": "brittle_moderate",
    "polystyrene": "brittle_moderate",
    "concrete": "rough_quasi_brittle",
    "cement": "rough_quasi_brittle",
    "plaster": "rough_quasi_brittle",
    "stone": "rough_quasi_brittle",
    "brick": "rough_quasi_brittle",
    "chalk": "rough_quasi_brittle",
}

CATEGORY_TO_FAMILY = {
    "glass": "sharp_brittle",
    "ice": "sharp_brittle",
    "ceramic": "brittle_moderate",
    "concrete": "rough_quasi_brittle",
    "stone": "rough_quasi_brittle",
    "polymer": "neutral_reference",
    "metal": "neutral_reference",
    "wood": "neutral_reference",
    "biological": "neutral_reference",
    "food": "brittle_moderate",
    "composite": "neutral_reference",
    "other": "neutral_reference",
}

FAMILY_STYLE_MULTIPLIERS = {
    "sharp_brittle": {
        "tau_init": 0.82,
        "growth_gain": 1.08,
        "band_width": 0.76,
        "band_fill_gain": 0.52,
        "open_gain": 1.42,
        "split_threshold": 0.74,
        "edge_break_rate": 1.28,
        "branching_bias": 0.55,
        "anisotropy_strength": 1.20,
    },
    "brittle_moderate": {
        "tau_init": 0.92,
        "growth_gain": 1.04,
        "band_width": 0.88,
        "band_fill_gain": 0.78,
        "open_gain": 1.14,
        "split_threshold": 0.86,
        "edge_break_rate": 1.08,
        "branching_bias": 0.82,
        "anisotropy_strength": 1.10,
    },
    "rough_quasi_brittle": {
        "tau_init": 1.02,
        "growth_gain": 0.95,
        "band_width": 1.26,
        "band_fill_gain": 1.38,
        "open_gain": 0.84,
        "split_threshold": 1.06,
        "edge_break_rate": 0.96,
        "branching_bias": 1.34,
        "anisotropy_strength": 0.92,
    },
    "diffuse_damage": {
        "tau_init": 1.34,
        "growth_gain": 0.42,
        "band_width": 1.62,
        "band_fill_gain": 1.48,
        "open_gain": 0.30,
        "split_threshold": 1.22,
        "edge_break_rate": 0.34,
        "branching_bias": 0.18,
        "anisotropy_strength": 0.55,
    },
    "neutral_reference": {
        "tau_init": 1.04,
        "growth_gain": 0.94,
        "band_width": 0.94,
        "band_fill_gain": 0.90,
        "open_gain": 0.90,
        "split_threshold": 1.00,
        "edge_break_rate": 0.92,
        "branching_bias": 0.85,
        "anisotropy_strength": 0.95,
    },
}

FAMILY_RUNTIME_PRESETS = {
    "sharp_brittle": {
        "manifold.material_family": "sharp_brittle",
        "manifold.enable_front_propagation": True,
        "manifold.material_drive_floor": 0.12,
        "manifold.diffuse_damage_gain": 0.10,
        "manifold.diffuse_neighborhood_steps": 1,
        "manifold.damage_source_scale": 0.62,
        "manifold.front_threshold": 0.03,
        "manifold.tip_propagation_scale": 0.98,
        "manifold.front_substeps": 3,
        "manifold.min_successor_score": 0.18,
        "manifold.impact_seed_H_multiplier": 0.10,
        "manifold.fragment_detect_every": 2,
        "manifold.fragment_damage_threshold": 0.34,
        "manifold.min_fragment_particles": 24,
        "manifold.fragment_opening_weight": 0.52,
        "manifold.fragment_active_tip_weight": 0.24,
        "manifold.fragment_recent_front_weight": 0.30,
        "manifold.fragment_pair_break_weight": 0.28,
        "manifold.fragment_edge_memory_decay": 0.988,
        "manifold.fragment_edge_memory_weight": 0.90,
        "manifold.fragment_cut_diffusion_alpha": 0.18,
        "manifold.fragment_cut_diffusion_iters": 1,
        "manifold.fragment_cut_cos_gate_tangent": 0.78,
        "manifold.fragment_cut_cos_gate_normal": 0.65,
        "manifold.fragment_primary_cut_ratio": 0.52,
        "manifold.fragment_fallback_cut_ratio": 0.34,
        "manifold.fragment_min_boundary_edges": 12,
        "manifold.fragment_detached_node_decay": 0.97,
        "manifold.fragment_persistent_min_size": 5,
        "manifold.fragment_component_hysteresis": 0.30,
        "manifold.fragment_post_split_threshold_scale": 0.82,
        "manifold.fragment_impulse_strength": 3.45,
        "manifold.fragment_upward_bias": 0.28,
        "manifold.fragment_visual_offset_scale": 0.017,
        "manifold.fragment_visual_ramp_frames": 5,
        "manifold.fragment_impulse_boost_frames": 6,
        "manifold.fragment_event_boost": 1.90,
        "manifold.shard_enable": True,
        "manifold.shard_count_scale": 1.00,
        "manifold.fragment_offset_gain": 1.65,
        "manifold.debris_motion_gain": 1.05,
        "manifold.cut_surface_enable": True,
        "manifold.cut_vote_strength": 1.20,
        "manifold.tau_cross": 0.50,
        "manifold.tau_tangent": 0.36,
        "manifold.cut_core_damage_threshold": 0.12,
        "manifold.cut_core_opening_threshold": 0.08,
        "manifold.cut_hard_break_threshold": 0.23,
        "manifold.authoritative_cut_decay": 0.975,
        "manifold.authoritative_cut_threshold": 0.14,
        "manifold.support_loss_enable": True,
        "manifold.support_anchor_quantile": 0.08,
        "manifold.support_release_threshold": 0.44,
        "manifold.support_promote_min_size": 3,
        "manifold.support_overlap_threshold": 0.06,
        "manifold.volumetric_cut_damage_scale": 0.72,
        "manifold.volumetric_auth_damage_floor": 0.82,
        "manifold.volumetric_detached_damage_floor": 0.96,
        "manifold.crack_volume_feedback_gain": 0.58,
        "manifold.crack_volume_opening_gain": 0.52,
        "manifold.crack_volume_visited_floor": 0.62,
        "manifold.crack_volume_tip_floor": 0.78,
        "manifold.crack_volume_interior_scale": 0.92,
        "manifold.fragment_physical_gap_scale": 0.00010,
        "manifold.fragment_physical_release_velocity": 0.010,
        "manifold.fragment_physical_downward_bias": 0.22,
        "manifold.fragment_physical_release_frames": 24,
        "manifold.splitting_enabled": True,
        "gaussian_splatting.material_family": "sharp_brittle",
        "gaussian_splatting.crack_band_weight": 0.55,
        "gaussian_splatting.crack_visited_weight": 0.10,
        "gaussian_splatting.crack_tip_weight": 1.35,
        "gaussian_splatting.crack_core_weight": 1.30,
        "gaussian_splatting.split_gap_gain": 1.48,
        "gaussian_splatting.fragment_shell_gain": 1.34,
        "gaussian_splatting.fragment_contrast_gain": 1.40,
        "gaussian_splatting.debris_darkening": 0.20,
        "gaussian_splatting.shard_scale_gain": 1.22,
        "gaussian_splatting.shard_opacity_gain": 1.15,
        "gaussian_splatting.damage_scale_shrink": 0.45,
        "gaussian_splatting.damage_center_opacity_reduction": 0.86,
        "gaussian_splatting.diffuse_damage_strength": 0.08,
        "gaussian_splatting.crack_tip_scale_boost": 0.46,
        "gaussian_splatting.crack_tip_opacity_boost": 0.24,
    },
    "brittle_moderate": {
        "manifold.material_family": "brittle_moderate",
        "manifold.enable_front_propagation": True,
        "manifold.material_drive_floor": 0.072,
        "manifold.diffuse_damage_gain": 0.14,
        "manifold.diffuse_neighborhood_steps": 1,
        "manifold.damage_source_scale": 0.43,
        "manifold.front_threshold": 0.04,
        "manifold.tip_propagation_scale": 0.86,
        "manifold.front_substeps": 3,
        "manifold.min_successor_score": 0.215,
        "manifold.impact_seed_H_multiplier": 0.05,
        "manifold.fragment_detect_every": 2,
        "manifold.fragment_damage_threshold": 0.40,
        "manifold.min_fragment_particles": 30,
        "manifold.fragment_opening_weight": 0.42,
        "manifold.fragment_active_tip_weight": 0.20,
        "manifold.fragment_recent_front_weight": 0.11,
        "manifold.fragment_pair_break_weight": 0.10,
        "manifold.fragment_edge_memory_decay": 0.980,
        "manifold.fragment_edge_memory_weight": 0.80,
        "manifold.fragment_cut_diffusion_alpha": 0.30,
        "manifold.fragment_cut_diffusion_iters": 1,
        "manifold.fragment_cut_cos_gate_tangent": 0.70,
        "manifold.fragment_cut_cos_gate_normal": 0.60,
        "manifold.fragment_primary_cut_ratio": 0.55,
        "manifold.fragment_fallback_cut_ratio": 0.38,
        "manifold.fragment_min_boundary_edges": 15,
        "manifold.fragment_detached_node_decay": 0.96,
        "manifold.fragment_persistent_min_size": 12,
        "manifold.fragment_component_hysteresis": 0.44,
        "manifold.fragment_post_split_threshold_scale": 0.92,
        "manifold.fragment_impulse_strength": 2.70,
        "manifold.fragment_upward_bias": 0.32,
        "manifold.fragment_visual_offset_scale": 0.012,
        "manifold.fragment_visual_ramp_frames": 6,
        "manifold.fragment_impulse_boost_frames": 4,
        "manifold.fragment_event_boost": 1.18,
        "manifold.shard_enable": False,
        "manifold.shard_count_scale": 0.0,
        "manifold.fragment_offset_gain": 1.05,
        "manifold.debris_motion_gain": 0.0,
        "manifold.cut_surface_enable": True,
        "manifold.cut_vote_strength": 0.68,
        "manifold.tau_cross": 0.60,
        "manifold.tau_tangent": 0.36,
        "manifold.cut_core_damage_threshold": 0.18,
        "manifold.cut_core_opening_threshold": 0.13,
        "manifold.cut_hard_break_threshold": 0.44,
        "manifold.authoritative_cut_decay": 0.968,
        "manifold.authoritative_cut_threshold": 0.20,
        "manifold.support_loss_enable": True,
        "manifold.support_anchor_quantile": 0.10,
        "manifold.support_release_threshold": 0.58,
        "manifold.support_promote_min_size": 10,
        "manifold.support_overlap_threshold": 0.12,
        "manifold.volumetric_cut_damage_scale": 0.52,
        "manifold.volumetric_auth_damage_floor": 0.70,
        "manifold.volumetric_detached_damage_floor": 0.84,
        "manifold.crack_volume_feedback_gain": 0.36,
        "manifold.crack_volume_opening_gain": 0.40,
        "manifold.crack_volume_visited_floor": 0.46,
        "manifold.crack_volume_tip_floor": 0.60,
        "manifold.crack_volume_interior_scale": 0.72,
        "manifold.fragment_physical_gap_scale": 0.00003,
        "manifold.fragment_physical_release_velocity": 0.002,
        "manifold.fragment_physical_downward_bias": 0.26,
        "manifold.fragment_physical_release_frames": 18,
        "manifold.splitting_enabled": False,
        "gaussian_splatting.material_family": "brittle_moderate",
        "gaussian_splatting.crack_band_weight": 0.72,
        "gaussian_splatting.crack_visited_weight": 0.28,
        "gaussian_splatting.crack_tip_weight": 1.10,
        "gaussian_splatting.crack_core_weight": 1.10,
        "gaussian_splatting.split_gap_gain": 1.05,
        "gaussian_splatting.fragment_shell_gain": 1.05,
        "gaussian_splatting.fragment_contrast_gain": 1.02,
        "gaussian_splatting.debris_darkening": 0.16,
        "gaussian_splatting.shard_scale_gain": 1.0,
        "gaussian_splatting.shard_opacity_gain": 1.0,
        "gaussian_splatting.damage_scale_shrink": 0.48,
        "gaussian_splatting.damage_center_opacity_reduction": 0.76,
        "gaussian_splatting.diffuse_damage_strength": 0.10,
        "gaussian_splatting.crack_tip_scale_boost": 0.28,
        "gaussian_splatting.crack_tip_opacity_boost": 0.14,
    },
    "rough_quasi_brittle": {
        "manifold.material_family": "rough_quasi_brittle",
        "manifold.enable_front_propagation": True,
        "manifold.material_drive_floor": 0.03,
        "manifold.diffuse_damage_gain": 0.18,
        "manifold.diffuse_neighborhood_steps": 2,
        "manifold.fragment_detect_every": 2,
        "manifold.fragment_damage_threshold": 0.55,
        "manifold.min_fragment_particles": 50,
        "manifold.fragment_opening_weight": 0.46,
        "manifold.fragment_active_tip_weight": 0.24,
        "manifold.fragment_recent_front_weight": 0.18,
        "manifold.fragment_pair_break_weight": 0.14,
        "manifold.fragment_edge_memory_decay": 0.990,
        "manifold.fragment_edge_memory_weight": 0.92,
        "manifold.fragment_cut_diffusion_alpha": 0.50,
        "manifold.fragment_cut_diffusion_iters": 2,
        "manifold.fragment_cut_cos_gate_tangent": 0.50,
        "manifold.fragment_cut_cos_gate_normal": 0.40,
        "manifold.fragment_primary_cut_ratio": 0.65,
        "manifold.fragment_fallback_cut_ratio": 0.45,
        "manifold.fragment_min_boundary_edges": 20,
        "manifold.fragment_detached_node_decay": 0.98,
        "manifold.fragment_persistent_min_size": 6,
        "manifold.fragment_component_hysteresis": 0.28,
        "manifold.fragment_post_split_threshold_scale": 0.84,
        "manifold.fragment_impulse_strength": 3.40,
        "manifold.fragment_upward_bias": 0.48,
        "manifold.fragment_visual_offset_scale": 0.016,
        "manifold.fragment_visual_ramp_frames": 6,
        "manifold.fragment_impulse_boost_frames": 5,
        "manifold.fragment_event_boost": 1.10,
        "manifold.shard_enable": False,
        "manifold.shard_count_scale": 0.0,
        "manifold.fragment_offset_gain": 1.32,
        "manifold.debris_motion_gain": 0.0,
        "manifold.cut_surface_enable": True,
        "manifold.cut_vote_strength": 0.42,
        "manifold.tau_cross": 0.54,
        "manifold.tau_tangent": 0.42,
        "manifold.cut_core_damage_threshold": 0.16,
        "manifold.cut_core_opening_threshold": 0.10,
        "manifold.cut_hard_break_threshold": 0.36,
        "manifold.authoritative_cut_decay": 0.982,
        "manifold.authoritative_cut_threshold": 0.18,
        "manifold.support_loss_enable": True,
        "manifold.support_anchor_quantile": 0.14,
        "manifold.support_release_threshold": 0.60,
        "manifold.support_promote_min_size": 20,
        "manifold.support_overlap_threshold": 0.18,
        "manifold.volumetric_cut_damage_scale": 0.86,
        "manifold.volumetric_auth_damage_floor": 0.88,
        "manifold.volumetric_detached_damage_floor": 0.98,
        "manifold.crack_volume_feedback_gain": 0.54,
        "manifold.crack_volume_opening_gain": 0.44,
        "manifold.crack_volume_visited_floor": 0.58,
        "manifold.crack_volume_tip_floor": 0.70,
        "manifold.crack_volume_interior_scale": 0.90,
        "manifold.fragment_physical_gap_scale": 0.00022,
        "manifold.fragment_physical_release_velocity": 0.020,
        "manifold.fragment_physical_downward_bias": 0.42,
        "manifold.fragment_physical_release_frames": 36,
        "manifold.splitting_enabled": False,
        "gaussian_splatting.material_family": "rough_quasi_brittle",
        "gaussian_splatting.crack_band_weight": 1.08,
        "gaussian_splatting.crack_visited_weight": 0.92,
        "gaussian_splatting.crack_tip_weight": 0.88,
        "gaussian_splatting.crack_core_weight": 0.92,
        "gaussian_splatting.split_gap_gain": 1.18,
        "gaussian_splatting.fragment_shell_gain": 1.40,
        "gaussian_splatting.fragment_contrast_gain": 1.18,
        "gaussian_splatting.debris_darkening": 0.34,
        "gaussian_splatting.shard_scale_gain": 1.28,
        "gaussian_splatting.shard_opacity_gain": 1.05,
        "gaussian_splatting.damage_scale_shrink": 0.54,
        "gaussian_splatting.damage_center_opacity_reduction": 0.68,
        "gaussian_splatting.diffuse_damage_strength": 0.18,
        "gaussian_splatting.crack_tip_scale_boost": 0.18,
        "gaussian_splatting.crack_tip_opacity_boost": 0.08,
    },
    "diffuse_damage": {
        "manifold.material_family": "diffuse_damage",
        "manifold.enable_front_propagation": False,
        "manifold.material_drive_floor": 0.0,
        "manifold.diffuse_damage_gain": 0.34,
        "manifold.diffuse_neighborhood_steps": 3,
        "manifold.fragment_detect_every": 3,
        "manifold.fragment_damage_threshold": 0.95,
        "manifold.min_fragment_particles": 100,
        "manifold.fragment_opening_weight": 0.08,
        "manifold.fragment_active_tip_weight": 0.00,
        "manifold.fragment_recent_front_weight": 0.00,
        "manifold.fragment_pair_break_weight": 0.00,
        "manifold.fragment_edge_memory_decay": 0.90,
        "manifold.fragment_edge_memory_weight": 0.20,
        "manifold.fragment_cut_diffusion_alpha": 0.0,
        "manifold.fragment_cut_diffusion_iters": 0,
        "manifold.fragment_cut_cos_gate_tangent": 0.95,
        "manifold.fragment_cut_cos_gate_normal": 0.95,
        "manifold.fragment_primary_cut_ratio": 0.90,
        "manifold.fragment_fallback_cut_ratio": 0.70,
        "manifold.fragment_min_boundary_edges": 40,
        "manifold.fragment_detached_node_decay": 0.85,
        "manifold.fragment_persistent_min_size": 9999,
        "manifold.fragment_component_hysteresis": 0.95,
        "manifold.fragment_post_split_threshold_scale": 1.00,
        "manifold.fragment_impulse_strength": 0.0,
        "manifold.fragment_upward_bias": 0.0,
        "manifold.fragment_visual_offset_scale": 0.0,
        "manifold.fragment_visual_ramp_frames": 1,
        "manifold.fragment_impulse_boost_frames": 0,
        "manifold.fragment_event_boost": 1.0,
        "manifold.shard_enable": False,
        "manifold.shard_count_scale": 0.0,
        "manifold.fragment_offset_gain": 1.0,
        "manifold.debris_motion_gain": 0.0,
        "manifold.cut_surface_enable": False,
        "manifold.cut_vote_strength": 0.0,
        "manifold.authoritative_cut_decay": 0.90,
        "manifold.authoritative_cut_threshold": 0.95,
        "manifold.support_loss_enable": False,
        "manifold.support_anchor_quantile": 0.10,
        "manifold.support_release_threshold": 1.0,
        "manifold.support_promote_min_size": 9999,
        "manifold.support_overlap_threshold": 1.0,
        "manifold.volumetric_cut_damage_scale": 0.0,
        "manifold.volumetric_auth_damage_floor": 0.0,
        "manifold.volumetric_detached_damage_floor": 0.0,
        "manifold.crack_volume_feedback_gain": 0.04,
        "manifold.crack_volume_opening_gain": 0.02,
        "manifold.crack_volume_visited_floor": 0.06,
        "manifold.crack_volume_tip_floor": 0.08,
        "manifold.crack_volume_interior_scale": 0.20,
        "manifold.fragment_physical_gap_scale": 0.0,
        "manifold.fragment_physical_release_velocity": 0.0,
        "manifold.fragment_physical_downward_bias": 0.0,
        "manifold.fragment_physical_release_frames": 0,
        "manifold.splitting_enabled": False,
        "gaussian_splatting.material_family": "diffuse_damage",
        "gaussian_splatting.crack_band_weight": 0.0,
        "gaussian_splatting.crack_visited_weight": 0.0,
        "gaussian_splatting.crack_tip_weight": 0.0,
        "gaussian_splatting.crack_core_weight": 0.0,
        "gaussian_splatting.split_gap_gain": 0.55,
        "gaussian_splatting.fragment_shell_gain": 0.0,
        "gaussian_splatting.fragment_contrast_gain": 0.0,
        "gaussian_splatting.debris_darkening": 0.0,
        "gaussian_splatting.shard_scale_gain": 1.0,
        "gaussian_splatting.shard_opacity_gain": 1.0,
        "gaussian_splatting.damage_scale_shrink": 0.22,
        "gaussian_splatting.damage_center_opacity_reduction": 0.18,
        "gaussian_splatting.diffuse_damage_strength": 1.00,
        "gaussian_splatting.crack_tip_scale_boost": 0.0,
        "gaussian_splatting.crack_tip_opacity_boost": 0.0,
    },
    "neutral_reference": {
        "manifold.material_family": "neutral_reference",
        "manifold.enable_front_propagation": True,
        "manifold.material_drive_floor": 0.025,
        "manifold.diffuse_damage_gain": 0.16,
        "manifold.diffuse_neighborhood_steps": 2,
        "manifold.fragment_detect_every": 2,
        "manifold.fragment_damage_threshold": 0.50,
        "manifold.min_fragment_particles": 40,
        "manifold.fragment_opening_weight": 0.36,
        "manifold.fragment_active_tip_weight": 0.18,
        "manifold.fragment_recent_front_weight": 0.12,
        "manifold.fragment_pair_break_weight": 0.10,
        "manifold.fragment_edge_memory_decay": 0.975,
        "manifold.fragment_edge_memory_weight": 0.76,
        "manifold.fragment_cut_diffusion_alpha": 0.18,
        "manifold.fragment_cut_diffusion_iters": 1,
        "manifold.fragment_cut_cos_gate_tangent": 0.62,
        "manifold.fragment_cut_cos_gate_normal": 0.52,
        "manifold.fragment_primary_cut_ratio": 0.70,
        "manifold.fragment_fallback_cut_ratio": 0.50,
        "manifold.fragment_min_boundary_edges": 18,
        "manifold.fragment_detached_node_decay": 0.95,
        "manifold.fragment_persistent_min_size": 8,
        "manifold.fragment_component_hysteresis": 0.34,
        "manifold.fragment_post_split_threshold_scale": 0.90,
        "manifold.fragment_impulse_strength": 2.40,
        "manifold.fragment_upward_bias": 0.36,
        "manifold.fragment_visual_offset_scale": 0.010,
        "manifold.fragment_visual_ramp_frames": 6,
        "manifold.fragment_impulse_boost_frames": 4,
        "manifold.fragment_event_boost": 1.0,
        "manifold.shard_enable": False,
        "manifold.shard_count_scale": 0.0,
        "manifold.fragment_offset_gain": 1.0,
        "manifold.debris_motion_gain": 0.0,
        "manifold.cut_surface_enable": False,
        "manifold.cut_vote_strength": 0.0,
        "manifold.authoritative_cut_decay": 0.95,
        "manifold.authoritative_cut_threshold": 0.24,
        "manifold.support_loss_enable": True,
        "manifold.support_anchor_quantile": 0.10,
        "manifold.support_release_threshold": 0.56,
        "manifold.support_promote_min_size": 6,
        "manifold.support_overlap_threshold": 0.10,
        "manifold.volumetric_cut_damage_scale": 0.58,
        "manifold.volumetric_auth_damage_floor": 0.72,
        "manifold.volumetric_detached_damage_floor": 0.90,
        "manifold.crack_volume_feedback_gain": 0.42,
        "manifold.crack_volume_opening_gain": 0.38,
        "manifold.crack_volume_visited_floor": 0.50,
        "manifold.crack_volume_tip_floor": 0.66,
        "manifold.crack_volume_interior_scale": 0.85,
        "manifold.fragment_physical_gap_scale": 0.00005,
        "manifold.fragment_physical_release_velocity": 0.004,
        "manifold.fragment_physical_downward_bias": 0.30,
        "manifold.fragment_physical_release_frames": 20,
        "manifold.splitting_enabled": False,
        "gaussian_splatting.material_family": "neutral_reference",
        "gaussian_splatting.crack_band_weight": 0.80,
        "gaussian_splatting.crack_visited_weight": 0.45,
        "gaussian_splatting.crack_tip_weight": 0.95,
        "gaussian_splatting.crack_core_weight": 1.00,
        "gaussian_splatting.split_gap_gain": 1.0,
        "gaussian_splatting.fragment_shell_gain": 1.0,
        "gaussian_splatting.fragment_contrast_gain": 1.0,
        "gaussian_splatting.debris_darkening": 0.20,
        "gaussian_splatting.shard_scale_gain": 1.0,
        "gaussian_splatting.shard_opacity_gain": 1.0,
        "gaussian_splatting.damage_scale_shrink": 0.50,
        "gaussian_splatting.damage_center_opacity_reduction": 0.70,
        "gaussian_splatting.diffuse_damage_strength": 0.12,
        "gaussian_splatting.crack_tip_scale_boost": 0.22,
        "gaussian_splatting.crack_tip_opacity_boost": 0.10,
    },
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


class MaterialPriorAdapter:
    """Blend CLIP retrieval outputs into physics + fracture priors."""

    def __init__(self, score_temperature: float = 10.0):
        self.score_temperature = float(score_temperature)

    def normalize_scores(self, scores: Sequence[float]) -> np.ndarray:
        scores_np = np.asarray(list(scores), dtype=np.float64)
        if scores_np.size == 0:
            return scores_np
        logits = scores_np * self.score_temperature
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= np.maximum(weights.sum(), 1e-12)
        return weights

    def style_for_entry(self, entry: MaterialEntry) -> Dict[str, float]:
        style = dict(DEFAULT_STYLE)
        style.update(CATEGORY_STYLE_DB.get(entry.category, {}))
        name_l = entry.name.lower()
        for pattern, override in NAME_STYLE_OVERRIDES.items():
            if pattern in name_l:
                style.update(override)
        return style

    def family_for_entry(self, entry: MaterialEntry) -> str:
        name_l = entry.name.lower()
        for pattern, family in NAME_FAMILY_OVERRIDES.items():
            if pattern in name_l:
                return family
        return CATEGORY_TO_FAMILY.get(entry.category, "neutral_reference")

    def build_family_prior(
        self,
        topk_materials: Sequence[MaterialEntry],
        weights: Sequence[float],
    ) -> Dict[str, object]:
        pairs = list(zip(topk_materials, weights))
        scores = {family: 0.0 for family in FAMILY_NAMES}
        for entry, weight in pairs:
            scores[self.family_for_entry(entry)] += float(weight)

        family, best_score = max(scores.items(), key=lambda kv: kv[1], default=("neutral_reference", 0.0))
        top_family = "neutral_reference"
        top_weight = 0.0
        if pairs:
            top_entry, top_weight = max(pairs, key=lambda ew: float(ew[1]))
            top_family = self.family_for_entry(top_entry)
        if best_score < 0.34:
            family = "neutral_reference"
        if top_family != "neutral_reference" and top_weight >= 0.28:
            family = top_family
        elif top_family == "diffuse_damage" and top_weight >= 0.24:
            family = "diffuse_damage"
        if family == "diffuse_damage" and scores["sharp_brittle"] + scores["brittle_moderate"] > 0.62:
            family = "brittle_moderate"
        return {
            "family": family,
            "scores": scores,
        }

    def build_physics_prior(
        self,
        topk_materials: Sequence[MaterialEntry],
        weights: Sequence[float],
    ) -> Dict[str, float]:
        entries = list(topk_materials)
        weights_np = np.asarray(list(weights), dtype=np.float64)
        E = float(np.exp(np.sum(weights_np * np.log([max(e.E, 1e-12) for e in entries]))))
        Gc = float(np.exp(np.sum(weights_np * np.log([max(e.Gc, 1e-12) for e in entries]))))
        nu = float(np.sum(weights_np * np.asarray([e.nu for e in entries], dtype=np.float64)))
        density = float(np.sum(weights_np * np.asarray([e.density for e in entries], dtype=np.float64)))
        return {
            "E": E,
            "Gc": Gc,
            "nu": nu,
            "density": density,
        }

    def build_fracture_prior(
        self,
        topk_materials: Sequence[MaterialEntry],
        weights: Sequence[float],
        physics_prior: Dict[str, float],
        family: str,
    ) -> Dict[str, float]:
        entries = list(topk_materials)
        weights_np = np.asarray(list(weights), dtype=np.float64)
        prior = {}
        for key in STYLE_KEYS:
            prior[key] = float(np.sum(
                weights_np * np.asarray([self.style_for_entry(entry)[key] for entry in entries], dtype=np.float64)
            ))

        prior = self._couple_fracture_to_physics(prior, physics_prior)
        multipliers = FAMILY_STYLE_MULTIPLIERS.get(family, FAMILY_STYLE_MULTIPLIERS["neutral_reference"])
        for key, value in multipliers.items():
            prior[key] *= float(value)

        prior["tau_init"] = _clamp(prior["tau_init"], 0.12, 0.90)
        prior["growth_gain"] = _clamp(prior["growth_gain"], 0.20, 1.90)
        prior["band_width"] = _clamp(prior["band_width"], 1.00, 3.40)
        prior["band_fill_gain"] = _clamp(prior["band_fill_gain"], 0.08, 1.10)
        prior["open_gain"] = _clamp(prior["open_gain"], 0.08, 1.90)
        prior["split_threshold"] = _clamp(prior["split_threshold"], 0.22, 0.95)
        prior["edge_break_rate"] = _clamp(prior["edge_break_rate"], 0.10, 1.90)
        prior["branching_bias"] = _clamp(prior["branching_bias"], 0.00, 0.95)
        prior["anisotropy_strength"] = _clamp(prior["anisotropy_strength"], 0.02, 0.95)
        return prior

    def _couple_fracture_to_physics(
        self,
        fracture_prior: Dict[str, float],
        physics_prior: Dict[str, float],
    ) -> Dict[str, float]:
        out = dict(fracture_prior)

        gc_ratio = max(physics_prior["Gc"] / 100.0, 1e-8)
        e_ratio = max(physics_prior["E"] / 3.0e10, 1e-8)
        nu = float(physics_prior["nu"])

        brittle = _clamp(-np.log10(gc_ratio) / 2.0, -1.0, 1.0)
        compliant = _clamp(-np.log10(e_ratio) / 2.0, -1.0, 1.0)
        incompressible = _clamp((nu - 0.30) / 0.18, -1.0, 1.0)

        out["tau_init"] *= 1.0 - 0.18 * brittle + 0.14 * compliant + 0.06 * incompressible
        out["growth_gain"] *= 1.0 + 0.22 * brittle - 0.18 * compliant
        out["band_width"] *= 1.0 + 0.30 * compliant + 0.10 * incompressible
        out["band_fill_gain"] *= 1.0 + 0.20 * compliant + 0.08 * incompressible
        out["open_gain"] *= 1.0 + 0.22 * brittle - 0.10 * compliant
        out["split_threshold"] *= 1.0 - 0.16 * brittle + 0.20 * compliant + 0.10 * incompressible
        out["edge_break_rate"] *= 1.0 + 0.18 * brittle - 0.16 * compliant
        out["branching_bias"] *= 1.0 + 0.12 * compliant + 0.08 * incompressible
        out["anisotropy_strength"] *= 1.0 - 0.10 * incompressible

        out["tau_init"] = _clamp(out["tau_init"], 0.12, 0.90)
        out["growth_gain"] = _clamp(out["growth_gain"], 0.20, 1.90)
        out["band_width"] = _clamp(out["band_width"], 1.00, 3.40)
        out["band_fill_gain"] = _clamp(out["band_fill_gain"], 0.08, 1.10)
        out["open_gain"] = _clamp(out["open_gain"], 0.08, 1.90)
        out["split_threshold"] = _clamp(out["split_threshold"], 0.22, 0.95)
        out["edge_break_rate"] = _clamp(out["edge_break_rate"], 0.10, 1.90)
        out["branching_bias"] = _clamp(out["branching_bias"], 0.00, 0.95)
        out["anisotropy_strength"] = _clamp(out["anisotropy_strength"], 0.02, 0.95)
        return out

    def dominant_category(
        self,
        topk_materials: Sequence[MaterialEntry],
        weights: Sequence[float],
    ) -> str:
        totals: Dict[str, float] = {}
        for entry, weight in zip(topk_materials, weights):
            totals[entry.category] = totals.get(entry.category, 0.0) + float(weight)
        if not totals:
            return "other"
        return max(totals.items(), key=lambda kv: kv[1])[0]

    def build_material_prior(
        self,
        topk_materials: Sequence[MaterialEntry],
        scores: Sequence[float],
    ) -> Dict[str, object]:
        entries = list(topk_materials)
        weights = self.normalize_scores(scores)
        physics = self.build_physics_prior(entries, weights)
        family_prior = self.build_family_prior(entries, weights)
        family = family_prior["family"]
        fracture = self.build_fracture_prior(entries, weights, physics, family)
        dominant_category = self.dominant_category(entries, weights)
        runtime = self.fracture_prior_to_runtime_overrides(fracture, family)

        return {
            "family": family,
            "family_scores": family_prior["scores"],
            "physics": physics,
            "fracture": fracture,
            "runtime": runtime,
            "weights": weights.tolist(),
            "dominant_category": dominant_category,
            "top_k": [
                {
                    "name": entry.name,
                    "category": entry.category,
                    "score": float(score),
                    "weight": float(weight),
                    "family": self.family_for_entry(entry),
                }
                for entry, score, weight in zip(entries, scores, weights)
            ],
        }

    @staticmethod
    def scale_physics_to_mpm(
        physics_prior: Dict[str, float],
        base_E: float = 1.5e7,
        base_Gc: float = 6.0e4,
        base_density: float = 1200.0,
        ref_E: float = 3.0e10,
        ref_Gc: float = 100.0,
        ref_density: float = 1200.0,
    ) -> Dict[str, float]:
        e_ratio = max(float(physics_prior["E"]) / ref_E, 1e-8)
        gc_ratio = max(float(physics_prior["Gc"]) / ref_Gc, 1e-8)
        rho_ratio = max(float(physics_prior["density"]) / ref_density, 1e-8)

        # Keep MPM physics in a stable range and let fracture priors carry most
        # of the material-style variation. Wide physical ratios still influence
        # behavior, but they no longer erase crack growth by over-suppressing
        # the contact-wave response.
        E = base_E * _clamp(e_ratio ** 0.12, 0.85, 1.45)
        Gc = base_Gc * _clamp(gc_ratio ** 0.18, 0.70, 1.60)
        density = base_density * _clamp(rho_ratio ** 0.20, 0.85, 1.35)
        nu = _clamp(float(physics_prior["nu"]), 0.10, 0.49)

        return {
            "E": float(E),
            "Gc": float(Gc),
            "nu": nu,
            "density": float(density),
        }

    @staticmethod
    def fracture_prior_to_runtime_overrides(
        fracture_prior: Dict[str, float],
        family: str = "neutral_reference",
    ) -> Dict[str, float]:
        tau = float(fracture_prior["tau_init"])
        growth = float(fracture_prior["growth_gain"])
        band_width = float(fracture_prior["band_width"])
        band_fill = float(fracture_prior["band_fill_gain"])
        open_gain = float(fracture_prior["open_gain"])
        split_threshold = float(fracture_prior["split_threshold"])
        edge_break_rate = float(fracture_prior["edge_break_rate"])
        branching = float(fracture_prior["branching_bias"])
        anisotropy = float(fracture_prior["anisotropy_strength"])

        successor_topk = 1 if branching < 0.18 else (2 if branching < 0.60 else 3)
        max_branching_tips = int(round(6 + 18 * branching))
        branch_score_ratio = _clamp(0.99 - 0.18 * branching, 0.75, 0.99)
        branch_drive_threshold = _clamp(0.82 - 0.42 * branching, 0.18, 0.85)
        damage_spread = _clamp(0.08 + 0.08 * (band_width - 1.0) + 0.48 * band_fill, 0.08, 0.95)
        damage_source_scale = _clamp(0.18 + 0.22 * growth + 0.10 * band_fill, 0.12, 0.70)
        tip_propagation_scale = _clamp(0.25 + 0.50 * growth, 0.20, 1.20)
        opening_scale = _clamp(0.006 + 0.018 * open_gain, 0.004, 0.050)
        dC_max = _clamp(0.014 + 0.010 * growth + 0.005 * open_gain, 0.018, 0.045)
        drive_quantile = _clamp(0.95 - 0.10 * branching - 0.10 * (band_width - 1.0), 0.55, 0.98)
        front_threshold = _clamp(0.02 + 0.16 * tau, 0.02, 0.18)
        gaussian_damage_threshold = _clamp(0.28 - 0.08 * open_gain + 0.03 * band_width, 0.12, 0.30)
        crack_max_opening = _clamp(0.008 + 0.010 * open_gain, 0.006, 0.035)
        crack_gap_fraction = _clamp(0.25 + 0.20 * open_gain, 0.20, 0.75)
        crack_opacity_reduction = _clamp(0.48 + 0.24 * open_gain, 0.30, 0.95)
        crack_edge_darken = _clamp(0.65 + 0.35 * open_gain, 0.40, 1.40)
        crack_red_accent = _clamp(0.06 + 0.10 * growth + 0.10 * branching, 0.06, 0.35)
        impact_seed_magnitude = _clamp(0.14 + 0.08 * (1.0 - tau) + 0.02 * open_gain, 0.14, 0.28)
        split_gap_gain = _clamp(0.85 + 0.32 * open_gain, 0.70, 1.60)
        fragment_shell_gain = _clamp(0.80 + 0.26 * band_width + 0.12 * branching, 0.80, 1.60)
        fragment_contrast_gain = _clamp(0.82 + 0.18 * growth + 0.16 * open_gain, 0.75, 1.55)
        debris_darkening = _clamp(0.08 + 0.16 * band_fill + 0.10 * branching, 0.04, 0.42)
        shard_scale_gain = _clamp(0.85 + 0.20 * open_gain + 0.15 * branching, 0.75, 1.50)
        shard_opacity_gain = _clamp(0.82 + 0.16 * growth + 0.10 * open_gain, 0.75, 1.35)
        shard_count_scale = _clamp(0.18 + 0.55 * edge_break_rate + 0.30 * branching, 0.0, 1.60)
        fragment_offset_gain = _clamp(0.82 + 0.18 * open_gain + 0.12 * edge_break_rate, 0.80, 1.60)
        debris_motion_gain = _clamp(0.06 + 0.30 * branching + 0.22 * max(edge_break_rate - 0.8, 0.0), 0.0, 1.60)
        fragment_damage_threshold = _clamp(
            0.58 * split_threshold + 0.12 / max(edge_break_rate, 1e-6),
            0.18,
            0.78,
        )
        if family == "sharp_brittle":
            successor_topk = 1
            max_branching_tips = min(max_branching_tips, 8)
            branch_score_ratio = max(branch_score_ratio, 0.92)
            branch_drive_threshold = max(branch_drive_threshold, 0.62)
            damage_spread *= 0.65
            damage_source_scale *= 1.12
            tip_propagation_scale *= 1.05
            opening_scale *= 1.35
            dC_max *= 0.92
            drive_quantile = max(drive_quantile, 0.78)
            gaussian_damage_threshold = min(gaussian_damage_threshold + 0.03, 0.34)
            crack_max_opening *= 1.35
            crack_gap_fraction *= 1.16
            crack_opacity_reduction = min(crack_opacity_reduction + 0.08, 0.98)
            crack_edge_darken = min(crack_edge_darken + 0.12, 1.55)
            crack_red_accent = max(crack_red_accent - 0.03, 0.05)
            fragment_damage_threshold *= 0.82
            split_gap_gain *= 1.18
            fragment_shell_gain *= 1.08
            fragment_contrast_gain *= 1.18
            debris_darkening *= 0.85
            shard_scale_gain *= 1.08
            shard_opacity_gain *= 1.08
            shard_count_scale *= 0.85
            fragment_offset_gain *= 1.10
            debris_motion_gain *= 0.85
        elif family == "brittle_moderate":
            successor_topk = min(successor_topk, 2)
            max_branching_tips = min(max_branching_tips, 10)
            branch_score_ratio = max(branch_score_ratio, 0.88)
            branch_drive_threshold = max(branch_drive_threshold, 0.48)
            damage_spread *= 0.84
            opening_scale *= 1.10
            crack_gap_fraction *= 1.05
            fragment_damage_threshold *= 0.92
            split_gap_gain *= 1.02
            fragment_shell_gain *= 0.96
            fragment_contrast_gain *= 0.95
            debris_darkening *= 0.90
            shard_scale_gain *= 0.92
            shard_opacity_gain *= 0.95
            shard_count_scale *= 0.15
            fragment_offset_gain *= 0.98
            debris_motion_gain *= 0.20
        elif family == "rough_quasi_brittle":
            successor_topk = max(successor_topk, 2)
            max_branching_tips = max(max_branching_tips, 16)
            branch_score_ratio = max(branch_score_ratio - 0.06, 0.72)
            branch_drive_threshold = max(branch_drive_threshold - 0.10, 0.18)
            damage_spread *= 1.22
            damage_source_scale *= 0.92
            opening_scale *= 0.88
            gaussian_damage_threshold = max(gaussian_damage_threshold - 0.04, 0.10)
            crack_gap_fraction *= 0.92
            crack_opacity_reduction = max(crack_opacity_reduction - 0.10, 0.28)
            crack_red_accent = min(crack_red_accent + 0.05, 0.40)
            fragment_damage_threshold *= 1.04
            split_gap_gain *= 1.06
            fragment_shell_gain *= 1.24
            fragment_contrast_gain *= 1.08
            debris_darkening *= 1.28
            shard_scale_gain *= 1.18
            shard_opacity_gain *= 0.96
            shard_count_scale *= 1.28
            fragment_offset_gain *= 1.12
            debris_motion_gain *= 1.30
        elif family == "diffuse_damage":
            successor_topk = 0
            max_branching_tips = 0
            branch_score_ratio = 0.999
            branch_drive_threshold = 0.99
            damage_spread *= 1.36
            damage_source_scale *= 0.55
            tip_propagation_scale *= 0.30
            opening_scale *= 0.14
            dC_max *= 0.58
            drive_quantile = min(drive_quantile + 0.12, 0.995)
            front_threshold = 0.999
            gaussian_damage_threshold = min(gaussian_damage_threshold + 0.08, 0.36)
            crack_max_opening *= 0.25
            crack_gap_fraction *= 0.30
            crack_opacity_reduction *= 0.20
            crack_edge_darken *= 0.35
            crack_red_accent = 0.0
            impact_seed_magnitude *= 0.40
            fragment_damage_threshold = min(fragment_damage_threshold + 0.18, 0.92)
            split_gap_gain *= 0.35
            fragment_shell_gain = 0.0
            fragment_contrast_gain = 0.0
            debris_darkening = 0.0
            shard_scale_gain = 1.0
            shard_opacity_gain = 1.0
            shard_count_scale = 0.0
            fragment_offset_gain = 1.0
            debris_motion_gain = 0.0
        else:
            successor_topk = min(max(successor_topk, 1), 2)
            max_branching_tips = min(max_branching_tips, 12)
            damage_spread *= 0.92
            opening_scale *= 0.90
            crack_gap_fraction *= 0.90
            fragment_damage_threshold *= 0.96
            split_gap_gain *= 0.92
            fragment_shell_gain *= 0.96
            fragment_contrast_gain *= 0.96
            shard_count_scale *= 0.40
            debris_motion_gain *= 0.35

        runtime = {
            "manifold.tau_init": tau,
            "manifold.growth_gain": growth,
            "manifold.band_width": band_width,
            "manifold.band_fill_gain": band_fill,
            "manifold.open_gain": open_gain,
            "manifold.edge_break_rate": edge_break_rate,
            "manifold.branching_bias": branching,
            "manifold.anisotropy_strength": anisotropy,
            "manifold.successor_topk": successor_topk,
            "manifold.max_branching_tips": max_branching_tips,
            "manifold.branch_score_ratio": branch_score_ratio,
            "manifold.branch_drive_threshold": branch_drive_threshold,
            "manifold.damage_spread": damage_spread,
            "manifold.damage_source_scale": damage_source_scale,
            "manifold.tip_propagation_scale": tip_propagation_scale,
            "manifold.opening_scale": opening_scale,
            "manifold.dC_max": dC_max,
            "manifold.drive_quantile": drive_quantile,
            "manifold.front_threshold": front_threshold,
            "manifold.impact_seed_magnitude": impact_seed_magnitude,
            "manifold.split_threshold": split_threshold,
            "manifold.opacity_threshold": _clamp(split_threshold * 0.82, 0.20, 0.88),
            "manifold.flatten_threshold": _clamp(split_threshold * 0.58, 0.16, 0.70),
            "manifold.split_offset_scale": _clamp(0.80 + 0.90 * open_gain, 0.70, 2.40),
            "manifold.fragment_damage_threshold": fragment_damage_threshold,
            "manifold.shard_count_scale": shard_count_scale,
            "manifold.fragment_offset_gain": fragment_offset_gain,
            "manifold.debris_motion_gain": debris_motion_gain,
            "gaussian_splatting.damage_threshold": gaussian_damage_threshold,
            "gaussian_splatting.crack_max_opening": crack_max_opening,
            "gaussian_splatting.crack_gap_fraction": crack_gap_fraction,
            "gaussian_splatting.crack_opacity_reduction": crack_opacity_reduction,
            "gaussian_splatting.crack_edge_darken": crack_edge_darken,
            "gaussian_splatting.crack_red_accent": crack_red_accent,
            "gaussian_splatting.split_gap_gain": split_gap_gain,
            "gaussian_splatting.fragment_shell_gain": fragment_shell_gain,
            "gaussian_splatting.fragment_contrast_gain": fragment_contrast_gain,
            "gaussian_splatting.debris_darkening": debris_darkening,
            "gaussian_splatting.shard_scale_gain": shard_scale_gain,
            "gaussian_splatting.shard_opacity_gain": shard_opacity_gain,
        }
        runtime.update(FAMILY_RUNTIME_PRESETS.get(family, FAMILY_RUNTIME_PRESETS["neutral_reference"]))
        return runtime
