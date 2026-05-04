"""
Material-prior adapter for CLIP-driven fracture behavior.

This layer translates semantic CLIP retrieval results into:
  1. physical priors   : E, Gc, nu, density
  2. fracture priors   : crack-growth style parameters
  3. runtime overrides : manifold + visualization controls
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

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

FRAGMENT_CAPABLE_FAMILIES = {
    "sharp_brittle",
    "brittle_moderate",
    "rough_quasi_brittle",
    "neutral_reference",
}


FRACTURE_PRIOR_BOUNDS = {
    "tau_init": (0.12, 0.90),
    "growth_gain": (0.20, 1.90),
    "band_width": (1.00, 3.40),
    "band_fill_gain": (0.08, 1.10),
    "open_gain": (0.08, 1.90),
    "split_threshold": (0.22, 0.95),
    "edge_break_rate": (0.10, 1.90),
    "branching_bias": (0.00, 0.95),
    "anisotropy_strength": (0.02, 0.95),
}


SENTENCE_STYLE_RULES = (
    {
        "name": "diffuse_microcrack",
        "tokens": (
            "diffuse", "microcrack", "micro crack", "tiny crack",
            "tiny surface", "scratch", "scratches", "shallow",
            "without visible fracture", "no visible fracture",
            "without brittle fracture", "denting", "deforming without",
        ),
        "fracture_mult": {
            "tau_init": 1.55,
            "growth_gain": 0.42,
            "band_width": 1.65,
            "band_fill_gain": 1.55,
            "open_gain": 0.22,
            "split_threshold": 1.45,
            "edge_break_rate": 0.26,
            "branching_bias": 0.35,
            "anisotropy_strength": 0.55,
        },
        "runtime": {
            "manifold.enable_front_propagation": False,
            "manifold.successor_topk": 0,
            "manifold.max_branching_tips": 0,
            "manifold.branch_drive_threshold": 0.99,
            "manifold.front_threshold": 0.98,
            "manifold.impact_seed_magnitude": 0.035,
            "manifold.edge_break_rate": 0.0,
            "manifold.fragment_damage_threshold": 0.95,
            "manifold.open_crack_release_enable": False,
            "manifold.brittle_release_intensity": 0.0,
            "gaussian_splatting.crack_gap_fraction": 0.08,
            "gaussian_splatting.crack_opacity_reduction": 0.16,
        },
    },
    {
        "name": "spiderweb_branching",
        "tokens": (
            "spiderweb", "spider web", "branching", "branched",
            "network", "web cracks", "wide branching",
            "connected cracks", "intersecting cracks", "cracks meet",
        ),
        "fracture_mult": {
            "tau_init": 0.92,
            "growth_gain": 1.14,
            "band_width": 1.10,
            "band_fill_gain": 1.18,
            "open_gain": 1.02,
            "split_threshold": 0.90,
            "edge_break_rate": 1.12,
            "branching_bias": 4.20,
            "anisotropy_strength": 0.68,
        },
        "runtime": {
            "manifold.max_seed_points": 8,
            "manifold.seed_quantile": 0.988,
            "manifold.min_seed_spacing": 0.024,
            "manifold.successor_topk": 3,
            "manifold.max_branching_tips": 24,
            "manifold.branch_score_ratio": 0.74,
            "manifold.branch_drive_threshold": 0.22,
            "manifold.branching_bias": 0.78,
            "manifold.drive_quantile": 0.58,
            "manifold.damage_spread": 0.14,
            "manifold.damage_source_scale": 0.48,
            "manifold.edge_break_rate": 1.00,
            "manifold.fragment_damage_threshold": 0.40,
            "manifold.fragment_edge_memory_weight": 0.78,
            "manifold.fragment_cut_diffusion_alpha": 0.08,
            "manifold.fragment_cut_diffusion_iters": 1,
            "manifold.fragment_persistent_min_size": 8,
            "manifold.fragment_persistent_min_size_ratio": 0.0,
            "manifold.cut_vote_strength": 0.58,
            "manifold.cut_core_damage_threshold": 0.20,
            "manifold.cut_core_opening_threshold": 0.14,
            "manifold.cut_hard_break_threshold": 0.42,
            "manifold.authoritative_cut_threshold": 0.22,
            "manifold.crack_connected_release_only": True,
            # No explicit released-ratio cap: spiderweb's visual is
            # carried by the fracture_mult (low branching_bias, wide
            # bands) and crack_connected_release_only=True, not by a
            # hand-tuned release cap.  Material physics decides the
            # actual release ratio.
            "manifold.open_crack_release_enable": False,
            "manifold.open_crack_release_max_patches": 0,
            "manifold.open_crack_release_threshold": 1.0,
            "manifold.brittle_release_intensity": 1.25,
            "manifold.impact_release_gain": 1.12,
            "manifold.fragment_impulse_strength": 0.0,
            "manifold.fragment_event_boost": 1.0,
            "manifold.fragment_impulse_boost_frames": 0,
            # Spiderweb: web cracks separate but the body holds more
            # than radial; moderate release window and lateral bias.
            "manifold.fragment_physical_gap_scale": 0.0007,
            "manifold.fragment_physical_release_velocity": 0.015,
            "manifold.fragment_physical_downward_bias": 0.30,
            "manifold.fragment_physical_release_frames": 50,
            "manifold.fragment_physical_lateral_bias": 0.45,
            "manifold.fragment_physical_spin_gain": 0.05,
            "manifold.debris_motion_gain": 0.0,
            "manifold.shard_enable": False,
            "manifold.shard_count_scale": 0.0,
        },
    },
    {
        "name": "radial_shatter",
        "tokens": (
            "radial", "many sharp", "shattering", "shatter",
            "many cracks", "starburst", "localized shatter",
            "localized cracks", "crack-connected", "connected radial",
            "shard", "shards", "detached shard", "detached shards",
        ),
        "priority_tokens": (
            "shard", "shards", "detached shard", "detached shards",
            "shattering", "shatter", "starburst",
        ),
        "fracture_mult": {
            "tau_init": 0.88,
            "growth_gain": 1.22,
            "band_width": 0.94,
            "band_fill_gain": 1.05,
            "open_gain": 1.25,
            "split_threshold": 0.82,
            "edge_break_rate": 1.24,
            "branching_bias": 2.80,
            "anisotropy_strength": 0.82,
        },
        "runtime": {
            "manifold.max_seed_points": 14,
            "manifold.seed_quantile": 0.978,
            "manifold.min_seed_spacing": 0.016,
            "manifold.successor_topk": 6,
            "manifold.max_branching_tips": 72,
            "manifold.branch_score_ratio": 0.74,
            "manifold.branch_drive_threshold": 0.10,
            "manifold.branching_bias": 0.90,
            "manifold.min_successor_score": 0.075,
            "manifold.drive_quantile": 0.44,
            # AT2 propagation: full burst mode (cracks complete in 1-2
            # sim frames).  Visual "sudden transform" feel is mitigated
            # at the playback layer (30/60 FPS viewer) where each sim
            # frame plays at ~16-33ms, so the ~1-2 frame burst spans
            # ~30-60ms of real playback -- close to how brittle glass
            # actually fractures.
            "manifold.front_substeps": 4,
            "manifold.at2_drive_gain": 1.0,
            "manifold.damage_spread": 0.12,
            "manifold.damage_source_scale": 0.50,
            "manifold.edge_break_rate": 1.00,
            "manifold.fragment_damage_threshold": 0.42,
            "manifold.fragment_primary_cut_ratio": 0.48,
            "manifold.fragment_fallback_cut_ratio": 0.32,
            "manifold.fragment_min_boundary_edges": 10,
            "manifold.fragment_edge_memory_weight": 0.78,
            "manifold.fragment_cut_diffusion_alpha": 0.08,
            "manifold.fragment_cut_diffusion_iters": 1,
            "manifold.cut_vote_strength": 0.62,
            "manifold.cut_core_damage_threshold": 0.22,
            "manifold.cut_core_opening_threshold": 0.14,
            "manifold.cut_hard_break_threshold": 0.44,
            "manifold.authoritative_cut_threshold": 0.24,
            "manifold.support_release_threshold": 0.34,
            "manifold.support_overlap_threshold": 0.045,
            "manifold.crack_connected_release_only": True,
            "manifold.open_crack_release_enable": False,
            "manifold.open_crack_release_max_patches": 0,
            "manifold.open_crack_release_threshold": 1.0,
            "manifold.brittle_release_intensity": 1.80,
            "manifold.impact_release_gain": 1.34,
            # Middle-ground thresholds: between v1.5 paper-baseline
            # (120/24 -> ~50 chunky fragments) and v6 (8/8 -> 308
            # micro-fragments, looked like exploding noise).  Values
            # below are calibrated for ~10K particles and roughly
            # ~100--150 fragments.  For higher particle counts the
            # ratio (0.0012) auto-scales the persistent threshold;
            # physical_min_size and render_min_size should additionally
            # be scaled by sqrt(N / 10K) at runtime if a coarser
            # fragment density is desired.
            "manifold.fragment_persistent_min_size": 40,
            "manifold.fragment_persistent_min_size_ratio": 0.0012,
            "manifold.fragment_component_hysteresis": 0.22,
            "manifold.fragment_render_min_size": 3,
            "manifold.fragment_physical_min_size": 12,
            # Curvature-weighted anisotropy: bias crack normals along the
            # local principal-curvature tangent so fractures follow
            # natural ridge / curvature lines instead of the kNN-grid
            # axes.  0.4 = moderate bias; flat regions are gated by
            # per-node anisotropy so radial cracks at impact still
            # propagate freely.
            "manifold.curvature_weight": 0.4,
            # Causal-support gate: a fragment candidate is rejected
            # unless the mean cut-vote across its boundary is at least
            # this ratio.  Eliminates the "fragments form before crack
            # tips arrive" artefact -- AT2 damage diffuses wider than
            # tips travel, and without this gate ~38% of boundaries are
            # formed without crack support.  0.55 lets natural radial
            # shatter still complete in 1-2 frames (burst mode) while
            # blocking the diffusion-only patches that previously
            # showed up as ghost fragments in raw-graph diagnostics.
            "manifold.fragment_boundary_cut_min_ratio": 0.55,
            "manifold.fragment_physical_overlap_threshold": 0.06,
            "manifold.fragment_impulse_strength": 0.0,
            "manifold.fragment_event_boost": 1.0,
            "manifold.fragment_impulse_boost_frames": 0,
            "manifold.fragment_offset_gain": 1.0,
            "manifold.debris_motion_gain": 0.0,
            # Radial shatter: small initial separation kick, then
            # natural gravity-driven fall.  Earlier (v5) settings of
            # release_velocity 0.025 + lateral_bias 0.55 over 60
            # substeps produced an explosive lateral burst rather
            # than a "rain down" look.  Softened to a gentle nudge
            # (release window 30 substeps) so gravity dominates the
            # rest of the post-impact trajectory.
            "manifold.fragment_physical_gap_scale": 0.0006,
            # Punchy release impulse: 0.4 over 10 frames (= 0.04 / frame
            # peak) carries the pre-impact kinetic energy outward into
            # fragments before MPM floor contact zeros v_com.  Earlier
            # 0.010 over 30 frames was too gentle -- fragments started
            # nearly stationary after impact ("느리게 처음 움직이는").
            "manifold.fragment_physical_release_velocity": 0.4,
            "manifold.fragment_physical_downward_bias": 0.65,
            "manifold.fragment_physical_release_frames": 10,
            "manifold.fragment_physical_lateral_bias": 0.20,
            "manifold.fragment_physical_spin_gain": 0.04,
            # Per-fragment random jitter on release_dir / spin_axis /
            # speed.  Without this all fragments share the same outward-
            # plus-down direction and synchronized release_velocity ramp,
            # v30 LOCKED radial-shatter physics profile.
            #
            # Visual hacks (release_velocity / jitter / fragment offset)
            # all OFF -- the only KE source for fragments is the
            # physically-motivated Griffith stress release at graduation,
            # and the only KE source for the base body is the unified
            # rigid-body impact response.  Shape matching is fully rigid
            # (0.97 / 0.97) for both body and fragments so post-impact
            # particles follow rigid-body kinematics on top of MPM grid
            # physics.  shape_match_velocity_blend = 0 disables the
            # correction-velocity injection that previously bypassed
            # mpm.damping and produced visible base-body oscillation.
            "manifold.fragment_release_jitter": 0.0,
            "manifold.fragment_offset_gain": 0.0,
            "manifold.fragment_visual_offset_scale": 0.0,
            "manifold.fragment_physical_release_velocity": 0.0,
            "manifold.fragment_physical_downward_bias": 0.0,
            "manifold.shape_match_strength": 0.97,
            "manifold.shape_match_fragment_strength": 0.97,
            "manifold.shape_match_velocity_blend": 0.0,
            # Unified impact impulse (rigid-body form): horizontal v_com
            # slide along body-COM-to-impact-center offset + tumble omega
            # around the perpendicular horizontal axis.  Per-particle
            # radial impulses sum to ~zero in v_com, so they get absorbed
            # by shape match within a few substeps -- only NET v_com +
            # omega changes survive.  scale=0.05 + tumble=0.20 gives a
            # measured horizontal slide and a clear sideways roll for
            # the intact base remnant.
            "manifold.unified_impact_impulse_scale": 0.05,
            "manifold.unified_impact_tumble_scale": 0.20,
            # Griffith stress-driven fragment release: per-particle KE
            # injection at fragment graduation, magnitude proportional
            # to sqrt(principal stress), direction along the principal
            # tensile eigenvector with a downward bias and a random
            # +/- sign per particle (a real bond opens to BOTH sides).
            # gain=0.001 puts the kick in the 0-2.4 m/s band consistent
            # with v_impact ~35 stored elastic energy converted at the
            # fracture surface.
            # Per-fragment-at-graduation NET rigid-body release.  Griffith
            # gives stress-correct per-particle KE but sums to ~0 at the
            # COM (signed +/- per particle), so without a NET COM kick the
            # detached fragments just inherit the parent's downward MPM
            # velocity and fall straight down -- visually they "pop, then
            # stop, then drop".  v_com_gain=0.55 m/s is in the same band
            # as the whole-body unified impact slide (0.05 * v_impact ~
            # 1.75 m/s) but per-fragment it scales naturally smaller.
            # tumble_gain=1.6 yields ~2.2 rad/s spin for a 0.4-unit mesh,
            # i.e. a visible roll over ~3 seconds before damping.
            # Baseline scatter values are calibrated for a GLASS-TIER
            # reference material (E=70 GPa, Gc=5 J/m^2 -> brittleness 1.0).
            # _apply_material_scatter_scaling multiplies these by the
            # actual material's brittleness, so ceramic / concrete /
            # rubber inherit reduced scatter without retuning the style.
            # Lateral scatter — radial_shatter is the pre-pulverization
            # tier so spread is significant but less than full pulverize.
            "manifold.fragment_release_v_com_gain": 20.0,
            "manifold.fragment_physical_max_speed": 50.0,
            # Voronoi pre-fracture: partial pulverization for radial_shatter.
            "manifold.voronoi_enable": True,
            "manifold.voronoi_n_cells": 80,
            "manifold.voronoi_seed_distribution": "impact_biased",
            "manifold.voronoi_bond_break_threshold": 0.20,
            "manifold.voronoi_bond_aging_per_frame": 0.005,
            "manifold.voronoi_impact_shock_radius": 0.12,
            "manifold.voronoi_cascade_radius": 1.0,
            "manifold.voronoi_force_shrink_max_frac": 0.40,
            "manifold.particle_speed_cap": 60.0,
            # Per-particle floor bounce override (recovers bounce that
            # the MPM "slip" BC would otherwise eat).  0.45 gives a
            # crisp half-velocity rebound for glass-tier shards.
            "manifold.fragment_floor_restitution": 0.45,
            # Post-impact gravity matches the free-fall magnitude so
            # acceleration is uniform across impact (no visible
            # "hovering" after contact).  Stronger damping (0.95) plus
            # velocity_blend=0 above kills residual elastic vibration
            # within a few substeps.
            "manifold.post_impact_gravity_z": -3500.0,
            "manifold.post_impact_damping": 0.95,
            "manifold.shard_enable": False,
            "manifold.shard_count_scale": 0.0,
        },
    },
    {
        "name": "complete_pulverization",
        "tokens": (
            "pulverize", "pulverized", "pulverizes", "completely pulverized",
            "shattered to dust", "fine powder", "ultra shatter",
            "explode into dust", "exploded into hundreds",
            "exploding into hundreds of tiny shards",
            "obliterate", "obliterated", "totally shattered",
        ),
        "priority_tokens": (
            "pulverize", "pulverized", "ultra shatter",
            "shattered to dust", "exploded into hundreds",
            "totally shattered", "obliterated",
        ),
        "fracture_mult": {
            "tau_init": 0.78,            # easier to start cracks
            "growth_gain": 1.45,         # cracks grow faster
            "band_width": 0.86,          # narrower bands -> sharper edges
            "band_fill_gain": 1.18,
            "open_gain": 1.42,           # more aperture
            "split_threshold": 0.72,     # more splits
            "edge_break_rate": 1.55,     # 50% more edges break
            "branching_bias": 4.20,      # extreme branching
            "anisotropy_strength": 0.78,
        },
        "runtime": {
            # Geometrically saturated radial shatter -> hundreds of tiny
            # shards, near-zero base remnant.  Inherits the v30 physics
            # profile (Griffith release + global p2g2p + AT2 halt) but
            # cranks the fracture-side parameters to "ultra-brittle":
            # nearly every graph patch graduates to a physical fragment,
            # the family cap is raised so the base remnant ratio drops
            # below ~10%, and Griffith gain doubles so the per-particle
            # release impulse is more visibly explosive.
            "manifold.max_seed_points": 24,
            "manifold.seed_quantile": 0.965,
            "manifold.min_seed_spacing": 0.012,
            "manifold.successor_topk": 8,
            "manifold.max_branching_tips": 110,
            "manifold.branch_score_ratio": 0.62,
            "manifold.branch_drive_threshold": 0.06,
            "manifold.branching_bias": 1.20,
            "manifold.min_successor_score": 0.060,
            "manifold.drive_quantile": 0.40,
            "manifold.front_substeps": 4,
            "manifold.damage_spread": 0.16,
            "manifold.damage_source_scale": 0.70,
            "manifold.edge_break_rate": 1.40,
            "manifold.fragment_damage_threshold": 0.34,
            "manifold.fragment_primary_cut_ratio": 0.42,
            "manifold.fragment_fallback_cut_ratio": 0.26,
            "manifold.fragment_min_boundary_edges": 6,
            "manifold.fragment_persistent_min_size": 12,
            "manifold.fragment_persistent_min_size_ratio": 0.0008,
            "manifold.fragment_render_min_size": 3,
            "manifold.fragment_physical_min_size": 6,
            "manifold.fragment_physical_overlap_threshold": 0.05,
            "manifold.support_release_threshold": 0.28,
            "manifold.crack_connected_release_only": True,
            "manifold.brittle_release_intensity": 2.10,
            "manifold.impact_release_gain": 1.60,
            # Inherit v30 motion profile (no visual hacks, Griffith
            # release on, AT2 halt on, etc.) -- but with stronger
            # Griffith gain so the explosive feel matches the prompt.
            "manifold.fragment_release_jitter": 0.0,
            "manifold.fragment_offset_gain": 0.0,
            "manifold.fragment_visual_offset_scale": 0.0,
            "manifold.fragment_physical_release_velocity": 0.0,
            "manifold.fragment_physical_downward_bias": 0.0,
            "manifold.shape_match_strength": 0.97,
            "manifold.shape_match_fragment_strength": 0.97,
            "manifold.shape_match_velocity_blend": 0.0,
            "manifold.unified_impact_impulse_scale": 0.05,
            "manifold.unified_impact_tumble_scale": 0.20,
            # Stronger NET COM release than radial_shatter -- complete
            # pulverization is the explosive end of the brittle spectrum,
            # so each shard carries away more of the released elastic
            # energy as bulk translational + rotational KE.
            # Baseline scatter values calibrated for glass-tier (brittleness 1.0).
            # complete_pulverization is the explosive end of the spectrum --
            # higher v_com gain than radial_shatter (a glass shatter prompt
            # with the "completely pulverized" wording should produce a
            # visibly more dramatic explosion than the same prompt with
            # "shattering into many sharp radial cracks").
            # Explosive lateral scatter: v_com_gain is the XY-radial
            # speed (m/s) the chunk takes away from the impact axis.
            # 8 m/s + max_speed 18 lets fragments traverse ~25% of the
            # world over their flight, which reads as "흩날린다" rather
            # than "주저앉는다".  upward_fraction 0.25 = brief arc lift,
            # the dominant motion is horizontal.
            "manifold.fragment_release_v_com_gain": 35.0,
            "manifold.fragment_physical_max_speed": 50.0,
            # Per-particle floor bounce override (recovers from slip BC).
            # 0.50 = moderate glass shard bounce.
            # Floor restitution disabled: slip BC at the Z=0 wall now
            # handles the floor, and combining the two creates a yo-yo
            # (slip kills v_z at the wall; restitution then injects
            # +v_z from saved pre-step velocity, sending the particle
            # back up; gravity pulls it down again; cycle).
            "manifold.fragment_floor_restitution": 0.0,
            # Voronoi pre-fracture: full pulverization, no base remnant.
            "manifold.voronoi_enable": True,
            "manifold.voronoi_n_cells": 250,
            "manifold.voronoi_seed_distribution": "impact_biased",
            "manifold.voronoi_bond_break_threshold": 0.15,
            # Time-driven bond aging disabled in favor of stress-wave
            # propagation: bonds outside the wave can't break by aging,
            # they wait for the wave-front to reach them.  This produces
            # impact-zone-first fracture (gradient breakage) instead of
            # uniform simultaneous breakage everywhere (which reads as
            # "explosion" not "shatter").
            "manifold.voronoi_bond_aging_per_frame": 0.0,
            "manifold.voronoi_impact_shock_radius": 0.10,
            # Wave propagation speed: bonds within `_wave_radius` of
            # impact_center can break each frame; radius grows by this
            # much per call.  0.025 = wave reaches body extent (~0.4)
            # in ~16 frames, matching the typical 30-60 frame post-
            # impact window.
            "manifold.voronoi_wave_speed_per_frame": 0.025,
            "manifold.voronoi_cascade_radius": 1.0,
            "manifold.voronoi_force_shrink_max_frac": 0.05,
            "manifold.fragment_release_position_offset": 0.05,
            "manifold.particle_speed_cap": 80.0,
            # Aggressive angular damping for residual base + fragments.
            # static_omega=25 puts kinetic damping (0.985) only above
            # 25 rad/s; everything else gets strong static damping
            # (0.5/frame), killing residual-base spin from impact tumble
            # within ~5-10 frames so the central fragment doesn't keep
            # spinning indefinitely.
            "manifold.shape_match_static_omega": 25.0,
            "manifold.post_impact_gravity_z": -4500.0,
            "manifold.post_impact_damping": 0.999,
            "manifold.curvature_weight": 0.4,
            "manifold.fragment_boundary_cut_min_ratio": 0.45,
            "manifold.shard_enable": False,
            "manifold.shard_count_scale": 0.0,
        },
    },
    {
        "name": "chunky_crumble",
        "tokens": (
            "crumbling", "crumble", "chunks", "chunk", "granular",
            "gritty", "rough pieces", "irregular chunks",
        ),
        "fracture_mult": {
            "tau_init": 0.94,
            "growth_gain": 1.04,
            "band_width": 1.25,
            "band_fill_gain": 1.22,
            "open_gain": 0.92,
            "split_threshold": 0.92,
            "edge_break_rate": 1.20,
            "branching_bias": 1.45,
            "anisotropy_strength": 0.78,
        },
        "runtime": {
            "manifold.max_seed_points": 6,
            "manifold.seed_quantile": 0.990,
            "manifold.min_seed_spacing": 0.030,
            "manifold.successor_topk": 3,
            "manifold.max_branching_tips": 28,
            "manifold.branch_score_ratio": 0.76,
            "manifold.branch_drive_threshold": 0.24,
            "manifold.branching_bias": 0.82,
            "manifold.open_crack_release_max_patches": 8,
            "manifold.open_crack_release_threshold": 0.36,
            "manifold.brittle_release_intensity": 1.35,
            "manifold.impact_release_gain": 1.18,
        },
    },
    {
        "name": "single_smooth",
        "tokens": (
            "one long", "single", "smooth", "clean fracture",
            "clean crack", "clean split", "fracture line", "split in half",
        ),
        "fracture_mult": {
            "tau_init": 1.12,
            "growth_gain": 0.92,
            "band_width": 0.78,
            "band_fill_gain": 0.62,
            "open_gain": 1.16,
            "split_threshold": 0.92,
            "edge_break_rate": 0.76,
            "branching_bias": 0.18,
            "anisotropy_strength": 1.85,
        },
        "runtime": {
            "manifold.successor_topk": 1,
            "manifold.max_branching_tips": 2,
            "manifold.branch_score_ratio": 0.99,
            "manifold.branch_drive_threshold": 0.88,
            "manifold.drive_quantile": 0.86,
            "manifold.damage_spread": 0.18,
            "manifold.fragment_persistent_min_size": 160,
            "manifold.fragment_persistent_min_size_ratio": 0.0040,
            "manifold.fragment_primary_cut_ratio": 0.66,
            "manifold.fragment_fallback_cut_ratio": 0.52,
            # No explicit released-ratio cap: single_smooth's visual is
            # carried by the fracture_mult (high tau_init, narrow
            # branching, drive_quantile=0.86) so cracks rarely propagate
            # to fragment-graduation in the first place.  Material
            # physics decides the actual release ratio.
            "manifold.open_crack_release_max_patches": 2,
            "manifold.open_crack_release_threshold": 0.44,
            "manifold.brittle_release_intensity": 0.90,
        },
    },
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

MATERIAL_CONTEXT_SPLITS = (
    " dropped on ",
    " falling on ",
    " colliding with ",
    " impacting ",
    " against ",
    " onto ",
    " on concrete",
)

MATERIAL_HINT_RULES = (
    {
        "name": "glass",
        "tokens": ("glass", "soda-lime", "tempered glass", "bottle", "pane", "window"),
        "category_bonus": {"glass": 0.46},
        "name_bonus": (("glass", 0.28), ("soda-lime", 0.36), ("tempered", 0.26)),
    },
    {
        "name": "rubber",
        "tokens": ("rubber", "latex", "elastomer", "neoprene", "silicone"),
        "category_bonus": {"polymer": 0.22},
        "name_bonus": (
            ("rubber", 0.68),
            ("latex", 0.48),
            ("elastomer", 0.46),
            ("neoprene", 0.40),
            ("silicone", 0.32),
        ),
    },
    {
        "name": "metal",
        "tokens": (
            "steel", "structural steel", "stainless steel", "iron",
            "cast iron", "aluminum", "aluminium", "metal",
        ),
        "category_bonus": {"metal": 0.58},
        "name_bonus": (
            ("structural steel", 0.78),
            ("stainless steel", 0.70),
            ("steel", 0.62),
            ("cast iron", 0.44),
            ("iron", 0.34),
            ("aluminum", 0.34),
            ("aluminium", 0.34),
        ),
    },
    {
        "name": "ceramic",
        "tokens": ("ceramic", "porcelain", "stoneware", "china", "mug"),
        "category_bonus": {"ceramic": 0.42},
        "name_bonus": (("porcelain", 0.44), ("ceramic", 0.34), ("stoneware", 0.34)),
    },
    {
        "name": "concrete",
        "tokens": ("concrete", "cement", "mortar"),
        "category_bonus": {"concrete": 0.42},
        "name_bonus": (("concrete", 0.34), ("cement", 0.28), ("mortar", 0.28)),
    },
    {
        "name": "stone",
        "tokens": ("sandstone", "limestone", "marble", "stone", "rock"),
        "category_bonus": {"stone": 0.34},
        "name_bonus": (("sandstone", 0.44), ("limestone", 0.34), ("marble", 0.34)),
    },
    {
        "name": "ice",
        "tokens": ("ice", "frozen"),
        "category_bonus": {"ice": 0.42},
        "name_bonus": (("ice", 0.40), ("frozen", 0.22)),
    },
)

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
        "manifold.impact_release_gain": 1.10,
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
        "manifold.impact_release_gain": 1.00,
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
        "manifold.impact_release_gain": 1.05,
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
        "manifold.impact_release_gain": 0.75,
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
        "manifold.shape_match_strength": 0.18,
        "manifold.shape_match_fragment_strength": 0.28,
        "manifold.shape_match_damaged_strength": 0.10,
        "manifold.shape_match_velocity_blend": 0.06,
        "manifold.rigid_contact_restitution": 0.72,
        "manifold.rigid_contact_angular_gain": 0.22,
        "manifold.rigid_contact_friction": 0.04,
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
        "manifold.impact_release_gain": 1.0,
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

    def __init__(
        self,
        score_temperature: float = 10.0,
        enable_style_head: bool = True,
        style_head_confidence: float = 0.55,
    ):
        self.score_temperature = float(score_temperature)
        self.enable_style_head = bool(enable_style_head)
        self.style_head_confidence = float(style_head_confidence)
        self._style_head = None  # lazy loaded
        self._style_head_encoder = None  # the CLIPTextEncoder used for training

    def normalize_scores(self, scores: Sequence[float]) -> np.ndarray:
        scores_np = np.asarray(list(scores), dtype=np.float64)
        if scores_np.size == 0:
            return scores_np
        logits = scores_np * self.score_temperature
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= np.maximum(weights.sum(), 1e-12)
        return weights

    @staticmethod
    def _clamp_fracture_prior(prior: Dict[str, float]) -> Dict[str, float]:
        out = dict(prior)
        for key, (lo, hi) in FRACTURE_PRIOR_BOUNDS.items():
            if key in out:
                out[key] = _clamp(float(out[key]), lo, hi)
        return out

    @staticmethod
    def sentence_style_for_text(text: str) -> Dict[str, object]:
        q = str(text or "").lower()
        matches = []
        for rule in SENTENCE_STYLE_RULES:
            tokens = tuple(rule.get("tokens", ()))
            priority_tokens = tuple(rule.get("priority_tokens", ()))
            hits = sum(1 for token in tokens if token in q)
            priority_hits = sum(1 for token in priority_tokens if token in q)
            if hits > 0 or priority_hits > 0:
                matches.append((100 * priority_hits + hits, rule))
        if matches:
            return max(matches, key=lambda item: item[0])[1]
        return {"name": "material_default", "fracture_mult": {}, "runtime": {}}

    def _ensure_style_head(self, encoder=None) -> None:
        """Lazy-load (or train) the CLIP-conditioned style head once."""
        if not self.enable_style_head:
            return
        if self._style_head is not None:
            return
        if encoder is None:
            return
        try:
            from .style_head import get_or_train_style_head
            self._style_head = get_or_train_style_head(encoder, verbose=False)
            self._style_head_encoder = encoder
        except Exception as e:
            print(f"[MaterialPriorAdapter] style head disabled: {e}")
            self.enable_style_head = False
            self._style_head = None

    def _rule_lookup(self, name: str) -> Optional[Dict[str, object]]:
        for rule in SENTENCE_STYLE_RULES:
            if str(rule.get("name", "")) == name:
                return rule
        return None

    def predict_sentence_style(
        self,
        text: str,
        encoder=None,
    ) -> Dict[str, object]:
        """Pick sentence style using the learned head when confident,
        falling back to the keyword rule otherwise.

        Reviewer-defensible flow:
          1. Encode the sentence with CLIP.
          2. Run the learned head; if its top-class softmax probability
             clears ``style_head_confidence`` AND the predicted style has
             a corresponding rule in ``SENTENCE_STYLE_RULES``, use it.
          3. Otherwise fall back to the keyword-rule selector
             (``sentence_style_for_text``).

        The learned head is trained on weak supervision derived from the
        same rule set on a template corpus, so it generalizes to
        paraphrased prompts that the rule's discrete keywords cannot
        match while remaining honest about its supervision source.
        """
        if encoder is not None:
            self._ensure_style_head(encoder)
        if self._style_head is not None and encoder is not None:
            try:
                with torch.no_grad():
                    emb = encoder.encode_text(text).to(
                        next(self._style_head.parameters()).device
                    )
                indices, confidences, _ = self._style_head.predict(emb)
                if indices and confidences[0] >= self.style_head_confidence:
                    style_name = self._style_head.style_name(int(indices[0]))
                    rule = self._rule_lookup(style_name)
                    if rule is not None:
                        out = dict(rule)
                        out["_source"] = "head"
                        out["_head_confidence"] = float(confidences[0])
                        return out
            except Exception as e:
                print(f"[MaterialPriorAdapter] style head predict failed: {e}")
        out = dict(self.sentence_style_for_text(text))
        out.setdefault("_source", "rule")
        return out

    @staticmethod
    def _material_hint_query(text: str) -> str:
        q = str(text or "").lower()
        for marker in MATERIAL_CONTEXT_SPLITS:
            if marker in q:
                q = q.split(marker, 1)[0]
                break
        return q

    def material_hint_logits(
        self,
        text: str,
        topk_materials: Sequence[MaterialEntry],
    ) -> np.ndarray:
        """Declarative lexical prior for explicit object material words.

        CLIP remains the retriever.  This only adjusts the blend weights of the
        retrieved candidates, and it ignores environment phrases such as
        "dropped on concrete" so the floor material does not dominate the object.
        """
        entries = list(topk_materials)
        q = self._material_hint_query(text)
        logits = np.zeros(len(entries), dtype=np.float64)
        if not q:
            return logits

        for rule in MATERIAL_HINT_RULES:
            token_hits = sum(1 for token in rule.get("tokens", ()) if token in q)
            if token_hits <= 0:
                continue
            for idx, entry in enumerate(entries):
                name_l = entry.name.lower()
                category_l = entry.category.lower()
                bonus = float(rule.get("category_bonus", {}).get(category_l, 0.0))
                for pattern, value in rule.get("name_bonus", ()):
                    if pattern in name_l:
                        bonus += float(value)
                logits[idx] += min(bonus * (1.0 + 0.12 * (token_hits - 1)), 1.20)
        return logits

    # Tokens that identify materials whose physics should override any
    # sentence-style request for fracture.  When the family is
    # `neutral_reference` AND the prompt mentions one of these, we
    # zero out the crack-front and treat the run as no-fragment --
    # honoring the material physics over the sentence style.
    _METAL_NO_FRACTURE_TOKENS: tuple = (
        "steel", "metal", "iron", "titanium", "aluminum",
        "aluminium", "stainless", "alloy", "copper", "brass",
        "bronze", "carbon steel",
    )

    @staticmethod
    def _enforce_metal_no_fracture(
        runtime: Dict[str, object],
        family: str,
        text: str,
    ) -> Dict[str, object]:
        """For ``neutral_reference`` family + a metal-token prompt, force
        zero-fracture runtime so the material physics overrides any
        sentence-style request.

        Defends the paper claim that material physics is the upper
        bound for fracture: even if the sentence asks for radial
        cracks, a steel/metal prompt produces no fragments because the
        material does not brittle-fracture under the simulated impact.
        """
        if family != "neutral_reference":
            return runtime
        text_l = str(text or "").lower()
        if not any(tok in text_l for tok in MaterialPriorAdapter._METAL_NO_FRACTURE_TOKENS):
            return runtime
        out = dict(runtime)
        out["manifold.successor_topk"] = 0
        out["manifold.max_branching_tips"] = 0
        out["manifold.branch_drive_threshold"] = 0.99
        out["manifold.front_threshold"] = 0.999
        out["manifold.enable_front_propagation"] = False
        out["manifold.impact_seed_magnitude"] = 0.02
        out["manifold.fragment_damage_threshold"] = 0.95
        out["manifold.crack_connected_release_only"] = False
        out["manifold.open_crack_release_enable"] = False
        out["manifold.open_crack_release_max_patches"] = 0
        out["manifold.shard_enable"] = False
        out["manifold.shard_count_scale"] = 0.0
        out["manifold.debris_motion_gain"] = 0.0
        return out

    @staticmethod
    def _apply_family_runtime_caps(
        runtime: Dict[str, object],
        family: str,
    ) -> Dict[str, object]:
        """Re-clamp tip/branching keys to family bounds.

        ``fracture_prior_to_runtime_overrides`` already clamps these by
        family, but a sentence-style ``runtime.update()`` can subsequently
        push them outside the family envelope.  This helper re-applies the
        per-family bounds so the family is the actual cap, not an
        easily-overwritten suggestion.  Each rule mirrors the corresponding
        clamp inside ``fracture_prior_to_runtime_overrides``.
        """
        out = dict(runtime)
        family_name = str(family or "neutral_reference")

        def _get(key: str, default):
            val = out.get(key, default)
            try:
                return type(default)(val)
            except (TypeError, ValueError):
                return default

        if family_name == "sharp_brittle":
            out["manifold.successor_topk"] = 1
            out["manifold.max_branching_tips"] = min(_get("manifold.max_branching_tips", 8), 8)
            out["manifold.branch_score_ratio"] = max(
                _get("manifold.branch_score_ratio", 0.92), 0.92)
            out["manifold.branch_drive_threshold"] = max(
                _get("manifold.branch_drive_threshold", 0.62), 0.62)
        elif family_name == "brittle_moderate":
            out["manifold.successor_topk"] = min(_get("manifold.successor_topk", 2), 2)
            out["manifold.max_branching_tips"] = min(_get("manifold.max_branching_tips", 10), 10)
            out["manifold.branch_score_ratio"] = max(
                _get("manifold.branch_score_ratio", 0.88), 0.88)
            out["manifold.branch_drive_threshold"] = max(
                _get("manifold.branch_drive_threshold", 0.48), 0.48)
        elif family_name == "rough_quasi_brittle":
            out["manifold.successor_topk"] = max(_get("manifold.successor_topk", 2), 2)
            out["manifold.max_branching_tips"] = max(_get("manifold.max_branching_tips", 16), 16)
            out["manifold.branch_score_ratio"] = max(
                _get("manifold.branch_score_ratio", 0.72), 0.72)
            out["manifold.branch_drive_threshold"] = max(
                _get("manifold.branch_drive_threshold", 0.18), 0.18)
        elif family_name == "diffuse_damage":
            out["manifold.successor_topk"] = 0
            out["manifold.max_branching_tips"] = 0
            out["manifold.branch_score_ratio"] = 0.999
            out["manifold.branch_drive_threshold"] = 0.99
            out["manifold.front_threshold"] = 0.999
        else:
            out["manifold.successor_topk"] = min(
                max(_get("manifold.successor_topk", 1), 1), 2)
            out["manifold.max_branching_tips"] = min(
                _get("manifold.max_branching_tips", 12), 12)
        return out

    @staticmethod
    def _enforce_crack_connected_fragment_runtime(
        runtime: Dict[str, object],
        family: str,
    ) -> Dict[str, object]:
        """Make crack closure the only fragment birth path.

        CLIP and sentence style still choose material family, crack growth,
        branch density, closure thresholds, and post-fragment scatter.  What
        they cannot do is bypass propagation with arbitrary damage patches.
        Fragment labels are born only from crack-connected closure/ring logic.
        Family-level tip/branching caps are re-applied so style overrides
        cannot push them outside the family envelope.
        """
        out = dict(runtime)
        family_name = str(family or "neutral_reference")
        if family_name not in FRAGMENT_CAPABLE_FAMILIES:
            out["manifold.crack_connected_release_only"] = False
            out["manifold.open_crack_release_enable"] = False
            out["manifold.open_crack_release_max_patches"] = 0
            out["manifold.shard_enable"] = False
            out["manifold.shard_count_scale"] = 0.0
            out["manifold.debris_motion_gain"] = 0.0
            out = MaterialPriorAdapter._apply_family_runtime_caps(out, family_name)
            return out

        cap_by_family = {
            # All family caps removed (1.00 = no cap).  The released
            # ratio is now an EMERGENT property of the underlying
            # material physics (E, Gc, nu, rho via fracture toughness,
            # damage propagation thresholds, and support gates).  Brittle
            # materials with low Gc (glass) self-pulverize fully; tough
            # materials with high Gc (concrete) self-limit at lower
            # ratios because cracks stop propagating before the body is
            # fully released.  No style-level or family-level hand-tuned
            # release cap is imposed.
            "sharp_brittle": 1.00,
            "brittle_moderate": 1.00,
            "rough_quasi_brittle": 1.00,
            "neutral_reference": 1.00,
        }
        family_cap = cap_by_family.get(family_name, 0.35)
        requested_cap = float(
            out.get("manifold.strict_closure_max_released_ratio", family_cap)
        )
        strict_cap = min(max(requested_cap, 0.0), family_cap)
        out.update({
            "manifold.crack_connected_release_only": True,
            "manifold.strict_closure_max_released_ratio": strict_cap,
            "manifold.open_crack_release_enable": False,
            "manifold.open_crack_release_max_patches": 0,
            "manifold.open_crack_release_threshold": 1.0,
            "manifold.fragment_impulse_strength": 0.0,
            "manifold.fragment_impulse_boost_frames": 0,
            "manifold.debris_motion_gain": 0.0,
            "manifold.shard_enable": False,
            "manifold.shard_count_scale": 0.0,
            "manifold.splitting_enabled": False,
        })
        out = MaterialPriorAdapter._apply_family_runtime_caps(out, family_name)
        return out

    @staticmethod
    def _apply_material_scatter_scaling(
        runtime: Dict[str, object],
        physics_prior: Dict[str, float],
    ) -> Dict[str, object]:
        """Scale fragment-scatter knobs by a material brittleness factor.

        Style runtime dicts set BASELINE scatter magnitudes calibrated
        for a glass-tier brittleness reference (E=70 GPa, Gc=5 J/m^2).
        This pass multiplies the scatter knobs by

            brittleness = sqrt(E/Gc) / sqrt(E_glass / Gc_glass)

        clipped to ``[0, 1.5]``.  Glass keeps the baseline (factor 1.0);
        ceramic (Gc ~50) drops to ~0.32; concrete (E=30 GPa, Gc ~100)
        to ~0.15; rubber (diffuse_damage) goes to ~0.  This is what makes
        the same `radial_shatter` style produce explosive scatter on
        glass and chunky slow-moving shards on concrete -- the "scatter
        magnitude is an emergent material property" claim.

        Knobs scaled:
          - fragment_release_v_com_gain (Mode-I bond-opening kick magnitude)
          - voronoi_impact_shock_radius (impact shock-front radius;
            harder material = smaller shock zone)

        Knobs NOT scaled (their tuning is independent of brittleness):
          - fragment_physical_max_speed (an upper-bound clamp; bumped
            in the style profile if needed)
          - shape_match_*  (per-fragment rigidity)
        """
        out = dict(runtime)
        try:
            E = max(float(physics_prior.get("E", 0.0)), 1.0)
            Gc = max(float(physics_prior.get("Gc", 1.0)), 1e-3)
        except (TypeError, ValueError):
            return out
        E_ref, Gc_ref = 70.0e9, 5.0
        brittleness = float(np.sqrt(E / Gc) / np.sqrt(E_ref / Gc_ref))
        brittleness = max(0.0, min(brittleness, 1.5))
        scatter_keys = (
            "manifold.fragment_release_v_com_gain",
            "manifold.voronoi_impact_shock_radius",
        )
        for key in scatter_keys:
            if key in out:
                try:
                    out[key] = float(out[key]) * brittleness
                except (TypeError, ValueError):
                    continue
        out["manifold.material_brittleness"] = brittleness
        return out

    def apply_sentence_style(
        self,
        material_prior: Dict[str, object],
        text: str,
        encoder=None,
    ) -> Dict[str, object]:
        """Apply crack-shape wording on top of material CLIP retrieval.

        The CLIP DB is intentionally material-centric, so shape phrases such as
        "single smooth crack" or "spiderweb cracks" need a small semantic style
        adapter after material retrieval.

        When ``encoder`` (a ``CLIPTextEncoder``) is provided and the learned
        style head is enabled, the head's prediction is used for high-confidence
        sentences; otherwise the keyword-rule selector handles the lookup.
        """
        if encoder is not None and self.enable_style_head:
            style = self.predict_sentence_style(text, encoder=encoder)
        else:
            style = self.sentence_style_for_text(text)
        style_name = str(style.get("name", "material_default"))
        if style_name == "material_default":
            out = dict(material_prior)
            runtime = dict(material_prior.get("runtime", {}))
            family = str(material_prior.get("family", "neutral_reference"))
            runtime = self._enforce_crack_connected_fragment_runtime(runtime, family)
            runtime = self._enforce_metal_no_fracture(runtime, family, text)
            runtime = self._apply_material_scatter_scaling(
                runtime, dict(material_prior.get("physics", {})))
            runtime["manifold.sentence_style"] = style_name
            out["runtime"] = runtime
            out["sentence_style"] = style_name
            return out

        fracture = dict(material_prior["fracture"])
        for key, mult in dict(style.get("fracture_mult", {})).items():
            if key in fracture:
                fracture[key] = float(fracture[key]) * float(mult)
        fracture = self._clamp_fracture_prior(fracture)

        family = str(material_prior.get("family", "neutral_reference"))
        runtime = self.fracture_prior_to_runtime_overrides(fracture, family)
        runtime.update(dict(style.get("runtime", {})))
        runtime = self._enforce_crack_connected_fragment_runtime(runtime, family)
        runtime = self._enforce_metal_no_fracture(runtime, family, text)
        runtime = self._apply_material_scatter_scaling(
            runtime, dict(material_prior.get("physics", {})))
        runtime["manifold.sentence_style"] = style_name

        out = dict(material_prior)
        out["fracture"] = fracture
        out["runtime"] = runtime
        out["sentence_style"] = style_name
        return out

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
        text: str | None = None,
    ) -> Dict[str, object]:
        entries = list(topk_materials)
        scores_np = np.asarray(list(scores), dtype=np.float64)
        hint_logits = (
            self.material_hint_logits(text, entries)
            if text is not None
            else np.zeros(len(entries), dtype=np.float64)
        )
        adjusted_scores = scores_np + hint_logits
        weights = self.normalize_scores(adjusted_scores)
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
            "material_hint_logits": hint_logits.tolist(),
            "dominant_category": dominant_category,
            "top_k": [
                {
                    "name": entry.name,
                    "category": entry.category,
                    "score": float(adjusted_score),
                    "raw_score": float(raw_score),
                    "material_hint_logit": float(hint),
                    "weight": float(weight),
                    "family": self.family_for_entry(entry),
                }
                for entry, raw_score, adjusted_score, hint, weight
                in zip(entries, scores_np, adjusted_scores, hint_logits, weights)
            ],
        }

    @staticmethod
    def scale_physics_to_mpm(
        physics_prior: Dict[str, float],
        family: str | None = None,
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
        if family == "diffuse_damage":
            E = base_E * _clamp(e_ratio ** 0.18, 0.20, 0.62)
        else:
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
        runtime = MaterialPriorAdapter._enforce_crack_connected_fragment_runtime(
            runtime,
            family,
        )
        return runtime
