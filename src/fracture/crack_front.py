"""
Crack front tracking on the Gaussian manifold.

This module explicitly tracks active crack tips and advances them along a
surface-aware neighborhood graph. It replaces broad diffusion-style growth
with selective successor activation.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional

from src.utils.knn import pairwise_distances


class CrackFront:
    """Tip-based crack front tracker."""

    def __init__(
        self,
        seed_quantile: float = 0.995,
        max_seed_points: int = 2,
        min_seed_spacing: float = 0.04,
        successor_topk: int = 2,
        min_successor_score: float = 0.25,
        drive_weight: float = 0.40,
        distance_weight: float = 0.12,
        align_weight: float = 0.24,
        tangent_weight: float = 0.16,
        continuity_weight: float = 0.10,
        radial_weight: float = 0.16,
        lift_weight: float = 0.40,
        max_tip_age: int = 2,
        revisit_drive_threshold: float = 0.8,
        branch_score_ratio: float = 0.97,
        branch_drive_threshold: float = 0.70,
        max_branching_tips: int = 12,
        tau_init: float = 0.30,
        growth_gain: float = 1.00,
        branching_bias: float = 0.20,
        anisotropy_strength: float = 0.10,
        crack_style: str = "material_default",
        material_family: str = "neutral_reference",
        growth_griffith_threshold: float = 0.50,
        branch_direction_mode: str = "energy",
        branch_angle_prior_floor: float = 0.50,
        branch_event_topk: int = 2,
        device: str = "cuda",
    ):
        self.seed_quantile = seed_quantile
        self.max_seed_points = max_seed_points
        self.min_seed_spacing = min_seed_spacing
        self.successor_topk = successor_topk
        self.min_successor_score = min_successor_score
        self.drive_weight = drive_weight
        self.distance_weight = distance_weight
        self.align_weight = align_weight
        self.tangent_weight = tangent_weight
        self.continuity_weight = continuity_weight
        self.radial_weight = radial_weight
        self.lift_weight = lift_weight
        self.max_tip_age = max_tip_age
        self.revisit_drive_threshold = revisit_drive_threshold
        self.branch_score_ratio = branch_score_ratio
        self.branch_drive_threshold = branch_drive_threshold
        self.max_branching_tips = max_branching_tips
        self.tau_init = tau_init
        self.growth_gain = growth_gain
        self.branching_bias = branching_bias
        self.anisotropy_strength = anisotropy_strength
        self.crack_style = str(crack_style)
        self.material_family = str(material_family)
        # Griffith-style hard gate: candidate must clear g_th of normalized
        # drive before it is allowed to advance, regardless of aggregate score.
        self.growth_griffith_threshold = min(
            max(float(growth_griffith_threshold), 0.0), 1.0)
        # Branch direction selection: "energy" picks lateral candidate with
        # highest local drive (Karma-Lobkovsky-like), "angle" preserves the
        # legacy hash-noise angle target.  "hybrid" blends them.
        self.branch_direction_mode = str(branch_direction_mode).lower()
        # In "energy" / "hybrid" mode, the angle_score acts as a soft
        # modulator floored at this value (so a misaligned but high-energy
        # candidate still receives `floor` of the score).
        self.branch_angle_prior_floor = min(
            max(float(branch_angle_prior_floor), 0.0), 1.0)
        # Resolution-invariant branch gate: only the top-K branch_event
        # scores per tip are allowed to branch, regardless of how dense
        # the local kNN graph is.  Replaces the old absolute
        # `branch_event_threshold` with a relative ranking that scales
        # automatically from 10K to 100K particles.
        self.branch_event_topk = max(int(branch_event_topk), 1)
        self.device = torch.device(device)

        self.tip_mask: Optional[Tensor] = None
        self.visited_mask: Optional[Tensor] = None
        self.parent_index: Optional[Tensor] = None
        self.tip_age: Optional[Tensor] = None
        self.branch_age: Optional[Tensor] = None
        self.activation_step: Optional[Tensor] = None
        self.path_length: Optional[Tensor] = None
        self.growth_dir: Optional[Tensor] = None
        self._advance_step: int = 0
        self.event_log: list[dict] = []
        self.last_step_events: list[dict] = []
        self._event_log_limit: int = 200000

    def _family_settings(self) -> dict:
        if self.material_family == "sharp_brittle":
            settings = {
                "front_enabled": True,
                "branch_scale": 0.35,
                "continuity_scale": 1.35,
                "align_scale": 1.20,
                "revisit_penalty": 0.95,
                "successor_cap": 1,
                "initial_seed_cap": 2,
                "seed_spacing_scale": 1.25,
                "lateral_branch_bonus": 0.00,
                "lateral_branch_threshold": 1.00,
                "branch_persist_steps": 0,
                "branch_persist_lateral": 1.00,
                "branch_extra_branches": 0,
                "closure_weight": 0.06,
                "closure_branch_threshold": 0.72,
                "closure_branch_bonus": 0.08,
                "closure_extra_branches": 0,
                "closure_target_cap": 48,
                "closure_min_dist_scale": 0.08,
                "closure_max_dist_scale": 0.22,
                "closure_height_scale": 0.10,
                "closure_score_weight": 0.04,
                "branch_event_bonus": 0.10,
                "branch_event_threshold": 0.34,
                "branch_event_min_depth": 3,
                "branch_angle_center": 0.56,
                "branch_angle_width": 0.34,
                "branch_angle_jitter": 0.08,
                "branch_guidance_steps": 3,
                "branch_guidance_weight": 0.32,
                "branch_turn_penalty": 0.18,
                "branch_forward_min": 0.10,
                "branch_hard_turn_penalty": 0.45,
                "branch_target_align_threshold": 0.34,
                "front_forward_min": -0.02,
                "front_turn_penalty": 0.12,
                "branch_hint_scale": 0.35,
                "branch_radial_scale": 0.55,
                "branch_lift_scale": 0.70,
                "closure_min_depth": 8,
                "closure_branch_age_max": 0,
                "max_tip_path_scale": 0.80,
                "duplicate_penalty": 0.42,
                "duplicate_parallel_threshold": 0.72,
                "revisit_drive_override": 0.95,
            }
            if self.crack_style == "radial_shatter":
                settings.update({
                    "branch_scale": 1.22,
                    "continuity_scale": 0.86,
                    "align_scale": 0.98,
                    "revisit_penalty": 0.62,
                    "successor_cap": max(self.successor_topk, 6),
                    "initial_seed_cap": 10,
                    "seed_spacing_scale": 0.36,
                    "lateral_branch_bonus": 0.30,
                    "lateral_branch_threshold": 0.14,
                    "branch_persist_steps": 1,
                    "branch_persist_lateral": 0.22,
                    "branch_extra_branches": 3,
                    "closure_weight": 0.05,
                    "closure_branch_threshold": 0.44,
                    "closure_branch_bonus": 0.08,
                    "closure_extra_branches": 1,
                    "closure_target_cap": 128,
                    "closure_max_dist_scale": 0.24,
                    "closure_height_scale": 0.16,
                    "closure_score_weight": 0.08,
                    "branch_event_bonus": 0.58,
                    "branch_event_threshold": 0.10,
                    "branch_event_min_depth": 1,
                    "branch_angle_center": 0.52,
                    "branch_angle_width": 0.48,
                    "branch_angle_jitter": 0.26,
                    "branch_guidance_steps": 18,
                    "branch_guidance_weight": 0.58,
                    "branch_turn_penalty": 0.34,
                    "branch_forward_min": 0.16,
                    "branch_hard_turn_penalty": 0.82,
                    "branch_target_align_threshold": 0.20,
                    "front_forward_min": 0.02,
                    "front_turn_penalty": 0.34,
                    "branch_hint_scale": 0.12,
                    "branch_radial_scale": 0.35,
                    "branch_lift_scale": 0.55,
                    "closure_min_depth": 6,
                    "max_tip_path_scale": 0.72,
                    "duplicate_penalty": 0.30,
                    "duplicate_parallel_threshold": 0.78,
                    "revisit_drive_override": 0.74,
                })
            elif self.crack_style == "spiderweb_branching":
                settings.update({
                    "branch_scale": 0.95,
                    "continuity_scale": 0.98,
                    "align_scale": 1.00,
                    "revisit_penalty": 0.70,
                    "successor_cap": max(self.successor_topk, 5),
                    "initial_seed_cap": 4,
                    "seed_spacing_scale": 0.50,
                    "lateral_branch_bonus": 0.34,
                    "lateral_branch_threshold": 0.16,
                    "branch_persist_steps": 1,
                    "branch_persist_lateral": 0.24,
                    "branch_extra_branches": 2,
                    "closure_weight": 0.04,
                    "closure_branch_threshold": 0.48,
                    "closure_branch_bonus": 0.08,
                    "closure_extra_branches": 1,
                    "closure_target_cap": 96,
                    "closure_max_dist_scale": 0.22,
                    "closure_height_scale": 0.16,
                    "closure_score_weight": 0.04,
                    "branch_event_bonus": 0.54,
                    "branch_event_threshold": 0.10,
                    "branch_event_min_depth": 1,
                    "branch_angle_center": 0.44,
                    "branch_angle_width": 0.44,
                    "branch_angle_jitter": 0.26,
                    "branch_guidance_steps": 32,
                    "branch_guidance_weight": 0.64,
                    "branch_turn_penalty": 0.38,
                    "branch_forward_min": 0.26,
                    "branch_hard_turn_penalty": 1.15,
                    "branch_target_align_threshold": 0.22,
                    "front_forward_min": 0.06,
                    "front_turn_penalty": 0.52,
                    "branch_hint_scale": 0.10,
                    "branch_radial_scale": 0.30,
                    "branch_lift_scale": 0.52,
                    "closure_min_depth": 7,
                    "max_tip_path_scale": 0.54,
                    "duplicate_penalty": 0.46,
                    "duplicate_parallel_threshold": 0.72,
                    "revisit_drive_override": 0.82,
                })
            elif self.crack_style == "single_smooth":
                settings.update({
                    "branch_scale": 0.08,
                    "continuity_scale": 1.55,
                    "align_scale": 1.35,
                    "revisit_penalty": 1.0,
                    "successor_cap": 1,
                    "initial_seed_cap": 1,
                    "seed_spacing_scale": 1.80,
                    "lateral_branch_bonus": 0.0,
                    "lateral_branch_threshold": 1.0,
                    "branch_persist_steps": 0,
                    "branch_extra_branches": 0,
                    "closure_weight": 0.0,
                    "closure_extra_branches": 0,
                    "closure_score_weight": 0.0,
                    "branch_event_bonus": 0.0,
                    "branch_event_threshold": 1.0,
                    "branch_event_min_depth": 999,
                    "branch_angle_jitter": 0.0,
                    "branch_guidance_steps": 0,
                    "branch_guidance_weight": 0.0,
                    "branch_turn_penalty": 0.0,
                    "branch_forward_min": -1.0,
                    "branch_hard_turn_penalty": 0.0,
                    "branch_target_align_threshold": 1.0,
                    "front_forward_min": -1.0,
                    "front_turn_penalty": 0.0,
                    "branch_hint_scale": 1.0,
                    "branch_radial_scale": 1.0,
                    "branch_lift_scale": 1.0,
                    "closure_min_depth": 999,
                    "max_tip_path_scale": 1.20,
                    "duplicate_penalty": 0.76,
                    "duplicate_parallel_threshold": 0.62,
                    "revisit_drive_override": 1.01,
                })
            return settings
        if self.material_family == "brittle_moderate":
            settings = {
                "front_enabled": True,
                "branch_scale": 0.70,
                "continuity_scale": 1.08,
                "align_scale": 1.00,
                "revisit_penalty": 0.85,
                "successor_cap": 3,
                "initial_seed_cap": 3,
                "seed_spacing_scale": 1.05,
                "lateral_branch_bonus": 0.12,
                "lateral_branch_threshold": 0.32,
                "branch_persist_steps": 0,
                "branch_persist_lateral": 0.35,
                "branch_extra_branches": 1,
                "closure_weight": 0.18,
                "closure_branch_threshold": 0.36,
                "closure_branch_bonus": 0.16,
                "closure_extra_branches": 1,
                "closure_target_cap": 64,
                "closure_min_dist_scale": 0.08,
                "closure_max_dist_scale": 0.28,
                "closure_height_scale": 0.14,
                "closure_score_weight": 0.16,
                "branch_event_bonus": 0.20,
                "branch_event_threshold": 0.28,
                "branch_event_min_depth": 2,
                "branch_angle_center": 0.54,
                "branch_angle_width": 0.38,
                "branch_angle_jitter": 0.10,
                "branch_guidance_steps": 4,
                "branch_guidance_weight": 0.38,
                "branch_turn_penalty": 0.22,
                "branch_forward_min": 0.14,
                "branch_hard_turn_penalty": 0.65,
                "branch_target_align_threshold": 0.36,
                "front_forward_min": 0.02,
                "front_turn_penalty": 0.22,
                "branch_hint_scale": 0.25,
                "branch_radial_scale": 0.55,
                "branch_lift_scale": 0.70,
                "closure_min_depth": 10,
                "closure_branch_age_max": 0,
                "max_tip_path_scale": 0.74,
                "duplicate_penalty": 0.32,
                "duplicate_parallel_threshold": 0.72,
                "revisit_drive_override": 0.90,
            }
            if self.crack_style == "spiderweb_branching":
                settings.update({
                    "branch_scale": 1.16,
                    "continuity_scale": 0.88,
                    "align_scale": 0.94,
                    "revisit_penalty": 0.58,
                    "successor_cap": max(self.successor_topk, 4),
                    "initial_seed_cap": 5,
                    "seed_spacing_scale": 0.58,
                    "lateral_branch_bonus": 0.28,
                    "lateral_branch_threshold": 0.18,
                    "branch_persist_steps": 1,
                    "branch_persist_lateral": 0.24,
                    "branch_extra_branches": 2,
                    "closure_weight": 0.30,
                    "closure_branch_threshold": 0.28,
                    "closure_branch_bonus": 0.24,
                    "closure_extra_branches": 2,
                    "closure_target_cap": 112,
                    "closure_max_dist_scale": 0.36,
                    "closure_height_scale": 0.20,
                    "closure_score_weight": 0.28,
                    "branch_event_bonus": 0.40,
                    "branch_event_threshold": 0.16,
                    "branch_event_min_depth": 1,
                    "branch_angle_center": 0.48,
                    "branch_angle_width": 0.46,
                    "branch_angle_jitter": 0.22,
                    "branch_guidance_steps": 16,
                    "branch_guidance_weight": 0.56,
                    "branch_turn_penalty": 0.32,
                    "branch_forward_min": 0.10,
                    "branch_hard_turn_penalty": 0.86,
                    "branch_target_align_threshold": 0.22,
                    "front_forward_min": -0.01,
                    "front_turn_penalty": 0.18,
                    "branch_hint_scale": 0.14,
                    "branch_radial_scale": 0.42,
                    "branch_lift_scale": 0.58,
                    "closure_min_depth": 6,
                    "max_tip_path_scale": 0.62,
                    "duplicate_penalty": 0.20,
                    "duplicate_parallel_threshold": 0.70,
                    "revisit_drive_override": 0.76,
                })
            elif self.crack_style == "radial_shatter":
                settings.update({
                    "branch_scale": 1.06,
                    "continuity_scale": 0.92,
                    "revisit_penalty": 0.64,
                    "successor_cap": max(self.successor_topk, 4),
                    "initial_seed_cap": 6,
                    "seed_spacing_scale": 0.52,
                    "lateral_branch_bonus": 0.24,
                    "lateral_branch_threshold": 0.20,
                    "branch_persist_steps": 1,
                    "branch_extra_branches": 2,
                    "closure_weight": 0.26,
                    "closure_branch_threshold": 0.30,
                    "closure_branch_bonus": 0.20,
                    "closure_extra_branches": 1,
                    "closure_target_cap": 96,
                    "closure_max_dist_scale": 0.32,
                    "closure_height_scale": 0.18,
                    "closure_score_weight": 0.22,
                    "branch_event_bonus": 0.34,
                    "branch_event_threshold": 0.18,
                    "branch_event_min_depth": 1,
                    "branch_angle_jitter": 0.18,
                    "branch_guidance_steps": 12,
                    "branch_guidance_weight": 0.50,
                    "max_tip_path_scale": 0.66,
                    "duplicate_penalty": 0.24,
                    "revisit_drive_override": 0.80,
                })
            elif self.crack_style == "single_smooth":
                settings.update({
                    "branch_scale": 0.10,
                    "continuity_scale": 1.45,
                    "align_scale": 1.22,
                    "revisit_penalty": 1.0,
                    "successor_cap": 1,
                    "initial_seed_cap": 1,
                    "seed_spacing_scale": 1.55,
                    "lateral_branch_bonus": 0.0,
                    "lateral_branch_threshold": 1.0,
                    "branch_extra_branches": 0,
                    "closure_weight": 0.04,
                    "closure_extra_branches": 0,
                    "closure_score_weight": 0.02,
                    "branch_event_bonus": 0.0,
                    "branch_event_threshold": 1.0,
                    "branch_event_min_depth": 999,
                    "branch_angle_jitter": 0.0,
                    "branch_guidance_steps": 0,
                    "max_tip_path_scale": 1.0,
                    "duplicate_penalty": 0.50,
                    "revisit_drive_override": 0.98,
                })
            return settings
        if self.material_family == "rough_quasi_brittle":
            return {
                "front_enabled": True,
                "branch_scale": 1.85,
                "continuity_scale": 0.70,
                "align_scale": 0.80,
                "revisit_penalty": 0.22,
                "successor_cap": max(self.successor_topk, 5),
                "initial_seed_cap": max(self.max_seed_points, 6),
                "seed_spacing_scale": 0.85,
                "lateral_branch_bonus": 0.26,
                "lateral_branch_threshold": 0.18,
                "branch_persist_steps": 1,
                "branch_persist_lateral": 0.22,
                "branch_extra_branches": 2,
                "closure_weight": 0.18,
                "closure_branch_threshold": 0.34,
                "closure_branch_bonus": 0.16,
                "closure_extra_branches": 1,
                "closure_target_cap": 96,
                "closure_min_dist_scale": 0.10,
                "closure_max_dist_scale": 0.40,
                "closure_height_scale": 0.18,
                "closure_score_weight": 0.20,
                "branch_event_bonus": 0.28,
                "branch_event_threshold": 0.22,
                "branch_event_min_depth": 2,
                "branch_angle_center": 0.46,
                "branch_angle_width": 0.46,
                "branch_angle_jitter": 0.22,
                "branch_guidance_steps": 16,
                "branch_guidance_weight": 0.48,
                "branch_turn_penalty": 0.28,
                "branch_forward_min": 0.18,
                "branch_hard_turn_penalty": 0.85,
                "branch_target_align_threshold": 0.34,
                "front_forward_min": -0.02,
                "front_turn_penalty": 0.16,
                "branch_hint_scale": 0.18,
                "branch_radial_scale": 0.45,
                "branch_lift_scale": 0.62,
                "closure_min_depth": 10,
                "closure_branch_age_max": 0,
                "max_tip_path_scale": 0.58,
                "duplicate_penalty": 0.24,
                "duplicate_parallel_threshold": 0.68,
                "revisit_drive_override": 0.72,
            }
        if self.material_family == "diffuse_damage":
            return {
                "front_enabled": False,
                "branch_scale": 0.0,
                "continuity_scale": 0.0,
                "align_scale": 0.0,
                "revisit_penalty": 1.0,
                "successor_cap": 0,
                "initial_seed_cap": 0,
                "seed_spacing_scale": 1.50,
                "lateral_branch_bonus": 0.0,
                "lateral_branch_threshold": 1.0,
                "branch_persist_steps": 0,
                "branch_persist_lateral": 1.0,
                "branch_extra_branches": 0,
                "closure_weight": 0.0,
                "closure_branch_threshold": 1.0,
                "closure_branch_bonus": 0.0,
                "closure_extra_branches": 0,
                "closure_target_cap": 0,
                "closure_min_dist_scale": 0.10,
                "closure_max_dist_scale": 0.20,
                "closure_height_scale": 0.10,
                "closure_score_weight": 0.0,
                "branch_event_bonus": 0.0,
                "branch_event_threshold": 1.0,
                "branch_event_min_depth": 999,
                "branch_angle_center": 0.5,
                "branch_angle_width": 0.5,
                "branch_angle_jitter": 0.0,
                "branch_guidance_steps": 0,
                "branch_guidance_weight": 0.0,
                "branch_turn_penalty": 0.0,
                "branch_forward_min": -1.0,
                "branch_hard_turn_penalty": 0.0,
                "branch_target_align_threshold": 1.0,
                "front_forward_min": -1.0,
                "front_turn_penalty": 0.0,
                "branch_hint_scale": 1.0,
                "branch_radial_scale": 1.0,
                "branch_lift_scale": 1.0,
                "closure_min_depth": 999,
                "closure_branch_age_max": 0,
                "max_tip_path_scale": 0.0,
                "duplicate_penalty": 0.0,
                "duplicate_parallel_threshold": 1.0,
                "revisit_drive_override": 1.01,
            }
        return {
            "front_enabled": True,
            "branch_scale": 1.0,
            "continuity_scale": 1.0,
            "align_scale": 1.0,
            "revisit_penalty": 0.75,
            "successor_cap": max(1, min(self.successor_topk, 2)),
            "initial_seed_cap": 2,
            "seed_spacing_scale": 1.0,
            "lateral_branch_bonus": 0.10,
            "lateral_branch_threshold": 0.28,
            "branch_persist_steps": 0,
            "branch_persist_lateral": 0.35,
            "branch_extra_branches": 1,
            "closure_weight": 0.12,
            "closure_branch_threshold": 0.40,
            "closure_branch_bonus": 0.12,
            "closure_extra_branches": 1,
            "closure_target_cap": 64,
            "closure_min_dist_scale": 0.08,
            "closure_max_dist_scale": 0.30,
            "closure_height_scale": 0.14,
            "closure_score_weight": 0.10,
            "branch_event_bonus": 0.14,
            "branch_event_threshold": 0.30,
            "branch_event_min_depth": 2,
            "branch_angle_center": 0.54,
            "branch_angle_width": 0.38,
            "branch_angle_jitter": 0.10,
            "branch_guidance_steps": 4,
            "branch_guidance_weight": 0.38,
            "branch_turn_penalty": 0.22,
            "branch_forward_min": 0.14,
            "branch_hard_turn_penalty": 0.65,
            "branch_target_align_threshold": 0.36,
            "front_forward_min": 0.02,
            "front_turn_penalty": 0.22,
            "branch_hint_scale": 0.25,
            "branch_radial_scale": 0.55,
            "branch_lift_scale": 0.70,
            "closure_min_depth": 10,
            "closure_branch_age_max": 0,
            "max_tip_path_scale": 0.74,
            "duplicate_penalty": 0.30,
            "duplicate_parallel_threshold": 0.72,
            "revisit_drive_override": 0.90,
        }

    def _duplicate_candidate_score(
        self,
        graph,
        candidate_idx: Tensor,
        candidate_edge: Tensor,
        tip_index: int,
        family_cfg: dict,
        closure_score: Tensor,
    ) -> Tensor:
        """Local crack occupancy penalty for parallel duplicate fronts."""
        if (
            candidate_idx.numel() == 0
            or self.visited_mask is None
            or self.growth_dir is None
            or getattr(graph, "knn_idx", None) is None
        ):
            return torch.zeros(candidate_idx.shape[0], device=candidate_idx.device, dtype=candidate_edge.dtype)

        near_idx = graph.knn_idx[candidate_idx]
        near_visited = self.visited_mask[near_idx]
        near_visited = near_visited & (near_idx != int(tip_index))
        if self.parent_index is not None:
            parent = int(self.parent_index[int(tip_index)].item())
            if parent >= 0:
                near_visited = near_visited & (near_idx != parent)
        if not bool(near_visited.any()):
            return torch.zeros(candidate_idx.shape[0], device=candidate_idx.device, dtype=candidate_edge.dtype)

        near_dir = self.growth_dir[near_idx]
        near_norm = near_dir.norm(dim=2, keepdim=True)
        near_dir = torch.where(
            near_norm > 1e-8,
            near_dir / near_norm.clamp(min=1e-8),
            torch.zeros_like(near_dir),
        )
        edge = candidate_edge / candidate_edge.norm(dim=1, keepdim=True).clamp(min=1e-8)
        parallel = (near_dir * edge.unsqueeze(1)).sum(dim=2).abs()
        parallel_threshold = float(family_cfg.get("duplicate_parallel_threshold", 0.72))
        parallel_mask = near_visited & (parallel >= parallel_threshold)
        parallel_score = (parallel * parallel_mask.float()).max(dim=1).values

        density_score = near_visited.float().mean(dim=1).clamp(0.0, 1.0)
        duplicate = torch.maximum(parallel_score, 0.35 * density_score)

        second_idx = graph.knn_idx[near_idx]
        second_visited = self.visited_mask[second_idx]
        second_visited = second_visited & (second_idx != int(tip_index))
        if self.parent_index is not None:
            parent = int(self.parent_index[int(tip_index)].item())
            if parent >= 0:
                second_visited = second_visited & (second_idx != parent)
        if bool(second_visited.any()):
            second_dir = self.growth_dir[second_idx]
            second_norm = second_dir.norm(dim=3, keepdim=True)
            second_dir = torch.where(
                second_norm > 1e-8,
                second_dir / second_norm.clamp(min=1e-8),
                torch.zeros_like(second_dir),
            )
            second_parallel = (second_dir * edge[:, None, None, :]).sum(dim=3).abs()
            second_parallel_mask = second_visited & (second_parallel >= parallel_threshold)
            second_score = (second_parallel * second_parallel_mask.float()).amax(dim=(1, 2))
            duplicate = torch.maximum(duplicate, 0.62 * second_score)

        closure_relief = (1.0 - 0.85 * closure_score.clamp(0.0, 1.0)).clamp(0.15, 1.0)
        return (duplicate * closure_relief).clamp(0.0, 1.0)

    def _lineage_depth(self, index: int, max_depth: int = 64) -> int:
        if self.parent_index is None:
            return 0
        depth = 0
        seen = set()
        parent = int(self.parent_index[int(index)].item())
        while parent >= 0 and parent not in seen and depth < max_depth:
            seen.add(parent)
            depth += 1
            parent = int(self.parent_index[parent].item())
        return depth

    def _stable_signed_noise(self, tip_index: int, candidate_idx: Tensor, salt: int = 0) -> Tensor:
        """Deterministic per-tip/candidate noise in [-1, 1] for branch angle jitter."""
        if candidate_idx.numel() == 0:
            return torch.zeros(candidate_idx.shape, device=candidate_idx.device, dtype=torch.float32)
        x = candidate_idx.to(dtype=torch.float32)
        x = x * 12.9898 + float(tip_index) * 78.233 + float(self._advance_step + salt) * 37.719
        return (torch.frac(torch.sin(x) * 43758.5453) * 2.0 - 1.0).clamp(-1.0, 1.0)

    def _branch_target_direction(
        self,
        tip_dir: Tensor,
        normal: Optional[Tensor],
        tip_index: int,
        family_cfg: dict,
    ) -> Tensor:
        """Pick one persistent random side direction for a new branch tip."""
        if tip_dir.norm() <= 1e-8:
            return tip_dir
        base = tip_dir / tip_dir.norm().clamp(min=1e-8)
        if normal is not None and normal.norm() > 1e-8:
            n = normal / normal.norm().clamp(min=1e-8)
            side = torch.cross(n, base, dim=0)
        else:
            ref = torch.zeros_like(base)
            ref[2] = 1.0
            if torch.abs(base @ ref) > 0.88:
                ref = torch.zeros_like(base)
                ref[0] = 1.0
            side = torch.cross(ref, base, dim=0)
        if side.norm() <= 1e-8:
            ref = torch.zeros_like(base)
            ref[1] = 1.0
            side = torch.cross(ref, base, dim=0)
        side = side / side.norm().clamp(min=1e-8)

        idx = torch.tensor([int(tip_index)], device=tip_dir.device, dtype=torch.long)
        sign_noise = self._stable_signed_noise(tip_index, idx, salt=211).to(
            device=tip_dir.device,
            dtype=tip_dir.dtype,
        )[0]
        angle_noise = self._stable_signed_noise(tip_index, idx, salt=409).to(
            device=tip_dir.device,
            dtype=tip_dir.dtype,
        )[0]
        sign = torch.where(
            sign_noise >= 0,
            torch.ones((), device=tip_dir.device, dtype=tip_dir.dtype),
            -torch.ones((), device=tip_dir.device, dtype=tip_dir.dtype),
        )
        angle_center = float(family_cfg.get("branch_angle_center", 0.54))
        angle_jitter = max(float(family_cfg.get("branch_angle_jitter", 0.0)), 0.0)
        angle_cos = (angle_center + angle_jitter * angle_noise).clamp(0.05, 0.95)
        angle_sin = torch.sqrt((1.0 - angle_cos * angle_cos).clamp(min=0.0))
        target = angle_cos * base + sign * angle_sin * side
        if normal is not None and normal.norm() > 1e-8:
            n = normal / normal.norm().clamp(min=1e-8)
            target = target - (target @ n) * n
        return target / target.norm().clamp(min=1e-8)

    def _compute_loop_closure_scores(
        self,
        tip_index: int,
        neighbor_idx: Tensor,
        edge_dir: Tensor,
        positions: Tensor,
        tip_dir: Tensor,
        family_cfg: dict,
        diag: float,
        height_scale: float,
    ) -> Tensor:
        closure_weight = float(family_cfg.get("closure_weight", 0.0))
        if closure_weight <= 0.0 or self.visited_mask is None:
            return torch.zeros(edge_dir.shape[0], device=positions.device, dtype=positions.dtype)

        target_mask = self.visited_mask.clone()
        target_mask[tip_index] = False
        parent = int(self.parent_index[tip_index].item()) if self.parent_index is not None else -1
        if parent >= 0:
            target_mask[parent] = False

        target_idx = torch.where(target_mask)[0]
        if target_idx.numel() == 0:
            return torch.zeros(edge_dir.shape[0], device=positions.device, dtype=positions.dtype)

        to_targets = positions[target_idx] - positions[tip_index].unsqueeze(0)
        dist_targets = to_targets.norm(dim=1)
        min_dist = max(0.02, family_cfg["closure_min_dist_scale"] * max(diag, 1e-6))
        max_dist = max(min_dist * 1.5, family_cfg["closure_max_dist_scale"] * max(diag, 1e-6))
        height_gate = max(0.02, family_cfg["closure_height_scale"] * max(height_scale, 1e-6))
        valid_targets = (
            (dist_targets >= min_dist)
            & (dist_targets <= max_dist)
            & (to_targets[:, 2].abs() <= height_gate)
        )
        if tip_dir.norm() > 1e-8:
            tip_dir = tip_dir / tip_dir.norm().clamp(min=1e-8)
            target_dir_all = to_targets / dist_targets.unsqueeze(1).clamp(min=1e-8)
            # Favor lateral / returning targets over purely forward continuation.
            valid_targets &= ((target_dir_all @ tip_dir).abs() <= 0.96)
        if not bool(valid_targets.any()):
            return torch.zeros(edge_dir.shape[0], device=positions.device, dtype=positions.dtype)

        target_idx = target_idx[valid_targets]
        dist_targets = dist_targets[valid_targets]
        to_targets = to_targets[valid_targets]
        if target_idx.numel() > int(family_cfg["closure_target_cap"]):
            order = dist_targets.argsort()
            target_idx = target_idx[order[: int(family_cfg["closure_target_cap"])]]
            dist_targets = dist_targets[order[: int(family_cfg["closure_target_cap"])]]
            to_targets = to_targets[order[: int(family_cfg["closure_target_cap"])]]

        target_dir = to_targets / dist_targets.unsqueeze(1).clamp(min=1e-8)
        neighbor_pos = positions[neighbor_idx]
        dist_next = pairwise_distances(neighbor_pos, positions[target_idx])
        progress = ((dist_targets.unsqueeze(0) - dist_next) / dist_targets.unsqueeze(0).clamp(min=1e-8)).clamp(0.0, 1.0)
        align = (edge_dir @ target_dir.T).clamp(min=0.0, max=1.0)

        if tip_dir.norm() > 1e-8:
            target_lateral = (1.0 - (target_dir @ tip_dir).abs()).clamp(0.0, 1.0)
        else:
            target_lateral = torch.ones(target_dir.shape[0], device=positions.device, dtype=positions.dtype)
        closure_pair = progress * (0.55 + 0.45 * align) * (0.45 + 0.55 * target_lateral.unsqueeze(0))
        return closure_pair.max(dim=1).values.clamp(0.0, 1.0)

    def initialize(self, N: int, device: Optional[torch.device] = None) -> None:
        device = device or self.device
        self.tip_mask = torch.zeros(N, dtype=torch.bool, device=device)
        self.visited_mask = torch.zeros(N, dtype=torch.bool, device=device)
        self.parent_index = torch.full((N,), -1, dtype=torch.long, device=device)
        self.tip_age = torch.zeros(N, dtype=torch.long, device=device)
        self.branch_age = torch.zeros(N, dtype=torch.long, device=device)
        self.activation_step = torch.full((N,), -1, dtype=torch.long, device=device)
        self.path_length = torch.zeros(N, device=device)
        self.growth_dir = torch.zeros(N, 3, device=device)
        self._advance_step = 0
        self.event_log = []
        self.last_step_events = []

    @staticmethod
    def _vec_to_list(value: Tensor) -> list[float]:
        return [float(v) for v in value.detach().float().cpu().tolist()]

    def _record_tip_event(
        self,
        kind: str,
        frame: int,
        parent: int,
        child: int,
        positions: Tensor,
        direction: Tensor,
        *,
        score: float = 0.0,
        drive: float = 0.0,
        closure_score: float = 0.0,
        branch_score: float = 0.0,
        branch_angle_deg: float = 0.0,
        branch_angle_target: float = 0.0,
        lineage_depth: int = 0,
    ) -> None:
        parent_pos = (
            self._vec_to_list(positions[parent])
            if parent >= 0 and parent < positions.shape[0]
            else [0.0, 0.0, 0.0]
        )
        event = {
            "kind": str(kind),
            "frame": int(frame),
            "parent": int(parent),
            "child": int(child),
            "parent_pos": parent_pos,
            "child_pos": self._vec_to_list(positions[child]),
            "direction": self._vec_to_list(direction),
            "score": float(score),
            "drive": float(drive),
            "closure_score": float(closure_score),
            "branch_score": float(branch_score),
            "branch_angle_deg": float(branch_angle_deg),
            "branch_angle_target_cos": float(branch_angle_target),
            "lineage_depth": int(lineage_depth),
        }
        self.last_step_events.append(event)
        self.event_log.append(event)
        if len(self.event_log) > self._event_log_limit:
            del self.event_log[: len(self.event_log) - self._event_log_limit]

    def has_active_tips(self) -> bool:
        return self.tip_mask is not None and bool(self.tip_mask.any())

    def seed_from_scores(
        self,
        init_score: Tensor,
        positions: Tensor,
        growth_dir_hint: Tensor,
        impact_center: Optional[Tensor] = None,
    ) -> int:
        """Create initial crack tips from sparse hotspot scores."""
        family_cfg = self._family_settings()
        if not family_cfg["front_enabled"]:
            return 0
        if self.tip_mask is None or self.tip_mask.shape[0] != positions.shape[0]:
            self.initialize(positions.shape[0], positions.device)
        if self.has_active_tips():
            return 0

        score = init_score.clamp(min=0.0)
        if score.max() <= 0:
            return 0

        q = float(min(max(self.seed_quantile, 0.0), 0.999))
        thresh = torch.quantile(score.detach(), q)
        thresh = torch.maximum(
            thresh,
            torch.tensor(self.tau_init, device=score.device, dtype=score.dtype),
        )
        candidate_idx = torch.where(score >= thresh)[0]
        if candidate_idx.numel() == 0 and score.max() < max(0.75 * self.tau_init, 0.12):
            return 0
        if candidate_idx.numel() == 0:
            candidate_idx = score.topk(min(self.max_seed_points, score.numel())).indices
        elif candidate_idx.numel() < min(self.max_seed_points, score.numel()):
            top_idx = score.topk(min(self.max_seed_points, score.numel())).indices
            candidate_idx = torch.unique(torch.cat([candidate_idx, top_idx], dim=0))

        order = score[candidate_idx].argsort(descending=True)
        candidate_idx = candidate_idx[order]

        selected = []
        min_seed_spacing = self.min_seed_spacing * family_cfg["seed_spacing_scale"]
        max_seed_points = min(
            self.max_seed_points,
            max(0, int(family_cfg.get("initial_seed_cap", family_cfg["successor_cap"] + 1))),
        )
        for idx in candidate_idx.tolist():
            if len(selected) >= max_seed_points:
                break
            pos_i = positions[idx]
            if selected:
                sel_pos = positions[torch.tensor(selected, device=positions.device)]
                min_dist = torch.norm(sel_pos - pos_i.unsqueeze(0), dim=1).min().item()
                if min_dist < min_seed_spacing:
                    continue
            selected.append(idx)

        if not selected:
            selected = [int(score.argmax().item())]

        sel_t = torch.tensor(selected, dtype=torch.long, device=positions.device)
        self.tip_mask[sel_t] = True
        self.visited_mask[sel_t] = True
        self.tip_age[sel_t] = 0
        if self.path_length is not None:
            self.path_length[sel_t] = 0.0
        if self.activation_step is not None:
            self.activation_step[sel_t] = 0

        seed_dir = growth_dir_hint[sel_t]
        seed_dir_norm = seed_dir.norm(dim=1, keepdim=True)
        if impact_center is not None:
            radial = positions[sel_t] - impact_center.unsqueeze(0)
            radial = radial / radial.norm(dim=1, keepdim=True).clamp(min=1e-8)
            seed_dir = torch.where(seed_dir_norm > 1e-8, seed_dir, radial)
        seed_dir = seed_dir / seed_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.growth_dir[sel_t] = seed_dir
        self.last_step_events = []
        for idx_i, dir_i in zip(sel_t.tolist(), seed_dir):
            self._record_tip_event(
                kind="seed",
                frame=self._advance_step,
                parent=-1,
                child=int(idx_i),
                positions=positions,
                direction=dir_i,
                score=float(score[int(idx_i)].item()),
                drive=float(score[int(idx_i)].item()),
            )
        return len(selected)

    def advance(
        self,
        graph,
        positions: Tensor,
        growth_drive: Tensor,
        growth_dir_hint: Tensor,
        impact_center: Optional[Tensor] = None,
    ) -> int:
        """Advance the active crack front by selecting a few successors."""
        family_cfg = self._family_settings()
        self.last_step_events = []
        if not family_cfg["front_enabled"]:
            return 0
        if not self.has_active_tips():
            return 0

        self._advance_step += 1
        device = positions.device
        normals = getattr(graph, "_normals", None)
        tip_indices = torch.where(self.tip_mask)[0]
        bbox_extent = positions.max(dim=0).values - positions.min(dim=0).values
        diag = float(bbox_extent.norm().item())
        height_scale = max(float(bbox_extent[2].item()), 1e-6)
        branch_scale = family_cfg["branch_scale"]
        successor_cap = max(0, family_cfg["successor_cap"])
        can_branch = (
            successor_cap > 1
            and tip_indices.numel()
            < max(1, int(round(self.max_branching_tips * max(branch_scale, 0.25))))
        )

        next_tip_mask = torch.zeros_like(self.tip_mask)
        next_growth_dir = torch.zeros_like(self.growth_dir)
        next_tip_age = torch.zeros_like(self.tip_age)
        next_branch_age = torch.zeros_like(self.tip_age)
        next_path_length = torch.zeros_like(self.path_length) if self.path_length is not None else None
        new_count = 0

        for tip_idx in tip_indices.tolist():
            i = int(tip_idx)
            branch_age_i = int(self.branch_age[i].item()) if self.branch_age is not None else 0
            is_guided_branch = branch_age_i > 0
            path_length_i = float(self.path_length[i].item()) if self.path_length is not None else 0.0
            max_path_scale = float(family_cfg.get("max_tip_path_scale", 0.0))
            if max_path_scale > 0.0 and path_length_i >= max_path_scale * max(diag, 1e-6):
                continue
            nbr_idx = graph.knn_idx[i]
            nbr_weights = graph.weights[i]
            valid = nbr_weights > 1e-8
            nbr_idx = nbr_idx[valid]
            if nbr_idx.numel() == 0:
                continue

            edge_raw = positions[nbr_idx] - positions[i].unsqueeze(0)
            edge_len = edge_raw.norm(dim=1)
            edge = edge_raw / edge_len.unsqueeze(1).clamp(min=1e-8)

            local_drive = growth_drive[nbr_idx].clamp(0.0, 1.0)
            local_drive = 1.0 - torch.pow(1.0 - local_drive, self.growth_gain)
            score = self.drive_weight * local_drive
            local_w = nbr_weights[valid]
            if local_w.numel() > 0:
                score = score + self.distance_weight * (
                    local_w / local_w.max().clamp(min=1e-8)
                )

            tip_dir = self.growth_dir[i]
            if tip_dir.norm() <= 1e-8:
                tip_dir = growth_dir_hint[i]
            if tip_dir.norm() > 1e-8:
                tip_dir = tip_dir / tip_dir.norm().clamp(min=1e-8)
                tip_forward_cos = edge @ tip_dir
                align_gain = (
                    self.align_weight
                    * family_cfg["align_scale"]
                    * (1.0 + 1.6 * self.anisotropy_strength)
                )
                score = score + align_gain * tip_forward_cos.clamp(min=0.0)
                if not is_guided_branch:
                    front_penalty = float(family_cfg.get("front_turn_penalty", 0.0))
                    if front_penalty > 0.0:
                        front_min = float(family_cfg.get("front_forward_min", -1.0))
                        score = score - front_penalty * (tip_forward_cos < front_min).float()

            hint_j = growth_dir_hint[nbr_idx]
            hint_j = hint_j / hint_j.norm(dim=1, keepdim=True).clamp(min=1e-8)
            hint_scale = float(family_cfg.get("branch_hint_scale", 1.0)) if is_guided_branch else 1.0
            score = score + (0.10 * hint_scale) * (edge * hint_j).sum(dim=1).clamp(min=0.0)

            parent = int(self.parent_index[i].item())
            if parent >= 0:
                parent_dir = positions[i] - positions[parent]
                parent_dir = parent_dir / parent_dir.norm().clamp(min=1e-8)
                score = score + (
                    self.continuity_weight
                    * family_cfg["continuity_scale"]
                    * (edge @ parent_dir).clamp(min=0.0)
                )
            lineage_depth = self._lineage_depth(i)
            can_branch_tip = (
                can_branch
                and lineage_depth >= int(family_cfg.get("branch_event_min_depth", 0))
            )
            closure_allowed = (
                lineage_depth >= int(family_cfg.get("closure_min_depth", 0))
                and branch_age_i <= int(family_cfg.get("closure_branch_age_max", 0))
            )
            tip_normal = (
                normals[i]
                if normals is not None and normals.shape[0] == positions.shape[0]
                else None
            )
            branch_target_dir = (
                self._branch_target_direction(tip_dir, tip_normal, i, family_cfg)
                if can_branch_tip and tip_dir.norm() > 1e-8
                else None
            )

            if normals is not None and normals.shape[0] == positions.shape[0]:
                n_i = normals[i].unsqueeze(0)
                n_j = normals[nbr_idx]
                tangent = (1.0 - (edge * n_i).sum(dim=1).abs())
                tangent = tangent * (1.0 - (edge * n_j).sum(dim=1).abs())
                tangent_gain = self.tangent_weight * (1.0 + self.anisotropy_strength)
                score = score + tangent_gain * tangent.clamp(0.0, 1.0)

            if impact_center is not None:
                radial = positions[nbr_idx] - impact_center.unsqueeze(0)
                radial = radial / radial.norm(dim=1, keepdim=True).clamp(min=1e-8)
                radial_gain = self.radial_weight * max(0.35, 1.0 - 0.55 * self.anisotropy_strength)
                if is_guided_branch:
                    radial_gain *= float(family_cfg.get("branch_radial_scale", 1.0))
                score = score + radial_gain * (edge * radial).sum(dim=1).clamp(min=0.0)

                escape_height = max(float((positions[i, 2] - impact_center[2]).item()), 0.0)
                escape_gain = 1.0 / (1.0 + 6.0 * escape_height)
            else:
                escape_gain = 1.0

            lift_gain = self.lift_weight * max(0.45, 1.0 - 0.35 * self.anisotropy_strength)
            if is_guided_branch:
                lift_gain *= float(family_cfg.get("branch_lift_scale", 1.0))
            score = score + (lift_gain * escape_gain) * edge[:, 2].clamp(min=0.0)

            revisit_penalty = self.visited_mask[nbr_idx].float()
            revisit_threshold = float(family_cfg.get("revisit_drive_override", self.revisit_drive_threshold))
            allow_revisit = local_drive > revisit_threshold
            score = score - family_cfg["revisit_penalty"] * (revisit_penalty * (~allow_revisit).float())

            score = score - 0.5 * self.tip_mask[nbr_idx].float()
            if tip_dir.norm() > 1e-8:
                lateral_score = (1.0 - (edge @ tip_dir).abs()).clamp(0.0, 1.0)
            else:
                lateral_score = torch.zeros(edge.shape[0], device=device, dtype=positions.dtype)
            score = score + family_cfg["lateral_branch_bonus"] * lateral_score * local_drive
            if is_guided_branch and tip_dir.norm() > 1e-8:
                guide_cos = edge @ tip_dir
                guide_score = guide_cos.clamp(0.0, 1.0)
                score = (
                    score
                    + float(family_cfg.get("branch_guidance_weight", 0.0)) * guide_score
                    - float(family_cfg.get("branch_turn_penalty", 0.0)) * (1.0 - guide_score)
                )
                forward_min = float(family_cfg.get("branch_forward_min", -1.0))
                hard_penalty = float(family_cfg.get("branch_hard_turn_penalty", 0.0))
                if hard_penalty > 0.0:
                    score = score - hard_penalty * (guide_cos < forward_min).float()

            if closure_allowed:
                closure_score = self._compute_loop_closure_scores(
                    tip_index=i,
                    neighbor_idx=nbr_idx,
                    edge_dir=edge,
                    positions=positions,
                    tip_dir=tip_dir if tip_dir.norm() > 1e-8 else torch.zeros(3, device=device),
                    family_cfg=family_cfg,
                    diag=diag,
                    height_scale=height_scale,
                )
            else:
                closure_score = torch.zeros(edge.shape[0], device=device, dtype=positions.dtype)
            duplicate_score = self._duplicate_candidate_score(
                graph=graph,
                candidate_idx=nbr_idx,
                candidate_edge=edge,
                tip_index=i,
                family_cfg=family_cfg,
                closure_score=closure_score,
            )

            closure_weight = float(family_cfg.get("closure_score_weight", 0.0))
            if closure_weight > 0.0:
                score = score + closure_weight * closure_score

            branch_event_score = torch.zeros_like(score)
            branch_angle_target = torch.full_like(score, float(family_cfg.get("branch_angle_center", 0.54)))
            branch_angle_noise = torch.zeros_like(score)
            branch_target_align = torch.zeros_like(score)
            if tip_dir.norm() > 1e-8:
                forward_cos = (edge @ tip_dir).clamp(0.0, 1.0)
                angle_center = float(family_cfg.get("branch_angle_center", 0.54))
                angle_width = max(float(family_cfg.get("branch_angle_width", 0.38)), 1e-4)
                angle_jitter = max(float(family_cfg.get("branch_angle_jitter", 0.0)), 0.0)
                branch_angle_noise = self._stable_signed_noise(
                    i,
                    nbr_idx,
                    salt=17 + 31 * max(lineage_depth, 0),
                ).to(device=device, dtype=score.dtype)
                branch_angle_target = (
                    angle_center + angle_jitter * branch_angle_noise
                ).clamp(0.05, 0.95)
                angle_score = (1.0 - (forward_cos - branch_angle_target).abs() / angle_width).clamp(0.0, 1.0)
                if branch_target_dir is not None:
                    branch_target_align = (edge @ branch_target_dir).clamp(0.0, 1.0)
                    angle_score = torch.maximum(angle_score, branch_target_align)

                # Branch direction selection.
                #   "energy" — pick lateral candidate with highest local drive
                #              (Karma-Lobkovsky-style: branch follows energy);
                #              the legacy hash-based angle target only acts as
                #              a soft modulator floored at branch_angle_prior_floor.
                #   "angle"  — legacy: angle_score is the dominant factor.
                #   "hybrid" — geometric mean of the two.
                energy_branch = (
                    lateral_score * local_drive * (1.0 - duplicate_score)
                ).clamp(0.0, 1.0)
                if self.branch_direction_mode == "angle":
                    branch_event_score = (
                        angle_score * energy_branch
                    ).clamp(0.0, 1.0)
                elif self.branch_direction_mode == "hybrid":
                    angle_mod = (
                        self.branch_angle_prior_floor
                        + (1.0 - self.branch_angle_prior_floor) * angle_score
                    )
                    branch_event_score = torch.sqrt(
                        (angle_mod * energy_branch).clamp(min=0.0)
                    ).clamp(0.0, 1.0)
                else:  # "energy"
                    angle_mod = (
                        self.branch_angle_prior_floor
                        + (1.0 - self.branch_angle_prior_floor) * angle_score
                    )
                    branch_event_score = (angle_mod * energy_branch).clamp(0.0, 1.0)
                branch_bonus = float(family_cfg.get("branch_event_bonus", 0.0))
                if can_branch_tip and branch_bonus > 0.0:
                    score = score + branch_bonus * branch_event_score

            duplicate_penalty = float(family_cfg.get("duplicate_penalty", 0.0))
            if duplicate_penalty > 0.0:
                score = score - duplicate_penalty * duplicate_score

            # Griffith-style gate: candidate must clear the local energy
            # drive threshold before it is allowed to advance, separate from
            # the soft aggregate score.  This makes propagation an
            # energy-conditioned event rather than a pure score quantile.
            griffith_threshold = self.growth_griffith_threshold
            griffith_pass = local_drive >= griffith_threshold
            keep = torch.where(
                (score > self.min_successor_score) & griffith_pass
            )[0]
            if keep.numel() == 0:
                if int(self.tip_age[i].item()) < self.max_tip_age:
                    next_tip_mask[i] = True
                    next_growth_dir[i] = tip_dir if tip_dir.norm() > 1e-8 else torch.zeros(3, device=device)
                    next_tip_age[i] = self.tip_age[i] + 1
                    next_branch_age[i] = max(branch_age_i - 1, 0)
                    if next_path_length is not None:
                        next_path_length[i] = path_length_i
                continue

            keep_score = score[keep]
            order = keep_score.argsort(descending=True)
            keep = keep[order]
            branch_drive_threshold = max(
                self.branch_drive_threshold - 0.58 * self.branching_bias * branch_scale,
                0.08,
            )
            main_keep = keep[:1]
            keep = main_keep
            branch_budget = int(family_cfg["branch_extra_branches"])
            closure_budget = int(family_cfg["closure_extra_branches"]) if closure_allowed else 0
            extra_budget = branch_budget + closure_budget
            if keep.numel() > 0 and keep.numel() < successor_cap and can_branch_tip and extra_budget > 0:
                extra_pool = keep.new_tensor([], dtype=keep.dtype)

                # Top-K relative gate on branch_event_score: only the K
                # strongest energy-branch candidates per tip are eligible,
                # which makes the gate resolution-invariant (the absolute
                # number of branches stays bounded as graph density grows).
                # The legacy per-family `branch_event_threshold` is kept as
                # a soft minimum-magnitude floor (filters near-zero noise).
                event_legacy_thresh = float(
                    family_cfg.get("branch_event_threshold", 1.0)
                )
                # Saturated >=1.0 means "branch only on perfect score" --
                # this already disables event-driven branching for the
                # corresponding style; honor it.
                if event_legacy_thresh >= 1.0 - 1e-6:
                    event_topk_floor = float("inf")
                else:
                    n_event_cand = int(branch_event_score.numel())
                    k_top = min(int(self.branch_event_topk), n_event_cand)
                    if k_top > 0:
                        event_topk_floor = float(
                            branch_event_score.topk(k_top).values.min().item()
                        )
                    else:
                        event_topk_floor = float("inf")
                event_soft_floor = min(event_legacy_thresh, 0.05)

                branch_candidates = torch.where(
                    (lateral_score >= family_cfg["lateral_branch_threshold"])
                    & (local_drive >= max(0.10, 0.65 * branch_drive_threshold))
                    & (branch_event_score >= event_topk_floor)
                    & (branch_event_score >= event_soft_floor)
                    & (branch_target_align >= float(family_cfg.get("branch_target_align_threshold", 0.0)))
                    & (score >= 0.50 * self.min_successor_score)
                )[0]
                closure_candidates = torch.where(
                    (closure_score >= family_cfg["closure_branch_threshold"])
                    & (lateral_score >= 0.45 * family_cfg["lateral_branch_threshold"])
                    & (score >= 0.78 * self.min_successor_score)
                )[0]
                candidate_mask = torch.zeros(score.shape[0], dtype=torch.bool, device=device)
                if branch_candidates.numel() > 0:
                    candidate_mask[branch_candidates] = True
                if closure_budget > 0 and closure_candidates.numel() > 0:
                    candidate_mask[closure_candidates] = True
                candidate_mask[main_keep] = False
                candidate_idx = torch.where(candidate_mask)[0]
                if candidate_idx.numel() > 0:
                    extra_aug = (
                        0.25 * score[candidate_idx]
                        + family_cfg["closure_branch_bonus"] * closure_score[candidate_idx]
                        + 0.60 * family_cfg["lateral_branch_bonus"] * lateral_score[candidate_idx]
                        + 1.35 * float(family_cfg.get("branch_event_bonus", 0.0)) * branch_event_score[candidate_idx]
                        + 0.90 * branch_target_align[candidate_idx]
                        - 0.35 * duplicate_score[candidate_idx]
                    )
                    extra_order = extra_aug.argsort(descending=True)
                    extra_pool = candidate_idx[extra_order]
                added = 0
                selected = keep
                for cand in extra_pool.tolist():
                    cand_t = keep.new_tensor([cand], dtype=keep.dtype)
                    if bool((selected == cand_t.item()).any()):
                        continue
                    selected = torch.cat([selected, cand_t], dim=0)
                    added += 1
                    if added >= extra_budget or selected.numel() >= successor_cap:
                        break
                keep = selected
            chosen = nbr_idx[keep]

            for local_k, j_t in enumerate(chosen.tolist()):
                j = int(j_t)
                next_tip_mask[j] = True
                self.parent_index[j] = i
                if self.activation_step is not None and not bool(self.visited_mask[j].item()):
                    self.activation_step[j] = self._advance_step
                self.visited_mask[j] = True
                next_tip_age[j] = 0
                next_branch_age[j] = max(branch_age_i - 1, 0)
                if next_path_length is not None:
                    next_path_length[j] = path_length_i + float(edge_len[keep[local_k]].item())
                grow_vec = edge[keep[local_k]]
                hint_vec = growth_dir_hint[j]
                if local_k == 0 and is_guided_branch and tip_dir.norm() > 1e-8:
                    mix = 0.82 * tip_dir + 0.14 * grow_vec + 0.04 * hint_vec
                elif local_k > 0 and tip_dir.norm() > 1e-8:
                    next_branch_age[j] = int(family_cfg.get("branch_guidance_steps", 0))
                    side_vec = grow_vec - (grow_vec @ tip_dir) * tip_dir
                    if side_vec.norm() <= 1e-8:
                        side_vec = hint_vec - (hint_vec @ tip_dir) * tip_dir
                    if (
                        normals is not None
                        and normals.shape[0] == positions.shape[0]
                        and side_vec.norm() > 1e-8
                    ):
                        n_i = normals[i]
                        if n_i.norm() > 1e-8:
                            n_i = n_i / n_i.norm().clamp(min=1e-8)
                            side_vec = side_vec - (side_vec @ n_i) * n_i
                    if branch_target_dir is not None:
                        mix = 0.84 * branch_target_dir + 0.12 * grow_vec + 0.04 * hint_vec
                    elif side_vec.norm() > 1e-8:
                        side_vec = side_vec / side_vec.norm().clamp(min=1e-8)
                        angle_cos = branch_angle_target[keep[local_k]].clamp(0.05, 0.95)
                        angle_sin = torch.sqrt((1.0 - angle_cos * angle_cos).clamp(min=0.0))
                        branch_dir = angle_cos * tip_dir + angle_sin * side_vec
                        if (
                            normals is not None
                            and normals.shape[0] == positions.shape[0]
                            and branch_dir.norm() > 1e-8
                        ):
                            n_i = normals[i]
                            if n_i.norm() > 1e-8:
                                n_i = n_i / n_i.norm().clamp(min=1e-8)
                                branch_dir = branch_dir - (branch_dir @ n_i) * n_i
                        branch_dir = branch_dir / branch_dir.norm().clamp(min=1e-8)
                        mix = 0.55 * branch_dir + 0.30 * grow_vec + 0.15 * hint_vec
                    else:
                        mix = 0.65 * grow_vec + 0.35 * hint_vec
                else:
                    mix = 0.65 * grow_vec + 0.35 * hint_vec
                if mix.norm() <= 1e-8:
                    mix = grow_vec
                next_growth_dir[j] = mix / mix.norm().clamp(min=1e-8)
                local_idx = keep[local_k]
                branch_angle_deg = 0.0
                if tip_dir.norm() > 1e-8 and next_growth_dir[j].norm() > 1e-8:
                    angle_cos = (next_growth_dir[j] @ tip_dir).clamp(-1.0, 1.0)
                    branch_angle_deg = float(torch.rad2deg(torch.acos(angle_cos)).item())
                self._record_tip_event(
                    kind="branch" if local_k > 0 else "advance",
                    frame=self._advance_step,
                    parent=i,
                    child=j,
                    positions=positions,
                    direction=next_growth_dir[j],
                    score=float(score[local_idx].item()),
                    drive=float(local_drive[local_idx].item()),
                    closure_score=float(closure_score[local_idx].item()),
                    branch_score=float(branch_event_score[local_idx].item()),
                    branch_angle_deg=branch_angle_deg,
                    branch_angle_target=float(branch_angle_target[local_idx].item()),
                    lineage_depth=int(lineage_depth),
                )
                new_count += 1

            if (
                chosen.numel() > 1
                and int(self.tip_age[i].item()) < int(family_cfg["branch_persist_steps"])
            ):
                chosen_lateral = lateral_score[keep].max().item() if keep.numel() > 0 else 0.0
                if chosen_lateral >= float(family_cfg["branch_persist_lateral"]):
                    next_tip_mask[i] = True
                    next_growth_dir[i] = tip_dir if tip_dir.norm() > 1e-8 else torch.zeros(3, device=device)
                    next_tip_age[i] = self.tip_age[i] + 1
                    next_branch_age[i] = max(branch_age_i - 1, 0)
                    if next_path_length is not None:
                        next_path_length[i] = path_length_i

        self.tip_mask = next_tip_mask
        self.growth_dir = next_growth_dir
        self.tip_age = next_tip_age
        self.branch_age = next_branch_age
        if next_path_length is not None:
            self.path_length = next_path_length
        self.visited_mask |= next_tip_mask
        return new_count

    def get_state(self) -> dict:
        return {
            "tip_mask": self.tip_mask,
            "visited_mask": self.visited_mask,
            "parent_index": self.parent_index,
            "tip_age": self.tip_age,
            "branch_age": self.branch_age,
            "activation_step": self.activation_step,
            "path_length": self.path_length,
            "growth_dir": self.growth_dir,
            "advance_step": self._advance_step,
        }

    def export_event_log(self) -> list[dict]:
        return list(self.event_log)

    def load_state(self, state: dict) -> None:
        self.tip_mask = state["tip_mask"]
        self.visited_mask = state["visited_mask"]
        self.parent_index = state["parent_index"]
        self.tip_age = state["tip_age"]
        self.branch_age = state.get("branch_age")
        if self.branch_age is None and self.tip_age is not None:
            self.branch_age = torch.zeros_like(self.tip_age)
        self.activation_step = state.get("activation_step")
        if self.activation_step is None and self.tip_age is not None:
            self.activation_step = torch.full_like(self.tip_age, -1)
        self.path_length = state.get("path_length")
        if self.path_length is None and self.tip_age is not None:
            self.path_length = torch.zeros_like(self.tip_age, dtype=torch.float32)
        self.growth_dir = state["growth_dir"]
        self._advance_step = int(state.get("advance_step", 0))
        self.event_log = []
        self.last_step_events = []
