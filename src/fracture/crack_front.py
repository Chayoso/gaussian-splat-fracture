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
        material_family: str = "neutral_reference",
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
        self.material_family = str(material_family)
        self.device = torch.device(device)

        self.tip_mask: Optional[Tensor] = None
        self.visited_mask: Optional[Tensor] = None
        self.parent_index: Optional[Tensor] = None
        self.tip_age: Optional[Tensor] = None
        self.growth_dir: Optional[Tensor] = None

    def _family_settings(self) -> dict:
        if self.material_family == "sharp_brittle":
            return {
                "front_enabled": True,
                "branch_scale": 0.35,
                "continuity_scale": 1.35,
                "align_scale": 1.20,
                "revisit_penalty": 0.95,
                "successor_cap": 1,
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
            }
        if self.material_family == "brittle_moderate":
            return {
                "front_enabled": True,
                "branch_scale": 0.70,
                "continuity_scale": 1.08,
                "align_scale": 1.00,
                "revisit_penalty": 0.85,
                "successor_cap": 3,
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
            }
        if self.material_family == "rough_quasi_brittle":
            return {
                "front_enabled": True,
                "branch_scale": 1.85,
                "continuity_scale": 0.70,
                "align_scale": 0.80,
                "revisit_penalty": 0.22,
                "successor_cap": max(self.successor_topk, 5),
                "seed_spacing_scale": 0.85,
                "lateral_branch_bonus": 0.26,
                "lateral_branch_threshold": 0.18,
                "branch_persist_steps": 1,
                "branch_persist_lateral": 0.22,
                "branch_extra_branches": 2,
                "closure_weight": 0.42,
                "closure_branch_threshold": 0.18,
                "closure_branch_bonus": 0.36,
                "closure_extra_branches": 2,
                "closure_target_cap": 96,
                "closure_min_dist_scale": 0.10,
                "closure_max_dist_scale": 0.40,
                "closure_height_scale": 0.18,
            }
        if self.material_family == "diffuse_damage":
            return {
                "front_enabled": False,
                "branch_scale": 0.0,
                "continuity_scale": 0.0,
                "align_scale": 0.0,
                "revisit_penalty": 1.0,
                "successor_cap": 0,
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
            }
        return {
            "front_enabled": True,
            "branch_scale": 1.0,
            "continuity_scale": 1.0,
            "align_scale": 1.0,
            "revisit_penalty": 0.75,
            "successor_cap": max(1, min(self.successor_topk, 2)),
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
        }

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
        dist_next = torch.cdist(neighbor_pos, positions[target_idx])
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
        self.growth_dir = torch.zeros(N, 3, device=device)

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
        max_seed_points = min(self.max_seed_points, family_cfg["successor_cap"] + 1)
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

        seed_dir = growth_dir_hint[sel_t]
        seed_dir_norm = seed_dir.norm(dim=1, keepdim=True)
        if impact_center is not None:
            radial = positions[sel_t] - impact_center.unsqueeze(0)
            radial = radial / radial.norm(dim=1, keepdim=True).clamp(min=1e-8)
            seed_dir = torch.where(seed_dir_norm > 1e-8, seed_dir, radial)
        seed_dir = seed_dir / seed_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.growth_dir[sel_t] = seed_dir
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
        if not family_cfg["front_enabled"]:
            return 0
        if not self.has_active_tips():
            return 0

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
        new_count = 0

        for tip_idx in tip_indices.tolist():
            i = int(tip_idx)
            nbr_idx = graph.knn_idx[i]
            nbr_weights = graph.weights[i]
            valid = nbr_weights > 1e-8
            nbr_idx = nbr_idx[valid]
            if nbr_idx.numel() == 0:
                continue

            edge = positions[nbr_idx] - positions[i].unsqueeze(0)
            edge = edge / edge.norm(dim=1, keepdim=True).clamp(min=1e-8)

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
                align_gain = (
                    self.align_weight
                    * family_cfg["align_scale"]
                    * (1.0 + 1.6 * self.anisotropy_strength)
                )
                score = score + align_gain * (edge @ tip_dir).clamp(min=0.0)

            hint_j = growth_dir_hint[nbr_idx]
            hint_j = hint_j / hint_j.norm(dim=1, keepdim=True).clamp(min=1e-8)
            score = score + 0.10 * (edge * hint_j).sum(dim=1).clamp(min=0.0)

            parent = int(self.parent_index[i].item())
            if parent >= 0:
                parent_dir = positions[i] - positions[parent]
                parent_dir = parent_dir / parent_dir.norm().clamp(min=1e-8)
                score = score + (
                    self.continuity_weight
                    * family_cfg["continuity_scale"]
                    * (edge @ parent_dir).clamp(min=0.0)
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
                score = score + radial_gain * (edge * radial).sum(dim=1).clamp(min=0.0)

                escape_height = max(float((positions[i, 2] - impact_center[2]).item()), 0.0)
                escape_gain = 1.0 / (1.0 + 6.0 * escape_height)
            else:
                escape_gain = 1.0

            lift_gain = self.lift_weight * max(0.45, 1.0 - 0.35 * self.anisotropy_strength)
            score = score + (lift_gain * escape_gain) * edge[:, 2].clamp(min=0.0)

            revisit_penalty = self.visited_mask[nbr_idx].float()
            allow_revisit = local_drive > self.revisit_drive_threshold
            score = score - family_cfg["revisit_penalty"] * (revisit_penalty * (~allow_revisit).float())

            score = score - 0.5 * self.tip_mask[nbr_idx].float()
            if tip_dir.norm() > 1e-8:
                lateral_score = (1.0 - (edge @ tip_dir).abs()).clamp(0.0, 1.0)
            else:
                lateral_score = torch.zeros(edge.shape[0], device=device, dtype=positions.dtype)
            score = score + family_cfg["lateral_branch_bonus"] * lateral_score * local_drive
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

            keep = torch.where(score > self.min_successor_score)[0]
            if keep.numel() == 0:
                if int(self.tip_age[i].item()) < self.max_tip_age:
                    next_tip_mask[i] = True
                    next_growth_dir[i] = tip_dir if tip_dir.norm() > 1e-8 else torch.zeros(3, device=device)
                    next_tip_age[i] = self.tip_age[i] + 1
                continue

            keep_score = score[keep]
            order = keep_score.argsort(descending=True)
            keep = keep[order]
            branch_topk = 1
            if keep.numel() > 1 and successor_cap > 1 and can_branch:
                second_drive = local_drive[keep[1]]
                branch_ratio = max(self.branch_score_ratio - 0.30 * self.branching_bias * branch_scale, 0.54)
                branch_drive_threshold = max(
                    self.branch_drive_threshold - 0.58 * self.branching_bias * branch_scale,
                    0.08,
                )
                if (
                    keep_score[order[1]] >= branch_ratio * keep_score[order[0]]
                    and second_drive >= branch_drive_threshold
                ):
                    branch_topk = min(successor_cap, keep.numel())
            keep = keep[:branch_topk]
            extra_budget = int(family_cfg["branch_extra_branches"]) + int(family_cfg["closure_extra_branches"])
            if keep.numel() > 0 and keep.numel() < successor_cap and can_branch and extra_budget > 0:
                extra_pool = keep.new_tensor([], dtype=keep.dtype)
                branch_candidates = torch.where(
                    (lateral_score >= family_cfg["lateral_branch_threshold"])
                    & (local_drive >= max(0.10, 0.65 * branch_drive_threshold))
                    & (score >= 0.72 * self.min_successor_score)
                )[0]
                closure_candidates = torch.where(
                    (closure_score >= family_cfg["closure_branch_threshold"])
                    & (score >= 0.78 * self.min_successor_score)
                )[0]
                candidate_mask = torch.zeros(score.shape[0], dtype=torch.bool, device=device)
                if branch_candidates.numel() > 0:
                    candidate_mask[branch_candidates] = True
                if closure_candidates.numel() > 0:
                    candidate_mask[closure_candidates] = True
                candidate_idx = torch.where(candidate_mask)[0]
                if candidate_idx.numel() > 0:
                    extra_aug = (
                        score[candidate_idx]
                        + family_cfg["closure_branch_bonus"] * closure_score[candidate_idx]
                        + 0.75 * family_cfg["lateral_branch_bonus"] * lateral_score[candidate_idx]
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
                self.visited_mask[j] = True
                next_tip_age[j] = 0
                grow_vec = edge[keep[local_k]]
                hint_vec = growth_dir_hint[j]
                mix = 0.65 * grow_vec + 0.35 * hint_vec
                if mix.norm() <= 1e-8:
                    mix = grow_vec
                next_growth_dir[j] = mix / mix.norm().clamp(min=1e-8)
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

        self.tip_mask = next_tip_mask
        self.growth_dir = next_growth_dir
        self.tip_age = next_tip_age
        self.visited_mask |= next_tip_mask
        return new_count

    def get_state(self) -> dict:
        return {
            "tip_mask": self.tip_mask,
            "visited_mask": self.visited_mask,
            "parent_index": self.parent_index,
            "tip_age": self.tip_age,
            "growth_dir": self.growth_dir,
        }

    def load_state(self, state: dict) -> None:
        self.tip_mask = state["tip_mask"]
        self.visited_mask = state["visited_mask"]
        self.parent_index = state["parent_index"]
        self.tip_age = state["tip_age"]
        self.growth_dir = state["growth_dir"]
