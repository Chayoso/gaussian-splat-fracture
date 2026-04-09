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
        self.device = torch.device(device)

        self.tip_mask: Optional[Tensor] = None
        self.visited_mask: Optional[Tensor] = None
        self.parent_index: Optional[Tensor] = None
        self.tip_age: Optional[Tensor] = None
        self.growth_dir: Optional[Tensor] = None

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
        if self.tip_mask is None or self.tip_mask.shape[0] != positions.shape[0]:
            self.initialize(positions.shape[0], positions.device)
        if self.has_active_tips():
            return 0

        score = init_score.clamp(min=0.0)
        if score.max() <= 0:
            return 0

        q = float(min(max(self.seed_quantile, 0.0), 0.999))
        thresh = torch.quantile(score.detach(), q)
        candidate_idx = torch.where(score >= thresh)[0]
        if candidate_idx.numel() == 0:
            candidate_idx = score.topk(min(self.max_seed_points, score.numel())).indices
        elif candidate_idx.numel() < min(self.max_seed_points, score.numel()):
            top_idx = score.topk(min(self.max_seed_points, score.numel())).indices
            candidate_idx = torch.unique(torch.cat([candidate_idx, top_idx], dim=0))

        order = score[candidate_idx].argsort(descending=True)
        candidate_idx = candidate_idx[order]

        selected = []
        for idx in candidate_idx.tolist():
            if len(selected) >= self.max_seed_points:
                break
            pos_i = positions[idx]
            if selected:
                sel_pos = positions[torch.tensor(selected, device=positions.device)]
                min_dist = torch.norm(sel_pos - pos_i.unsqueeze(0), dim=1).min().item()
                if min_dist < self.min_seed_spacing:
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
        if not self.has_active_tips():
            return 0

        device = positions.device
        normals = getattr(graph, "_normals", None)
        tip_indices = torch.where(self.tip_mask)[0]
        can_branch = tip_indices.numel() < self.max_branching_tips

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
                score = score + self.align_weight * (edge @ tip_dir).clamp(min=0.0)

            hint_j = growth_dir_hint[nbr_idx]
            hint_j = hint_j / hint_j.norm(dim=1, keepdim=True).clamp(min=1e-8)
            score = score + 0.10 * (edge * hint_j).sum(dim=1).clamp(min=0.0)

            parent = int(self.parent_index[i].item())
            if parent >= 0:
                parent_dir = positions[i] - positions[parent]
                parent_dir = parent_dir / parent_dir.norm().clamp(min=1e-8)
                score = score + self.continuity_weight * (edge @ parent_dir).clamp(min=0.0)

            if normals is not None and normals.shape[0] == positions.shape[0]:
                n_i = normals[i].unsqueeze(0)
                n_j = normals[nbr_idx]
                tangent = (1.0 - (edge * n_i).sum(dim=1).abs())
                tangent = tangent * (1.0 - (edge * n_j).sum(dim=1).abs())
                score = score + self.tangent_weight * tangent.clamp(0.0, 1.0)

            if impact_center is not None:
                radial = positions[nbr_idx] - impact_center.unsqueeze(0)
                radial = radial / radial.norm(dim=1, keepdim=True).clamp(min=1e-8)
                score = score + self.radial_weight * (edge * radial).sum(dim=1).clamp(min=0.0)

                escape_height = max(float((positions[i, 2] - impact_center[2]).item()), 0.0)
                escape_gain = 1.0 / (1.0 + 6.0 * escape_height)
            else:
                escape_gain = 1.0

            score = score + (self.lift_weight * escape_gain) * edge[:, 2].clamp(min=0.0)

            revisit_penalty = self.visited_mask[nbr_idx].float()
            allow_revisit = local_drive > self.revisit_drive_threshold
            score = score - 0.75 * (revisit_penalty * (~allow_revisit).float())

            score = score - 0.5 * self.tip_mask[nbr_idx].float()

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
            if keep.numel() > 1 and self.successor_topk > 1 and can_branch:
                second_drive = local_drive[keep[1]]
                if (
                    keep_score[order[1]] >= self.branch_score_ratio * keep_score[order[0]]
                    and second_drive >= self.branch_drive_threshold
                ):
                    branch_topk = min(self.successor_topk, keep.numel())
            keep = keep[:branch_topk]
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
