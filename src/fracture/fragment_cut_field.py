"""Cut-field helpers for graph fragment detection."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .graph_builder import GaussianGraph


class FragmentCutFieldMixin:
    """Compute directional cut fields and boundary crack support."""

    @staticmethod
    def _normalize_vectors(v: Tensor) -> Tensor:
        norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.where(norm > 1e-8, v / norm, torch.zeros_like(v))

    def _break_threshold_from_edge_damage(
        self,
        damage_threshold: float,
        edge_break_rate: float,
    ) -> float:
        break_thresh = 1.0 - float(max(1.0 - damage_threshold, 0.0)) ** (
            1.0 / max(edge_break_rate, 1e-8)
        )
        return float(min(max(break_thresh, 0.0), 1.0))

    def _directional_edge_gate(
        self,
        graph: GaussianGraph,
        crack_normal: Optional[Tensor],
        crack_tangent: Optional[Tensor],
    ) -> Optional[Tensor]:
        if graph.knn_idx is None or crack_normal is None:
            return None

        cut_dir = self._normalize_vectors(crack_normal)
        if crack_tangent is not None and crack_tangent.shape == cut_dir.shape:
            tangent_dir = self._normalize_vectors(crack_tangent)
        elif graph._normals is not None and graph._normals.shape == cut_dir.shape:
            surf_normal = self._normalize_vectors(graph._normals)
            tangent_dir = self._normalize_vectors(torch.cross(surf_normal, cut_dir, dim=1))
        else:
            tangent_dir = torch.zeros_like(cut_dir)

        tan_i = tangent_dir.unsqueeze(1).expand(-1, graph.knn_idx.shape[1], -1)
        tan_j = tangent_dir[graph.knn_idx]
        norm_i = cut_dir.unsqueeze(1).expand_as(tan_i)
        norm_j = cut_dir[graph.knn_idx]

        tan_align = (tan_i * tan_j).sum(dim=2).abs()
        norm_align = (norm_i * norm_j).sum(dim=2).abs()

        tan_gate = (
            (tan_align - self.cut_cos_gate_tangent)
            / max(1.0 - self.cut_cos_gate_tangent, 1e-6)
        ).clamp(0.0, 1.0)
        norm_gate = (
            (norm_align - self.cut_cos_gate_normal)
            / max(1.0 - self.cut_cos_gate_normal, 1e-6)
        ).clamp(0.0, 1.0)
        return (tan_gate * norm_gate).clamp(0.0, 1.0)

    def _diffuse_cut_corridor(
        self,
        graph: GaussianGraph,
        seed_d_cut: Tensor,
        crack_normal: Optional[Tensor],
        crack_tangent: Optional[Tensor],
    ) -> Tensor:
        if (
            graph.knn_idx is None
            or self.cut_diffusion_iters <= 0
            or self.cut_diffusion_alpha <= 0.0
        ):
            return seed_d_cut.clamp(0.0, 1.0)

        gate = self._directional_edge_gate(graph, crack_normal, crack_tangent)
        if gate is None:
            return seed_d_cut.clamp(0.0, 1.0)

        d_cut = seed_d_cut.clamp(0.0, 1.0)
        for _ in range(self.cut_diffusion_iters):
            node_support = d_cut.max(dim=1).values
            node_support = torch.maximum(node_support, graph.weighted_neighbor_max(node_support))
            support_i = node_support.unsqueeze(1).expand_as(d_cut)
            support_j = node_support[graph.knn_idx]
            propagated = self.cut_diffusion_alpha * torch.maximum(support_i, support_j) * gate
            d_cut = torch.maximum(d_cut, propagated)
        return d_cut.clamp(0.0, 1.0)

    def _compute_boundary_cut_stats(
        self,
        labels: Tensor,
        graph: GaussianGraph,
        effective_cut_damage: Tensor,
        cut_threshold: float,
        excluded_mask: Optional[Tensor] = None,
    ) -> Tuple[dict, dict]:
        label_i = labels.unsqueeze(1).expand_as(graph.knn_idx)
        label_j = labels[graph.knn_idx]
        cross_component = label_i != label_j
        if excluded_mask is not None:
            excluded = excluded_mask.to(device=labels.device, dtype=torch.bool)
            cross_component = (
                cross_component
                & (~excluded.unsqueeze(1))
                & (~excluded[graph.knn_idx])
            )
        cut_mask = effective_cut_damage >= cut_threshold

        ratio_by_old = {}
        boundary_edges_by_old = {}
        valid_ratios = []
        for old_label in labels.unique().tolist():
            mask = labels == old_label
            boundary_mask = cross_component & mask.unsqueeze(1)
            boundary_edges = int(boundary_mask.sum().item())
            boundary_edges_by_old[old_label] = boundary_edges
            if boundary_edges <= 0:
                ratio_by_old[old_label] = 0.0
                continue
            cut_ratio = float(cut_mask[boundary_mask].float().mean().item())
            ratio_by_old[old_label] = cut_ratio
            valid_ratios.append(cut_ratio)

        if valid_ratios:
            ratio_tensor = torch.tensor(valid_ratios, dtype=effective_cut_damage.dtype)
            self.last_boundary_cut_ratio_mean = float(ratio_tensor.mean().item())
            self.last_boundary_cut_ratio_q50 = float(torch.quantile(ratio_tensor, 0.50).item())
            self.last_boundary_cut_ratio_q90 = float(torch.quantile(ratio_tensor, 0.90).item())
            self.last_boundary_cut_ratio_max = float(ratio_tensor.max().item())
        else:
            self.last_boundary_cut_ratio_mean = 0.0
            self.last_boundary_cut_ratio_q50 = 0.0
            self.last_boundary_cut_ratio_q90 = 0.0
            self.last_boundary_cut_ratio_max = 0.0

        return ratio_by_old, boundary_edges_by_old


