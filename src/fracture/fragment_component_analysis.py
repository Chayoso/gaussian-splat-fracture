"""Component analysis helpers for graph fragment detection."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .graph_builder import GaussianGraph


class FragmentComponentAnalysisMixin:
    """Score component interfaces, support loss, and cut-surface votes."""

    def _cluster_boundary_candidates(
        self,
        labels: Tensor,
        positions: Optional[Tensor],
        candidate_labels: set,
        score_by_old: dict,
        authoritative_cut_mask: Optional[Tensor],
        min_group_size: int,
    ) -> Tuple[dict, set, dict]:
        def size_filtered_labels(label_set: set) -> set:
            keep = set()
            for old_label in label_set:
                size = int((labels == old_label).sum().item())
                if size >= max(int(min_group_size), 1):
                    keep.add(old_label)
            return keep

        if positions is None or not candidate_labels:
            return {}, size_filtered_labels(candidate_labels), dict(score_by_old)
        if self.material_family not in {"rough_quasi_brittle", "brittle_moderate"}:
            return {}, size_filtered_labels(candidate_labels), dict(score_by_old)

        z = positions[:, 2]
        z_min = float(z.min().item())
        z_max = float(z.max().item())
        height_scale = max(z_max - z_min, 1e-6)
        bbox_extent = positions.max(dim=0).values - positions.min(dim=0).values
        diag = float(bbox_extent.norm().item())
        merge_radius = 0.09 * max(diag, 1e-6)
        if self.material_family == "brittle_moderate":
            merge_radius *= 0.72

        stats = {}
        for old_label in sorted(candidate_labels):
            mask = labels == old_label
            if not bool(mask.any()):
                continue
            auth_overlap = 0.0
            if authoritative_cut_mask is not None:
                auth_overlap = float((mask & authoritative_cut_mask).sum().item()) / max(int(mask.sum().item()), 1)
            stats[old_label] = {
                "com": positions[mask].mean(dim=0),
                "mean_z": float(z[mask].mean().item()),
                "auth_overlap": auth_overlap,
                "size": int(mask.sum().item()),
            }
        if len(stats) <= 1:
            return {}, size_filtered_labels(candidate_labels), dict(score_by_old)

        parent = {lbl: lbl for lbl in stats}

        def find(lbl):
            while parent[lbl] != lbl:
                parent[lbl] = parent[parent[lbl]]
                lbl = parent[lbl]
            return lbl

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if stats[ra]["size"] < stats[rb]["size"]:
                ra, rb = rb, ra
            parent[rb] = ra

        label_list = list(stats.keys())
        for i, a in enumerate(label_list):
            for b in label_list[i + 1:]:
                sa = stats[a]
                sb = stats[b]
                dist = float((sa["com"] - sb["com"]).norm().item())
                if dist > merge_radius:
                    continue
                if abs(sa["mean_z"] - sb["mean_z"]) > 0.18 * height_scale:
                    continue
                auth_gate = max(sa["auth_overlap"], sb["auth_overlap"])
                if auth_gate < 0.04 and min(
                    float(score_by_old.get(a, 0.0)),
                    float(score_by_old.get(b, 0.0)),
                ) < self.fallback_cut_ratio:
                    continue
                union(a, b)

        component_group_map = {}
        grouped_labels = set()
        grouped_scores = {}
        members_by_group = {}
        for old_label in stats:
            root = find(old_label)
            component_group_map[old_label] = root
            members_by_group.setdefault(root, []).append(old_label)

        for root, members in members_by_group.items():
            aggregated_size = sum(int(stats[lbl]["size"]) for lbl in members)
            aggregated_score = max(float(score_by_old.get(lbl, 0.0)) for lbl in members)
            grouped_scores[root] = aggregated_score
            if aggregated_size >= max(int(min_group_size), 1):
                grouped_labels.add(root)

        for old_label, score in score_by_old.items():
            if old_label not in component_group_map:
                grouped_scores[old_label] = float(score)

        return component_group_map, grouped_labels, grouped_scores

    def _compute_component_interface_graph(
        self,
        labels: Tensor,
        graph: GaussianGraph,
        broken_edge_mask: Tensor,
        edge_damage: Tensor,
    ) -> dict:
        adjacency = {}
        edge_rows, edge_cols = torch.where(broken_edge_mask)
        if edge_rows.numel() == 0:
            return adjacency

        src_labels = labels[edge_rows].tolist()
        dst_labels = labels[graph.knn_idx[edge_rows, edge_cols]].tolist()
        edge_strength = edge_damage[edge_rows, edge_cols].tolist()
        for src, dst, strength in zip(src_labels, dst_labels, edge_strength):
            if src == dst:
                continue
            a, b = (src, dst) if src < dst else (dst, src)
            adjacency.setdefault(a, {})
            adjacency.setdefault(b, {})
            count_a, max_a = adjacency[a].get(b, (0, 0.0))
            count_b, max_b = adjacency[b].get(a, (0, 0.0))
            updated = (count_a + 1, max(max_a, float(strength)))
            adjacency[a][b] = updated
            adjacency[b][a] = updated
        return adjacency

    def _compute_component_stats(
        self,
        labels: Tensor,
        positions: Optional[Tensor],
        authoritative_cut_mask: Optional[Tensor],
        boundary_ratio_by_old: dict,
    ) -> dict:
        stats = {}
        if positions is None:
            for old_label in labels.unique().tolist():
                size = int((labels == old_label).sum().item())
                stats[old_label] = {
                    "size": size,
                    "com": None,
                    "mean_z": 0.0,
                    "anchor_ratio": 0.0,
                    "auth_overlap": 0.0,
                    "boundary_ratio": float(boundary_ratio_by_old.get(old_label, 0.0)),
                }
            return stats

        z = positions[:, 2]
        z_min = float(z.min().item())
        z_max = float(z.max().item())
        height_scale = max(z_max - z_min, 1e-6)
        anchor_quantile = self.support_anchor_quantile
        if self.material_family == "rough_quasi_brittle":
            anchor_quantile = max(anchor_quantile, 0.14)
        elif self.material_family == "sharp_brittle":
            anchor_quantile = min(anchor_quantile, 0.08)
        anchor_z = float(torch.quantile(z.detach(), anchor_quantile).item()) + 0.02 * height_scale
        support_anchor_mask = z <= anchor_z
        auth_mask = authoritative_cut_mask
        if auth_mask is None:
            auth_mask = torch.zeros_like(labels, dtype=torch.bool)

        for old_label in labels.unique().tolist():
            mask = labels == old_label
            size = int(mask.sum().item())
            if size <= 0:
                continue
            stats[old_label] = {
                "size": size,
                "com": positions[mask].mean(dim=0),
                "mean_z": float(z[mask].mean().item()),
                "anchor_ratio": float((mask & support_anchor_mask).sum().item()) / max(size, 1),
                "auth_overlap": float((mask & auth_mask).sum().item()) / max(size, 1),
                "boundary_ratio": float(boundary_ratio_by_old.get(old_label, 0.0)),
            }
        return stats

    def _absorb_release_neighbors(
        self,
        labels: Tensor,
        positions: Optional[Tensor],
        main_label: Optional[int],
        component_group_map: dict,
        primary_group_ids: set,
        component_stats: dict,
        interface_graph: dict,
    ) -> Tuple[dict, int]:
        if positions is None or not primary_group_ids or not component_stats:
            return component_group_map, 0
        if self.material_family not in {"rough_quasi_brittle", "brittle_moderate"}:
            return component_group_map, 0

        bbox_extent = positions.max(dim=0).values - positions.min(dim=0).values
        diag = float(bbox_extent.norm().item())
        z = positions[:, 2]
        height_scale = max(float(z.max().item() - z.min().item()), 1e-6)

        absorb_radius = 0.16 * max(diag, 1e-6)
        absorb_height_delta = 0.22 * height_scale
        absorb_min_interface_edges = 3
        absorb_boundary_ratio = 0.08
        absorb_auth_ratio = 0.04
        absorb_anchor_ratio = 0.06
        absorb_max_component_size = 80
        if self.material_family == "brittle_moderate":
            absorb_radius *= 0.68
            absorb_height_delta *= 0.85
            absorb_min_interface_edges = 2
            absorb_boundary_ratio = 0.10
            absorb_auth_ratio = 0.05
            absorb_anchor_ratio = 0.03
            absorb_max_component_size = 24

        members_by_group = {group_id: set() for group_id in primary_group_ids}
        for old_label in labels.unique().tolist():
            group_id = component_group_map.get(old_label, old_label)
            if group_id in primary_group_ids:
                members_by_group[group_id].add(old_label)
        claimed = {old_label for members in members_by_group.values() for old_label in members}
        absorbed_count = 0

        group_order = sorted(
            primary_group_ids,
            key=lambda gid: -sum(component_stats.get(lbl, {}).get("size", 0) for lbl in members_by_group.get(gid, (gid,))),
        )
        for group_id in group_order:
            changed = True
            while changed:
                changed = False
                members = members_by_group.get(group_id, set())
                if not members:
                    continue
                total_size = sum(component_stats[lbl]["size"] for lbl in members if lbl in component_stats)
                if total_size <= 0:
                    continue
                weighted_com = sum(
                    component_stats[lbl]["com"] * float(component_stats[lbl]["size"])
                    for lbl in members
                    if component_stats[lbl]["com"] is not None
                ) / float(total_size)
                weighted_mean_z = sum(
                    component_stats[lbl]["mean_z"] * float(component_stats[lbl]["size"])
                    for lbl in members
                ) / float(total_size)

                candidate_neighbors = set()
                for lbl in list(members):
                    candidate_neighbors.update(interface_graph.get(lbl, {}).keys())

                for nb in candidate_neighbors:
                    if nb == main_label or nb in members or nb in claimed:
                        continue
                    if nb not in component_stats:
                        continue
                    stat = component_stats[nb]
                    if stat["size"] > absorb_max_component_size:
                        continue
                    if stat["anchor_ratio"] > absorb_anchor_ratio:
                        continue
                    if stat["boundary_ratio"] < absorb_boundary_ratio and stat["auth_overlap"] < absorb_auth_ratio:
                        continue
                    if abs(stat["mean_z"] - weighted_mean_z) > absorb_height_delta:
                        continue
                    if stat["com"] is None:
                        continue
                    dist = float((stat["com"] - weighted_com).norm().item())
                    if dist > absorb_radius:
                        continue

                    strong_interface = False
                    for member in members:
                        edge_info = interface_graph.get(member, {}).get(nb, None)
                        if edge_info is None:
                            continue
                        interface_count, interface_strength = edge_info
                        if (
                            interface_count >= absorb_min_interface_edges
                            or interface_strength >= self.fallback_cut_ratio
                        ):
                            strong_interface = True
                            break
                    if not strong_interface:
                        continue

                    component_group_map[nb] = group_id
                    members_by_group[group_id].add(nb)
                    claimed.add(nb)
                    absorbed_count += 1
                    changed = True

        return component_group_map, absorbed_count

    def _compute_authoritative_cut_score(
        self,
        graph: GaussianGraph,
        damage: Tensor,
        opening: Optional[Tensor],
        cut_vote: Optional[Tensor],
        cut_core_mask: Optional[Tensor],
        cut_edge_mask: Optional[Tensor],
    ) -> Tensor:
        score = torch.zeros_like(damage)
        if cut_vote is None or cut_core_mask is None:
            return score

        edge_support = cut_vote.max(dim=1).values
        edge_support = torch.maximum(edge_support, graph.weighted_neighbor_max(edge_support))
        node_cut_density = torch.zeros_like(damage)
        if cut_edge_mask is not None:
            node_cut_density = cut_edge_mask.float().mean(dim=1)
        damage_score = damage.clamp(0.0, 1.0)
        if opening is not None:
            opening_scale = torch.quantile(opening.detach(), 0.85).clamp(min=1e-8)
            opening_score = (opening / opening_scale).clamp(0.0, 1.0)
            damage_score = torch.maximum(damage_score, 0.55 * damage_score + 0.45 * opening_score)

        score = torch.maximum(
            0.54 * damage_score * cut_core_mask.float(),
            0.72 * edge_support + 0.20 * node_cut_density,
        )
        if self.material_family == "rough_quasi_brittle":
            score = torch.maximum(score, 0.64 * graph.weighted_neighbor_max(score))
        elif self.material_family == "sharp_brittle":
            score = torch.maximum(score, 0.52 * graph.weighted_neighbor_max(score))
        elif self.material_family == "brittle_moderate":
            score = torch.maximum(score, 0.42 * graph.weighted_neighbor_max(score))
        return score.clamp(0.0, 1.0)

    def _compute_support_loss_candidates(
        self,
        graph: GaussianGraph,
        labels: Tensor,
        positions: Tensor,
        label_sizes: List[int],
        authoritative_cut_mask: Optional[Tensor],
        cut_core_mask: Optional[Tensor],
        cut_vote: Optional[Tensor],
        hard_cut: Optional[Tensor],
    ) -> Tuple[dict, dict, set, Tensor]:
        release_score_by_old = {}
        support_score_by_old = {}
        support_lost_labels = set()
        support_lost_mask = torch.zeros_like(labels, dtype=torch.bool)
        if positions is None or labels.numel() == 0:
            return release_score_by_old, support_score_by_old, support_lost_labels, support_lost_mask

        z = positions[:, 2]
        z_min = float(z.min().item())
        z_max = float(z.max().item())
        height_scale = max(z_max - z_min, 1e-6)
        anchor_quantile = self.support_anchor_quantile
        if self.material_family == "rough_quasi_brittle":
            anchor_quantile = max(anchor_quantile, 0.14)
        elif self.material_family == "sharp_brittle":
            anchor_quantile = min(anchor_quantile, 0.08)
        anchor_z = float(torch.quantile(z.detach(), anchor_quantile).item()) + 0.02 * height_scale
        support_anchor_mask = z <= anchor_z
        self.last_support_anchor_nodes = int(support_anchor_mask.sum().item())

        label_i = labels.unsqueeze(1).expand_as(graph.knn_idx)
        label_j = labels[graph.knn_idx]
        cross_component = label_i != label_j
        if authoritative_cut_mask is None:
            authoritative_cut_mask = torch.zeros_like(labels, dtype=torch.bool)
        if cut_core_mask is None:
            cut_core_mask = torch.zeros_like(labels, dtype=torch.bool)

        unique_labels = labels.unique().tolist()
        if not unique_labels:
            return release_score_by_old, support_score_by_old, support_lost_labels, support_lost_mask
        main_label = max(unique_labels, key=lambda lbl: int((labels == lbl).sum().item()))

        base_release_thresh = self.support_release_threshold
        min_raw_candidate_size = 2
        if self.material_family == "sharp_brittle":
            base_release_thresh *= 0.86
            min_raw_candidate_size = 2
        elif self.material_family == "rough_quasi_brittle":
            base_release_thresh *= 0.92
            min_raw_candidate_size = 3
        elif self.material_family == "brittle_moderate":
            base_release_thresh *= 0.96
            min_raw_candidate_size = 4

        detached_overlap_mask = None
        if self.detached_node_memory is not None:
            detached_overlap_mask = self.detached_node_memory > 0.20

        for old_label in unique_labels:
            mask = labels == old_label
            size = int(mask.sum().item())
            if size <= 0:
                continue

            anchor_ratio = float((mask & support_anchor_mask).sum().item()) / max(size, 1)
            supported = anchor_ratio >= 0.02
            support_score = 1.0 if supported else 0.0
            support_score_by_old[old_label] = support_score

            boundary_mask = cross_component & mask.unsqueeze(1)
            boundary_cut = 0.0
            boundary_hard = 0.0
            if bool(boundary_mask.any()) and cut_vote is not None:
                boundary_cut = float(cut_vote[boundary_mask].mean().item())
            if bool(boundary_mask.any()) and hard_cut is not None:
                boundary_hard = float(hard_cut[boundary_mask].float().mean().item())

            auth_overlap = float((mask & authoritative_cut_mask).sum().item()) / max(size, 1)
            core_overlap = float((mask & cut_core_mask).sum().item()) / max(size, 1)
            comp_z = float(z[mask].mean().item())
            height_score = max(0.0, min(1.0, (comp_z - anchor_z) / max(0.45 * height_scale, 1e-6)))
            detached_overlap = 0.0
            if detached_overlap_mask is not None:
                detached_overlap = float((mask & detached_overlap_mask).sum().item()) / max(size, 1)

            cut_support = max(boundary_cut, boundary_hard, auth_overlap, core_overlap)
            release_score = (
                0.48 * (1.0 - support_score)
                + 0.34 * cut_support
                + 0.18 * height_score
                + 0.18 * detached_overlap
            )
            release_score = max(0.0, min(1.0, release_score))
            release_score_by_old[old_label] = release_score

            if old_label == main_label:
                continue
            if supported:
                continue
            if size < min_raw_candidate_size:
                continue
            if cut_support < self.support_overlap_threshold:
                continue
            if release_score >= base_release_thresh:
                support_lost_labels.add(old_label)
                support_lost_mask[mask] = True

        return release_score_by_old, support_score_by_old, support_lost_labels, support_lost_mask

    def _cluster_release_components(
        self,
        labels: Tensor,
        positions: Tensor,
        support_lost_labels: set,
        release_score_by_old: dict,
        support_score_by_old: dict,
        authoritative_cut_mask: Optional[Tensor],
    ) -> Tuple[dict, set, dict, dict]:
        if positions is None or not support_lost_labels:
            return {}, set(support_lost_labels), dict(release_score_by_old), dict(support_score_by_old)
        if self.material_family not in {"rough_quasi_brittle", "brittle_moderate"}:
            return {}, set(support_lost_labels), dict(release_score_by_old), dict(support_score_by_old)

        z = positions[:, 2]
        z_min = float(z.min().item())
        z_max = float(z.max().item())
        height_scale = max(z_max - z_min, 1e-6)
        bbox_extent = positions.max(dim=0).values - positions.min(dim=0).values
        diag = float(bbox_extent.norm().item())
        merge_radius = 0.09 * max(diag, 1e-6)
        if self.material_family == "brittle_moderate":
            merge_radius *= 0.72
        min_group_size = self.support_promote_min_size
        if self.material_family == "rough_quasi_brittle":
            min_group_size = max(12, min_group_size)
        elif self.material_family == "sharp_brittle":
            min_group_size = max(3, min_group_size - 2)
        elif self.material_family == "brittle_moderate":
            min_group_size = max(8, min_group_size)

        candidate_labels = sorted(support_lost_labels)
        if len(candidate_labels) <= 1:
            return {}, set(support_lost_labels), dict(release_score_by_old), dict(support_score_by_old)

        stats = {}
        for old_label in candidate_labels:
            mask = labels == old_label
            if not bool(mask.any()):
                continue
            com = positions[mask].mean(dim=0)
            mean_z = float(z[mask].mean().item())
            auth_overlap = 0.0
            if authoritative_cut_mask is not None:
                auth_overlap = float((mask & authoritative_cut_mask).sum().item()) / max(int(mask.sum().item()), 1)
            stats[old_label] = {
                "com": com,
                "mean_z": mean_z,
                "auth_overlap": auth_overlap,
                "size": int(mask.sum().item()),
            }
        if len(stats) <= 1:
            return {}, set(support_lost_labels), dict(release_score_by_old), dict(support_score_by_old)

        parent = {lbl: lbl for lbl in stats}

        def find(lbl):
            while parent[lbl] != lbl:
                parent[lbl] = parent[parent[lbl]]
                lbl = parent[lbl]
            return lbl

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # Keep the larger component as the representative.
            if stats[ra]["size"] < stats[rb]["size"]:
                ra, rb = rb, ra
            parent[rb] = ra

        label_list = list(stats.keys())
        for i, a in enumerate(label_list):
            for b in label_list[i + 1:]:
                sa = stats[a]
                sb = stats[b]
                dist = float((sa["com"] - sb["com"]).norm().item())
                if dist > merge_radius:
                    continue
                if abs(sa["mean_z"] - sb["mean_z"]) > 0.18 * height_scale:
                    continue
                auth_gate = max(sa["auth_overlap"], sb["auth_overlap"])
                if auth_gate < 0.04 and min(
                    release_score_by_old.get(a, 0.0),
                    release_score_by_old.get(b, 0.0),
                ) < 0.72:
                    continue
                union(a, b)

        component_group_map = {}
        grouped_labels = set()
        grouped_release_scores = {}
        grouped_support_scores = {}
        members_by_group = {}
        for old_label in stats:
            root = find(old_label)
            component_group_map[old_label] = root
            members_by_group.setdefault(root, []).append(old_label)

        for root, members in members_by_group.items():
            aggregated_size = sum(int(stats[lbl]["size"]) for lbl in members)
            aggregated_release = max(
                float(release_score_by_old.get(lbl, 0.0)) for lbl in members
            )
            aggregated_support = min(
                float(support_score_by_old.get(lbl, 1.0)) for lbl in members
            )
            grouped_release_scores[root] = aggregated_release
            grouped_support_scores[root] = aggregated_support
            if aggregated_size >= min_group_size:
                grouped_labels.add(root)

        for old_label, score in release_score_by_old.items():
            if old_label not in component_group_map:
                grouped_release_scores[old_label] = float(score)
                grouped_support_scores[old_label] = float(support_score_by_old.get(old_label, 1.0))

        return component_group_map, grouped_labels, grouped_release_scores, grouped_support_scores

    def _compute_cut_surface_votes(
        self,
        graph: GaussianGraph,
        positions: Optional[Tensor],
        damage: Tensor,
        opening: Optional[Tensor],
        active_tip_mask: Optional[Tensor],
        recent_front_mask: Optional[Tensor],
        crack_normal: Optional[Tensor],
        crack_tangent: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        if positions is None or opening is None or crack_normal is None:
            return None, None, None, None
        if graph.knn_idx is None or graph._normals is None:
            return None, None, None, None
        if positions.shape[0] != damage.shape[0]:
            return None, None, None, None

        recent_mask = torch.zeros_like(damage, dtype=torch.bool)
        if recent_front_mask is not None:
            recent_mask |= recent_front_mask
        if active_tip_mask is not None:
            recent_mask |= active_tip_mask
        if not bool(recent_mask.any()):
            return None, None, None, None

        opening_scale = torch.quantile(opening.detach(), 0.85).clamp(min=1e-8)
        opening_norm = (opening / opening_scale).clamp(0.0, 1.0)
        damage_core = damage >= self.cut_core_damage_threshold
        opening_core = opening_norm >= self.cut_core_opening_threshold
        cut_core_mask = recent_mask & damage_core & opening_core
        if not bool(cut_core_mask.any()) and self.material_family == "sharp_brittle":
            cut_core_mask = (
                recent_mask
                & (damage >= 0.75 * self.cut_core_damage_threshold)
                & (opening_norm >= 0.55 * self.cut_core_opening_threshold)
            )
        if not bool(cut_core_mask.any()):
            return None, None, None, None

        damage_score = (
            (damage - self.cut_core_damage_threshold)
            / max(1.0 - self.cut_core_damage_threshold, 1e-6)
        ).clamp(0.0, 1.0)
        opening_score = (
            (opening_norm - self.cut_core_opening_threshold)
            / max(1.0 - self.cut_core_opening_threshold, 1e-6)
        ).clamp(0.0, 1.0)
        cut_core_score = (0.55 * damage_score + 0.45 * opening_score) * cut_core_mask.float()
        support_field = cut_core_score.clone()
        neighbor_support = graph.weighted_neighbor_max(cut_core_score)
        support_min = 0.0
        if self.material_family == "sharp_brittle":
            support_field = torch.maximum(support_field, 0.90 * neighbor_support)
            second_ring = graph.weighted_neighbor_max(support_field)
            support_field = torch.maximum(support_field, 0.72 * second_ring)
            support_min = 0.10
        elif self.material_family == "brittle_moderate":
            support_field = torch.maximum(support_field, 0.76 * neighbor_support)
            second_ring = graph.weighted_neighbor_max(support_field)
            support_field = torch.maximum(support_field, 0.56 * second_ring)
            support_min = 0.12
        elif self.material_family == "rough_quasi_brittle":
            support_field = torch.maximum(support_field, 0.82 * neighbor_support)
            second_ring = graph.weighted_neighbor_max(support_field)
            support_field = torch.maximum(support_field, 0.62 * second_ring)
            support_min = 0.08

        edge_vec = positions[graph.knn_idx] - positions.unsqueeze(1)
        edge_unit = self._normalize_vectors(edge_vec)

        cut_dir = self._normalize_vectors(crack_normal)
        surf_normal = self._normalize_vectors(graph._normals)
        tangent_from_surface = self._normalize_vectors(
            torch.cross(surf_normal, cut_dir, dim=1)
        )
        if crack_tangent is not None and crack_tangent.shape == cut_dir.shape:
            tangent_hint = self._normalize_vectors(crack_tangent)
            tangent_valid = tangent_hint.norm(dim=1) > 1e-5
            tangent_dir = tangent_from_surface.clone()
            tangent_dir[tangent_valid] = tangent_hint[tangent_valid]
        else:
            tangent_dir = tangent_from_surface

        core_i = support_field.unsqueeze(1)
        core_j = support_field[graph.knn_idx]
        cut_pair = (core_i > support_min) | (core_j > support_min)
        if not bool(cut_pair.any()):
            return None, None, cut_core_mask, None

        nbr_support = torch.where(
            core_j > support_min,
            core_j,
            torch.zeros_like(core_j),
        )
        self_support = torch.where(
            support_field > support_min,
            support_field,
            torch.zeros_like(support_field),
        ).unsqueeze(1)
        nbr_weight_sum = nbr_support.sum(dim=1, keepdim=True)
        total_support = (self_support + nbr_weight_sum).clamp(min=1e-6)
        local_center = (
            positions * self_support
            + (positions[graph.knn_idx] * nbr_support.unsqueeze(2)).sum(dim=1)
        ) / total_support

        cut_i = cut_dir.unsqueeze(1).expand_as(edge_unit)
        cut_j = cut_dir[graph.knn_idx]
        tan_i = tangent_dir.unsqueeze(1).expand_as(edge_unit)
        tan_j = tangent_dir[graph.knn_idx]
        use_i = core_i >= core_j
        chosen_cut = torch.where(use_i.unsqueeze(2), cut_i, cut_j)
        chosen_tangent = torch.where(use_i.unsqueeze(2), tan_i, tan_j)
        chosen_cut = self._normalize_vectors(chosen_cut)
        chosen_tangent = self._normalize_vectors(chosen_tangent)

        cross_align = (edge_unit * chosen_cut).sum(dim=2).abs()
        tangent_align = (edge_unit * chosen_tangent).sum(dim=2).abs()
        cross_score = (
            (cross_align - self.tau_cross)
            / max(1.0 - self.tau_cross, 1e-6)
        ).clamp(0.0, 1.0)
        tangent_gate = (
            (self.tau_tangent - tangent_align)
            / max(self.tau_tangent, 1e-6)
        ).clamp(0.0, 1.0)
        core_score = torch.maximum(core_i, core_j)
        side_score = torch.zeros_like(core_score)
        side_weight = 0.0
        if self.material_family == "sharp_brittle":
            side_weight = 1.0
        elif self.material_family == "brittle_moderate":
            side_weight = 0.55
        elif self.material_family == "rough_quasi_brittle":
            side_weight = 0.35
        if side_weight > 0.0:
            center_i = local_center.unsqueeze(1).expand_as(edge_unit)
            center_j = local_center[graph.knn_idx]
            chosen_center = torch.where(use_i.unsqueeze(2), center_i, center_j)
            pos_i = positions.unsqueeze(1).expand_as(edge_unit)
            pos_j = positions[graph.knn_idx]
            signed_i = ((pos_i - chosen_center) * chosen_cut).sum(dim=2)
            signed_j = ((pos_j - chosen_center) * chosen_cut).sum(dim=2)
            edge_len = edge_vec.norm(dim=2).clamp(min=1e-6)
            side_margin = 0.12 * edge_len
            opposite_side = (signed_i * signed_j) < -(side_margin ** 2)
            side_sep = ((signed_i - signed_j).abs() / edge_len).clamp(0.0, 1.0)
            side_score = (
                side_weight
                * opposite_side.float()
                * side_sep
                * core_score.clamp(0.0, 1.0)
            )
        cut_vote = (
            self.cut_vote_strength
            * core_score
            * cross_score
            * tangent_gate
            * cut_pair.float()
        ).clamp(0.0, 1.0)
        if side_weight > 0.0:
            side_boost = (
                0.46
                * self.cut_vote_strength
                * cross_score
                * torch.sqrt(tangent_gate.clamp(min=0.0))
                * side_score
                * cut_pair.float()
            )
            cut_vote = (cut_vote + side_boost).clamp(0.0, 1.0)
        cut_edge_threshold = 0.05
        if self.material_family == "sharp_brittle":
            cut_edge_threshold = 0.07
        elif self.material_family == "brittle_moderate":
            cut_edge_threshold = 0.09
        cut_edge_mask = cut_vote > cut_edge_threshold
        hard_cut = cut_vote > self.cut_hard_break_threshold
        if self.material_family == "sharp_brittle":
            hard_cut = hard_cut | (
                cut_pair
                & (cross_align > min(self.tau_cross + 0.10, 0.98))
                & (tangent_align < 0.85 * self.tau_tangent)
                & (core_score > 0.32)
            )
            hard_cut = hard_cut | (
                cut_pair
                & (side_score > 0.10)
                & (cross_align > max(self.tau_cross - 0.04, 0.34))
                & (tangent_align < min(1.08 * self.tau_tangent, 0.52))
                & (core_score > 0.22)
            )
        elif self.material_family == "brittle_moderate":
            hard_cut = hard_cut | (
                cut_pair
                & (cross_align > min(self.tau_cross + 0.06, 0.96))
                & (tangent_align < 0.92 * self.tau_tangent)
                & (core_score > 0.40)
            )
            hard_cut = hard_cut | (
                cut_pair
                & (side_score > 0.08)
                & (cross_align > max(self.tau_cross - 0.03, 0.38))
                & (tangent_align < min(1.02 * self.tau_tangent, 0.48))
                & (core_score > 0.30)
            )
        elif self.material_family == "rough_quasi_brittle":
            hard_cut = hard_cut | (
                cut_pair
                & (cross_align > min(self.tau_cross + 0.04, 0.94))
                & (tangent_align < 1.02 * self.tau_tangent)
                & (core_score > 0.34)
            )
            hard_cut = hard_cut | (
                cut_pair
                & (side_score > 0.06)
                & (cross_align > max(self.tau_cross - 0.06, 0.30))
                & (core_score > 0.26)
            )

        if self.material_family == "sharp_brittle":
            cut_edge_mask = cut_edge_mask | (side_score > 0.06)
        elif self.material_family == "brittle_moderate":
            cut_edge_mask = cut_edge_mask | (side_score > 0.08)
        elif self.material_family == "rough_quasi_brittle":
            cut_edge_mask = cut_edge_mask | (
                cut_pair
                & (cross_align > max(self.tau_cross - 0.06, 0.28))
                & (core_score > 0.18)
            )
            cut_edge_mask = cut_edge_mask | (side_score > 0.10)
        cut_edge_mask = cut_edge_mask | hard_cut
        return cut_vote, hard_cut, cut_core_mask, cut_edge_mask


