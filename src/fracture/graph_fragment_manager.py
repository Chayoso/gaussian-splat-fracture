"""
Graph Fragment Manager

Phase 3: Detect material fragments using graph connectivity analysis.
Instead of grid-based connected components (scipy.ndimage.label),
operates directly on the Gaussian kNN graph with damage-weakened edges.

Fragments are connected components of the Gaussian graph where
edges with high damage are removed.
"""

import torch
from torch import Tensor
from typing import Dict, Optional, List, Tuple

from .graph_builder import GaussianGraph


class GraphFragmentManager:
    """
    Fragment detection via graph connectivity on the Gaussian manifold.

    Pipeline:
        1. Compute edge connectivity from damage field
        2. Remove edges where max(c_i, c_j) > threshold
        3. Find connected components via BFS/union-find
        4. Assign fragment labels to each Gaussian
        5. Compute per-fragment properties (COM, velocity, size)
    """

    def __init__(
        self,
        damage_threshold: float = 0.5,
        min_fragment_size: int = 20,
        edge_break_rate: float = 1.0,
        opening_weight: float = 0.35,
        active_tip_weight: float = 0.18,
        recent_front_weight: float = 0.12,
        pair_break_weight: float = 0.10,
        edge_memory_decay: float = 0.97,
        edge_memory_weight: float = 0.72,
        cut_diffusion_alpha: float = 0.0,
        cut_diffusion_iters: int = 0,
        cut_cos_gate_tangent: float = 0.5,
        cut_cos_gate_normal: float = 0.4,
        primary_cut_ratio: float = 0.75,
        fallback_cut_ratio: float = 0.55,
        min_boundary_edges: int = 12,
        detached_node_decay: float = 0.95,
        persistent_min_fragment_size: int = 8,
        component_hysteresis: float = 0.35,
        post_split_threshold_scale: float = 0.92,
        cut_surface_enable: bool = False,
        cut_vote_strength: float = 0.0,
        tau_cross: float = 0.60,
        tau_tangent: float = 0.45,
        cut_core_damage_threshold: float = 0.18,
        cut_core_opening_threshold: float = 0.16,
        cut_hard_break_threshold: float = 0.42,
        authoritative_cut_decay: float = 0.96,
        authoritative_cut_threshold: float = 0.20,
        support_loss_enable: bool = True,
        support_anchor_quantile: float = 0.10,
        support_release_threshold: float = 0.56,
        support_promote_min_size: int = 6,
        support_overlap_threshold: float = 0.10,
        material_family: str = "neutral_reference",
        device: str = "cuda",
    ):
        """
        Args:
            damage_threshold: edge damage above which connection breaks
            min_fragment_size: minimum Gaussians to count as fragment
            device: torch device
        """
        self.damage_threshold = damage_threshold
        self.min_fragment_size = min_fragment_size
        self.edge_break_rate = edge_break_rate
        self.opening_weight = opening_weight
        self.active_tip_weight = active_tip_weight
        self.recent_front_weight = recent_front_weight
        self.pair_break_weight = pair_break_weight
        self.edge_memory_decay = edge_memory_decay
        self.edge_memory_weight = edge_memory_weight
        self.cut_diffusion_alpha = float(cut_diffusion_alpha)
        self.cut_diffusion_iters = max(int(cut_diffusion_iters), 0)
        self.cut_cos_gate_tangent = float(cut_cos_gate_tangent)
        self.cut_cos_gate_normal = float(cut_cos_gate_normal)
        self.primary_cut_ratio = float(primary_cut_ratio)
        self.fallback_cut_ratio = float(fallback_cut_ratio)
        self.min_boundary_edges = max(int(min_boundary_edges), 0)
        self.detached_node_decay = detached_node_decay
        self.persistent_min_fragment_size = persistent_min_fragment_size
        self.component_hysteresis = component_hysteresis
        self.post_split_threshold_scale = post_split_threshold_scale
        self.cut_surface_enable = bool(cut_surface_enable)
        self.cut_vote_strength = float(cut_vote_strength)
        self.tau_cross = float(tau_cross)
        self.tau_tangent = float(tau_tangent)
        self.cut_core_damage_threshold = float(cut_core_damage_threshold)
        self.cut_core_opening_threshold = float(cut_core_opening_threshold)
        self.cut_hard_break_threshold = float(cut_hard_break_threshold)
        self.authoritative_cut_decay = float(authoritative_cut_decay)
        self.authoritative_cut_threshold = float(authoritative_cut_threshold)
        self.support_loss_enable = bool(support_loss_enable)
        self.support_anchor_quantile = float(support_anchor_quantile)
        self.support_release_threshold = float(support_release_threshold)
        self.support_promote_min_size = int(support_promote_min_size)
        self.support_overlap_threshold = float(support_overlap_threshold)
        self.material_family = str(material_family)
        self.device = torch.device(device)

        self.n_fragments: int = 0
        self.fragment_ids: Optional[Tensor] = None   # (N,) fragment label per Gaussian
        self.fragment_sizes: List[int] = []
        self.fragment_indices: List[Tensor] = []      # per-fragment Gaussian indices
        self.last_broken_edges: int = 0
        self.last_total_edges: int = 0
        self.last_mean_edge_damage: float = 0.0
        self.last_max_edge_damage: float = 0.0
        self.last_raw_components: int = 1
        self.last_damage_threshold: float = damage_threshold
        self.last_edge_break_rate: float = edge_break_rate
        self.last_edge_damage_break_threshold: float = 0.0
        self.last_promoted_components: int = 0
        self.last_primary_promoted_components: int = 0
        self.last_fallback_promoted_components: int = 0
        self.last_cut_core_nodes: int = 0
        self.last_cut_edges: int = 0
        self.last_cut_corridor_edges: int = 0
        self.last_cross_edge_breaks: int = 0
        self.last_cut_vote_max: float = 0.0
        self.last_top_component_sizes: List[int] = []
        self.last_authoritative_cut_nodes: int = 0
        self.last_authoritative_cut_score_max: float = 0.0
        self.last_support_lost_components: int = 0
        self.last_support_loss_score_max: float = 0.0
        self.last_release_candidate_count: int = 0
        self.last_support_anchor_nodes: int = 0
        self.last_boundary_cut_ratio_mean: float = 0.0
        self.last_boundary_cut_ratio_q50: float = 0.0
        self.last_boundary_cut_ratio_q90: float = 0.0
        self.last_boundary_cut_ratio_max: float = 0.0
        self.last_components_above_primary: int = 0
        self.last_components_above_fallback: int = 0
        self.last_absorbed_components: int = 0
        self.last_closure_candidate_count: int = 0
        self.last_closure_candidate_nodes: int = 0
        self.last_closure_score_max: float = 0.0
        self.last_closure_candidate_sizes: List[int] = []
        self.last_effective_edge_damage: Optional[Tensor] = None
        self.edge_cut_memory: Optional[Tensor] = None
        self.authoritative_cut_memory: Optional[Tensor] = None
        self.detached_node_memory: Optional[Tensor] = None
        self.last_cut_core_mask: Optional[Tensor] = None
        self.last_cut_edge_mask: Optional[Tensor] = None
        self.last_authoritative_cut_mask: Optional[Tensor] = None
        self.last_support_lost_mask: Optional[Tensor] = None
        self.last_closure_candidate_mask: Optional[Tensor] = None
        self.last_closure_boundary_mask: Optional[Tensor] = None
        self.has_split_once: bool = False
        self.fragment_release_scores: List[float] = []
        self.fragment_support_scores: List[float] = []
        self.fragment_support_lost: List[bool] = []

    def detect_fragments(
        self,
        graph: GaussianGraph,
        damage: Tensor,
        positions: Optional[Tensor] = None,
        opening: Optional[Tensor] = None,
        active_tip_mask: Optional[Tensor] = None,
        recent_front_mask: Optional[Tensor] = None,
        crack_normal: Optional[Tensor] = None,
        crack_tangent: Optional[Tensor] = None,
    ) -> int:
        """
        Detect fragments from damage-weakened graph.

        Args:
            graph: kNN graph on Gaussians
            damage: (N,) per-Gaussian damage values

        Returns:
            n_fragments: number of detected fragments
        """
        if self.material_family == "diffuse_damage":
            N = damage.shape[0]
            self.fragment_ids = torch.zeros(N, dtype=torch.long, device=self.device)
            self.fragment_sizes = [N]
            self.fragment_indices = [torch.arange(N, device=self.device)]
            self.n_fragments = 1
            self.last_broken_edges = 0
            self.last_total_edges = 0
            self.last_mean_edge_damage = 0.0
            self.last_max_edge_damage = 0.0
            self.last_raw_components = 1
            self.last_edge_break_rate = self.edge_break_rate
            self.last_edge_damage_break_threshold = 0.0
            self.last_promoted_components = 0
            self.last_primary_promoted_components = 0
            self.last_fallback_promoted_components = 0
            self.last_cut_core_nodes = 0
            self.last_cut_edges = 0
            self.last_cut_corridor_edges = 0
            self.last_cross_edge_breaks = 0
            self.last_cut_vote_max = 0.0
            self.last_top_component_sizes = [N]
            self.last_authoritative_cut_nodes = 0
            self.last_authoritative_cut_score_max = 0.0
            self.last_support_lost_components = 0
            self.last_support_loss_score_max = 0.0
            self.last_release_candidate_count = 0
            self.last_support_anchor_nodes = 0
            self.last_boundary_cut_ratio_mean = 0.0
            self.last_boundary_cut_ratio_q50 = 0.0
            self.last_boundary_cut_ratio_q90 = 0.0
            self.last_boundary_cut_ratio_max = 0.0
            self.last_components_above_primary = 0
            self.last_components_above_fallback = 0
            self.last_absorbed_components = 0
            self.last_closure_candidate_count = 0
            self.last_closure_candidate_nodes = 0
            self.last_closure_score_max = 0.0
            self.last_closure_candidate_sizes = []
            self.edge_cut_memory = None
            self.authoritative_cut_memory = None
            self.detached_node_memory = None
            self.last_effective_edge_damage = None
            self.last_cut_core_mask = None
            self.last_cut_edge_mask = None
            self.last_authoritative_cut_mask = None
            self.last_support_lost_mask = None
            self.last_closure_candidate_mask = None
            self.last_closure_boundary_mask = None
            self.has_split_once = False
            self.fragment_release_scores = [0.0]
            self.fragment_support_scores = [1.0]
            self.fragment_support_lost = [False]
            return 1

        N = damage.shape[0]
        if graph.knn_idx is None:
            self.fragment_ids = torch.zeros(N, dtype=torch.long, device=self.device)
            self.n_fragments = 1
            self.last_effective_edge_damage = None
            self.fragment_release_scores = [0.0]
            self.fragment_support_scores = [1.0]
            self.fragment_support_lost = [False]
            return 1

        if (self.edge_cut_memory is None
                or self.edge_cut_memory.shape != graph.knn_idx.shape):
            self.edge_cut_memory = torch.zeros(
                graph.knn_idx.shape,
                dtype=damage.dtype,
                device=self.device,
            )
        if (self.detached_node_memory is None
                or self.detached_node_memory.shape[0] != N):
            self.detached_node_memory = torch.zeros(
                N,
                dtype=damage.dtype,
                device=self.device,
            )
        if (self.authoritative_cut_memory is None
                or self.authoritative_cut_memory.shape[0] != N):
            self.authoritative_cut_memory = torch.zeros(
                N,
                dtype=damage.dtype,
                device=self.device,
            )
        self.last_cut_core_nodes = 0
        self.last_cut_edges = 0
        self.last_cross_edge_breaks = 0
        self.last_cut_vote_max = 0.0
        self.last_cut_core_mask = None
        self.last_cut_edge_mask = None
        self.last_effective_edge_damage = None
        self.last_top_component_sizes = []
        self.last_primary_promoted_components = 0
        self.last_fallback_promoted_components = 0
        self.last_authoritative_cut_nodes = 0
        self.last_authoritative_cut_score_max = 0.0
        self.last_support_lost_components = 0
        self.last_support_loss_score_max = 0.0
        self.last_release_candidate_count = 0
        self.last_support_anchor_nodes = 0
        self.last_cut_corridor_edges = 0
        self.last_boundary_cut_ratio_mean = 0.0
        self.last_boundary_cut_ratio_q50 = 0.0
        self.last_boundary_cut_ratio_q90 = 0.0
        self.last_boundary_cut_ratio_max = 0.0
        self.last_components_above_primary = 0
        self.last_components_above_fallback = 0
        self.last_absorbed_components = 0
        self.last_closure_candidate_count = 0
        self.last_closure_candidate_nodes = 0
        self.last_closure_score_max = 0.0
        self.last_closure_candidate_sizes = []
        self.last_authoritative_cut_mask = None
        self.last_support_lost_mask = None
        self.last_closure_candidate_mask = None
        self.last_closure_boundary_mask = None
        self.fragment_release_scores = []
        self.fragment_support_scores = []
        self.fragment_support_lost = []

        node_break = damage.clamp(0.0, 1.0)
        if opening is not None:
            opening = opening.clamp(min=0.0)
            opening_scale = torch.quantile(opening.detach(), 0.90).clamp(min=1e-8)
            opening_norm = (opening / opening_scale).clamp(0.0, 1.0)
            node_break = torch.maximum(node_break, (damage + self.opening_weight * opening_norm).clamp(0.0, 1.0))
        if active_tip_mask is not None:
            node_break = (node_break + self.active_tip_weight * active_tip_mask.float()).clamp(0.0, 1.0)
        if recent_front_mask is not None:
            node_break = (node_break + self.recent_front_weight * recent_front_mask.float()).clamp(0.0, 1.0)

        c_i = node_break.unsqueeze(1).expand_as(graph.knn_idx.float())
        c_j = node_break[graph.knn_idx]
        cut_vote = None
        hard_cut = None
        cut_core_mask = None
        raw_cut_edge_mask = None
        authoritative_cut_score = torch.zeros_like(damage)
        authoritative_cut_mask = torch.zeros_like(damage, dtype=torch.bool)
        if self._supports_cut_surface():
            cut_vote, hard_cut, cut_core_mask, cut_edge_mask = self._compute_cut_surface_votes(
                graph=graph,
                positions=positions,
                damage=damage,
                opening=opening,
                active_tip_mask=active_tip_mask,
                recent_front_mask=recent_front_mask,
                crack_normal=crack_normal,
                crack_tangent=crack_tangent,
            )
            if cut_vote is not None:
                self.last_cut_core_mask = cut_core_mask
                self.last_cut_core_nodes = int(cut_core_mask.sum().item())
                self.last_cut_vote_max = float(cut_vote.max().item())
                raw_cut_edge_mask = cut_edge_mask
                authoritative_cut_score = self._compute_authoritative_cut_score(
                    graph=graph,
                    damage=damage,
                    opening=opening,
                    cut_vote=cut_vote,
                    cut_core_mask=cut_core_mask,
                    cut_edge_mask=cut_edge_mask,
                )
        if self.authoritative_cut_memory is not None:
            self.authoritative_cut_memory = torch.maximum(
                self.authoritative_cut_memory * self.authoritative_cut_decay,
                authoritative_cut_score,
            )
            auth_thresh = self._authoritative_cut_threshold()
            authoritative_cut_mask = self.authoritative_cut_memory > auth_thresh
            self.last_authoritative_cut_nodes = int(authoritative_cut_mask.sum().item())
            self.last_authoritative_cut_score_max = float(self.authoritative_cut_memory.max().item())
            self.last_authoritative_cut_mask = authoritative_cut_mask.clone()

        seed_d_cut = torch.maximum(c_i, c_j)
        if active_tip_mask is not None:
            tip_i = active_tip_mask.unsqueeze(1).expand_as(seed_d_cut)
            tip_j = active_tip_mask[graph.knn_idx]
            seed_d_cut = (
                seed_d_cut
                + self.pair_break_weight * torch.maximum(tip_i.float(), tip_j.float())
            ).clamp(0.0, 1.0)
        if recent_front_mask is not None:
            recent_i = recent_front_mask.unsqueeze(1).expand_as(seed_d_cut)
            recent_j = recent_front_mask[graph.knn_idx]
            seed_d_cut = (
                seed_d_cut
                + 0.5 * self.pair_break_weight * torch.maximum(recent_i.float(), recent_j.float())
            ).clamp(0.0, 1.0)
        if cut_vote is not None:
            seed_d_cut = (seed_d_cut + cut_vote).clamp(0.0, 1.0)
        if self.authoritative_cut_memory is not None:
            auth_i = self.authoritative_cut_memory.unsqueeze(1).expand_as(seed_d_cut)
            auth_j = self.authoritative_cut_memory[graph.knn_idx]
            seed_d_cut = torch.maximum(seed_d_cut, 0.85 * torch.maximum(auth_i, auth_j))

        d_cut = self._diffuse_cut_corridor(
            graph=graph,
            seed_d_cut=seed_d_cut,
            crack_normal=crack_normal,
            crack_tangent=crack_tangent,
        )
        self.edge_cut_memory = torch.maximum(
            self.edge_cut_memory * self.edge_memory_decay,
            d_cut,
        )
        effective_cut_damage = torch.maximum(
            d_cut,
            self.edge_memory_weight * self.edge_cut_memory,
        ).clamp(0.0, 1.0)
        edge_break_rate = max(self.edge_break_rate, 1e-4)
        damage_threshold = self.damage_threshold
        effective_min_fragment_size = self.min_fragment_size
        if self.has_split_once:
            damage_threshold *= self.post_split_threshold_scale
        cut_break_threshold = self._break_threshold_from_edge_damage(
            damage_threshold=damage_threshold,
            edge_break_rate=edge_break_rate,
        )

        edge_alive = effective_cut_damage < cut_break_threshold
        if hard_cut is not None:
            edge_alive = edge_alive & (~hard_cut)
        corridor_edge_mask = effective_cut_damage >= cut_break_threshold
        if hard_cut is not None:
            corridor_edge_mask = corridor_edge_mask | hard_cut
        self.last_cut_edge_mask = corridor_edge_mask.clone()
        self.last_broken_edges = int((~edge_alive).sum().item())
        self.last_total_edges = int(edge_alive.numel())
        self.last_mean_edge_damage = float(effective_cut_damage.mean().item())
        self.last_max_edge_damage = float(effective_cut_damage.max().item())
        self.last_damage_threshold = float(damage_threshold)
        self.last_edge_break_rate = float(edge_break_rate)
        self.last_edge_damage_break_threshold = float(cut_break_threshold)
        self.last_effective_edge_damage = effective_cut_damage.detach().flatten().cpu()
        self.last_cut_edges = int(corridor_edge_mask.sum().item())
        self.last_cut_corridor_edges = self.last_cut_edges
        if raw_cut_edge_mask is not None:
            self.last_cross_edge_breaks = int((((~edge_alive) & raw_cut_edge_mask)).sum().item())

        # Union-Find on CPU (graph CC is inherently serial)
        knn_idx_cpu = graph.knn_idx.cpu()
        edge_alive_cpu = edge_alive.cpu()

        labels = self._union_find_cc(N, knn_idx_cpu, edge_alive_cpu)
        labels = torch.from_numpy(labels).to(self.device)

        # Remap to contiguous labels and filter small fragments
        unique_labels = labels.unique()
        self.last_raw_components = int(unique_labels.numel())
        label_sizes = [(labels == lbl).sum().item() for lbl in unique_labels]
        promoted_labels = set()
        main_label = None
        if unique_labels.numel() > 0:
            main_label = max(unique_labels.tolist(), key=lambda lbl: int((labels == lbl).sum().item()))
        boundary_cut_ratio_by_old, boundary_edges_by_old = self._compute_boundary_cut_stats(
            labels=labels,
            graph=graph,
            effective_cut_damage=effective_cut_damage,
            cut_threshold=cut_break_threshold,
        )
        boundary_candidate_labels = set()
        boundary_score_by_old = {}
        for old_label, size in zip(unique_labels.tolist(), label_sizes):
            if old_label == main_label:
                continue
            boundary_edges = int(boundary_edges_by_old.get(old_label, 0))
            cut_ratio = float(boundary_cut_ratio_by_old.get(old_label, 0.0))
            if boundary_edges < self.min_boundary_edges:
                continue
            if cut_ratio < self.fallback_cut_ratio:
                continue
            boundary_candidate_labels.add(old_label)
            boundary_score_by_old[old_label] = cut_ratio
        component_group_map, grouped_boundary_labels, grouped_boundary_scores = self._cluster_boundary_candidates(
            labels=labels,
            positions=positions,
            candidate_labels=boundary_candidate_labels,
            score_by_old=boundary_score_by_old,
            authoritative_cut_mask=authoritative_cut_mask,
            min_group_size=effective_min_fragment_size,
        )
        candidate_group_ids = {
            component_group_map.get(old_label, old_label)
            for old_label in boundary_candidate_labels
        }
        group_closure_scores = self._compute_group_closure_scores(
            labels=labels,
            positions=positions,
            graph=graph,
            corridor_edge_mask=corridor_edge_mask,
            component_group_map=component_group_map,
            group_ids=candidate_group_ids,
        )
        _, _, closure_debug_threshold = self._closure_params()
        primary_promoted_labels = set()
        fallback_candidate_labels = set()
        for group_id in grouped_boundary_labels:
            cut_ratio = float(grouped_boundary_scores.get(group_id, 0.0))
            if cut_ratio >= self.primary_cut_ratio:
                primary_promoted_labels.add(group_id)
            elif cut_ratio >= self.fallback_cut_ratio:
                fallback_candidate_labels.add(group_id)
        members_by_group = {}
        for old_label in unique_labels.tolist():
            group_id = component_group_map.get(old_label, old_label)
            members_by_group.setdefault(group_id, []).append(old_label)
        explicit_patches = self._extract_explicit_closure_patches(
            labels=labels,
            positions=positions,
            graph=graph,
            corridor_edge_mask=corridor_edge_mask,
            grouped_boundary_labels=candidate_group_ids,
            grouped_boundary_scores=grouped_boundary_scores,
            group_closure_scores=group_closure_scores,
            members_by_group=members_by_group,
        )
        closure_candidate_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        closure_boundary_mask = torch.zeros_like(corridor_edge_mask)
        closure_candidate_sizes = []
        closure_candidate_groups = {
            group_id for group_id in candidate_group_ids
            if float(group_closure_scores.get(group_id, 0.0)) >= closure_debug_threshold
        }
        for group_id in sorted(closure_candidate_groups):
            members = members_by_group.get(group_id, [])
            if not members:
                continue
            in_group = torch.zeros(N, dtype=torch.bool, device=self.device)
            group_size = 0
            for old_label in members:
                member_mask = labels == old_label
                in_group |= member_mask
                group_size += int(member_mask.sum().item())
            closure_candidate_mask |= in_group
            closure_boundary_mask |= (
                corridor_edge_mask
                & (in_group.unsqueeze(1) ^ in_group[graph.knn_idx])
            )
            closure_candidate_sizes.append(group_size)
        if explicit_patches:
            closure_candidate_mask.zero_()
            closure_boundary_mask.zero_()
            closure_candidate_sizes = []
            for patch in explicit_patches:
                closure_candidate_mask |= patch["mask"]
                closure_boundary_mask |= patch["boundary_mask"]
                closure_candidate_sizes.append(int(patch["size"]))
        self.last_closure_candidate_count = len(closure_candidate_sizes)
        self.last_closure_candidate_nodes = int(closure_candidate_mask.sum().item())
        self.last_closure_score_max = (
            max(float(group_closure_scores.get(group_id, 0.0)) for group_id in candidate_group_ids)
            if candidate_group_ids else 0.0
        )
        self.last_closure_candidate_sizes = sorted(closure_candidate_sizes, reverse=True)
        self.last_closure_candidate_mask = closure_candidate_mask.clone()
        self.last_closure_boundary_mask = closure_boundary_mask.clone()
        component_stats = self._compute_component_stats(
            labels=labels,
            positions=positions,
            authoritative_cut_mask=authoritative_cut_mask,
            boundary_ratio_by_old=boundary_cut_ratio_by_old,
        )
        interface_graph = self._compute_component_interface_graph(
            labels=labels,
            graph=graph,
            broken_edge_mask=~edge_alive,
            edge_damage=effective_cut_damage,
        )
        component_group_map, absorbed_count = self._absorb_release_neighbors(
            labels=labels,
            positions=positions,
            main_label=main_label,
            component_group_map=component_group_map,
            primary_group_ids=primary_promoted_labels,
            component_stats=component_stats,
            interface_graph=interface_graph,
        )
        self.last_absorbed_components = absorbed_count
        self.last_components_above_primary = len(primary_promoted_labels)
        self.last_components_above_fallback = len(primary_promoted_labels) + len(fallback_candidate_labels)
        promoted_labels.update(primary_promoted_labels)
        self.last_primary_promoted_components = len(primary_promoted_labels)
        release_score_by_old = {}
        support_score_by_old = {}
        support_lost_labels = set()
        support_lost_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        if self.support_loss_enable and positions is not None:
            (
                release_score_by_old,
                support_score_by_old,
                support_lost_labels,
                support_lost_mask,
            ) = self._compute_support_loss_candidates(
                graph=graph,
                labels=labels,
                positions=positions,
                label_sizes=label_sizes,
                authoritative_cut_mask=authoritative_cut_mask,
                cut_core_mask=cut_core_mask,
                cut_vote=cut_vote,
                hard_cut=hard_cut,
            )
            support_lost_group_ids = {
                component_group_map.get(old_label, old_label)
                for old_label in support_lost_labels
            }
            support_lost_labels = support_lost_group_ids & fallback_candidate_labels
            if support_lost_labels:
                filtered_support_lost_mask = torch.zeros_like(support_lost_mask)
                for old_label in labels.unique().tolist():
                    group_id = component_group_map.get(old_label, old_label)
                    if group_id in support_lost_labels:
                        filtered_support_lost_mask |= (labels == old_label)
                support_lost_mask = filtered_support_lost_mask
            else:
                support_lost_mask = torch.zeros_like(support_lost_mask)
        grouped_support_lost_labels = set(support_lost_labels)
        grouped_release_scores = {}
        grouped_support_scores = {}
        for old_label in unique_labels.tolist():
            group_id = component_group_map.get(old_label, old_label)
            grouped_release_scores[group_id] = max(
                grouped_release_scores.get(group_id, 0.0),
                float(boundary_score_by_old.get(old_label, release_score_by_old.get(old_label, 0.0))),
            )
            grouped_support_scores[group_id] = min(
                grouped_support_scores.get(group_id, 1.0),
                float(support_score_by_old.get(old_label, 1.0)),
            )
        promoted_labels.update(grouped_support_lost_labels)
        self.last_fallback_promoted_components = len(grouped_support_lost_labels)
        self.last_support_lost_components = len(grouped_support_lost_labels)
        self.last_release_candidate_count = (
            len(primary_promoted_labels)
            + len(grouped_support_lost_labels)
            + len(explicit_patches)
        )
        self.last_support_loss_score_max = (
            max(grouped_release_scores.values()) if grouped_release_scores else 0.0
        )
        explicit_support_mask = torch.zeros_like(support_lost_mask)
        for patch in explicit_patches:
            if bool(patch.get("support_lost", False)):
                explicit_support_mask |= patch["mask"]
        support_lost_mask = support_lost_mask | explicit_support_mask
        self.last_support_lost_components += int(sum(1 for patch in explicit_patches if bool(patch.get("support_lost", False))))
        self.last_support_lost_mask = support_lost_mask.clone()
        if self.has_split_once and self.detached_node_memory is not None:
            detached_mask = self.detached_node_memory > 0.25
            if bool(detached_mask.any()):
                overlap_labels = labels[detached_mask].unique()
                for old_label in overlap_labels.tolist():
                    mask = labels == old_label
                    size = int(mask.sum().item())
                    if size < self.persistent_min_fragment_size:
                        continue
                    overlap = int((mask & detached_mask).sum().item())
                    overlap_ratio = overlap / max(size, 1)
                    if overlap_ratio >= self.component_hysteresis:
                        promoted_labels.add(component_group_map.get(old_label, old_label))
        self.last_primary_promoted_components += len(explicit_patches)
        self.last_promoted_components = len(promoted_labels) + len(explicit_patches)

        # Sort by size (largest first)
        sorted_pairs = sorted(zip(unique_labels.tolist(), label_sizes),
                              key=lambda x: -x[1])
        self.last_top_component_sizes = [int(size) for _, size in sorted_pairs[:8]]
        grouped_sizes = {}
        for old_label, size in sorted_pairs:
            group_id = component_group_map.get(old_label, old_label)
            grouped_sizes[group_id] = grouped_sizes.get(group_id, 0) + int(size)

        # Remap
        new_labels = torch.zeros(N, dtype=torch.long, device=self.device)
        kept_pairs = []
        assigned_groups = {}
        for idx, (old_label, size) in enumerate(sorted_pairs):
            group_id = component_group_map.get(old_label, old_label)
            group_size = grouped_sizes.get(group_id, size)
            keep_component = (
                idx == 0
                or group_size >= effective_min_fragment_size
                or group_id in promoted_labels
            )
            if keep_component:
                if group_id not in assigned_groups:
                    kept_pairs.append((group_id, group_size))
                    assigned_groups[group_id] = len(kept_pairs) - 1
                new_labels[labels == old_label] = assigned_groups[group_id]

        explicit_meta = {}
        next_explicit_label = int(new_labels.max().item()) + 1 if new_labels.numel() > 0 else 1
        occupied_explicit = torch.zeros(N, dtype=torch.bool, device=self.device)
        for patch in sorted(explicit_patches, key=lambda item: -int(item["size"])):
            patch_mask = patch["mask"] & (~occupied_explicit)
            patch_size = int(patch_mask.sum().item())
            if patch_size < max(self.persistent_min_fragment_size, 1):
                continue
            new_labels[patch_mask] = next_explicit_label
            explicit_meta[next_explicit_label] = {
                "release_score": float(patch.get("release_score", 0.0)),
                "support_score": 0.0 if bool(patch.get("support_lost", False)) else 0.25,
                "support_lost": bool(patch.get("support_lost", False)),
            }
            occupied_explicit |= patch_mask
            next_explicit_label += 1

        self.fragment_ids = new_labels
        unique_new_labels = self.fragment_ids.unique(sorted=True)
        self.fragment_sizes = []
        self.fragment_indices = []
        self.fragment_release_scores = []
        self.fragment_support_scores = []
        self.fragment_support_lost = []
        for new_label in unique_new_labels.tolist():
            mask = self.fragment_ids == new_label
            size = int(mask.sum().item())
            if size <= 0:
                continue
            self.fragment_sizes.append(size)
            self.fragment_indices.append(torch.where(mask)[0])
            if new_label in explicit_meta:
                meta = explicit_meta[new_label]
                self.fragment_release_scores.append(float(meta["release_score"]))
                self.fragment_support_scores.append(float(meta["support_score"]))
                self.fragment_support_lost.append(bool(meta["support_lost"]))
            else:
                old_label = kept_pairs[new_label][0] if new_label < len(kept_pairs) else None
                self.fragment_release_scores.append(float(grouped_release_scores.get(old_label, 0.0)))
                self.fragment_support_scores.append(float(grouped_support_scores.get(old_label, 1.0)))
                self.fragment_support_lost.append(bool(old_label in grouped_support_lost_labels))

        self.n_fragments = len(self.fragment_indices) if self.fragment_indices else 1
        if self.n_fragments > 1:
            self.has_split_once = True
        if self.detached_node_memory is not None:
            current_detached = (
                (self.fragment_ids > 0).float()
                if self.n_fragments > 1 else
                torch.zeros_like(damage)
            )
            self.detached_node_memory = torch.maximum(
                self.detached_node_memory * self.detached_node_decay,
                current_detached,
            )

        if self.n_fragments > 1:
            print(f"[GraphFrag] Detected {self.n_fragments} fragments: "
                  f"sizes={self.fragment_sizes[:10]} "
                  f"promoted={self.last_promoted_components}")

        return self.n_fragments

    def _supports_cut_surface(self) -> bool:
        return self.cut_surface_enable and self.material_family in {
            "sharp_brittle",
            "brittle_moderate",
            "rough_quasi_brittle",
        }

    def _authoritative_cut_threshold(self) -> float:
        thresh = self.authoritative_cut_threshold
        if self.material_family == "sharp_brittle":
            thresh *= 0.82
        elif self.material_family == "rough_quasi_brittle":
            thresh *= 0.92
        elif self.material_family == "brittle_moderate":
            thresh *= 1.05
        return float(thresh)

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
    ) -> Tuple[dict, dict]:
        label_i = labels.unsqueeze(1).expand_as(graph.knn_idx)
        label_j = labels[graph.knn_idx]
        cross_component = label_i != label_j
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

    def _closure_params(self) -> Tuple[int, float, float]:
        if self.material_family == "sharp_brittle":
            return 18, 0.22, 0.58
        if self.material_family == "brittle_moderate":
            return 20, 0.20, 0.60
        if self.material_family == "rough_quasi_brittle":
            return 16, 0.16, 0.52
        return 16, 0.18, 0.58

    def _explicit_closure_params(self) -> Tuple[float, int, float, float]:
        if self.material_family == "sharp_brittle":
            return 0.68, max(28, self.min_fragment_size), 1.04, 0.040
        if self.material_family == "brittle_moderate":
            return 0.72, max(40, self.min_fragment_size), 1.08, 0.050
        if self.material_family == "rough_quasi_brittle":
            return 0.70, max(72, 2 * self.min_fragment_size), 1.12, 0.060
        return 0.72, max(48, self.min_fragment_size), 1.08, 0.050

    def _compute_group_closure_scores(
        self,
        labels: Tensor,
        positions: Optional[Tensor],
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        component_group_map: dict,
        group_ids: set,
    ) -> dict:
        if positions is None or graph.knn_idx is None or not group_ids:
            return {}

        angle_bins, compactness_target, _ = self._closure_params()
        two_pi = float(2.0 * torch.pi)
        scores = {}

        for group_id in sorted(group_ids):
            in_group = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
            group_size = 0
            for old_label in labels.unique().tolist():
                if component_group_map.get(old_label, old_label) != group_id:
                    continue
                mask = labels == old_label
                in_group |= mask
                group_size += int(mask.sum().item())
            if group_size <= 0:
                scores[group_id] = 0.0
                continue

            boundary_mask = corridor_edge_mask & (in_group.unsqueeze(1) ^ in_group[graph.knn_idx])
            edge_rows, edge_cols = torch.where(boundary_mask)
            if edge_rows.numel() < max(self.min_boundary_edges, 8):
                scores[group_id] = 0.0
                continue

            nbr_idx = graph.knn_idx[edge_rows, edge_cols]
            mids = 0.5 * (positions[edge_rows] + positions[nbr_idx])
            center = positions[in_group].mean(dim=0, keepdim=True)
            centered = mids - center
            if centered.shape[0] < 6:
                scores[group_id] = 0.0
                continue

            cov = centered.T @ centered / float(max(centered.shape[0] - 1, 1))
            try:
                _, eigvecs = torch.linalg.eigh(cov)
            except RuntimeError:
                scores[group_id] = 0.0
                continue
            basis = eigvecs[:, -2:]
            uv = centered @ basis
            radii = uv.norm(dim=1)
            mean_radius = float(radii.mean().item())
            if mean_radius < 1e-6:
                scores[group_id] = 0.0
                continue

            valid = radii > (0.15 * mean_radius)
            if int(valid.sum().item()) < 6:
                scores[group_id] = 0.0
                continue
            uv = uv[valid]
            radii = radii[valid]
            angles = torch.atan2(uv[:, 1], uv[:, 0])

            bin_pos = ((angles + torch.pi) / (2.0 * torch.pi) * angle_bins).floor().long()
            bin_pos = bin_pos.clamp(0, angle_bins - 1)
            occupied = torch.bincount(bin_pos, minlength=angle_bins) > 0
            angular_coverage = float(occupied.float().mean().item())

            sorted_angles = torch.sort(angles).values
            wrapped = torch.cat([sorted_angles, sorted_angles[:1] + two_pi])
            gaps = wrapped[1:] - wrapped[:-1]
            max_gap = float(gaps.max().item()) if gaps.numel() > 0 else two_pi
            gap_score = max(0.0, min(1.0, 1.0 - max_gap / two_pi))

            compactness = float(group_size) / float(max(int(edge_rows.numel()), 1))
            compactness_score = max(0.0, min(1.0, compactness / max(compactness_target, 1e-6)))

            radial_cv = float(radii.std(unbiased=False).item()) / max(float(radii.mean().item()), 1e-6)
            radial_score = max(0.0, min(1.0, 1.0 - radial_cv / 0.85))

            closure_score = (
                0.42 * angular_coverage
                + 0.33 * gap_score
                + 0.17 * compactness_score
                + 0.08 * radial_score
            )
            scores[group_id] = float(max(0.0, min(1.0, closure_score)))

        return scores

    def _build_explicit_patch_from_group(
        self,
        labels: Tensor,
        positions: Tensor,
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        members: List[int],
        min_patch_size: int,
        inflate_scale: float,
        slab_scale: float,
    ) -> Optional[dict]:
        if graph.knn_idx is None or positions is None or not members:
            return None

        seed_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        for old_label in members:
            seed_mask |= (labels == old_label)
        seed_size = int(seed_mask.sum().item())
        if seed_size < max(self.min_boundary_edges, 8):
            return None

        boundary_mask = corridor_edge_mask & (seed_mask.unsqueeze(1) ^ seed_mask[graph.knn_idx])
        edge_rows, edge_cols = torch.where(boundary_mask)
        if edge_rows.numel() < max(self.min_boundary_edges, 10):
            return None

        nbr_idx = graph.knn_idx[edge_rows, edge_cols]
        boundary_mids = 0.5 * (positions[edge_rows] + positions[nbr_idx])
        boundary_nodes = torch.unique(torch.cat([edge_rows, nbr_idx], dim=0))
        if boundary_mids.shape[0] < 8:
            return None

        center = boundary_mids.mean(dim=0)
        centered = boundary_mids - center.unsqueeze(0)
        cov = centered.T @ centered / float(max(centered.shape[0] - 1, 1))
        try:
            _, eigvecs = torch.linalg.eigh(cov)
        except RuntimeError:
            return None
        basis = eigvecs[:, -2:]
        plane_normal = eigvecs[:, 0]
        uv_boundary = centered @ basis
        radii = uv_boundary.norm(dim=1)
        mean_radius = float(radii.mean().item())
        if mean_radius < 1e-5:
            return None

        angle_bins = max(self._closure_params()[0] * 2, 24)
        angles = torch.atan2(uv_boundary[:, 1], uv_boundary[:, 0])
        bin_pos = ((angles + torch.pi) / (2.0 * torch.pi) * angle_bins).floor().long()
        bin_pos = bin_pos.clamp(0, angle_bins - 1)

        radius_bins = torch.zeros(angle_bins, dtype=positions.dtype, device=positions.device)
        occupied = torch.zeros(angle_bins, dtype=torch.bool, device=positions.device)
        for b in range(angle_bins):
            mask_b = bin_pos == b
            if bool(mask_b.any()):
                radius_bins[b] = radii[mask_b].max()
                occupied[b] = True
        if int(occupied.sum().item()) < max(angle_bins // 3, 8):
            return None

        filled_bins = radius_bins.clone()
        occ_idx = torch.where(occupied)[0]
        for b in range(angle_bins):
            if occupied[b]:
                continue
            circular_dist = torch.minimum(
                (occ_idx - b).abs(),
                angle_bins - (occ_idx - b).abs(),
            )
            nearest = occ_idx[int(torch.argmin(circular_dist).item())]
            filled_bins[b] = radius_bins[nearest]
        for _ in range(2):
            filled_bins = torch.maximum(
                filled_bins,
                0.5 * (torch.roll(filled_bins, 1) + torch.roll(filled_bins, -1)),
            )
        filled_bins = inflate_scale * filled_bins

        seed_plane_dist = ((positions[seed_mask] - center.unsqueeze(0)) @ plane_normal).abs()
        slab = max(
            float(torch.quantile(seed_plane_dist, 0.90).item()) * 1.8,
            slab_scale,
        )
        uv_all = (positions - center.unsqueeze(0)) @ basis
        node_radii = uv_all.norm(dim=1)
        node_angles = torch.atan2(uv_all[:, 1], uv_all[:, 0])
        node_bins = ((node_angles + torch.pi) / (2.0 * torch.pi) * angle_bins).floor().long()
        node_bins = node_bins.clamp(0, angle_bins - 1)
        radius_limit = filled_bins[node_bins]
        plane_dist = ((positions - center.unsqueeze(0)) @ plane_normal).abs()
        inside_proj = (
            (node_radii <= radius_limit.clamp(min=1e-6))
            & (plane_dist <= slab)
        )
        candidate_mask = inside_proj | seed_mask
        candidate_mask[boundary_nodes] = True

        patch_mask = seed_mask.clone()
        allowed_edge = (~corridor_edge_mask) & candidate_mask.unsqueeze(1) & candidate_mask[graph.knn_idx]
        for _ in range(32):
            row_in = patch_mask.unsqueeze(1).expand_as(graph.knn_idx)
            col_in = patch_mask[graph.knn_idx]
            touch = allowed_edge & (row_in | col_in)
            if not bool(touch.any()):
                break
            expanded = patch_mask.clone()
            expanded |= torch.any(touch, dim=1)
            expanded[graph.knn_idx[touch]] = True
            delta = expanded & (~patch_mask)
            patch_mask = expanded
            if not bool(delta.any()):
                break

        patch_size = int(patch_mask.sum().item())
        if patch_size < min_patch_size or patch_size <= seed_size:
            return None

        cross_boundary = patch_mask.unsqueeze(1) ^ patch_mask[graph.knn_idx]
        boundary_edges = int(cross_boundary.sum().item())
        if boundary_edges < max(self.min_boundary_edges, 12):
            return None
        patch_boundary_mask = corridor_edge_mask & cross_boundary
        patch_cut_ratio = float(patch_boundary_mask[cross_boundary].float().mean().item()) if bool(cross_boundary.any()) else 0.0
        if patch_cut_ratio < max(0.32, 0.85 * self.fallback_cut_ratio):
            return None

        patch_center = positions[patch_mask].mean(dim=0)
        seed_center = positions[seed_mask].mean(dim=0)
        return {
            "mask": patch_mask,
            "boundary_mask": patch_boundary_mask,
            "size": patch_size,
            "seed_size": seed_size,
            "cut_ratio": patch_cut_ratio,
            "center": patch_center,
            "seed_center": seed_center,
            "plane_normal": plane_normal,
        }

    def _extract_explicit_closure_patches(
        self,
        labels: Tensor,
        positions: Optional[Tensor],
        graph: GaussianGraph,
        corridor_edge_mask: Tensor,
        grouped_boundary_labels: set,
        grouped_boundary_scores: dict,
        group_closure_scores: dict,
        members_by_group: dict,
    ) -> List[dict]:
        if positions is None or graph.knn_idx is None or not grouped_boundary_labels:
            return []

        closure_threshold, min_patch_size, inflate_scale, slab_scale = self._explicit_closure_params()
        patches: List[dict] = []
        used_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        ordered_groups = sorted(
            grouped_boundary_labels,
            key=lambda gid: (
                float(group_closure_scores.get(gid, 0.0)),
                float(grouped_boundary_scores.get(gid, 0.0)),
            ),
            reverse=True,
        )
        for group_id in ordered_groups:
            closure_score = float(group_closure_scores.get(group_id, 0.0))
            boundary_score = float(grouped_boundary_scores.get(group_id, 0.0))
            if closure_score < closure_threshold:
                continue
            if boundary_score < max(self.fallback_cut_ratio, 0.30):
                continue
            members = members_by_group.get(group_id, [])
            patch = self._build_explicit_patch_from_group(
                labels=labels,
                positions=positions,
                graph=graph,
                corridor_edge_mask=corridor_edge_mask,
                members=members,
                min_patch_size=min_patch_size,
                inflate_scale=inflate_scale,
                slab_scale=slab_scale,
            )
            if patch is None:
                continue
            patch_mask = patch["mask"] & (~used_mask)
            patch_size = int(patch_mask.sum().item())
            if patch_size < min_patch_size:
                continue
            patch["mask"] = patch_mask
            patch["group_id"] = group_id
            patch["closure_score"] = closure_score
            patch["boundary_score"] = boundary_score
            patch["release_score"] = float(min(1.0, max(boundary_score, 0.52 + 0.42 * closure_score)))
            patch["support_lost"] = bool(
                self.material_family == "rough_quasi_brittle"
                or closure_score >= 0.88
            )
            used_mask |= patch_mask
            patches.append(patch)

        return patches

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
        if self.material_family == "sharp_brittle":
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
            side_score = opposite_side.float() * side_sep * core_score.clamp(0.0, 1.0)
        cut_vote = (
            self.cut_vote_strength
            * core_score
            * cross_score
            * tangent_gate
            * cut_pair.float()
        ).clamp(0.0, 1.0)
        if self.material_family == "sharp_brittle":
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
        elif self.material_family == "rough_quasi_brittle":
            hard_cut = hard_cut | (
                cut_pair
                & (cross_align > min(self.tau_cross + 0.04, 0.94))
                & (tangent_align < 1.02 * self.tau_tangent)
                & (core_score > 0.34)
            )

        if self.material_family == "sharp_brittle":
            cut_edge_mask = cut_edge_mask | (side_score > 0.06)
        elif self.material_family == "rough_quasi_brittle":
            cut_edge_mask = cut_edge_mask | (
                cut_pair
                & (cross_align > max(self.tau_cross - 0.06, 0.28))
                & (core_score > 0.18)
            )
        cut_edge_mask = cut_edge_mask | hard_cut
        return cut_vote, hard_cut, cut_core_mask, cut_edge_mask

    def _union_find_cc(self, N: int, knn_idx, edge_alive) -> 'np.ndarray':
        """
        Union-Find connected components on CPU.

        Args:
            N: number of nodes
            knn_idx: (N, K) neighbor indices
            edge_alive: (N, K) boolean edge mask

        Returns:
            labels: (N,) numpy array of component labels
        """
        import numpy as np

        parent = np.arange(N, dtype=np.int64)
        rank = np.zeros(N, dtype=np.int64)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        knn_np = knn_idx.numpy()
        alive_np = edge_alive.numpy()
        K = knn_np.shape[1]

        for i in range(N):
            for ki in range(K):
                if alive_np[i, ki]:
                    j = knn_np[i, ki]
                    union(i, j)

        # Flatten
        labels = np.array([find(i) for i in range(N)], dtype=np.int64)
        return labels

    def apply_fragment_impulse(
        self,
        positions: Tensor,
        velocities: Tensor,
        impact_center: Optional[Tensor] = None,
        impulse_strength: float = 2.0,
        upward_bias: float = 0.5,
    ) -> Tensor:
        """
        Apply separation impulse to detected fragments.

        Each fragment gets a velocity impulse directed away from
        the impact center with an upward bias.

        Args:
            positions: (N, 3) Gaussian positions
            velocities: (N, 3) current velocities
            impact_center: (3,) impact point (default: centroid)
            impulse_strength: magnitude of impulse
            upward_bias: additional upward (z+) component

        Returns:
            velocities: (N, 3) updated velocities
        """
        if self.n_fragments <= 1:
            return velocities

        if impact_center is None:
            impact_center = positions.mean(dim=0)

        total_particles = positions.shape[0]
        v_out = velocities.clone()
        base_com = positions.mean(dim=0)
        if self.fragment_indices:
            base_com = positions[self.fragment_indices[0]].mean(dim=0)

        for frag_id, frag_idx in enumerate(self.fragment_indices):
            min_impulse_size = 6
            if self.material_family == "sharp_brittle":
                min_impulse_size = 3
            elif self.material_family == "rough_quasi_brittle":
                min_impulse_size = 5
            elif self.material_family == "brittle_moderate":
                min_impulse_size = 8
            if len(frag_idx) < min_impulse_size:
                continue

            com = positions[frag_idx].mean(dim=0)
            release_score = 0.0
            support_lost = False
            if frag_id < len(self.fragment_release_scores):
                release_score = float(self.fragment_release_scores[frag_id])
            if frag_id < len(self.fragment_support_lost):
                support_lost = bool(self.fragment_support_lost[frag_id])

            direction = com - impact_center
            if frag_id > 0 and support_lost:
                direction = com - base_com
            dist = direction.norm() + 1e-8
            direction = direction / dist

            # Released fragments should separate and then fall, not only jump upward.
            if support_lost and frag_id > 0:
                direction[2] -= 0.22 + 0.40 * release_score
            else:
                direction[2] += upward_bias
            direction = direction / (direction.norm() + 1e-8)

            # Scale inversely with fragment size
            size_ratio = len(frag_idx) / (total_particles + 1e-8)
            strength = impulse_strength * max(0.3, min(1.0, size_ratio * 5.0))
            if support_lost and frag_id > 0:
                strength *= 1.0 + 0.85 * release_score

            v_out[frag_idx] += strength * direction.unsqueeze(0)

        return v_out

    def get_fragment_coms(self, positions: Tensor) -> List[Tensor]:
        """Compute center of mass for each fragment."""
        coms = []
        for frag_idx in self.fragment_indices:
            if len(frag_idx) > 0:
                coms.append(positions[frag_idx].mean(dim=0))
        return coms
