"""
Graph Fragment Manager

Phase 3: Detect material fragments using graph connectivity analysis.
Instead of grid-based connected components (scipy.ndimage.label),
operates directly on the Gaussian kNN graph with damage-weakened edges.

Fragments are connected components of the Gaussian graph where
edges with high damage are removed.
"""

import math

import torch
from torch import Tensor
from typing import Dict, Optional, List, Tuple

from .graph_builder import GaussianGraph
from .fragment_phase_approval import FragmentPhaseApprovalMixin
from .fragment_cut_field import FragmentCutFieldMixin
from .fragment_closure_patches import FragmentClosurePatchMixin
from .fragment_release_patches import FragmentReleasePatchMixin
from .fragment_component_analysis import FragmentComponentAnalysisMixin
from .fragment_manager_utils import FragmentManagerUtilityMixin


class GraphFragmentManager(
    FragmentPhaseApprovalMixin,
    FragmentCutFieldMixin,
    FragmentClosurePatchMixin,
    FragmentReleasePatchMixin,
    FragmentComponentAnalysisMixin,
    FragmentManagerUtilityMixin,
):
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
        persistent_min_fragment_size_ratio: float = 0.0,
        persistent_min_fragment_reference_nodes: int = 10000,
        persistent_min_fragment_resolution_exponent: float = 0.5,
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
        fragment_boundary_cut_min_ratio: float = 0.0,
        open_crack_release_enable: bool = True,
        open_crack_release_threshold: float = 0.0,
        open_crack_release_max_patches: int = 2,
        crack_connected_release_only: bool = False,
        strict_closure_max_released_ratio: float = 0.54,
        crack_style: str = "material_default",
        brittle_release_intensity: float = 1.0,
        impact_release_gain: float = 1.0,
        impact_closure_target_ratio: float = -1.0,
        impact_closure_max_patches: int = -1,
        impact_closure_active_frames: int = 2,
        impact_closure_sector_count: int = 0,
        impact_closure_band_count: int = 0,
        impact_closure_layer_count: int = 0,
        impact_closure_min_size_ratio: float = 0.0,
        impact_closure_max_size_ratio: float = 0.0,
        impact_closure_adaptive_extra_frames: int = 6,
        impact_closure_completion_ratio: float = 0.92,
        phase_approval_enable: bool = True,
        phase_approval_threshold_scale: float = 1.0,
        phase_approval_threshold_offset: float = 0.0,
        phase_cc_modulation_enable: bool = True,
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
        self.persistent_min_fragment_size_ratio = max(
            float(persistent_min_fragment_size_ratio), 0.0)
        self.persistent_min_fragment_reference_nodes = max(
            int(persistent_min_fragment_reference_nodes), 1)
        self.persistent_min_fragment_resolution_exponent = max(
            float(persistent_min_fragment_resolution_exponent), 0.0)
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
        # Causal-support gate: a fragment candidate is rejected when the
        # mean cut-vote across its boundary is below this ratio.  0.0
        # disables (paper-baseline behavior); 0.4--0.6 enforces that
        # fragment boundaries actually align with cracked edges, fixing
        # the "premature fragmentation" mismatch where AT2 damage
        # diffusion forms patches before crack tips arrive.
        self.fragment_boundary_cut_min_ratio = max(
            min(float(fragment_boundary_cut_min_ratio), 1.0), 0.0)
        self.open_crack_release_enable = bool(open_crack_release_enable)
        self.open_crack_release_threshold = float(open_crack_release_threshold)
        self.open_crack_release_max_patches = max(int(open_crack_release_max_patches), 0)
        self.crack_connected_release_only = bool(crack_connected_release_only)
        self.strict_closure_max_released_ratio = float(strict_closure_max_released_ratio)
        self.crack_style = str(crack_style)
        self.brittle_release_intensity = float(brittle_release_intensity)
        self.impact_release_gain = float(impact_release_gain)
        self.configured_impact_closure_target_ratio = float(impact_closure_target_ratio)
        self.configured_impact_closure_max_patches = int(impact_closure_max_patches)
        self.configured_impact_closure_active_frames = max(
            int(impact_closure_active_frames), 0)
        self.impact_closure_sector_count = max(int(impact_closure_sector_count), 0)
        self.impact_closure_band_count = max(int(impact_closure_band_count), 0)
        self.impact_closure_layer_count = max(int(impact_closure_layer_count), 0)
        self.impact_closure_min_size_ratio = max(
            float(impact_closure_min_size_ratio), 0.0)
        self.impact_closure_max_size_ratio = max(
            float(impact_closure_max_size_ratio), 0.0)
        self.impact_closure_adaptive_extra_frames = max(
            int(impact_closure_adaptive_extra_frames), 0)
        self.impact_closure_completion_ratio = min(
            max(float(impact_closure_completion_ratio), 0.0), 1.0)
        self.phase_approval_enable = bool(phase_approval_enable)
        # CC-detection narrow-band modulation: when on, edge_phase_gate is
        # multiplied into seed_d_cut and used to gate hard_cut.  This
        # implements the v1.5 narrow-band doctrine but couples connectivity
        # detection to the phase field; turning it off lets ablations
        # measure pure surface-graph CC behavior.
        self.phase_cc_modulation_enable = bool(phase_cc_modulation_enable)
        self.phase_approval_threshold_scale = float(phase_approval_threshold_scale)
        self.phase_approval_threshold_offset = float(phase_approval_threshold_offset)
        self.material_family = str(material_family)
        self.device = torch.device(device)
        self.impact_closure_active = False
        self.impact_closure_target_ratio = 0.0
        self.impact_closure_max_patches = 0

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
        self.last_open_release_patches: int = 0
        self.last_open_release_nodes: int = 0
        self.last_open_release_score_max: float = 0.0
        self.last_impact_closure_patches: int = 0
        self.last_impact_closure_nodes: int = 0
        self.last_impact_closure_score_max: float = 0.0
        self.last_phase_gate_max: float = 0.0
        self.last_phase_gate_mean: float = 0.0
        self.last_phase_cut_edges: int = 0
        self.last_phase_candidate_patches: int = 0
        self.last_phase_approved_patches: int = 0
        self.last_phase_rejected_patches: int = 0
        self.last_phase_approved_nodes: int = 0
        self.last_phase_approval_score_max: float = 0.0
        self.last_phase_approval_score_mean: float = 0.0
        self.last_phase_approval_cvol_max: float = 0.0
        self.last_phase_approval_gate_max: float = 0.0
        self.last_pseudo_thickness_mass: float = 0.0
        self.last_birth_phase_score: float = 0.0
        self.last_birth_cvol_max: float = 0.0
        self.last_birth_phase_gate_max: float = 0.0
        self.last_birth_phase_approved: bool = False
        self.last_effective_edge_damage: Optional[Tensor] = None
        self.edge_cut_memory: Optional[Tensor] = None
        self.authoritative_cut_memory: Optional[Tensor] = None
        self.detached_node_memory: Optional[Tensor] = None
        self.detached_node_mask: Optional[Tensor] = None
        self.detached_fragment_ids: Optional[Tensor] = None
        self.detached_boundary_mask: Optional[Tensor] = None
        self.detached_fragment_meta: dict[int, dict] = {}
        self.last_cut_core_mask: Optional[Tensor] = None
        self.last_cut_edge_mask: Optional[Tensor] = None
        self.last_authoritative_cut_mask: Optional[Tensor] = None
        self.last_support_lost_mask: Optional[Tensor] = None
        self.last_closure_candidate_mask: Optional[Tensor] = None
        self.last_closure_boundary_mask: Optional[Tensor] = None
        self.last_hard_detached_nodes: int = 0
        self.last_new_detached_nodes: int = 0
        self.last_detached_boundary_edges: int = 0
        self.has_split_once: bool = False
        self.fragment_release_scores: List[float] = []
        self.fragment_support_scores: List[float] = []
        self.fragment_support_lost: List[bool] = []

    def _effective_persistent_min_size(self, total_nodes: int) -> int:
        base_size = max(int(self.persistent_min_fragment_size), 1)
        resolution_scale = max(
            float(total_nodes) / float(self.persistent_min_fragment_reference_nodes),
            1.0,
        ) ** self.persistent_min_fragment_resolution_exponent
        resolution_size = int(round(float(base_size) * resolution_scale))
        ratio_size = int(round(float(total_nodes) * self.persistent_min_fragment_size_ratio))
        return max(base_size, resolution_size, ratio_size, 1)

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
        phase_gate: Optional[Tensor] = None,
        volume_damage: Optional[Tensor] = None,
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
            self.last_open_release_patches = 0
            self.last_open_release_nodes = 0
            self.last_open_release_score_max = 0.0
            self.last_impact_closure_patches = 0
            self.last_impact_closure_nodes = 0
            self.last_impact_closure_score_max = 0.0
            self.last_phase_gate_max = 0.0
            self.last_phase_gate_mean = 0.0
            self.last_phase_cut_edges = 0
            self._reset_phase_approval_stats()
            self.edge_cut_memory = None
            self.authoritative_cut_memory = None
            self.detached_node_memory = None
            self.detached_node_mask = None
            self.detached_fragment_ids = None
            self.detached_boundary_mask = None
            self.detached_fragment_meta = {}
            self.last_effective_edge_damage = None
            self.last_cut_core_mask = None
            self.last_cut_edge_mask = None
            self.last_authoritative_cut_mask = None
            self.last_support_lost_mask = None
            self.last_closure_candidate_mask = None
            self.last_closure_boundary_mask = None
            self.last_hard_detached_nodes = 0
            self.last_new_detached_nodes = 0
            self.last_detached_boundary_edges = 0
            self.has_split_once = False
            self.fragment_release_scores = [0.0]
            self.fragment_support_scores = [1.0]
            self.fragment_support_lost = [False]
            return 1

        N = damage.shape[0]
        effective_persistent_min_size = self._effective_persistent_min_size(N)
        if graph.knn_idx is None:
            self.fragment_ids = torch.zeros(N, dtype=torch.long, device=self.device)
            self.n_fragments = 1
            self.last_effective_edge_damage = None
            self.fragment_release_scores = [0.0]
            self.fragment_support_scores = [1.0]
            self.fragment_support_lost = [False]
            self.last_phase_gate_max = 0.0
            self.last_phase_gate_mean = 0.0
            self.last_phase_cut_edges = 0
            self._reset_phase_approval_stats()
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
        if (self.detached_node_mask is None
                or self.detached_node_mask.shape[0] != N):
            self.detached_node_mask = torch.zeros(
                N,
                dtype=torch.bool,
                device=self.device,
            )
            self.detached_fragment_ids = torch.zeros(
                N,
                dtype=torch.long,
                device=self.device,
            )
            self.detached_boundary_mask = torch.zeros(
                graph.knn_idx.shape,
                dtype=torch.bool,
                device=self.device,
            )
            self.detached_fragment_meta = {}
        elif (self.detached_fragment_ids is None
                or self.detached_fragment_ids.shape[0] != N):
            self.detached_fragment_ids = torch.zeros(
                N,
                dtype=torch.long,
                device=self.device,
            )
            if bool(self.detached_node_mask.any()):
                self.detached_fragment_ids[self.detached_node_mask] = 1
            self.detached_fragment_meta = {}
        if (self.detached_boundary_mask is None
                or self.detached_boundary_mask.shape != graph.knn_idx.shape):
            self.detached_boundary_mask = torch.zeros(
                graph.knn_idx.shape,
                dtype=torch.bool,
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
        self.last_open_release_patches = 0
        self.last_open_release_nodes = 0
        self.last_open_release_score_max = 0.0
        self.last_impact_closure_patches = 0
        self.last_impact_closure_nodes = 0
        self.last_impact_closure_score_max = 0.0
        self.last_phase_gate_max = 0.0
        self.last_phase_gate_mean = 0.0
        self.last_phase_cut_edges = 0
        self._reset_phase_approval_stats()
        self.last_authoritative_cut_mask = None
        self.last_support_lost_mask = None
        self.last_closure_candidate_mask = None
        self.last_closure_boundary_mask = None
        self.last_hard_detached_nodes = int(
            self.detached_node_mask.sum().item()
        ) if self.detached_node_mask is not None else 0
        self.last_new_detached_nodes = 0
        self.last_detached_boundary_edges = int(
            self.detached_boundary_mask.sum().item()
        ) if self.detached_boundary_mask is not None else 0
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

        phase_gate_local = None
        edge_phase_gate = None
        volume_damage_local = None
        if phase_gate is not None:
            phase_gate_local = phase_gate.to(device=self.device, dtype=damage.dtype)[:N].clamp(0.0, 1.0)
            self.last_phase_gate_max = float(phase_gate_local.max().item()) if phase_gate_local.numel() else 0.0
            self.last_phase_gate_mean = float(phase_gate_local.mean().item()) if phase_gate_local.numel() else 0.0
        if volume_damage is not None:
            volume_damage_local = volume_damage.to(
                device=self.device,
                dtype=damage.dtype,
            )[:N].clamp(0.0, 1.0)

        c_i = node_break.unsqueeze(1).expand_as(graph.knn_idx.float())
        c_j = node_break[graph.knn_idx]
        if phase_gate_local is not None:
            g_i = phase_gate_local.unsqueeze(1).expand_as(graph.knn_idx.float())
            g_j = phase_gate_local[graph.knn_idx]
            edge_phase_gate = torch.maximum(g_i, g_j).clamp(0.0, 1.0)
            self.last_phase_cut_edges = int((edge_phase_gate > 0.0).sum().item())
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
        if edge_phase_gate is not None and self.phase_cc_modulation_enable:
            seed_d_cut = seed_d_cut * edge_phase_gate
            if hard_cut is not None:
                hard_cut = hard_cut & (edge_phase_gate > 0.12)

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
        hard_detached_mask = (
            self.detached_node_mask.bool()
            if (
                self.crack_connected_release_only
                and self.detached_node_mask is not None
                and self.detached_node_mask.shape[0] == N
            )
            else torch.zeros(N, dtype=torch.bool, device=self.device)
        )
        if bool(hard_detached_mask.any()):
            detached_edge = (
                hard_detached_mask.unsqueeze(1)
                | hard_detached_mask[graph.knn_idx]
            )
            edge_alive = edge_alive & (~detached_edge)
            corridor_edge_mask = corridor_edge_mask & (~detached_edge)
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

        labels = self._connected_components_cpu(N, knn_idx_cpu, edge_alive_cpu)
        labels = torch.from_numpy(labels).to(self.device)

        # Remap to contiguous labels and filter small fragments.  In strict
        # crack-connected mode, previously detached patches are no longer part
        # of the active mesh; they keep their fragment labels and cannot seed
        # future closure patches.
        unique_labels = labels.unique()
        unique_label_list = [int(lbl) for lbl in unique_labels.tolist()]
        active_label_sizes = {
            old_label: int(
                ((labels == old_label) & (~hard_detached_mask)).sum().item()
            )
            for old_label in unique_label_list
        }
        active_label_list = [
            old_label for old_label in unique_label_list
            if active_label_sizes.get(old_label, 0) > 0
        ]
        persistent_fragment_labels = set()
        if (
            self.crack_connected_release_only
            and self.detached_fragment_ids is not None
            and bool(hard_detached_mask.any())
        ):
            persistent_fragment_labels = {
                int(lbl)
                for lbl in self.detached_fragment_ids[hard_detached_mask].unique().tolist()
                if int(lbl) > 0
            }
        self.last_raw_components = (
            len(active_label_list) + len(persistent_fragment_labels)
        )
        label_sizes = [
            active_label_sizes.get(old_label, 0)
            for old_label in unique_label_list
        ]
        promoted_labels = set()
        main_label = None
        if active_label_list:
            main_label = max(
                active_label_list,
                key=lambda lbl: active_label_sizes.get(lbl, 0),
            )
        boundary_cut_ratio_by_old, boundary_edges_by_old = self._compute_boundary_cut_stats(
            labels=labels,
            graph=graph,
            effective_cut_damage=effective_cut_damage,
            cut_threshold=cut_break_threshold,
            excluded_mask=hard_detached_mask,
        )
        boundary_candidate_labels = set()
        boundary_score_by_old = {}
        for old_label in active_label_list:
            size = active_label_sizes.get(old_label, 0)
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
            excluded_mask=hard_detached_mask,
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
        for old_label in active_label_list:
            group_id = component_group_map.get(old_label, old_label)
            members_by_group.setdefault(group_id, []).append(old_label)
        explicit_patches = self._collect_fragment_patches(
            labels=labels,
            positions=positions,
            graph=graph,
            corridor_edge_mask=corridor_edge_mask,
            damage=damage,
            opening=opening,
            active_tip_mask=active_tip_mask,
            recent_front_mask=recent_front_mask,
            grouped_boundary_labels=candidate_group_ids,
            grouped_boundary_scores=grouped_boundary_scores,
            group_closure_scores=group_closure_scores,
            members_by_group=members_by_group,
            excluded_mask=hard_detached_mask,
        )
        if self.crack_connected_release_only:
            explicit_patches = self._filter_phase_approved_patches(
                patches=explicit_patches,
                graph=graph,
                damage=damage,
                opening=opening,
                phase_gate=phase_gate_local,
                volume_damage=volume_damage_local,
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
                member_mask = (labels == old_label) & (~hard_detached_mask)
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
            primary_group_ids=(
                set() if self.crack_connected_release_only else primary_promoted_labels
            ),
            component_stats=component_stats,
            interface_graph=interface_graph,
        )
        self.last_absorbed_components = absorbed_count
        self.last_components_above_primary = len(primary_promoted_labels)
        self.last_components_above_fallback = len(primary_promoted_labels) + len(fallback_candidate_labels)
        if not self.crack_connected_release_only:
            promoted_labels.update(primary_promoted_labels)
            self.last_primary_promoted_components = len(primary_promoted_labels)
        else:
            self.last_primary_promoted_components = 0
        release_score_by_old = {}
        support_score_by_old = {}
        support_lost_labels = set()
        support_lost_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        if (
            self.support_loss_enable
            and positions is not None
            and not self.crack_connected_release_only
        ):
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
                for old_label in active_label_list:
                    group_id = component_group_map.get(old_label, old_label)
                    if group_id in support_lost_labels:
                        filtered_support_lost_mask |= (
                            (labels == old_label) & (~hard_detached_mask)
                        )
                support_lost_mask = filtered_support_lost_mask
            else:
                support_lost_mask = torch.zeros_like(support_lost_mask)
        grouped_support_lost_labels = set(support_lost_labels)
        grouped_release_scores = {}
        grouped_support_scores = {}
        for old_label in active_label_list:
            group_id = component_group_map.get(old_label, old_label)
            grouped_release_scores[group_id] = max(
                grouped_release_scores.get(group_id, 0.0),
                float(boundary_score_by_old.get(old_label, release_score_by_old.get(old_label, 0.0))),
            )
            grouped_support_scores[group_id] = min(
                grouped_support_scores.get(group_id, 1.0),
                float(support_score_by_old.get(old_label, 1.0)),
            )
        if not self.crack_connected_release_only:
            promoted_labels.update(grouped_support_lost_labels)
            self.last_fallback_promoted_components = len(grouped_support_lost_labels)
            self.last_support_lost_components = len(grouped_support_lost_labels)
        else:
            grouped_support_lost_labels = set()
            self.last_fallback_promoted_components = 0
            self.last_support_lost_components = 0
        self.last_release_candidate_count = (
            (0 if self.crack_connected_release_only else len(primary_promoted_labels))
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
        if (
            self.has_split_once
            and self.detached_node_memory is not None
            and not self.crack_connected_release_only
        ):
            detached_mask = self.detached_node_memory > 0.25
            if bool(detached_mask.any()):
                overlap_labels = labels[detached_mask].unique()
                for old_label in overlap_labels.tolist():
                    old_label = int(old_label)
                    if old_label not in active_label_sizes:
                        continue
                    mask = (labels == old_label) & (~hard_detached_mask)
                    size = int(mask.sum().item())
                    if size < effective_persistent_min_size:
                        continue
                    overlap = int((mask & detached_mask).sum().item())
                    overlap_ratio = overlap / max(size, 1)
                    if overlap_ratio >= self.component_hysteresis:
                        promoted_labels.add(component_group_map.get(old_label, old_label))
        self.last_primary_promoted_components += len(explicit_patches)
        self.last_promoted_components = len(promoted_labels) + len(explicit_patches)

        # Sort by size (largest first)
        sorted_pairs = sorted(
            [
                (old_label, active_label_sizes.get(old_label, 0))
                for old_label in active_label_list
            ],
            key=lambda x: -x[1],
        )
        self.last_top_component_sizes = [int(size) for _, size in sorted_pairs[:8]]
        grouped_sizes = {}
        for old_label, size in sorted_pairs:
            group_id = component_group_map.get(old_label, old_label)
            grouped_sizes[group_id] = grouped_sizes.get(group_id, 0) + int(size)

        # Remap active mesh components while preserving previously detached
        # patch labels.  Existing detached patches are hard occupied, so new
        # closure patches can only claim nodes still belonging to the main mesh.
        new_labels = torch.zeros(N, dtype=torch.long, device=self.device)
        persistent_meta = {}
        if (
            self.crack_connected_release_only
            and self.detached_fragment_ids is not None
            and bool(hard_detached_mask.any())
        ):
            persistent_ids = self.detached_fragment_ids.to(
                device=self.device,
                dtype=torch.long,
            ).clone()
            persistent_ids[~hard_detached_mask] = 0
            missing_persistent = hard_detached_mask & (persistent_ids <= 0)
            if bool(missing_persistent.any()):
                persistent_ids[missing_persistent] = 1
            new_labels[hard_detached_mask] = persistent_ids[hard_detached_mask]
            for frag_label in sorted(persistent_fragment_labels):
                persistent_meta[frag_label] = self.detached_fragment_meta.get(
                    frag_label,
                    {
                        "release_score": 0.0,
                        "support_score": 0.25,
                        "support_lost": False,
                    },
                )
            if bool(missing_persistent.any()):
                persistent_meta.setdefault(
                    1,
                    {
                        "release_score": 0.0,
                        "support_score": 0.25,
                        "support_lost": False,
                    },
                )

        base_group_id = (
            component_group_map.get(main_label, main_label)
            if main_label is not None else None
        )
        assigned_groups = {}
        kept_group_by_label = {}
        if base_group_id is not None:
            assigned_groups[base_group_id] = 0
            kept_group_by_label[0] = base_group_id
        next_component_label = (
            int(new_labels.max().item()) + 1 if new_labels.numel() > 0 else 1
        )

        for idx, (old_label, size) in enumerate(sorted_pairs):
            group_id = component_group_map.get(old_label, old_label)
            group_size = grouped_sizes.get(group_id, size)
            is_base_group = group_id == base_group_id
            if self.crack_connected_release_only:
                keep_component = (
                    is_base_group
                    or (
                        group_id in promoted_labels
                        and group_size >= effective_persistent_min_size
                    )
                )
            else:
                keep_component = (
                    is_base_group
                    or group_size >= effective_min_fragment_size
                    or group_id in promoted_labels
                )
            if not keep_component:
                continue
            if group_id not in assigned_groups:
                assigned_groups[group_id] = next_component_label
                kept_group_by_label[next_component_label] = group_id
                next_component_label += 1
            assign_mask = (labels == old_label) & (~hard_detached_mask)
            new_labels[assign_mask] = assigned_groups[group_id]

        explicit_meta = {}
        next_explicit_label = int(new_labels.max().item()) + 1 if new_labels.numel() > 0 else 1
        occupied_explicit = hard_detached_mask.clone()
        new_detached_boundary_mask = torch.zeros_like(corridor_edge_mask)
        assigned_phase_scores: List[float] = []
        assigned_phase_cvol: List[float] = []
        assigned_phase_gates: List[float] = []
        assigned_pseudo_masses: List[float] = []
        assigned_phase_nodes = 0
        strict_remaining_nodes = N
        if self.crack_connected_release_only:
            strict_max_detached_nodes = int(
                round(float(N) * self._strict_closure_release_cap())
            )
            strict_remaining_nodes = max(
                strict_max_detached_nodes - int(hard_detached_mask.sum().item()),
                0,
            )
        for patch in sorted(explicit_patches, key=lambda item: -int(item["size"])):
            if self.crack_connected_release_only and strict_remaining_nodes <= 0:
                break
            patch_mask = patch["mask"] & (~occupied_explicit)
            patch_size = int(patch_mask.sum().item())
            if patch_size < effective_persistent_min_size:
                continue
            if (
                self.crack_connected_release_only
                and patch_size > strict_remaining_nodes
            ):
                continue
            new_labels[patch_mask] = next_explicit_label
            patch_boundary = patch.get("boundary_mask")
            if (
                patch_boundary is not None
                and patch_boundary.shape == new_detached_boundary_mask.shape
            ):
                new_detached_boundary_mask |= patch_boundary.bool()
            explicit_meta[next_explicit_label] = {
                "release_score": float(patch.get("release_score", 0.0)),
                "support_score": 0.0 if bool(patch.get("support_lost", False)) else 0.25,
                "support_lost": bool(patch.get("support_lost", False)),
                "phase_score": float(patch.get("phase_score", 0.0)),
                "phase_cvol_max": float(patch.get("phase_cvol_max", 0.0)),
                "phase_gate_max": float(patch.get("phase_gate_max", 0.0)),
                "pseudo_thickness_mass": float(patch.get("pseudo_thickness_mass", patch_size)),
            }
            assigned_phase_scores.append(float(patch.get("phase_score", 0.0)))
            assigned_phase_cvol.append(float(patch.get("phase_cvol_max", 0.0)))
            assigned_phase_gates.append(float(patch.get("phase_gate_max", 0.0)))
            assigned_pseudo_masses.append(float(patch.get("pseudo_thickness_mass", patch_size)))
            assigned_phase_nodes += patch_size
            occupied_explicit |= patch_mask
            if self.crack_connected_release_only:
                strict_remaining_nodes -= patch_size
            next_explicit_label += 1

        if assigned_phase_scores:
            self.last_phase_approved_patches = len(assigned_phase_scores)
            self.last_phase_approved_nodes = int(assigned_phase_nodes)
            self.last_phase_approval_score_max = max(assigned_phase_scores)
            self.last_phase_approval_score_mean = (
                sum(assigned_phase_scores) / max(len(assigned_phase_scores), 1)
            )
            self.last_phase_approval_cvol_max = max(assigned_phase_cvol or [0.0])
            self.last_phase_approval_gate_max = max(assigned_phase_gates or [0.0])
            self.last_pseudo_thickness_mass = sum(assigned_pseudo_masses)

        self.fragment_ids = new_labels
        unique_new_labels = self.fragment_ids.unique(sorted=True)
        self.fragment_sizes = []
        self.fragment_indices = []
        max_new_label = (
            int(unique_new_labels.max().item())
            if unique_new_labels.numel() > 0 else 0
        )
        self.fragment_release_scores = [0.0 for _ in range(max_new_label + 1)]
        self.fragment_support_scores = [1.0 for _ in range(max_new_label + 1)]
        self.fragment_support_lost = [False for _ in range(max_new_label + 1)]
        label_meta_by_new_label = {}
        for new_label in unique_new_labels.tolist():
            new_label = int(new_label)
            mask = self.fragment_ids == new_label
            size = int(mask.sum().item())
            if size <= 0:
                continue
            self.fragment_sizes.append(size)
            self.fragment_indices.append(torch.where(mask)[0])
            if new_label in explicit_meta:
                meta = explicit_meta[new_label]
            elif new_label in persistent_meta:
                meta = persistent_meta[new_label]
            else:
                group_id = kept_group_by_label.get(new_label)
                meta = {
                    "release_score": float(grouped_release_scores.get(group_id, 0.0)),
                    "support_score": float(grouped_support_scores.get(group_id, 1.0)),
                    "support_lost": bool(group_id in grouped_support_lost_labels),
                }
            label_meta_by_new_label[new_label] = {
                "release_score": float(meta.get("release_score", 0.0)),
                "support_score": float(meta.get("support_score", 1.0)),
                "support_lost": bool(meta.get("support_lost", False)),
            }
            self.fragment_release_scores[new_label] = float(
                label_meta_by_new_label[new_label]["release_score"]
            )
            self.fragment_support_scores[new_label] = float(
                label_meta_by_new_label[new_label]["support_score"]
            )
            self.fragment_support_lost[new_label] = bool(
                label_meta_by_new_label[new_label]["support_lost"]
            )

        self.n_fragments = len(self.fragment_indices) if self.fragment_indices else 1
        if self.n_fragments > 1:
            self.has_split_once = True
        if self.crack_connected_release_only:
            previous_hard_detached = hard_detached_mask
            current_detached = self.fragment_ids > 0
            self.last_new_detached_nodes = int(
                (current_detached & (~previous_hard_detached)).sum().item()
            )
            if self.last_new_detached_nodes > 0 and assigned_phase_scores:
                self.last_birth_phase_score = max(assigned_phase_scores)
                self.last_birth_cvol_max = max(assigned_phase_cvol or [0.0])
                self.last_birth_phase_gate_max = max(assigned_phase_gates or [0.0])
                self.last_birth_phase_approved = True
            self.detached_node_mask = previous_hard_detached | current_detached
            self.detached_fragment_ids = torch.zeros(
                N,
                dtype=torch.long,
                device=self.device,
            )
            self.detached_fragment_ids[self.detached_node_mask] = (
                self.fragment_ids[self.detached_node_mask]
            )
            if self.detached_boundary_mask is None:
                self.detached_boundary_mask = torch.zeros_like(new_detached_boundary_mask)
            current_detached_boundary = (
                current_detached.unsqueeze(1)
                ^ current_detached[graph.knn_idx]
            )
            self.detached_boundary_mask = (
                self.detached_boundary_mask.bool()
                | new_detached_boundary_mask.bool()
                | current_detached_boundary.bool()
            )
            self.detached_fragment_meta = {
                int(label): dict(meta)
                for label, meta in label_meta_by_new_label.items()
                if int(label) > 0
            }
            self.last_hard_detached_nodes = int(
                self.detached_node_mask.sum().item()
            )
            self.last_detached_boundary_edges = int(
                self.detached_boundary_mask.sum().item()
            )
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
