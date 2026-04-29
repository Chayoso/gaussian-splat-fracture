"""Utility methods for graph fragment detection."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .graph_builder import GaussianGraph


class FragmentManagerUtilityMixin:
    """Connected components and post-detection fragment utilities."""

    def _connected_components_cpu(self, N: int, knn_idx, edge_alive) -> 'np.ndarray':
        """
        Connected components on the CPU sparse kNN adjacency.

        Args:
            N: number of nodes
            knn_idx: (N, K) neighbor indices
            edge_alive: (N, K) boolean edge mask

        Returns:
            labels: (N,) numpy array of component labels
        """
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        knn_np = knn_idx.numpy()
        alive_np = edge_alive.numpy()
        if knn_np.size == 0:
            return np.arange(N, dtype=np.int64)
        rows = np.repeat(np.arange(N, dtype=np.int64), knn_np.shape[1])
        cols = knn_np.reshape(-1).astype(np.int64, copy=False)
        alive = alive_np.reshape(-1).astype(bool, copy=False)
        if not np.any(alive):
            return np.arange(N, dtype=np.int64)
        data = np.ones(int(alive.sum()), dtype=np.uint8)
        adjacency = coo_matrix(
            (data, (rows[alive], cols[alive])),
            shape=(N, N),
            dtype=np.uint8,
        )
        adjacency = adjacency.maximum(adjacency.T)
        _, labels = connected_components(
            adjacency.tocsr(),
            directed=False,
            return_labels=True,
        )
        return labels.astype(np.int64, copy=False)

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

