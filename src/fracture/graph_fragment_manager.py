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
from typing import Optional, List, Tuple

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
        self.material_family = str(material_family)
        self.device = torch.device(device)

        self.n_fragments: int = 0
        self.fragment_ids: Optional[Tensor] = None   # (N,) fragment label per Gaussian
        self.fragment_sizes: List[int] = []
        self.fragment_indices: List[Tensor] = []      # per-fragment Gaussian indices

    def detect_fragments(
        self,
        graph: GaussianGraph,
        damage: Tensor,
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
            return 1

        N = damage.shape[0]
        if graph.knn_idx is None:
            self.fragment_ids = torch.zeros(N, dtype=torch.long, device=self.device)
            self.n_fragments = 1
            return 1

        # Compute edge connectivity
        connectivity = graph.edge_damage_strength(damage)  # (N, K)
        edge_break_rate = max(self.edge_break_rate, 1e-4)
        damage_threshold = self.damage_threshold
        if self.material_family == "sharp_brittle":
            edge_break_rate *= 1.20
            damage_threshold *= 0.84
        elif self.material_family == "brittle_moderate":
            edge_break_rate *= 1.08
            damage_threshold *= 0.92
        elif self.material_family == "rough_quasi_brittle":
            edge_break_rate *= 0.96
            damage_threshold *= 1.04
        elif self.material_family == "neutral_reference":
            edge_break_rate *= 0.92
            damage_threshold *= 0.96
        connectivity = connectivity.clamp(0.0, 1.0) ** edge_break_rate

        # Binary edge mask: edge is intact if connectivity > threshold
        edge_alive = connectivity > (1.0 - damage_threshold)

        # Union-Find on CPU (graph CC is inherently serial)
        knn_idx_cpu = graph.knn_idx.cpu()
        edge_alive_cpu = edge_alive.cpu()

        labels = self._union_find_cc(N, knn_idx_cpu, edge_alive_cpu)
        labels = torch.from_numpy(labels).to(self.device)

        # Remap to contiguous labels and filter small fragments
        unique_labels = labels.unique()
        label_sizes = [(labels == lbl).sum().item() for lbl in unique_labels]

        # Sort by size (largest first)
        sorted_pairs = sorted(zip(unique_labels.tolist(), label_sizes),
                              key=lambda x: -x[1])

        # Remap
        new_labels = torch.zeros(N, dtype=torch.long, device=self.device)
        fragment_idx = 0
        self.fragment_sizes = []
        self.fragment_indices = []

        for old_label, size in sorted_pairs:
            mask = labels == old_label
            if size < self.min_fragment_size:
                # Assign small fragments to nearest large fragment
                continue
            new_labels[mask] = fragment_idx
            self.fragment_sizes.append(size)
            self.fragment_indices.append(torch.where(mask)[0])
            fragment_idx += 1

        # Assign orphans (small fragments) to nearest large fragment
        orphan_mask = new_labels == 0
        # Mark the actual fragment-0 Gaussians
        if len(self.fragment_indices) > 0:
            frag0_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
            frag0_mask[self.fragment_indices[0]] = True
            real_orphans = orphan_mask & ~frag0_mask
        else:
            real_orphans = orphan_mask

        # Orphan assignment skipped if no large fragments exist
        # (in that case everything is one fragment)

        self.fragment_ids = new_labels
        self.n_fragments = fragment_idx if fragment_idx > 0 else 1

        if self.n_fragments > 1:
            print(f"[GraphFrag] Detected {self.n_fragments} fragments: "
                  f"sizes={self.fragment_sizes[:10]}")

        return self.n_fragments

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

        for frag_idx in self.fragment_indices:
            if len(frag_idx) < 10:
                continue

            com = positions[frag_idx].mean(dim=0)
            direction = com - impact_center
            dist = direction.norm() + 1e-8
            direction = direction / dist

            # Add upward component
            direction[2] += upward_bias
            direction = direction / (direction.norm() + 1e-8)

            # Scale inversely with fragment size
            size_ratio = len(frag_idx) / (total_particles + 1e-8)
            strength = impulse_strength * max(0.3, min(1.0, size_ratio * 5.0))

            v_out[frag_idx] += strength * direction.unsqueeze(0)

        return v_out

    def get_fragment_coms(self, positions: Tensor) -> List[Tensor]:
        """Compute center of mass for each fragment."""
        coms = []
        for frag_idx in self.fragment_indices:
            if len(frag_idx) > 0:
                coms.append(positions[frag_idx].mean(dim=0))
        return coms
