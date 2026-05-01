"""
Gaussian Graph Builder

Constructs and maintains a kNN neighborhood graph over Gaussian positions.
Used for graph-based fracture propagation, fragment detection, and
Laplacian-like damage diffusion.
"""

import torch
from torch import Tensor
from typing import Optional

from src.utils.knn import knn_search


class GaussianGraph:
    """
    kNN neighborhood graph on 3D Gaussian positions.

    Stores:
        edges:   (2, E) edge index pairs (bidirectional)
        weights: (E,)  edge weights (Gaussian kernel of distance)
        knn_idx: (N, K) k-nearest neighbor indices per node
        knn_dist:(N, K) distances to k-nearest neighbors
    """

    def __init__(
        self,
        k: int = 12,
        sigma: float = 0.02,
        rebuild_every: int = 5,
        normal_compat_threshold: float = 0.0,
        device: str = "cuda",
    ):
        """
        Args:
            k: number of nearest neighbors
            sigma: Gaussian kernel bandwidth for edge weights
            rebuild_every: rebuild graph every N frames (0 = every frame)
            normal_compat_threshold: min dot(n_i, n_j) to keep edge.
                0.0 = only reject opposing normals (<0).
                Higher values are stricter.
            device: torch device
        """
        self.k = k
        self.sigma = sigma
        self.rebuild_every = rebuild_every
        self.normal_compat_threshold = normal_compat_threshold
        self.device = torch.device(device)

        # Graph state
        self.knn_idx: Optional[Tensor] = None   # (N, K)
        self.knn_dist: Optional[Tensor] = None   # (N, K)
        self.weights: Optional[Tensor] = None     # (N, K) normalized weights
        self.N: int = 0
        self._build_count: int = 0
        self._normals: Optional[Tensor] = None    # cached normals
        # Curvature-weighted anisotropy: per-particle in-shell direction of
        # max curvature ("principal tangent") and the corresponding
        # anisotropy strength.  Used downstream by the AT2 fracture field
        # to bias crack normals along natural ridge / curvature lines.
        self._principal_tangent: Optional[Tensor] = None     # (N, 3) unit
        self._principal_anisotropy: Optional[Tensor] = None  # (N,) in [0, 1]

    def set_normals(self, normals: Tensor) -> None:
        """Store surface normals for normal-aware edge filtering.

        Args:
            normals: (N, 3) unit normals per Gaussian
        """
        self._normals = normals

    @torch.no_grad()
    def compute_curvature_directions(self, positions: Tensor) -> None:
        """Per-particle principal in-shell direction via local tangent-plane PCA.

        For each particle, gather kNN neighbours, project their offsets onto
        the local tangent plane (perpendicular to the stored surface
        normal), and eigendecompose the weighted covariance.  The
        eigenvector of the largest eigenvalue is the in-shell direction of
        maximum local extent -- the natural axis along which thin-shell
        cracks tend to align (ridge / curvature lines).

        Sets:
            self._principal_tangent     (N, 3) unit vectors in the tangent
                                        plane.
            self._principal_anisotropy  (N,) ratio (lambda1 - lambda2) /
                                        (lambda1 + lambda2) clamped to
                                        [0, 1].  Near 1 = strongly
                                        anisotropic (ridge), near 0 =
                                        isotropic (flat region).
        """
        if (self.knn_idx is None
                or self._normals is None
                or self.knn_idx.shape[0] != positions.shape[0]):
            self._principal_tangent = None
            self._principal_anisotropy = None
            return
        K = int(self.knn_idx.shape[1])
        if K < 3:
            self._principal_tangent = None
            self._principal_anisotropy = None
            return

        eps = 1e-8
        N = positions.shape[0]
        nbr = positions[self.knn_idx]                            # (N, K, 3)
        centered = nbr - positions.unsqueeze(1)                  # (N, K, 3)

        # Project out the surface-normal component to get in-plane offsets.
        n = self._normals                                        # (N, 3)
        n_dot = (centered * n.unsqueeze(1)).sum(dim=2, keepdim=True)
        in_plane = centered - n_dot * n.unsqueeze(1)             # (N, K, 3)

        # Use existing kNN Gaussian weights if available, else uniform.
        if self.weights is not None and self.weights.shape == (N, K):
            w = self.weights.clamp(min=0.0)
        else:
            w = torch.full((N, K), 1.0 / K, device=positions.device,
                           dtype=positions.dtype)

        weighted = in_plane * w.unsqueeze(2)                     # (N, K, 3)
        cov = torch.einsum('nki,nkj->nij', in_plane, weighted)   # (N, 3, 3)

        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)            # ascending
        except RuntimeError:
            self._principal_tangent = None
            self._principal_anisotropy = None
            return

        # Largest in-plane eigenvalue is the last; second-largest is -2.
        principal = eigvecs[..., -1]                             # (N, 3)
        principal = principal / principal.norm(dim=1, keepdim=True).clamp(min=eps)

        l1 = eigvals[..., -1]
        l2 = eigvals[..., -2]
        aniso = ((l1 - l2) / (l1 + l2 + eps)).clamp(0.0, 1.0)

        # If a particle has nearly-degenerate covariance (l1 ~ l2), aniso
        # is near 0 and the principal direction is unreliable; downstream
        # blending uses aniso as the gate, so this is safe to leave.
        self._principal_tangent = principal
        self._principal_anisotropy = aniso

    def build(self, positions: Tensor, force: bool = False) -> None:
        """
        Build or rebuild the kNN graph from Gaussian positions.

        If normals are set (via set_normals), edges between Gaussians with
        opposing normals are pruned. This prevents damage from leaking
        through thin geometry to the opposite surface.

        Args:
            positions: (N, 3) Gaussian world-space positions
            force: rebuild even if not due
        """
        N = positions.shape[0]
        if N <= 1:
            self.knn_idx = torch.empty(N, 0, dtype=torch.long, device=positions.device)
            self.knn_dist = torch.empty(N, 0, dtype=positions.dtype, device=positions.device)
            self.weights = torch.empty(N, 0, dtype=positions.dtype, device=positions.device)
            self.N = N
            self._build_count += 1
            return

        # Skip rebuild if not due
        if (not force
                and self.knn_idx is not None
                and self.N == N
                and self.rebuild_every > 0
                and self._build_count % self.rebuild_every != 0):
            self._build_count += 1
            return

        # Over-fetch neighbors so we have enough after pruning
        has_normals = (self._normals is not None
                       and self._normals.shape[0] == N)
        k_fetch = min(self.k * 2 if has_normals else self.k, N - 1)
        k_final = min(self.k, N - 1)

        knn_dist, knn_idx = knn_search(
            positions,
            positions,
            k_fetch,
            exclude_self=True,
        )
        if self._build_count == 0:
            print(f"[Graph] faiss kNN: N={N}, k_fetch={k_fetch}")

        # --- Normal compatibility filtering ---
        if has_normals:
            normals = self._normals  # (N, 3)
            n_i = normals.unsqueeze(1).expand(-1, k_fetch, -1)  # (N, K_fetch, 3)
            n_j = normals[knn_idx]  # (N, K_fetch, 3)

            # dot(n_i, n_j): same-side ≈ +1, opposite-side ≈ -1
            ndot = (n_i * n_j).sum(dim=2)  # (N, K_fetch)

            # Also check edge direction vs normal: reject if the edge
            # vector is roughly aligned with the normal (= goes "through"
            # the surface rather than along it).
            edge_dir = positions[knn_idx] - positions.unsqueeze(1)  # (N,K,3)
            edge_len = edge_dir.norm(dim=2).clamp(min=1e-8)
            edge_unit = edge_dir / edge_len.unsqueeze(2)
            edge_normal_align = (edge_unit * n_i).sum(dim=2).abs()  # (N,K)

            # Keep edge if normals agree AND edge is roughly tangent
            # ndot > threshold  AND  |edge · n| < 0.7
            compatible = (ndot > self.normal_compat_threshold) & (edge_normal_align < 0.7)

            # Set incompatible edges to inf distance so they sort last
            knn_dist = knn_dist.clone()
            knn_dist[~compatible] = float('inf')

            # Re-sort by distance to pick the K best compatible neighbors
            sorted_dist, sort_idx = knn_dist.sort(dim=1)
            knn_idx = knn_idx.gather(1, sort_idx)
            knn_dist = sorted_dist

            # Trim to k_final
            knn_idx = knn_idx[:, :k_final]
            knn_dist = knn_dist[:, :k_final]

            # Edges with inf distance get zero weight (handled below)
            n_pruned = (knn_dist == float('inf')).sum().item()
            if self._build_count == 0:
                n_total = N * k_final
                print(f"[Graph] Normal filter: {n_pruned}/{n_total} edges pruned "
                      f"({100*n_pruned/max(n_total,1):.1f}%)")
        else:
            knn_idx = knn_idx[:, :k_final]
            knn_dist = knn_dist[:, :k_final]

        # Gaussian kernel weights: w_ij = exp(-d_ij^2 / (2*sigma^2))
        # inf-distance edges get weight ≈ 0
        weights = torch.exp(-knn_dist.clamp(max=100.0) ** 2 / (2.0 * self.sigma ** 2))
        weights[knn_dist >= 1e6] = 0.0  # explicitly zero out pruned edges
        # Row-normalize
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        self.knn_idx = knn_idx
        self.knn_dist = knn_dist
        self.weights = weights
        self.N = N
        self._build_count += 1

    def graph_laplacian(self, values: Tensor) -> Tensor:
        """
        Compute graph Laplacian: L[i] = sum_j w_ij * (values[j] - values[i])

        This is the weighted graph analog of the continuous Laplacian nabla^2.

        Args:
            values: (N,) scalar field on nodes

        Returns:
            lap: (N,) discrete graph Laplacian
        """
        # Gather neighbor values: (N, K)
        neighbor_vals = values[self.knn_idx]
        # Weighted difference
        diff = neighbor_vals - values.unsqueeze(1)  # (N, K)
        lap = (self.weights * diff).sum(dim=1)  # (N,)
        return lap

    def graph_gradient(self, values: Tensor, positions: Tensor) -> Tensor:
        """
        Compute approximate graph gradient of a scalar field.

        Uses weighted least-squares on the kNN neighborhood:
        grad[i] ≈ sum_j w_ij * (v_j - v_i) * (x_j - x_i) / |x_j - x_i|^2

        Args:
            values: (N,) scalar field
            positions: (N, 3) node positions

        Returns:
            grad: (N, 3) gradient vectors
        """
        # Neighbor positions and values
        nbr_pos = positions[self.knn_idx]   # (N, K, 3)
        nbr_val = values[self.knn_idx]       # (N, K)

        # Differences
        dx = nbr_pos - positions.unsqueeze(1)  # (N, K, 3)
        dv = nbr_val - values.unsqueeze(1)     # (N, K)

        # Distance squared
        dist_sq = (dx ** 2).sum(dim=2).clamp(min=1e-12)  # (N, K)

        # Weighted gradient: sum w * dv * dx / |dx|^2
        coeff = self.weights * dv / dist_sq  # (N, K)
        grad = (coeff.unsqueeze(2) * dx).sum(dim=1)  # (N, 3)

        return grad

    def gather_neighbors(self, values: Tensor) -> Tensor:
        """
        Gather neighbor values for arbitrary per-node tensor.

        Args:
            values: (N,) or (N, D) field on nodes

        Returns:
            (N, K) or (N, K, D) neighbor values
        """
        return values[self.knn_idx]

    def weighted_neighbor_max(self, values: Tensor) -> Tensor:
        """
        Weighted maximum over neighbors.
        Useful for directional crack propagation.

        Args:
            values: (N,) scalar field

        Returns:
            (N,) weighted max of neighbor values
        """
        nbr_vals = values[self.knn_idx]  # (N, K)
        return nbr_vals.max(dim=1).values

    def edge_damage_strength(self, damage: Tensor) -> Tensor:
        """
        Compute edge damage strength for fragment detection.
        An edge is "broken" when either endpoint has high damage.

        Args:
            damage: (N,) per-node damage values

        Returns:
            (N, K) edge connectivity strength in [0, 1]
                    1 = fully connected, 0 = fully broken
        """
        c_i = damage.unsqueeze(1).expand_as(self.knn_idx.float())  # (N, K)
        c_j = damage[self.knn_idx]  # (N, K)
        # Edge breaks when max(c_i, c_j) is high
        edge_damage = torch.maximum(c_i, c_j)
        connectivity = (1.0 - edge_damage).clamp(0.0, 1.0)
        return connectivity
