"""
Physics Projector: MPM → Gaussian Signal Transfer

Projects physics-derived driving signals (tensile strain energy, principal
stress directions, deformation gradients) from MPM particles to Gaussians.

Replaces the old damage_mapper's role: instead of projecting scalar damage,
we project the raw physics driving force so that fracture evolution happens
on the Gaussian manifold.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple

from src.utils.knn import knn_search


class PhysicsProjector:
    """
    Project physics signals from MPM volumetric particles to Gaussian nodes.

    Transfers:
        - Tensile strain energy ψ⁺ (scalar driving force)
        - Principal stress direction n₁ (crack orientation cue)
        - Deformation gradient F (for Gaussian deformation)

    Uses kNN weighted interpolation (same math as old DamageMapper,
    but applied to driving signals instead of damage).
    """

    def __init__(
        self,
        k_neighbors: int = 8,
        influence_radius: float = 0.05,
        device: str = "cuda",
    ):
        """
        Args:
            k_neighbors: K for kNN interpolation
            influence_radius: Gaussian kernel sigma (MPM space units)
            device: torch device
        """
        self.k = k_neighbors
        self.sigma = influence_radius
        self.device = torch.device(device)

        # Cached kNN index for repeated queries with same geometry
        self._cached_knn_idx: Optional[Tensor] = None
        self._cached_knn_weights: Optional[Tensor] = None
        self._cache_frame: int = -1

    def project_scalar(
        self,
        values: Tensor,
        x_mpm: Tensor,
        x_gaussian: Tensor,
        frame: int = -1,
    ) -> Tensor:
        """
        Project scalar field from MPM particles to Gaussians via kNN.

        Args:
            values: (N_mpm,) scalar values on MPM particles
            x_mpm: (N_mpm, 3) MPM particle positions [0,1]^3
            x_gaussian: (N_gauss, 3) Gaussian positions (MPM space)
            frame: current frame for cache invalidation

        Returns:
            (N_gauss,) interpolated scalar values
        """
        knn_idx, weights = self._get_knn_weights(x_mpm, x_gaussian, frame)
        # Gather and weight
        vals_k = values[knn_idx]  # (N_gauss, K)
        return (weights * vals_k).sum(dim=1)

    def project_vector(
        self,
        vectors: Tensor,
        x_mpm: Tensor,
        x_gaussian: Tensor,
        frame: int = -1,
    ) -> Tensor:
        """
        Project vector field from MPM particles to Gaussians.

        Args:
            vectors: (N_mpm, 3) vector values on MPM particles
            x_mpm: (N_mpm, 3) MPM particle positions
            x_gaussian: (N_gauss, 3) Gaussian positions (MPM space)
            frame: current frame for cache invalidation

        Returns:
            (N_gauss, 3) interpolated vector values
        """
        knn_idx, weights = self._get_knn_weights(x_mpm, x_gaussian, frame)
        # Gather: (N_gauss, K, 3)
        vecs_k = vectors[knn_idx]
        # Weighted average: (N_gauss, 3)
        return (weights.unsqueeze(2) * vecs_k).sum(dim=1)

    def project_matrix(
        self,
        matrices: Tensor,
        x_mpm: Tensor,
        x_gaussian: Tensor,
        frame: int = -1,
    ) -> Tensor:
        """
        Project 3x3 matrix field (e.g., deformation gradient F) to Gaussians.

        Args:
            matrices: (N_mpm, 3, 3) matrix values on MPM particles
            x_mpm: (N_mpm, 3) positions
            x_gaussian: (N_gauss, 3) Gaussian positions (MPM space)
            frame: current frame for cache invalidation

        Returns:
            (N_gauss, 3, 3) interpolated matrices
        """
        knn_idx, weights = self._get_knn_weights(x_mpm, x_gaussian, frame)
        # Gather: (N_gauss, K, 3, 3)
        mats_k = matrices[knn_idx]
        # Weighted average
        w = weights.unsqueeze(2).unsqueeze(3)  # (N_gauss, K, 1, 1)
        return (w * mats_k).sum(dim=1)

    def project_principal_stress_direction(
        self,
        stress: Tensor,
        x_mpm: Tensor,
        x_gaussian: Tensor,
        frame: int = -1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute and project the principal tensile stress direction.

        For each MPM particle, eigendecompose the stress tensor,
        take the eigenvector of the largest positive eigenvalue.
        Then project to Gaussians via kNN.

        Args:
            stress: (N_mpm, 3, 3) Kirchhoff stress tensors
            x_mpm: (N_mpm, 3) positions
            x_gaussian: (N_gauss, 3) Gaussian positions (MPM space)
            frame: current frame

        Returns:
            n1: (N_gauss, 3) principal tensile direction (unit vectors)
            sigma1: (N_gauss,) principal tensile stress magnitude
        """
        # Symmetrize stress
        S = 0.5 * (stress + stress.transpose(1, 2))
        S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e8, 1e8)
        eigenvalues, eigenvectors = self._symmetric_eigh_chunked(S)

        # Largest eigenvalue and its eigenvector
        sigma1_mpm = eigenvalues[:, -1].clamp(min=0.0)  # (N_mpm,)
        n1_mpm = eigenvectors[:, :, -1]  # (N_mpm, 3)

        # Only keep directions where the local principal stress is tensile.
        tensile_mask = sigma1_mpm > 0.0
        n1_mpm = torch.where(
            tensile_mask.unsqueeze(1),
            torch.nn.functional.normalize(n1_mpm, dim=-1),
            torch.zeros_like(n1_mpm),
        )

        # Project to Gaussians
        sigma1 = self.project_scalar(sigma1_mpm, x_mpm, x_gaussian, frame)
        n1 = self.project_vector(n1_mpm, x_mpm, x_gaussian, frame)
        n1_norm = n1.norm(dim=-1, keepdim=True)
        n1 = torch.where(
            n1_norm > 1e-8,
            n1 / n1_norm.clamp(min=1e-8),
            torch.zeros_like(n1),
        )

        return n1, sigma1

    @staticmethod
    def _diagonal_principal_fallback(S: Tensor) -> Tuple[Tensor, Tensor]:
        """Return a bounded principal direction from the largest diagonal term."""
        diag = torch.diagonal(S, dim1=1, dim2=2)
        values, axis = diag.max(dim=1)
        vectors = torch.zeros_like(S[:, :, 0])
        vectors.scatter_(1, axis.unsqueeze(1), 1.0)
        return values.unsqueeze(1), vectors.unsqueeze(2)

    @classmethod
    def _symmetric_eigh_chunked(
        cls,
        S: Tensor,
        *,
        chunk_size: int = 8192,
    ) -> Tuple[Tensor, Tensor]:
        """
        Eigendecompose many 3x3 symmetric tensors without one huge cuSolver batch.

        At 50K surface particles, a single batched CUDA `eigh` can fail before
        the fracture code runs. Chunking keeps the normal principal-stress path
        intact. If a chunk still fails, only that chunk degrades to a diagonal
        principal-axis approximation instead of aborting the whole sweep.
        """
        if S.ndim != 3 or S.shape[1:] != (3, 3):
            raise ValueError(f"expected stress tensor shape (N,3,3), got {tuple(S.shape)}")
        values_out = []
        vectors_out = []
        n = int(S.shape[0])
        for start in range(0, n, max(int(chunk_size), 1)):
            chunk = S[start:start + chunk_size]
            try:
                values, vectors = torch.linalg.eigh(chunk)
            except RuntimeError:
                max_values, max_vectors = cls._diagonal_principal_fallback(chunk)
                min_values = torch.zeros_like(max_values)
                values = torch.cat([min_values, min_values, max_values], dim=1)
                zero = torch.zeros_like(max_vectors)
                vectors = torch.cat([zero, zero, max_vectors], dim=2)
            values_out.append(values)
            vectors_out.append(vectors)
        return torch.cat(values_out, dim=0), torch.cat(vectors_out, dim=0)

    def _get_knn_weights(
        self,
        x_mpm: Tensor,
        x_gaussian: Tensor,
        frame: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute or retrieve cached kNN indices and weights.

        Returns:
            knn_idx: (N_gauss, K) indices into x_mpm
            weights: (N_gauss, K) normalized Gaussian weights
        """
        # Use cache if same frame
        if (self._cached_knn_idx is not None
                and frame == self._cache_frame
                and frame >= 0):
            return self._cached_knn_idx, self._cached_knn_weights

        N_mpm = x_mpm.shape[0]
        k = min(self.k, N_mpm)

        knn_dist, knn_idx = knn_search(x_gaussian, x_mpm, k)

        # Gaussian kernel weights
        weights = torch.exp(-knn_dist ** 2 / (2.0 * self.sigma ** 2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        # Cache
        self._cached_knn_idx = knn_idx
        self._cached_knn_weights = weights
        self._cache_frame = frame

        return knn_idx, weights
