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
import numpy as np


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
        device = stress.device
        N = stress.shape[0]

        # Symmetrize stress
        S = 0.5 * (stress + stress.transpose(1, 2))

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(S)
        except Exception:
            # Fallback: return zero directions
            N_g = x_gaussian.shape[0]
            return (torch.zeros(N_g, 3, device=device),
                    torch.zeros(N_g, device=device))

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

        N_gauss = x_gaussian.shape[0]
        N_mpm = x_mpm.shape[0]
        k = min(self.k, N_mpm)
        device = x_mpm.device

        # Compute kNN (chunked for memory)
        if N_gauss <= 20000:
            dists = torch.cdist(x_gaussian, x_mpm)  # (N_gauss, N_mpm)
            knn_dist, knn_idx = dists.topk(k, largest=False, dim=1)
        else:
            knn_idx = torch.empty(N_gauss, k, dtype=torch.long, device=device)
            knn_dist = torch.empty(N_gauss, k, device=device)
            chunk = 4096
            for i in range(0, N_gauss, chunk):
                j = min(i + chunk, N_gauss)
                d = torch.cdist(x_gaussian[i:j], x_mpm)
                dist_c, idx_c = d.topk(k, largest=False, dim=1)
                knn_idx[i:j] = idx_c
                knn_dist[i:j] = dist_c

        # Gaussian kernel weights
        weights = torch.exp(-knn_dist ** 2 / (2.0 * self.sigma ** 2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        # Cache
        self._cached_knn_idx = knn_idx
        self._cached_knn_weights = weights
        self._cache_frame = frame

        return knn_idx, weights
