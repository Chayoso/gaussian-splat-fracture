"""
Volumetric-to-Surface Damage Mapper

Projects volumetric phase field damage (from MPM particles) to surface particles
(for Gaussian Splatting visualization).

Strategy: KNN weighted average with GPU acceleration
"""

import torch
from torch import Tensor
from typing import Optional
import numpy as np


class VolumetricToSurfaceDamageMapper:
    """
    Project volumetric damage field to surface particles

    Methods:
    - knn_weighted: K-nearest neighbor with Gaussian weighting (recommended)
    - direct: Extract surface damage directly (if surface-only phase field)
    - ray_casting: Ray-based accumulation (experimental, slower)
    """

    def __init__(
        self,
        projection_method: str = "knn_weighted",
        k_neighbors: int = 8,
        influence_radius: float = 0.05,  # MPM grid units
        damage_threshold: float = 0.01,   # Ignore c < threshold
        use_faiss: bool = False,           # GPU acceleration (requires FAISS)
        device: torch.device = None
    ):
        """
        Args:
            projection_method: "knn_weighted", "direct", or "ray_casting"
            k_neighbors: Number of nearest neighbors for KNN
            influence_radius: Gaussian kernel width (σ in MPM space)
            damage_threshold: Minimum damage to consider (noise reduction)
            use_faiss: Use FAISS for GPU-accelerated KNN (faster for N > 10K)
            device: PyTorch device
        """
        if projection_method not in ["knn_weighted", "direct", "ray_casting"]:
            raise ValueError(f"Unknown method: {projection_method}")

        self.method = projection_method
        self.k = k_neighbors
        self.sigma = influence_radius
        self.threshold = damage_threshold
        self.use_faiss = use_faiss
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # FAISS index (lazy initialization)
        self.faiss_index = None

        print(f"[DamageMapper] Initialized")
        print(f"  - Method: {projection_method}")
        print(f"  - K-neighbors: {k_neighbors}")
        print(f"  - Influence radius (σ): {influence_radius}")
        print(f"  - Use FAISS: {use_faiss}")
        print(f"  - Device: {self.device}")

    def project_damage(
        self,
        c_volumetric: Tensor,  # (N_mpm,) damage for all MPM particles
        x_mpm: Tensor,         # (N_mpm, 3) all MPM particle positions [0, 1]³
        x_surface: Tensor,     # (N_surf, 3) surface particle positions [0, 1]³
        surface_mask: Optional[Tensor] = None  # (N_mpm,) bool - which particles are surface
    ) -> Tensor:
        """
        Project volumetric damage to surface particles

        Args:
            c_volumetric: (N_mpm,) damage values for all particles
            x_mpm: (N_mpm, 3) positions of all MPM particles
            x_surface: (N_surf, 3) positions of surface particles
            surface_mask: Optional mask indicating surface particles in x_mpm

        Returns:
            c_surface: (N_surf,) projected damage values for surface particles
        """
        if self.method == "direct":
            return self._direct_projection(c_volumetric, surface_mask)
        elif self.method == "knn_weighted":
            return self._knn_weighted_projection(c_volumetric, x_mpm, x_surface)
        elif self.method == "ray_casting":
            return self._ray_casting_projection(c_volumetric, x_mpm, x_surface)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

    def _direct_projection(
        self,
        c_vol: Tensor,
        surface_mask: Tensor
    ) -> Tensor:
        """
        Direct extraction: surface particles already in MPM volume

        Args:
            c_vol: (N_mpm,) all particle damage
            surface_mask: (N_mpm,) boolean mask for surface particles

        Returns:
            c_surf: (N_surf,) surface damage
        """
        if surface_mask is None:
            raise ValueError("Direct projection requires surface_mask")

        return c_vol[surface_mask]

    def _knn_weighted_projection(
        self,
        c_vol: Tensor,
        x_mpm: Tensor,
        x_surf: Tensor
    ) -> Tensor:
        """
        KNN-weighted projection: c_surf[i] = Σ w_ij * c_vol[j]
        where w_ij = exp(-d_ij² / (2σ²)) / Σ_k exp(-d_ik² / (2σ²))

        Args:
            c_vol: (N_mpm,) volumetric damage
            x_mpm: (N_mpm, 3) MPM particle positions
            x_surf: (N_surf, 3) surface particle positions

        Returns:
            c_surf: (N_surf,) projected surface damage
        """
        device = c_vol.device
        N_surf = x_surf.shape[0]
        N_mpm = x_mpm.shape[0]

        # Use FAISS for large datasets
        if self.use_faiss and N_mpm > 5000:
            return self._knn_weighted_faiss(c_vol, x_mpm, x_surf)

        # PyTorch implementation (works for moderate sizes)
        # Compute pairwise distances (N_surf, N_mpm)
        dists = torch.cdist(x_surf, x_mpm)  # Euclidean distance

        # Find k-nearest neighbors
        k_actual = min(self.k, N_mpm)
        knn_dists, knn_indices = torch.topk(dists, k=k_actual, largest=False, dim=1)

        # Gaussian weighting: w = exp(-d² / (2σ²))
        weights = torch.exp(-knn_dists**2 / (2 * self.sigma**2))  # (N_surf, k)

        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        # Gather damage values from k-nearest neighbors
        c_knn = c_vol[knn_indices]  # (N_surf, k)

        # Weighted sum
        c_surf = (weights * c_knn).sum(dim=1)  # (N_surf,)

        # Apply damage threshold (noise reduction)
        c_surf = torch.where(c_surf > self.threshold, c_surf, torch.zeros_like(c_surf))

        return c_surf

    def _knn_weighted_faiss(
        self,
        c_vol: Tensor,
        x_mpm: Tensor,
        x_surf: Tensor
    ) -> Tensor:
        """
        FAISS-accelerated KNN weighted projection

        Requires: pip install faiss-gpu
        """
        try:
            import faiss
        except ImportError:
            print("[Warning] FAISS not available, falling back to PyTorch")
            return self._knn_weighted_projection(c_vol, x_mpm, x_surf)

        device = c_vol.device
        N_surf = x_surf.shape[0]

        # Convert to numpy for FAISS
        x_mpm_np = x_mpm.detach().cpu().numpy().astype(np.float32)
        x_surf_np = x_surf.detach().cpu().numpy().astype(np.float32)

        # Build FAISS index (GPU if available)
        if device.type == 'cuda' and hasattr(faiss, 'StandardGpuResources'):
            # GPU FAISS
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, 3)  # 3D points, L2 distance
        else:
            # CPU FAISS
            index = faiss.IndexFlatL2(3)

        # Add MPM particles to index
        index.add(x_mpm_np)

        # Query k-nearest neighbors
        k_actual = min(self.k, x_mpm_np.shape[0])
        distances, indices = index.search(x_surf_np, k_actual)  # (N_surf, k)

        # Convert back to PyTorch
        distances = torch.from_numpy(distances).to(device)
        indices = torch.from_numpy(indices).long().to(device)

        # Gaussian weighting
        weights = torch.exp(-distances / (2 * self.sigma**2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        # Gather and weight damage values
        c_knn = c_vol[indices]  # (N_surf, k)
        c_surf = (weights * c_knn).sum(dim=1)

        # Apply threshold
        c_surf = torch.where(c_surf > self.threshold, c_surf, torch.zeros_like(c_surf))

        return c_surf

    def _ray_casting_projection(
        self,
        c_vol: Tensor,
        x_mpm: Tensor,
        x_surf: Tensor
    ) -> Tensor:
        """
        Ray casting projection: cast rays from surface toward center,
        accumulate damage along ray

        More accurate for deep cracks but slower

        Args:
            c_vol: (N_mpm,) volumetric damage
            x_mpm: (N_mpm, 3) MPM particle positions
            x_surf: (N_surf, 3) surface particle positions

        Returns:
            c_surf: (N_surf,) projected surface damage
        """
        device = c_vol.device
        N_surf = x_surf.shape[0]

        # Domain center (assume [0.5, 0.5, 0.5])
        center = torch.tensor([0.5, 0.5, 0.5], device=device)

        c_surf = torch.zeros(N_surf, device=device)

        # Process in batches to avoid memory issues
        batch_size = 100
        for i in range(0, N_surf, batch_size):
            j = min(i + batch_size, N_surf)
            origins = x_surf[i:j]  # (batch, 3)

            # Ray directions (toward center)
            directions = center.unsqueeze(0) - origins  # (batch, 3)
            ray_lengths = directions.norm(dim=1, keepdim=True)
            directions = directions / (ray_lengths + 1e-12)

            # Sample points along rays
            n_samples = 20
            t_samples = torch.linspace(0, 1, n_samples, device=device).view(1, n_samples, 1)
            ray_points = origins.unsqueeze(1) + directions.unsqueeze(1) * ray_lengths.unsqueeze(1) * t_samples
            # ray_points: (batch, n_samples, 3)

            # Find nearest MPM particle for each ray point
            ray_points_flat = ray_points.reshape(-1, 3)  # (batch * n_samples, 3)
            dists_to_mpm = torch.cdist(ray_points_flat, x_mpm)  # (batch * n_samples, N_mpm)
            nearest_indices = dists_to_mpm.argmin(dim=1)  # (batch * n_samples,)

            # Accumulate damage along ray
            c_along_ray = c_vol[nearest_indices].reshape(-1, n_samples)  # (batch, n_samples)
            c_surf[i:j] = c_along_ray.max(dim=1)[0]  # Maximum damage along ray

        return c_surf

    def to(self, device: torch.device):
        """Move mapper to specified device"""
        self.device = device
        return self

    def __repr__(self) -> str:
        return (f"VolumetricToSurfaceDamageMapper(method='{self.method}', "
                f"k={self.k}, sigma={self.sigma})")


def test_damage_mapper():
    """Test damage projection with synthetic data"""
    print("Testing VolumetricToSurfaceDamageMapper...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create synthetic volumetric damage field
    N_mpm = 1000
    N_surf = 300

    x_mpm = torch.rand(N_mpm, 3, device=device)  # Random MPM particles
    c_vol = torch.rand(N_mpm, device=device)     # Random damage

    # Create localized high damage region (center of domain)
    center_mask = (x_mpm - 0.5).norm(dim=1) < 0.2  # Within radius 0.2
    c_vol[center_mask] = torch.rand(center_mask.sum(), device=device) * 0.5 + 0.5  # High damage

    # Surface particles (first 300 particles)
    surface_mask = torch.zeros(N_mpm, dtype=torch.bool, device=device)
    surface_mask[:N_surf] = True
    x_surf = x_mpm[:N_surf]

    # Test KNN projection
    mapper = VolumetricToSurfaceDamageMapper(
        projection_method="knn_weighted",
        k_neighbors=8,
        influence_radius=0.05,
        device=device
    )

    c_surf = mapper.project_damage(c_vol, x_mpm, x_surf, surface_mask)

    # Verify
    assert c_surf.shape[0] == N_surf, "Incorrect output size"
    assert c_surf.min() >= 0.0 and c_surf.max() <= 1.0, "Damage out of bounds"
    assert not torch.isnan(c_surf).any(), "NaN in projected damage"

    print(f"✓ Projection successful")
    print(f"  - Input damage: mean={c_vol.mean():.4f}, max={c_vol.max():.4f}")
    print(f"  - Output damage: mean={c_surf.mean():.4f}, max={c_surf.max():.4f}")

    # Test direct projection
    mapper_direct = VolumetricToSurfaceDamageMapper(projection_method="direct", device=device)
    c_surf_direct = mapper_direct.project_damage(c_vol, x_mpm, x_surf, surface_mask)

    assert torch.allclose(c_surf_direct, c_vol[:N_surf]), "Direct projection mismatch"
    print(f"✓ Direct projection verified")

    print("✓ All tests passed!\n")


if __name__ == "__main__":
    test_damage_mapper()
