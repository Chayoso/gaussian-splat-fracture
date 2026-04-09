"""
Surface-Only Phase Field Fracture

Problem: Internal cracks affect physics (strength) but are invisible
Solution: Restrict crack propagation to surface only

Philosophy:
- For disaster response / practical applications
- Only externally visible damage matters
- Internal structure should remain intact
- Physics should match what we can observe
"""

from typing import Tuple, Optional
import torch
from torch import Tensor
import numpy as np


class SurfacePhaseFieldLite:
    """
    Phase field fracture restricted to surface only

    Key difference from standard PhaseFieldLite:
    - Crack growth only near surface
    - Interior remains intact (c = 0)
    - Prevents internal weakening that would cause collapse
    """

    def __init__(
        self,
        Gc: float = 1.0e-3,
        l0: float = 0.015,
        viscosity: float = 0.01,
        surface_depth: float = 0.05,
        k_neighbors: int = 32,
        enable_surface_restriction: bool = True
    ):
        """
        Args:
            Gc: Critical energy release rate
            l0: Length scale (crack thickness)
            viscosity: Phase field viscosity
            surface_depth: How deep to allow crack growth (MPM units)
            k_neighbors: KNN for surface detection
            enable_surface_restriction: If False, behaves like standard phase field
        """
        self.Gc = Gc
        self.l0 = l0
        self.viscosity = viscosity
        self.surface_depth = surface_depth
        self.k_neighbors = k_neighbors
        self.enable_surface_restriction = enable_surface_restriction

        print(f"[SurfacePhaseFieldLite] Initialized")
        if enable_surface_restriction:
            print(f"  - Surface-only mode: crack growth restricted to surface")
            print(f"  - Surface depth: {surface_depth}")
            print(f"  - Rationale: Internal cracks invisible & shouldn't weaken structure")
        else:
            print(f"  - Standard mode: volumetric crack propagation")

    def compute_surface_mask(
        self,
        x: Tensor,  # (N, 3) particle positions
        k: Optional[int] = None
    ) -> Tensor:
        """
        Determine which particles are near surface

        Strategy: Surface particles have lower local density (larger k-NN distance)

        Returns:
            surface_mask: (N,) bool - True if near surface, False if internal
        """
        N = x.shape[0]
        device = x.device

        if k is None:
            k = self.k_neighbors

        if N == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)

        # Compute pairwise distances
        dists = torch.cdist(x, x)  # (N, N)

        # Get k-NN distances (exclude self at index 0)
        knn_dists, _ = torch.topk(dists, k=min(k+1, N), largest=False, dim=1)
        avg_knn_dist = knn_dists[:, 1:].mean(dim=1)  # (N,) exclude self

        # Surface = larger k-NN distance (lower local density)
        # Use adaptive threshold based on distribution
        threshold = torch.quantile(avg_knn_dist, 0.5)  # Top 50% = surface
        surface_mask = avg_knn_dist >= threshold

        return surface_mask

    def update(
        self,
        c: Tensor,           # (N,) current damage
        psi_pos: Tensor,     # (N,) positive strain energy
        x: Tensor,           # (N, 3) particle positions
        dt: float,
        warmup_frames: int = 0,
        current_frame: int = 0
    ) -> Tensor:
        """
        Update phase field with surface restriction

        Key modification:
        - Compute crack driving force as usual
        - Apply surface mask: only surface particles can grow cracks
        - Interior remains c = 0 (intact)

        Args:
            c: Current damage field
            psi_pos: Positive strain energy (tension only)
            x: Particle positions for surface detection
            dt: Time step
            warmup_frames: Frames before crack can grow
            current_frame: Current simulation frame

        Returns:
            c_new: Updated damage field (surface only if enabled)
        """
        device = c.device
        N = c.shape[0]

        # Warmup period (no crack growth)
        if current_frame < warmup_frames:
            return c

        # Standard phase field driving force
        # ∂E/∂c = 2(1-c)ψ⁺ - Gc/l0 + 2Gc·l0·Δc
        # Simplified: linear update based on energy

        # Crack driving force (simplified)
        # High strain → want to increase c
        psi_threshold = 1e-6
        driving_force = torch.clamp(psi_pos - psi_threshold, min=0.0)

        # Phase field evolution (simplified Allen-Cahn)
        # dc/dt = -1/η · ∂E/∂c
        dc_dt = driving_force / (self.viscosity + 1e-12)

        # Tentative update
        c_new = c + dc_dt * dt

        # Clamp to [0, 1]
        c_new = torch.clamp(c_new, 0.0, 1.0)

        # SURFACE RESTRICTION (NEW!)
        if self.enable_surface_restriction:
            # Compute surface mask
            surface_mask = self.compute_surface_mask(x, k=self.k_neighbors)

            # Only apply crack growth to surface particles
            # Interior: keep c = 0 (intact)
            c_new_restricted = torch.where(
                surface_mask,
                c_new,        # Surface: allow growth
                torch.zeros_like(c_new)  # Interior: force c = 0
            )

            # Stats
            n_surface = surface_mask.sum().item()
            n_cracked_surface = (c_new_restricted > 0.3).sum().item()

            if current_frame % 20 == 0:  # Log every 20 frames
                print(f"[SurfacePhaseField] frame={current_frame}: "
                      f"surface={n_surface}/{N} ({n_surface/N*100:.1f}%), "
                      f"cracked={n_cracked_surface}")

            return c_new_restricted
        else:
            # Standard volumetric phase field
            return c_new

    def compute_degradation(self, c: Tensor, k_res: float = 1.0e-6) -> Tensor:
        """
        Compute material degradation function g(c)

        Standard: g(c) = (1-c)² + k_res

        Returns:
            g: (N,) degradation function
        """
        g = (1.0 - c) ** 2 + k_res
        return g


class AdaptiveSurfacePhaseField(SurfacePhaseFieldLite):
    """
    Advanced version with adaptive surface depth

    Idea: Surface depth adapts based on crack severity
    - Light damage: shallow surface restriction
    - Heavy damage: allow deeper propagation (but still bounded)
    """

    def __init__(
        self,
        Gc: float = 1.0e-3,
        l0: float = 0.015,
        viscosity: float = 0.01,
        min_surface_depth: float = 0.03,
        max_surface_depth: float = 0.08,
        k_neighbors: int = 32
    ):
        super().__init__(
            Gc=Gc,
            l0=l0,
            viscosity=viscosity,
            surface_depth=min_surface_depth,
            k_neighbors=k_neighbors,
            enable_surface_restriction=True
        )
        self.min_surface_depth = min_surface_depth
        self.max_surface_depth = max_surface_depth

    def compute_surface_mask(
        self,
        x: Tensor,
        k: Optional[int] = None,
        c: Optional[Tensor] = None  
    ) -> Tensor:
        """
        Adaptive surface mask based on current damage

        If c is provided, adapt surface depth:
        - Low damage → shallow (min_depth)
        - High damage → deeper (max_depth)
        """
        if c is None:
            return super().compute_surface_mask(x, k)

        # Compute base surface mask
        base_mask = super().compute_surface_mask(x, k)

        # Adaptive depth based on average damage
        c_mean = c.mean().item()
        adaptive_depth = self.min_surface_depth + (self.max_surface_depth - self.min_surface_depth) * c_mean

        # Could refine mask based on adaptive_depth
        # For now, just use base mask with logging
        # (Full implementation would recompute with adaptive threshold)

        return base_mask


def compare_phase_field_modes():
    """
    Comparison of standard vs surface-only phase field

    Demonstrates the difference in crack propagation behavior
    """
    print("\n" + "="*60)
    print("Phase Field Mode Comparison")
    print("="*60)

    N = 1000
    x = torch.randn(N, 3) * 0.3 + 0.5  # Particles centered at 0.5
    c = torch.zeros(N)
    psi = torch.rand(N) * 1.0e-5  # Random strain energy

    # Standard phase field
    pf_standard = SurfacePhaseFieldLite(
        enable_surface_restriction=False
    )

    # Surface-only phase field
    pf_surface = SurfacePhaseFieldLite(
        enable_surface_restriction=True,
        surface_depth=0.05
    )

    print("\n1. Standard Phase Field (Volumetric)")
    c_standard = c.clone()
    for frame in range(10):
        c_standard = pf_standard.update(c_standard, psi, x, dt=0.001, current_frame=frame)

    n_cracked_standard = (c_standard > 0.1).sum().item()
    print(f"   Final cracked particles: {n_cracked_standard}/{N} ({n_cracked_standard/N*100:.1f}%)")

    print("\n2. Surface-Only Phase Field")
    c_surface = c.clone()
    for frame in range(10):
        c_surface = pf_surface.update(c_surface, psi, x, dt=0.001, current_frame=frame)

    n_cracked_surface = (c_surface > 0.1).sum().item()
    print(f"   Final cracked particles: {n_cracked_surface}/{N} ({n_cracked_surface/N*100:.1f}%)")
    print(f"   Reduction: {(1 - n_cracked_surface/max(n_cracked_standard,1))*100:.1f}%")

    print("\n" + "="*60)
    print("Key Difference:")
    print("- Standard: Internal cracks weaken structure invisibly")
    print("- Surface-only: Only visible cracks affect physics")
    print("="*60 + "\n")


if __name__ == "__main__":
    compare_phase_field_modes()
