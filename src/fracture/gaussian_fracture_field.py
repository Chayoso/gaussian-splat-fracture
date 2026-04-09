"""
Gaussian Fracture Field

Phase-field-inspired fracture propagation on the Gaussian manifold.
Evolves damage, crack normals, and opening magnitudes directly on
Gaussian nodes using graph-based Laplacian coupling.

This replaces the volumetric AT2 PDE + damage projection pipeline
with a unified, render-native fracture representation.

Per-Gaussian fracture state:
    c_i  : damage magnitude ∈ [0, 1]
    H_i  : history variable (irreversible driving force)
    n_i  : crack normal direction (3D unit vector)
    a_i  : opening magnitude (scalar displacement)
    f_i  : fragment label (int, assigned by GraphFragmentManager)
"""

import torch
from torch import Tensor
from typing import Optional, Tuple

from .graph_builder import GaussianGraph
from .physics_projector import PhysicsProjector


class GaussianFractureField:
    """
    Graph-based fracture field on Gaussian nodes.

    Update cycle per frame:
        1. Receive physics driving signal (ψ⁺, stress) from PhysicsProjector
        2. Update history H_i = max(H_i, ψ⁺_i)
        3. Build/update kNN graph on Gaussian positions
        4. Evolve damage via graph AT2: c_eq = (H_ratio + l0²·lap_c) / (1 + H_ratio)
        5. Estimate crack normal from damage gradient
        6. Compute opening magnitude from damage + deformation
    """

    def __init__(
        self,
        Gc: float = 60000.0,
        l0: float = 0.025,
        dC_max: float = 0.015,
        warmup_frames: int = 5,
        aniso_ratio: float = 3.0,
        opening_scale: float = 0.02,
        damage_source_scale: float = 0.35,
        damage_spread: float = 0.18,
        drive_quantile: float = 0.90,
        front_threshold: float = 0.05,
        radial_bias: float = 2.5,
        tip_propagation_scale: float = 0.75,
        graph: Optional[GaussianGraph] = None,
        device: str = "cuda",
    ):
        """
        Args:
            Gc: fracture toughness
            l0: regularization length (graph analog of AT2 l0)
            dC_max: max damage increment per step (rate limiter)
            warmup_frames: skip H accumulation during initial transient
            aniso_ratio: anisotropy factor for directional propagation
            opening_scale: scaling factor for crack opening displacement
            damage_source_scale: gain for physics-driven damage advance
            damage_spread: gain for crack-front propagation on the graph
            drive_quantile: quantile used to remove broad low-level drive
            front_threshold: minimum local/front damage needed to advance
            radial_bias: prefer propagation away from the seed/impact center
            tip_propagation_scale: aggressive crack-tip advance along favored edges
            graph: pre-built GaussianGraph (created if None)
            device: torch device
        """
        self.Gc = Gc
        self.l0 = l0
        self.dC_max = dC_max
        self.warmup_frames = warmup_frames
        self.aniso_ratio = aniso_ratio
        self.opening_scale = opening_scale
        self.damage_source_scale = damage_source_scale
        self.damage_spread = damage_spread
        self.drive_quantile = drive_quantile
        self.front_threshold = front_threshold
        self.radial_bias = radial_bias
        self.tip_propagation_scale = tip_propagation_scale
        self.device = torch.device(device)

        # Graph (shared or owned)
        self.graph = graph or GaussianGraph(device=device)

        # Per-Gaussian fracture state (initialized lazily)
        self.c: Optional[Tensor] = None       # (N,) damage
        self.H: Optional[Tensor] = None       # (N,) history variable
        self.n: Optional[Tensor] = None       # (N, 3) crack normal
        self.a: Optional[Tensor] = None       # (N,) opening magnitude
        self.f: Optional[Tensor] = None       # (N,) fragment labels
        self.seed_center: Optional[Tensor] = None

        self._frame_count: int = 0
        self._initialized: bool = False

    def initialize(self, N: int) -> None:
        """
        Initialize fracture state for N Gaussians.

        Args:
            N: number of Gaussians
        """
        device = self.device
        self.c = torch.zeros(N, device=device)
        self.H = torch.zeros(N, device=device)
        self.n = torch.zeros(N, 3, device=device)
        self.a = torch.zeros(N, device=device)
        self.f = torch.zeros(N, dtype=torch.long, device=device)
        self.seed_center = None
        self._frame_count = 0
        self._initialized = True

    def update(
        self,
        positions: Tensor,
        psi_plus: Tensor,
        stress: Optional[Tensor] = None,
        stress_dir: Optional[Tensor] = None,
        F_gaussian: Optional[Tensor] = None,
    ) -> None:
        """
        Full per-frame fracture field update.

        Args:
            positions: (N, 3) current Gaussian positions (world or MPM space)
            psi_plus: (N,) tensile strain energy projected to Gaussians
            stress: (N, 3, 3) stress tensor projected to Gaussians (optional)
            stress_dir: (N, 3) principal stress direction (optional, avoids recompute)
            F_gaussian: (N, 3, 3) deformation gradient per Gaussian (optional)
        """
        N = positions.shape[0]

        # Lazy init
        if not self._initialized or self.c.shape[0] != N:
            self.initialize(N)

        self._frame_count += 1

        # 1. Warmup: skip H accumulation
        if self._frame_count <= self.warmup_frames:
            return

        # 2. Update history variable (irreversible)
        psi_clamped = psi_plus.clamp(min=0.0, max=1e8)
        self.H = torch.maximum(self.H, psi_clamped)

        # 3. Build/update graph
        self.graph.build(positions)

        # 4. Damage evolution on the Gaussian graph
        self._evolve_damage(positions, stress_dir)

        # 5. Crack normal estimation
        self._estimate_crack_normal(positions, stress_dir)

        # 6. Opening magnitude
        self._compute_opening(F_gaussian)

    def _evolve_damage(
        self,
        positions: Tensor,
        stress_dir: Optional[Tensor] = None,
    ) -> None:
        """
        Graph-based damage evolution using localized excess drive.

        AT2 equilibrium on graph:
            c_eq = (H_ratio + l0² · lap_c) / (1 + H_ratio)
        where:
            H_ratio = 2 · l0 · H / Gc
            lap_c = graph Laplacian of c

        With optional anisotropic weighting: propagation is faster
        perpendicular to the principal tensile stress direction.
        """
        H_ratio = 2.0 * self.l0 * self.H / self.Gc  # (N,)
        if H_ratio.numel() == 0:
            return

        q = float(min(max(self.drive_quantile, 0.0), 0.999))
        drive_floor = torch.quantile(H_ratio.detach(), q)
        drive_span = (H_ratio.max() - drive_floor).clamp(min=1e-8)
        drive = ((H_ratio - drive_floor) / drive_span).clamp(0.0, 1.0)

        front = torch.maximum(self.c, self.graph.weighted_neighbor_max(self.c))
        front_active = front > self.front_threshold

        if stress_dir is not None and self.aniso_ratio > 1.0:
            prop = self._directional_positive_propagation(
                self.c, positions, stress_dir)
            tip_support = self._directional_neighbor_peak(
                self.c, positions, stress_dir)
        else:
            prop = self._positive_propagation(self.c)
            tip_support = self.graph.weighted_neighbor_max(self.c)

        dc_source = self.damage_source_scale * drive * front_active.float()
        # Propagation should work even without local drive —
        # the crack front carries itself forward once initiated.
        dc_prop = self.damage_spread * prop * front_active.float()
        dc_tip = (
            self.tip_propagation_scale
            * (tip_support - self.c).clamp(min=0.0)
            * front_active.float()
        )
        dc = (dc_source + dc_prop + dc_tip).clamp(0.0, self.dC_max)
        self.c = (self.c + dc).clamp(0.0, 1.0)

        # Diagnostics
        n_growing = (dc > 1e-6).sum().item()
        if self._frame_count % 10 == 0 or self._frame_count < 5:
            print(f"[GFF] frame={self._frame_count} "
                  f"c_max={self.c.max():.4f} c_mean={self.c.mean():.4f} "
                  f"H_max={self.H.max():.2e} growing={n_growing}/{self.c.shape[0]} "
                  f"drive=[{drive.min():.3f},{drive.max():.3f}] "
                  f"front={front_active.sum().item()}/{self.c.shape[0]}")

    def _positive_propagation(
        self,
        values: Tensor,
    ) -> Tensor:
        """One-sided propagation from more-damaged neighbors only."""
        nbr_vals = values[self.graph.knn_idx]  # (N, K)
        diff_pos = (nbr_vals - values.unsqueeze(1)).clamp(min=0.0)
        return (self.graph.weights * diff_pos).sum(dim=1)

    def _directional_positive_propagation(
        self,
        values: Tensor,
        positions: Tensor,
        stress_dir: Tensor,
    ) -> Tensor:
        """
        One-sided propagation with directional bias.

        Prefer edges aligned with the stress cue and moving outward from the
        original seed / impact center when available.
        """
        nbr_pos = positions[self.graph.knn_idx]   # (N, K, 3)
        edge_dir = nbr_pos - positions.unsqueeze(1)  # (N, K, 3)
        edge_len = edge_dir.norm(dim=2).clamp(min=1e-8)
        edge_dir_norm = edge_dir / edge_len.unsqueeze(2)

        n1 = stress_dir.unsqueeze(1)
        cos_theta = (edge_dir_norm * n1).sum(dim=2).abs()
        align_factor = 1.0 + (self.aniso_ratio - 1.0) * (cos_theta ** 2)

        radial_factor = 1.0
        if self.seed_center is not None and self.radial_bias > 0.0:
            radial_dir = positions - self.seed_center.unsqueeze(0)
            radial_dir = radial_dir / radial_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
            outward = (edge_dir_norm * radial_dir.unsqueeze(1)).sum(dim=2).clamp(min=0.0)
            radial_factor = 1.0 + self.radial_bias * outward

        weights_dir = self.graph.weights * align_factor * radial_factor
        weights_dir = weights_dir / (weights_dir.sum(dim=1, keepdim=True) + 1e-12)

        nbr_vals = values[self.graph.knn_idx]
        diff_pos = (nbr_vals - values.unsqueeze(1)).clamp(min=0.0)
        return (weights_dir * diff_pos).sum(dim=1)

    def _directional_neighbor_peak(
        self,
        values: Tensor,
        positions: Tensor,
        stress_dir: Tensor,
    ) -> Tensor:
        """
        Strong crack-tip support along favored edges.

        This behaves more like a front advancer than a diffusive smoother:
        a node can inherit a strongly damaged upstream neighbor if that
        neighbor lies along the preferred propagation direction.
        """
        nbr_pos = positions[self.graph.knn_idx]
        edge_dir = nbr_pos - positions.unsqueeze(1)
        edge_dir = edge_dir / edge_dir.norm(dim=2, keepdim=True).clamp(min=1e-8)

        n1 = stress_dir.unsqueeze(1)
        cos_theta = (edge_dir * n1).sum(dim=2).abs()
        bias = 1.0 + (self.aniso_ratio - 1.0) * (cos_theta ** 2)

        if self.seed_center is not None and self.radial_bias > 0.0:
            radial_dir = positions - self.seed_center.unsqueeze(0)
            radial_dir = radial_dir / radial_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
            outward = (edge_dir * radial_dir.unsqueeze(1)).sum(dim=2).clamp(min=0.0)
            bias = bias * (1.0 + self.radial_bias * outward)

        bias = bias / bias.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        nbr_vals = values[self.graph.knn_idx]
        return (nbr_vals * bias).max(dim=1).values

    def _anisotropic_positive_propagation(
        self,
        values: Tensor,
        positions: Tensor,
        stress_dir: Tensor,
    ) -> Tensor:
        """
        Anisotropic graph Laplacian: propagation is enhanced perpendicular
        to the principal tensile stress direction (crack grows ⊥ to tension).

        Weight modulation:
            w'_ij = w_ij * (1 + (aniso_ratio - 1) * sin²(θ_ij))
        where θ_ij is the angle between edge (i→j) and stress_dir[i].
        """
        # Edge directions: neighbor - self
        nbr_pos = positions[self.graph.knn_idx]   # (N, K, 3)
        edge_dir = nbr_pos - positions.unsqueeze(1)  # (N, K, 3)
        edge_len = edge_dir.norm(dim=2).clamp(min=1e-8)
        edge_dir_norm = edge_dir / edge_len.unsqueeze(2)

        # Angle between edge and stress direction
        n1 = stress_dir.unsqueeze(1)  # (N, 1, 3)
        cos_theta = (edge_dir_norm * n1).sum(dim=2).abs()  # (N, K)
        sin2_theta = 1.0 - cos_theta ** 2  # (N, K)

        # Modulate weights: higher weight ⊥ to tension
        aniso_factor = 1.0 + (self.aniso_ratio - 1.0) * sin2_theta
        weights_aniso = self.graph.weights * aniso_factor
        weights_aniso = weights_aniso / (weights_aniso.sum(dim=1, keepdim=True) + 1e-12)

        # One-sided weighted propagation
        nbr_vals = values[self.graph.knn_idx]  # (N, K)
        diff_pos = (nbr_vals - values.unsqueeze(1)).clamp(min=0.0)
        return (weights_aniso * diff_pos).sum(dim=1)

    def _estimate_crack_normal(
        self,
        positions: Tensor,
        stress_dir: Optional[Tensor] = None,
    ) -> None:
        """
        Estimate crack normal direction for each Gaussian.

        Strategy:
            1. Primary: gradient of damage field → crack grows along ∇c
               The crack normal is ∇c / |∇c| (perpendicular to crack surface)
            2. Fallback: principal stress direction n1 (from physics)
            3. Default: zero vector (no crack)

        The crack normal is used for:
            - Gaussian covariance deformation (flatten along n)
            - Split direction (offset along ±n)
            - Fragment boundary detection
        """
        # Only update normals for damaged Gaussians
        damaged = self.c > 0.05
        if not damaged.any():
            return

        # Gradient of damage field
        grad_c = self.graph.graph_gradient(self.c, positions)  # (N, 3)
        grad_mag = grad_c.norm(dim=1)  # (N,)

        # Use gradient where strong enough
        has_grad = damaged & (grad_mag > 1e-4)
        if has_grad.any():
            self.n[has_grad] = grad_c[has_grad] / grad_mag[has_grad].unsqueeze(1)

        # Fallback to stress direction where gradient is weak
        if stress_dir is not None:
            needs_fallback = damaged & ~has_grad
            if needs_fallback.any():
                self.n[needs_fallback] = stress_dir[needs_fallback]

    def _compute_opening(self, F_gaussian: Optional[Tensor] = None) -> None:
        """
        Compute crack opening magnitude.

        Opening is proportional to damage and deformation stretch
        along the crack normal direction.

        a_i = opening_scale * c_i^2 * stretch_along_n
        """
        if F_gaussian is not None and self.n is not None:
            # Stretch along crack normal: |F @ n|
            Fn = torch.bmm(F_gaussian, self.n.unsqueeze(2)).squeeze(2)  # (N, 3)
            stretch = Fn.norm(dim=1)  # (N,)
            # Opening = damage^2 * (stretch - 1) clamped positive
            tensile_stretch = (stretch - 1.0).clamp(min=0.0)
            self.a = self.opening_scale * (self.c ** 2) * (1.0 + tensile_stretch * 5.0)
        else:
            # Without F, opening is purely damage-based
            self.a = self.opening_scale * (self.c ** 2)

    def seed_damage(
        self,
        positions: Tensor,
        center: Tensor,
        radius: float,
        magnitude: float = 0.8,
        H_multiplier: float = 0.0,
    ) -> None:
        """
        Seed damage at a localized region (e.g., impact zone).

        Args:
            positions: (N, 3) Gaussian positions
            center: (3,) seed center
            radius: seed radius
            magnitude: peak damage value
            H_multiplier: H seed as multiple of H_ref = Gc/(2*l0)
        """
        if not self._initialized:
            self.initialize(positions.shape[0])

        radius = max(float(radius), 1e-6)
        dist = (positions - center.unsqueeze(0)).norm(dim=1)
        t = (1.0 - dist / radius).clamp(0.0, 1.0)
        influence = t * t

        # Damage seed
        c_seed = influence * magnitude
        self.c = torch.maximum(self.c, c_seed)
        self.seed_center = center.detach().clone()

        # Optional H seed. Keep this local to avoid turning broad impact
        # regions into permanently pre-damaged background.
        if H_multiplier > 0.0:
            H_ref = self.Gc / (2.0 * self.l0)
            H_seed = influence * H_ref * H_multiplier
            self.H = torch.maximum(self.H, H_seed)

        n_seeded = (c_seed > 1e-6).sum().item()
        print(f"[GFF] Seeded damage: {n_seeded} Gaussians, "
              f"c_max={self.c.max():.3f} H_max={self.H.max():.1f}")

    def get_state(self) -> dict:
        """Return current fracture state as a dict for checkpointing."""
        return {
            'c': self.c,
            'H': self.H,
            'n': self.n,
            'a': self.a,
            'f': self.f,
            'frame_count': self._frame_count,
        }

    def load_state(self, state: dict) -> None:
        """Restore fracture state from checkpoint."""
        self.c = state['c']
        self.H = state['H']
        self.n = state['n']
        self.a = state['a']
        self.f = state.get('f', torch.zeros_like(self.c, dtype=torch.long))
        self._frame_count = state.get('frame_count', 0)
        self._initialized = True
