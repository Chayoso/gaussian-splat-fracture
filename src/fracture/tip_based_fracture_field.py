"""
Tip-based Gaussian fracture field.

Fracture growth is represented as an explicit crack front on the Gaussian
surface graph. The damage field is only updated near that front, which keeps
the crack narrow and makes upward propagation possible when the projected
physics cues support it.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional

from .crack_front import CrackFront
from .graph_builder import GaussianGraph


class GaussianFractureField:
    """Tip-driven fracture state on Gaussian nodes."""

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
        front_substeps: int = 2,
        graph: Optional[GaussianGraph] = None,
        crack_front: Optional[CrackFront] = None,
        device: str = "cuda",
    ):
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
        self.front_substeps = max(int(front_substeps), 1)
        self.device = torch.device(device)

        self.graph = graph or GaussianGraph(device=device)
        self.crack_front = crack_front or CrackFront(device=device)

        self.c: Optional[Tensor] = None
        self.H: Optional[Tensor] = None
        self.n: Optional[Tensor] = None
        self.a: Optional[Tensor] = None
        self.f: Optional[Tensor] = None
        self.seed_center: Optional[Tensor] = None

        self._frame_count: int = 0
        self._initialized: bool = False

    def initialize(self, N: int) -> None:
        device = self.device
        self.c = torch.zeros(N, device=device)
        self.H = torch.zeros(N, device=device)
        self.n = torch.zeros(N, 3, device=device)
        self.a = torch.zeros(N, device=device)
        self.f = torch.zeros(N, dtype=torch.long, device=device)
        self.seed_center = None
        self._frame_count = 0
        self._initialized = True
        self.crack_front.initialize(N, device)

    @staticmethod
    def _safe_normalize(v: Tensor) -> Tensor:
        if v.numel() == 0:
            return v
        if v.ndim == 1:
            scale = v.max().clamp(min=1e-8)
            return (v / scale).clamp(0.0, 1.0)
        norm = v.norm(dim=1, keepdim=True)
        return torch.where(norm > 1e-8, v / norm.clamp(min=1e-8), torch.zeros_like(v))

    def _normalize_drive(self, values: Tensor, quantile: Optional[float] = None) -> Tensor:
        if values.numel() == 0:
            return values
        q = float(min(max(quantile if quantile is not None else self.drive_quantile, 0.5), 0.999))
        floor = torch.quantile(values.detach(), q)
        span = (values.max() - floor).clamp(min=1e-8)
        return ((values - floor) / span).clamp(0.0, 1.0)

    def update(
        self,
        positions: Tensor,
        init_score: Tensor,
        growth_drive: Tensor,
        growth_dir: Optional[Tensor] = None,
        F_gaussian: Optional[Tensor] = None,
        impact_center: Optional[Tensor] = None,
    ) -> None:
        N = positions.shape[0]
        if not self._initialized or self.c.shape[0] != N:
            self.initialize(N)

        self._frame_count += 1
        self.graph.build(positions)

        if impact_center is not None:
            self.seed_center = impact_center.detach().clone()

        init_score = self._normalize_drive(init_score.clamp(min=0.0), quantile=0.96)
        growth_drive = self._normalize_drive(growth_drive.clamp(min=0.0))
        if growth_dir is None:
            growth_dir = torch.zeros(N, 3, device=positions.device)
        growth_dir = self._safe_normalize(growth_dir)

        self.H = torch.maximum(self.H, growth_drive)
        if self._frame_count <= self.warmup_frames:
            return

        seeded = 0
        advanced = 0
        if not self.crack_front.has_active_tips():
            seeded = self._seed_front(positions, init_score, growth_drive, growth_dir)
        else:
            for _ in range(self.front_substeps):
                if not self.crack_front.has_active_tips():
                    break
                advanced += self.crack_front.advance(
                    self.graph,
                    positions,
                    growth_drive,
                    growth_dir,
                    impact_center=self.seed_center,
                )
            if not self.crack_front.has_active_tips():
                seeded = self._seed_front(positions, init_score, growth_drive, growth_dir)

        self._update_damage_band(init_score, growth_drive)
        self._estimate_crack_normal(positions, growth_dir)
        self._compute_opening(F_gaussian)

        if self._frame_count % 10 == 0 or self._frame_count < 5:
            tip_count = int(self.crack_front.tip_mask.sum().item())
            visited_count = int(self.crack_front.visited_mask.sum().item())
            cracked_count = int((self.c > 0.3).sum().item())
            print(
                f"[GFF] frame={self._frame_count} "
                f"c_max={self.c.max():.4f} c_mean={self.c.mean():.4f} "
                f"tips={tip_count} visited={visited_count} "
                f"seeded={seeded} advanced={advanced} cracked={cracked_count}"
            )

    def _seed_front(
        self,
        positions: Tensor,
        init_score: Tensor,
        growth_drive: Tensor,
        growth_dir: Tensor,
    ) -> int:
        front_support = torch.maximum(self.c, self.graph.weighted_neighbor_max(self.c))
        front_support = self._safe_normalize(front_support)
        candidate_score = (
            0.65 * init_score
            + 0.35 * front_support
            + 0.20 * growth_drive * (front_support > self.front_threshold).float()
        ).clamp(0.0, 1.0)
        candidate_score = candidate_score * (front_support > 0.5 * self.front_threshold).float()
        return self.crack_front.seed_from_scores(
            candidate_score,
            positions,
            growth_dir,
            impact_center=self.seed_center,
        )

    def _update_damage_band(self, init_score: Tensor, growth_drive: Tensor) -> None:
        tip_mask = self.crack_front.tip_mask
        visited_mask = self.crack_front.visited_mask
        if tip_mask is None or visited_mask is None or not bool(visited_mask.any()):
            return

        tip_f = tip_mask.float()
        visited_f = visited_mask.float()
        nbr_tip = tip_mask[self.graph.knn_idx].float()
        tip_ring = (self.graph.weights * nbr_tip).sum(dim=1).clamp(0.0, 1.0)

        parent_mask = torch.zeros_like(tip_f)
        active_tip_idx = torch.where(tip_mask)[0]
        if active_tip_idx.numel() > 0:
            parent_idx = self.crack_front.parent_index[active_tip_idx]
            valid_parent = parent_idx >= 0
            if valid_parent.any():
                parent_mask[parent_idx[valid_parent]] = 1.0

        drive_local = growth_drive.clamp(0.0, 1.0)
        init_local = init_score.clamp(0.0, 1.0)

        tip_floor = (0.22 + 0.18 * drive_local) * tip_f
        core_floor = (0.18 + 0.10 * drive_local) * parent_mask
        self.c = torch.maximum(self.c, tip_floor)
        self.c = torch.maximum(self.c, core_floor)

        dc_tip = self.damage_source_scale * (0.45 + 0.55 * drive_local) * tip_f
        dc_core = 0.65 * self.tip_propagation_scale * (0.25 + 0.75 * drive_local) * parent_mask
        dc_band = self.damage_spread * (0.15 + 0.85 * drive_local) * tip_ring * (1.0 - tip_f)
        dc_seed = 0.20 * self.damage_source_scale * init_local * tip_ring * (1.0 - visited_f)

        dc = (dc_tip + dc_core + dc_band + dc_seed).clamp(0.0, self.dC_max)
        self.c = (self.c + dc).clamp(0.0, 1.0)
        self.H = torch.maximum(self.H, drive_local * torch.maximum(tip_f, tip_ring))

    def _estimate_crack_normal(
        self,
        positions: Tensor,
        growth_dir_hint: Optional[Tensor] = None,
    ) -> None:
        damaged = self.c > 0.05
        if not damaged.any():
            return

        grad_c = self.graph.graph_gradient(self.c, positions)
        grad_mag = grad_c.norm(dim=1)
        has_grad = damaged & (grad_mag > 1e-4)
        if has_grad.any():
            self.n[has_grad] = grad_c[has_grad] / grad_mag[has_grad].unsqueeze(1)

        remaining_idx = torch.where(damaged & ~has_grad)[0]
        if remaining_idx.numel() == 0:
            return

        normals = getattr(self.graph, "_normals", None)
        if growth_dir_hint is not None and normals is not None and normals.shape[0] == positions.shape[0]:
            opening_hint = torch.cross(
                normals[remaining_idx],
                growth_dir_hint[remaining_idx],
                dim=1,
            )
            opening_mag = opening_hint.norm(dim=1, keepdim=True)
            valid = opening_mag.squeeze(1) > 1e-8
            if valid.any():
                self.n[remaining_idx[valid]] = opening_hint[valid] / opening_mag[valid].clamp(min=1e-8)
                remaining_idx = remaining_idx[~valid]

        if remaining_idx.numel() == 0 or growth_dir_hint is None:
            return

        hint_norm = growth_dir_hint[remaining_idx].norm(dim=1, keepdim=True)
        valid = hint_norm.squeeze(1) > 1e-8
        if valid.any():
            self.n[remaining_idx[valid]] = (
                growth_dir_hint[remaining_idx[valid]]
                / hint_norm[valid].clamp(min=1e-8)
            )

    def _compute_opening(self, F_gaussian: Optional[Tensor] = None) -> None:
        if F_gaussian is not None and self.n is not None:
            Fn = torch.bmm(F_gaussian, self.n.unsqueeze(2)).squeeze(2)
            stretch = Fn.norm(dim=1)
            tensile_stretch = (stretch - 1.0).clamp(min=0.0)
            self.a = self.opening_scale * (self.c ** 2) * (1.0 + 4.0 * tensile_stretch)
        else:
            self.a = self.opening_scale * (self.c ** 2)

        if self.crack_front.tip_mask is not None:
            self.a = self.a * (1.0 + 0.25 * self.crack_front.tip_mask.float())

    def seed_damage(
        self,
        positions: Tensor,
        center: Tensor,
        radius: float,
        magnitude: float = 0.8,
        H_multiplier: float = 0.0,
    ) -> None:
        if not self._initialized:
            self.initialize(positions.shape[0])

        radius = max(float(radius), 1e-6)
        dist = (positions - center.unsqueeze(0)).norm(dim=1)
        t = (1.0 - dist / radius).clamp(0.0, 1.0)
        influence = t * t

        c_seed = influence * magnitude
        self.c = torch.maximum(self.c, c_seed)
        self.seed_center = center.detach().clone()

        if H_multiplier > 0.0:
            H_ref = self.Gc / (2.0 * self.l0)
            H_seed = influence * H_ref * H_multiplier
            self.H = torch.maximum(self.H, H_seed)

        n_seeded = int((c_seed > 1e-6).sum().item())
        print(
            f"[GFF] Seeded damage: {n_seeded} Gaussians, "
            f"c_max={self.c.max():.3f} H_max={self.H.max():.1f}"
        )

    def get_state(self) -> dict:
        return {
            "c": self.c,
            "H": self.H,
            "n": self.n,
            "a": self.a,
            "f": self.f,
            "seed_center": self.seed_center,
            "frame_count": self._frame_count,
            "crack_front": self.crack_front.get_state(),
        }

    def load_state(self, state: dict) -> None:
        self.c = state["c"]
        self.H = state["H"]
        self.n = state["n"]
        self.a = state["a"]
        self.f = state.get("f", torch.zeros_like(self.c, dtype=torch.long))
        self.seed_center = state.get("seed_center", None)
        self._frame_count = state.get("frame_count", 0)
        self._initialized = True
        if "crack_front" in state:
            self.crack_front.load_state(state["crack_front"])
        else:
            self.crack_front.initialize(self.c.shape[0], self.c.device)
