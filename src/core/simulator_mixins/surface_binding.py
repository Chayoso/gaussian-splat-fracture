"""SurfaceBindingMixin for ManifoldSimulator.

This module is behavior-preserving extraction from manifold_simulator.py.
"""

import torch
from torch import Tensor
from src.utils.knn import knn_search


class SurfaceBindingMixin:
    def _build_rest_surface_binding(self, chunk_size: int = 2048) -> None:
        """Bind each particle to a rest-state nearest surface Gaussian."""
        if self.init_positions is None:
            return
        if self._surface_indices is None:
            self._surface_indices = torch.where(self.surface_mask)[0]
        if self._interior_indices is None:
            self._interior_indices = torch.where(~self.surface_mask)[0]

        surface_indices = self._surface_indices
        interior_indices = self._interior_indices
        device = self.init_positions.device
        n_total = self.init_positions.shape[0]
        n_surf = int(surface_indices.shape[0])
        if n_surf <= 0:
            self._particle_to_surface_local = None
            self._surface_local_index = None
            return

        particle_to_surface = torch.full(
            (n_total,), -1, dtype=torch.long, device=device
        )
        surface_local_index = torch.full(
            (n_total,), -1, dtype=torch.long, device=device
        )
        surface_local = torch.arange(n_surf, device=device, dtype=torch.long)
        particle_to_surface[surface_indices] = surface_local
        surface_local_index[surface_indices] = surface_local

        if interior_indices.numel() > 0:
            x_surf = self.init_positions[surface_indices]
            _, nearest = knn_search(
                self.init_positions[interior_indices],
                x_surf,
                1,
            )
            nearest = nearest[:, 0]
            particle_to_surface[interior_indices] = nearest

        self._particle_to_surface_local = particle_to_surface
        self._surface_local_index = surface_local_index
        print(
            f"[ManifoldSim] Rest surface binding built: "
            f"{n_total} particles -> {n_surf} surface anchors"
        )

    def _project_surface_scalar_to_particles(
        self,
        surface_values: Tensor,
        interior_scale: float = 1.0,
    ) -> Tensor:
        """Project a per-surface scalar field to all particles via rest binding."""
        out = torch.zeros(
            self.x_mpm.shape[0],
            dtype=surface_values.dtype,
            device=surface_values.device,
        )
        if self._surface_indices is None or self._particle_to_surface_local is None:
            return out

        n_assign = min(surface_values.shape[0], self._surface_indices.shape[0])
        if n_assign <= 0:
            return out

        surface_indices = self._surface_indices[:n_assign]
        out[surface_indices] = surface_values[:n_assign]
        if self._interior_indices is not None and self._interior_indices.numel() > 0:
            mapped = self._particle_to_surface_local[self._interior_indices]
            valid = (mapped >= 0) & (mapped < n_assign)
            if bool(valid.any()):
                out[self._interior_indices[valid]] = (
                    surface_values[mapped[valid]] * interior_scale
                )
        return out

    def _map_surface_labels_to_particles(
        self,
        surface_labels: Tensor,
    ) -> Tensor:
        """Map per-surface fragment labels to all particles via rest binding."""
        out = torch.zeros(
            self.x_mpm.shape[0],
            dtype=torch.long,
            device=surface_labels.device,
        )
        if self._surface_indices is None or self._particle_to_surface_local is None:
            return out

        n_assign = min(surface_labels.shape[0], self._surface_indices.shape[0])
        if n_assign <= 0:
            return out

        surface_indices = self._surface_indices[:n_assign]
        out[surface_indices] = surface_labels[:n_assign]
        if self._interior_indices is not None and self._interior_indices.numel() > 0:
            mapped = self._particle_to_surface_local[self._interior_indices]
            valid = (mapped >= 0) & (mapped < n_assign)
            if bool(valid.any()):
                out[self._interior_indices[valid]] = surface_labels[mapped[valid]]
        return out
