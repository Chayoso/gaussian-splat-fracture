"""Fragment detection from AT2 phase-field damage via connected components."""

import torch
from torch import Tensor
from typing import List, Optional
import numpy as np
from scipy.ndimage import label as scipy_label


class FragmentManager:
    """
    Detect and track material fragments from the AT2 damage field.

    Pipeline:
    1. Threshold c_grid → binary crack boundary
    2. AND with grid_occupied → material mask
    3. scipy.ndimage.label (26-connectivity) → connected components
    4. Map grid labels → particles via B-spline weighted vote
    """

    def __init__(
        self,
        damage_threshold: float = 0.5,
        min_fragment_particles: int = 50,
        device: str = "cuda",
    ):
        self.damage_threshold = damage_threshold
        self.min_fragment_particles = min_fragment_particles
        self.device = torch.device(device)

        self.n_fragments: int = 0
        self.fragment_ids: Optional[Tensor] = None
        self.fragment_particle_indices: List[Tensor] = []
        self.surface_fragment_ids: Optional[Tensor] = None

    def detect_fragments(
        self,
        c_grid: Tensor,
        grid_occupied: Tensor,
        x_mpm: Tensor,
        mpm_model,
    ) -> int:
        """Run connected components and map labels to particles."""
        label_grid_np, n_labels = self._label_grid(c_grid, grid_occupied)
        if n_labels <= 1:
            self.n_fragments = n_labels
            self.fragment_ids = torch.zeros(
                x_mpm.shape[0], dtype=torch.long, device=self.device)
            self._build_particle_indices(n_labels)
            return n_labels

        label_grid = torch.from_numpy(label_grid_np).to(
            dtype=torch.long, device=self.device)

        self.fragment_ids = self._map_grid_labels_to_particles(
            label_grid, x_mpm, mpm_model)

        # Remap labels to 0..K-1 (scipy labels are 1..K)
        unique_labels = self.fragment_ids.unique()
        unique_labels = unique_labels[unique_labels > 0]
        remap = torch.zeros(
            unique_labels.max().item() + 1, dtype=torch.long, device=self.device)
        for new_id, old_id in enumerate(unique_labels):
            remap[old_id] = new_id
        valid = self.fragment_ids > 0
        self.fragment_ids[valid] = remap[self.fragment_ids[valid]]
        # label=0 particles (in crack) → assign to nearest fragment
        self._assign_orphans(x_mpm)

        self.n_fragments = int(unique_labels.numel())
        self._build_particle_indices(self.n_fragments)
        return self.n_fragments

    def _label_grid(self, c_grid: Tensor, grid_occupied: Tensor):
        """Binary threshold + connected components via scipy."""
        c_np = c_grid.detach().cpu().numpy()
        occ_np = grid_occupied.detach().cpu().bool().numpy()
        material_mask = occ_np & (c_np < self.damage_threshold)
        struct = np.ones((3, 3, 3), dtype=np.int32)  # 26-connectivity
        label_array, n_features = scipy_label(material_mask, structure=struct)
        return label_array, n_features

    def _map_grid_labels_to_particles(
        self,
        label_grid: Tensor,
        x_mpm: Tensor,
        mpm_model,
    ) -> Tensor:
        """Weighted vote: gather labels from 27 B-spline nodes, pick majority."""
        n = mpm_model.num_grids
        inv_dx = mpm_model.inv_dx
        dx = mpm_model.dx
        offset = mpm_model.offset  # (27, 3)
        label_flat = label_grid.reshape(-1)  # (n³,)

        N = x_mpm.shape[0]
        frag_ids = torch.zeros(N, dtype=torch.long, device=self.device)
        CHUNK = 50000

        for ci in range(0, N, CHUNK):
            cj = min(ci + CHUNK, N)
            x_c = x_mpm[ci:cj]
            M = cj - ci

            px = x_c * inv_dx
            base = (px - 0.5).long()
            fx = px - base.float()

            w = [0.5 * (1.5 - fx) ** 2,
                 0.75 - (fx - 1) ** 2,
                 0.5 * (fx - 0.5) ** 2]
            w = torch.stack(w, dim=-1)
            w_e = torch.einsum('bi,bj,bk->bijk', w[:, 0], w[:, 1], w[:, 2])
            weight = w_e.reshape(M, 27)  # (M, 27)

            idx3d = base.unsqueeze(1) + offset.unsqueeze(0).long()
            idx_flat = (idx3d[:, :, 0] * n * n
                        + idx3d[:, :, 1] * n
                        + idx3d[:, :, 2])
            idx_flat = idx_flat.clamp(0, n ** 3 - 1)  # (M, 27)

            node_labels = label_flat[idx_flat]  # (M, 27)

            # Weighted vote: for each particle, find label with highest weight sum
            max_label = int(node_labels.max().item())
            if max_label == 0:
                continue

            best_w = torch.zeros(M, device=self.device)
            best_lab = torch.zeros(M, dtype=torch.long, device=self.device)

            for lab in range(1, max_label + 1):
                mask_lab = (node_labels == lab)
                w_lab = (weight * mask_lab.float()).sum(dim=1)
                update = w_lab > best_w
                if update.any():
                    best_w[update] = w_lab[update]
                    best_lab[update] = lab

            assigned = best_lab > 0
            if assigned.any():
                frag_ids[ci:cj][assigned] = best_lab[assigned]

        return frag_ids

    def _assign_orphans(self, x_mpm: Tensor):
        """Assign label=0 (crack interior) particles to nearest fragment."""
        orphan_mask = self.fragment_ids == 0
        n_orphans = orphan_mask.sum().item()
        if n_orphans == 0:
            return

        labeled_mask = self.fragment_ids > 0
        if not labeled_mask.any():
            return

        orphan_pos = x_mpm[orphan_mask]
        labeled_pos = x_mpm[labeled_mask]
        labeled_ids = self.fragment_ids[labeled_mask]

        CHUNK = 10000
        orphan_indices = torch.where(orphan_mask)[0]

        for ci in range(0, n_orphans, CHUNK):
            cj = min(ci + CHUNK, n_orphans)
            pos_c = orphan_pos[ci:cj]
            dists = torch.cdist(pos_c, labeled_pos)
            nearest = dists.argmin(dim=1)
            self.fragment_ids[orphan_indices[ci:cj]] = labeled_ids[nearest]

    def _build_particle_indices(self, n_fragments: int):
        """Build per-fragment particle index lists."""
        self.fragment_particle_indices = []
        if self.fragment_ids is None:
            return

        for k in range(n_fragments):
            indices = torch.where(self.fragment_ids == k)[0]
            self.fragment_particle_indices.append(indices)

        # Merge small fragments into nearest large fragment
        large_threshold = self.min_fragment_particles
        large_frags = [
            k for k in range(n_fragments)
            if len(self.fragment_particle_indices[k]) >= large_threshold
        ]
        if not large_frags:
            return

        small_frags = [
            k for k in range(n_fragments)
            if len(self.fragment_particle_indices[k]) < large_threshold
            and len(self.fragment_particle_indices[k]) > 0
        ]
        if not small_frags:
            return

        # Compute centroids of large fragments
        large_centroids = []
        for k in large_frags:
            idx = self.fragment_particle_indices[k]
            # We don't have x_mpm here, so skip centroid merge for now
            # Small fragments will use ballistic physics in step_physics
            large_centroids.append(k)

    def map_to_surface(self, surface_mask: Tensor) -> Tensor:
        """Map fragment IDs to surface particles."""
        if self.fragment_ids is None:
            return None
        self.surface_fragment_ids = self.fragment_ids[surface_mask]
        return self.surface_fragment_ids
