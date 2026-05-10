"""RenderFragmentMixin for ManifoldSimulator.

Maintains a persistent fragment-id label per particle so the renderer
can colour fragments consistently across frames.  The historical
``offset/velocity/age`` per-fragment render-side displacement was
removed: rendered positions now equal raw MPM positions, since the
MPM release impulse + gravity already encode the fracture motion and
the offset only added a visible "invisible floor" artefact.
"""

from typing import Optional

import torch
from torch import Tensor


class RenderFragmentMixin:
    def _ensure_render_fragment_registry(self, count: int, device) -> None:
        if (self._render_fragment_labels is not None
                and self._render_fragment_labels.shape[0] == count
                and self._render_fragment_labels.device == device):
            return
        self._render_fragment_labels = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self._next_render_fragment_id = 1

    def _register_render_fragments(
        self,
        positions: Tensor,
        fragment_ids: Optional[Tensor],
        opening: Optional[Tensor] = None,
    ) -> None:
        if fragment_ids is None:
            return
        if self._render_fragment_labels is None:
            self._ensure_render_fragment_registry(positions.shape[0], positions.device)

        active_labels = fragment_ids.unique(sorted=True)
        for frag_id in active_labels.tolist():
            if frag_id <= 0:
                continue
            mask = fragment_ids == frag_id
            frag_size = int(mask.sum().item())
            if frag_size < self.fragment_render_min_size:
                continue

            overlap_vals, overlap_counts = self._render_fragment_labels[mask].unique(
                return_counts=True
            )
            persistent_label = None
            if overlap_vals.numel() > 0:
                overlap_mask = overlap_vals > 0
                if bool(overlap_mask.any()):
                    overlap_vals = overlap_vals[overlap_mask]
                    overlap_counts = overlap_counts[overlap_mask]
                    best_idx = int(torch.argmax(overlap_counts).item())
                    best_count = int(overlap_counts[best_idx].item())
                    if best_count / max(frag_size, 1) >= self.fragment_render_overlap_threshold:
                        persistent_label = int(overlap_vals[best_idx].item())

            if persistent_label is None:
                persistent_label = self._next_render_fragment_id
                self._next_render_fragment_id += 1

            self._render_fragment_labels[mask] = persistent_label

    def _apply_persistent_fragment_separation(
        self,
        positions: Tensor,
        fragment_ids: Optional[Tensor],
        opening: Optional[Tensor] = None,
    ) -> Tensor:
        self._ensure_render_fragment_registry(positions.shape[0], positions.device)
        if fragment_ids is not None and bool((fragment_ids > 0).any()):
            self._register_render_fragments(positions, fragment_ids, opening)
        render_ids = self._render_fragment_labels.clone()
        return positions, render_ids
