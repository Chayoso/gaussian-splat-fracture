"""Physical-fragment diagnostic plot.

Renders a 3-view scatter using actual MPM particle positions and the
filtered ``_physical_fragment_labels`` (the registry that keeps only
coherent, persistent chunks above ``fragment_physical_min_size``).
This is the visualization the smoke test produces as `physical_crack_*`
and `physical_fragment_*` PNGs -- it avoids the "floating Gaussian"
artefact you get when you plot graph-level fragment IDs at the
visualizer's offset positions, because every node here is either
(a) base body (label 0) or (b) a member of a physical fragment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor


def _to_numpy(t: Optional[Tensor]) -> Optional[np.ndarray]:
    if t is None:
        return None
    return t.detach().cpu().numpy()


def save_physical_diagnostic(
    *,
    positions: Tensor,                     # (N, 3) world-space positions
    damage: Optional[Tensor],              # (N,) c field
    physical_fragment_ids: Optional[Tensor],  # (N,) physical labels (0 = base)
    visited: Optional[Tensor],
    tips: Optional[Tensor],
    out_path: Path,
    title: str,
) -> None:
    """Save a 3-view scatter of damage + physical fragments.

    Floating-Gaussian guard: nodes with ``physical_fragment_ids == 0`` are
    rendered as the base body (low alpha grey), nodes with id > 0 are
    rendered as colored physical fragments.  Graph-level small patches
    that didn't make the physical registry are simply painted as base,
    so they cannot appear as orphaned floating splats.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = _to_numpy(positions)
    if x is None or x.shape[0] == 0:
        return
    c = _to_numpy(damage)
    if c is None:
        c = np.zeros(x.shape[0])
    v = _to_numpy(visited)
    if v is None:
        v = np.zeros(x.shape[0], dtype=bool)
    t_arr = _to_numpy(tips)
    if t_arr is None:
        t_arr = np.zeros(x.shape[0], dtype=bool)

    frag = _to_numpy(physical_fragment_ids)
    if frag is None:
        frag = np.zeros(x.shape[0], dtype=np.int64)

    # Truncate to common length so floating dimension mismatches don't
    # produce orphan points.
    n = min(x.shape[0], c.shape[0], v.shape[0], t_arr.shape[0], frag.shape[0])
    x, c, v, t_arr, frag = x[:n], c[:n], v[:n], t_arr[:n], frag[:n]

    detached_ids = [int(fid) for fid in np.unique(frag) if int(fid) > 0]
    n_phys_fragments = len(detached_ids)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    views = [
        ("X-Y (top)", 0, 1),
        ("X-Z (front)", 0, 2),
        ("Y-Z (side)", 1, 2),
    ]

    cmap_frag = plt.get_cmap("tab20")
    last_sc = None

    for ax, (label, i, j) in zip(axes, views):
        # Base body: low alpha grey, hides any node not in a physical
        # fragment so they don't look like orphan floating points.
        base_mask = frag == 0
        if base_mask.any():
            ax.scatter(
                x[base_mask, i], x[base_mask, j],
                c=c[base_mask], cmap="hot",
                vmin=0.0, vmax=1.0,
                s=0.7, alpha=0.18, linewidths=0,
            )
            last_sc = ax.collections[-1]

        # Physical fragments: full opacity colored chunks.
        for fid in detached_ids:
            mask = frag == fid
            if not mask.any():
                continue
            color = cmap_frag((fid - 1) % 20)
            ax.scatter(
                x[mask, i], x[mask, j],
                color=[color], s=8.0, alpha=0.95, linewidths=0,
            )
            com = x[mask][:, [i, j]].mean(axis=0)
            ax.scatter(
                [com[0]], [com[1]],
                c="white", edgecolors="black", s=28, linewidths=0.7,
            )

        # Visited (crack-front coverage) and active tips kept for
        # diagnostic cross-reference.
        if v.any():
            ax.scatter(
                x[v, i], x[v, j],
                c="#34d399", s=2.0, alpha=0.45, linewidths=0,
            )
        if t_arr.any():
            ax.scatter(
                x[t_arr, i], x[t_arr, j],
                c="#38bdf8", s=14.0, alpha=1.0,
                marker="x", linewidths=0.7,
            )

        ax.set_xlabel("XYZ"[i])
        ax.set_ylabel("XYZ"[j])
        ax.set_title(label)
        ax.set_aspect("equal")
        margin = 0.04
        ax.set_xlim(x[:, i].min() - margin, x[:, i].max() + margin)
        ax.set_ylim(x[:, j].min() - margin, x[:, j].max() + margin)

    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes, label="Damage c", shrink=0.6)
    fig.suptitle(
        f"{title}  |  physical_fragments={n_phys_fragments}  "
        f"c_max={c.max():.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
