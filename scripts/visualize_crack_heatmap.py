"""Render crack-path density heatmaps from a saved simulator checkpoint."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _collect_path_points(crack_paths):
    pts = []
    for path in crack_paths:
        if path is None:
            continue
        p = path.detach().cpu().numpy() if torch.is_tensor(path) else np.asarray(path)
        if p.ndim == 2 and p.shape[1] == 3 and p.shape[0] > 0:
            pts.append(p)
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(pts, axis=0)


def _densify_paths(crack_paths, samples_per_segment: int = 24):
    pts = []
    for path in crack_paths:
        if path is None:
            continue
        p = path.detach().cpu().numpy() if torch.is_tensor(path) else np.asarray(path)
        if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] == 0:
            continue
        if p.shape[0] == 1:
            pts.append(p.astype(np.float32, copy=False))
            continue
        dense = [p[0:1]]
        for a, b in zip(p[:-1], p[1:]):
            t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False, dtype=np.float32)[:, None]
            seg = (1.0 - t) * a[None, :] + t * b[None, :]
            dense.append(seg.astype(np.float32, copy=False))
        dense.append(p[-1:])  # keep exact endpoints
        pts.append(np.concatenate(dense, axis=0))
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(pts, axis=0)


def _compute_view_range(pts: np.ndarray, axis_i: int, axis_j: int, margin: float = 0.04):
    mins = pts[:, [axis_i, axis_j]].min(axis=0)
    maxs = pts[:, [axis_i, axis_j]].max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    lo = np.maximum(0.0, mins - np.maximum(margin, 0.25 * span))
    hi = np.minimum(1.0, maxs + np.maximum(margin, 0.25 * span))
    return [[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]]


def visualize_crack_heatmap(ckpt_path: str, output_dir: str = "output/crack_heatmaps",
                            bins: int = 96) -> Path:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    frame = ckpt.get("frame", "?")
    crack_paths = ckpt.get("crack_paths", [])
    c_vol = ckpt.get("c_vol")
    x_mpm = ckpt.get("x_mpm")

    pts = _collect_path_points(crack_paths)
    if pts.shape[0] == 0:
        raise ValueError(f"No crack path points found in {ckpt_path}")
    dense_pts = _densify_paths(crack_paths)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    views = [
        ("XY", 0, 1),
        ("XZ", 0, 2),
        ("YZ", 1, 2),
    ]

    bg_x = x_mpm.numpy() if x_mpm is not None else None
    bg_c = c_vol.numpy() if c_vol is not None else None

    for ax, (title, i, j) in zip(axes, views):
        view_range = _compute_view_range(dense_pts, i, j)
        if bg_x is not None and bg_c is not None:
            cracked = bg_c > 0.15
            if cracked.any():
                x_lo, x_hi = view_range[0]
                y_lo, y_hi = view_range[1]
                cropped = (
                    (bg_x[:, i] >= x_lo) & (bg_x[:, i] <= x_hi) &
                    (bg_x[:, j] >= y_lo) & (bg_x[:, j] <= y_hi)
                )
                cracked = cracked & cropped
                ax.scatter(
                    bg_x[cracked, i],
                    bg_x[cracked, j],
                    c=bg_c[cracked],
                    cmap="gray",
                    s=2.5,
                    alpha=0.32,
                    vmin=0.0,
                    vmax=1.0,
                )

        h = ax.hist2d(
            dense_pts[:, i], dense_pts[:, j],
            bins=bins,
            range=view_range,
            cmap="inferno",
            cmin=1,
        )
        ax.scatter(pts[:, i], pts[:, j], c="cyan", s=8, alpha=0.9, linewidths=0.0)
        ax.set_title(f"{title} Crack Path Heatmap")
        ax.set_xlim(*view_range[0])
        ax.set_ylim(*view_range[1])
        ax.set_aspect("equal")
        ax.set_xlabel(["x", "x", "y"][views.index((title, i, j))])
        ax.set_ylabel(["y", "z", "z"][views.index((title, i, j))])
        plt.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04, label="Path density")

    plt.suptitle(
        f"Crack Path Density Heatmap - Frame {frame} - {Path(ckpt_path).name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path = out / f"crack_path_heatmap_{Path(ckpt_path).stem}.png"
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/visualize_crack_heatmap.py <checkpoint.pt> [output_dir]")

    ckpt_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/crack_heatmaps"
    save_path = visualize_crack_heatmap(ckpt_path, output_dir)
    print(save_path)
