"""Visualize pure phase-field checkpoints as narrow-band heatmaps."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch


def _projection_bbox(mask_2d: np.ndarray, pad: int = 4):
    ys, xs = np.where(mask_2d)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = max(int(xs.min()) - pad, 0)
    x1 = int(xs.max()) + pad + 1
    y0 = max(int(ys.min()) - pad, 0)
    y1 = int(ys.max()) + pad + 1
    return x0, x1, y0, y1


def _crop_projection(img: np.ndarray, bbox):
    if bbox is None:
        return img
    x0, x1, y0, y1 = bbox
    return img[y0:y1, x0:x1]


def _compute_band_stats(c_grid: torch.Tensor, threshold: float):
    active = torch.nonzero(c_grid > threshold, as_tuple=False).float()
    if active.numel() == 0:
        return {
            "active_cells": 0,
            "bbox_cells": [0, 0, 0],
            "singular_values": [0.0, 0.0, 0.0],
            "anisotropy_ratio": 0.0,
        }
    centered = active - active.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(centered, full_matrices=False)
    s = s.cpu().numpy()
    mins = active.min(dim=0).values
    maxs = active.max(dim=0).values
    bbox = (maxs - mins + 1).cpu().numpy().astype(int).tolist()
    return {
        "active_cells": int(active.shape[0]),
        "bbox_cells": bbox,
        "singular_values": [float(v) for v in s[:3]],
        "anisotropy_ratio": float(s[0] / max(s[-1], 1e-6)),
    }


def visualize_checkpoint(ckpt_path: str, output_dir: str = "output/heatmaps",
                         active_thresh: float = 0.10, crack_thresh: float = 0.30):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    frame = ckpt.get("frame", "?")
    x_mpm = ckpt.get("x_mpm")
    c_vol = ckpt.get("c_vol")
    c_grid = ckpt.get("c_grid")
    H_grid = ckpt.get("H_grid")
    surface_mask = ckpt.get("surface_mask")
    support_grid = ckpt.get("pf_support_grid")

    if x_mpm is None or c_vol is None:
        raise ValueError(f"Checkpoint missing x_mpm/c_vol: {ckpt_path}")
    if c_grid is None:
        raise ValueError(f"Checkpoint missing c_grid: {ckpt_path}")

    x = x_mpm.numpy()
    c = c_vol.numpy()
    n = c_grid.shape[0]
    c_np = c_grid.numpy()
    H_np = H_grid.numpy() if H_grid is not None else np.zeros_like(c_np)
    S_np = support_grid.numpy().astype(np.float32) if support_grid is not None else None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    band_stats = _compute_band_stats(c_grid, active_thresh)
    crack_mask_3d = c_np > active_thresh
    proj_bboxes = {
        "xy": _projection_bbox(crack_mask_3d.max(axis=2)),
        "xz": _projection_bbox(crack_mask_3d.max(axis=1)),
        "yz": _projection_bbox(crack_mask_3d.max(axis=0)),
    }

    projections = [
        ("XY", c_np.max(axis=2), H_np.max(axis=2), S_np.max(axis=2) if S_np is not None else None, (0, 1), proj_bboxes["xy"]),
        ("XZ", c_np.max(axis=1), H_np.max(axis=1), S_np.max(axis=1) if S_np is not None else None, (0, 2), proj_bboxes["xz"]),
        ("YZ", c_np.max(axis=0), H_np.max(axis=0), S_np.max(axis=0) if S_np is not None else None, (1, 2), proj_bboxes["yz"]),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(16, 16))

    for col, (title, c_proj, h_proj, s_proj, plot_axes, bbox) in enumerate(projections):
        c_img = _crop_projection(c_proj.T, bbox)
        h_img = _crop_projection(h_proj.T, bbox)
        s_img = _crop_projection(s_proj.T, bbox) if s_proj is not None else None
        c_vmax = max(float(c_img.max()), active_thresh * 1.25, 1e-3)
        crack_mask = np.zeros_like(c_img, dtype=np.int32)
        crack_mask[c_img > active_thresh] = 1
        crack_mask[c_img > crack_thresh] = 2

        ax_h = axes[0, col]
        if s_img is not None:
            im_h = ax_h.imshow(s_img, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
            ax_h.set_title(f"{title} support max-proj")
        else:
            im_h = ax_h.imshow(h_img, origin="lower", cmap="magma")
            ax_h.set_title(f"{title} H max-proj")
        plt.colorbar(im_h, ax=ax_h, fraction=0.046)

        ax_c = axes[1, col]
        im_c = ax_c.imshow(
            c_img,
            origin="lower",
            cmap="hot",
            norm=colors.PowerNorm(gamma=0.65, vmin=0.0, vmax=c_vmax),
        )
        if c_img.max() > active_thresh:
            yy, xx = np.mgrid[0:c_img.shape[0], 0:c_img.shape[1]]
            ax_c.contour(xx, yy, c_img, levels=[active_thresh], colors="cyan", linewidths=1.0)
        if c_img.max() > crack_thresh:
            yy, xx = np.mgrid[0:c_img.shape[0], 0:c_img.shape[1]]
            ax_c.contour(xx, yy, c_img, levels=[crack_thresh], colors="lime", linewidths=1.0)
        ax_c.set_title(f"{title} c_grid max-proj (adaptive)")
        plt.colorbar(im_c, ax=ax_c, fraction=0.046)

        ax_m = axes[2, col]
        mask_cmap = colors.ListedColormap(["black", "#ffb000", "#ff2d55"])
        mask_norm = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], mask_cmap.N)
        ax_m.imshow(crack_mask, origin="lower", cmap=mask_cmap, norm=mask_norm)
        ax_m.set_title(f"{title} band mask ({active_thresh:.2f}/{crack_thresh:.2f})")

        ax_p = axes[3, col]
        pts = x[:, plot_axes]
        vals = c
        if surface_mask is not None:
            sm = surface_mask.bool().numpy()
            pts = pts[sm]
            vals = vals[sm]
            ax_p.set_title(f"{title} surface c_vol")
        else:
            ax_p.set_title(f"{title} particle c_vol")
        cracked = vals > 0.01
        if cracked.any():
            ax_p.scatter(
                pts[cracked, 0], pts[cracked, 1],
                c=vals[cracked], cmap="hot", vmin=0.0, vmax=max(float(vals[cracked].max()), crack_thresh), s=4, alpha=0.9
            )
        else:
            ax_p.text(0.5, 0.5, "No cracked particles", ha="center", va="center",
                      transform=ax_p.transAxes)
        ax_p.set_aspect("equal")

    sv = band_stats["singular_values"]
    fig.suptitle(
        f"Pure PF Heatmap | frame={frame} | "
        f"cells>{active_thresh:.2f}={band_stats['active_cells']} | "
        f"bbox={band_stats['bbox_cells']} | "
        f"sv=[{sv[0]:.1f}, {sv[1]:.1f}, {sv[2]:.1f}] | "
        f"anisotropy={band_stats['anisotropy_ratio']:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path = out / f"pure_pf_heatmap_{Path(ckpt_path).stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cracked_particles = int((c > crack_thresh).sum())
    surface_cracked = None
    if surface_mask is not None:
        sm = surface_mask.bool().numpy()
        surface_cracked = int((c[sm] > crack_thresh).sum())
    support_cells = int(support_grid.sum().item()) if support_grid is not None else None

    print(f"Checkpoint: {Path(ckpt_path).name}")
    print(f"  frame={frame}")
    print(f"  c_grid max={float(c_grid.max()):.4f}, H_grid max={float(H_grid.max()) if H_grid is not None else 0.0:.4e}")
    print(f"  active cells > {active_thresh:.2f}: {band_stats['active_cells']}")
    print(f"  bbox cells: {band_stats['bbox_cells']}")
    print(f"  singular values: {[round(v, 3) for v in band_stats['singular_values']]}")
    print(f"  anisotropy ratio: {band_stats['anisotropy_ratio']:.3f}")
    if support_cells is not None:
        print(f"  support cells: {support_cells}")
    print(f"  cracked particles > {crack_thresh:.2f}: {cracked_particles}")
    if surface_cracked is not None:
        print(f"  cracked surface particles > {crack_thresh:.2f}: {surface_cracked}")
    print(f"  saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize pure PF checkpoints")
    parser.add_argument("checkpoint", nargs="+", help="Checkpoint path(s)")
    parser.add_argument("--output-dir", default="output/heatmaps")
    parser.add_argument("--active-thresh", type=float, default=0.10)
    parser.add_argument("--crack-thresh", type=float, default=0.30)
    args = parser.parse_args()

    for ckpt in args.checkpoint:
        visualize_checkpoint(
            ckpt,
            output_dir=args.output_dir,
            active_thresh=args.active_thresh,
            crack_thresh=args.crack_thresh,
        )


if __name__ == "__main__":
    main()
