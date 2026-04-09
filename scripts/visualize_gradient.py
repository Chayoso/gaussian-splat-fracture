"""Visualize c_grid and ∇c as 2D heatmap slices."""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "gaussian-splatting"))


def visualize_from_checkpoint(ckpt_path: str, output_dir: str = "output/grad_viz"):
    """Load checkpoint and visualize c_grid + gradient."""
    ckpt = torch.load(ckpt_path, map_location='cpu')

    c_grid = ckpt.get('c_grid')
    if c_grid is None:
        print("No c_grid in checkpoint!")
        return

    n = c_grid.shape[0]
    dx = 1.0 / n
    print(f"Grid: {n}³, c_grid range: [{c_grid.min():.4f}, {c_grid.max():.4f}]")
    print(f"Cracked cells (c>0.3): {(c_grid > 0.3).sum().item()}")

    # Compute gradient
    cp = torch.nn.functional.pad(
        c_grid.unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1, 1, 1), mode='replicate'
    )[0, 0]

    grad_x = (cp[2:, 1:-1, 1:-1] - cp[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = (cp[1:-1, 2:, 1:-1] - cp[1:-1, :-2, 1:-1]) / (2 * dx)
    grad_z = (cp[1:-1, 1:-1, 2:] - cp[1:-1, 1:-1, :-2]) / (2 * dx)

    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    print(f"Gradient magnitude range: [{grad_mag.min():.4f}, {grad_mag.max():.4f}]")

    # Output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Find interesting Z slices (where damage exists)
    c_per_z = c_grid.sum(dim=(0, 1))
    top_z = torch.topk(c_per_z, 5).indices.sort().values

    fig, axes = plt.subplots(2, len(top_z), figsize=(4 * len(top_z), 8))

    for i, z_idx in enumerate(top_z):
        z = z_idx.item()

        # Row 1: c_grid slice
        ax1 = axes[0, i]
        im1 = ax1.imshow(c_grid[:, :, z].numpy().T, origin='lower',
                          cmap='hot', vmin=0, vmax=1)
        ax1.set_title(f'c_grid (z={z}/{n})')
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # Row 2: gradient magnitude slice
        ax2 = axes[1, i]
        gm = grad_mag[:, :, z].numpy().T
        im2 = ax2.imshow(gm, origin='lower',
                          cmap='viridis', vmin=0, vmax=gm.max() * 0.8)
        ax2.set_title(f'|∇c| (z={z}/{n})')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # Overlay gradient direction arrows (subsampled)
        step = max(1, n // 16)
        xx, yy = np.meshgrid(range(0, n, step), range(0, n, step))
        gx_sub = grad_x[::step, ::step, z].numpy().T
        gy_sub = grad_y[::step, ::step, z].numpy().T
        mag_sub = np.sqrt(gx_sub**2 + gy_sub**2) + 1e-8
        # Only show arrows where gradient is significant
        mask = mag_sub > mag_sub.max() * 0.1
        ax2.quiver(xx[mask], yy[mask],
                   gx_sub[mask] / mag_sub[mask],
                   gy_sub[mask] / mag_sub[mask],
                   color='white', alpha=0.7, scale=30)

    plt.suptitle(f'Checkpoint: {Path(ckpt_path).name}', fontsize=14)
    plt.tight_layout()

    save_path = out / f'gradient_heatmap_{Path(ckpt_path).stem}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Also save a single mid-slice for quick view
    mid_z = n // 2
    fig2, (ax_c, ax_g) = plt.subplots(1, 2, figsize=(12, 5))

    im_c = ax_c.imshow(c_grid[:, :, mid_z].numpy().T, origin='lower',
                        cmap='hot', vmin=0, vmax=1)
    ax_c.set_title(f'c_grid (z={mid_z}, mid-slice)')
    plt.colorbar(im_c, ax=ax_c)

    gm_mid = grad_mag[:, :, mid_z].numpy().T
    im_g = ax_g.imshow(gm_mid, origin='lower', cmap='viridis')
    ax_g.set_title(f'|∇c| (z={mid_z}, mid-slice)')
    plt.colorbar(im_g, ax=ax_g)

    # Arrows
    step = max(1, n // 20)
    xx, yy = np.meshgrid(range(0, n, step), range(0, n, step))
    gx_sub = grad_x[::step, ::step, mid_z].numpy().T
    gy_sub = grad_y[::step, ::step, mid_z].numpy().T
    mag_sub = np.sqrt(gx_sub**2 + gy_sub**2) + 1e-8
    mask = mag_sub > mag_sub.max() * 0.05
    if mask.any():
        ax_g.quiver(xx[mask], yy[mask],
                    gx_sub[mask] / mag_sub[mask],
                    gy_sub[mask] / mag_sub[mask],
                    color='white', alpha=0.8, scale=25)

    plt.tight_layout()
    save_path2 = out / f'gradient_midslice_{Path(ckpt_path).stem}.png'
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path2}")


if __name__ == "__main__":
    import glob
    ckpts = sorted(glob.glob("output/checkpoints/checkpoint_*.pt"))
    if not ckpts:
        print("No checkpoints found!")
    else:
        for cp in ckpts:
            print(f"\n{'='*50}")
            visualize_from_checkpoint(cp)
