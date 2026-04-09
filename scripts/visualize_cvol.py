"""Visualize c_vol (particle damage) as 2D scatter heatmap slices."""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_cvol(ckpt_path: str, output_dir: str = "output/cvol_viz"):
    """Load checkpoint and visualize c_vol as scatter heatmap."""
    ckpt = torch.load(ckpt_path, map_location='cpu')

    x_mpm = ckpt.get('x_mpm')
    c_vol = ckpt.get('c_vol')

    if x_mpm is None or c_vol is None:
        print(f"Missing x_mpm or c_vol in {ckpt_path}")
        return

    x = x_mpm.numpy()
    c = c_vol.numpy()
    frame = ckpt.get('frame', '?')

    print(f"Checkpoint: {Path(ckpt_path).name}")
    print(f"  Frame: {frame}")
    print(f"  Particles: {len(c)}")
    print(f"  c_vol range: [{c.min():.4f}, {c.max():.4f}]")
    print(f"  Cracked (c>0.3): {(c > 0.3).sum()} ({100*(c > 0.3).mean():.1f}%)")
    print(f"  Cracked (c>0.5): {(c > 0.5).sum()} ({100*(c > 0.5).mean():.1f}%)")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 3 orthogonal slices through the object center ---
    center = x.mean(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Slice definitions: (axis_name, axis_idx, other_axes)
    slices = [
        ('XY (top view, z-slice)', 2, (0, 1)),
        ('XZ (front view, y-slice)', 1, (0, 2)),
        ('YZ (side view, x-slice)', 0, (1, 2)),
    ]

    for col, (title, slice_axis, plot_axes) in enumerate(slices):
        # Select particles near the center along slice axis
        slice_thickness = 0.03
        mask = np.abs(x[:, slice_axis] - center[slice_axis]) < slice_thickness

        x_plot = x[mask][:, plot_axes[0]]
        y_plot = x[mask][:, plot_axes[1]]
        c_plot = c[mask]

        # Top row: all particles in slice
        ax1 = axes[0, col]
        scatter1 = ax1.scatter(x_plot, y_plot, c=c_plot, cmap='hot',
                               s=1, vmin=0, vmax=1, alpha=0.8)
        ax1.set_title(f'{title}\n(all particles, n={mask.sum()})')
        ax1.set_aspect('equal')
        plt.colorbar(scatter1, ax=ax1, label='c_vol')

        # Bottom row: only cracked particles
        cracked = c_plot > 0.1
        ax2 = axes[1, col]
        if cracked.any():
            scatter2 = ax2.scatter(x_plot[cracked], y_plot[cracked],
                                   c=c_plot[cracked], cmap='hot',
                                   s=3, vmin=0, vmax=1)
            ax2.set_title(f'{title}\n(cracked only, n={cracked.sum()})')
            plt.colorbar(scatter2, ax=ax2, label='c_vol')
        else:
            ax2.text(0.5, 0.5, 'No cracked particles',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'{title}\n(cracked only)')
        ax2.set_aspect('equal')
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())

    plt.suptitle(f'c_vol Heatmap — Frame {frame} — {Path(ckpt_path).name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = out / f'cvol_heatmap_{Path(ckpt_path).stem}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # --- Surface vs Volume comparison ---
    # First 50% = surface, rest = volume (based on surface_ratio=0.5)
    n_surf = len(c) // 2
    c_surf = c[:n_surf]
    c_vol_only = c[n_surf:]

    fig2, (ax_s, ax_v) = plt.subplots(1, 2, figsize=(12, 4))

    ax_s.hist(c_surf[c_surf > 0.01], bins=50, color='blue', alpha=0.7)
    ax_s.set_title(f'Surface particles c_vol distribution\n(n={len(c_surf)}, cracked={( c_surf > 0.3).sum()})')
    ax_s.set_xlabel('c_vol')
    ax_s.axvline(0.3, color='r', linestyle='--', label='threshold 0.3')
    ax_s.legend()

    ax_v.hist(c_vol_only[c_vol_only > 0.01], bins=50, color='orange', alpha=0.7)
    ax_v.set_title(f'Volume particles c_vol distribution\n(n={len(c_vol_only)}, cracked={(c_vol_only > 0.3).sum()})')
    ax_v.set_xlabel('c_vol')
    ax_v.axvline(0.3, color='r', linestyle='--', label='threshold 0.3')
    ax_v.legend()

    plt.suptitle(f'Surface vs Volume Damage — Frame {frame}', fontsize=13)
    plt.tight_layout()

    save_path2 = out / f'cvol_surf_vs_vol_{Path(ckpt_path).stem}.png'
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path2}")


def visualize_crack_paths(ckpt_path: str, output_dir: str = "output/cvol_viz"):
    """Overlay crack_paths polylines on c_vol heatmap."""
    ckpt = torch.load(ckpt_path, map_location='cpu')

    x_mpm = ckpt.get('x_mpm')
    c_vol = ckpt.get('c_vol')
    crack_paths = ckpt.get('crack_paths', [])
    frame = ckpt.get('frame', '?')

    if x_mpm is None or c_vol is None:
        print(f"Missing data in {ckpt_path}")
        return

    x = x_mpm.numpy()
    c = c_vol.numpy()

    print(f"  Crack paths: {len(crack_paths)}")
    total_pts = sum(p.shape[0] for p in crack_paths) if crack_paths else 0
    print(f"  Total path points: {total_pts}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 3 views with crack paths overlaid
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    views = [
        ('XY (top)', 0, 1, 2),      # plot x,y; slice on z
        ('XZ (front)', 0, 2, 1),     # plot x,z; slice on y
        ('YZ (side)', 1, 2, 0),      # plot y,z; slice on x
    ]

    center = x.mean(axis=0)

    for col, (title, ax0, ax1, slice_ax) in enumerate(views):
        ax = axes[col]

        # Slice particles near center
        slice_thick = 0.05
        mask = np.abs(x[:, slice_ax] - center[slice_ax]) < slice_thick

        x_plot = x[mask][:, ax0]
        y_plot = x[mask][:, ax1]
        c_plot = c[mask]

        # Background: all particles (dim)
        ax.scatter(x_plot, y_plot, c=c_plot, cmap='hot', s=0.5,
                   vmin=0, vmax=1, alpha=0.5)

        # Cracked particles (bright)
        cracked = c_plot > 0.3
        if cracked.any():
            ax.scatter(x_plot[cracked], y_plot[cracked], c=c_plot[cracked],
                       cmap='hot', s=2, vmin=0, vmax=1)

        # Overlay crack paths as lines
        for i, path in enumerate(crack_paths):
            p = path.numpy()
            # Filter path points near this slice
            slice_mask = np.abs(p[:, slice_ax] - center[slice_ax]) < slice_thick * 2
            if slice_mask.sum() >= 2:
                p_vis = p[slice_mask]
                ax.plot(p_vis[:, ax0], p_vis[:, ax1], 'c-', linewidth=1.5, alpha=0.8)
                ax.plot(p_vis[0, ax0], p_vis[0, ax1], 'go', markersize=3)  # start
                ax.plot(p_vis[-1, ax0], p_vis[-1, ax1], 'r^', markersize=3)  # tip

        ax.set_title(f'{title}\n(cyan=polyline, green=start, red=tip)')
        ax.set_aspect('equal')

    plt.suptitle(f'c_vol + Crack Paths — Frame {frame} — {Path(ckpt_path).name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = out / f'crack_paths_overlay_{Path(ckpt_path).stem}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    import glob
    ckpts = sorted(glob.glob("output/checkpoints/checkpoint_*.pt"))
    if not ckpts:
        print("No checkpoints found! Run simulation first.")
    else:
        for cp in ckpts:
            print(f"\n{'='*50}")
            visualize_cvol(cp)
            visualize_crack_paths(cp)
