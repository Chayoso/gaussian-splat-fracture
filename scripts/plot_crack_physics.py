"""
Plot all crack-related physics quantities from simulation statistics CSV.

Usage:
    python plot_crack_physics.py [--csv output/statistics.csv] [--out output/plots]
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_stats(csv_path: str) -> dict:
    """Load statistics CSV into dict of numpy arrays."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    data = {}
    for key in rows[0].keys():
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except (ValueError, KeyError):
                vals.append(0.0)
        data[key] = np.array(vals)

    return data


def plot_all(data: dict, out_dir: str):
    """Generate comprehensive crack physics plots."""
    os.makedirs(out_dir, exist_ok=True)
    frames = data.get('frame', np.arange(len(next(iter(data.values())))))

    # Color palette
    C1, C2, C3, C4, C5 = '#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0'

    # =====================================================================
    # Figure 1: Damage Evolution (2x2)
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Damage Evolution', fontsize=16, fontweight='bold')

    # 1a: c_max and c_mean
    ax = axes[0, 0]
    ax.plot(frames, data['c_max'], color=C2, linewidth=2, label='$c_{max}$')
    ax.plot(frames, data['c_mean'], color=C1, linewidth=2, label='$c_{mean}$')
    if 'c_surface_max' in data:
        ax.plot(frames, data['c_surface_max'], color=C2, linewidth=1, linestyle='--',
                alpha=0.6, label='$c_{surf,max}$')
        ax.plot(frames, data['c_surface_mean'], color=C1, linewidth=1, linestyle='--',
                alpha=0.6, label='$c_{surf,mean}$')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Damage $c$')
    ax.set_title('Phase Field Damage')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)

    # 1b: Cracked particle counts at multiple thresholds
    ax = axes[0, 1]
    if 'n_cracked_03' in data:
        ax.plot(frames, data['n_cracked_03'], color=C4, linewidth=2, label='$c > 0.3$')
    if 'n_cracked_05' in data:
        ax.plot(frames, data['n_cracked_05'], color=C2, linewidth=2, label='$c > 0.5$')
    if 'n_cracked_08' in data:
        ax.plot(frames, data['n_cracked_08'], color=C5, linewidth=2, label='$c > 0.8$')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Particle Count')
    ax.set_title('Cracked Particles by Threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 1c: Damage rate (dc/dframe)
    ax = axes[1, 0]
    if len(frames) > 1:
        dc_max = np.diff(data['c_max'])
        dc_mean = np.diff(data['c_mean'])
        ax.plot(frames[1:], dc_max, color=C2, linewidth=1.5, alpha=0.7, label='$\\Delta c_{max}$')
        ax.plot(frames[1:], dc_mean, color=C1, linewidth=1.5, alpha=0.7, label='$\\Delta c_{mean}$')
    ax.set_xlabel('Frame')
    ax.set_ylabel('$\\Delta c$ / frame')
    ax.set_title('Damage Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 1d: Cracked fraction
    ax = axes[1, 1]
    n_particles = data.get('n_particles', np.ones(len(frames)))
    n_particles[n_particles == 0] = 1
    if 'n_cracked_03' in data:
        frac_03 = data['n_cracked_03'] / n_particles * 100
        frac_05 = data.get('n_cracked_05', np.zeros_like(frames)) / n_particles * 100
        frac_08 = data.get('n_cracked_08', np.zeros_like(frames)) / n_particles * 100
        ax.fill_between(frames, 0, frac_03, alpha=0.3, color=C4, label='$c > 0.3$')
        ax.fill_between(frames, 0, frac_05, alpha=0.3, color=C2, label='$c > 0.5$')
        ax.fill_between(frames, 0, frac_08, alpha=0.3, color=C5, label='$c > 0.8$')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Cracked Fraction (%)')
    ax.set_title('Volume Fraction of Damaged Material')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_damage_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [1/5] Damage evolution saved")

    # =====================================================================
    # Figure 2: Strain Energy & Griffith Criterion (2x2)
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Strain Energy & Griffith Criterion', fontsize=16, fontweight='bold')

    # 2a: H_max, H_mean, H_ref
    ax = axes[0, 0]
    if 'H_max' in data:
        ax.plot(frames, data['H_max'], color=C2, linewidth=2, label='$H_{max}$')
        ax.plot(frames, data['H_mean'], color=C1, linewidth=2, label='$H_{mean}$ (nonzero)')
        if 'H_ref' in data:
            ax.axhline(y=data['H_ref'][0], color='k', linestyle='--', linewidth=1.5,
                       alpha=0.7, label=f'$H_{{ref}} = G_c / 2l_0$ = {data["H_ref"][0]:.0f}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Strain Energy History $H$')
    ax.set_title('H Field Statistics')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2b: G/Gc ratio
    ax = axes[0, 1]
    if 'G_Gc_max' in data:
        ax.plot(frames, data['G_Gc_max'], color=C2, linewidth=2, label='$G_{max}/G_c$')
        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Griffith threshold ($G/G_c = 1$)')
        ax.fill_between(frames, 1.0, data['G_Gc_max'],
                        where=data['G_Gc_max'] > 1.0,
                        alpha=0.2, color=C2, label='Supercritical zone')
    ax.set_xlabel('Frame')
    ax.set_ylabel('$G / G_c$')
    ax.set_title('Energy Release Rate Ratio')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2c: Number of supercritical cells
    ax = axes[1, 0]
    if 'n_supercritical' in data:
        ax.plot(frames, data['n_supercritical'], color=C5, linewidth=2)
        ax.fill_between(frames, 0, data['n_supercritical'], alpha=0.2, color=C5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Grid Cells with $H > H_{ref}$')
    ax.set_title('Supercritical Volume')
    ax.grid(True, alpha=0.3)

    # 2d: Stress max
    ax = axes[1, 1]
    if 'stress_max' in data:
        ax.plot(frames, data['stress_max'], color=C3, linewidth=2, label='$\\|\\sigma\\|_{max}$')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Stress Norm (Pa)')
    ax.set_title('Maximum Stress')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_energy_griffith.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [2/5] Energy & Griffith saved")

    # =====================================================================
    # Figure 3: Crack Path Geometry (2x2)
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Crack Path Geometry', fontsize=16, fontweight='bold')

    # 3a: Number of crack paths
    ax = axes[0, 0]
    if 'n_paths' in data:
        ax.plot(frames, data['n_paths'], color=C1, linewidth=2, label='Active paths')
        if 'total_branches' in data:
            ax.plot(frames, data['total_branches'], color=C5, linewidth=2,
                    linestyle='--', label='Cumulative branches')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Count')
    ax.set_title('Crack Paths & Branches')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3b: Total crack length
    ax = axes[0, 1]
    if 'total_crack_length' in data:
        ax.plot(frames, data['total_crack_length'], color=C2, linewidth=2,
                label='Total crack length')
        ax.plot(frames, data['mean_path_length'], color=C4, linewidth=2,
                linestyle='--', label='Mean path length')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Length (normalized units)')
    ax.set_title('Crack Length')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3c: Max segment count (longest path)
    ax = axes[1, 0]
    if 'max_segment_count' in data:
        ax.plot(frames, data['max_segment_count'], color=C3, linewidth=2)
        ax.fill_between(frames, 0, data['max_segment_count'], alpha=0.15, color=C3)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Segments')
    ax.set_title('Longest Path Segment Count')
    ax.grid(True, alpha=0.3)

    # 3d: Total path points
    ax = axes[1, 1]
    if 'total_path_pts' in data:
        ax.plot(frames, data['total_path_pts'], color=C4, linewidth=2)
        # Crack growth rate
        if len(frames) > 1:
            growth = np.diff(data['total_path_pts'])
            ax2 = ax.twinx()
            ax2.bar(frames[1:], growth, alpha=0.3, color=C2, width=0.8, label='Growth/frame')
            ax2.set_ylabel('New Points/Frame', color=C2)
            ax2.tick_params(axis='y', labelcolor=C2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Total Points')
    ax.set_title('Path Points & Growth Rate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_crack_geometry.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [3/5] Crack geometry saved")

    # =====================================================================
    # Figure 4: Kinematics (2x2)
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Kinematics & Deformation', fontsize=16, fontweight='bold')

    # 4a: Velocity
    ax = axes[0, 0]
    if 'v_max' in data:
        ax.plot(frames, data['v_max'], color=C2, linewidth=2, label='$|v|_{max}$')
        ax.plot(frames, data['v_mean'], color=C1, linewidth=2, label='$|v|_{mean}$')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Particle Velocities')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4b: Displacement
    ax = axes[0, 1]
    if 'disp_max' in data:
        ax.plot(frames, data['disp_max'], color=C3, linewidth=2, label='$|u|_{max}$')
        ax.plot(frames, data['disp_mean'], color=C1, linewidth=2, label='$|u|_{mean}$')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Displacement (normalized)')
    ax.set_title('Particle Displacement')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4c: Seismic response — velocity oscillation
    ax = axes[1, 0]
    if 'v_max' in data and 'time' in data:
        ax.plot(data['time'], data['v_max'], color=C2, linewidth=1.5)
        ax.set_xlabel('Simulation Time (s)')
        ax.set_ylabel('$|v|_{max}$ (m/s)')
        ax.set_title('Seismic Response (Velocity vs Time)')
        ax.grid(True, alpha=0.3)

    # 4d: Phase diagram — v_max vs c_max
    ax = axes[1, 1]
    if 'v_max' in data and 'c_max' in data:
        sc = ax.scatter(data['v_max'], data['c_max'], c=frames, cmap='viridis',
                        s=10, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='Frame')
        ax.set_xlabel('$|v|_{max}$')
        ax.set_ylabel('$c_{max}$')
        ax.set_title('Phase Diagram: Velocity vs Damage')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_kinematics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [4/5] Kinematics saved")

    # =====================================================================
    # Figure 5: Summary Dashboard (3x2)
    # =====================================================================
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Crack Simulation Summary Dashboard', fontsize=18, fontweight='bold')
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 5a: Damage timeline
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(frames, data['c_max'], color=C2, linewidth=2, label='$c_{max}$')
    n_cracked = data.get('n_cracked_03', data.get('n_cracked', np.zeros_like(frames)))
    ax2 = ax.twinx()
    ax2.plot(frames, n_cracked, color=C4, linewidth=2, linestyle='--')
    ax2.set_ylabel('Cracked Count', color=C4)
    ax.set_ylabel('$c_{max}$', color=C2)
    ax.set_xlabel('Frame')
    ax.set_title('Damage Timeline')
    ax.grid(True, alpha=0.3)

    # 5b: Energy ratio
    ax = fig.add_subplot(gs[0, 1])
    if 'G_Gc_max' in data:
        ax.plot(frames, data['G_Gc_max'], color=C2, linewidth=2)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.fill_between(frames, 1.0, data['G_Gc_max'],
                        where=data['G_Gc_max'] > 1.0, alpha=0.15, color=C2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('$G/G_c$')
    ax.set_title('Griffith Energy Ratio')
    ax.grid(True, alpha=0.3)

    # 5c: Crack paths
    ax = fig.add_subplot(gs[1, 0])
    if 'n_paths' in data:
        ax.plot(frames, data['n_paths'], color=C1, linewidth=2, label='Paths')
        ax.plot(frames, data.get('total_branches', np.zeros_like(frames)),
                color=C5, linewidth=2, linestyle='--', label='Branches')
        ax.legend(fontsize=9)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Count')
    ax.set_title('Crack Paths')
    ax.grid(True, alpha=0.3)

    # 5d: Total crack length
    ax = fig.add_subplot(gs[1, 1])
    if 'total_crack_length' in data:
        ax.plot(frames, data['total_crack_length'], color=C3, linewidth=2)
        ax.fill_between(frames, 0, data['total_crack_length'], alpha=0.15, color=C3)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Length')
    ax.set_title('Total Crack Length')
    ax.grid(True, alpha=0.3)

    # 5e: Velocity
    ax = fig.add_subplot(gs[2, 0])
    if 'v_max' in data:
        ax.plot(frames, data['v_max'], color=C2, linewidth=1.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('$|v|_{max}$')
    ax.set_title('Peak Velocity')
    ax.grid(True, alpha=0.3)

    # 5f: Key metrics text
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    final = {k: v[-1] for k, v in data.items()}
    peak_G = data['G_Gc_max'].max() if 'G_Gc_max' in data else 0
    text = (
        f"{'='*40}\n"
        f"  Final Frame Statistics\n"
        f"{'='*40}\n"
        f"  Total Frames:     {int(final.get('frame', 0))}\n"
        f"  Sim Time:         {final.get('time', 0):.4f} s\n"
        f"{'─'*40}\n"
        f"  Damage (c_max):   {final.get('c_max', 0):.4f}\n"
        f"  Cracked (>0.3):   {int(final.get('n_cracked_03', final.get('n_cracked', 0)))}\n"
        f"  Cracked (>0.5):   {int(final.get('n_cracked_05', 0))}\n"
        f"  Cracked (>0.8):   {int(final.get('n_cracked_08', 0))}\n"
        f"{'─'*40}\n"
        f"  Crack Paths:      {int(final.get('n_paths', 0))}\n"
        f"  Total Length:     {final.get('total_crack_length', 0):.4f}\n"
        f"  Branches:         {int(final.get('total_branches', 0))}\n"
        f"{'─'*40}\n"
        f"  Peak G/Gc:        {peak_G:.2f}\n"
        f"  H_max:            {final.get('H_max', 0):.1f}\n"
        f"  H_ref:            {final.get('H_ref', 0):.1f}\n"
        f"{'─'*40}\n"
        f"  Peak |v|:         {data['v_max'].max() if 'v_max' in data else 0:.4f}\n"
        f"  Max Displacement: {data['disp_max'].max() if 'disp_max' in data else 0:.6f}\n"
        f"  Max Stress:       {data['stress_max'].max() if 'stress_max' in data else 0:.1f}\n"
        f"{'='*40}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.savefig(os.path.join(out_dir, '05_summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [5/6] Summary dashboard saved")

    # =====================================================================
    # Figure 6: Momentum & Energy Conservation (2x2)
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Momentum & Energy Conservation', fontsize=16, fontweight='bold')

    # 6a: Linear momentum components
    ax = axes[0, 0]
    if 'momentum_x' in data:
        ax.plot(frames, data['momentum_x'], color=C2, linewidth=1.5, label='$p_x$')
        ax.plot(frames, data['momentum_y'], color=C3, linewidth=1.5, label='$p_y$')
        ax.plot(frames, data['momentum_z'], color=C1, linewidth=1.5, label='$p_z$')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Momentum (kg m/s)')
    ax.set_title('Linear Momentum Components')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6b: Momentum magnitude
    ax = axes[0, 1]
    if 'momentum_mag' in data:
        ax.plot(frames, data['momentum_mag'], color=C5, linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('$|p|$ (kg m/s)')
    ax.set_title('Total Momentum Magnitude')
    ax.grid(True, alpha=0.3)

    # 6c: Energy partition
    ax = axes[1, 0]
    if 'kinetic_energy' in data:
        ax.plot(frames, data['kinetic_energy'], color=C2, linewidth=2, label='Kinetic Energy')
    if 'strain_energy_total' in data:
        ax.plot(frames, data['strain_energy_total'], color=C3, linewidth=2, label='Strain Energy (H)')
    if 'total_damage' in data:
        # Scale damage to be visible alongside energy
        d = data['total_damage']
        if d.max() > 0:
            Gc_val = data.get('H_ref', np.ones_like(d))[0] * 2.0 * 0.025 if 'H_ref' in data else 1.0
            fracture_energy = d * Gc_val  # approximate fracture dissipation
            ax.plot(frames, fracture_energy, color=C5, linewidth=2,
                    linestyle='--', label=f'Fracture Dissipation (~$G_c \\cdot \\Sigma c$)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Partition')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6d: Damage monotonicity + total damage
    ax = axes[1, 1]
    if 'total_damage' in data:
        ax.plot(frames, data['total_damage'], color=C4, linewidth=2, label='$\\Sigma c$ (total damage)')
        ax.set_ylabel('Total Damage', color=C4)
    if 'damage_monotonic' in data:
        violations = data['damage_monotonic'] < 0.5
        if violations.any():
            ax.axvspan(frames[violations].min(), frames[violations].max(),
                       alpha=0.3, color='red', label='Monotonicity violation!')
        else:
            ax.text(0.98, 0.02, 'Damage monotonicity OK',
                    transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                    color='green', fontweight='bold')
    ax.set_xlabel('Frame')
    ax.set_title('Total Damage & Monotonicity Check')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '06_conservation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [6/6] Conservation plots saved")

    print(f"\nAll plots saved to: {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot crack simulation physics')
    parser.add_argument('--csv', default='output/statistics.csv', help='Statistics CSV path')
    parser.add_argument('--out', default='output/plots', help='Output directory for plots')
    args = parser.parse_args()

    print(f"Loading statistics from: {args.csv}")
    data = load_stats(args.csv)
    print(f"  Loaded {len(data.get('frame', []))} frames, {len(data)} columns")

    print(f"\nGenerating plots...")
    plot_all(data, args.out)
