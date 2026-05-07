"""Multi-mesh falling + mesh-to-mesh collision (scaffold).

NOTE — this is a SCAFFOLD, not a fully working pipeline.  Mesh-to-mesh
collision via the shared MPM grid is "free" once particles from
multiple bodies share the same `x_mpm` tensor — the existing P2G2P
step automatically averages velocities at each grid cell, so adjacent
bodies exchange momentum on contact.

What's left to wire (so each body fractures independently):
    1. Per-mesh impact detection: detect when EACH body's lowest
       particle reaches `ground_z` (currently the simulator handles
       only a single body-wide impact event).
    2. Per-mesh Voronoi tessellation: instantiate a separate
       `VoronoiDecomposer` for each mesh, masking particles by
       `mesh_id`.  Bond breakage / Mode-I kicks then propagate only
       within their owning body.
    3. Per-mesh material/family routing: each spec entry's prompt
       runs through CLIP-MaterialDB independently, so different
       bodies can have different materials in the same scene.

Items 1–3 require a focused refactor in `manifold_simulator.py` and
`voronoi_pipeline.py` (~1–2 days).  Without those, this script will
still drop multiple bodies onto the floor and let them collide via
the shared grid, but they will all use a single material/style and
share one impact event.

Spec format (JSON list):
    [{"mesh": "bunny", "drop_z": 0.45, "drop_xy": [0.40, 0.50],
      "scale": 0.20, "prompt": "soda-lime glass shattering"},
     {"mesh": "bunny", "drop_z": 0.75, "drop_xy": [0.50, 0.50],
      "scale": 0.20, "prompt": "soda-lime glass shattering"},
     {"mesh": "bunny", "drop_z": 1.05, "drop_xy": [0.60, 0.50],
      "scale": 0.20, "prompt": "soda-lime glass shattering"}]

Usage:
    python scripts/run_multi_mesh.py \\
        --spec '[...]' --particles-per-mesh 10000 --frames 120 \\
        --grid 64 --out output/multi_mesh_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _sample_surface_particles(mesh_path: Path, n: int, seed: int = 0) -> np.ndarray:
    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float32)


def _normalize_and_place(
    pts: np.ndarray,
    scale: float,
    cx: float,
    cy: float,
    cz: float,
) -> np.ndarray:
    bbox_min, bbox_max = pts.min(axis=0), pts.max(axis=0)
    bbox_size = (bbox_max - bbox_min).clip(min=1e-6)
    diag = float(np.linalg.norm(bbox_size))
    pts = (pts - 0.5 * (bbox_min + bbox_max)) / max(diag, 1e-6) * scale
    pts[:, 0] += cx
    pts[:, 1] += cy
    pts[:, 2] += cz
    return pts.astype(np.float32)


def build_combined_cloud(spec: list[dict], particles_per_mesh: int, default_scale: float):
    """Build combined particle cloud from multi-mesh spec.

    Returns (x_init, mesh_ids, per_mesh_meta).
    """
    parts: list[np.ndarray] = []
    mesh_ids: list[np.ndarray] = []
    metas: list[dict] = []
    for i, entry in enumerate(spec):
        mesh_path = ROOT / "assets" / "meshes" / f"{entry['mesh']}.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"mesh not found: {mesh_path}")
        pts_local = _sample_surface_particles(
            mesh_path, particles_per_mesh, seed=1234 + i)
        cx, cy = entry.get("drop_xy", [0.5, 0.5])
        scale = float(entry.get("scale", default_scale))
        pts_world = _normalize_and_place(pts_local, scale, float(cx), float(cy),
                                         float(entry["drop_z"]))
        parts.append(pts_world)
        mesh_ids.append(np.full(pts_world.shape[0], i, dtype=np.int64))
        metas.append({
            "mesh_id": i,
            "mesh_name": entry["mesh"],
            "drop_z": float(entry["drop_z"]),
            "drop_xy": [float(cx), float(cy)],
            "scale": scale,
            "prompt": entry.get("prompt", "soda-lime glass shattering"),
            "n_particles": pts_world.shape[0],
        })
    return (
        np.concatenate(parts, axis=0),
        np.concatenate(mesh_ids, axis=0),
        metas,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=str, required=True,
                        help="JSON list of {mesh, drop_z, drop_xy, scale, prompt} entries.")
    parser.add_argument("--particles-per-mesh", type=int, default=10000)
    parser.add_argument("--scale", type=float, default=0.20)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec)
    x_init, mesh_ids, metas = build_combined_cloud(
        spec, args.particles_per_mesh, args.scale)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist combined cloud + per-mesh metadata so a follow-up
    # implementation can wire the per-mesh fracture path.
    np.savez(
        out_dir / "multi_mesh_init.npz",
        x_init=x_init,
        mesh_ids=mesh_ids,
    )
    with open(out_dir / "multi_mesh_meta.json", "w") as f:
        json.dump({"meshes": metas, "n_total": int(x_init.shape[0])}, f, indent=2)

    print(f"[multi-mesh scaffold] wrote {x_init.shape[0]} particles "
          f"across {len(spec)} meshes -> {out_dir}")
    for m in metas:
        print(f"  mesh {m['mesh_id']}: {m['mesh_name']} @ "
              f"z={m['drop_z']:.2f} xy={m['drop_xy']}  "
              f"n={m['n_particles']}  prompt='{m['prompt']}'")
    print()
    print("[multi-mesh scaffold] next steps to actually run the sim:")
    print("  1. Modify ManifoldSimulator.initialize() to accept a")
    print("     `mesh_ids` tensor and store it as `self.mesh_ids`.")
    print("  2. In _handle_ground_impact(), iterate per mesh_id and")
    print("     fire impact only for the body whose lowest particle")
    print("     just reached ground_z (use a `_impacted_mesh_ids` set).")
    print("  3. In _init_voronoi_tessellation(), tessellate per mesh:")
    print("     for each unfracture mesh_id, run VoronoiDecomposer on")
    print("     the subset of x_mpm with that mesh_id.")
    print("  4. Per-particle prompt routing: build a per-particle")
    print("     fracture_cfg that picks values from the per-mesh CLIP")
    print("     routing tables.")


if __name__ == "__main__":
    main()
