"""Side-by-side viewer for three Houdini geo directories in one window.

Loads three ``houdini_export`` directories and renders them in a single
Open3D window with horizontal X-offsets, so drop-height variants can
be compared frame-synchronously.  Supports the same colour modes as
the single-run viewer, including oriented Gaussian splat disks
(modes 5 / 6) — i.e. *pure Gaussian rendering*.

Usage:

    python scripts/view_houdini_geo_tri.py \
        output/run_10k_z0.22_v12/houdini_export \
        output/run_10k_z0.42_v12/houdini_export \
        output/run_10k_z0.80_v12/houdini_export

Keys:

    Space  toggle play / pause
    N / →  next frame
    P / ←  previous frame
    1      points coloured by base Cd
    2      points coloured by fragment_id
    3      points coloured by damage (heat)
    4      per-fragment thin-shell PCA-Delaunay surfaces
    5      oriented splat disks (fragment-coloured)        ← pure Gaussian
    6      oriented splat disks (base Cd, photoreal)       ← pure Gaussian
    R      reset camera
    Q      quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.view_houdini_geo import (  # noqa: E402
    read_houdini_geo,
    color_for_mode,
    build_fragment_hull_mesh,
    build_splat_disk_mesh,
    _list_frames,
)


def _shifted_array(arr: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    return arr + offset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs=3, type=Path,
                        help="Three houdini_export directories.")
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--x-offset", type=float, default=1.6,
                        help="Horizontal spacing between the three runs.")
    parser.add_argument("--mode", choices=("cd", "fragment", "damage",
                                            "surface", "splat_frag",
                                            "splat_cd"),
                        default="splat_cd",
                        help="Initial colour / render mode (default: "
                             "splat_cd, pure Gaussian disks with base Cd).")
    parser.add_argument("--max-splats", type=int, default=0,
                        help="Optional per-panel cap on rendered splats.")
    parser.add_argument("--align-impact", dest="align_impact",
                        action="store_true", default=True,
                        help="Align playback so common-time T=0 is each "
                             "run's first impact frame (default: on). "
                             "Pre-impact frames play with T<0.")
    parser.add_argument("--no-align-impact", dest="align_impact",
                        action="store_false",
                        help="Use absolute frame indices across runs "
                             "(legacy lockstep mode).")
    args = parser.parse_args()

    import open3d as o3d

    print("loading three runs...")
    runs: List[Tuple[Path, List[Path]]] = []
    impact_idx_per_panel: List[int] = []
    for d in args.dirs:
        frames = _list_frames(d)
        # First file whose phase tag is `impact_` marks the impact frame.
        impact_pos = next(
            (i for i, f in enumerate(frames) if "_impact_" in f.name),
            len(frames) - 1,  # never impacted: pin to last frame
        )
        impact_idx_per_panel.append(impact_pos)
        print(f"  {d.parent.name}/{d.name}: {len(frames)} frames "
              f"(impact at frame {impact_pos})")
        runs.append((d, frames))

    if args.align_impact:
        # Common time T: T=0 is each panel's impact frame.  Pre-impact
        # window is min(impact_idx); post-impact window extends until
        # the *latest*-finishing run runs out of frames (others freeze
        # on their last frame).
        T_min = -min(impact_idx_per_panel)
        T_max = max(len(runs[k][1]) - 1 - impact_idx_per_panel[k]
                    for k in range(3))
        n_frames = T_max - T_min + 1
        print(f"impact-aligned: T ∈ [{T_min}, {T_max}], "
              f"{n_frames} common-time steps "
              f"(impact offsets: {impact_idx_per_panel})")
    else:
        T_min = 0
        n_frames = min(len(frames) for _, frames in runs)
        T_max = n_frames - 1
        print(f"absolute lockstep: {n_frames} frames")

    # Per-panel caches: each is keyed by (frame_idx, ...) and stores the
    # already-translated geometry arrays.
    frame_caches: List[Dict[int, Dict[str, np.ndarray]]] = [{}, {}, {}]
    splat_caches: List[Dict[Tuple[int, str], object]] = [{}, {}, {}]
    surface_caches: List[Dict[int, object]] = [{}, {}, {}]
    offsets = np.array([
        [-args.x_offset, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [+args.x_offset, 0.0, 0.0],
    ], dtype=np.float64)

    def panel_idx_for_T(panel: int, T: int) -> int:
        """Map common-time T to this panel's local frame index.

        With ``--align-impact`` (default), T=0 is each panel's first
        impact frame, so we add the panel's impact-offset and clamp
        to the run's frame range (so panels that ended early freeze
        on their last frame).
        """
        if args.align_impact:
            local = impact_idx_per_panel[panel] + T
        else:
            local = T
        return max(0, min(local, len(runs[panel][1]) - 1))

    def get_frame(panel: int, idx: int) -> Dict[str, np.ndarray]:
        cache = frame_caches[panel]
        if idx not in cache:
            cache[idx] = read_houdini_geo(runs[panel][1][idx])
        return cache[idx]

    def get_splat(panel: int, idx: int, color_src: str):
        key = (idx, color_src)
        cache = splat_caches[panel]
        if key not in cache:
            cache[key] = build_splat_disk_mesh(
                get_frame(panel, idx),
                include_base=True,
                color_source=color_src,
                min_fragment_size=0,
                max_splats=int(args.max_splats),
            )
        return cache[key]

    def get_surface(panel: int, idx: int):
        cache = surface_caches[panel]
        if idx not in cache:
            cache[idx] = build_fragment_hull_mesh(
                get_frame(panel, idx), include_base=True)
        return cache[idx]

    # state["idx"] is a 0-based step index into [0, n_frames - 1].
    # The current common-time T is `state["idx"] + T_min`, so T=0
    # (impact moment) corresponds to idx = -T_min.  Start playback at
    # T = T_min (the earliest pre-impact frame any panel can show).
    state = {
        "idx": 0,
        "mode": args.mode,
        "playing": True,
        "last_step": time.time(),
    }

    def current_T() -> int:
        return state["idx"] + T_min

    pcds = [o3d.geometry.PointCloud() for _ in range(3)]
    meshes = [o3d.geometry.TriangleMesh() for _ in range(3)]
    empty_pts = np.zeros((0, 3), dtype=np.float64)

    # Bootstrap bbox from each panel's impact frame so the camera
    # frames the post-impact body, not the high-altitude pre-impact
    # bunny that may be far from the floor.
    bbox_min = None
    bbox_max = None
    for panel in range(3):
        rec = get_frame(panel, impact_idx_per_panel[panel])
        if rec["P"].size:
            P = rec["P"].astype(np.float64) + offsets[panel]
            bm = P.min(axis=0)
            bM = P.max(axis=0)
            bbox_min = bm if bbox_min is None else np.minimum(bbox_min, bm)
            bbox_max = bM if bbox_max is None else np.maximum(bbox_max, bM)
    if bbox_min is None:
        bbox_min, bbox_max = np.zeros(3), np.ones(3)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=("Tri-view: " + " | ".join(d.parent.name for d, _ in runs)),
        width=1920, height=720,
    )
    for pcd in pcds:
        vis.add_geometry(pcd)
    for mesh in meshes:
        vis.add_geometry(mesh)

    # Floor: one slab spanning all three panels.  Bake the slab at the
    # global minimum Z across runs (impact ground) and centre it on
    # the body's XY centroid at impact time.
    floor_z = None
    centre_xy = np.zeros(2)
    n_centre = 0
    for k, (_, frames_k) in enumerate(runs):
        # Sample a few late local frames per run for the floor Z.
        n_local = len(frames_k)
        sample_idx = sorted({impact_idx_per_panel[k],
                             max(impact_idx_per_panel[k], n_local - 5),
                             n_local - 1})
        for si in sample_idx:
            rec = get_frame(k, si)
            if rec["P"].size:
                z = float(rec["P"][:, 2].min())
                floor_z = z if floor_z is None else min(floor_z, z)
        rec_imp = get_frame(k, impact_idx_per_panel[k])
        if rec_imp["P"].size:
            centre_xy = centre_xy + rec_imp["P"][:, :2].mean(axis=0)
            n_centre += 1
    floor_z = 0.0 if floor_z is None else floor_z
    centre_xy = centre_xy / max(n_centre, 1)
    panel_span = 2.0 * args.x_offset  # leftmost panel to rightmost panel
    half_w = max(panel_span * 1.5 + bbox_diag * 2.0, 1.2)
    floor_cx, floor_cy = float(centre_xy[0]), float(centre_xy[1])
    floor = o3d.geometry.TriangleMesh.create_box(
        width=2.0 * half_w, height=2.0 * half_w, depth=0.002)
    floor.translate((floor_cx - half_w, floor_cy - half_w, floor_z - 0.002))
    floor.paint_uniform_color([0.18, 0.18, 0.20])
    floor.compute_vertex_normals()
    vis.add_geometry(floor)

    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.background_color = np.array([0.05, 0.05, 0.06])
    opt.mesh_show_back_face = True
    opt.light_on = False
    try:
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    except AttributeError:
        pass

    view_ctrl = vis.get_view_control()
    view_ctrl.set_up([0.0, 0.0, 1.0])
    view_ctrl.set_front([0.7, -0.7, 0.35])
    bbox_center = ((bbox_min + bbox_max) * 0.5).tolist()
    view_ctrl.set_lookat(bbox_center)
    view_ctrl.set_zoom(0.4)

    def _clear_pcd(panel: int):
        pcds[panel].points = o3d.utility.Vector3dVector(empty_pts)
        pcds[panel].colors = o3d.utility.Vector3dVector(empty_pts)

    def _clear_mesh(panel: int):
        meshes[panel].vertices = o3d.utility.Vector3dVector(empty_pts)
        meshes[panel].triangles = o3d.utility.Vector3iVector(
            np.zeros((0, 3), dtype=np.int32))
        meshes[panel].vertex_colors = o3d.utility.Vector3dVector(empty_pts)

    # Per-panel RGB tint applied in "cd" mode so the three runs are
    # immediately distinguishable as red/green/blue while preserving
    # the texture's luminance variation.
    panel_tints = np.array([
        [1.00, 0.35, 0.35],   # z=0.22 → red
        [0.35, 1.00, 0.35],   # z=0.50 → green
        [0.35, 0.35, 1.00],   # z=0.80 → blue
    ], dtype=np.float64)

    def _maybe_tint(panel: int, mode: str, cols: np.ndarray) -> np.ndarray:
        if mode in ("cd", "splat_cd"):
            return cols * panel_tints[panel]
        return cols

    def _set_pcd(panel: int, rec: Dict[str, np.ndarray], mode: str):
        P = _shifted_array(rec["P"].astype(np.float64), offsets[panel])
        cols = color_for_mode(rec, mode).astype(np.float64)
        cols = _maybe_tint(panel, mode, cols)
        pcds[panel].points = o3d.utility.Vector3dVector(P)
        pcds[panel].colors = o3d.utility.Vector3dVector(cols)

    def _set_mesh(panel: int, geom, mode: str = ""):
        if geom is None:
            _clear_mesh(panel)
            return
        V, F, C = geom
        V = _shifted_array(V.astype(np.float64), offsets[panel])
        C = _maybe_tint(panel, mode, C.astype(np.float64))
        meshes[panel].vertices = o3d.utility.Vector3dVector(V)
        meshes[panel].triangles = o3d.utility.Vector3iVector(F)
        meshes[panel].vertex_colors = o3d.utility.Vector3dVector(C)
        meshes[panel].compute_vertex_normals()

    def update_geom():
        T = current_T()
        for panel in range(3):
            local_i = panel_idx_for_T(panel, T)
            rec = get_frame(panel, local_i)
            if state["mode"] == "surface":
                _clear_pcd(panel)
                _set_mesh(panel, get_surface(panel, local_i), state["mode"])
            elif state["mode"] in ("splat_frag", "splat_cd"):
                color_src = "fragment" if state["mode"] == "splat_frag" else "cd"
                _clear_pcd(panel)
                _set_mesh(panel, get_splat(panel, local_i, color_src),
                          state["mode"])
            else:
                _clear_mesh(panel)
                _set_pcd(panel, rec, state["mode"])
            vis.update_geometry(pcds[panel])
            vis.update_geometry(meshes[panel])
        sys.stdout.write(
            f"\rT={T:+4d}  ({state['idx']+1:3d}/{n_frames})  "
            f"mode={state['mode']:11s}        ")
        sys.stdout.flush()

    update_geom()

    def step(delta: int):
        state["idx"] = (state["idx"] + delta) % n_frames
        update_geom()
        return False

    def toggle_play(_v):
        state["playing"] = not state["playing"]
        state["last_step"] = time.time()
        return False

    def set_mode(name: str):
        def cb(_v):
            state["mode"] = name
            update_geom()
            return False
        return cb

    def reset_cam(_v):
        view_ctrl = vis.get_view_control()
        view_ctrl.set_up([0.0, 0.0, 1.0])
        view_ctrl.set_front([0.7, -0.7, 0.35])
        view_ctrl.set_lookat(bbox_center)
        view_ctrl.set_zoom(0.4)
        return False

    def quit_cb(_v):
        vis.close()
        return False

    vis.register_key_callback(ord(" "), toggle_play)
    vis.register_key_callback(ord("N"), lambda _v: step(+1))
    vis.register_key_callback(ord("P"), lambda _v: step(-1))
    vis.register_key_callback(262, lambda _v: step(+1))
    vis.register_key_callback(263, lambda _v: step(-1))
    vis.register_key_callback(ord("1"), set_mode("cd"))
    vis.register_key_callback(ord("2"), set_mode("fragment"))
    vis.register_key_callback(ord("3"), set_mode("damage"))
    vis.register_key_callback(ord("4"), set_mode("surface"))
    vis.register_key_callback(ord("5"), set_mode("splat_frag"))
    vis.register_key_callback(ord("6"), set_mode("splat_cd"))
    vis.register_key_callback(ord("R"), reset_cam)
    vis.register_key_callback(ord("Q"), quit_cb)

    period = 1.0 / max(args.fps, 1.0)
    try:
        while True:
            now = time.time()
            if state["playing"] and (now - state["last_step"]) >= period:
                state["idx"] = (state["idx"] + 1) % n_frames
                state["last_step"] = now
                update_geom()
            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        sys.stdout.write("\n")
        vis.destroy_window()

    return 0


if __name__ == "__main__":
    sys.exit(main())
