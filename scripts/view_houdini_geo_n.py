"""N-panel side-by-side viewer for arbitrary number of houdini_export
directories (2 - 6 panels).  Same controls as ``view_houdini_geo_tri.py``
but extended to handle 4, 5, or 6 simultaneous runs.

Usage:

    python scripts/view_houdini_geo_n.py \
        output/abl10k_A_glass_full_z0.80_v25f/houdini_export \
        output/abl10k_A_ceramic_full_z0.80_v25f/houdini_export \
        output/abl10k_A_concrete_full_z0.80_v25f/houdini_export \
        output/abl10k_A_rubber_full_z0.80_v25f/houdini_export \
        output/abl10k_A_wood_full_z0.80_v25f/houdini_export

Keys: same as tri-viewer (Space play/pause, N/P frames, 1-6 modes,
R reset cam, Q quit).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="Two or more houdini_export directories.")
    ap.add_argument("--point-size", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--x-offset", type=float, default=0.40)
    ap.add_argument("--mode", choices=("cd", "fragment", "damage",
                                        "surface", "splat_frag", "splat_cd"),
                    default="splat_cd")
    ap.add_argument("--max-splats", type=int, default=0)
    ap.add_argument("--align-impact", dest="align_impact",
                    action="store_true", default=True)
    ap.add_argument("--no-align-impact", dest="align_impact",
                    action="store_false")
    args = ap.parse_args()
    if len(args.dirs) < 2 or len(args.dirs) > 6:
        ap.error("Need 2-6 directories.")

    import open3d as o3d
    n_panels = len(args.dirs)
    panel_idx_range = range(n_panels)

    print(f"loading {n_panels} runs...")
    runs: List[Tuple[Path, List[Path]]] = []
    impact_idx_per_panel: List[int] = []
    for d in args.dirs:
        frames = _list_frames(d)
        impact_pos = next(
            (i for i, f in enumerate(frames) if "_impact_" in f.name),
            len(frames) - 1,
        )
        impact_idx_per_panel.append(impact_pos)
        print(f"  {d.parent.name}/{d.name}: {len(frames)} frames "
              f"(impact at frame {impact_pos})")
        runs.append((d, frames))

    if args.align_impact:
        T_min = -min(impact_idx_per_panel)
        T_max = max(len(runs[k][1]) - 1 - impact_idx_per_panel[k]
                    for k in panel_idx_range)
        n_frames = T_max - T_min + 1
        print(f"impact-aligned: T ∈ [{T_min}, {T_max}], "
              f"{n_frames} common-time steps")
    else:
        T_min = 0
        n_frames = min(len(frames) for _, frames in runs)
        T_max = n_frames - 1
        print(f"absolute lockstep: {n_frames} frames")

    # Centred X offsets so the whole row is symmetric around 0.
    offsets = np.zeros((n_panels, 3), dtype=np.float64)
    for k in panel_idx_range:
        offsets[k, 0] = (k - (n_panels - 1) / 2.0) * args.x_offset

    # Per-panel tint: rainbow palette generated from HSV so any N looks ok.
    panel_tints = np.zeros((n_panels, 3), dtype=np.float64)
    for k in panel_idx_range:
        h = k / max(n_panels, 1)
        # Saturated colour, roughly: red, orange, yellow, green, blue, purple
        if n_panels <= 3:
            base = np.array([
                [1.00, 0.35, 0.35],
                [0.35, 1.00, 0.35],
                [0.35, 0.35, 1.00],
            ])
            panel_tints[k] = base[k]
        else:
            # 6-color rotation.
            ring = np.array([
                [1.00, 0.40, 0.40],   # red
                [1.00, 0.70, 0.30],   # orange
                [0.95, 0.95, 0.40],   # yellow
                [0.40, 0.95, 0.50],   # green
                [0.40, 0.65, 1.00],   # blue
                [0.85, 0.50, 1.00],   # purple
            ])
            panel_tints[k] = ring[k % 6]

    frame_caches: List[Dict[int, Dict[str, np.ndarray]]] = [{} for _ in panel_idx_range]
    splat_caches: List[Dict[Tuple[int, str], object]] = [{} for _ in panel_idx_range]
    surface_caches: List[Dict[int, object]] = [{} for _ in panel_idx_range]

    def panel_idx_for_T(panel: int, T: int) -> int:
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

    state = {"idx": 0, "mode": args.mode, "playing": True,
             "last_step": time.time()}

    def current_T() -> int:
        return state["idx"] + T_min

    pcds = [o3d.geometry.PointCloud() for _ in panel_idx_range]
    meshes = [o3d.geometry.TriangleMesh() for _ in panel_idx_range]
    empty_pts = np.zeros((0, 3), dtype=np.float64)

    bbox_min = None
    bbox_max = None
    for panel in panel_idx_range:
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
        window_name=("N-view: " + " | ".join(d.parent.name for d, _ in runs)),
        width=min(1920, 380 * n_panels), height=720,
    )
    for pcd in pcds:
        vis.add_geometry(pcd)
    for mesh in meshes:
        vis.add_geometry(mesh)

    floor_z = None
    centre_xy = np.zeros(2)
    n_centre = 0
    for k, (_, frames_k) in enumerate(runs):
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
    panel_span = (n_panels - 1) * args.x_offset
    half_w = max(panel_span * 1.0 + bbox_diag * 2.0, 1.2)
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
    # Back-face culling on: previously the floor's bottom face was
    # rendered too, producing a visible "mirror" of the bunny below
    # the floor whenever the camera looked up through the slab.
    opt.mesh_show_back_face = False
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

    def _maybe_tint(panel: int, mode: str, cols: np.ndarray) -> np.ndarray:
        if mode in ("cd", "splat_cd"):
            return cols * panel_tints[panel]
        return cols

    def _clear_pcd(panel: int):
        pcds[panel].points = o3d.utility.Vector3dVector(empty_pts)
        pcds[panel].colors = o3d.utility.Vector3dVector(empty_pts)

    def _clear_mesh(panel: int):
        meshes[panel].vertices = o3d.utility.Vector3dVector(empty_pts)
        meshes[panel].triangles = o3d.utility.Vector3iVector(
            np.zeros((0, 3), dtype=np.int32))
        meshes[panel].vertex_colors = o3d.utility.Vector3dVector(empty_pts)

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
        for panel in panel_idx_range:
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
