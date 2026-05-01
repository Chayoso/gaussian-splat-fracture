"""Interactive 3D viewer for Houdini ``.geo`` / ``.geo.gz`` snapshots.

Reads the JSON-pair Houdini geo format written by
``src/diagnostics/houdini_export.py`` and renders it with Open3D.

Usage
-----
View a single file:

    python scripts/view_houdini_geo.py path/to/frame.geo.gz

Play a directory of frames:

    python scripts/view_houdini_geo.py path/to/houdini_export/

Keys
----
    Space  toggle play/pause
    N / →  next frame
    P / ←  previous frame
    1      points colored by base Cd
    2      points colored by fragment_id
    3      points colored by damage (heat)
    4      per-fragment thin-shell PCA-Delaunay surfaces
    5      oriented splat disks (fragment-colored, mirrors Houdini)
    6      oriented splat disks (base Cd, photoreal preview)
    B      toggle base body
    R      reset camera
    Q      quit
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------

def _pairs(seq) -> Dict:
    """Houdini's JSON geo stores dicts as flat ``[k, v, k, v, ...]`` lists."""
    return {seq[i]: seq[i + 1] for i in range(0, len(seq), 2)}


def _read_attr(entry) -> Optional[Dict]:
    header = _pairs(entry[0])
    body = _pairs(entry[1])
    name = header.get("name")
    size = int(body.get("size", 1))

    values = body.get("values", body)
    vp = _pairs(values) if isinstance(values, list) else {}
    if "tuples" in vp:
        arr = np.asarray(vp["tuples"], dtype=np.float32)
    elif "arrays" in vp:
        flat = vp["arrays"][0]
        arr = np.asarray(flat)
        if size > 1:
            arr = arr.reshape(-1, size)
    else:
        return None

    storage = vp.get("storage", "fpreal32")
    if "int" in storage:
        arr = arr.astype(np.int32)
    else:
        arr = arr.astype(np.float32)
    return {"name": name, "size": size, "data": arr}


def read_houdini_geo(path: Path) -> Dict[str, np.ndarray]:
    """Return a dict of point attributes from a Houdini ``.geo`` / ``.geo.gz``."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            blob = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    top = _pairs(blob)
    n = int(top.get("pointcount", 0))
    attribs_block = top.get("attributes", [])
    attribs = _pairs(attribs_block) if isinstance(attribs_block, list) else {}
    point_attrs = attribs.get("pointattributes", [])

    out: Dict[str, np.ndarray] = {}
    for entry in point_attrs:
        rec = _read_attr(entry)
        if rec is None:
            continue
        out[rec["name"]] = rec["data"]
    out["__count__"] = np.array([n])
    return out


# --------------------------------------------------------------------------
# Color modes
# --------------------------------------------------------------------------

_BASE_COLOR = np.array([0.55, 0.55, 0.58], dtype=np.float32)
_GOLDEN_RATIO_INV = 0.6180339887498948


def _hsv_single_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    rgb = [(v, t, p), (q, v, p), (p, v, t),
           (p, q, v), (t, p, v), (v, p, q)][i]
    return np.array(rgb, dtype=np.float32)


def _hsv_array_to_rgb(h: np.ndarray, s: float, v: float) -> np.ndarray:
    h = np.mod(h, 1.0)
    i = (h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    R = np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p,
        np.where(i == 3, p, np.where(i == 4, t, v)))))
    G = np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v,
        np.where(i == 3, q, np.where(i == 4, p, p)))))
    B = np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t,
        np.where(i == 3, v, np.where(i == 4, v, q)))))
    return np.stack([R, G, B], axis=-1).astype(np.float32)


def _heat(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32).clip(0.0, 1.0)
    r = np.clip(1.5 * v, 0.0, 1.0)
    g = np.clip(1.5 * v - 0.5, 0.0, 1.0)
    b = np.clip(1.5 * v - 1.0, 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


def _fragment_palette_for(fids: np.ndarray) -> np.ndarray:
    """HSV golden-angle hues per fragment id. id==0 -> grey base color.

    Saturated, bright colors so fragment boundaries pop out even when
    hundreds of fragments coexist on screen.
    """
    fids = np.asarray(fids).reshape(-1).astype(np.int64)
    base_mask = fids <= 0
    hues = ((fids.astype(np.float64) * _GOLDEN_RATIO_INV) % 1.0).astype(np.float32)
    rgb = _hsv_array_to_rgb(hues, s=0.85, v=0.95)
    rgb[base_mask] = _BASE_COLOR
    return rgb


def color_for_mode(rec: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    n = int(rec["P"].shape[0])
    if mode == "cd" and "Cd" in rec:
        return rec["Cd"].reshape(n, 3).clip(0.0, 1.0)
    if mode == "fragment" and "fragment_id" in rec:
        return _fragment_palette_for(rec["fragment_id"])
    if mode == "damage" and "damage" in rec:
        return _heat(rec["damage"].reshape(n))
    return np.full((n, 3), 0.7, dtype=np.float32)


def _color_for_fragment(f_id: int) -> np.ndarray:
    if f_id <= 0:
        return _BASE_COLOR.copy()
    h = float(f_id) * _GOLDEN_RATIO_INV
    return _hsv_single_to_rgb(h - int(h), 0.85, 0.95)


def _pca_plane_triangulate(points: np.ndarray) -> Optional[np.ndarray]:
    """Triangulate a thin-shell fragment by projecting onto its PCA plane.

    Returns triangle indices (M, 3) referencing rows of ``points``, or None
    if the fragment is too small / too degenerate / triangulation fails.
    """
    n = points.shape[0]
    if n < 4:
        return None
    centroid = points.mean(axis=0)
    centered = points - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    # vh rows are principal axes sorted by descending singular value;
    # the first two span the dominant plane (in-shell axes), the third
    # is the normal (through-shell axis).
    plane = vh[:2].T   # (3, 2)
    proj = centered @ plane  # (N, 2)
    spread = proj.max(axis=0) - proj.min(axis=0)
    if float(spread.min()) < 1e-7:
        return None
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(proj, qhull_options="QJ")
    except Exception:
        return None
    simplices = np.asarray(tri.simplices, dtype=np.int32)
    if simplices.shape[0] == 0:
        return None
    # Filter out very stretched triangles (poorly-conditioned far-field
    # connections caused by the convex Delaunay envelope).
    v0 = points[simplices[:, 0]]
    v1 = points[simplices[:, 1]]
    v2 = points[simplices[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    longest = np.maximum(np.maximum(e0, e1), e2)
    shortest = np.minimum(np.minimum(e0, e1), e2)
    median_long = float(np.median(longest))
    if median_long <= 0.0:
        return None
    aspect_ok = longest <= max(median_long * 4.0, 0.005)
    nontrivial = shortest > 1e-7
    keep = aspect_ok & nontrivial
    simplices = simplices[keep]
    if simplices.shape[0] == 0:
        return None
    return simplices


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Vectorized (N, 4) quaternion (x, y, z, w; Houdini ijks order) -> (N, 3, 3)."""
    q = np.asarray(q, dtype=np.float32)
    n = q.shape[0]
    norm = np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-8)
    q = q / norm
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


def _fragment_size_filter(fid: np.ndarray, min_size: int) -> np.ndarray:
    """Return a boolean keep-mask: drop splats whose fragment has < min_size splats."""
    if min_size <= 1:
        return np.ones(fid.shape[0], dtype=bool)
    uniq, inv, counts = np.unique(fid, return_inverse=True, return_counts=True)
    keep_label = counts >= int(min_size)
    return keep_label[inv]


def build_splat_disk_mesh(rec: Dict[str, np.ndarray],
                          include_base: bool = False,
                          color_source: str = "fragment",
                          target_disk_frac: float = 0.006,
                          min_fragment_size: int = 0,
                          max_splats: int = 0):
    """Render each Gaussian as an oriented quad-disk using its P, orient, scale.

    The disk lies in the plane perpendicular to the splat's smallest local
    axis (i.e. the ellipsoid's thin direction).  This mirrors what Houdini's
    Copy-to-Points + per-point ``orient`` + per-point ``scale`` does, so the
    viewer preview matches the eventual photoreal render reasonably well.

    ``color_source``: ``"fragment"`` for fragment_id palette colors,
    ``"cd"`` for stored base RGB.
    """
    P_all = rec["P"]
    orient_all = rec.get("orient")
    scale_all = rec.get("scale")
    if orient_all is None or scale_all is None:
        return None
    fid_all = rec.get("fragment_id",
                      np.zeros(P_all.shape[0], dtype=np.int32)).reshape(-1).astype(np.int64)
    cd_all = rec.get("Cd")

    if include_base:
        keep = np.ones(P_all.shape[0], dtype=bool)
    else:
        keep = fid_all > 0
    if min_fragment_size > 1:
        keep = keep & _fragment_size_filter(fid_all, min_fragment_size)
    if not keep.any():
        return None
    # Render-time subsample for high-particle sims (150K disks at 2 tris
    # each = 300K tris/frame -- mesh swap stutters in Open3D).  Drop a
    # deterministic random subset down to max_splats; per-frame keeps
    # the same seed so the same particles render across frames (no
    # flicker).
    if max_splats > 0 and int(keep.sum()) > max_splats:
        kept_idx = np.where(keep)[0]
        rng = np.random.default_rng(seed=0xC0FFEE)
        sample = rng.choice(kept_idx, size=int(max_splats), replace=False)
        new_keep = np.zeros_like(keep)
        new_keep[sample] = True
        keep = new_keep

    P = P_all[keep].astype(np.float32)
    orient = orient_all[keep].astype(np.float32)
    scale = scale_all[keep].astype(np.float32)
    fid = fid_all[keep]
    n = P.shape[0]

    R = _quat_xyzw_to_rotmat(orient)

    # Adaptive disk size: stored scales for untrained Gaussians can be
    # absurdly large relative to the world bbox (mean ~ 1.0 for a 0.4-unit
    # bunny).  Re-normalize so disks span ~target_disk_frac of bbox max.
    bbox_size = float((P_all.max(axis=0) - P_all.min(axis=0)).max())
    target = max(bbox_size * float(target_disk_frac), 1e-6)
    median_scale = float(np.median(scale))
    if median_scale > 1e-7:
        scale_norm = target / median_scale
    else:
        scale_norm = target

    # Smallest local axis is the splat normal (thin direction); the disk
    # extents along the other two axes (rescaled to a sane visual size).
    smallest = np.argmin(scale, axis=1)
    a_idx = np.where(smallest == 0, 1, 0)
    b_idx = np.where(smallest == 2, 1, 2)
    rows = np.arange(n)
    sa = scale[rows, a_idx] * scale_norm
    sb = scale[rows, b_idx] * scale_norm

    # Local-frame axis vectors as rotation-matrix columns
    a_world = R[rows, :, a_idx] * sa[:, None]
    b_world = R[rows, :, b_idx] * sb[:, None]

    c0 = P - a_world - b_world
    c1 = P + a_world - b_world
    c2 = P + a_world + b_world
    c3 = P - a_world + b_world

    verts = np.empty((n * 4, 3), dtype=np.float32)
    verts[0::4] = c0
    verts[1::4] = c1
    verts[2::4] = c2
    verts[3::4] = c3

    base = (rows * 4).reshape(n, 1)
    tri_a = np.concatenate(
        [base + 0, base + 1, base + 2], axis=1)
    tri_b = np.concatenate(
        [base + 0, base + 2, base + 3], axis=1)
    tris = np.empty((n * 2, 3), dtype=np.int32)
    tris[0::2] = tri_a
    tris[1::2] = tri_b

    if color_source == "cd" and cd_all is not None:
        per_splat_color = cd_all[keep].astype(np.float32).clip(0.0, 1.0)
    else:
        per_splat_color = _fragment_palette_for(fid)

    colors = np.repeat(per_splat_color, 4, axis=0).astype(np.float32)
    return verts.astype(np.float64), tris, colors.astype(np.float64)


def build_fragment_hull_mesh(rec: Dict[str, np.ndarray],
                             include_base: bool = False,
                             min_points: int = 6):
    """Build one combined TriangleMesh of per-fragment thin-shell surfaces.

    Each fragment is triangulated in its PCA plane (so a thin shell stays
    a thin shell instead of becoming a solid volume).  Returns
    (vertices Nx3, triangles Mx3 int32, vertex_colors Nx3) or None if
    nothing could be built.
    """
    P = rec["P"]
    if "fragment_id" not in rec:
        return None
    fid = rec["fragment_id"].reshape(-1).astype(np.int64)
    unique = np.unique(fid)

    Vs: List[np.ndarray] = []
    Fs: List[np.ndarray] = []
    Cs: List[np.ndarray] = []
    voff = 0
    for f in unique:
        if not include_base and f == 0:
            continue
        mask = fid == f
        pts = P[mask].astype(np.float64)
        if pts.shape[0] < min_points:
            continue
        tris = _pca_plane_triangulate(pts)
        if tris is None:
            continue
        color = _color_for_fragment(int(f))
        Vs.append(pts)
        Fs.append(tris + voff)
        Cs.append(np.tile(color.astype(np.float64), (pts.shape[0], 1)))
        voff += pts.shape[0]

    if not Vs:
        return None
    return (np.concatenate(Vs, axis=0),
            np.concatenate(Fs, axis=0).astype(np.int32),
            np.concatenate(Cs, axis=0))


# --------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------

def _list_frames(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir()
                   if p.suffix == ".gz" or p.suffix == ".geo")
    if not files:
        raise FileNotFoundError(f"No .geo / .geo.gz files in {path}")
    return files


def run_viewer(target: Path, point_size: float = 6.0,
               fps: float = 30.0, start_mode: str = "fragment",
               hide_base: bool = False,
               min_fragment_size: int = 0,
               max_splats: int = 0) -> None:
    import open3d as o3d

    frames = _list_frames(target)
    print(f"loaded {len(frames)} frame(s) from {target}")

    cache: Dict[int, Dict[str, np.ndarray]] = {}
    surface_cache: Dict[tuple, object] = {}
    splat_cache: Dict[tuple, object] = {}

    def get_frame(i: int) -> Dict[str, np.ndarray]:
        if i not in cache:
            cache[i] = read_houdini_geo(frames[i])
        return cache[i]

    def get_surface(i: int, include_base: bool):
        key = (i, include_base)
        if key not in surface_cache:
            surface_cache[key] = build_fragment_hull_mesh(
                get_frame(i), include_base=include_base)
        return surface_cache[key]

    def get_splat(i: int, include_base: bool, color_source: str,
                  min_frag: int, max_n: int):
        key = (i, include_base, color_source, int(min_frag), int(max_n))
        if key not in splat_cache:
            splat_cache[key] = build_splat_disk_mesh(
                get_frame(i), include_base=include_base,
                color_source=color_source,
                min_fragment_size=int(min_frag),
                max_splats=int(max_n))
        return splat_cache[key]

    state = {
        "idx": 0,
        "mode": start_mode,
        "playing": False,
        "last_step": time.time(),
        "show_base": not hide_base,
        "min_frag_size": int(min_fragment_size),
        "max_splats": int(max_splats),
    }

    rec0 = get_frame(0)
    empty_pts = np.zeros((0, 3), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rec0["P"].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        color_for_mode(rec0, "fragment").astype(np.float64))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(empty_pts)
    mesh.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Houdini geo viewer  [{frames[0].parent.name}]",
                      width=1280, height=800)
    vis.add_geometry(pcd)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([0.05, 0.05, 0.06])
    opt.mesh_show_back_face = True
    # Disable Phong shading so per-vertex fragment colors render as
    # specified (lighting otherwise washes saturated palette into greys).
    opt.light_on = False
    try:
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    except AttributeError:
        pass

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=[0, 0, 0])
    vis.add_geometry(axes)

    # Z-up scene: gravity is along -Z, so the floor is a thin slab in XY at
    # the global minimum Z across all frames.  Using only frame 0 (which
    # may be pre-impact, with the bunny still high in the air) leaves the
    # floor floating mid-scene; sample a few late frames to find the
    # actual ground level.
    sample_idx = sorted(set(
        [0, len(frames) // 2, max(len(frames) - 1, 0),
         max(len(frames) - 5, 0), max(len(frames) - 10, 0)]
    ))
    floor_z = None
    for si in sample_idx:
        rec = get_frame(si)
        if rec["P"].size:
            zmin = float(rec["P"][:, 2].min())
            floor_z = zmin if floor_z is None else min(floor_z, zmin)
    if floor_z is None:
        floor_z = 0.0
    if rec0["P"].size:
        cx = float(rec0["P"][:, 0].mean())
        cy = float(rec0["P"][:, 1].mean())
        bbox_diag = float(np.linalg.norm(
            rec0["P"].max(axis=0) - rec0["P"].min(axis=0)))
        half = max(bbox_diag * 1.5, 0.25)
    else:
        cx, cy, half = 0.0, 0.0, 1.0
    floor = o3d.geometry.TriangleMesh.create_box(
        width=2.0 * half, height=2.0 * half, depth=0.002)
    floor.translate((cx - half, cy - half, floor_z - 0.002))
    floor.paint_uniform_color([0.18, 0.18, 0.20])
    floor.compute_vertex_normals()
    vis.add_geometry(floor)

    # Camera defaults: Z-up world, 3/4 perspective from +X / -Y / +Z corner,
    # lookat at the bbox center of the first frame (bbox center is more
    # stable than mean if a few fragments fly far away).
    if rec0["P"].size:
        bbox_min = rec0["P"].min(axis=0)
        bbox_max = rec0["P"].max(axis=0)
        bbox_center = ((bbox_min + bbox_max) * 0.5).tolist()
    else:
        bbox_center = [0.0, 0.0, 0.0]

    view_ctrl = vis.get_view_control()
    view_ctrl.set_up([0.0, 0.0, 1.0])
    view_ctrl.set_front([0.7, -0.7, 0.35])
    view_ctrl.set_lookat(bbox_center)
    view_ctrl.set_zoom(0.7)

    title_path = frames[0]

    def _set_pcd_visible(rec, mode):
        P = rec["P"].astype(np.float64)
        cols = color_for_mode(rec, mode).astype(np.float64)
        fid = rec.get("fragment_id", None)
        if fid is not None:
            fid_flat = fid.reshape(-1).astype(np.int64)
            keep = np.ones(P.shape[0], dtype=bool)
            if not state["show_base"]:
                keep &= fid_flat > 0
            if state["min_frag_size"] > 1:
                keep &= _fragment_size_filter(fid_flat, state["min_frag_size"])
            P = P[keep]
            cols = cols[keep]
        pcd.points = o3d.utility.Vector3dVector(P)
        pcd.colors = o3d.utility.Vector3dVector(cols)

    def _set_pcd_empty():
        pcd.points = o3d.utility.Vector3dVector(empty_pts)
        pcd.colors = o3d.utility.Vector3dVector(empty_pts)

    def _set_mesh_visible(surface):
        V, F, C = surface
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.triangles = o3d.utility.Vector3iVector(F)
        mesh.vertex_colors = o3d.utility.Vector3dVector(C)
        mesh.compute_vertex_normals()

    def _set_mesh_empty():
        mesh.vertices = o3d.utility.Vector3dVector(empty_pts)
        mesh.triangles = o3d.utility.Vector3iVector(
            np.zeros((0, 3), dtype=np.int32))
        mesh.vertex_colors = o3d.utility.Vector3dVector(empty_pts)

    def update_geometry():
        rec = get_frame(state["idx"])
        nfrag = 0
        if "fragment_id" in rec:
            fids = rec["fragment_id"].reshape(-1).astype(np.int64)
            uniq = np.unique(fids)
            nfrag = int((uniq != 0).sum())
        if state["mode"] == "surface":
            surface = get_surface(state["idx"], state["show_base"])
            if surface is None:
                _set_mesh_empty()
            else:
                _set_mesh_visible(surface)
            _set_pcd_empty()
        elif state["mode"] in ("splat_frag", "splat_cd"):
            color_src = "fragment" if state["mode"] == "splat_frag" else "cd"
            splat = get_splat(state["idx"], state["show_base"],
                              color_src, state["min_frag_size"],
                              state["max_splats"])
            if splat is None:
                _set_mesh_empty()
            else:
                _set_mesh_visible(splat)
            _set_pcd_empty()
        else:
            _set_mesh_empty()
            _set_pcd_visible(rec, state["mode"])
        vis.update_geometry(pcd)
        vis.update_geometry(mesh)
        nonlocal title_path
        title_path = frames[state["idx"]]
        sys.stdout.write(
            f"\rframe {state['idx']+1:3d}/{len(frames)}  "
            f"mode={state['mode']:9s}  pts={rec['P'].shape[0]:6d}  "
            f"frags={nfrag}  base={'on' if state['show_base'] else 'off'}  "
            f"file={title_path.name}        ")
        sys.stdout.flush()

    update_geometry()

    def step(delta: int):
        new_idx = (state["idx"] + delta) % len(frames)
        state["idx"] = new_idx
        update_geometry()
        return False

    def set_mode(mode: str):
        state["mode"] = mode
        update_geometry()
        return False

    def toggle_base(v):
        state["show_base"] = not state["show_base"]
        update_geometry()
        return False

    vis.register_key_callback(ord("N"), lambda v: step(+1))
    vis.register_key_callback(ord("P"), lambda v: step(-1))
    vis.register_key_callback(262, lambda v: step(+1))   # right arrow
    vis.register_key_callback(263, lambda v: step(-1))   # left arrow
    vis.register_key_callback(ord("1"), lambda v: set_mode("cd"))
    vis.register_key_callback(ord("2"), lambda v: set_mode("fragment"))
    vis.register_key_callback(ord("3"), lambda v: set_mode("damage"))
    vis.register_key_callback(ord("4"), lambda v: set_mode("surface"))
    vis.register_key_callback(ord("5"), lambda v: set_mode("splat_frag"))
    vis.register_key_callback(ord("6"), lambda v: set_mode("splat_cd"))
    vis.register_key_callback(ord("B"), toggle_base)

    def toggle_play(v):
        state["playing"] = not state["playing"]
        state["last_step"] = time.time()
        return False

    def reset_view(v):
        rec = get_frame(state["idx"])
        vc = vis.get_view_control()
        vc.set_up([0.0, 0.0, 1.0])
        vc.set_front([0.7, -0.7, 0.35])
        if rec["P"].size:
            bb_min = rec["P"].min(axis=0)
            bb_max = rec["P"].max(axis=0)
            vc.set_lookat(((bb_min + bb_max) * 0.5).tolist())
        else:
            vc.set_lookat([0.0, 0.0, 0.0])
        vc.set_zoom(0.7)
        return False

    vis.register_key_callback(ord(" "), toggle_play)
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("Q"), lambda v: (vis.close(), False)[1])

    period = 1.0 / max(0.1, fps)
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
        if state["playing"] and len(frames) > 1:
            now = time.time()
            if now - state["last_step"] >= period:
                state["last_step"] = now
                state["idx"] = (state["idx"] + 1) % len(frames)
                update_geometry()
    sys.stdout.write("\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path,
                    help="single .geo[.gz] file or directory of frames")
    ap.add_argument("--point-size", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=30.0,
                    help="playback fps (try 30, 60). 8 = old default, very slow.")
    ap.add_argument("--mode",
                    choices=["cd", "fragment", "damage", "surface",
                             "splat_frag", "splat_cd"],
                    default="splat_frag")
    ap.add_argument("--hide-base", action="store_true",
                    help="hide base body (fragment_id == 0) in point modes")
    ap.add_argument("--min-fragment-size", type=int, default=0,
                    help="hide fragments with fewer than N splats (any mode)")
    ap.add_argument("--max-splats", type=int, default=0,
                    help="cap splat-disk render count (subsample for "
                         "smooth playback at 100K+ particle sims)")
    ap.add_argument("--dump", action="store_true",
                    help="just print attributes of the first frame and exit")
    args = ap.parse_args()

    if args.dump:
        frames = _list_frames(args.path)
        rec = read_houdini_geo(frames[0])
        print(f"file: {frames[0]}")
        for k, v in rec.items():
            if k.startswith("__"):
                continue
            print(f"  {k:14s}  shape={v.shape}  dtype={v.dtype}")
        return 0

    run_viewer(args.path, point_size=args.point_size,
               fps=args.fps, start_mode=args.mode,
               hide_base=args.hide_base,
               min_fragment_size=args.min_fragment_size,
               max_splats=args.max_splats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
