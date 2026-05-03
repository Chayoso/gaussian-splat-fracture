"""Scatter-quality metric for brittle shatter sims.

Beyond verify_smoke (base==0, lateral>thr, bbox>thr), this measures
*outward* radial motion (away from body center, not just one direction),
*direction entropy* (sphere coverage), and *floater count* (orphan
phantom Gaussians).

Targets a "흩날린다" shatter:
  - outward_frac >= 0.50            # frags whose XY com moved outward
  - radial_growth_p90 >= 0.10       # 90th-pct radial displacement
  - direction_entropy >= 1.5 bits   # 8-sector XY direction histogram
  - bbox_spread_peak >= 1.8         # XY bbox grew at least 1.8x at peak
  - floater_count <= 5              # phantom orphan count
  - base_frac <= 0.001              # essentially 0 base remnant
  - n_unique_fragments >= 50        # not too few chunks
"""
from __future__ import annotations
import sys, gzip, json, math
from pathlib import Path
from typing import Iterable
import numpy as np


def _pairs(seq):
    if not isinstance(seq, list):
        return {}
    return {seq[i]: seq[i + 1] for i in range(0, len(seq) - 1, 2)}


def _read_attr(entry):
    if not isinstance(entry, list) or len(entry) < 2:
        return None
    head = _pairs(entry[0])
    body = _pairs(entry[1])
    name = head.get("name")
    if name is None:
        return None
    values = body.get("values")
    if values is None:
        return None
    vals = _pairs(values) if isinstance(values, list) else {}
    arrays = vals.get("arrays")
    tuples = vals.get("tuples")
    raw = arrays if arrays is not None else tuples
    if raw is None or not isinstance(raw, list):
        return None
    flat = []
    for chunk in raw:
        if isinstance(chunk, list):
            for v in chunk:
                flat.append(v)
        else:
            flat.append(chunk)
    sz = int(vals.get("size", 1))
    arr = np.asarray(flat)
    if sz > 1 and arr.size % sz == 0:
        arr = arr.reshape(-1, sz)
    return {"name": name, "data": arr}


def read_geo(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            blob = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    top = _pairs(blob)
    attribs = _pairs(top.get("attributes", []))
    out = {}
    for entry in attribs.get("pointattributes", []):
        rec = _read_attr(entry)
        if rec is not None:
            out[rec["name"]] = rec["data"]
    return out


def list_frames(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir()
                  if p.suffix in (".geo", ".gz"))


def compute_metrics(houdini_dir: Path) -> dict:
    frames = list_frames(houdini_dir)
    if not frames:
        return {"ok": False, "reason": "no frames"}

    rec0 = read_geo(frames[0])
    rec1 = read_geo(frames[-1])
    P0 = np.asarray(rec0.get("P"), dtype=np.float32)
    P1 = np.asarray(rec1.get("P"), dtype=np.float32)
    fid0 = np.asarray(rec0.get("fragment_id"), dtype=np.int64).reshape(-1)
    fid1 = np.asarray(rec1.get("fragment_id"), dtype=np.int64).reshape(-1)
    if P0.ndim != 2 or P1.ndim != 2 or P0.shape[1] != 3 or P1.shape[1] != 3:
        return {"ok": False, "reason": "bad P shape"}

    n_last = len(fid1)
    n_base_last = int((fid1 == 0).sum())
    base_frac = n_base_last / max(1, n_last)
    floater_count = int((fid1 == -1).sum())

    valid_mask = fid1 > 0
    if not valid_mask.any():
        return {"ok": False, "reason": "no fragments at last"}
    fid_v = fid1[valid_mask]
    P_v = P1[valid_mask]

    # Body XY center proxy = first-frame median XY
    body_com_xy = np.array([
        float(np.median(P0[:, 0])),
        float(np.median(P0[:, 1])),
    ], dtype=np.float32)

    unique_fids = np.unique(fid_v)
    com_xy_end = []
    com_xy_start = []
    for f in unique_fids:
        m_e = fid_v == f
        if m_e.sum() < 2:
            continue
        com_xy_end.append(P_v[m_e, :2].mean(0))
        m_s = fid0 == f
        if m_s.any():
            com_xy_start.append(P0[m_s, :2].mean(0))
        else:
            com_xy_start.append(body_com_xy.copy())
    if not com_xy_end:
        return {"ok": False, "reason": "no valid fragment coms"}
    com_xy_end = np.stack(com_xy_end)
    com_xy_start = np.stack(com_xy_start)
    n_unique = len(com_xy_end)

    r_end = np.linalg.norm(com_xy_end - body_com_xy[None], axis=1)
    r_start = np.linalg.norm(com_xy_start - body_com_xy[None], axis=1)
    r_growth = r_end - r_start
    r_growth_mean = float(np.mean(r_growth))
    r_growth_p90 = float(np.percentile(r_growth, 90))
    outward_frac = float(np.mean(r_growth > 0.01))

    dxy = com_xy_end - body_com_xy[None]
    angles = np.arctan2(dxy[:, 1], dxy[:, 0])
    sectors = ((angles + math.pi) / (2 * math.pi) * 8).astype(np.int64)
    sectors = sectors.clip(0, 7)
    bin_counts = np.bincount(sectors, minlength=8).astype(np.float32)
    p = bin_counts / bin_counts.sum().clip(min=1)
    p_safe = np.where(p > 0, p, 1.0)
    direction_entropy = float(-np.sum(p * np.log2(p_safe)))

    # Anisotropy: PCA on fragment-COM XY distribution.  Perfect circular
    # explosion -> ratio ~1.0 (isotropic).  Natural fracture is irregular
    # -> elongated cloud, ratio > 1.5.
    if len(com_xy_end) >= 3:
        cxy = com_xy_end - com_xy_end.mean(axis=0, keepdims=True)
        cov = (cxy.T @ cxy) / max(len(cxy), 1)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        anisotropy_ratio = float(np.sqrt(max(eigvals[0], 1e-12)
                                         / max(eigvals[1], 1e-12)))
    else:
        anisotropy_ratio = 1.0

    # Radial-uniformity gap: variance of radial distance.  Pure circular
    # explosion has all frags at similar radius -> very low variance.
    # Natural fracture has chunks at varied radii (clusters near impact,
    # scattered elsewhere).
    radial_cv = float(np.std(r_end) / max(np.mean(r_end), 1e-6))

    # bbox spread peak across all frames
    bbox0 = P0[:, :2].max(0) - P0[:, :2].min(0)
    bbox0_max = float(bbox0.max())
    peak_spread = 1.0
    for f in frames:
        rec = read_geo(f)
        Pf = np.asarray(rec.get("P"), dtype=np.float32)
        fidf = np.asarray(rec.get("fragment_id"), dtype=np.int64).reshape(-1)
        if Pf.ndim != 2 or Pf.shape[1] != 3:
            continue
        m = fidf > 0
        if m.sum() < 5:
            continue
        bb = Pf[m, :2].max(0) - Pf[m, :2].min(0)
        peak_spread = max(peak_spread, float(bb.max() / max(bbox0_max, 1e-6)))

    return {
        "ok": True,
        "n_frames": len(frames),
        "n_particles_last": n_last,
        "n_base_last": n_base_last,
        "base_frac": base_frac,
        "n_unique_fragments": n_unique,
        "floater_count": floater_count,
        "outward_frac": outward_frac,
        "radial_growth_mean": r_growth_mean,
        "radial_growth_p90": r_growth_p90,
        "direction_entropy": direction_entropy,
        "bbox_spread_peak": peak_spread,
        "anisotropy_ratio": anisotropy_ratio,
        "radial_cv": radial_cv,
    }


TARGETS = {
    "outward_frac_min": 0.50,
    "radial_growth_p90_min": 0.10,
    "direction_entropy_min": 1.5,
    "bbox_spread_peak_min": 1.8,
    "floater_count_max": 5,
    "base_frac_max": 0.001,
    "n_unique_fragments_min": 50,
    # Natural fracture targets: anisotropy > 1.4 = NOT a perfect circle.
    # radial_cv > 0.30 = chunks at varied radii (clusters + scatter),
    # not all at the same radius like a uniform shockwave.
    "anisotropy_ratio_min": 1.40,
    "radial_cv_min": 0.30,
    # Direction entropy upper bound: too high (>2.7) = perfect uniform
    # circle, NOT natural fracture.  Target [1.5, 2.7] band: directional
    # but not single-axis drift.
    "direction_entropy_max": 2.85,
}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: scatter_metric.py <houdini_export_dir>")
        return 2
    m = compute_metrics(Path(sys.argv[1]))
    if not m.get("ok"):
        print(f"FAIL: {m.get('reason')}")
        return 1
    print("=== Scatter Quality Metric ===")
    for k in ("n_frames", "n_particles_last", "n_base_last", "base_frac",
              "n_unique_fragments", "floater_count", "outward_frac",
              "radial_growth_mean", "radial_growth_p90",
              "direction_entropy", "bbox_spread_peak"):
        v = m[k]
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.4f}")
        else:
            print(f"  {k:30s} = {v}")
    print("=== Targets ===")
    pass_all = True
    def check(name, val, op, target):
        nonlocal pass_all
        ok = (val >= target) if op == ">=" else (val <= target)
        tag = "OK  " if ok else "FAIL"
        print(f"  {tag} {name:25s} {val:.4f} {op} {target}")
        if not ok:
            pass_all = False
    check("outward_frac",         m["outward_frac"],          ">=", TARGETS["outward_frac_min"])
    check("radial_growth_p90",    m["radial_growth_p90"],     ">=", TARGETS["radial_growth_p90_min"])
    check("direction_entropy",    m["direction_entropy"],     ">=", TARGETS["direction_entropy_min"])
    check("direction_entropy_max",m["direction_entropy"],     "<=", TARGETS["direction_entropy_max"])
    check("bbox_spread_peak",     m["bbox_spread_peak"],      ">=", TARGETS["bbox_spread_peak_min"])
    check("floater_count",        m["floater_count"],         "<=", TARGETS["floater_count_max"])
    check("base_frac",            m["base_frac"],             "<=", TARGETS["base_frac_max"])
    check("n_unique_fragments",   m["n_unique_fragments"],    ">=", TARGETS["n_unique_fragments_min"])
    check("anisotropy_ratio",     m["anisotropy_ratio"],      ">=", TARGETS["anisotropy_ratio_min"])
    check("radial_cv",            m["radial_cv"],             ">=", TARGETS["radial_cv_min"])
    print("=== Verdict ===")
    print("PASS_TARGETS" if pass_all else "NEED_MORE")
    return 0 if pass_all else 3


if __name__ == "__main__":
    sys.exit(main())
