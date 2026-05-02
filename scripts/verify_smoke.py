"""Quick verifier for smoke iterations.

Reads the last Houdini geo frame from a sim output dir and prints:
  - total particle count
  - label==0 (base) count and fraction
  - per-fragment size stats
  - z-velocity-like signal: max z-displacement among fragment particles between
    last two frames (proxy for whether bounces are happening)

Exit code 0 if base==0 AND fragments are clearly moving; else 1.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np


def _pairs(seq):
    return {seq[i]: seq[i + 1] for i in range(0, len(seq), 2)}


def _read_attr(entry):
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
    arr = arr.astype(np.int32 if "int" in storage else np.float32)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", type=Path)
    args = ap.parse_args()

    files = sorted(p for p in args.export_dir.iterdir()
                   if p.suffix in (".geo", ".gz"))
    if len(files) < 2:
        print("ERROR: need at least 2 frames")
        return 2

    first = read_geo(files[0])
    mid = read_geo(files[len(files) // 2])
    last = read_geo(files[-1])
    prev_last = read_geo(files[-2])

    fid_last = last.get("fragment_id")
    P_last = last.get("P")
    P_prev = prev_last.get("P")
    if fid_last is None or P_last is None or P_prev is None:
        print("ERROR: missing required attributes")
        return 2

    n = fid_last.shape[0]
    n_base = int((fid_last == 0).sum())
    n_frag = n - n_base

    P_last_3 = P_last.reshape(n, 3)
    P_prev_3 = P_prev.reshape(P_prev.shape[0], 3)
    common_n = min(P_last_3.shape[0], P_prev_3.shape[0])
    delta = P_last_3[:common_n] - P_prev_3[:common_n]
    fid_common = fid_last[:common_n].reshape(-1)

    # Fragment-only motion
    frag_mask = fid_common > 0
    base_mask = fid_common == 0

    # Mid-frame fragment count to check explosion peak
    fid_mid = mid.get("fragment_id")
    n_base_mid = int((fid_mid == 0).sum()) if fid_mid is not None else -1

    print(f"=== Smoke Verifier ===")
    print(f"frames: {len(files)}  first={files[0].name}  last={files[-1].name}")
    print(f"total particles: {n}")
    print(f"final base (fid==0): {n_base} ({100.0*n_base/n:.1f}%)")
    print(f"final fragment particles: {n_frag} ({100.0*n_frag/n:.1f}%)")
    print(f"mid-frame base: {n_base_mid}")
    print(f"unique fragment labels (final): {int(np.unique(fid_last[fid_last>0]).size)}")

    if frag_mask.any():
        z_delta = delta[frag_mask, 2]
        max_z_up = float(z_delta.max())
        max_z_down = float(z_delta.min())
        mean_speed = float(np.linalg.norm(delta[frag_mask], axis=1).mean())
        print(f"fragment dz (last 2 frames): up_max={max_z_up:.5f}  down_max={max_z_down:.5f}")
        print(f"fragment mean displacement / frame: {mean_speed:.5f}")
    else:
        max_z_up = 0.0
        mean_speed = 0.0
        print("no fragment particles in final frame")

    # Scan ALL frame pairs for the largest upward dz (proxy for bounce).
    # The bounce typically happens early after first floor contact, so
    # the final-frame dz under-reports peak bounce.
    max_up_global = 0.0
    max_up_idx = -1
    for i in range(1, len(files)):
        prev_g = read_geo(files[i - 1])
        cur_g = read_geo(files[i])
        if "P" not in prev_g or "P" not in cur_g or "fragment_id" not in cur_g:
            continue
        n_cur = cur_g["P"].shape[0] // 3 if cur_g["P"].ndim == 1 else cur_g["P"].shape[0]
        n_prev = prev_g["P"].shape[0] // 3 if prev_g["P"].ndim == 1 else prev_g["P"].shape[0]
        n_common = min(n_cur, n_prev)
        cur_P = cur_g["P"].reshape(-1, 3)[:n_common]
        prev_P = prev_g["P"].reshape(-1, 3)[:n_common]
        cur_fid = cur_g["fragment_id"][:n_common].reshape(-1)
        dz = cur_P[:, 2] - prev_P[:, 2]
        frag_dz = dz[cur_fid > 0]
        if frag_dz.size > 0:
            max_up_here = float(frag_dz.max())
            if max_up_here > max_up_global:
                max_up_global = max_up_here
                max_up_idx = i
    print(f"max upward dz across all frames: {max_up_global:.5f} at frame index {max_up_idx}")

    if base_mask.any():
        base_displacement = float(np.linalg.norm(delta[base_mask], axis=1).mean())
        print(f"base mean displacement / frame: {base_displacement:.5f}")
    else:
        base_displacement = 0.0

    # Verdict
    print()
    print("=== Verdict ===")
    issues = []
    if n_base > 0:
        issues.append(f"BASE_PERSISTS ({n_base} particles still labeled 0)")
    if max_up_global < 1e-3:
        issues.append(f"NO_VISIBLE_BOUNCE (peak upward dz = {max_up_global:.6f})")
    if mean_speed < 1e-4:
        issues.append(f"FRAGMENTS_FROZEN (mean speed = {mean_speed:.6f})")

    # Lateral scatter check: max XY displacement across all frame pairs.
    max_lateral = 0.0
    for i in range(1, len(files)):
        prev_g = read_geo(files[i - 1])
        cur_g = read_geo(files[i])
        if "P" not in prev_g or "P" not in cur_g or "fragment_id" not in cur_g:
            continue
        n_cur = cur_g["P"].shape[0] // 3 if cur_g["P"].ndim == 1 else cur_g["P"].shape[0]
        n_prev = prev_g["P"].shape[0] // 3 if prev_g["P"].ndim == 1 else prev_g["P"].shape[0]
        n_common = min(n_cur, n_prev)
        cur_P = cur_g["P"].reshape(-1, 3)[:n_common]
        prev_P = prev_g["P"].reshape(-1, 3)[:n_common]
        cur_fid = cur_g["fragment_id"][:n_common].reshape(-1)
        dxy = np.linalg.norm(cur_P[:, :2] - prev_P[:, :2], axis=1)
        frag_dxy = dxy[cur_fid > 0]
        if frag_dxy.size > 0:
            max_lateral = max(max_lateral, float(frag_dxy.max()))
    print(f"max lateral (XY) displacement / frame: {max_lateral:.5f}")

    # Also: spread of final fragment positions vs initial.
    fid_first = first.get("fragment_id")
    P_first = first.get("P")
    if P_first is not None and fid_last is not None:
        n_min = min(P_first.shape[0] // 3 if P_first.ndim == 1 else P_first.shape[0], n)
        first_P = P_first.reshape(-1, 3)[:n_min]
        last_P_n = P_last.reshape(-1, 3)[:n_min]
        bbox_first = first_P.max(axis=0) - first_P.min(axis=0)
        bbox_last = last_P_n.max(axis=0) - last_P_n.min(axis=0)
        spread_growth = float((bbox_last[:2] / bbox_first[:2].clip(min=1e-6)).max())
        print(f"XY bbox spread growth (last/first): {spread_growth:.2f}")
    else:
        spread_growth = 0.0

    if max_lateral < 5e-4:
        issues.append(f"NO_LATERAL_SCATTER (max XY dxy = {max_lateral:.6f})")
    if spread_growth < 1.3:
        issues.append(f"INSUFFICIENT_SPREAD (XY bbox grew {spread_growth:.2f}x)")

    if not issues:
        print("PASS - base eliminated AND fragments scattering laterally with bounce")
        return 0
    else:
        for issue in issues:
            print(f"  FAIL: {issue}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
