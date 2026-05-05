"""Material Consistency Score (MCS) — quantitative ablation metric.

For the dual-channel claim ("material physics dominates over morphology
style under conflicting prompts") we need a number that says, given a
prompt that asks rubber to behave like glass, *did the rendered
animation actually behave like rubber (correct) or like glass
(incorrect)?*

We compute MCS as a fragmentation-statistic distance between the
observed run output and the *family-expected* range:

    family            n_frags expected (15K particles)   bbox_spread
    -------           --------------------------------   -----------
    sharp_brittle     50 - 200                           2.0 - 3.0
    brittle_moderate  20 - 80                            1.5 - 2.5
    rough_quasi      8 - 35                              1.0 - 2.0
    diffuse_damage    1 - 5  (essentially intact)        0.8 - 1.4
    neutral_reference 1 - 3  (no fracture)               0.8 - 1.2

MCS = exp(-||obs - centre|| / span) ∈ [0, 1].

Higher = more consistent with the prompt's *material* family (not
its *style* request).  Under our adversarial prompt
"rubber shattering into glass-like shards", correct behaviour is
to land in the diffuse_damage envelope (1-5 frags); the no-clamp
ablation lands in the sharp_brittle envelope (50+ frags) -> low MCS,
showing the family clamp is what enforces material physics.

Usage:
    python scripts/material_consistency_score.py \
        --houdini-dir output/run_30k_ablation_rubber_glass_full_z0.50_v25e/houdini_export \
        --target-family diffuse_damage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.view_houdini_geo import read_houdini_geo, _list_frames  # noqa: E402


# Per-family expected envelope at 15K-30K particles, single drop impact.
# (centre_n_frags, span_n_frags, centre_bbox, span_bbox)
FAMILY_ENVELOPE: Dict[str, Tuple[float, float, float, float]] = {
    "sharp_brittle":      (95.0, 60.0, 2.5, 0.6),
    "brittle_moderate":   (35.0, 25.0, 2.0, 0.5),
    "rough_quasi_brittle": (18.0, 12.0, 1.5, 0.5),
    "diffuse_damage":      (3.0,  3.0, 1.1, 0.3),
    "neutral_reference":   (1.5,  1.5, 1.0, 0.2),
}


def _last_frame_metrics(houdini_dir: Path) -> Tuple[int, float]:
    frames = _list_frames(houdini_dir)
    if not frames:
        raise FileNotFoundError(f"no frames in {houdini_dir}")
    rec = read_houdini_geo(frames[-1])
    fid = rec.get("fragment_id")
    P = rec["P"]
    n_frag = int(np.unique(fid.reshape(-1)).size) if fid is not None else 1
    bbox_diag = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    return n_frag, bbox_diag


def material_consistency_score(
    n_frag: int,
    bbox_diag: float,
    target_family: str,
) -> float:
    """Distance-from-envelope MCS in [0, 1]; 1 = perfect match."""
    if target_family not in FAMILY_ENVELOPE:
        raise KeyError(f"unknown family: {target_family}")
    cn, sn, cb, sb = FAMILY_ENVELOPE[target_family]
    z_n = (n_frag - cn) / max(sn, 1e-6)
    z_b = (bbox_diag - cb) / max(sb, 1e-6)
    distance = float(np.sqrt(z_n * z_n + z_b * z_b))
    return float(np.exp(-distance))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--houdini-dir", type=Path, required=True,
                    help="Path to a run's houdini_export directory.")
    ap.add_argument("--target-family", type=str, required=True,
                    choices=sorted(FAMILY_ENVELOPE.keys()),
                    help="Material family expected from prompt's material "
                         "(NOT from prompt's style request).")
    ap.add_argument("--json", action="store_true",
                    help="Emit a single-line JSON record.")
    args = ap.parse_args()

    n_frag, bbox = _last_frame_metrics(args.houdini_dir)
    mcs = material_consistency_score(n_frag, bbox, args.target_family)
    cn, sn, cb, sb = FAMILY_ENVELOPE[args.target_family]

    if args.json:
        print(json.dumps({
            "houdini_dir": str(args.houdini_dir),
            "target_family": args.target_family,
            "n_frag": n_frag,
            "bbox_diag": bbox,
            "expected_n_frag_centre": cn,
            "expected_bbox_centre": cb,
            "mcs": mcs,
        }))
    else:
        print(f"  run            : {args.houdini_dir}")
        print(f"  target family  : {args.target_family}")
        print(f"  observed       : n_frag={n_frag}  bbox={bbox:.2f}")
        print(f"  expected       : n_frag~{cn:.0f}±{sn:.0f}  bbox~{cb:.2f}±{sb:.2f}")
        print(f"  MCS            : {mcs:.3f}  (1=perfect, 0=far)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
