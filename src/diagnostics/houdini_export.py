"""Houdini-native ``.geo`` (JSON) writer for per-frame Gaussian splat state.

Produces files Houdini reads directly via the File SOP without any
external library (no ``hou``, no ``partio``).  The format is the
documented Houdini ASCII geo JSON:

    https://www.sidefx.com/docs/houdini/io/formats/geo.html

Per-Gaussian point attributes written:

| name          | size | semantics                                  |
|---------------|------|--------------------------------------------|
| ``P``         |  3   | world-space position                       |
| ``Cd``        |  3   | base RGB (sigmoided DC SH * material LUT)  |
| ``Alpha``     |  1   | sigmoided opacity                          |
| ``scale``     |  3   | per-axis Gaussian sigmas (exp of log-scale)|
| ``pscale``    |  1   | mean sigma -- isotropic fallback           |
| ``orient``    |  4   | Houdini quaternion ``(i, j, k, s)``        |
| ``fragment_id`` | 1  | int physical fragment label                |
| ``damage``    |  1   | AT2 c field value                          |
| ``N``         |  3   | crack normal (zero where undefined)        |

Houdini convention for ``orient`` is ``(i, j, k, s)`` (xyz then w),
while 3DGS stores quaternions in ``(w, x, y, z)`` order.  We rotate
into Houdini convention at write time.

Output extension: ``.geo`` (uncompressed JSON) or ``.geo.gz``
(gzipped JSON, also natively Houdini-readable).  Houdini's File SOP
recognizes both.  After the paper is done, batch-convert to
``.bgeo.sc`` inside Houdini if disk space matters.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _numeric_attr(name: str, values: np.ndarray, default: float = 0.0,
                  attr_kind: Optional[str] = None) -> list:
    """Build one Houdini point-attribute entry.

    ``values`` shape must be (N,) for a scalar attribute or (N, K) for
    a K-tuple attribute (K in {2, 3, 4}).
    """
    arr = np.asarray(values)
    if arr.ndim == 1:
        size = 1
        tuples = arr.astype(np.float32).tolist()
    elif arr.ndim == 2:
        size = int(arr.shape[1])
        tuples = arr.astype(np.float32).tolist()
    else:
        raise ValueError(f"unsupported attribute shape {arr.shape}")

    options: Dict[str, object] = {}
    if attr_kind:
        options["type"] = {"type": "string", "value": attr_kind}

    if size == 1:
        # Houdini stores scalar arrays in a flat list under "arrays".
        flat = arr.astype(np.float32).reshape(-1).tolist()
        values_block = [
            "size", 1,
            "storage", "fpreal32",
            "arrays", [flat],
        ]
    else:
        values_block = [
            "size", size,
            "storage", "fpreal32",
            "tuples", tuples,
        ]

    return [
        [
            "scope", "public",
            "type", "numeric",
            "name", name,
            "options", options,
        ],
        [
            "size", size,
            "storage", "fpreal32",
            "defaults", [
                "size", 1,
                "storage", "fpreal64",
                "values", [float(default)],
            ],
            "values", values_block,
        ],
    ]


def _int_attr(name: str, values: np.ndarray, default: int = 0) -> list:
    arr = np.asarray(values).astype(np.int32).reshape(-1).tolist()
    return [
        [
            "scope", "public",
            "type", "numeric",
            "name", name,
            "options", {},
        ],
        [
            "size", 1,
            "storage", "int32",
            "defaults", [
                "size", 1,
                "storage", "int32",
                "values", [int(default)],
            ],
            "values", [
                "size", 1,
                "storage", "int32",
                "arrays", [arr],
            ],
        ],
    ]


def _quat_wxyz_to_houdini(q_wxyz: np.ndarray) -> np.ndarray:
    """3DGS stores quaternions as (w, x, y, z); Houdini expects
    (i, j, k, s) i.e. (x, y, z, w)."""
    q = np.asarray(q_wxyz, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"expected (N, 4) quaternion, got {q.shape}")
    return q[:, [1, 2, 3, 0]]


def write_houdini_geo(
    path: Path,
    *,
    positions: np.ndarray,         # (N, 3)
    color: Optional[np.ndarray] = None,    # (N, 3) RGB in [0, 1]
    alpha: Optional[np.ndarray] = None,    # (N,)
    scale_xyz: Optional[np.ndarray] = None,    # (N, 3) per-axis sigma
    pscale: Optional[np.ndarray] = None,   # (N,) isotropic fallback
    orient_wxyz: Optional[np.ndarray] = None,  # (N, 4) wxyz
    fragment_id: Optional[np.ndarray] = None,  # (N,)
    damage: Optional[np.ndarray] = None,   # (N,)
    crack_normal: Optional[np.ndarray] = None,  # (N, 3)
    extra_scalars: Optional[Dict[str, np.ndarray]] = None,
    compress: bool = True,
) -> Path:
    """Write a Houdini-readable ``.geo`` (JSON) file.

    Set ``compress=True`` (default) to gzip and write ``<path>.gz``.
    """
    path = Path(path)
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {positions.shape}")
    n = int(positions.shape[0])

    point_attribs: list = [_numeric_attr("P", positions, attr_kind="point")]

    if color is not None:
        col = np.asarray(color, dtype=np.float32).reshape(n, 3)
        point_attribs.append(_numeric_attr("Cd", col, default=1.0,
                                          attr_kind="color"))
    if alpha is not None:
        a = np.asarray(alpha, dtype=np.float32).reshape(n)
        point_attribs.append(_numeric_attr("Alpha", a, default=1.0))
    if scale_xyz is not None:
        s = np.asarray(scale_xyz, dtype=np.float32).reshape(n, 3)
        point_attribs.append(_numeric_attr("scale", s, default=1.0,
                                          attr_kind="vector"))
    if pscale is not None:
        ps = np.asarray(pscale, dtype=np.float32).reshape(n)
        point_attribs.append(_numeric_attr("pscale", ps, default=1.0))
    if orient_wxyz is not None:
        q = _quat_wxyz_to_houdini(np.asarray(orient_wxyz))
        point_attribs.append(_numeric_attr("orient", q, default=0.0,
                                          attr_kind="quaternion"))
    if fragment_id is not None:
        fid = np.asarray(fragment_id).reshape(n).astype(np.int32)
        point_attribs.append(_int_attr("fragment_id", fid, default=0))
    if damage is not None:
        d = np.asarray(damage, dtype=np.float32).reshape(n)
        point_attribs.append(_numeric_attr("damage", d, default=0.0))
    if crack_normal is not None:
        nrm = np.asarray(crack_normal, dtype=np.float32).reshape(n, 3)
        point_attribs.append(_numeric_attr("N", nrm, default=0.0,
                                          attr_kind="normal"))

    if extra_scalars:
        for name, vals in extra_scalars.items():
            arr = np.asarray(vals, dtype=np.float32).reshape(n)
            point_attribs.append(_numeric_attr(name, arr, default=0.0))

    geo: list = [
        "fileversion", "20.0.751",
        "hasindex", False,
        "pointcount", n,
        "vertexcount", 0,
        "primitivecount", 0,
        "info", {
            "software": "gaussian_phase_field",
            "exporter": "houdini_export.write_houdini_geo",
        },
        "topology", ["pointref", ["indices", []]],
        "attributes", ["pointattributes", point_attribs],
        "primitives", [],
        "pointgroups", [],
        "primitivegroups", [],
    ]

    payload = json.dumps(geo, separators=(",", ":"))
    if compress:
        out_path = path.with_suffix(path.suffix + ".gz") if path.suffix == ".geo" else path
        if not out_path.name.endswith(".gz"):
            out_path = Path(str(out_path) + ".gz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            f.write(payload)
    else:
        out_path = path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    return out_path


def export_simulator_state(
    out_path: Path,
    *,
    simulator,
    crack_normal: Optional[np.ndarray] = None,
    compress: bool = True,
) -> Optional[Path]:
    """Convenience wrapper: pull all per-Gaussian attributes from a live
    simulator and write a Houdini ``.geo`` snapshot.  Returns the actual
    output path, or ``None`` if the simulator is not yet initialized.
    """
    import torch

    gaussians = getattr(simulator, "gaussians", None)
    if gaussians is None:
        return None
    xyz = gaussians._xyz.data.detach()
    n = int(xyz.shape[0])
    if n <= 0:
        return None

    pos_np = xyz.cpu().numpy()

    dc = gaussians._features_dc.data.detach()
    if dc.ndim == 3 and dc.shape[1] == 1 and dc.shape[2] == 3:
        col_raw = dc[:, 0, :].cpu().numpy()
    else:
        col_raw = dc.reshape(n, -1)[:, :3].cpu().numpy()
    color_np = (0.5 + col_raw * 0.28209479).clip(0.0, 1.0)  # SH DC -> RGB approx

    op = gaussians._opacity.data.detach()
    alpha_np = torch.sigmoid(op).reshape(n).cpu().numpy()

    sc = gaussians._scaling.data.detach()
    if sc.ndim == 2 and sc.shape[1] == 3:
        scale_np = torch.exp(sc).cpu().numpy()
    else:
        scale_np = np.tile(np.exp(sc.reshape(n, -1).mean(dim=1, keepdim=True).cpu().numpy()), (1, 3))
    pscale_np = scale_np.mean(axis=1)

    rot = gaussians._rotation.data.detach()
    rot = rot / rot.norm(dim=1, keepdim=True).clamp(min=1e-8)
    orient_np = rot.cpu().numpy()

    # Per-Gaussian fragment / damage / normal attributes.  The simulator's
    # _physical_fragment_labels is per-MPM-particle; we map it to surface
    # particles (one-to-one with the original Gaussians) via _surface_indices.
    # Densification can grow Gaussian count past N_surface, so we always
    # truncate-or-pad to the current Gaussian count instead of refusing the
    # assignment when sizes diverge.
    physical_labels = getattr(simulator, "_physical_fragment_labels", None)
    surface_indices = getattr(simulator, "_surface_indices", None)
    # Initialize fid to -1 ("no mapping") so densified Gaussian slots
    # without a corresponding surface particle aren't conflated with
    # genuine label==0 base body particles.  Any fid <= 0 is hidden by
    # the viewer's B-key, so phantom slots stay correctly hidden.  We
    # write -1 (rather than 0) so downstream code that distinguishes
    # base from phantom can do so on the sign of fid.
    fid_np = np.full(n, -1, dtype=np.int32)
    if (physical_labels is not None
            and surface_indices is not None
            and physical_labels.numel() > 0):
        try:
            sid = physical_labels[surface_indices].cpu().numpy().astype(np.int32)
            m = min(int(sid.shape[0]), n)
            fid_np[:m] = sid[:m]
        except Exception as exc:
            print(f"[houdini_export] fragment_id alignment skipped: {exc}")

    fracture_field = getattr(simulator, "fracture_field", None)
    damage_np = np.zeros(n, dtype=np.float32)
    if fracture_field is not None and getattr(fracture_field, "c", None) is not None:
        c = fracture_field.c
        m = min(int(c.shape[0]), n)
        damage_np[:m] = c[:m].cpu().numpy().astype(np.float32)

    if crack_normal is None and fracture_field is not None and getattr(fracture_field, "n", None) is not None:
        nfield = fracture_field.n
        m = min(int(nfield.shape[0]), n)
        if m > 0:
            crack_normal = np.zeros((n, 3), dtype=np.float32)
            crack_normal[:m] = nfield[:m].cpu().numpy().astype(np.float32)

    return write_houdini_geo(
        out_path,
        positions=pos_np,
        color=color_np,
        alpha=alpha_np,
        scale_xyz=scale_np,
        pscale=pscale_np,
        orient_wxyz=orient_np,
        fragment_id=fid_np,
        damage=damage_np,
        crack_normal=crack_normal,
        compress=compress,
    )
