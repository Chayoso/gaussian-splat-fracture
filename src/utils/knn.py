"""Faiss-backed kNN and local distance helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _pad_row(
    idx_row: np.ndarray,
    dist_row: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if idx_row.shape[0] >= k:
        return idx_row[:k], dist_row[:k]
    if idx_row.shape[0] == 0:
        return (
            np.zeros((k,), dtype=np.int64),
            np.full((k,), np.inf, dtype=np.float32),
        )
    pad = k - idx_row.shape[0]
    return (
        np.pad(idx_row, (0, pad), mode="edge"),
        np.pad(dist_row, (0, pad), constant_values=np.inf),
    )


def _remove_self(
    idx: np.ndarray,
    dist: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_idx = np.empty((idx.shape[0], k), dtype=np.int64)
    out_dist = np.empty((idx.shape[0], k), dtype=np.float32)
    for row in range(idx.shape[0]):
        keep = idx[row] != row
        row_idx, row_dist = _pad_row(idx[row, keep], dist[row, keep], k)
        out_idx[row] = row_idx
        out_dist[row] = row_dist
    return out_idx, out_dist


_GPU_RES = None


def _gpu_resources():
    """Lazy-cache a single ``StandardGpuResources`` per process."""
    global _GPU_RES
    if _GPU_RES is None:
        import faiss  # type: ignore
        _GPU_RES = faiss.StandardGpuResources()
    return _GPU_RES


def knn_search(
    query: Tensor,
    database: Tensor,
    k: int,
    *,
    exclude_self: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return Euclidean distances and indices for k nearest database points.

    Validation and gravity runs assume the `diffmpm_v2.3.0` environment where
    Faiss is available; missing Faiss should fail loudly instead of silently
    changing the neighbor backend.
    """
    if query.ndim != 2 or database.ndim != 2:
        raise ValueError("query and database must be rank-2 tensors")
    if query.shape[1] != database.shape[1]:
        raise ValueError("query and database dimensions must match")

    device = query.device
    dtype = query.dtype
    db_count = int(database.shape[0])
    if db_count <= 0:
        raise ValueError("database must contain at least one vector")

    k = max(int(k), 1)
    usable_k = min(k, max(db_count - (1 if exclude_self else 0), 1))
    search_k = min(db_count, usable_k + (1 if exclude_self else 0))

    query_np = query.detach().float().cpu().contiguous().numpy()
    database_np = database.detach().float().cpu().contiguous().numpy()

    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Faiss is required for kNN search in this project environment") from exc

    # Use GPU faiss when available -- the search itself runs on the GPU
    # so kNN doesn't bottleneck the CPU thread between MPM kernels.
    # Falls back to CPU IndexFlatL2 if faiss-gpu is not present (e.g.
    # conda-forge faiss-cpu builds where get_num_gpus is unavailable).
    use_gpu = False
    try:
        if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0 \
                and hasattr(faiss, "StandardGpuResources"):
            use_gpu = True
    except Exception:
        use_gpu = False

    if use_gpu:
        # Cache resources per process so we don't allocate every call.
        res = _gpu_resources()
        index = faiss.GpuIndexFlatL2(res, database_np.shape[1])
        index.add(database_np)
        dist_sq, idx = index.search(query_np, search_k)
    else:
        index = faiss.IndexFlatL2(database_np.shape[1])
        index.add(database_np)
        dist_sq, idx = index.search(query_np, search_k)
    dist = np.sqrt(np.maximum(dist_sq, 0.0)).astype(np.float32, copy=False)
    idx = idx.astype(np.int64, copy=False)

    if exclude_self and query.shape[0] == database.shape[0]:
        idx, dist = _remove_self(idx, dist, usable_k)
    else:
        idx = idx[:, :usable_k]
        dist = dist[:, :usable_k]

    dist_t = torch.from_numpy(dist).to(device=device, dtype=dtype)
    idx_t = torch.from_numpy(idx).to(device=device, dtype=torch.long)
    return dist_t, idx_t


def pairwise_distances(query: Tensor, database: Tensor) -> Tensor:
    """Dense Euclidean distances via matrix products for small local scoring."""
    q_sq = (query * query).sum(dim=1, keepdim=True)
    db_sq = (database * database).sum(dim=1).unsqueeze(0)
    dist_sq = (q_sq + db_sq - 2.0 * (query @ database.T)).clamp(min=0.0)
    return torch.sqrt(dist_sq)
