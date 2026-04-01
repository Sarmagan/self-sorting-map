"""
Distance Preservation Quality (DPQ) for 2-D grid layouts.

Same definition as LAS / rgb_las: compares spatially ordered HD distances
to the ideal HD-only ordering.
"""

from __future__ import annotations

import numpy as np


def _squared_l2_distance_rows(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Squared L2 distance matrix between rows of *p* (N,D) and *q* (M,D); shape (N, M)."""
    ps = np.sum(p * p, axis=-1, keepdims=True)
    qs = np.sum(q * q, axis=-1, keepdims=True)
    return np.clip(ps - 2 * np.matmul(p, q.T) + qs.T, 0, np.inf)


def _compute_spatial_distances_wrapped(grid_shape: tuple[int, int]) -> np.ndarray:
    n_x, n_y = grid_shape
    wrap1 = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, n_y],
        [0, n_y],
        [n_x, 0],
        [n_x, 0],
        [n_x, n_y],
    ]
    wrap2 = [
        [0, n_y],
        [n_x, 0],
        [n_x, n_y],
        [0, 0],
        [n_x, 0],
        [0, 0],
        [0, n_y],
        [0, 0],
    ]
    a, b = np.indices(grid_shape)
    mat_flat = np.concatenate(
        [np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1
    ).reshape(-1, 2)

    d = _squared_l2_distance_rows(mat_flat, mat_flat)
    for w1, w2 in zip(wrap1, wrap2):
        d = np.minimum(
            d, _squared_l2_distance_rows(mat_flat + w1, mat_flat + w2)
        )
    return d


def _compute_spatial_distances_non_wrapped(grid_shape: tuple[int, int]) -> np.ndarray:
    a, b = np.indices(grid_shape)
    mat_flat = np.concatenate(
        [np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1
    ).reshape(-1, 2)
    return _squared_l2_distance_rows(mat_flat, mat_flat)


def compute_spatial_distances_for_grid(
    grid_shape: tuple[int, int], wrap: bool
) -> np.ndarray:
    """Matrix of squared spatial distances for a 2-D grid (flat index order)."""
    if wrap:
        return _compute_spatial_distances_wrapped(grid_shape)
    return _compute_spatial_distances_non_wrapped(grid_shape)


def _sort_hddists_by_2d_dists(
    hd_dists: np.ndarray, ld_dists: np.ndarray
) -> np.ndarray:
    """Sort each row of *hd_dists* by ascending *ld_dists*, tie-break by HD value."""
    max_hd = np.max(hd_dists) * 1.0001
    combined = hd_dists / max_hd + ld_dists
    combined_sorted = np.sort(combined, axis=1)
    return np.fmod(combined_sorted, 1) * max_hd


def _get_distance_preservation_gain(
    sorted_d_mat: np.ndarray, d_mean: float
) -> np.ndarray:
    nums = np.arange(1, len(sorted_d_mat))
    cumsum = np.cumsum(sorted_d_mat[:, 1:], axis=1)
    d_k = (cumsum / nums).mean(axis=0)
    return np.clip((d_mean - d_k) / d_mean, 0, np.inf)


def distance_preservation_quality(
    sorted_X: np.ndarray, p: float = 2.0, wrap: bool = False
) -> float:
    """
    Distance Preservation Quality DPQ_p(S) in [0, 1].

    Measures how well 2-D spatial neighbourhoods reflect high-dimensional
    distance structure; 1.0 is optimal.

    Parameters
    ----------
    sorted_X : np.ndarray
        Sorted grid of shape (H, W, D).
    p : float
        Norm order used to aggregate per-k gains (default 2).
    wrap : bool
        If True, assume toroidal grid boundaries.
    """
    grid_shape = sorted_X.shape[:-1]
    n = int(np.prod(grid_shape))
    flat_x = sorted_X.reshape(n, -1)

    dists_hd = np.sqrt(_squared_l2_distance_rows(flat_x, flat_x))
    sorted_d = np.sort(dists_hd, axis=1)
    mean_d = sorted_d[:, 1:].mean()

    dists_spatial = compute_spatial_distances_for_grid(grid_shape, wrap)
    sorted_hd_by_2d = _sort_hddists_by_2d_dists(dists_hd, dists_spatial)

    delta_dp_k_2d = _get_distance_preservation_gain(sorted_hd_by_2d, mean_d)
    delta_dp_k_hd = _get_distance_preservation_gain(sorted_d, mean_d)

    return float(
        np.linalg.norm(delta_dp_k_2d, ord=p)
        / np.linalg.norm(delta_dp_k_hd, ord=p)
    )
