#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pillow",
# ]
# ///
"""
Self-Sorting Map (SSM) Algorithm
Run with:
    uv run self_sorting_map.py
"""

from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    row: int
    col: int
    item: Any = None


class SelfSortingMap:
    """
    Self-Sorting Map (SSM).

    Parameters
    ----------
    grid_size : int
        N for an N×N grid. Must be a power of two and >= 8
        (so that N//4 >= 2 for the initial 4x4 block arrangement).
    distance_fn : callable
        delta(s, t) -> float
    data_mode : str
        'real'    - target = mean vector
        'nominal' - target = exact medoid of the full neighborhood (Eq. 6)
    max_iters : int
        Max swap passes per stage (L). Set to 4 as per the paper.
    seed : int | None
    """

    def __init__(
        self,
        grid_size: int,
        distance_fn: Callable,
        data_mode: str = "real",
        max_iters: int = 4,
        seed: int | None = 42,
    ):
        if grid_size & (grid_size - 1):
            raise ValueError("grid_size must be a power of two")
        if grid_size < 8:
            raise ValueError(
                "grid_size must be >= 8 (need at least a 4x4 block arrangement)"
            )
        self.N = grid_size
        self.delta = distance_fn
        self.data_mode = data_mode
        self.max_iters = max_iters
        self.rng = random.Random(seed)

        self.grid: list[list[Cell]] = [
            [Cell(r, c) for c in range(self.N)] for r in range(self.N)
        ]

    def fit(self, items: list[Any], *, verbose: bool = True) -> "SelfSortingMap":
        """
        Place N*N items on the grid and run SSM sorting.

        verbose
            If True, print per-block timings.
        """
        if len(items) != self.N * self.N:
            raise ValueError(f"Expected {self.N*self.N} items, got {len(items)}")

        shuffled = list(items)
        self.rng.shuffle(shuffled)
        idx = 0
        for r in range(self.N):
            for c in range(self.N):
                self.grid[r][c].item = shuffled[idx]
                idx += 1

        # Paper starts with a 4x4 block arrangement -> block_size = N // 4
        block_size = self.N // 4
        while block_size >= 1:
            t0 = time.perf_counter()
            self._run_stage(block_size)
            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"  block={block_size:3d}  {elapsed*1000:7.1f}ms")
            block_size //= 2

        return self

    def get_layout(self) -> list[list[Any]]:
        return [[self.grid[r][c].item for c in range(self.N)] for r in range(self.N)]

    # ------------------------------------------------------------------
    # Stage / grouping
    # ------------------------------------------------------------------

    def _run_stage(self, block_size: int) -> None:
        for _ in range(self.max_iters):
            improved = False
            for offset in (0, 1):
                if self._run_grouping(block_size, offset):
                    improved = True
            if not improved:
                break

    def _run_grouping(self, block_size: int, offset: int) -> bool:
        step = block_size * 2
        any_swap = False

        def get_starts(off: int) -> list[int]:
            starts: list[int] = []
            s = block_size * off
            while s + 2 * block_size <= self.N:
                starts.append(s)
                s += step
            return starts

        starts_r = get_starts(offset)
        starts_c = get_starts(offset)

        # 1. Compute targets for all active blocks first (prevents race conditions)
        group_targets = {}
        for r0 in starts_r:
            for c0 in starts_c:
                origins = [
                    (r0, c0),
                    (r0, c0 + block_size),
                    (r0 + block_size, c0),
                    (r0 + block_size, c0 + block_size),
                ]
                group_targets[(r0, c0)] = [
                    self._compute_target(br, bc, block_size, origins)
                    for (br, bc) in origins
                ]

        # 2. Swap items within quadruples using the generated targets
        for r0 in starts_r:
            for c0 in starts_c:
                origins = [
                    (r0, c0),
                    (r0, c0 + block_size),
                    (r0 + block_size, c0),
                    (r0 + block_size, c0 + block_size),
                ]
                targets = group_targets[(r0, c0)]

                for dr in range(block_size):
                    for dc in range(block_size):
                        cells = [self.grid[br + dr][bc + dc] for (br, bc) in origins]
                        if self._swap_quadruple(cells, targets):
                            any_swap = True

        return any_swap

    # ------------------------------------------------------------------
    # Target generation
    # ------------------------------------------------------------------

    def _compute_target(
        self,
        br: int,
        bc: int,
        block_size: int,
        group_origins: list[tuple[int, int]],
    ) -> Any:
        other = [(r, c) for (r, c) in group_origins if (r, c) != (br, bc)]
        centroid_r = sum(r for r, _ in other) / 3
        centroid_c = sum(c for _, c in other) / 3
        
        dir_r = int(np.sign(br - centroid_r))
        dir_c = int(np.sign(bc - centroid_c))

        # EXACT FAITHFUL REVISION: The 1-block shift.
        # This mirrors the 1D example in the paper exactly, keeping the window offset
        # by exactly 1 block away from the centroid of the paired grouping.
        start_rb = (br // block_size) - (2 if dir_r == -1 else 1)
        start_cb = (bc // block_size) - (2 if dir_c == -1 else 1)
        win_blocks = 4

        if self.data_mode == "real":
            block_means = []
            for d_r in range(win_blocks):
                for d_c in range(win_blocks):
                    rb = start_rb + d_r
                    cb = start_cb + d_c
                    
                    r_s = max(0, min(self.N, rb * block_size))
                    r_e = max(0, min(self.N, (rb + 1) * block_size))
                    c_s = max(0, min(self.N, cb * block_size))
                    c_e = max(0, min(self.N, (cb + 1) * block_size))
                    
                    # FAITHFUL REVISION: Equation (4) explicitly calculates the average 
                    # of block means, not the raw mean of all neighborhood cells.
                    if r_s < r_e and c_s < c_e:
                        b_items = [self.grid[r][c].item for r in range(r_s, r_e) for c in range(c_s, c_e)]
                        block_means.append(np.mean(b_items, axis=0))
            
            if not block_means:  # Boundary fallback safety
                return np.zeros_like(self.grid[0][0].item)
            return np.mean(block_means, axis=0)

        # FAITHFUL REVISION: Nominal mode uses exact mathematical definition from Equation 6.
        # Target is the item from the entire neighborhood Ω(B_i) that minimizes total dissimilarity.
        neighborhood_items = []
        for d_r in range(win_blocks):
            for d_c in range(win_blocks):
                rb = start_rb + d_r
                cb = start_cb + d_c
                
                r_s = max(0, min(self.N, rb * block_size))
                r_e = max(0, min(self.N, (rb + 1) * block_size))
                c_s = max(0, min(self.N, cb * block_size))
                c_e = max(0, min(self.N, (cb + 1) * block_size))
                
                if r_s < r_e and c_s < c_e:
                    neighborhood_items.extend(
                        [self.grid[r][c].item for r in range(r_s, r_e) for c in range(c_s, c_e)]
                    )
        
        if not neighborhood_items:
            return self.grid[br][bc].item
            
        target = min(
            neighborhood_items, 
            key=lambda u: sum(self.delta(u, v) for v in neighborhood_items)
        )
        return target

    # ------------------------------------------------------------------
    # Swapping
    # ------------------------------------------------------------------

    def _swap_quadruple(self, cells: list[Cell], targets: list[Any]) -> bool:
        items = [cell.item for cell in cells]

        def cost(perm):
            return sum(self.delta(items[perm[i]], targets[i]) for i in range(4))

        best_perm = (0, 1, 2, 3)
        best_cost = cost(best_perm)

        for perm in itertools.permutations(range(4)):
            c = cost(perm)
            if c < best_cost - 1e-12:
                best_cost = c
                best_perm = perm

        if best_perm == (0, 1, 2, 3):
            return False

        original = list(items)
        for i, src in enumerate(best_perm):
            cells[i].item = original[src]
        return True


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def generate_random_colors_numpy(nx=32, ny=32):
    """Return a (nx, ny, 3) uint8-range integer numpy array."""
    np.random.seed(3)
    return np.random.uniform(0, 255, size=(nx, ny, 3)).astype(int)


if __name__ == "__main__":
    from PIL import Image

    try:
        from dpq import distance_preservation_quality
        has_dpq = True
    except ImportError:
        has_dpq = False

    grid_size = 32
    nx = ny = grid_size
    colors = generate_random_colors_numpy(nx, ny)

    items = [colors[r, c] for r in range(nx) for c in range(ny)]

    ssm = SelfSortingMap(
        grid_size=grid_size,
        distance_fn=euclidean_distance,
        data_mode="real",
        max_iters=1000,
        seed=1,
    )

    # Same shuffle as fit(seed=1): raster for README / before–after figures
    rng = random.Random(1)
    shuffled = list(items)
    rng.shuffle(shuffled)
    initial_arr = np.zeros((nx, ny, 3), dtype=np.uint8)
    idx = 0
    for r in range(nx):
        for c in range(ny):
            initial_arr[r, c] = shuffled[idx]
            idx += 1
    Image.fromarray(initial_arr.astype(np.uint8)).save("ssm_rgb_initial.png")
    print("Saved -> ssm_rgb_initial.png")

    t0 = time.perf_counter()
    ssm.fit(items)
    elapsed = time.perf_counter() - t0
    print(f"SSM fit completed in {elapsed:.3f}s  ({elapsed*1000:.1f}ms)")

    layout = ssm.get_layout()
    x_sorted = np.array(
        [[layout[r][c] for c in range(ny)] for r in range(nx)], dtype=np.uint8
    )

    if has_dpq:
        dpq = distance_preservation_quality(x_sorted.astype(np.float64), p=16, wrap=False)
        print(f"Distance Preservation Quality: {dpq}")

    Image.fromarray(x_sorted.astype(np.uint8)).save("ssm_rgb.png")
    print("Saved -> ssm_rgb.png")