"""Lightweight geometry/mesh serialisation for the web 3-D viewport.

Produces compact position/index/line buffers (plain float/int lists) that the
front-end feeds straight into Three.js BufferGeometries.
"""
from __future__ import annotations

import numpy as np

from ..geometry.surface import BodySurface
from ..mesh.multiblock import MultiBlockMesh
from ..mesh.block import Face


def _grid_triangles(grid: np.ndarray, flip: bool) -> list[list[int]]:
    ni, nj, _ = grid.shape
    tris = []
    for i in range(ni - 1):
        for j in range(nj - 1):
            a = i * nj + j
            b = (i + 1) * nj + j
            c = (i + 1) * nj + (j + 1)
            d = i * nj + (j + 1)
            if flip:
                tris += [[a, c, b], [a, d, c]]
            else:
                tris += [[a, b, c], [a, c, d]]
    return tris


def _grid_lines(grid: np.ndarray, base: int, stride: int = 1) -> list[list[int]]:
    ni, nj, _ = grid.shape
    lines = []
    for i in range(0, ni, stride):
        for j in range(nj - 1):
            lines.append([base + i * nj + j, base + i * nj + j + 1])
    for j in range(0, nj, stride):
        for i in range(ni - 1):
            lines.append([base + i * nj + j, base + (i + 1) * nj + j])
    return lines


def surface_payload(surface: BodySurface) -> dict:
    """Triangulated upper+lower surface plus grid wireframe for the viewer."""
    upper = surface.upper
    lower = surface.lower
    nU = upper.shape[0] * upper.shape[1]

    pos = np.concatenate([upper.reshape(-1, 3), lower.reshape(-1, 3)], axis=0)
    tris = _grid_triangles(upper, flip=True)
    tris += [[a + nU, b + nU, c + nU] for a, b, c in _grid_triangles(lower, flip=False)]

    stride = max(1, surface.n_span // 30)
    lines = _grid_lines(upper, 0, stride) + _grid_lines(lower, nU, stride)

    le = surface.leading_edge
    lo, hi = surface.bbox()
    return {
        "positions": pos.astype(float).ravel().tolist(),
        "indices": np.asarray(tris, dtype=int).ravel().tolist(),
        "lines": np.asarray(lines, dtype=int).ravel().tolist(),
        "leading_edge": le.astype(float).ravel().tolist(),
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
    }


def mesh_payload(mesh: MultiBlockMesh) -> dict:
    """Legible volume-mesh wireframe for the viewer.

    Returns separate, cleanly separable layers so the front-end can render a
    readable picture instead of an overlapping cloud:

    * ``wall_indices``  -- solid body surface
    * ``wall_lines``    -- surface grid on the body
    * ``farfield_lines``-- the outer domain boundary (rings + streamwise spars),
      so the domain extent is clearly outlined
    * ``cut_lines``     -- a few radial cut planes (wall -> farfield) that reveal
      the interior clustering, including the near-wall boundary-layer bunching
    """
    positions: list[float] = []
    wall_tri: list[int] = []
    wall_lines: list[int] = []
    farfield_lines: list[int] = []
    cut_lines: list[int] = []

    for block in mesh.blocks:
        c = block.coords
        ni, nj, nk = block.shape
        base = len(positions) // 3
        positions.extend(c.reshape(-1, 3).astype(float).ravel().tolist())

        def idx(i, j, k):
            return base + (i * nj + j) * nk + k

        # --- solid body (wall, j = 0) + a light surface grid ---
        si = max(1, ni // 32)
        sk = max(1, nk // 32)
        for i in range(ni - 1):
            for k in range(nk - 1):
                a, b = idx(i, 0, k), idx(i + 1, 0, k)
                cc, d = idx(i + 1, 0, k + 1), idx(i, 0, k + 1)
                wall_tri += [a, b, cc, a, cc, d]
        for i in range(0, ni, si):
            for k in range(nk - 1):
                wall_lines += [idx(i, 0, k), idx(i, 0, k + 1)]
        for k in range(0, nk, sk):
            for i in range(ni - 1):
                wall_lines += [idx(i, 0, k), idx(i + 1, 0, k)]

        # --- outer domain boundary (farfield, j = nj-1): rings + spars ---
        jf = nj - 1
        for k in range(0, nk, sk):                       # cross-section rings
            for i in range(ni - 1):
                farfield_lines += [idx(i, jf, k), idx(i + 1, jf, k)]
        for i in range(0, ni, si):                       # streamwise spars
            for k in range(nk - 1):
                farfield_lines += [idx(i, jf, k), idx(i, jf, k + 1)]

        # --- radial cut planes at a few azimuths: wall -> farfield structure ---
        # A handful of constant-wrap "fans" show the wall-normal clustering (the
        # boundary layer bunches near j=0) along the whole body, without the
        # dense every-cell soup.
        n_fans = 6
        fan_i = sorted(set(int(round(f * (ni - 1))) for f in
                           [t / n_fans for t in range(n_fans + 1)]))
        sk2 = max(1, nk // 24)
        for i in fan_i:
            for k in range(0, nk, sk2):                  # radial lines (show BL bunching)
                for j in range(nj - 1):
                    cut_lines += [idx(i, j, k), idx(i, j + 1, k)]
            for j in range(0, nj, max(1, nj // 16)):     # streamwise lines on the fan
                for k in range(nk - 1):
                    cut_lines += [idx(i, j, k), idx(i, j, k + 1)]

    lo, hi = mesh.bounding_box()
    return {
        "positions": positions,
        "wall_indices": wall_tri,
        "wall_lines": wall_lines,
        "farfield_lines": farfield_lines,
        "cut_lines": cut_lines,
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
    }
