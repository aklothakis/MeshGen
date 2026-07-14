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


def mesh_payload(mesh: MultiBlockMesh, max_lines: int = 60000) -> dict:
    """Wall surface + block-edge / cross-section wireframe for the viewer."""
    positions: list[float] = []
    wall_tri: list[int] = []
    wall_lines: list[int] = []
    outer_lines: list[int] = []
    slice_lines: list[int] = []       # interior cross-section grids (volume)

    for block in mesh.blocks:
        c = block.coords
        ni, nj, nk = block.shape
        base = len(positions) // 3
        positions.extend(c.reshape(-1, 3).astype(float).ravel().tolist())

        def idx(i, j, k):
            return base + (i * nj + j) * nk + k

        # Wall (jmin) surface triangles + grid lines.
        si = max(1, ni // 40)
        sk = max(1, nk // 40)
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

        # Interior volume cross-sections: full (wrap x wall-normal) grid at a few
        # streamwise stations, so the mesh reads as a 3-D volume, not a surface.
        sw = max(1, ni // 60)
        sn = max(1, nj // 24)
        k_slices = sorted(set([0, nk - 1] + [round(f * (nk - 1)) for f in (0.5,)]))
        for k in k_slices:
            for i in range(0, ni, sw):
                for j in range(nj - 1):
                    slice_lines += [idx(i, j, k), idx(i, j + 1, k)]
            for j in range(0, nj, sn):
                for i in range(ni - 1):
                    slice_lines += [idx(i, j, k), idx(i + 1, j, k)]

        # Block outline edges (the 12 edges of the logical box).
        corners = [(0, 0, 0), (ni - 1, 0, 0), (ni - 1, nj - 1, 0), (0, nj - 1, 0),
                   (0, 0, nk - 1), (ni - 1, 0, nk - 1), (ni - 1, nj - 1, nk - 1), (0, nj - 1, nk - 1)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        cidx = [idx(*cc) for cc in corners]
        for a, b in edges:
            outer_lines += [cidx[a], cidx[b]]

    lo, hi = mesh.bounding_box()
    return {
        "positions": positions,
        "wall_indices": wall_tri,
        "wall_lines": wall_lines,
        "slice_lines": slice_lines,
        "block_edges": outer_lines,
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
    }
