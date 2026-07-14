"""Assemble a MultiBlockMesh into a watertight unstructured representation.

Structured blocks are converted to hexahedra; coincident nodes (block
interfaces and the O-grid seam) are merged by spatial hashing so the result is
watertight.  Physical boundary faces are collected per marker for the solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mesh.block import BCType, Face, StructuredBlock
from ..mesh.multiblock import MultiBlockMesh

__all__ = ["UnstructuredMesh", "assemble"]

# Faces that become tagged solver boundaries (INTERFACE is internal).
_PHYSICAL_BC = {
    BCType.WALL,
    BCType.FARFIELD,
    BCType.SYMMETRY,
    BCType.INFLOW,
    BCType.OUTFLOW,
}


@dataclass
class UnstructuredMesh:
    points: np.ndarray                              # (n_pts, 3)
    hexes: np.ndarray                               # (n_hex, 8) int
    markers: dict[str, np.ndarray] = field(default_factory=dict)  # tag -> (m, 4)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def n_cells(self) -> int:
        return self.hexes.shape[0]


def _block_hexes(block: StructuredBlock, base: int) -> np.ndarray:
    """Local hex connectivity (VTK ordering) offset by ``base`` node index."""
    ni, nj, nk = block.shape

    def idx(i, j, k):
        return base + (i * nj + j) * nk + k

    i, j, k = np.meshgrid(
        np.arange(ni - 1), np.arange(nj - 1), np.arange(nk - 1), indexing="ij"
    )
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    hexes = np.stack(
        [
            idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
            idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1),
        ],
        axis=1,
    )
    return hexes


def _face_quads(block: StructuredBlock, face: Face, base: int) -> np.ndarray:
    """Boundary quad connectivity for one logical face (outward-ordered)."""
    ni, nj, nk = block.shape

    def idx(i, j, k):
        return base + (i * nj + j) * nk + k

    quads = []
    if face in (Face.IMIN, Face.IMAX):
        i = 0 if face == Face.IMIN else ni - 1
        for j in range(nj - 1):
            for k in range(nk - 1):
                quads.append([idx(i, j, k), idx(i, j + 1, k),
                              idx(i, j + 1, k + 1), idx(i, j, k + 1)])
    elif face in (Face.JMIN, Face.JMAX):
        j = 0 if face == Face.JMIN else nj - 1
        for i in range(ni - 1):
            for k in range(nk - 1):
                quads.append([idx(i, j, k), idx(i, j, k + 1),
                              idx(i + 1, j, k + 1), idx(i + 1, j, k)])
    else:  # KMIN / KMAX
        k = 0 if face == Face.KMIN else nk - 1
        for i in range(ni - 1):
            for j in range(nj - 1):
                quads.append([idx(i, j, k), idx(i + 1, j, k),
                              idx(i + 1, j + 1, k), idx(i, j + 1, k)])
    return np.asarray(quads, dtype=np.int64).reshape(-1, 4)


def assemble(mesh: MultiBlockMesh, merge_tol: float | None = None) -> UnstructuredMesh:
    """Build a watertight unstructured mesh with per-marker boundary faces."""
    all_pts = []
    all_hexes = []
    marker_quads: dict[str, list] = {}
    base = 0
    for block in mesh.blocks:
        pts = block.coords.reshape(-1, 3)
        all_pts.append(pts)
        all_hexes.append(_block_hexes(block, base))
        for face, bc in block.bc.items():
            if bc in _PHYSICAL_BC:
                q = _face_quads(block, face, base)
                if q.size:
                    marker_quads.setdefault(bc.value, []).append(q)
        base += pts.shape[0]

    points = np.concatenate(all_pts, axis=0)
    hexes = np.concatenate(all_hexes, axis=0)

    # Merge coincident nodes (interfaces + seam) so the mesh is watertight.
    if merge_tol is None:
        diag = np.linalg.norm(points.max(0) - points.min(0))
        merge_tol = max(diag * 1e-9, 1e-12)
    remap, unique_pts = _merge_nodes(points, merge_tol)
    hexes = remap[hexes]
    markers = {
        tag: remap[np.concatenate(qs, axis=0)] for tag, qs in marker_quads.items()
    }
    return UnstructuredMesh(points=unique_pts, hexes=hexes, markers=markers)


def _merge_nodes(points: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(remap, unique_points)`` merging nodes within ``tol``."""
    keys = np.round(points / tol).astype(np.int64)
    # np.unique returns (unique, index, inverse) in that fixed order.
    _, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.ravel()
    # Order unique points by first appearance for stable output.
    order = np.argsort(first)
    new_id = np.empty(order.shape[0], dtype=np.int64)
    new_id[order] = np.arange(order.shape[0])
    remap = new_id[inverse]
    unique_pts = points[first[order]]
    return remap, unique_pts
