"""Multiblock structured-mesh container with quality diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .block import BCType, Face, StructuredBlock

__all__ = ["MultiBlockMesh", "MeshQuality"]


@dataclass
class MeshQuality:
    n_blocks: int
    n_cells: int
    n_nodes: int
    min_cell_volume: float
    n_negative_cells: int
    min_orthogonality_deg: float
    max_aspect_ratio: float
    max_expansion_ratio: float

    def is_valid(self) -> bool:
        return self.n_negative_cells == 0 and self.min_cell_volume > 0.0

    def as_dict(self) -> dict:
        return {
            "n_blocks": self.n_blocks,
            "n_cells": self.n_cells,
            "n_nodes": self.n_nodes,
            "min_cell_volume": self.min_cell_volume,
            "n_negative_cells": self.n_negative_cells,
            "min_orthogonality_deg": self.min_orthogonality_deg,
            "max_aspect_ratio": self.max_aspect_ratio,
            "max_expansion_ratio": self.max_expansion_ratio,
            "valid": self.is_valid(),
        }


@dataclass
class MultiBlockMesh:
    blocks: list[StructuredBlock] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(self, block: StructuredBlock) -> None:
        self.blocks.append(block)

    @property
    def n_cells(self) -> int:
        return sum(b.n_cells for b in self.blocks)

    @property
    def n_nodes(self) -> int:
        return sum(b.n_nodes for b in self.blocks)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for b in self.blocks:
            pts = b.coords.reshape(-1, 3)
            lo = np.minimum(lo, pts.min(axis=0))
            hi = np.maximum(hi, pts.max(axis=0))
        return lo, hi

    # ---- quality ------------------------------------------------------
    def quality(self) -> MeshQuality:
        min_vol = np.inf
        n_neg = 0
        min_ortho = 180.0
        max_ar = 0.0
        max_er = 0.0
        for b in self.blocks:
            vols = b.cell_volumes()
            min_vol = min(min_vol, float(vols.min()))
            n_neg += int((vols <= 0.0).sum())
            o, ar, er = _block_quality(b.coords)
            min_ortho = min(min_ortho, o)
            max_ar = max(max_ar, ar)
            max_er = max(max_er, er)
        return MeshQuality(
            n_blocks=len(self.blocks),
            n_cells=self.n_cells,
            n_nodes=self.n_nodes,
            min_cell_volume=float(min_vol),
            n_negative_cells=int(n_neg),
            min_orthogonality_deg=float(min_ortho),
            max_aspect_ratio=float(max_ar),
            max_expansion_ratio=float(max_er),
        )


def _block_quality(coords: np.ndarray) -> tuple[float, float, float]:
    """Return ``(min_orthogonality_deg, max_aspect_ratio, max_expansion_ratio)``.

    Orthogonality is the minimum angle between the three grid-line directions
    at each node; aspect ratio and expansion ratio use edge lengths along each
    logical direction.
    """
    di = np.diff(coords, axis=0)
    dj = np.diff(coords, axis=1)
    dk = np.diff(coords, axis=2)

    li = np.linalg.norm(di, axis=-1)
    lj = np.linalg.norm(dj, axis=-1)
    lk = np.linalg.norm(dk, axis=-1)

    # Aspect ratio: use interior overlap of the three edge-length fields.
    ni, nj, nk = coords.shape[:3]
    a = li[:, : nj - 1, : nk - 1]
    b = lj[: ni - 1, :, : nk - 1]
    c = lk[: ni - 1, : nj - 1, :]
    emax = np.maximum(np.maximum(a, b), c)
    emin = np.minimum(np.minimum(a, b), c)
    max_ar = float((emax / np.maximum(emin, 1e-30)).max())

    # Expansion ratio: consecutive spacings along each direction.
    def expansion(l: np.ndarray, axis: int) -> float:
        if l.shape[axis] < 2:
            return 1.0
        s1 = np.take(l, range(l.shape[axis] - 1), axis=axis)
        s2 = np.take(l, range(1, l.shape[axis]), axis=axis)
        r = np.maximum(s1, s2) / np.maximum(np.minimum(s1, s2), 1e-30)
        return float(r.max())

    max_er = max(expansion(li, 0), expansion(lj, 1), expansion(lk, 2))

    # Orthogonality: angle between edge directions at shared interior nodes.
    ui = di[:, : nj - 1, : nk - 1] / np.maximum(a[..., None], 1e-30)
    uj = dj[: ni - 1, :, : nk - 1] / np.maximum(b[..., None], 1e-30)
    uk = dk[: ni - 1, : nj - 1, :] / np.maximum(c[..., None], 1e-30)

    def angle(u, v):
        d = np.abs(np.einsum("...i,...i->...", u, v)).clip(0, 1)
        return np.degrees(np.arccos(d))

    ang = np.minimum(np.minimum(angle(ui, uj), angle(uj, uk)), angle(ui, uk))
    # 90 deg is perfect; report the worst deviation as an angle.
    min_ortho = float(90.0 - (90.0 - ang).max()) if ang.size else 90.0
    return min_ortho, max_ar, max_er
