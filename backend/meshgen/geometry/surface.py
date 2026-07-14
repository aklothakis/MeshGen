"""Shared body-surface representation consumed by the mesher.

Both the parametric waverider generator and the STEP importer produce a
:class:`BodySurface`: an upper (freestream) and a lower (compression) patch as
structured ``(n_span, n_stream, 3)`` grids sharing a leading-edge row.  This is
the single interface the domain/mesh stages depend on, so parametric and
imported geometry travel identical downstream paths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BodySurface"]


@dataclass
class BodySurface:
    upper: np.ndarray              # (n_span, n_stream, 3)
    lower: np.ndarray              # (n_span, n_stream, 3)
    source: str = "parametric"     # "parametric" | "step"
    meta: dict | None = None

    def __post_init__(self) -> None:
        if self.upper.shape != self.lower.shape:
            raise ValueError("upper and lower surfaces must share their shape")
        if self.upper.ndim != 3 or self.upper.shape[2] != 3:
            raise ValueError("surfaces must have shape (n_span, n_stream, 3)")
        if self.meta is None:
            self.meta = {}

    @property
    def n_span(self) -> int:
        return self.upper.shape[0]

    @property
    def n_stream(self) -> int:
        return self.upper.shape[1]

    @property
    def leading_edge(self) -> np.ndarray:
        return self.lower[:, 0, :]

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        allpts = np.concatenate(
            [self.upper.reshape(-1, 3), self.lower.reshape(-1, 3)], axis=0
        )
        return allpts.min(axis=0), allpts.max(axis=0)

    def reference_length(self) -> float:
        lo, hi = self.bbox()
        return float(hi[0] - lo[0])

    def to_lens(self, n_span: int | None = None, n_stream: int | None = None) -> "BodySurface":
        """Return a copy re-gridded to constant-x lens cross-sections.

        This canonical form (upper/lower meeting at the spanwise tips, one nose
        station) is what the O-grid topology builder requires.  Applying it to
        an already-lens surface simply resamples it.
        """
        from .regrid import triangulate_structured, regrid_to_lens

        n_span = n_span or self.n_span
        n_stream = n_stream or self.n_stream
        verts, faces, normals = triangulate_structured(self.upper, self.lower)
        upper, lower, rmeta = regrid_to_lens(verts, faces, normals, n_span, n_stream)
        return BodySurface(upper=upper, lower=lower, source=self.source,
                           meta={**(self.meta or {}), "regridded": "lens", **rmeta})

    @classmethod
    def from_waverider(cls, geom) -> "BodySurface":
        return cls(
            upper=geom.upper.copy(),
            lower=geom.lower.copy(),
            source="parametric",
            meta=geom.as_dict(),
        )
