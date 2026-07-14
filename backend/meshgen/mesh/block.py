"""Structured block and boundary-condition primitives."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

__all__ = ["BCType", "Face", "StructuredBlock"]


class BCType(str, enum.Enum):
    WALL = "wall"                    # viscous no-slip body surface
    FARFIELD = "farfield"           # freestream / characteristic outer boundary
    SYMMETRY = "symmetry"           # symmetry plane (y = 0 half-model)
    INFLOW = "inflow"               # supersonic inflow
    OUTFLOW = "outflow"             # supersonic outflow / extrapolation
    INTERFACE = "interface"         # block-to-block 1-to-1 connection


class Face(str, enum.Enum):
    IMIN = "imin"
    IMAX = "imax"
    JMIN = "jmin"
    JMAX = "jmax"
    KMIN = "kmin"
    KMAX = "kmax"


@dataclass
class StructuredBlock:
    """A single structured (curvilinear) block.

    ``coords`` has shape ``(ni, nj, nk, 3)``.  ``bc`` maps each of the six
    logical faces to a :class:`BCType`; unset faces default to INTERFACE and
    are expected to be resolved by block connectivity.
    """

    name: str
    coords: np.ndarray                        # (ni, nj, nk, 3)
    bc: dict[Face, BCType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coords.ndim != 4 or self.coords.shape[3] != 3:
            raise ValueError("coords must have shape (ni, nj, nk, 3)")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.coords.shape[:3]

    @property
    def n_nodes(self) -> int:
        ni, nj, nk = self.shape
        return ni * nj * nk

    @property
    def n_cells(self) -> int:
        ni, nj, nk = self.shape
        return max(ni - 1, 1) * max(nj - 1, 1) * max(nk - 1, 1)

    def face_nodes(self, face: Face) -> np.ndarray:
        c = self.coords
        return {
            Face.IMIN: c[0, :, :, :],
            Face.IMAX: c[-1, :, :, :],
            Face.JMIN: c[:, 0, :, :],
            Face.JMAX: c[:, -1, :, :],
            Face.KMIN: c[:, :, 0, :],
            Face.KMAX: c[:, :, -1, :],
        }[face]

    def cell_volumes(self) -> np.ndarray:
        """Signed volumes of every hexahedral cell (decomposed into tetrahedra)."""
        return _hex_cell_volumes(self.coords)

    def min_cell_volume(self) -> float:
        return float(self.cell_volumes().min())


def _hex_cell_volumes(coords: np.ndarray) -> np.ndarray:
    """Signed volume of each hex cell via a 6-tetra decomposition.

    Node ordering per cell follows the VTK/CGNS hexahedron convention.
    """
    p = coords
    # Eight corners of every cell.
    p000 = p[:-1, :-1, :-1]
    p100 = p[1:, :-1, :-1]
    p110 = p[1:, 1:, :-1]
    p010 = p[:-1, 1:, :-1]
    p001 = p[:-1, :-1, 1:]
    p101 = p[1:, :-1, 1:]
    p111 = p[1:, 1:, 1:]
    p011 = p[:-1, 1:, 1:]

    def tet(a, b, c, d):
        return np.einsum("...i,...i->...", np.cross(b - a, c - a), d - a) / 6.0

    # Standard 6-tetra split of a hexahedron.
    vol = (
        tet(p000, p100, p110, p111)
        + tet(p000, p110, p010, p111)
        + tet(p000, p010, p011, p111)
        + tet(p000, p011, p001, p111)
        + tet(p000, p001, p101, p111)
        + tet(p000, p101, p100, p111)
    )
    return vol
