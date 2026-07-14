"""Structured multiblock CGNS writer (CGNS/HDF5 file mapping).

Writes each structured block as a CGNS ``Zone_t`` of ``ZoneType`` Structured,
with ``GridCoordinates`` and per-face ``ZoneBC`` boundary conditions -- the
native representation for a multiblock structured mesh, readable by CGNS tools,
SU2, and most CFD solvers.

The CGNS/HDF5 mapping represents every SIDS node as an HDF5 group carrying the
attributes ``name``/``label``/``flags``/``type`` and an optional ``" data"``
child dataset.  Multidimensional arrays are stored with reversed (Fortran)
dimension order, per the mapping.

Reference: CGNS/HDF5 File Mapping, https://cgns.github.io/
"""
from __future__ import annotations

import numpy as np

from ..mesh.block import BCType, Face, StructuredBlock
from ..mesh.multiblock import MultiBlockMesh

__all__ = ["write_cgns"]

# CGNS BCType_t names for our boundary kinds.
_BC_CGNS = {
    BCType.WALL: "BCWallViscous",
    BCType.FARFIELD: "BCFarfield",
    BCType.SYMMETRY: "BCSymmetryPlane",
    BCType.INFLOW: "BCInflowSupersonic",
    BCType.OUTFLOW: "BCOutflowSupersonic",
}


def _s(text: str, length: int) -> np.ndarray:
    """Null-padded fixed-length byte string as an HDF5 attribute value."""
    b = text.encode("ascii")[: length - 1]
    return np.frombuffer(b + b"\x00" * (length - len(b)), dtype="S1")


def _set_node_attrs(grp, name: str, label: str, dtype: str) -> None:
    # CGNS expects fixed-length char arrays for these attributes.
    for attr, val, ln in (("name", name, 33), ("label", label, 33), ("type", dtype, 3)):
        grp.attrs.create(attr, _s(val, ln))
    grp.attrs.create("flags", np.array([1], dtype="int32"))


def _node(parent, name: str, label: str, dtype: str = "MT", data=None):
    """Create a CGNS/HDF5 node (group) under ``parent``."""
    grp = parent.create_group(name)
    _set_node_attrs(grp, name, label, dtype)
    if data is not None:
        arr = np.asarray(data)
        # Reverse dimension order for the Fortran-ordered CGNS mapping.
        if arr.ndim > 1:
            ds = arr.T
        else:
            ds = arr
        grp.create_dataset(" data", data=np.ascontiguousarray(ds))
    return grp


def _char_node(parent, name: str, label: str, text: str):
    grp = parent.create_group(name)
    _set_node_attrs(grp, name, label, "C1")
    grp.create_dataset(" data", data=_s(text, len(text) + 1)[: len(text)])
    return grp


def write_cgns(mesh: MultiBlockMesh, path: str) -> str:
    """Write ``mesh`` as a structured multiblock CGNS/HDF5 file."""
    import h5py

    with h5py.File(path, "w") as f:
        root = f["/"]
        _set_node_attrs(root, "HDF5 MotherNode", "Root Node of HDF5 File", "MT")
        root.create_dataset(" format", data=_s("IEEE_LITTLE_32", 15)[:14])
        root.create_dataset(" hdf5version", data=_s(h5py.version.hdf5_version, 33)[:32])

        _node(root, "CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4",
              np.array([3.4], dtype="float32"))

        base = _node(root, "Base", "CGNSBase_t", "I4",
                     np.array([3, 3], dtype="int32"))  # cell dim 3, phys dim 3

        for b, block in enumerate(mesh.blocks):
            _write_zone(base, block, f"Zone{b + 1}")

    return path


def _write_zone(base, block: StructuredBlock, name: str) -> None:
    ni, nj, nk = block.shape
    # Zone size: [ [vertex, cell, boundary] per index dim ] -> shape (3, 3).
    zsize = np.array(
        [[ni, ni - 1, 0], [nj, nj - 1, 0], [nk, nk - 1, 0]], dtype="int32"
    )
    zone = _node(base, name, "Zone_t", "I4", zsize)
    _char_node(zone, "ZoneType", "ZoneType_t", "Structured")

    gc = _node(zone, "GridCoordinates", "GridCoordinates_t")
    x = np.ascontiguousarray(block.coords[..., 0])
    y = np.ascontiguousarray(block.coords[..., 1])
    z = np.ascontiguousarray(block.coords[..., 2])
    _node(gc, "CoordinateX", "DataArray_t", "R8", x)
    _node(gc, "CoordinateY", "DataArray_t", "R8", y)
    _node(gc, "CoordinateZ", "DataArray_t", "R8", z)

    # ZoneBC: one BC_t per physical face, as a PointRange over the face.
    zbc = _node(zone, "ZoneBC", "ZoneBC_t")
    for face, bc in block.bc.items():
        if bc not in _BC_CGNS:
            continue
        rng = _face_point_range(face, ni, nj, nk)
        bc_node = _char_node(zbc, f"{face.value}", "BC_t", _BC_CGNS[bc])
        # PointRange as an IndexRange_t (2 x 3): begin, end vertices.
        _node(bc_node, "PointRange", "IndexRange_t", "I4",
              np.array(rng, dtype="int32"))
        _char_node(bc_node, "GridLocation", "GridLocation_t", "Vertex")


def _face_point_range(face: Face, ni: int, nj: int, nk: int) -> list[list[int]]:
    """1-based inclusive [begin, end] vertex range for a logical face."""
    full = {"i": (1, ni), "j": (1, nj), "k": (1, nk)}
    lo = {"i": 1, "j": 1, "k": 1}
    if face == Face.IMIN:
        return [[1, 1, 1], [1, nj, nk]]
    if face == Face.IMAX:
        return [[ni, 1, 1], [ni, nj, nk]]
    if face == Face.JMIN:
        return [[1, 1, 1], [ni, 1, nk]]
    if face == Face.JMAX:
        return [[1, nj, 1], [ni, nj, nk]]
    if face == Face.KMIN:
        return [[1, 1, 1], [ni, nj, 1]]
    return [[1, 1, nk], [ni, nj, nk]]
