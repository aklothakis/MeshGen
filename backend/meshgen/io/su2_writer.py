"""Native SU2 mesh writer (.su2).

Writes a 3-D unstructured hexahedral mesh with tagged boundary markers, the
format the SU2 CFD suite consumes directly.  VTK element codes: 12 = hexahedron,
9 = quadrilateral.
"""
from __future__ import annotations

from ..mesh.multiblock import MultiBlockMesh
from .assembler import UnstructuredMesh, assemble

__all__ = ["write_su2"]

_VTK_HEX = 12
_VTK_QUAD = 9


def write_su2(mesh: MultiBlockMesh | UnstructuredMesh, path: str) -> str:
    um = mesh if isinstance(mesh, UnstructuredMesh) else assemble(mesh)

    lines: list[str] = []
    lines.append("NDIME= 3")

    # Volume elements.
    lines.append(f"NELEM= {um.n_cells}")
    for e, h in enumerate(um.hexes):
        lines.append(f"{_VTK_HEX} " + " ".join(map(str, h.tolist())) + f" {e}")

    # Points.
    lines.append(f"NPOIN= {um.n_points}")
    for i, p in enumerate(um.points):
        lines.append(f"{p[0]:.10e} {p[1]:.10e} {p[2]:.10e} {i}")

    # Boundary markers.
    tags = sorted(um.markers.keys())
    lines.append(f"NMARK= {len(tags)}")
    for tag in tags:
        quads = um.markers[tag]
        lines.append(f"MARKER_TAG= {tag}")
        lines.append(f"MARKER_ELEMS= {quads.shape[0]}")
        for q in quads:
            lines.append(f"{_VTK_QUAD} " + " ".join(map(str, q.tolist())))

    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return path
