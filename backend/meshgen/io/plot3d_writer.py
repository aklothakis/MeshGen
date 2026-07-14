"""Plot3D multiblock structured-grid writer (formatted ASCII, 3-D, whole).

Plot3D is the canonical structured-multiblock exchange format and preserves the
block topology exactly, so it doubles as a faithful round-trip/verification
format alongside SU2/CGNS.
"""
from __future__ import annotations

from ..mesh.multiblock import MultiBlockMesh

__all__ = ["write_plot3d"]


def write_plot3d(mesh: MultiBlockMesh, path: str) -> str:
    lines: list[str] = [str(len(mesh.blocks))]
    for block in mesh.blocks:
        ni, nj, nk = block.shape
        lines.append(f"{ni} {nj} {nk}")
    for block in mesh.blocks:
        c = block.coords
        # Plot3D orders as X(all), Y(all), Z(all) in Fortran (i fastest) order.
        for d in range(3):
            flat = c[..., d].ravel(order="F")
            lines.extend(f"{v:.10e}" for v in flat)
    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return path
