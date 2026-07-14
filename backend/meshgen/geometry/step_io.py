"""STEP (ISO 10303) import via OpenCASCADE (OCP).

Reads a STEP solid, tessellates it for display and geometry queries, and
re-grids the wall into the structured upper/lower :class:`BodySurface` the
mesher consumes.  The re-gridding assumes a slender, z-single-valued body over
its planform -- exactly the waverider class this tool targets -- by splitting
triangles into an upper and a lower shell (by facet normal) and sampling each
onto a structured planform grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surface import BodySurface

__all__ = ["StepModel", "load_step", "write_step"]


@dataclass
class StepModel:
    """A tessellated STEP body."""

    vertices: np.ndarray           # (n_v, 3)
    faces: np.ndarray              # (n_f, 3) int triangle indices
    normals: np.ndarray            # (n_f, 3) per-face outward normals
    path: str

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def reference_length(self) -> float:
        lo, hi = self.bbox()
        return float(hi[0] - lo[0])

    def triangle_soup(self) -> dict:
        """Flat data for a web viewer (position + index buffers)."""
        return {
            "positions": self.vertices.astype(float).ravel().tolist(),
            "indices": self.faces.astype(int).ravel().tolist(),
            "bbox_min": self.bbox()[0].tolist(),
            "bbox_max": self.bbox()[1].tolist(),
        }

    def to_body_surface(self, n_span: int = 61, n_stream: int = 41) -> BodySurface:
        """Re-grid the tessellated wall into structured upper/lower patches."""
        upper, lower = _regrid_structured(
            self.vertices, self.faces, self.normals, n_span, n_stream
        )
        return BodySurface(
            upper=upper,
            lower=lower,
            source="step",
            meta={
                "path": self.path,
                "reference_length": self.reference_length(),
                "n_triangles": int(self.faces.shape[0]),
            },
        )


def load_step(path: str, linear_deflection: float = 0.0, angular_deflection: float = 0.3) -> StepModel:
    """Load and tessellate a STEP file.

    Parameters
    ----------
    linear_deflection:
        Max chordal deviation of the triangulation; ``0`` auto-sizes it to the
        bounding-box diagonal (finer for larger models).
    angular_deflection:
        Max angular deviation (radians) between adjacent facets.
    """
    # Imported lazily so the rest of the package works without OCP installed.
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise IOError(f"Failed to read STEP file: {path} (status={status})")
    reader.TransferRoots()
    shape = reader.OneShape()

    # Auto-size the deflection to the model if not given.
    if linear_deflection <= 0.0:
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
        linear_deflection = max(diag * 2e-3, 1e-6)

    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    normals: list[tuple[float, float, float]] = []

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            base = len(verts)
            n_nodes = tri.NbNodes()
            pts = []
            for k in range(1, n_nodes + 1):
                p = tri.Node(k).Transformed(trsf)
                pts.append((p.X(), p.Y(), p.Z()))
                verts.append((p.X(), p.Y(), p.Z()))
            reversed_face = face.Orientation() == 1  # TopAbs_REVERSED
            for k in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(k)
                a, b, c = t.Get()
                ia, ib, ic = a - 1 + base, b - 1 + base, c - 1 + base
                if reversed_face:
                    ib, ic = ic, ib
                faces.append((ia, ib, ic))
                pa, pb, pc = pts[a - 1], pts[b - 1], pts[c - 1]
                n = _tri_normal(pa, pb, pc)
                if reversed_face:
                    n = (-n[0], -n[1], -n[2])
                normals.append(n)
        explorer.Next()

    if not faces:
        raise ValueError(f"No triangulable faces found in STEP file: {path}")

    return StepModel(
        vertices=np.asarray(verts, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        normals=np.asarray(normals, dtype=float),
        path=path,
    )


def write_step(surface: "BodySurface | object", path: str) -> str:
    """Write a faceted solid built from upper+lower structured grids to STEP.

    Each surface quad is split into two planar triangular faces; the two shells
    plus the base are sewn into a shell and exported.  Primarily for inspecting
    generated geometry in a CAD system (and for round-trip testing STEP import).
    """
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_Sewing,
    )
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs

    upper = np.asarray(surface.upper)
    lower = np.asarray(surface.lower)

    def _pnt(v):
        return gp_Pnt(float(v[0]), float(v[1]), float(v[2]))

    def _tri_face(p0, p1, p2):
        poly = BRepBuilderAPI_MakePolygon(_pnt(p0), _pnt(p1), _pnt(p2), True)
        return BRepBuilderAPI_MakeFace(poly.Wire()).Face()

    sew = BRepBuilderAPI_Sewing(1e-6)

    def _add_grid(grid, flip=False):
        ni, nj, _ = grid.shape
        for i in range(ni - 1):
            for j in range(nj - 1):
                a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
                if flip:
                    sew.Add(_tri_face(a, c, b))
                    sew.Add(_tri_face(a, d, c))
                else:
                    sew.Add(_tri_face(a, b, c))
                    sew.Add(_tri_face(a, c, d))

    _add_grid(lower, flip=False)
    _add_grid(upper, flip=True)
    # Close the base (last stream row) between upper and lower.
    ni = upper.shape[0]
    for i in range(ni - 1):
        a, b = lower[i, -1], lower[i + 1, -1]
        c, d = upper[i + 1, -1], upper[i, -1]
        sew.Add(_tri_face(a, b, c))
        sew.Add(_tri_face(a, c, d))

    sew.Perform()
    shell = sew.SewedShape()

    writer = STEPControl_Writer()
    writer.Transfer(shell, STEPControl_AsIs)
    writer.Write(path)
    return path


def _tri_normal(a, b, c) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    mag = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
    return (nx / mag, ny / mag, nz / mag)


def _regrid_structured(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    n_span: int,
    n_stream: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Body-fit a structured surface grid to a slender, z-single-valued body.

    The body is swept from a nose to a base, so a bounding rectangle would
    include off-body corners.  Instead, for each streamwise station ``x`` we
    find the body's spanwise extent ``[y_min(x), y_max(x)]`` from the projected
    triangulation and lay ``n_span`` nodes across it.  Upper/lower z come from
    barycentric sampling of the up/down-facing triangles.
    """
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    tri_pts = vertices[faces]                       # (n_f, 3, 3)
    nz = normals[:, 2]
    tris_up = tri_pts[nz > 1e-6]
    tris_lo = tri_pts[nz < -1e-6]
    all_xy = tri_pts[:, :, :2]

    # Slight inward inset so stations sit on the body, not exactly on the tips.
    eps = 1e-4 * (hi[0] - lo[0])
    xs = np.linspace(lo[0] + eps, hi[0] - eps, n_stream)
    s = np.linspace(0.0, 1.0, n_span)

    upper = np.zeros((n_span, n_stream, 3))
    lower = np.zeros((n_span, n_stream, 3))

    for j, x in enumerate(xs):
        ymin, ymax = _y_extent_at_x(all_xy, x, lo[1], hi[1])
        if ymax - ymin < 1e-9:
            ymin, ymax = -eps, eps
        ys = ymin + s * (ymax - ymin)
        z_up = _sample_z_line(x, ys, tris_up, fill="max")
        z_lo = _sample_z_line(x, ys, tris_lo, fill="min")
        upper[:, j, 0] = x
        upper[:, j, 1] = ys
        upper[:, j, 2] = z_up
        lower[:, j, 0] = x
        lower[:, j, 1] = ys
        lower[:, j, 2] = z_lo

    # Share the leading-edge row (nose station) between the two shells.
    le = 0.5 * (upper[:, 0, :] + lower[:, 0, :])
    upper[:, 0, :] = le
    lower[:, 0, :] = le
    return upper, lower


def _y_extent_at_x(all_xy: np.ndarray, x: float, ylo: float, yhi: float) -> tuple[float, float]:
    """Spanwise extent of the body at streamwise station ``x``.

    Scans candidate y positions and keeps those whose ``(x, y)`` falls inside
    some projected triangle.
    """
    ys = np.linspace(ylo, yhi, 200)
    inside = np.array([_point_in_any(x, y, all_xy) for y in ys])
    if not inside.any():
        return 0.0, 0.0
    idx = np.where(inside)[0]
    return float(ys[idx[0]]), float(ys[idx[-1]])


def _point_in_any(px: float, py: float, tris_xy: np.ndarray) -> bool:
    ax, ay = tris_xy[:, 0, 0], tris_xy[:, 0, 1]
    bx, by = tris_xy[:, 1, 0], tris_xy[:, 1, 1]
    cx, cy = tris_xy[:, 2, 0], tris_xy[:, 2, 1]
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
    l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
    l3 = 1.0 - l1 - l2
    return bool(np.any((l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)))


def _sample_z_line(x: float, ys: np.ndarray, tris: np.ndarray, fill: str) -> np.ndarray:
    """Sample z at ``(x, y)`` for each y from a triangle set (xy projection)."""
    out = np.full(ys.shape, np.nan)
    if tris.shape[0] == 0:
        return np.zeros_like(ys)
    ax, ay = tris[:, 0, 0], tris[:, 0, 1]
    bx, by = tris[:, 1, 0], tris[:, 1, 1]
    cx, cy = tris[:, 2, 0], tris[:, 2, 1]
    az, bz, cz = tris[:, 0, 2], tris[:, 1, 2], tris[:, 2, 2]
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    for i, py in enumerate(ys):
        l1 = ((by - cy) * (x - cx) + (cx - bx) * (py - cy)) / det
        l2 = ((cy - ay) * (x - cx) + (ax - cx) * (py - cy)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if np.any(inside):
            zc = (l1 * az + l2 * bz + l3 * cz)[inside]
            out[i] = zc.max() if fill == "max" else zc.min()
    if np.isnan(out).any():
        good = ~np.isnan(out)
        if good.any():
            out = np.interp(np.arange(out.size),
                            np.where(good)[0], out[good])
        else:
            out = np.zeros_like(ys)
    return out
