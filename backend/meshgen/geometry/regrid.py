"""Re-grid a triangulated or structured body surface into constant-x lens
cross-sections.

The mesher needs the wall as a stack of planar (constant-x) cross-sections that
close cleanly at their spanwise tips -- a "lens" whose lower and upper arcs meet
at the two side edges.  Both the parametric waverider and an imported STEP body
are funnelled through this so the downstream O-grid topology is identical.
"""
from __future__ import annotations

import numpy as np

__all__ = ["triangulate_structured", "regrid_to_lens"]


def triangulate_structured(
    upper: np.ndarray, lower: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate two structured (n_span, n_stream, 3) shells.

    Returns ``(vertices, faces, face_normals)`` with normals oriented outward
    (upper facing +z, lower facing -z).
    """
    verts: list = []
    faces: list = []
    normals: list = []

    def add_grid(grid: np.ndarray, want_up: bool) -> None:
        ni, nj, _ = grid.shape
        base = len(verts)
        verts.extend(grid.reshape(-1, 3))
        for i in range(ni - 1):
            for j in range(nj - 1):
                a = base + i * nj + j
                b = base + (i + 1) * nj + j
                c = base + (i + 1) * nj + (j + 1)
                d = base + i * nj + (j + 1)
                for tri in ((a, b, c), (a, c, d)):
                    p0, p1, p2 = (np.asarray(verts[t]) for t in tri)
                    n = np.cross(p1 - p0, p2 - p0)
                    nn = np.linalg.norm(n) or 1.0
                    n = n / nn
                    if (n[2] > 0) != want_up:
                        tri = (tri[0], tri[2], tri[1])
                        n = -n
                    faces.append(tri)
                    normals.append(n)

    add_grid(lower, want_up=False)
    add_grid(upper, want_up=True)
    return (
        np.asarray(verts, dtype=float),
        np.asarray(faces, dtype=np.int64),
        np.asarray(normals, dtype=float),
    )


def regrid_to_lens(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    n_span: int,
    n_stream: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Body-fit constant-x lens cross-sections to a slender, z-single-valued body.

    For each streamwise station ``x`` the spanwise extent ``[y_min, y_max]`` is
    found from the projected triangulation and ``n_span`` nodes are laid across
    it; upper/lower z come from barycentric sampling of the up/down-facing
    triangles.  The two spanwise tips are forced coincident so each lens closes.
    """
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    tri_pts = vertices[faces]
    nz = normals[:, 2]
    tris_up = tri_pts[nz > 1e-6]
    tris_lo = tri_pts[nz < -1e-6]
    all_xy = tri_pts[:, :, :2]

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

    # Force the spanwise tips coincident so each lens closes to a sharp edge.
    for tip in (0, -1):
        mid = 0.5 * (upper[tip, :, :] + lower[tip, :, :])
        upper[tip, :, :] = mid
        lower[tip, :, :] = mid
    # Do NOT collapse the nose station: keep it a small finite lens so the
    # O-grid's first cross-plane has positive-volume cells (a fully coincident
    # nose row produces a degenerate polar cap).
    return upper, lower


def _y_extent_at_x(all_xy: np.ndarray, x: float, ylo: float, yhi: float) -> tuple[float, float]:
    ys = np.linspace(ylo, yhi, 240)
    inside = _points_in_any(np.full_like(ys, x), ys, all_xy)
    if not inside.any():
        return 0.0, 0.0
    idx = np.where(inside)[0]
    return float(ys[idx[0]]), float(ys[idx[-1]])


def _bary(px, py, tris):
    ax, ay = tris[:, 0, 0], tris[:, 0, 1]
    bx, by = tris[:, 1, 0], tris[:, 1, 1]
    cx, cy = tris[:, 2, 0], tris[:, 2, 1]
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
    l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
    l3 = 1.0 - l1 - l2
    return l1, l2, l3


def _points_in_any(pxs: np.ndarray, pys: np.ndarray, tris_xy: np.ndarray) -> np.ndarray:
    out = np.zeros(pxs.shape, dtype=bool)
    for i, (px, py) in enumerate(zip(pxs, pys)):
        l1, l2, l3 = _bary(px, py, tris_xy)
        out[i] = bool(np.any((l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)))
    return out


def _sample_z_line(x: float, ys: np.ndarray, tris: np.ndarray, fill: str) -> np.ndarray:
    out = np.full(ys.shape, np.nan)
    if tris.shape[0] == 0:
        return np.zeros_like(ys)
    az, bz, cz = tris[:, 0, 2], tris[:, 1, 2], tris[:, 2, 2]
    tris_xy = tris[:, :, :2]
    for i, py in enumerate(ys):
        l1, l2, l3 = _bary(x, py, tris_xy)
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if np.any(inside):
            zc = (l1 * az + l2 * bz + l3 * cz)[inside]
            out[i] = zc.max() if fill == "max" else zc.min()
    if np.isnan(out).any():
        good = ~np.isnan(out)
        if good.any():
            out = np.interp(np.arange(out.size), np.where(good)[0], out[good])
        else:
            out = np.zeros_like(ys)
    return out
