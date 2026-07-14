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
    blunt_tol: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Body-fit constant-x lens cross-sections to a slender, z-single-valued body.

    For each streamwise station ``x`` the spanwise extent ``[y_min, y_max]`` is
    found from the projected triangulation and ``n_span`` nodes are laid across
    it.  The upper/lower z at each node are the **vertical envelope** -- the
    highest and lowest surface directly above/below that ``(x, y)`` -- computed
    over *all* triangles (no normal classification, which misfires near blunt or
    vertical edges).

    Spanwise tips are collapsed to a sharp edge only when the body is actually
    sharp there; a **blunt** edge (finite thickness at the tip) is preserved.
    ``blunt_tol`` is the tip-thickness fraction (of the max cross-section
    thickness) above which an edge is treated as blunt.

    Returns ``(upper, lower, meta)`` where ``meta['blunt_tips'] = (bool, bool)``
    for the ``i=0`` and ``i=-1`` spanwise tips.
    """
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    tri_pts = vertices[faces]
    all_xy = tri_pts[:, :, :2]
    az, bz, cz = tri_pts[:, 0, 2], tri_pts[:, 1, 2], tri_pts[:, 2, 2]

    xspan = hi[0] - lo[0]
    eps = 1e-3 * xspan
    xs = np.linspace(lo[0] + eps, hi[0] - eps, n_stream)
    s = np.linspace(0.0, 1.0, n_span)

    upper = np.zeros((n_span, n_stream, 3))
    lower = np.zeros((n_span, n_stream, 3))

    for j, x in enumerate(xs):
        ymin, ymax = _y_extent_at_x(all_xy, x, lo[1], hi[1])
        if ymax - ymin < 1e-9:
            ymin, ymax = -eps, eps
        # Inset a hair so the tip nodes sit on the top/bottom faces (capturing
        # blunt-edge thickness) rather than exactly on the vertical edge.
        pad = 1e-3 * (ymax - ymin)
        ys = (ymin + pad) + s * ((ymax - pad) - (ymin + pad))
        z_up, z_lo = _envelope_z_line(x, ys, all_xy, az, bz, cz)
        upper[:, j, 0] = x
        upper[:, j, 1] = ys
        upper[:, j, 2] = z_up
        lower[:, j, 0] = x
        lower[:, j, 1] = ys
        lower[:, j, 2] = z_lo

    # Classify each spanwise tip as sharp or blunt from its thickness.
    thick = upper[:, :, 2] - lower[:, :, 2]
    max_thick = float(thick.max()) if thick.size else 0.0
    blunt = []
    for tip in (0, -1):
        tip_thick = float(np.median(thick[tip]))
        is_blunt = max_thick > 1e-12 and tip_thick > blunt_tol * max_thick
        blunt.append(is_blunt)
        if not is_blunt:
            mid = 0.5 * (upper[tip, :, :] + lower[tip, :, :])
            upper[tip, :, :] = mid
            lower[tip, :, :] = mid
    return upper, lower, {"blunt_tips": (blunt[0], blunt[1])}


def _y_extent_at_x(all_xy: np.ndarray, x: float, ylo: float, yhi: float) -> tuple[float, float]:
    ys = np.linspace(ylo, yhi, 400)
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


def _envelope_z_line(
    x: float, ys: np.ndarray, tris_xy: np.ndarray,
    az: np.ndarray, bz: np.ndarray, cz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertical envelope at ``(x, y)`` for each y: (upper z, lower z).

    Interpolates z on every triangle whose xy-projection contains the point and
    returns the max (top surface) and min (bottom surface).  Using all triangles
    -- rather than pre-classifying by normal -- makes this robust at blunt and
    vertical edges, which is where the old min/max-per-class sampler spiked.
    """
    up = np.full(ys.shape, np.nan)
    dn = np.full(ys.shape, np.nan)
    for i, py in enumerate(ys):
        l1, l2, l3 = _bary(x, py, tris_xy)
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if np.any(inside):
            zc = (l1 * az + l2 * bz + l3 * cz)[inside]
            up[i] = zc.max()
            dn[i] = zc.min()
    up = _fill_gaps(up)
    dn = _fill_gaps(dn)
    return up, dn


def _fill_gaps(v: np.ndarray) -> np.ndarray:
    if np.isnan(v).any():
        good = ~np.isnan(v)
        if good.any():
            return np.interp(np.arange(v.size), np.where(good)[0], v[good])
        return np.zeros_like(v)
    return v
