"""Elliptic (Winslow) smoothing of structured cross-planes.

State-of-the-art structured grid generation refines an algebraic starting grid
by solving elliptic PDEs for the node coordinates, which yields smooth,
near-orthogonal grid lines (Thompson–Thames–Mastin; Winslow).  Here each
constant-x O-grid cross-section (wrap x wall-normal, in the y-z plane) is
relaxed with the inverted-Laplace (Winslow) operator, holding the wall and
farfield boundaries fixed and treating the wrap direction as periodic.

To keep the boundary-layer clustering that a pure smoother would wash out, each
radial line is re-clustered to the target first-cell height after smoothing --
so the result is smoother and more orthogonal *and* still resolves the wall.
"""
from __future__ import annotations

import numpy as np

from .clustering import distribute_wall_normal

__all__ = ["smooth_crossplanes"]


def smooth_crossplanes(
    coords: np.ndarray,
    first_cell: float,
    n_iter: int = 150,
    omega: float = 0.6,
    recluster: bool = True,
) -> np.ndarray:
    """Winslow-smooth every (wrap, normal) cross-plane of an O-grid volume.

    Parameters
    ----------
    coords:
        ``(n_wrap, n_normal, n_stream, 3)`` grid.  X is constant within each
        cross-plane (constant-x lens stations); only Y, Z are relaxed.
    first_cell:
        Absolute target first wall-cell height, re-imposed per radial line.
    n_iter:
        Gauss–Seidel/Jacobi sweeps.
    omega:
        Under-relaxation factor in (0, 1].
    """
    from .block import _hex_cell_volumes

    original = coords
    coords = coords.copy()
    Y = coords[..., 1]                          # (nw, nn, ns)
    Z = coords[..., 2]

    # Freeze wrap columns at the sharp lens tips: Winslow smoothing folds around
    # concave corners, so hold a band around each high-curvature wall node.
    free = _wrap_free_mask(coords)              # (nw,) bool, True = updatable
    free_col = free[:, None, None]

    for _ in range(n_iter):
        # Interior wall-normal (eta, axis 1) rows only; wall & farfield fixed.
        Yip, Yim = np.roll(Y, -1, axis=0), np.roll(Y, 1, axis=0)  # xi +/- (periodic)
        Zip, Zim = np.roll(Z, -1, axis=0), np.roll(Z, 1, axis=0)

        Yc, Zc = Y[:, 1:-1], Z[:, 1:-1]
        Yjp, Yjm = Y[:, 2:], Y[:, :-2]
        Zjp, Zjm = Z[:, 2:], Z[:, :-2]

        Y_xi = 0.5 * (Yip[:, 1:-1] - Yim[:, 1:-1])
        Z_xi = 0.5 * (Zip[:, 1:-1] - Zim[:, 1:-1])
        Y_eta = 0.5 * (Yjp - Yjm)
        Z_eta = 0.5 * (Zjp - Zjm)

        alpha = Y_eta ** 2 + Z_eta ** 2
        gamma = Y_xi ** 2 + Z_xi ** 2
        beta = Y_xi * Y_eta + Z_xi * Z_eta
        denom = 2.0 * (alpha + gamma)
        denom = np.where(denom < 1e-30, 1e-30, denom)

        for P, Pip, Pim in ((Y, Yip, Yim), (Z, Zip, Zim)):
            Pc = P[:, 1:-1]
            Pjp, Pjm = P[:, 2:], P[:, :-2]
            cross = 0.25 * (Pip[:, 2:] - Pip[:, :-2] - Pim[:, 2:] + Pim[:, :-2])
            num = (
                alpha * (Pip[:, 1:-1] + Pim[:, 1:-1])
                + gamma * (Pjp + Pjm)
                - 0.5 * beta * cross
            )
            updated = (1.0 - omega) * Pc + omega * (num / denom)
            # Only move nodes on non-frozen wrap columns (free_col is (nw,1,1)).
            P[:, 1:-1] = np.where(free_col, updated, Pc)

    coords[..., 1] = Y
    coords[..., 2] = Z

    if recluster:
        coords = _recluster_radial(coords, first_cell)

    # Validity guard: never return a folded grid -- fall back to the algebraic one.
    if np.median(_hex_cell_volumes(coords)) < 0:
        coords = coords[::-1].copy()  # match handedness before testing
    if (_hex_cell_volumes(coords) <= 0).any():
        return original
    return coords


def _wrap_free_mask(coords: np.ndarray, band: int = 4, thresh_deg: float = 55.0) -> np.ndarray:
    """Mark wrap columns free to move, freezing bands around sharp wall corners.

    Sharp corners are wall (eta=0) nodes whose turning angle exceeds a
    threshold; Winslow smoothing folds if these are relaxed.
    """
    wall = coords[:, 0, :, :].mean(axis=1)      # representative wall ring (nw, 3)
    prev = np.roll(wall, 1, axis=0)
    nxt = np.roll(wall, -1, axis=0)
    a = wall - prev
    b = nxt - wall
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-30)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-30)
    turn = np.degrees(np.arccos(np.einsum("ij,ij->i", a, b).clip(-1, 1)))

    nw = coords.shape[0]
    free = np.ones(nw, dtype=bool)
    corners = np.where(turn > thresh_deg)[0]
    for c in corners:
        for d in range(-band, band + 1):
            free[(c + d) % nw] = False
    # Always freeze the O-grid seam neighbourhood.
    for d in range(-1, 2):
        free[d % nw] = False
    return free


def _recluster_radial(coords: np.ndarray, first_cell: float) -> np.ndarray:
    """Redistribute nodes along each smoothed radial line to hit ``first_cell``.

    Line shape (hence orthogonality) is preserved; only the along-line node
    positions change, restoring boundary-layer clustering.
    """
    nw, nn, ns, _ = coords.shape
    out = coords.copy()
    for i in range(nw):
        for j in range(ns):
            line = coords[i, :, j, :]                 # (nn, 3)
            seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            total = s[-1]
            if total < 1e-14:
                continue
            fc = min(first_cell, 0.4 * total)
            target = distribute_wall_normal(nn, total, fc)
            for d in range(3):
                out[i, :, j, d] = np.interp(target, s, line[:, d])
    return out
