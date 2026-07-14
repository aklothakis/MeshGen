"""Structured multiblock topology builder for waverider-class bodies.

Builds a body-fitted **O-grid**: each streamwise station's cross-section (a
closed loop wrapping the lower then upper surface) is extruded outward to a
circular farfield with hyperbolic-tangent wall-normal clustering.  Stacking the
cross-planes along the streamwise direction gives a curvilinear volume grid,
which is then split into several connected blocks -> a genuine structured
multiblock mesh.

Boundary conditions:
    * m = 0            -> WALL   (body) / OUTFLOW behind base handled by BC on j
    * m = n_normal-1   -> FARFIELD
    * j = 0            -> INFLOW  (supersonic, leading-edge plane)
    * j = n_stream-1   -> OUTFLOW (base plane)
    * wrap seam        -> INTERFACE (periodic O-grid closure)
    * interior j/l splits -> INTERFACE (block-to-block)
"""
from __future__ import annotations

import numpy as np

from ..domain.farfield import DomainParams
from ..flow.conditions import FlowConditions
from ..geometry.surface import BodySurface
from .block import BCType, Face, StructuredBlock
from .clustering import distribute_wall_normal, tanh_one_sided
from .multiblock import MultiBlockMesh

__all__ = ["build_multiblock"]


def _wrap_loop(surface: BodySurface) -> np.ndarray:
    """Closed cross-section loop per station: (n_wrap, n_stream, 3).

    Order: lower arc across the span (tip- -> tip+), then the upper arc back,
    dropping the two shared tip nodes so the lens closes with no coincident
    duplicates.  ``n_wrap = 2 * n_span - 2``.
    """
    lower = surface.lower                       # (n_span, n_stream, 3)
    upper = surface.upper
    # lower[0 .. n-1] then upper[n-2 .. 1]; upper[0]==lower[0] and
    # upper[-1]==lower[-1] are the coincident tips, so exclude them.
    loop = np.concatenate([lower, upper[-2:0:-1]], axis=0)
    return loop


def _refine_wrap(loop: np.ndarray, mult: int) -> np.ndarray:
    if mult <= 1:
        return loop
    n_wrap, n_stream, _ = loop.shape
    # Periodic spline-free linear refinement around the loop.
    idx = np.arange(n_wrap)
    fine_idx = np.linspace(0, n_wrap, n_wrap * mult, endpoint=False)
    out = np.zeros((fine_idx.size, n_stream, 3))
    for j in range(n_stream):
        for d in range(3):
            out[:, j, d] = np.interp(
                fine_idx, np.append(idx, n_wrap),
                np.append(loop[:, j, d], loop[0, j, d]),
                period=n_wrap,
            )
    return out


def build_multiblock(
    surface: BodySurface,
    domain: DomainParams | None = None,
    flow: FlowConditions | None = None,
) -> MultiBlockMesh:
    """Generate a structured multiblock O-grid around ``surface``."""
    domain = domain or DomainParams()

    # Canonicalise to constant-x lens cross-sections so the wrap is well posed.
    if surface.meta is None or surface.meta.get("regridded") != "lens":
        surface = surface.to_lens()

    loop = _wrap_loop(surface)
    loop = _refine_wrap(loop, domain.n_wrap_mult)
    n_wrap, n_stream, _ = loop.shape

    lo, hi = surface.bbox()
    body_length = float(hi[0] - lo[0])
    cross_scale = 0.5 * float(max(hi[1] - lo[1], hi[2] - lo[2]))
    R_far = domain.farfield_radius(cross_scale, body_length)
    first_cell = domain.resolve_first_cell(flow, body_length)

    n_normal = domain.n_normal

    # Per-station cross-section centroid (in y-z), used as the radial origin.
    coords = np.zeros((n_wrap, n_normal, n_stream, 3))

    # Wall-normal normalised distribution, sized to the representative segment.
    # (Recomputed per station to keep the first cell near-constant physically.)
    for j in range(n_stream):
        ring = loop[:, j, :]                      # (n_wrap, 3)
        cy = ring[:, 1].mean()
        cz = ring[:, 2].mean()
        cx = ring[:, 0]                           # keep station x per node
        # Outward radial direction in the y-z plane from the centroid.
        dy = ring[:, 1] - cy
        dz = ring[:, 2] - cz
        rad = np.hypot(dy, dz)
        rad_safe = np.where(rad < 1e-12, 1e-12, rad)
        uy = dy / rad_safe
        uz = dz / rad_safe
        # Degenerate station (leading edge): give a tiny outward fan so cells
        # have positive volume instead of collapsing.
        if np.all(rad < 1e-9):
            ang = np.linspace(0.0, 2.0 * np.pi, n_wrap, endpoint=False)
            uy = np.cos(ang)
            uz = np.sin(ang)
            rad = np.zeros(n_wrap)

        seg_len = R_far - rad                      # radial distance wall->farfield
        seg_len = np.maximum(seg_len, 0.05 * R_far)
        rep_len = float(np.median(seg_len))
        s = _wall_normal_dist(n_normal, rep_len, first_cell)  # normalised [0,1]

        for m in range(n_normal):
            t = s[m]
            coords[:, m, j, 0] = cx
            coords[:, m, j, 1] = cy + (rad + t * seg_len) * uy
            coords[:, m, j, 2] = cz + (rad + t * seg_len) * uz

    coords = _orient_positive(coords)
    return _split_blocks(coords, domain, surface)


def _orient_positive(coords: np.ndarray) -> np.ndarray:
    """Flip the wrap axis if the grid handedness yields negative cell volumes."""
    from .block import _hex_cell_volumes

    vols = _hex_cell_volumes(coords)
    if np.median(vols) < 0.0:
        coords = coords[::-1].copy()
    return coords


def _wall_normal_dist(n_normal: int, total: float, first_cell: float) -> np.ndarray:
    """Normalised [0,1] wall-normal distribution hitting ``first_cell``."""
    d = distribute_wall_normal(n_normal, total, min(first_cell, 0.2 * total))
    return d / d[-1]


def _split_blocks(
    coords: np.ndarray,
    domain: DomainParams,
    surface: BodySurface,
) -> MultiBlockMesh:
    """Split the (n_wrap, n_normal, n_stream) grid into connected blocks.

    Splits along the wrap (i) and streamwise (k) directions with one node of
    overlap so neighbouring blocks share a face (1-to-1 interface).
    """
    n_wrap, n_normal, n_stream = coords.shape[:3]
    mesh = MultiBlockMesh()

    k_splits = _split_indices(n_stream, domain.n_stream_blocks)
    i_splits = _split_indices(n_wrap, domain.n_wrap_blocks)

    n_ib = len(i_splits) - 1
    n_kb = len(k_splits) - 1

    for kb in range(n_kb):
        k0, k1 = k_splits[kb], k_splits[kb + 1]
        for ib in range(n_ib):
            i0, i1 = i_splits[ib], i_splits[ib + 1]
            sub = coords[i0 : i1 + 1, :, k0 : k1 + 1, :].copy()
            block = StructuredBlock(name=f"block_i{ib}_k{kb}", coords=sub)

            bc: dict[Face, BCType] = {}
            bc[Face.JMIN] = BCType.WALL           # m = 0
            bc[Face.JMAX] = BCType.FARFIELD       # m = n_normal-1
            # Streamwise ends.
            if kb == 0:
                bc[Face.KMIN] = BCType.INFLOW
            else:
                bc[Face.KMIN] = BCType.INTERFACE
            if kb == n_kb - 1:
                bc[Face.KMAX] = BCType.OUTFLOW
            else:
                bc[Face.KMAX] = BCType.INTERFACE
            # Wrap direction: interior splits are interfaces; the O-grid seam is
            # also an interface (block 0 imin meets block n_ib-1 imax).
            bc[Face.IMIN] = BCType.INTERFACE
            bc[Face.IMAX] = BCType.INTERFACE
            block.bc = bc
            mesh.add(block)

    mesh.meta = {
        "topology": "O-grid",
        "n_wrap": int(n_wrap),
        "n_normal": int(n_normal),
        "n_stream": int(n_stream),
        "n_blocks": len(mesh.blocks),
        "source": surface.source,
        "wrap_periodic": True,
    }
    return mesh


def _split_indices(n: int, n_blocks: int) -> list[int]:
    n_blocks = max(1, min(n_blocks, n - 1))
    edges = np.linspace(0, n - 1, n_blocks + 1)
    return sorted(set(int(round(e)) for e in edges))
