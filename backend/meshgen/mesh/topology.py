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

    Order: lower arc across the span (tip- -> tip+), then the upper arc back.
    A **sharp** tip (upper==lower) is dropped to avoid a zero-length wrap edge;
    a **blunt** tip is kept so its finite edge thickness is meshed.
    """
    lower = surface.lower                       # (n_span, n_stream, 3)
    upper = surface.upper
    n_span = lower.shape[0]
    blunt0, blunt_last = (surface.meta or {}).get("blunt_tips", (False, False))

    idx = list(range(n_span - 1, -1, -1))       # upper arc, tip+ -> tip-
    if not blunt_last:
        idx = idx[1:]                           # drop coincident upper[-1]
    if not blunt0:
        idx = idx[:-1]                          # drop coincident upper[0]
    loop = np.concatenate([lower, upper[idx]], axis=0)
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

    # Precompute per-station extrusion data: centroid, radial + wall-normal
    # directions, and the wall-normal clustering distribution.
    stations = []
    for j in range(n_stream):
        ring = loop[:, j, :]                      # (n_wrap, 3)
        cy = ring[:, 1].mean()
        cz = ring[:, 2].mean()
        cx = ring[:, 0].copy()                    # keep station x per node
        dy = ring[:, 1] - cy
        dz = ring[:, 2] - cz
        rad = np.hypot(dy, dz)
        rad_safe = np.where(rad < 1e-12, 1e-12, rad)
        uy = dy / rad_safe                        # centroid-radial direction
        uz = dz / rad_safe
        if np.all(rad < 1e-9):                     # degenerate LE: radial fan
            ang = np.linspace(0.0, 2.0 * np.pi, n_wrap, endpoint=False)
            uy, uz = np.cos(ang), np.sin(ang)

        oy = cy + R_far * uy                       # farfield circle points
        oz = cz + R_far * uz

        # Wall-normal direction (periodic tangent via roll, rotated 90 deg).
        ty = 0.5 * (np.roll(ring[:, 1], -1) - np.roll(ring[:, 1], 1))
        tz = 0.5 * (np.roll(ring[:, 2], -1) - np.roll(ring[:, 2], 1))
        ny, nz = tz.copy(), -ty.copy()
        flip = (ny * dy + nz * dz) < 0.0
        ny[flip] *= -1.0
        nz[flip] *= -1.0
        nlen = np.hypot(ny, nz)
        nlen = np.where(nlen < 1e-12, 1e-12, nlen)
        ny /= nlen
        nz /= nlen
        if np.all(rad < 1e-9):
            ny, nz = uy.copy(), uz.copy()

        seg_len = np.maximum(np.hypot(oy - ring[:, 1], oz - ring[:, 2]), 0.05 * R_far)
        s = _wall_normal_dist(n_normal, float(np.median(seg_len)), first_cell)
        stations.append(dict(ring=ring, cx=cx, oy=oy, oz=oz,
                             ny=ny, nz=nz, seg_len=seg_len, s=s))

    def place(g: float) -> np.ndarray:
        """Extrude with wall-normal weight ``g`` blended into radial-to-circle."""
        coords = np.zeros((n_wrap, n_normal, n_stream, 3))
        for j, st in enumerate(stations):
            ring, s = st["ring"], st["s"]
            ny, nz, seg = st["ny"], st["nz"], st["seg_len"]
            oy, oz = st["oy"], st["oz"]
            for m in range(n_normal):
                t = s[m]
                b = t ** 0.6                       # near wall -> normal, far -> circle
                py_n = ring[:, 1] + g * t * seg * ny + (1 - g) * t * (oy - ring[:, 1])
                pz_n = ring[:, 2] + g * t * seg * nz + (1 - g) * t * (oz - ring[:, 2])
                py_r = ring[:, 1] + t * (oy - ring[:, 1])
                pz_r = ring[:, 2] + t * (oz - ring[:, 2])
                coords[:, m, j, 0] = st["cx"]
                coords[:, m, j, 1] = (1.0 - b) * py_n + b * py_r
                coords[:, m, j, 2] = (1.0 - b) * pz_n + b * pz_r
        return coords

    # Adaptive: use the most wall-normal (orthogonal) extrusion that stays valid.
    from .block import _hex_cell_volumes

    coords = None
    used_g = 0.0
    for g in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0):
        trial = place(g)
        v = _hex_cell_volumes(trial)
        if np.median(v) < 0:
            trial = trial[::-1].copy()
            v = _hex_cell_volumes(trial)
        if (v > 0).all():
            coords, used_g = trial, g
            break
    if coords is None:
        coords = _orient_positive(place(0.0))     # radial fallback (always valid)

    # Optional state-of-the-art refinement: elliptic (Winslow) smoothing.
    if domain.smoothing_iters > 0:
        from .elliptic import smooth_crossplanes

        coords = smooth_crossplanes(
            coords, first_cell,
            n_iter=domain.smoothing_iters, omega=domain.smoothing_omega,
        )

    coords = _orient_positive(coords)
    mesh = _split_blocks(coords, domain, surface)
    mesh.meta["wall_normal_weight"] = used_g
    return mesh


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
