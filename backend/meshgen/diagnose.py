"""Diagnose an imported STEP body for the mesher's assumptions.

The re-gridder assumes a slender body whose **length runs along X**, whose
**up direction is Z**, and which is a **closed solid** (has both an upper and a
lower skin).  When a real vehicle violates one of these, surfaces come out
wrong (e.g. a missing bottom).  This tool reports what the importer actually
sees so the fix is obvious.

Usage:
    python -m meshgen.diagnose path/to/body.step
"""
from __future__ import annotations

import sys

import numpy as np

from .geometry.step_io import load_step

_AX = {0: "X", 1: "Y", 2: "Z"}


def diagnose(path: str) -> None:
    print(f"Loading STEP: {path}")
    model = load_step(path)
    v = model.vertices
    n = model.normals
    faces = model.faces
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    ext = hi - lo

    print("\n== Bounding box ==")
    for a in range(3):
        print(f"  {_AX[a]}: {lo[a]:+.4g} .. {hi[a]:+.4g}   (extent {ext[a]:.4g})")
    print(f"  triangles: {faces.shape[0]}")

    length_ax = int(np.argmax(ext))
    up_ax = int(np.argmin(ext))
    print("\n== Orientation (inferred from extents) ==")
    print(f"  longest axis  (length)   : {_AX[length_ax]}  (extent {ext[length_ax]:.4g})")
    print(f"  shortest axis (thickness): {_AX[up_ax]}  (extent {ext[up_ax]:.4g})")
    if length_ax != 0:
        print(f"  !! Mesher expects LENGTH along X, but it looks like {_AX[length_ax]}.")
        print(f"     -> rotate the body so nose-to-tail is +X (or ask for an orientation option).")
    if up_ax != 2:
        print(f"  !! Mesher expects UP along Z, but thickness is along {_AX[up_ax]}.")
        print(f"     -> rotate so the thin (up) direction is Z.")
    if length_ax == 0 and up_ax == 2:
        print("  OK: length=X, up=Z  (matches the mesher's assumptions).")

    # Solid vs shell: distribution of facet normals along the up axis.
    nu = n[:, up_ax]
    up_faces = int((nu > 0.3).sum())
    dn_faces = int((nu < -0.3).sum())
    side_faces = int((np.abs(nu) <= 0.3).sum())
    print("\n== Solid vs shell (facet normals along the up axis) ==")
    print(f"  up-facing   : {up_faces}")
    print(f"  down-facing : {dn_faces}")
    print(f"  side/vertical: {side_faces}")
    if dn_faces == 0 or up_faces == 0:
        missing = "bottom" if dn_faces == 0 else "top"
        print(f"  !! No {missing} skin found -> this looks like an OPEN SHELL missing the {missing}.")
        print("     -> re-export as a CLOSED SOLID (watertight), or export both skins.")
    else:
        print("  OK: both top and bottom skins present (looks like a closed solid).")

    # Actual re-grid result: thickness the mesher would produce.
    try:
        bs = model.to_body_surface(41, 31)
        th = bs.upper[:, :, 2] - bs.lower[:, :, 2]
        collapsed = float((th < 1e-6 * max(ext.max(), 1e-9)).mean()) * 100.0
        print("\n== Re-gridded lens (what the mesher builds) ==")
        print(f"  thickness: min {th.min():.4g}  mean {th.mean():.4g}  max {th.max():.4g}")
        print(f"  ~zero-thickness nodes: {collapsed:.1f}%")
        print(f"  blunt tips detected: {bs.meta.get('blunt_tips')}")
        if th.mean() < 1e-3 * ext.max():
            print("  !! Nearly zero thickness -> upper and lower collapsed (missing bottom).")
    except Exception as exc:  # pragma: no cover
        print(f"\n  re-grid failed: {exc}")

    print("\nShare this output and I can pinpoint the fix.")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m meshgen.diagnose path/to/body.step")
        return 2
    diagnose(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
