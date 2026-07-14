"""Command-line interface for the waverider mesh generator.

Examples
--------
    # Parametric waverider -> SU2 + CGNS
    python -m meshgen.cli --mach 6 --shock-angle 14 --length 2 \
        --altitude 30000 --out mesh --formats su2,cgns

    # Import a STEP body and mesh it
    python -m meshgen.cli --step body.step --out mesh --formats su2,plot3d
"""
from __future__ import annotations

import argparse
import json
import sys

from .domain.farfield import DomainParams
from .geometry.waverider import WaveriderParams
from .pipeline import PipelineConfig, export_mesh, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Automatic waverider multiblock mesh generator")
    g = p.add_argument_group("geometry")
    g.add_argument("--step", help="STEP file to import (else parametric waverider)")
    g.add_argument("--mach", type=float, default=6.0)
    g.add_argument("--shock-angle", type=float, default=14.0, help="deg")
    g.add_argument("--length", type=float, default=2.0)
    g.add_argument("--height-ratio", type=float, default=0.35)
    g.add_argument("--n-span", type=int, default=61)
    g.add_argument("--n-stream", type=int, default=41)

    fl = p.add_argument_group("flow")
    fl.add_argument("--altitude", type=float, default=30000.0, help="m")
    fl.add_argument("--no-flow", action="store_true", help="ignore flow-based wall spacing")

    d = p.add_argument_group("domain / mesh")
    d.add_argument("--n-normal", type=int, default=48)
    d.add_argument("--farfield", type=float, default=6.0, help="farfield radius factor")
    d.add_argument("--y-plus", type=float, default=1.0)
    d.add_argument("--stream-blocks", type=int, default=2)
    d.add_argument("--wrap-blocks", type=int, default=2)

    o = p.add_argument_group("output")
    o.add_argument("--out", default="mesh", help="output basename")
    o.add_argument("--formats", default="su2,cgns", help="comma list: su2,cgns,plot3d")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = PipelineConfig(
        geometry_source="step" if args.step else "parametric",
        step_path=args.step,
        step_n_span=args.n_span,
        step_n_stream=args.n_stream,
        waverider=WaveriderParams(
            mach=args.mach,
            shock_angle_deg=args.shock_angle,
            length=args.length,
            height_ratio=args.height_ratio,
            n_span=args.n_span,
            n_stream=args.n_stream,
        ),
        mach=args.mach,
        altitude_m=args.altitude,
        use_flow=not args.no_flow,
        domain=DomainParams(
            n_normal=args.n_normal,
            farfield_radius_factor=args.farfield,
            y_plus=args.y_plus,
            n_stream_blocks=args.stream_blocks,
            n_wrap_blocks=args.wrap_blocks,
        ),
    )

    print("Running pipeline ...", file=sys.stderr)
    result = run_pipeline(cfg)
    print(json.dumps({"geometry": result.geometry_info,
                      "quality": result.quality}, indent=2, default=float))
    if result.flow is not None:
        print("Flow:", json.dumps(result.flow.summary(), indent=2, default=float))

    for fmt in [x.strip() for x in args.formats.split(",") if x.strip()]:
        ext = {"su2": "su2", "cgns": "cgns", "plot3d": "xyz"}.get(fmt, fmt)
        path = f"{args.out}.{ext}"
        export_mesh(result.mesh, path, fmt)
        print(f"wrote {path}", file=sys.stderr)

    if not result.quality["valid"]:
        print("WARNING: mesh has non-positive cells", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
