"""End-to-end mesh-generation pipeline.

Ties the workflow stages together -- geometry (parametric or STEP) -> flow
conditions -> domain/mesh -> export -- behind one config object, shared by the
CLI and the web API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .domain.farfield import DomainParams
from .flow.conditions import FlowConditions
from .geometry.surface import BodySurface
from .geometry.waverider import WaveriderParams, generate_waverider
from .mesh.multiblock import MultiBlockMesh
from .mesh.topology import build_multiblock

__all__ = ["PipelineConfig", "PipelineResult", "run_pipeline", "export_mesh"]


@dataclass
class PipelineConfig:
    geometry_source: Literal["parametric", "step"] = "parametric"
    waverider: WaveriderParams = field(default_factory=WaveriderParams)
    step_path: str | None = None
    step_n_span: int = 61
    step_n_stream: int = 41

    # Flow conditions (optional; drives wall spacing).
    mach: float | None = None            # defaults to waverider.mach
    altitude_m: float = 30000.0
    use_flow: bool = True

    domain: DomainParams = field(default_factory=DomainParams)


@dataclass
class PipelineResult:
    surface: BodySurface
    mesh: MultiBlockMesh
    flow: FlowConditions | None
    geometry_info: dict
    quality: dict


def _build_surface(cfg: PipelineConfig) -> tuple[BodySurface, dict]:
    if cfg.geometry_source == "step":
        if not cfg.step_path:
            raise ValueError("step_path required for STEP geometry source")
        from .geometry.step_io import load_step

        model = load_step(cfg.step_path)
        surface = model.to_body_surface(cfg.step_n_span, cfg.step_n_stream)
        info = {
            "source": "step",
            "path": cfg.step_path,
            "reference_length": model.reference_length(),
            "n_triangles": int(model.faces.shape[0]),
        }
        return surface, info

    geom = generate_waverider(cfg.waverider)
    return BodySurface.from_waverider(geom), {"source": "parametric", **geom.as_dict()}


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    surface, info = _build_surface(cfg)

    lo, hi = surface.bbox()
    body_length = float(hi[0] - lo[0])

    flow = None
    if cfg.use_flow:
        mach = cfg.mach if cfg.mach is not None else cfg.waverider.mach
        flow = FlowConditions.from_altitude(mach, cfg.altitude_m, body_length)

    mesh = build_multiblock(surface, cfg.domain, flow)
    quality = mesh.quality().as_dict()

    return PipelineResult(
        surface=surface,
        mesh=mesh,
        flow=flow,
        geometry_info=info,
        quality=quality,
    )


def export_mesh(mesh: MultiBlockMesh, path: str, fmt: str) -> str:
    """Export a mesh to ``fmt`` in {"su2", "cgns", "plot3d"}."""
    fmt = fmt.lower()
    if fmt == "su2":
        from .io.su2_writer import write_su2

        return write_su2(mesh, path)
    if fmt == "cgns":
        from .io.cgns_writer import write_cgns

        return write_cgns(mesh, path)
    if fmt in ("plot3d", "p3d", "xyz"):
        from .io.plot3d_writer import write_plot3d

        return write_plot3d(mesh, path)
    raise ValueError(f"Unknown export format: {fmt}")
