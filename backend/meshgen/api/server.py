"""FastAPI application exposing the waverider mesh-generation workflow.

Endpoints mirror the left-to-right UI stages:
    geometry (parametric | STEP)  ->  flow  ->  mesh  ->  export
State for each browser session is held in an in-memory store keyed by a
server-issued ``session_id``.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..domain.farfield import DomainParams
from ..flow.conditions import FlowConditions
from ..geometry.surface import BodySurface
from ..geometry.waverider import WaveriderParams, generate_waverider
from ..mesh.topology import build_multiblock
from ..pipeline import export_mesh
from .schemas import DomainRequest, ExportRequest, FlowRequest, WaveriderRequest
from .viz import mesh_payload, surface_payload

app = FastAPI(title="Waverider MeshGen", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id -> {"surface": BodySurface, "mesh": MultiBlockMesh, "flow": ...}
_SESSIONS: dict[str, dict] = {}
_EXPORT_DIR = Path(tempfile.gettempdir()) / "meshgen_exports"
_EXPORT_DIR.mkdir(exist_ok=True)


def _session(sid: str) -> dict:
    if sid not in _SESSIONS:
        raise HTTPException(404, f"Unknown session {sid}")
    return _SESSIONS[sid]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "sessions": len(_SESSIONS)}


@app.post("/api/geometry/parametric")
def geometry_parametric(req: WaveriderRequest) -> dict:
    params = WaveriderParams(
        mach=req.mach, shock_angle_deg=req.shock_angle_deg, length=req.length,
        height_ratio=req.height_ratio, width_trim=req.width_trim,
        gamma=req.gamma, n_span=req.n_span, n_stream=req.n_stream,
    )
    try:
        geom = generate_waverider(params)
    except Exception as exc:  # infeasible design point
        raise HTTPException(400, str(exc))
    surface = BodySurface.from_waverider(geom)
    sid = uuid.uuid4().hex[:12]
    _SESSIONS[sid] = {"surface": surface, "info": geom.as_dict()}
    return {"session_id": sid, "info": geom.as_dict(), "geometry": surface_payload(surface)}


@app.post("/api/geometry/step")
async def geometry_step(
    file: UploadFile = File(...),
    n_span: int = Form(61),
    n_stream: int = Form(41),
) -> dict:
    from ..geometry.step_io import load_step

    suffix = Path(file.filename or "body.step").suffix or ".step"
    tmp = _EXPORT_DIR / f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    tmp.write_bytes(await file.read())
    try:
        model = load_step(str(tmp))
    except Exception as exc:
        raise HTTPException(400, f"STEP import failed: {exc}")
    finally:
        pass
    surface = model.to_body_surface(n_span, n_stream)
    sid = uuid.uuid4().hex[:12]
    info = {
        "source": "step", "filename": file.filename,
        "reference_length": model.reference_length(),
        "n_triangles": int(model.faces.shape[0]),
    }
    _SESSIONS[sid] = {"surface": surface, "info": info, "step_model": model}
    return {
        "session_id": sid,
        "info": info,
        "geometry": surface_payload(surface),
        "raw_step": model.triangle_soup(),
    }


@app.post("/api/flow")
def flow(req: FlowRequest) -> dict:
    fc = FlowConditions.from_altitude(
        req.mach, req.altitude_m, req.reference_length, req.gamma
    )
    return fc.summary()


@app.post("/api/mesh")
def mesh(req: DomainRequest) -> dict:
    sess = _session(req.session_id)
    surface: BodySurface = sess["surface"]

    lo, hi = surface.bbox()
    body_length = float(hi[0] - lo[0])
    flow = None
    if req.use_flow:
        m = req.mach if req.mach is not None else sess["info"].get("mach", 6.0)
        flow = FlowConditions.from_altitude(m, req.altitude_m, body_length)

    dom = DomainParams(
        n_normal=req.n_normal, farfield_radius_factor=req.farfield_radius_factor,
        y_plus=req.y_plus, first_cell_height=req.first_cell_height,
        n_stream_blocks=req.n_stream_blocks, n_wrap_blocks=req.n_wrap_blocks,
        n_wrap_mult=req.n_wrap_mult,
    )
    try:
        mb = build_multiblock(surface, dom, flow)
    except Exception as exc:
        raise HTTPException(400, f"Meshing failed: {exc}")
    sess["mesh"] = mb
    sess["flow"] = flow

    q = mb.quality().as_dict()
    payload = mesh_payload(mb)
    resp = {"quality": q, "mesh": payload, "topology": mb.meta}
    if flow is not None:
        resp["first_cell_height"] = flow.first_cell_height(req.y_plus)
    return resp


@app.post("/api/export")
def export(req: ExportRequest) -> FileResponse:
    sess = _session(req.session_id)
    if "mesh" not in sess:
        raise HTTPException(400, "No mesh generated yet for this session")
    ext = {"su2": "su2", "cgns": "cgns", "plot3d": "xyz"}[req.format]
    out = _EXPORT_DIR / f"waverider_{req.session_id}.{ext}"
    export_mesh(sess["mesh"], str(out), req.format)
    return FileResponse(
        str(out), filename=f"waverider_mesh.{ext}", media_type="application/octet-stream"
    )


# Serve the built front-end (if present) so the whole app runs from one process.
_FRONTEND_DIST = Path(__file__).resolve().parents[2].parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
