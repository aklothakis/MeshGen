"""Pydantic request/response models for the web API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class WaveriderRequest(BaseModel):
    mach: float = Field(6.0, gt=1.0)
    shock_angle_deg: float = Field(14.0, gt=0.0, lt=90.0)
    length: float = Field(2.0, gt=0.0)
    height_ratio: float = Field(0.35, gt=0.0, lt=1.0)
    width_trim: float = Field(0.98, gt=0.1, le=1.0)
    n_span: int = Field(61, ge=11, le=301)
    n_stream: int = Field(41, ge=5, le=201)
    gamma: float = Field(1.4, gt=1.0, lt=2.0)


class FlowRequest(BaseModel):
    mach: float = Field(6.0, gt=1.0)
    altitude_m: float = Field(30000.0, ge=0.0, le=86000.0)
    reference_length: float = Field(2.0, gt=0.0)
    gamma: float = Field(1.4, gt=1.0, lt=2.0)


class DomainRequest(BaseModel):
    session_id: str
    n_normal: int = Field(48, ge=8, le=256)
    farfield_radius_factor: float = Field(6.0, ge=1.5, le=50.0)
    y_plus: float = Field(1.0, gt=0.0)
    first_cell_height: float | None = None
    n_stream_blocks: int = Field(2, ge=1, le=16)
    n_wrap_blocks: int = Field(2, ge=1, le=16)
    n_wrap_mult: int = Field(1, ge=1, le=8)
    smoothing_iters: int = Field(0, ge=0, le=1000)
    use_flow: bool = True
    mach: float | None = None
    altitude_m: float = 30000.0


class ExportRequest(BaseModel):
    session_id: str
    format: str = Field("su2", pattern="^(su2|cgns|plot3d)$")
