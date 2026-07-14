"""Regression tests for the mesh-generation pipeline.

Run: ``cd backend && python -m pytest -q``
"""
import math

import numpy as np
import pytest

from meshgen.flow.oblique_shock import (
    beta_from_theta, theta_from_beta, oblique_shock, max_deflection,
)
from meshgen.flow.taylor_maccoll import solve_taylor_maccoll
from meshgen.flow.conditions import FlowConditions, standard_atmosphere
from meshgen.geometry.waverider import WaveriderParams, generate_waverider
from meshgen.geometry.surface import BodySurface
from meshgen.mesh.clustering import distribute_wall_normal
from meshgen.domain.farfield import DomainParams
from meshgen.mesh.topology import build_multiblock
from meshgen.io.assembler import assemble


# ---- aerodynamics (validated vs textbook references) --------------------
def test_oblique_shock_anderson():
    # M=2, theta=20 deg -> weak beta ~53.4 deg, strong ~74.3 deg.
    b_weak = beta_from_theta(2.0, math.radians(20.0), weak=True)
    b_strong = beta_from_theta(2.0, math.radians(20.0), weak=False)
    assert math.degrees(b_weak) == pytest.approx(53.42, abs=0.1)
    assert math.degrees(b_strong) == pytest.approx(74.27, abs=0.1)
    # Round trip.
    assert math.degrees(theta_from_beta(2.0, b_weak)) == pytest.approx(20.0, abs=1e-3)


def test_max_deflection():
    tmax, _ = max_deflection(2.0)
    assert math.degrees(tmax) == pytest.approx(22.97, abs=0.1)


def test_detached_shock_raises():
    with pytest.raises(ValueError):
        beta_from_theta(2.0, math.radians(30.0))  # exceeds theta_max


def test_taylor_maccoll_cone_naca1135():
    # M=2, conical shock 37.8 deg -> cone ~20 deg, surface Mach ~1.57.
    f = solve_taylor_maccoll(2.0, math.radians(37.8))
    assert f.theta_c_deg == pytest.approx(20.0, abs=0.3)
    assert f.mach_c == pytest.approx(1.57, abs=0.02)


def test_standard_atmosphere():
    p, t, rho = standard_atmosphere(11000.0)
    assert t == pytest.approx(216.65, abs=0.5)
    assert p == pytest.approx(22632.0, rel=0.02)


def test_first_cell_height_positive():
    fc = FlowConditions.from_altitude(8.0, 30000.0, 2.0)
    y1 = fc.first_cell_height(1.0)
    assert 0 < y1 < 1e-3
    assert fc.reynolds > 0


# ---- clustering ---------------------------------------------------------
def test_wall_normal_first_cell():
    d = distribute_wall_normal(40, total_height=1.0, first_cell=1e-5)
    assert (d[1] - d[0]) == pytest.approx(1e-5, rel=0.05)
    assert np.all(np.diff(d) > 0)
    assert d[-1] == pytest.approx(1.0)


# ---- geometry -----------------------------------------------------------
def test_waverider_leading_edge_on_shock():
    p = WaveriderParams(mach=6.0, shock_angle_deg=14.0, length=2.0, n_span=41, n_stream=31)
    g = generate_waverider(p)
    le = g.leading_edge
    rho = np.hypot(le[:, 1], le[:, 2])
    ratio = rho / le[:, 0]
    assert np.abs(ratio - math.tan(math.radians(14.0))).max() < 1e-9
    assert not np.isnan(g.upper).any() and not np.isnan(g.lower).any()
    assert g.volume > 0 and g.planform_area > 0


# ---- meshing ------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_mesh():
    g = generate_waverider(WaveriderParams(n_span=31, n_stream=21))
    bs = BodySurface.from_waverider(g)
    flow = FlowConditions.from_altitude(6.0, 30000.0, 2.0)
    return build_multiblock(bs, DomainParams(n_normal=24, n_stream_blocks=2, n_wrap_blocks=2), flow)


def test_mesh_positive_volumes(sample_mesh):
    q = sample_mesh.quality()
    assert q.n_negative_cells == 0
    assert q.min_cell_volume > 0
    assert q.is_valid()
    assert q.n_blocks == 4


def test_assemble_watertight(sample_mesh):
    um = assemble(sample_mesh)
    # Shared nodes were merged, so unique < raw block-node total.
    assert um.n_points < sample_mesh.n_nodes
    assert um.hexes.max() < um.n_points
    assert um.hexes.min() >= 0
    for tag in ("wall", "farfield"):
        assert tag in um.markers
        assert um.markers[tag].max() < um.n_points


def test_su2_roundtrip(tmp_path, sample_mesh):
    import meshio
    from meshgen.io.su2_writer import write_su2

    out = str(tmp_path / "m.su2")
    write_su2(sample_mesh, out)
    m = meshio.read(out)
    assert any(c.type == "hexahedron" for c in m.cells)


def test_cgns_structured(tmp_path, sample_mesh):
    import h5py
    from meshgen.io.cgns_writer import write_cgns

    out = str(tmp_path / "m.cgns")
    write_cgns(sample_mesh, out)
    with h5py.File(out, "r") as f:
        zones = [k for k in f["Base"].keys() if k.startswith("Zone")]
        assert len(zones) == 4
        z = f["Base"][zones[0]]
        ztype = b"".join(np.asarray(z["ZoneType"][" data"][:]).astype("S1").tolist())
        assert ztype == b"Structured"
