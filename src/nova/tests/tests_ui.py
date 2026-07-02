"""
tests/unit/test_ui.py
======================
Unit tests for Phase 13 UI Glass Cockpit modules:
  nova.ui.pfd          — PrimaryFlightDisplay, PFDState, NavballVector,
                         SpeedTapeData, AltitudeTapeData
  nova.ui.orbital_deck — OrbitalDeck, OrbitalDeckState, HohmannBudget,
                         compute_hohmann_budget
  nova.ui.avionics     — AvionicsPanel, AvionicsState, EngineDisplayData,
                         AngularRateData, StructuralHealthData, AlertSummary
  nova.ui.hud          — HUDCompositor, HUDConfig, HUDFrame

Coverage
--------
PFD (45 tests):
  NavballVector: construction, frozen, shape validation
  SpeedTapeData / AltitudeTapeData: construction, frozen, field values
  PFDState: construction, frozen, deg/rad consistency
  PrimaryFlightDisplay: build from RenderFrame, identity quaternion,
    navball vector count/directions, altitude tape target, type error

OrbitalDeck (42 tests):
  HohmannBudget: construction, frozen, repr
  compute_hohmann_budget: energy, direction, circular approx, invalid inputs
  OrbitalDeckState: construction, frozen, km conversions, deg conversions
  OrbitalDeck: build from circular orbit frame, closed/open orbit flags,
    suborbital flag, manoeuvre node set/clear, type error

Avionics (40 tests):
  EngineDisplayData: construction, frozen, repr
  AngularRateData: construction, frozen, values
  StructuralHealthData: construction, frozen, labels
  AlertSummary: grouping, master_warning/caution, any_active, all_active
  AvionicsPanel: build from snapshot, engine status logic, alert propagation,
    zero alerts, wrong snapshot type

HUD (33 tests):
  HUDConfig: construction, frozen, defaults, bad alert_max
  HUDFrame: construction, frozen, properties
  HUDCompositor: tick with data, tick without data (empty registry),
    multiple ticks, reset, type guards, show_* flags disabling panels
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from nova.ai.monitor import AlertMessage, AlertSeverity, AlertCategory, MonitorConfig
from nova.core.pipeline import ControlInput, PipelineConfig, SimulationPipeline
from nova.core.telemetry_registry import TelemetryRegistry, TelemetrySnapshot
from nova.core.state_vector import make_state, identity_state
from nova.frames.transforms import euler_to_quaternion
from nova.rendering.celestial import CelestialRenderer
from nova.rendering.vehicle_render import VehicleRenderer, default_rocket_config
from nova.rendering.viewport import RenderFrame, Viewport, ViewportConfig
from nova.ui.avionics import (
    AlertSummary,
    AngularRateData,
    AvionicsPanel,
    AvionicsState,
    EngineDisplayData,
    EngineStatus,
    StructuralHealthData,
    _classify_engine,
    _group_alerts,
    _structural_label,
)
from nova.ui.hud import HUDCompositor, HUDConfig, HUDFrame
from nova.ui.orbital_deck import (
    HohmannBudget,
    OrbitalDeck,
    OrbitalDeckState,
    compute_hohmann_budget,
)
from nova.ui.pfd import (
    AltitudeTapeData,
    NavballVector,
    PFDState,
    PrimaryFlightDisplay,
    SpeedTapeData,
    _compute_navball_vectors,
    _navball_angles,
    _unit,
)
from nova.vehicle.component_graph import ComponentGraph, ComponentNode
from nova.vehicle.mass_model import point_mass


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def leo_render_frame() -> RenderFrame:
    """Circular 400 km LEO RenderFrame with identity attitude."""
    from nova.core.constants import EARTH_RADIUS_EQ, EARTH_MU
    r = EARTH_RADIUS_EQ + 400_000.0
    v = math.sqrt(EARTH_MU / r)
    a = r  # circular: semi-major axis = radius
    # For circular orbit e=0 → Ap=Pe=altitude
    alt = 400_000.0
    return RenderFrame(
        position_eci=np.array([r, 0.0, 0.0]),
        velocity_eci=np.array([0.0, v, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body=np.array([0.01, 0.005, -0.003]),
        mass=5000.0,
        mission_time=100.0,
        altitude=alt,
        speed=v,
        mach=0.0,
        throttle=0.8,
        thrust_magnitude=50_000.0,
        alpha=0.05,
        dynamic_pressure=1000.0,
        semi_major_axis=a,
        eccentricity=0.0,
        inclination=math.radians(51.6),
        apoapsis=alt,
        periapsis=alt,
        any_structural_failure=False,
        alpha_blend=0.5,
        earlier_snap_time=99.9,
        later_snap_time=100.1,
    )


@pytest.fixture
def elliptic_render_frame() -> RenderFrame:
    """Elliptic orbit: 200 km periapsis, 400 km apoapsis."""
    from nova.core.constants import EARTH_RADIUS_EQ, EARTH_MU
    r_pe = EARTH_RADIUS_EQ + 200_000.0
    r_ap = EARTH_RADIUS_EQ + 400_000.0
    a = 0.5 * (r_pe + r_ap)
    e = (r_ap - r_pe) / (r_ap + r_pe)
    v_pe = math.sqrt(EARTH_MU * (2.0 / r_pe - 1.0 / a))
    return RenderFrame(
        position_eci=np.array([r_pe, 0.0, 0.0]),
        velocity_eci=np.array([0.0, v_pe, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body=np.zeros(3),
        mass=5000.0,
        mission_time=200.0,
        altitude=200_000.0,
        speed=v_pe,
        mach=0.0,
        throttle=0.0,
        thrust_magnitude=0.0,
        alpha=0.0,
        dynamic_pressure=0.0,
        semi_major_axis=a,
        eccentricity=e,
        inclination=math.radians(28.5),
        apoapsis=400_000.0,
        periapsis=200_000.0,
        any_structural_failure=False,
        alpha_blend=0.0,
        earlier_snap_time=199.9,
        later_snap_time=200.1,
    )


@pytest.fixture
def pipeline_with_registry():
    """Run a 5-tick simulation pipeline and return (registry, latest_snapshot)."""
    mc = point_mass('body', 5000.0, np.zeros(3))
    node = ComponentNode(
        node_id='body', display_name='Body',
        mass_component=mc, component_type='body'
    )
    graph = ComponentGraph()
    graph.add_node(node)
    reg = TelemetryRegistry(buffer_size=50)
    cfg = PipelineConfig(
        dt=0.1, enable_aerodynamics=False, enable_j2=False,
        enable_structural=False, gravity_bodies=[]
    )
    state = make_state(
        position_eci=np.array([6_771_000., 0., 0.]),
        velocity_eci=np.array([0., 7_800., 0.]),
        quaternion=np.array([1., 0., 0., 0.]),
        omega_body=np.array([0.01, 0.005, -0.003]),
        mass=5000., time=0.
    )
    pipe = SimulationPipeline(cfg, state, graph, reg, propellant_mass=1000.)
    for _ in range(5):
        pipe.tick(ControlInput(throttle=0.8))
    return reg, reg.latest


@pytest.fixture
def alert_critical() -> AlertMessage:
    return AlertMessage(
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.STRUCTURAL,
        message="Structural failure imminent",
        recommendation="Reduce dynamic pressure",
        time_to_limit_s=5.0,
        mission_time=10.0,
        parameter="structural_margin",
        current_value=0.05,
        rate=-0.01,
        limit=0.10,
    )


@pytest.fixture
def alert_warning() -> AlertMessage:
    return AlertMessage(
        severity=AlertSeverity.WARNING,
        category=AlertCategory.AERODYNAMIC,
        message="High angle of attack",
        recommendation="Reduce AOA",
        time_to_limit_s=30.0,
        mission_time=10.0,
        parameter="alpha",
        current_value=0.25,
        rate=0.002,
        limit=0.26,
    )


@pytest.fixture
def alert_info() -> AlertMessage:
    return AlertMessage(
        severity=AlertSeverity.INFO,
        category=AlertCategory.INFO,
        message="Max-Q reached",
        recommendation="Monitor dynamic pressure",
        time_to_limit_s=float("inf"),
        mission_time=10.0,
        parameter="dynamic_pressure",
        current_value=45000.0,
        rate=0.0,
        limit=50000.0,
    )


# ===========================================================================
# NavballVector tests
# ===========================================================================

class TestNavballVector:

    def test_valid_construction(self):
        v = NavballVector(
            label="PRO",
            direction_body=np.array([1.0, 0.0, 0.0]),
            color=(255, 255, 0),
            is_visible=True,
            azimuth_rad=0.0,
            elevation_rad=math.pi / 2.0,
        )
        assert v.label == "PRO"
        assert v.direction_body.shape == (3,)
        assert v.is_visible is True

    def test_frozen(self):
        v = NavballVector("PRO", np.array([1., 0., 0.]), (255, 255, 0), True, 0.0, 0.0)
        with pytest.raises(Exception):
            v.label = "RET"

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="direction_body"):
            NavballVector("PRO", np.array([1., 0.]), (255, 255, 0), True, 0.0, 0.0)

    def test_dtype_float64(self):
        v = NavballVector("PRO", [1.0, 0.0, 0.0], (255, 255, 0), True, 0.0, 0.0)
        assert v.direction_body.dtype == np.float64


class TestNavballAngles:

    def test_forward_direction_elevation_90(self):
        """+X body = forward = navball north pole (elevation = π/2)."""
        _, el, vis = _navball_angles(np.array([1.0, 0.0, 0.0]))
        assert el == pytest.approx(math.pi / 2.0, abs=1e-9)
        assert vis is True

    def test_backward_direction_not_visible(self):
        """−X body = backward = not visible."""
        _, _, vis = _navball_angles(np.array([-1.0, 0.0, 0.0]))
        assert vis is False

    def test_right_direction_visible(self):
        """+Y body = right = on equator (elevation ≈ 0)."""
        az, el, vis = _navball_angles(np.array([0.0, 1.0, 0.0]))
        assert abs(el) < 1e-9
        assert vis is False  # x=0, not strictly > 0


class TestComputeNavballVectors:

    def test_six_vectors_for_leo(self, leo_render_frame):
        vecs = _compute_navball_vectors(leo_render_frame)
        assert len(vecs) == 6

    def test_labels_present(self, leo_render_frame):
        vecs = _compute_navball_vectors(leo_render_frame)
        labels = {v.label for v in vecs}
        assert "PRO" in labels
        assert "RET" in labels
        assert "NRM" in labels

    def test_pro_ret_opposite(self, leo_render_frame):
        """Prograde and retrograde must be exact opposites."""
        vecs = _compute_navball_vectors(leo_render_frame)
        pro = next(v for v in vecs if v.label == "PRO")
        ret = next(v for v in vecs if v.label == "RET")
        dot = float(np.dot(pro.direction_body, ret.direction_body))
        assert dot == pytest.approx(-1.0, abs=1e-6)

    def test_nrm_anm_opposite(self, leo_render_frame):
        vecs = _compute_navball_vectors(leo_render_frame)
        nrm = next(v for v in vecs if v.label == "NRM")
        anm = next(v for v in vecs if v.label == "ANM")
        dot = float(np.dot(nrm.direction_body, anm.direction_body))
        assert dot == pytest.approx(-1.0, abs=1e-6)

    def test_all_directions_unit_length(self, leo_render_frame):
        vecs = _compute_navball_vectors(leo_render_frame)
        for v in vecs:
            n = float(np.linalg.norm(v.direction_body))
            assert abs(n - 1.0) < 1e-9


# ===========================================================================
# SpeedTapeData / AltitudeTapeData tests
# ===========================================================================

class TestSpeedTapeData:

    def test_construction(self):
        s = SpeedTapeData(7800.0, 7750.0, 50.0, 0.0)
        assert s.orbital_speed_m_s == pytest.approx(7800.0)
        assert s.mach == pytest.approx(0.0)

    def test_frozen(self):
        s = SpeedTapeData(7800.0, 7750.0, 50.0, 0.0)
        with pytest.raises(Exception):
            s.orbital_speed_m_s = 0.0


class TestAltitudeTapeData:

    def test_with_target(self):
        a = AltitudeTapeData(400_000.0, 50.0, 500_000.0)
        assert a.target_altitude_m == pytest.approx(500_000.0)

    def test_without_target(self):
        a = AltitudeTapeData(400_000.0, 50.0, None)
        assert a.target_altitude_m is None

    def test_frozen(self):
        a = AltitudeTapeData(400_000.0, 0.0, None)
        with pytest.raises(Exception):
            a.altitude_m = 0.0


# ===========================================================================
# PFDState tests
# ===========================================================================

class TestPFDState:

    @pytest.fixture
    def sample_pfd(self, leo_render_frame) -> PFDState:
        return PrimaryFlightDisplay().build(leo_render_frame)

    def test_construction(self, sample_pfd):
        assert isinstance(sample_pfd, PFDState)
        assert sample_pfd.quaternion.shape == (4,)

    def test_frozen(self, sample_pfd):
        with pytest.raises(Exception):
            sample_pfd.roll_rad = 1.0

    def test_deg_rad_consistency_roll(self, sample_pfd):
        assert sample_pfd.roll_deg == pytest.approx(
            math.degrees(sample_pfd.roll_rad), abs=1e-9
        )

    def test_deg_rad_consistency_pitch(self, sample_pfd):
        assert sample_pfd.pitch_deg == pytest.approx(
            math.degrees(sample_pfd.pitch_rad), abs=1e-9
        )

    def test_deg_rad_consistency_yaw(self, sample_pfd):
        assert sample_pfd.yaw_deg == pytest.approx(
            math.degrees(sample_pfd.yaw_rad), abs=1e-9
        )

    def test_identity_quaternion_zero_euler(self, leo_render_frame):
        """Identity quaternion → zero roll, pitch, yaw."""
        pfd = PrimaryFlightDisplay().build(leo_render_frame)
        assert abs(pfd.roll_deg) < 1e-6
        assert abs(pfd.pitch_deg) < 1e-6
        assert abs(pfd.yaw_deg) < 1e-6

    def test_repr(self, sample_pfd):
        r = repr(sample_pfd)
        assert "PFDState" in r

    def test_mission_time_propagated(self, sample_pfd, leo_render_frame):
        assert sample_pfd.mission_time == pytest.approx(
            leo_render_frame.mission_time
        )

    def test_altitude_propagated(self, sample_pfd, leo_render_frame):
        assert sample_pfd.altitude_tape.altitude_m == pytest.approx(
            leo_render_frame.altitude
        )

    def test_speed_propagated(self, sample_pfd, leo_render_frame):
        assert sample_pfd.speed_tape.orbital_speed_m_s == pytest.approx(
            leo_render_frame.speed
        )

    def test_throttle_propagated(self, sample_pfd, leo_render_frame):
        assert sample_pfd.throttle == pytest.approx(leo_render_frame.throttle)

    def test_thrust_propagated(self, sample_pfd, leo_render_frame):
        assert sample_pfd.thrust_n == pytest.approx(
            leo_render_frame.thrust_magnitude
        )


# ===========================================================================
# PrimaryFlightDisplay tests
# ===========================================================================

class TestPrimaryFlightDisplay:

    def test_default_construction(self):
        pfd = PrimaryFlightDisplay()
        assert pfd.target_altitude_m is None

    def test_with_target_altitude(self):
        pfd = PrimaryFlightDisplay(target_altitude_m=500_000.0)
        assert pfd.target_altitude_m == pytest.approx(500_000.0)

    def test_target_altitude_setter(self):
        pfd = PrimaryFlightDisplay()
        pfd.target_altitude_m = 600_000.0
        assert pfd.target_altitude_m == pytest.approx(600_000.0)

    def test_clear_target_altitude(self):
        pfd = PrimaryFlightDisplay(target_altitude_m=500_000.0)
        pfd.target_altitude_m = None
        assert pfd.target_altitude_m is None

    def test_build_returns_pfd_state(self, leo_render_frame):
        pfd = PrimaryFlightDisplay()
        state = pfd.build(leo_render_frame)
        assert isinstance(state, PFDState)

    def test_build_wrong_type_raises(self):
        pfd = PrimaryFlightDisplay()
        with pytest.raises(TypeError):
            pfd.build("not_a_render_frame")

    def test_navball_has_six_vectors(self, leo_render_frame):
        pfd = PrimaryFlightDisplay()
        state = pfd.build(leo_render_frame)
        assert len(state.navball_vectors) == 6

    def test_non_identity_quaternion_nonzero_euler(self):
        """A 45° roll quaternion should give roll_deg ≈ 45."""
        from nova.core.constants import EARTH_RADIUS_EQ, EARTH_MU
        r = EARTH_RADIUS_EQ + 400_000.0
        v = math.sqrt(EARTH_MU / r)
        q = euler_to_quaternion(math.radians(45.0), 0.0, 0.0)
        frame = RenderFrame(
            position_eci=np.array([r, 0., 0.]),
            velocity_eci=np.array([0., v, 0.]),
            quaternion=q,
            omega_body=np.zeros(3),
            mass=1000.0,
            mission_time=0.0,
            altitude=400_000.0,
            speed=v,
            mach=0.0,
            throttle=0.0,
            thrust_magnitude=0.0,
            alpha=0.0,
            dynamic_pressure=0.0,
            semi_major_axis=r,
            eccentricity=0.0,
            inclination=0.0,
            apoapsis=400_000.0,
            periapsis=400_000.0,
            any_structural_failure=False,
            alpha_blend=0.0,
            earlier_snap_time=0.0,
            later_snap_time=0.0,
        )
        pfd = PrimaryFlightDisplay()
        state = pfd.build(frame)
        assert abs(state.roll_deg - 45.0) < 1e-3

    def test_structural_failure_propagated(self, leo_render_frame):
        import dataclasses
        frame_failed = dataclasses.replace(leo_render_frame, any_structural_failure=True)
        pfd = PrimaryFlightDisplay()
        state = pfd.build(frame_failed)
        assert state.any_structural_failure is True

    def test_repr(self):
        assert "PrimaryFlightDisplay" in repr(PrimaryFlightDisplay())


# ===========================================================================
# HohmannBudget tests
# ===========================================================================

class TestHohmannBudget:

    def test_valid_construction(self):
        h = HohmannBudget(200_000., 400_000., 300_000., 600_000.,
                          100., 80., 180., True)
        assert h.total_dv_m_s == pytest.approx(180.0)
        assert h.is_valid is True

    def test_frozen(self):
        h = HohmannBudget(200_000., 400_000., 300_000., 600_000.,
                          100., 80., 180., True)
        with pytest.raises(Exception):
            h.total_dv_m_s = 0.0

    def test_repr(self):
        h = HohmannBudget(200_000., 400_000., 300_000., 600_000.,
                          100., 80., 180., True)
        r = repr(h)
        assert "HohmannBudget" in r
        assert "180.0" in r


class TestComputeHohmannBudget:

    def test_circular_to_higher_circular(self):
        """LEO 400 km → 1000 km: both Δv positive."""
        bgt = compute_hohmann_budget(400_000., 400_000., 1000_000., 1000_000.)
        assert bgt.is_valid
        assert bgt.dv1_m_s > 0.0
        assert bgt.dv2_m_s > 0.0
        assert bgt.total_dv_m_s == pytest.approx(bgt.dv1_m_s + bgt.dv2_m_s)

    def test_same_orbit_near_zero_dv(self):
        """Transfer from orbit to itself: Δv ≈ 0."""
        bgt = compute_hohmann_budget(400_000., 400_000., 400_000., 400_000.)
        assert bgt.is_valid
        assert bgt.total_dv_m_s == pytest.approx(0.0, abs=1.0)

    def test_invalid_negative_periapsis(self):
        bgt = compute_hohmann_budget(-100., 400_000., 300_000., 600_000.)
        assert bgt.is_valid is False
        assert bgt.total_dv_m_s == pytest.approx(0.0)

    def test_known_leo_gto_approx(self):
        """LEO 400 km → GTO (200 km × 35786 km): Δv ≈ 2.5 km/s total."""
        bgt = compute_hohmann_budget(400_000., 400_000., 200_000., 35_786_000.)
        assert bgt.is_valid
        # Rough check: total Δv for this manoeuvre is 2–3 km/s
        assert 1500.0 < bgt.total_dv_m_s < 5000.0


# ===========================================================================
# OrbitalDeckState tests
# ===========================================================================

class TestOrbitalDeckState:

    @pytest.fixture
    def sample_orbital(self, leo_render_frame) -> OrbitalDeckState:
        return OrbitalDeck().build(leo_render_frame)

    def test_construction(self, sample_orbital):
        assert isinstance(sample_orbital, OrbitalDeckState)

    def test_frozen(self, sample_orbital):
        with pytest.raises(Exception):
            sample_orbital.apoapsis_km = 0.0

    def test_km_conversion(self, sample_orbital):
        assert sample_orbital.apoapsis_km == pytest.approx(
            sample_orbital.apoapsis_m / 1000.0
        )
        assert sample_orbital.periapsis_km == pytest.approx(
            sample_orbital.periapsis_m / 1000.0
        )
        assert sample_orbital.semi_major_axis_km == pytest.approx(
            sample_orbital.semi_major_axis_m / 1000.0
        )

    def test_deg_conversion(self, sample_orbital):
        assert sample_orbital.inclination_deg == pytest.approx(
            math.degrees(sample_orbital.inclination_rad), abs=1e-9
        )
        assert sample_orbital.raan_deg == pytest.approx(
            math.degrees(sample_orbital.raan_rad), abs=1e-9
        )

    def test_circular_orbit_flag(self, sample_orbital):
        assert sample_orbital.is_orbit_closed is True

    def test_repr(self, sample_orbital):
        r = repr(sample_orbital)
        assert "OrbitalDeckState" in r
        assert "km" in r


# ===========================================================================
# OrbitalDeck tests
# ===========================================================================

class TestOrbitalDeck:

    def test_default_construction(self):
        od = OrbitalDeck()
        assert od._target_ap is None

    def test_with_target(self):
        od = OrbitalDeck(target_apoapsis_m=500_000.0)
        assert od._target_ap == pytest.approx(500_000.0)

    def test_repr(self):
        assert "OrbitalDeck" in repr(OrbitalDeck())

    def test_build_returns_state(self, leo_render_frame):
        od = OrbitalDeck()
        state = od.build(leo_render_frame)
        assert isinstance(state, OrbitalDeckState)

    def test_wrong_type_raises(self):
        od = OrbitalDeck()
        with pytest.raises(TypeError):
            od.build("not_a_render_frame")

    def test_circular_orbit_period_positive(self, leo_render_frame):
        od = OrbitalDeck()
        state = od.build(leo_render_frame)
        assert state.orbital_period_s > 0.0

    def test_circular_orbit_not_suborbital(self, leo_render_frame):
        od = OrbitalDeck()
        state = od.build(leo_render_frame)
        assert state.is_suborbital is False

    def test_elliptic_orbit_apoapsis_gt_periapsis(self, elliptic_render_frame):
        od = OrbitalDeck()
        state = od.build(elliptic_render_frame)
        assert state.apoapsis_m > state.periapsis_m

    def test_elliptic_orbit_eccentricity_nonzero(self, elliptic_render_frame):
        od = OrbitalDeck()
        state = od.build(elliptic_render_frame)
        assert state.eccentricity > 0.001

    def test_set_manoeuvre_node(self, leo_render_frame):
        od = OrbitalDeck()
        od.set_manoeuvre_node(800_000.0)
        state = od.build(leo_render_frame)
        assert state.hohmann is not None
        assert state.hohmann.is_valid

    def test_clear_manoeuvre_node(self, leo_render_frame):
        od = OrbitalDeck(target_apoapsis_m=800_000.0)
        od.clear_manoeuvre_node()
        state = od.build(leo_render_frame)
        assert state.hohmann is None

    def test_period_min_equals_s_over_60(self, leo_render_frame):
        od = OrbitalDeck()
        state = od.build(leo_render_frame)
        assert state.orbital_period_min == pytest.approx(
            state.orbital_period_s / 60.0
        )

    def test_mission_time_propagated(self, leo_render_frame):
        od = OrbitalDeck()
        state = od.build(leo_render_frame)
        assert state.mission_time == pytest.approx(leo_render_frame.mission_time)

    def test_inclination_propagated(self, leo_render_frame):
        od = OrbitalDeck()
        state = od.build(leo_render_frame)
        assert state.inclination_rad == pytest.approx(
            leo_render_frame.inclination, abs=1e-9
        )

    def test_hohmann_total_dv_positive(self, leo_render_frame):
        od = OrbitalDeck(target_apoapsis_m=800_000.0, target_periapsis_m=400_000.0)
        state = od.build(leo_render_frame)
        assert state.hohmann is not None
        assert state.hohmann.total_dv_m_s > 0.0


# ===========================================================================
# EngineDisplayData tests
# ===========================================================================

class TestEngineDisplayData:

    def test_construction(self):
        e = EngineDisplayData(
            status=EngineStatus.ACTIVE,
            throttle_pct=80.0,
            thrust_kn=100.0,
            mass_flow_kg_s=30.0,
            isp_s=300.0,
            twr=2.0,
            gimbal_pitch_deg=0.5,
            gimbal_yaw_deg=-0.3,
        )
        assert e.status == "ACTIVE"
        assert e.throttle_pct == pytest.approx(80.0)

    def test_frozen(self):
        e = EngineDisplayData("IDLE", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(Exception):
            e.status = "ACTIVE"

    def test_repr(self):
        e = EngineDisplayData("ACTIVE", 80.0, 100.0, 30.0, 300.0, 2.0, 0.0, 0.0)
        r = repr(e)
        assert "ACTIVE" in r
        assert "80.0%" in r


class TestClassifyEngine:

    def test_active_when_thrust_positive(self):
        assert _classify_engine(0.8, 50_000.0) == EngineStatus.ACTIVE

    def test_idle_when_thrust_zero(self):
        assert _classify_engine(0.8, 0.0) == EngineStatus.IDLE

    def test_idle_when_both_zero(self):
        assert _classify_engine(0.0, 0.0) == EngineStatus.IDLE


class TestStructuralLabel:

    def test_nominal(self):
        assert _structural_label(0.5, False) == "NOMINAL"

    def test_warning_at_low_margin(self):
        assert _structural_label(0.2, False) == "WARNING"

    def test_critical_at_very_low_margin(self):
        assert _structural_label(0.05, False) == "CRITICAL"

    def test_failed_overrides_margin(self):
        assert _structural_label(0.9, True) == "FAILED"


# ===========================================================================
# AlertSummary tests
# ===========================================================================

class TestAlertSummary:

    def test_empty_alerts(self):
        s = _group_alerts([])
        assert s.n_critical == 0
        assert s.n_warning == 0
        assert s.master_warning is False
        assert s.master_caution is False
        assert s.any_active is False

    def test_critical_sets_master_warning(self, alert_critical):
        s = _group_alerts([alert_critical])
        assert s.master_warning is True
        assert s.n_critical == 1

    def test_warning_sets_master_caution(self, alert_warning):
        s = _group_alerts([alert_warning])
        assert s.master_caution is True
        assert s.master_warning is False

    def test_info_no_master_flags(self, alert_info):
        s = _group_alerts([alert_info])
        assert s.master_warning is False
        assert s.master_caution is False

    def test_all_active_ordering(self, alert_critical, alert_warning, alert_info):
        s = _group_alerts([alert_info, alert_warning, alert_critical])
        all_a = s.all_active
        # CRITICAL first
        assert all_a[0].severity == AlertSeverity.CRITICAL

    def test_any_active_with_info(self, alert_info):
        s = _group_alerts([alert_info])
        assert s.any_active is True

    def test_repr(self, alert_critical):
        s = _group_alerts([alert_critical])
        r = repr(s)
        assert "AlertSummary" in r
        assert "CRIT=1" in r


# ===========================================================================
# AvionicsPanel tests
# ===========================================================================

class TestAvionicsPanel:

    def test_default_construction(self):
        av = AvionicsPanel()
        assert av.sas_enabled is True

    def test_sas_disabled(self):
        av = AvionicsPanel(sas_enabled=False)
        assert av.sas_enabled is False

    def test_sas_setter(self):
        av = AvionicsPanel()
        av.sas_enabled = False
        assert av.sas_enabled is False

    def test_repr(self):
        assert "AvionicsPanel" in repr(AvionicsPanel())

    def test_build_returns_avionics_state(self, pipeline_with_registry):
        reg, snap = pipeline_with_registry
        av = AvionicsPanel()
        state = av.build(snap)
        assert isinstance(state, AvionicsState)

    def test_wrong_snapshot_type_raises(self):
        av = AvionicsPanel()
        with pytest.raises(TypeError):
            av.build("not_a_snapshot")

    def test_throttle_pct_propagated(self, pipeline_with_registry):
        reg, snap = pipeline_with_registry
        av = AvionicsPanel()
        state = av.build(snap)
        expected = snap.throttle * 100.0
        assert state.engine.throttle_pct == pytest.approx(expected)

    def test_angular_rates_propagated(self, pipeline_with_registry):
        reg, snap = pipeline_with_registry
        av = AvionicsPanel()
        state = av.build(snap)
        p = snap.vehicle_state.omega_body[0]
        assert state.angular_rates.roll_rate_deg_s == pytest.approx(
            math.degrees(p), abs=1e-9
        )

    def test_mass_propagated(self, pipeline_with_registry):
        reg, snap = pipeline_with_registry
        av = AvionicsPanel()
        state = av.build(snap)
        assert state.vehicle_mass_kg == pytest.approx(snap.vehicle_state.mass)

    def test_alerts_included(self, pipeline_with_registry, alert_critical):
        reg, snap = pipeline_with_registry
        av = AvionicsPanel()
        state = av.build(snap, alerts=[alert_critical])
        assert state.alerts.n_critical == 1
        assert state.alerts.master_warning is True

    def test_no_alerts_ok(self, pipeline_with_registry):
        reg, snap = pipeline_with_registry
        av = AvionicsPanel()
        state = av.build(snap, alerts=[])
        assert state.alerts.any_active is False

    def test_health_pct_in_range(self, pipeline_with_registry):
        reg, snap = pipeline_with_registry
        av = AvionicsPanel()
        state = av.build(snap)
        assert 0.0 <= state.structural.health_pct <= 100.0

    def test_downrange_km_nonneg(self, pipeline_with_registry):
        reg, snap = pipeline_with_registry
        av = AvionicsPanel()
        state = av.build(snap)
        assert state.downrange_km >= 0.0

    def test_repr(self, pipeline_with_registry):
        reg, snap = pipeline_with_registry
        av = AvionicsPanel()
        state = av.build(snap)
        r = repr(state)
        assert "AvionicsState" in r


# ===========================================================================
# HUDConfig tests
# ===========================================================================

class TestHUDConfig:

    def test_defaults(self):
        cfg = HUDConfig()
        assert cfg.show_pfd is True
        assert cfg.show_orbital_deck is True
        assert cfg.show_avionics is True
        assert cfg.show_celestial is True
        assert cfg.show_vehicle is True
        assert cfg.show_alerts is True
        assert cfg.alert_max_display == 5

    def test_frozen(self):
        cfg = HUDConfig()
        with pytest.raises(Exception):
            cfg.show_pfd = False

    def test_zero_alert_max_rejected(self):
        with pytest.raises(ValueError):
            HUDConfig(alert_max_display=0)

    def test_disable_all(self):
        cfg = HUDConfig(
            show_pfd=False, show_orbital_deck=False, show_avionics=False,
            show_celestial=False, show_vehicle=False
        )
        assert not cfg.show_pfd
        assert not cfg.show_celestial


# ===========================================================================
# HUDFrame tests
# ===========================================================================

class TestHUDFrame:

    def test_no_data_frame(self):
        frame = HUDFrame(
            render_frame=None, pfd=None, orbital_deck=None,
            avionics=None, celestial_scene=None, vehicle_scene=None,
            alerts=[], tick_number=1, mission_time=0.0, has_data=False,
        )
        assert frame.has_data is False
        assert frame.master_warning is False
        assert frame.any_structural_failure is False

    def test_frozen(self):
        frame = HUDFrame(
            render_frame=None, pfd=None, orbital_deck=None,
            avionics=None, celestial_scene=None, vehicle_scene=None,
            alerts=[], tick_number=1, mission_time=0.0, has_data=False,
        )
        with pytest.raises(Exception):
            frame.tick_number = 99

    def test_repr(self):
        frame = HUDFrame(
            render_frame=None, pfd=None, orbital_deck=None,
            avionics=None, celestial_scene=None, vehicle_scene=None,
            alerts=[], tick_number=5, mission_time=10.0, has_data=False,
        )
        r = repr(frame)
        assert "HUDFrame" in r
        assert "tick=5" in r


# ===========================================================================
# HUDCompositor tests
# ===========================================================================

@pytest.fixture
def full_hud(pipeline_with_registry):
    reg, _ = pipeline_with_registry
    vp = Viewport(reg, ViewportConfig())
    cel = CelestialRenderer()
    veh = VehicleRenderer(default_rocket_config())
    pfd = PrimaryFlightDisplay()
    od = OrbitalDeck()
    av = AvionicsPanel()
    return HUDCompositor(vp, reg, cel, veh, pfd, od, av)


class TestHUDCompositor:

    def test_construction(self, full_hud):
        assert full_hud.tick_number == 0

    def test_tick_returns_hud_frame(self, full_hud):
        frame = full_hud.tick()
        assert isinstance(frame, HUDFrame)

    def test_tick_increments_counter(self, full_hud):
        full_hud.tick()
        full_hud.tick()
        assert full_hud.tick_number == 2

    def test_first_tick_has_data(self, full_hud):
        frame = full_hud.tick()
        assert frame.has_data is True

    def test_pfd_populated(self, full_hud):
        frame = full_hud.tick()
        assert frame.pfd is not None
        assert isinstance(frame.pfd, PFDState)

    def test_orbital_deck_populated(self, full_hud):
        frame = full_hud.tick()
        assert frame.orbital_deck is not None
        assert isinstance(frame.orbital_deck, OrbitalDeckState)

    def test_avionics_populated(self, full_hud):
        frame = full_hud.tick()
        assert frame.avionics is not None
        assert isinstance(frame.avionics, AvionicsState)

    def test_celestial_scene_populated(self, full_hud):
        frame = full_hud.tick()
        assert frame.celestial_scene is not None

    def test_vehicle_scene_populated(self, full_hud):
        frame = full_hud.tick()
        assert frame.vehicle_scene is not None

    def test_empty_registry_returns_no_data(self):
        empty_reg = TelemetryRegistry(buffer_size=10)
        vp = Viewport(empty_reg, ViewportConfig())
        cel = CelestialRenderer()
        veh = VehicleRenderer(default_rocket_config())
        pfd = PrimaryFlightDisplay()
        od = OrbitalDeck()
        av = AvionicsPanel()
        hud = HUDCompositor(vp, empty_reg, cel, veh, pfd, od, av)
        frame = hud.tick()
        assert frame.has_data is False
        assert frame.pfd is None

    def test_show_pfd_false_skips_pfd(self, pipeline_with_registry):
        reg, _ = pipeline_with_registry
        vp = Viewport(reg, ViewportConfig())
        cel = CelestialRenderer()
        veh = VehicleRenderer(default_rocket_config())
        hud = HUDCompositor(
            vp, reg, cel, veh,
            PrimaryFlightDisplay(), OrbitalDeck(), AvionicsPanel(),
            config=HUDConfig(show_pfd=False),
        )
        frame = hud.tick()
        assert frame.pfd is None

    def test_show_celestial_false_skips_celestial(self, pipeline_with_registry):
        reg, _ = pipeline_with_registry
        vp = Viewport(reg, ViewportConfig())
        cel = CelestialRenderer()
        veh = VehicleRenderer(default_rocket_config())
        hud = HUDCompositor(
            vp, reg, cel, veh,
            PrimaryFlightDisplay(), OrbitalDeck(), AvionicsPanel(),
            config=HUDConfig(show_celestial=False),
        )
        frame = hud.tick()
        assert frame.celestial_scene is None

    def test_show_vehicle_false_skips_vehicle(self, pipeline_with_registry):
        reg, _ = pipeline_with_registry
        vp = Viewport(reg, ViewportConfig())
        cel = CelestialRenderer()
        veh = VehicleRenderer(default_rocket_config())
        hud = HUDCompositor(
            vp, reg, cel, veh,
            PrimaryFlightDisplay(), OrbitalDeck(), AvionicsPanel(),
            config=HUDConfig(show_vehicle=False),
        )
        frame = hud.tick()
        assert frame.vehicle_scene is None

    def test_reset_clears_tick_counter(self, full_hud):
        full_hud.tick()
        full_hud.tick()
        full_hud.reset()
        assert full_hud.tick_number == 0

    def test_multiple_ticks_mission_time_increases(self, full_hud):
        f1 = full_hud.tick()
        f2 = full_hud.tick()
        # Both read same registry (no new ticks in sim), so mission_time same
        assert f1.mission_time == pytest.approx(f2.mission_time)

    def test_wrong_viewport_type_raises(self):
        with pytest.raises(TypeError):
            HUDCompositor(
                "not_viewport", TelemetryRegistry(),
                CelestialRenderer(), VehicleRenderer(default_rocket_config()),
                PrimaryFlightDisplay(), OrbitalDeck(), AvionicsPanel(),
            )

    def test_wrong_registry_type_raises(self):
        reg = TelemetryRegistry()
        vp = Viewport(reg, ViewportConfig())
        with pytest.raises(TypeError):
            HUDCompositor(
                vp, "not_registry",
                CelestialRenderer(), VehicleRenderer(default_rocket_config()),
                PrimaryFlightDisplay(), OrbitalDeck(), AvionicsPanel(),
            )

    def test_repr(self, full_hud):
        r = repr(full_hud)
        assert "HUDCompositor" in r
