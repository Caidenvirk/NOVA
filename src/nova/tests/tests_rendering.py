import math
import pytest
import numpy as np
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock

# --- Viewport Imports ---
from nova.rendering.viewport import (
    ViewportConfig,
    RenderFrame,
    _lerp,
    _lerp_vec,
    _slerp,
    interpolate_snapshots,
    Viewport,
    _find_bounding_pair,
)

# --- Celestial Imports ---
from nova.rendering.celestial import (
    CelestialConfig,
    _camera_basis,
    project_eci_to_screen,
    _orbit_eci_points,
    _planet_wireframe_eci,
    CelestialMarker,
    GroundTrackPoint,
    CelestialScene,
    CelestialRenderer
)

# --- Vehicle Render Imports ---
from nova.rendering.vehicle_render import (
    VehicleSegmentConfig,
    VehicleRenderConfig,
    SegmentGeometry,
    PlumeGeometry,
    VehicleScene,
    VehicleRenderer,
    default_rocket_config,
    _body_camera_basis,
    _project_body,
    _segment_outline,
    _plume_outline,
    SHAPE_CYLINDER,
    SHAPE_CONE
)

# ==============================================================================
# FIXTURES & DUMMY OBJECTS
# ==============================================================================

@dataclass
class DummyVehicleState:
    position_eci: np.ndarray
    velocity_eci: np.ndarray
    quaternion: np.ndarray
    omega_body: np.ndarray
    mass: float

@dataclass
class DummyTelemetrySnapshot:
    vehicle_state: DummyVehicleState
    _mission_time: float
    altitude: float = 1000.0
    speed: float = 500.0
    mach: float = 1.5
    throttle: float = 1.0
    thrust_magnitude: float = 10000.0
    alpha: float = 0.1
    dynamic_pressure: float = 50000.0
    semi_major_axis: float = 7000e3
    eccentricity: float = 0.0
    inclination: float = 0.0
    apoapsis: float = 1000.0
    periapsis: float = 1000.0
    any_structural_failure: bool = False

    def mission_time(self):
        return self._mission_time

@dataclass
class DummyTelemetryRegistry:
    _history: list

    def history(self):
        return self._history

def create_dummy_snapshot(t: float, pos_x: float = 0.0, is_failed: bool = False) -> DummyTelemetrySnapshot:
    state = DummyVehicleState(
        position_eci=np.array([pos_x, 0.0, 0.0]),
        velocity_eci=np.array([0.0, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body=np.array([0.0, 0.0, 0.0]),
        mass=1000.0,
    )
    return DummyTelemetrySnapshot(
        vehicle_state=state,
        _mission_time=t,
        any_structural_failure=is_failed
    )

def _create_mock_celestial_frame(semi_major_axis=7000e3, eccentricity=0.0, pos=np.array([7000e3, 0, 0]), vel=np.array([0, 7500, 0])) -> RenderFrame:
    return RenderFrame(
        position_eci=pos,
        velocity_eci=vel,
        quaternion=np.array([1, 0, 0, 0]),
        omega_body=np.zeros(3),
        mass=1000.0,
        mission_time=10.0,
        altitude=500e3,
        speed=7500.0,
        mach=25.0,
        throttle=1.0,
        thrust_magnitude=10000.0,
        alpha=0.0,
        dynamic_pressure=0.0,
        semi_major_axis=semi_major_axis,
        eccentricity=eccentricity,
        inclination=0.0,
        apoapsis=500e3,
        periapsis=500e3,
        any_structural_failure=False,
        alpha_blend=0.5,
        earlier_snap_time=9.0,
        later_snap_time=11.0,
    )

def _create_mock_vehicle_frame(throttle=1.0, mach=0.5, is_failed=False) -> RenderFrame:
    return RenderFrame(
        position_eci=np.zeros(3),
        velocity_eci=np.zeros(3),
        quaternion=np.array([1, 0, 0, 0]),
        omega_body=np.zeros(3),
        mass=1000.0,
        mission_time=10.0,
        altitude=500e3,
        speed=150.0,
        mach=mach,
        throttle=throttle,
        thrust_magnitude=10000.0,
        alpha=0.0,
        dynamic_pressure=0.0,
        semi_major_axis=0.0,
        eccentricity=0.0,
        inclination=0.0,
        apoapsis=0.0,
        periapsis=0.0,
        any_structural_failure=is_failed,
        alpha_blend=0.5,
        earlier_snap_time=9.0,
        later_snap_time=11.0,
    )

# ==============================================================================
# VIEWPORT TESTS
# ==============================================================================

def test_viewport_config_valid():
    cfg = ViewportConfig(width_px=1920, height_px=1080)
    assert cfg.aspect_ratio == 1920 / 1080
    assert cfg.frame_budget_s == 1.0 / 60.0
    assert cfg.center_px == (960, 540)

@pytest.mark.parametrize("kwargs, expected_exc", [
    ({"width_px": 0}, ValueError),
    ({"height_px": -10}, ValueError),
    ({"target_fps": 0.0}, ValueError),
    ({"background_color": (-1, 5, 15)}, ValueError),
    ({"background_color": (5, 256, 15)}, ValueError),
    ({"near_clip": 0.0}, ValueError),
    ({"far_clip": 0.1, "near_clip": 0.5}, ValueError),
    ({"fov_deg": 0.0}, ValueError),
    ({"fov_deg": 180.0}, ValueError),
])
def test_viewport_config_invalid(kwargs, expected_exc):
    with pytest.raises(expected_exc):
        ViewportConfig(**kwargs)

def test_render_frame_properties():
    frame = RenderFrame(
        position_eci=np.zeros(3),
        velocity_eci=np.zeros(3),
        quaternion=np.array([1,0,0,0]),
        omega_body=np.zeros(3),
        mass=100.0,
        mission_time=15.5,
        altitude=15000.0,
        speed=1500.0,
        mach=4.0,
        throttle=0.8,
        thrust_magnitude=5000.0,
        alpha=0.0,
        dynamic_pressure=10.0,
        semi_major_axis=10.0,
        eccentricity=0.0,
        inclination=0.0,
        apoapsis=10.0,
        periapsis=10.0,
        any_structural_failure=False,
        alpha_blend=0.5,
        earlier_snap_time=15.0,
        later_snap_time=16.0
    )
    assert frame.altitude_km == 15.0
    assert frame.speed_km_s == 1.5
    assert "t=15.5" in repr(frame)

def test_lerp():
    assert math.isclose(_lerp(10.0, 20.0, 0.5), 15.0)
    assert math.isclose(_lerp(10.0, 20.0, 0.0), 10.0)
    assert math.isclose(_lerp(10.0, 20.0, 1.0), 20.0)

def test_lerp_vec():
    a = np.array([0.0, 10.0, 20.0])
    b = np.array([10.0, 10.0, 0.0])
    res = _lerp_vec(a, b, 0.5)
    np.testing.assert_allclose(res, [5.0, 10.0, 10.0])

def test_slerp():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 1.0, 0.0, 0.0])
    res = _slerp(q0, q1, 0.5)
    expected = np.array([math.sqrt(2)/2, math.sqrt(2)/2, 0.0, 0.0])
    np.testing.assert_allclose(res, expected)

def test_slerp_shortest_path():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([-math.sqrt(2)/2, -math.sqrt(2)/2, 0.0, 0.0])
    res = _slerp(q0, q1, 0.5)
    assert res[0] > 0

def test_slerp_fallback_to_lerp():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([1.0, 1e-4, 0.0, 0.0]) 
    q1 /= np.linalg.norm(q1)
    res = _slerp(q0, q1, 0.5)
    assert math.isclose(np.linalg.norm(res), 1.0)

def test_interpolate_snapshots():
    snap_a = create_dummy_snapshot(10.0, pos_x=100.0)
    snap_b = create_dummy_snapshot(20.0, pos_x=200.0, is_failed=True)
    frame = interpolate_snapshots(snap_a, snap_b, 0.5)
    
    assert frame.mission_time == 15.0
    assert frame.position_eci[0] == 150.0
    assert frame.any_structural_failure is True

def test_viewport_initialization():
    reg = DummyTelemetryRegistry([])
    vp = Viewport(registry=reg) # type: ignore
    assert vp.config.width_px == 1280
    assert vp.frame_count == 0
    assert vp.display_open is False

def test_viewport_find_bounding_pair():
    h = [create_dummy_snapshot(float(t)) for t in [10, 20, 30, 40]]
    sa, sb, t = _find_bounding_pair(h, 25.0)
    assert sa.mission_time() == 20.0
    assert sb.mission_time() == 30.0
    assert math.isclose(t, 0.5)
    
    sa, sb, t = _find_bounding_pair(h, 5.0)
    assert sa.mission_time() == 10.0
    assert t == 0.0
    
    sa, sb, t = _find_bounding_pair(h, 50.0)
    assert sa.mission_time() == 30.0
    assert sb.mission_time() == 40.0
    assert t == 1.0

def test_viewport_get_render_frame():
    reg = DummyTelemetryRegistry([
        create_dummy_snapshot(10.0, pos_x=100.0),
        create_dummy_snapshot(20.0, pos_x=200.0)
    ])
    vp = Viewport(registry=reg) # type: ignore
    
    f_latest = vp.get_render_frame()
    assert f_latest is not None
    assert f_latest.mission_time == 20.0
    
    f_interp = vp.get_render_frame(display_time=15.0)
    assert f_interp.mission_time == 15.0
    assert f_interp.position_eci[0] == 150.0
    assert vp.frame_count == 2

def test_viewport_empty_registry():
    reg = DummyTelemetryRegistry([])
    vp = Viewport(registry=reg) # type: ignore
    assert vp.get_render_frame() is None

def test_viewport_single_snapshot():
    reg = DummyTelemetryRegistry([create_dummy_snapshot(10.0)])
    vp = Viewport(registry=reg) # type: ignore
    frame = vp.get_render_frame(15.0)
    assert frame is not None
    assert frame.mission_time == 10.0
    assert frame.alpha_blend == 1.0


# ==============================================================================
# CELESTIAL TESTS
# ==============================================================================

def test_celestial_config_valid():
    cfg = CelestialConfig()
    assert cfg.orbit_n_points == 360

@pytest.mark.parametrize("kwargs, expected_exc", [
    ({"planet_radius_m": 0.0}, ValueError),
    ({"camera_distance_m": 100.0, "planet_radius_m": 200.0}, ValueError),
    ({"orbit_n_points": 2}, ValueError),
    ({"ground_track_n_points": 0}, ValueError),
    ({"camera_elevation_rad": math.pi}, ValueError),
    ({"planet_color": (300, 0, 0)}, ValueError),
])
def test_celestial_config_invalid(kwargs, expected_exc):
    with pytest.raises(expected_exc):
        CelestialConfig(**kwargs)

def test_camera_basis():
    basis = _camera_basis(0.0, 0.0)
    np.testing.assert_allclose(basis[:, 0], [0.0, 1.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(basis[:, 1], [0.0, 0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(basis[:, 2], [-1.0, 0.0, 0.0], atol=1e-10)

def test_project_eci_to_screen_visible():
    cam_pos = np.array([1000.0, 0.0, 0.0])
    basis = _camera_basis(0.0, 0.0)
    point = np.array([0.0, 0.0, 0.0])
    px = project_eci_to_screen(point, cam_pos, basis, focal_length=1.0, screen_center=(500, 500))
    assert px is not None
    assert math.isclose(px[0], 500.0)
    assert math.isclose(px[1], 500.0)

def test_project_eci_to_screen_behind_camera():
    cam_pos = np.array([1000.0, 0.0, 0.0])
    basis = _camera_basis(0.0, 0.0)
    point = np.array([2000.0, 0.0, 0.0])
    px = project_eci_to_screen(point, cam_pos, basis, focal_length=1.0, screen_center=(500, 500))
    assert px is None

def test_orbit_eci_points_circular():
    frame = _create_mock_celestial_frame(semi_major_axis=7000e3, eccentricity=0.0)
    pts = _orbit_eci_points(frame, n_points=10)
    assert len(pts) == 10
    for pt in pts:
        assert math.isclose(np.linalg.norm(pt), 7000e3, rel_tol=1e-5)

def test_orbit_eci_points_hyperbolic():
    frame = _create_mock_celestial_frame(eccentricity=1.5)
    pts = _orbit_eci_points(frame, n_points=10)
    assert pts == []

def test_planet_wireframe():
    lines = _planet_wireframe_eci(6371e3, n_meridians=4, n_parallels=3)
    assert len(lines) == 6

def test_celestial_marker():
    marker = CelestialMarker("Ap", np.array([1,2,3]), (255,0,0))
    assert marker.label == "Ap"
    assert marker.radius_px == 4

def test_ground_track_point():
    pt = GroundTrackPoint(1.0, 0.5, 100.0)
    assert pt.longitude_rad == 1.0

@patch('nova.frames.transforms.T_ECI_to_ECEF')
@patch('nova.rendering.celestial._ecef_to_geodetic')
def test_celestial_renderer_build(mock_ecef, mock_t_eci):
    mock_t_eci.return_value = np.eye(3)
    mock_ecef.return_value = (1.0, 0.5, 0.0)

    cfg = CelestialConfig(ground_track_n_points=2)
    renderer = CelestialRenderer(cfg)
    
    frame1 = _create_mock_celestial_frame()
    frame2 = _create_mock_celestial_frame(pos=np.array([7100e3, 0, 0]))
    frame3 = _create_mock_celestial_frame(pos=np.array([7200e3, 0, 0]))

    renderer.build(frame1)
    renderer.build(frame2)
    scene = renderer.build(frame3)

    assert len(scene.ground_track) == 2
    assert scene.ground_track[-1].mission_time == frame3.mission_time
    assert len(scene.markers) >= 3
    assert len(scene.orbit_points_eci) == cfg.orbit_n_points

def test_celestial_renderer_clear():
    renderer = CelestialRenderer()
    renderer._ground_track.append(GroundTrackPoint(0,0,0))
    renderer.clear_ground_track()
    assert len(renderer.ground_track) == 0


# ==============================================================================
# VEHICLE RENDER TESTS
# ==============================================================================

def test_vehicle_segment_config():
    seg = VehicleSegmentConfig(
        segment_id="test",
        shape=SHAPE_CYLINDER,
        x_start=10.0,
        x_end=5.0,
        radius_start=2.0,
        radius_end=2.0,
        color=(255, 255, 255)
    )
    assert seg.length == 5.0
    assert seg.mean_radius == 2.0

@pytest.mark.parametrize("kwargs, expected_exc", [
    ({"segment_id": "", "shape": SHAPE_CYLINDER}, ValueError),
    ({"segment_id": "ok", "shape": "invalid_shape"}, ValueError),
    ({"segment_id": "ok", "shape": SHAPE_CYLINDER, "radius_start": -1.0}, ValueError),
    ({"segment_id": "ok", "shape": SHAPE_CYLINDER, "n_sides": 2}, ValueError),
    ({"segment_id": "ok", "shape": SHAPE_CYLINDER, "color": (300, 0, 0)}, ValueError),
])
def test_vehicle_segment_config_invalid(kwargs, expected_exc):
    base_kwargs = {
        "x_start": 10.0, "x_end": 0.0, "radius_start": 1.0, 
        "radius_end": 1.0, "color": (255,255,255)
    }
    base_kwargs.update(kwargs)
    with pytest.raises(expected_exc):
        VehicleSegmentConfig(**base_kwargs)

def test_vehicle_render_config():
    seg = VehicleSegmentConfig("test", SHAPE_CYLINDER, 10, 0, 1, 1, (255,255,255))
    cfg = VehicleRenderConfig(segments=[seg], failed_joint_ids=frozenset(["test"]))
    assert "test" in cfg.failed_joint_ids
    
    with pytest.raises(ValueError):
        VehicleRenderConfig(segments=[])

def test_body_camera_basis():
    basis = _body_camera_basis(0.0, 0.0)
    np.testing.assert_allclose(basis[:, 2], [-1.0, 0.0, 0.0], atol=1e-10)

def test_project_body():
    cam_pos = np.array([50.0, 0.0, 0.0])
    basis = _body_camera_basis(0.0, 0.0)
    point = np.array([0.0, 0.0, 0.0])
    px = _project_body(point, cam_pos, basis, focal_length=1.0, screen_cx=500.0, screen_cy=500.0, scale=1.0)
    assert px is not None
    assert math.isclose(px[0], 500.0)
    assert math.isclose(px[1], 500.0)

def test_segment_outline():
    seg = VehicleSegmentConfig("t", SHAPE_CYLINDER, 10, 0, 1, 1, (255,255,255), n_sides=4)
    cam_pos = np.array([50.0, 0.0, 0.0])
    basis = _body_camera_basis(0.0, 0.0)
    outline = _segment_outline(seg, cam_pos, basis, 1.0, 500.0, 500.0, 1.0)
    assert len(outline) == 8

def test_plume_outline():
    cam_pos = np.array([50.0, 0.0, 0.0])
    basis = _body_camera_basis(0.0, 0.0)
    outer, inner = _plume_outline(0.0, 1.0, 10.0, cam_pos, basis, 1.0, 500.0, 500.0, 1.0, n_sides=4)
    assert len(outer) == 5
    assert len(inner) == 5

def test_vehicle_renderer_build():
    seg1 = VehicleSegmentConfig("body", SHAPE_CYLINDER, 20, 10, 2, 2, (200,200,200))
    seg2 = VehicleSegmentConfig("engine", SHAPE_CONE, 10, 0, 2, 1, (100,100,100), is_engine=True)
    cfg = VehicleRenderConfig(segments=[seg1, seg2], failed_joint_ids=frozenset(["body"]))
    renderer = VehicleRenderer(cfg, screen_width=800, screen_height=600)
    
    frame = _create_mock_vehicle_frame(throttle=1.0, mach=2.0)
    scene = renderer.build(frame)
    
    assert len(scene.segments) == 2
    assert scene.segments[0].is_failed is True
    assert scene.segments[0].color == seg1.failure_color
    assert scene.any_structural_failure is True
    
    assert scene.plume.active is True
    assert scene.plume.length_m > 0
    assert math.isclose(scene.plume.length_m, 18.0)
    assert len(scene.reference_axes) == 3

def test_default_rocket_config():
    cfg = default_rocket_config(total_length_m=50.0)
    assert len(cfg.segments) == 5
    assert cfg.segments[-1].is_engine is True
    assert cfg.camera_distance_m == 100.0
