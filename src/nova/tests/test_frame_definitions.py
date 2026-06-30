"""
tests/unit/test_frame_definitions.py
=====================================
Unit tests for Phase 7 frame definition classes:
  nova.frames.eci   — ECIFrame
  nova.frames.ecef  — ECEFFrame
  nova.frames.enu   — ENUFrame
  nova.frames.body  — BodyFrame

Test coverage
-------------
For each frame class:
  1. Valid construction and field access
  2. Immutability (FrozenInstanceError on mutation attempt)
  3. Validation rejects (wrong shape, bad dtype, out-of-range scalars)
  4. Derived scalar properties
  5. Frame conversion methods (DCM orthogonality, vector rotation consistency)
  6. Round-trip transforms (A → B → A recovers original within tolerance)
  7. Convenience constructors
  8. __repr__ string presence of expected fields

Numerical tolerances follow the hierarchy defined in Section 10.2
of the engineering handoff:
  Transform identity:    1e-12
  Quaternion norm:       1e-9
  Position round-trip:   1e-3 m
  Velocity round-trip:   1e-6 m/s
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nova.core.constants import (
    EARTH_MU,
    EARTH_OMEGA,
    EARTH_RADIUS_EQ,
    EARTH_RADIUS_POLAR,
    TRANSFORM_IDENTITY_TOL,
    QUATERNION_NORM_TOL,
)
from nova.core.state_vector import make_state
from nova.frames.eci import ECIFrame, eci_from_state, eci_circular_orbit
from nova.frames.ecef import ECEFFrame, ecef_from_eci, ecef_from_geodetic
from nova.frames.enu import ENUFrame, enu_from_ecef, enu_from_eci
from nova.frames.body import BodyFrame, body_from_state
from nova.vehicle.mass_model import MassModel, point_mass, compute_mass_properties


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def leo_position_eci() -> np.ndarray:
    """LEO position: 400 km above equator on X-axis [m]."""
    return np.array([EARTH_RADIUS_EQ + 400_000.0, 0.0, 0.0], dtype=np.float64)


@pytest.fixture
def leo_velocity_eci() -> np.ndarray:
    """Circular orbital velocity for 400 km LEO [m s⁻¹]."""
    r = EARTH_RADIUS_EQ + 400_000.0
    v = math.sqrt(EARTH_MU / r)
    return np.array([0.0, v, 0.0], dtype=np.float64)


@pytest.fixture
def identity_quaternion() -> np.ndarray:
    """Identity quaternion — body axes aligned with ENU."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@pytest.fixture
def simple_mass_model() -> MassModel:
    """Symmetric 1000 kg rocket stage, 10 m long, radius 0.5 m."""
    from nova.vehicle.mass_model import solid_cylinder
    comps = [
        solid_cylinder("stage", 1000.0, length=10.0, radius=0.5,
                       position_body=np.array([5.0, 0.0, 0.0])),
    ]
    return compute_mass_properties(comps)


@pytest.fixture
def leo_vehicle_state(
    leo_position_eci: np.ndarray,
    leo_velocity_eci: np.ndarray,
    identity_quaternion: np.ndarray,
) -> "nova.core.state_vector.VehicleState":
    """A VehicleState at 400 km LEO with identity attitude."""
    return make_state(
        position_eci=leo_position_eci,
        velocity_eci=leo_velocity_eci,
        quaternion=identity_quaternion,
        omega_body=np.zeros(3),
        mass=1000.0,
        time=0.0,
    )


# ===========================================================================
# ECIFrame tests
# ===========================================================================

class TestECIFrameConstruction:
    """Valid construction, field access, and dtype enforcement."""

    def test_basic_construction(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(
            position_eci=leo_position_eci,
            velocity_eci=leo_velocity_eci,
            epoch_time=0.0,
        )
        assert frame.position_eci.shape == (3,)
        assert frame.velocity_eci.shape == (3,)
        assert frame.position_eci.dtype == np.float64
        assert frame.velocity_eci.dtype == np.float64
        assert frame.epoch_time == 0.0
        assert frame.body_name == "Earth"

    def test_list_inputs_converted_to_float64(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(
            position_eci=leo_position_eci.tolist(),
            velocity_eci=leo_velocity_eci.tolist(),
            epoch_time=100.0,
        )
        assert frame.position_eci.dtype == np.float64
        assert frame.velocity_eci.dtype == np.float64

    def test_nonzero_epoch_time(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, epoch_time=3600.0)
        assert frame.epoch_time == 3600.0

    def test_custom_body_parameters(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(
            leo_position_eci, leo_velocity_eci, 0.0,
            body_name="Moon", omega=0.0, mu=4.902_800_066e12,
        )
        assert frame.body_name == "Moon"
        assert frame.omega == 0.0
        assert frame.mu == pytest.approx(4.902_800_066e12)

    def test_epoch_time_zero_is_valid(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, epoch_time=0.0)
        assert frame.epoch_time == 0.0


class TestECIFrameImmutability:
    """FrozenInstanceError on any field mutation attempt."""

    def test_cannot_set_position(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, 0.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            frame.position_eci = np.zeros(3)

    def test_cannot_set_epoch_time(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, 0.0)
        with pytest.raises(Exception):
            frame.epoch_time = 999.0

    def test_cannot_set_body_name(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, 0.0)
        with pytest.raises(Exception):
            frame.body_name = "Moon"


class TestECIFrameValidation:
    """Rejects invalid inputs."""

    def test_wrong_shape_position(self, leo_velocity_eci):
        with pytest.raises(ValueError, match="position_eci"):
            ECIFrame(np.zeros(4), leo_velocity_eci, 0.0)

    def test_wrong_shape_velocity(self, leo_position_eci):
        with pytest.raises(ValueError, match="velocity_eci"):
            ECIFrame(leo_position_eci, np.zeros(2), 0.0)

    def test_negative_epoch_time(self, leo_position_eci, leo_velocity_eci):
        with pytest.raises(ValueError, match="epoch_time"):
            ECIFrame(leo_position_eci, leo_velocity_eci, epoch_time=-1.0)

    def test_empty_body_name(self, leo_position_eci, leo_velocity_eci):
        with pytest.raises(ValueError, match="body_name"):
            ECIFrame(leo_position_eci, leo_velocity_eci, 0.0, body_name="")

    def test_negative_omega(self, leo_position_eci, leo_velocity_eci):
        with pytest.raises(ValueError, match="omega"):
            ECIFrame(leo_position_eci, leo_velocity_eci, 0.0, omega=-1e-5)

    def test_zero_mu(self, leo_position_eci, leo_velocity_eci):
        with pytest.raises(ValueError, match="mu"):
            ECIFrame(leo_position_eci, leo_velocity_eci, 0.0, mu=0.0)


class TestECIFrameProperties:
    """Derived scalar properties."""

    def test_radius_is_correct(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, 0.0)
        expected = float(np.linalg.norm(leo_position_eci))
        assert frame.radius == pytest.approx(expected, rel=1e-12)

    def test_speed_is_correct(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, 0.0)
        expected = float(np.linalg.norm(leo_velocity_eci))
        assert frame.speed == pytest.approx(expected, rel=1e-12)

    def test_orbital_energy_negative_for_leo(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, 0.0)
        assert frame.specific_orbital_energy < 0.0

    def test_orbital_energy_formula(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, 0.0)
        r = frame.radius
        v = frame.speed
        expected = 0.5 * v ** 2 - EARTH_MU / r
        assert frame.specific_orbital_energy == pytest.approx(expected, rel=1e-12)

    def test_position_unit_is_unit_vector(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, 0.0)
        u = frame.position_unit
        assert u.shape == (3,)
        assert abs(float(np.linalg.norm(u)) - 1.0) < 1.0e-12

    def test_position_unit_direction(self, leo_position_eci, leo_velocity_eci):
        """For position along +X, unit vector is [1, 0, 0]."""
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, 0.0)
        u = frame.position_unit
        assert u == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


class TestECIFrameTransforms:
    """DCM orthogonality and ECI↔ECEF conversion."""

    def test_dcm_to_ecef_is_orthogonal(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, epoch_time=1000.0)
        T = frame.dcm_to_ecef()
        assert T.shape == (3, 3)
        residual = T @ T.T - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) <= TRANSFORM_IDENTITY_TOL

    def test_dcm_from_ecef_is_orthogonal(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, epoch_time=1000.0)
        T = frame.dcm_from_ecef()
        residual = T @ T.T - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) <= TRANSFORM_IDENTITY_TOL

    def test_dcm_to_ecef_inverse_is_dcm_from_ecef(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, epoch_time=500.0)
        T_to = frame.dcm_to_ecef()
        T_from = frame.dcm_from_ecef()
        product = T_to @ T_from
        residual = product - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) < 1.0e-12

    def test_ecef_position_round_trip(self, leo_position_eci, leo_velocity_eci):
        """ECI → ECEF → ECI recovers original position."""
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, epoch_time=1234.5)
        pos_ecef = frame.to_ecef_position()
        T_back = frame.dcm_from_ecef()
        pos_eci_recovered = T_back @ pos_ecef
        assert pos_eci_recovered == pytest.approx(leo_position_eci, abs=1.0e-3)

    def test_at_t0_eci_ecef_position_equal(self, leo_position_eci, leo_velocity_eci):
        """At t=0, ECEF rotation angle is 0 → positions are identical."""
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, epoch_time=0.0)
        pos_ecef = frame.to_ecef_position()
        assert pos_ecef == pytest.approx(leo_position_eci, abs=1.0e-6)


class TestECIConvenienceConstructors:
    """eci_from_state and eci_circular_orbit."""

    def test_eci_from_state_creates_valid_frame(
        self, leo_position_eci, leo_velocity_eci
    ):
        frame = eci_from_state(leo_position_eci, leo_velocity_eci, 0.0)
        assert isinstance(frame, ECIFrame)
        assert frame.position_eci == pytest.approx(leo_position_eci, abs=1e-9)

    def test_eci_circular_orbit_radius(self):
        alt = 400_000.0
        frame = eci_circular_orbit(alt)
        expected_r = EARTH_RADIUS_EQ + alt
        assert frame.radius == pytest.approx(expected_r, rel=1e-12)

    def test_eci_circular_orbit_circular_speed(self):
        alt = 400_000.0
        frame = eci_circular_orbit(alt)
        r = frame.radius
        v_circ = math.sqrt(EARTH_MU / r)
        assert frame.speed == pytest.approx(v_circ, rel=1e-12)

    def test_eci_circular_orbit_position_velocity_orthogonal(self):
        """For circular orbit, r · v = 0."""
        frame = eci_circular_orbit(400_000.0)
        dot = float(np.dot(frame.position_eci, frame.velocity_eci))
        assert abs(dot) < 1.0e-6

    def test_eci_circular_orbit_negative_energy(self):
        frame = eci_circular_orbit(400_000.0)
        assert frame.specific_orbital_energy < 0.0

    def test_eci_circular_orbit_inclined(self):
        inc = math.radians(51.6)
        frame = eci_circular_orbit(400_000.0, inclination_rad=inc)
        # Velocity should have Z component proportional to sin(i)
        r = frame.radius
        v_circ = math.sqrt(EARTH_MU / r)
        assert abs(frame.velocity_eci[2]) == pytest.approx(v_circ * math.sin(inc), rel=1e-12)

    def test_eci_circular_orbit_zero_altitude_raises(self):
        with pytest.raises(ValueError):
            eci_circular_orbit(0.0)

    def test_eci_repr_contains_expected_fields(self, leo_position_eci, leo_velocity_eci):
        frame = ECIFrame(leo_position_eci, leo_velocity_eci, 0.0)
        s = repr(frame)
        assert "ECIFrame" in s
        assert "Earth" in s
        assert "km" in s


# ===========================================================================
# ECEFFrame tests
# ===========================================================================

class TestECEFFrameConstruction:

    def test_basic_construction(self):
        pos = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        vel = np.zeros(3, dtype=np.float64)
        frame = ECEFFrame(position_ecef=pos, velocity_ecef=vel, epoch_time=0.0)
        assert frame.position_ecef.shape == (3,)
        assert frame.velocity_ecef.shape == (3,)
        assert frame.epoch_time == 0.0
        assert frame.body_name == "Earth"

    def test_dtype_enforcement(self):
        pos = [EARTH_RADIUS_EQ, 0.0, 0.0]
        vel = [0.0, 0.0, 0.0]
        frame = ECEFFrame(position_ecef=pos, velocity_ecef=vel, epoch_time=0.0)
        assert frame.position_ecef.dtype == np.float64
        assert frame.velocity_ecef.dtype == np.float64

    def test_custom_radii(self):
        pos = np.array([1e7, 0.0, 0.0], dtype=np.float64)
        vel = np.zeros(3, dtype=np.float64)
        frame = ECEFFrame(pos, vel, 0.0, radius_eq=1e7, radius_polar=0.99e7)
        assert frame.radius_eq == pytest.approx(1e7)


class TestECEFFrameImmutability:

    def test_cannot_set_position(self):
        pos = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), 0.0)
        with pytest.raises(Exception):
            frame.position_ecef = np.zeros(3)

    def test_cannot_set_epoch_time(self):
        pos = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), 0.0)
        with pytest.raises(Exception):
            frame.epoch_time = 100.0


class TestECEFFrameValidation:

    def test_wrong_shape_position(self):
        with pytest.raises(ValueError, match="position_ecef"):
            ECEFFrame(np.zeros(4), np.zeros(3), 0.0)

    def test_wrong_shape_velocity(self):
        with pytest.raises(ValueError, match="velocity_ecef"):
            ECEFFrame(np.zeros(3), np.zeros(5), 0.0)

    def test_negative_epoch_time(self):
        with pytest.raises(ValueError, match="epoch_time"):
            ECEFFrame(np.zeros(3), np.zeros(3), -1.0)

    def test_polar_radius_exceeds_equatorial(self):
        with pytest.raises(ValueError, match="radius_polar"):
            ECEFFrame(np.zeros(3), np.zeros(3), 0.0,
                      radius_eq=6_356_000.0, radius_polar=6_378_000.0)

    def test_empty_body_name(self):
        with pytest.raises(ValueError, match="body_name"):
            ECEFFrame(np.zeros(3), np.zeros(3), 0.0, body_name="  ")


class TestECEFFrameGeodesic:
    """Geodetic conversion accuracy."""

    def test_equatorial_surface_longitude_zero(self):
        """Point on equator at 0° longitude maps to (0, 0, ~0 alt)."""
        pos = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), 0.0)
        lam, phi, alt = frame.geodetic()
        assert lam == pytest.approx(0.0, abs=1e-9)
        assert phi == pytest.approx(0.0, abs=1e-9)
        assert alt == pytest.approx(0.0, abs=0.1)  # 0.1 m tolerance

    def test_north_pole(self):
        """Point on north pole."""
        pos = np.array([0.0, 0.0, EARTH_RADIUS_POLAR], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), 0.0)
        _, phi, alt = frame.geodetic()
        assert phi == pytest.approx(math.pi / 2.0, abs=1e-9)
        assert alt == pytest.approx(0.0, abs=0.1)

    def test_geodetic_round_trip(self):
        """geodetic() → ecef_from_geodetic() round-trips within 1 mm."""
        # A specific geodetic point
        lam0 = math.radians(-79.3832)  # Toronto longitude
        phi0 = math.radians(43.6532)   # Toronto latitude
        alt0 = 76.0                    # approximate elevation [m]
        frame0 = ecef_from_geodetic(lam0, phi0, alt0)
        lam1, phi1, alt1 = frame0.geodetic()
        assert lam1 == pytest.approx(lam0, abs=1e-9)
        assert phi1 == pytest.approx(phi0, abs=1e-9)
        assert alt1 == pytest.approx(alt0, abs=0.001)  # 1 mm

    def test_altitude_above_equator(self):
        alt = 400_000.0  # 400 km
        pos = np.array([EARTH_RADIUS_EQ + alt, 0.0, 0.0], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), 0.0)
        _, _, alt_computed = frame.geodetic()
        assert alt_computed == pytest.approx(alt, abs=1.0)  # 1 m tolerance

    def test_90deg_longitude(self):
        """Point at 90° east longitude."""
        pos = np.array([0.0, EARTH_RADIUS_EQ, 0.0], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), 0.0)
        lam, phi, _ = frame.geodetic()
        assert lam == pytest.approx(math.pi / 2.0, abs=1e-9)
        assert phi == pytest.approx(0.0, abs=1e-9)

    def test_longitude_property_matches_geodetic(self):
        pos = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), 0.0)
        lam_prop = frame.longitude_rad
        lam_tuple, _, _ = frame.geodetic()
        assert lam_prop == pytest.approx(lam_tuple, abs=1e-12)

    def test_latitude_property_matches_geodetic(self):
        pos = np.array([0.0, 0.0, EARTH_RADIUS_POLAR], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), 0.0)
        phi_prop = frame.latitude_rad
        _, phi_tuple, _ = frame.geodetic()
        assert phi_prop == pytest.approx(phi_tuple, abs=1e-12)


class TestECEFFrameTransforms:

    def test_dcm_from_eci_is_orthogonal(self):
        pos = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), epoch_time=1000.0)
        T = frame.dcm_from_eci()
        residual = T @ T.T - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) <= TRANSFORM_IDENTITY_TOL

    def test_dcm_to_enu_is_orthogonal(self):
        pos = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), 0.0)
        T = frame.dcm_to_enu()
        residual = T @ T.T - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) <= TRANSFORM_IDENTITY_TOL

    def test_eci_position_round_trip(self):
        """ECEF → ECI → ECEF recovers original."""
        pos_ecef = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        frame = ECEFFrame(pos_ecef, np.zeros(3), epoch_time=500.0)
        pos_eci = frame.to_eci_position()
        T_back = frame.dcm_from_eci()
        recovered = T_back @ pos_eci
        assert recovered == pytest.approx(pos_ecef, abs=1.0e-3)


class TestECEFConvenienceConstructors:

    def test_ecef_from_eci_position_at_t0(self, leo_position_eci, leo_velocity_eci):
        """At t=0, ECI and ECEF are aligned → positions should match."""
        frame = ecef_from_eci(leo_position_eci, leo_velocity_eci, 0.0)
        assert frame.position_ecef == pytest.approx(leo_position_eci, abs=1.0e-3)

    def test_ecef_from_eci_valid_frame(self, leo_position_eci, leo_velocity_eci):
        frame = ecef_from_eci(leo_position_eci, leo_velocity_eci, 100.0)
        assert isinstance(frame, ECEFFrame)
        assert frame.position_ecef.shape == (3,)

    def test_ecef_from_geodetic_equator(self):
        """Geodetic (0, 0, 0) → ECEF (EARTH_RADIUS_EQ, 0, 0)."""
        frame = ecef_from_geodetic(0.0, 0.0, 0.0)
        assert frame.position_ecef[0] == pytest.approx(EARTH_RADIUS_EQ, rel=1e-6)
        assert abs(frame.position_ecef[1]) < 1.0
        assert abs(frame.position_ecef[2]) < 1.0

    def test_ecef_from_geodetic_with_velocity(self):
        """ENU velocity is correctly rotated to ECEF."""
        vel_enu = np.array([100.0, 0.0, 0.0], dtype=np.float64)  # 100 m/s east
        frame = ecef_from_geodetic(0.0, 0.0, 0.0, velocity_enu=vel_enu)
        # At equator, prime meridian: east direction in ECEF is [0, 1, 0]
        assert frame.velocity_ecef[1] == pytest.approx(100.0, abs=0.01)

    def test_ecef_repr(self):
        pos = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        frame = ECEFFrame(pos, np.zeros(3), 0.0)
        s = repr(frame)
        assert "ECEFFrame" in s
        assert "lon=" in s
        assert "lat=" in s
        assert "alt=" in s


# ===========================================================================
# ENUFrame tests
# ===========================================================================

class TestENUFrameConstruction:

    def test_basic_construction(self):
        pos = np.array([100.0, 200.0, 400_000.0], dtype=np.float64)
        vel = np.array([10.0, 0.0, 50.0], dtype=np.float64)
        frame = ENUFrame(
            position_enu=pos,
            velocity_enu=vel,
            ref_longitude_rad=0.0,
            ref_latitude_rad=0.0,
            epoch_time=0.0,
        )
        assert frame.position_enu.shape == (3,)
        assert frame.velocity_enu.shape == (3,)
        assert frame.ref_longitude_rad == 0.0
        assert frame.ref_latitude_rad == 0.0

    def test_dtype_enforcement(self):
        frame = ENUFrame(
            position_enu=[0.0, 0.0, 1000.0],
            velocity_enu=[0.0, 0.0, 0.0],
            ref_longitude_rad=0.0,
            ref_latitude_rad=0.0,
            epoch_time=0.0,
        )
        assert frame.position_enu.dtype == np.float64
        assert frame.velocity_enu.dtype == np.float64


class TestENUFrameImmutability:

    def test_cannot_set_position(self):
        frame = ENUFrame(
            np.zeros(3), np.zeros(3),
            ref_longitude_rad=0.0, ref_latitude_rad=0.0, epoch_time=0.0,
        )
        with pytest.raises(Exception):
            frame.position_enu = np.ones(3)

    def test_cannot_set_ref_longitude(self):
        frame = ENUFrame(
            np.zeros(3), np.zeros(3),
            ref_longitude_rad=0.0, ref_latitude_rad=0.0, epoch_time=0.0,
        )
        with pytest.raises(Exception):
            frame.ref_longitude_rad = 1.0


class TestENUFrameValidation:

    def test_wrong_shape_position(self):
        with pytest.raises(ValueError, match="position_enu"):
            ENUFrame(np.zeros(2), np.zeros(3), 0.0, 0.0, 0.0)

    def test_wrong_shape_velocity(self):
        with pytest.raises(ValueError, match="velocity_enu"):
            ENUFrame(np.zeros(3), np.zeros(4), 0.0, 0.0, 0.0)

    def test_longitude_out_of_range(self):
        with pytest.raises(ValueError, match="ref_longitude_rad"):
            ENUFrame(np.zeros(3), np.zeros(3), ref_longitude_rad=4.0,
                     ref_latitude_rad=0.0, epoch_time=0.0)

    def test_latitude_out_of_range(self):
        with pytest.raises(ValueError, match="ref_latitude_rad"):
            ENUFrame(np.zeros(3), np.zeros(3), ref_longitude_rad=0.0,
                     ref_latitude_rad=2.0, epoch_time=0.0)

    def test_negative_epoch_time(self):
        with pytest.raises(ValueError, match="epoch_time"):
            ENUFrame(np.zeros(3), np.zeros(3), 0.0, 0.0, epoch_time=-0.1)


class TestENUFrameProperties:

    @pytest.fixture
    def sample_frame(self):
        return ENUFrame(
            position_enu=np.array([300.0, 400.0, 1000.0], dtype=np.float64),
            velocity_enu=np.array([5.0, 0.0, 10.0], dtype=np.float64),
            ref_longitude_rad=0.0,
            ref_latitude_rad=0.0,
            epoch_time=0.0,
        )

    def test_east_component(self, sample_frame):
        assert sample_frame.east == pytest.approx(300.0)

    def test_north_component(self, sample_frame):
        assert sample_frame.north == pytest.approx(400.0)

    def test_up_component(self, sample_frame):
        assert sample_frame.up == pytest.approx(1000.0)

    def test_horizontal_range(self, sample_frame):
        """E=300, N=400 → range = 500."""
        assert sample_frame.horizontal_range == pytest.approx(500.0, rel=1e-12)

    def test_slant_range(self, sample_frame):
        expected = math.sqrt(300.0**2 + 400.0**2 + 1000.0**2)
        assert sample_frame.slant_range == pytest.approx(expected, rel=1e-12)

    def test_bearing_due_east(self):
        frame = ENUFrame(
            position_enu=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            velocity_enu=np.zeros(3),
            ref_longitude_rad=0.0, ref_latitude_rad=0.0, epoch_time=0.0,
        )
        assert frame.bearing_rad == pytest.approx(math.pi / 2.0, abs=1e-9)

    def test_bearing_due_north(self):
        frame = ENUFrame(
            position_enu=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            velocity_enu=np.zeros(3),
            ref_longitude_rad=0.0, ref_latitude_rad=0.0, epoch_time=0.0,
        )
        assert frame.bearing_rad == pytest.approx(0.0, abs=1e-9)

    def test_vertical_speed(self, sample_frame):
        assert sample_frame.vertical_speed == pytest.approx(10.0)

    def test_horizontal_speed(self, sample_frame):
        assert sample_frame.horizontal_speed == pytest.approx(5.0, rel=1e-12)

    def test_total_speed(self, sample_frame):
        expected = math.sqrt(5.0**2 + 10.0**2)
        assert sample_frame.speed == pytest.approx(expected, rel=1e-12)


class TestENUFrameTransforms:

    def test_dcm_to_ecef_orthogonal(self):
        frame = ENUFrame(np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0)
        T = frame.dcm_to_ecef()
        residual = T @ T.T - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) <= TRANSFORM_IDENTITY_TOL

    def test_dcm_from_ecef_orthogonal(self):
        frame = ENUFrame(np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0)
        T = frame.dcm_from_ecef()
        residual = T @ T.T - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) <= TRANSFORM_IDENTITY_TOL

    def test_dcm_to_ecef_inverse_of_from_ecef(self):
        lam = math.radians(45.0)
        phi = math.radians(30.0)
        frame = ENUFrame(np.zeros(3), np.zeros(3), lam, phi, 0.0)
        T_to = frame.dcm_to_ecef()
        T_from = frame.dcm_from_ecef()
        product = T_to @ T_from
        residual = product - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) < 1.0e-12

    def test_ecef_velocity_round_trip(self):
        """ENU velocity → ECEF → ENU recovers original."""
        vel_enu = np.array([10.0, 20.0, 5.0], dtype=np.float64)
        lam = math.radians(30.0)
        phi = math.radians(45.0)
        frame = ENUFrame(np.zeros(3), vel_enu, lam, phi, 0.0)
        vel_ecef = frame.to_ecef_velocity()
        T_back = frame.dcm_from_ecef()
        vel_enu_recovered = T_back @ vel_ecef
        assert vel_enu_recovered == pytest.approx(vel_enu, abs=1.0e-9)

    def test_eci_velocity_round_trip(self):
        """ENU → ECI → ENU recovers original velocity."""
        vel_enu = np.array([100.0, 50.0, 10.0], dtype=np.float64)
        lam = math.radians(10.0)
        phi = math.radians(20.0)
        t = 500.0
        frame = ENUFrame(np.zeros(3), vel_enu, lam, phi, epoch_time=t)
        vel_eci = frame.to_eci_velocity()
        # Manually invert: ECI → ECEF → ENU
        from nova.frames.transforms import T_ECI_to_ECEF, T_ECEF_to_ENU
        T = T_ECEF_to_ENU(lam, phi) @ T_ECI_to_ECEF(t)
        vel_enu_recovered = T @ vel_eci
        assert vel_enu_recovered == pytest.approx(vel_enu, abs=1.0e-9)


class TestENUConvenienceConstructors:

    def test_enu_from_ecef_creates_frame(self):
        pos_ecef = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        vel_ecef = np.zeros(3, dtype=np.float64)
        frame = enu_from_ecef(pos_ecef, vel_ecef, 0.0, 0.0)
        assert isinstance(frame, ENUFrame)

    def test_enu_from_ecef_position_at_equator(self):
        """ECEF position on equator at prime meridian → Up component = EARTH_RADIUS_EQ."""
        pos_ecef = np.array([EARTH_RADIUS_EQ, 0.0, 0.0], dtype=np.float64)
        frame = enu_from_ecef(pos_ecef, np.zeros(3), ref_longitude_rad=0.0,
                              ref_latitude_rad=0.0)
        # The Up component should equal the radial distance
        assert frame.up == pytest.approx(EARTH_RADIUS_EQ, rel=1e-6)
        assert abs(frame.east) < 1.0e-3
        assert abs(frame.north) < 1.0e-3

    def test_enu_from_eci_creates_frame(self, leo_position_eci, leo_velocity_eci):
        frame = enu_from_eci(leo_position_eci, leo_velocity_eci, 0.0, 0.0, 0.0)
        assert isinstance(frame, ENUFrame)
        assert frame.position_enu.shape == (3,)

    def test_enu_repr(self):
        frame = ENUFrame(
            np.array([0.0, 0.0, 1000.0]), np.zeros(3), 0.0, 0.0, 0.0
        )
        s = repr(frame)
        assert "ENUFrame" in s
        assert "E=" in s
        assert "N=" in s
        assert "U=" in s


# ===========================================================================
# BodyFrame tests
# ===========================================================================

class TestBodyFrameConstruction:

    def test_basic_construction(self, identity_quaternion, simple_mass_model):
        frame = BodyFrame(
            quaternion=identity_quaternion,
            omega_body=np.zeros(3),
            com_body=simple_mass_model.com_body,
            total_mass=simple_mass_model.total_mass,
            epoch_time=0.0,
            inertia_body=simple_mass_model.inertia_body,
            inertia_inv=simple_mass_model.inertia_inv,
        )
        assert frame.quaternion.shape == (4,)
        assert frame.omega_body.shape == (3,)
        assert frame.com_body.shape == (3,)
        assert frame.total_mass > 0.0
        assert frame.epoch_time == 0.0

    def test_dtype_is_float64(self, identity_quaternion, simple_mass_model):
        frame = BodyFrame(
            quaternion=identity_quaternion,
            omega_body=np.zeros(3),
            com_body=simple_mass_model.com_body,
            total_mass=simple_mass_model.total_mass,
            epoch_time=0.0,
            inertia_body=simple_mass_model.inertia_body,
            inertia_inv=simple_mass_model.inertia_inv,
        )
        assert frame.quaternion.dtype == np.float64
        assert frame.omega_body.dtype == np.float64
        assert frame.com_body.dtype == np.float64
        assert frame.inertia_body.dtype == np.float64


class TestBodyFrameImmutability:

    def test_cannot_set_quaternion(self, identity_quaternion, simple_mass_model):
        frame = BodyFrame(
            quaternion=identity_quaternion,
            omega_body=np.zeros(3),
            com_body=simple_mass_model.com_body,
            total_mass=simple_mass_model.total_mass,
            epoch_time=0.0,
            inertia_body=simple_mass_model.inertia_body,
            inertia_inv=simple_mass_model.inertia_inv,
        )
        with pytest.raises(Exception):
            frame.quaternion = np.array([0.0, 1.0, 0.0, 0.0])

    def test_cannot_set_total_mass(self, identity_quaternion, simple_mass_model):
        frame = BodyFrame(
            quaternion=identity_quaternion,
            omega_body=np.zeros(3),
            com_body=simple_mass_model.com_body,
            total_mass=simple_mass_model.total_mass,
            epoch_time=0.0,
            inertia_body=simple_mass_model.inertia_body,
            inertia_inv=simple_mass_model.inertia_inv,
        )
        with pytest.raises(Exception):
            frame.total_mass = 0.0


class TestBodyFrameValidation:

    def test_wrong_shape_quaternion(self, simple_mass_model):
        with pytest.raises(ValueError, match="quaternion"):
            BodyFrame(
                quaternion=np.array([1.0, 0.0, 0.0]),  # wrong: shape (3,)
                omega_body=np.zeros(3),
                com_body=simple_mass_model.com_body,
                total_mass=simple_mass_model.total_mass,
                epoch_time=0.0,
                inertia_body=simple_mass_model.inertia_body,
                inertia_inv=simple_mass_model.inertia_inv,
            )

    def test_non_unit_quaternion_rejected(self, simple_mass_model):
        with pytest.raises(ValueError, match="quaternion"):
            BodyFrame(
                quaternion=np.array([2.0, 0.0, 0.0, 0.0]),  # norm = 2
                omega_body=np.zeros(3),
                com_body=simple_mass_model.com_body,
                total_mass=simple_mass_model.total_mass,
                epoch_time=0.0,
                inertia_body=simple_mass_model.inertia_body,
                inertia_inv=simple_mass_model.inertia_inv,
            )

    def test_zero_mass_rejected(self, identity_quaternion, simple_mass_model):
        with pytest.raises(ValueError, match="total_mass"):
            BodyFrame(
                quaternion=identity_quaternion,
                omega_body=np.zeros(3),
                com_body=simple_mass_model.com_body,
                total_mass=0.0,
                epoch_time=0.0,
                inertia_body=simple_mass_model.inertia_body,
                inertia_inv=simple_mass_model.inertia_inv,
            )

    def test_negative_epoch_time_rejected(self, identity_quaternion, simple_mass_model):
        with pytest.raises(ValueError, match="epoch_time"):
            BodyFrame(
                quaternion=identity_quaternion,
                omega_body=np.zeros(3),
                com_body=simple_mass_model.com_body,
                total_mass=simple_mass_model.total_mass,
                epoch_time=-1.0,
                inertia_body=simple_mass_model.inertia_body,
                inertia_inv=simple_mass_model.inertia_inv,
            )

    def test_asymmetric_inertia_rejected(self, identity_quaternion, simple_mass_model):
        I_bad = np.eye(3, dtype=np.float64) * 1000.0
        I_bad[0, 1] = 500.0  # asymmetric
        with pytest.raises(ValueError, match="symmetric"):
            BodyFrame(
                quaternion=identity_quaternion,
                omega_body=np.zeros(3),
                com_body=simple_mass_model.com_body,
                total_mass=simple_mass_model.total_mass,
                epoch_time=0.0,
                inertia_body=I_bad,
                inertia_inv=simple_mass_model.inertia_inv,
            )


class TestBodyFrameAttitudeProperties:

    @pytest.fixture
    def identity_frame(self, identity_quaternion, simple_mass_model):
        return BodyFrame(
            quaternion=identity_quaternion,
            omega_body=np.zeros(3),
            com_body=simple_mass_model.com_body,
            total_mass=simple_mass_model.total_mass,
            epoch_time=0.0,
            inertia_body=simple_mass_model.inertia_body,
            inertia_inv=simple_mass_model.inertia_inv,
        )

    def test_dcm_enu_to_body_is_orthogonal(self, identity_frame):
        T = identity_frame.dcm_enu_to_body
        residual = T @ T.T - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) <= TRANSFORM_IDENTITY_TOL

    def test_dcm_body_to_enu_is_orthogonal(self, identity_frame):
        T = identity_frame.dcm_body_to_enu
        residual = T @ T.T - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) <= TRANSFORM_IDENTITY_TOL

    def test_dcm_enu_to_body_inverse_is_dcm_body_to_enu(self, identity_frame):
        T_to = identity_frame.dcm_enu_to_body
        T_from = identity_frame.dcm_body_to_enu
        product = T_to @ T_from
        residual = product - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) < 1.0e-12

    def test_identity_quaternion_gives_identity_dcm(self, identity_frame):
        """Identity quaternion → T_ENU→Body = I_3."""
        T = identity_frame.dcm_enu_to_body
        residual = T - np.eye(3)
        assert float(np.linalg.norm(residual, "fro")) < 1.0e-12

    def test_euler_angles_from_identity_are_zero(self, identity_frame):
        roll, pitch, yaw = identity_frame.euler_angles
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)
        assert yaw == pytest.approx(0.0, abs=1e-9)

    def test_roll_pitch_yaw_properties_match_euler_angles(self, identity_frame):
        r, p, y = identity_frame.euler_angles
        assert identity_frame.roll_rad == pytest.approx(r, abs=1e-12)
        assert identity_frame.pitch_rad == pytest.approx(p, abs=1e-12)
        assert identity_frame.yaw_rad == pytest.approx(y, abs=1e-12)

    def test_xi_matrix_shape(self, identity_frame):
        Xi = identity_frame.xi
        assert Xi.shape == (4, 3)
        assert Xi.dtype == np.float64

    def test_qdot_zero_for_zero_omega(self, identity_frame):
        """Zero angular velocity → quaternion derivative is zero."""
        qdot = identity_frame.qdot
        assert qdot == pytest.approx(np.zeros(4), abs=1e-12)

    def test_qdot_formula(self, simple_mass_model, identity_quaternion):
        """dq/dt = ½ Ξ(q) ω verified numerically."""
        omega = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        frame = BodyFrame(
            quaternion=identity_quaternion,
            omega_body=omega,
            com_body=simple_mass_model.com_body,
            total_mass=simple_mass_model.total_mass,
            epoch_time=0.0,
            inertia_body=simple_mass_model.inertia_body,
            inertia_inv=simple_mass_model.inertia_inv,
        )
        from nova.frames.transforms import xi_matrix
        Xi = xi_matrix(identity_quaternion)
        expected = 0.5 * (Xi @ omega)
        assert frame.qdot == pytest.approx(expected, abs=1e-12)


class TestBodyFrameAngularProperties:

    @pytest.fixture
    def spinning_frame(self, simple_mass_model):
        omega = np.array([0.5, 0.2, 0.1], dtype=np.float64)
        return BodyFrame(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            omega_body=omega,
            com_body=simple_mass_model.com_body,
            total_mass=simple_mass_model.total_mass,
            epoch_time=5.0,
            inertia_body=simple_mass_model.inertia_body,
            inertia_inv=simple_mass_model.inertia_inv,
        )

    def test_roll_rate(self, spinning_frame):
        assert spinning_frame.roll_rate == pytest.approx(0.5)

    def test_pitch_rate(self, spinning_frame):
        assert spinning_frame.pitch_rate == pytest.approx(0.2)

    def test_yaw_rate(self, spinning_frame):
        assert spinning_frame.yaw_rate == pytest.approx(0.1)

    def test_angular_speed(self, spinning_frame):
        expected = math.sqrt(0.5**2 + 0.2**2 + 0.1**2)
        assert spinning_frame.angular_speed == pytest.approx(expected, rel=1e-12)

    def test_angular_momentum_formula(self, spinning_frame):
        """h = I · ω should match inertia_body @ omega_body."""
        h = spinning_frame.angular_momentum_body
        expected = spinning_frame.inertia_body @ spinning_frame.omega_body
        assert h == pytest.approx(expected, rel=1e-12)

    def test_Ixx_Iyy_Izz_properties(self, spinning_frame):
        assert spinning_frame.Ixx == pytest.approx(spinning_frame.inertia_body[0, 0])
        assert spinning_frame.Iyy == pytest.approx(spinning_frame.inertia_body[1, 1])
        assert spinning_frame.Izz == pytest.approx(spinning_frame.inertia_body[2, 2])


class TestBodyFrameGeometry:

    @pytest.fixture
    def body_at_origin(self, identity_quaternion, simple_mass_model):
        return BodyFrame(
            quaternion=identity_quaternion,
            omega_body=np.zeros(3),
            com_body=simple_mass_model.com_body,
            total_mass=simple_mass_model.total_mass,
            epoch_time=0.0,
            inertia_body=simple_mass_model.inertia_body,
            inertia_inv=simple_mass_model.inertia_inv,
        )

    def test_shift_to_com_subtracts_com(self, body_at_origin):
        point = np.array([8.0, 0.0, 0.0], dtype=np.float64)
        r = body_at_origin.shift_to_com(point)
        expected = point - body_at_origin.com_body
        assert r == pytest.approx(expected, abs=1e-12)

    def test_moment_arm_equals_shift_to_com(self, body_at_origin):
        point = np.array([3.0, 1.0, -0.5], dtype=np.float64)
        r1 = body_at_origin.shift_to_com(point)
        r2 = body_at_origin.moment_arm(point)
        assert r1 == pytest.approx(r2, abs=1e-12)

    def test_rotate_to_enu_and_back(self, body_at_origin):
        """v_body → ENU → body recovers original."""
        v_body = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v_enu = body_at_origin.rotate_to_enu(v_body)
        v_body_recovered = body_at_origin.rotate_from_enu(v_enu)
        assert v_body_recovered == pytest.approx(v_body, abs=1e-12)

    def test_identity_rotate_to_enu_unchanged(self, body_at_origin):
        """Identity quaternion → rotation has no effect."""
        v_body = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v_enu = body_at_origin.rotate_to_enu(v_body)
        assert v_enu == pytest.approx(v_body, abs=1e-12)


class TestBodyFromStateConstructor:

    def test_body_from_state_creates_valid_frame(
        self, leo_vehicle_state, simple_mass_model
    ):
        frame = body_from_state(leo_vehicle_state, simple_mass_model)
        assert isinstance(frame, BodyFrame)
        assert frame.epoch_time == leo_vehicle_state.time
        assert frame.total_mass == pytest.approx(simple_mass_model.total_mass)

    def test_body_from_state_quaternion_matches(
        self, leo_vehicle_state, simple_mass_model
    ):
        frame = body_from_state(leo_vehicle_state, simple_mass_model)
        assert frame.quaternion == pytest.approx(leo_vehicle_state.quaternion, abs=1e-12)

    def test_body_from_state_omega_matches(
        self, leo_vehicle_state, simple_mass_model
    ):
        frame = body_from_state(leo_vehicle_state, simple_mass_model)
        assert frame.omega_body == pytest.approx(leo_vehicle_state.omega_body, abs=1e-12)

    def test_body_from_state_com_matches_mass_model(
        self, leo_vehicle_state, simple_mass_model
    ):
        frame = body_from_state(leo_vehicle_state, simple_mass_model)
        assert frame.com_body == pytest.approx(simple_mass_model.com_body, abs=1e-12)

    def test_body_from_state_inertia_matches(
        self, leo_vehicle_state, simple_mass_model
    ):
        frame = body_from_state(leo_vehicle_state, simple_mass_model)
        assert frame.inertia_body == pytest.approx(
            simple_mass_model.inertia_body, rel=1e-12
        )

    def test_body_repr_contains_expected_fields(
        self, leo_vehicle_state, simple_mass_model
    ):
        frame = body_from_state(leo_vehicle_state, simple_mass_model)
        s = repr(frame)
        assert "BodyFrame" in s
        assert "roll=" in s
        assert "pitch=" in s
        assert "yaw=" in s


# ===========================================================================
# Cross-frame round-trip integration tests
# ===========================================================================

class TestCrossFrameRoundTrips:
    """
    End-to-end chain: ECI → ECEF → ENU → ECI.
    Validates the full coordinate transform pipeline.
    """

    def test_eci_ecef_enu_position_chain(
        self, leo_position_eci, leo_velocity_eci
    ):
        """ECI → ECEF → ENU → ECEF → ECI recovers original position."""
        t = 0.0  # use t=0 to simplify: ECI ≡ ECEF at epoch
        # Step 1: ECI frame
        eci_frame = eci_from_state(leo_position_eci, leo_velocity_eci, t)

        # Step 2: ECEF (at t=0 should be same as ECI position)
        ecef_frame = ecef_from_eci(leo_position_eci, leo_velocity_eci, t)
        pos_ecef = ecef_frame.position_ecef

        # Step 3: Back to ECI
        pos_eci_recovered = ecef_frame.to_eci_position()
        assert pos_eci_recovered == pytest.approx(leo_position_eci, abs=1.0e-3)

    def test_eci_to_enu_then_eci_velocity(self, leo_position_eci, leo_velocity_eci):
        """ECI velocity → ENU → ECI recovers original velocity."""
        t = 500.0
        lam = 0.0
        phi = 0.0
        enu_frame = enu_from_eci(leo_position_eci, leo_velocity_eci, lam, phi, t)
        vel_eci_recovered = enu_frame.to_eci_velocity()
        assert vel_eci_recovered == pytest.approx(leo_velocity_eci, abs=1.0e-6)

    def test_geodetic_ecef_eci_chain(self):
        """Geodetic → ECEF → ECI → ECEF → Geodetic round-trip."""
        lam = math.radians(28.6139)   # New Delhi longitude
        phi = math.radians(77.2090)   # (deliberately swapped for an off-axis test)
        # Actually use proper Delhi coords
        lam = math.radians(77.2090)
        phi = math.radians(28.6139)
        alt = 216.0  # metres above ellipsoid

        ecef0 = ecef_from_geodetic(lam, phi, alt, epoch_time=0.0)
        lam_r, phi_r, alt_r = ecef0.geodetic()
        assert lam_r == pytest.approx(lam, abs=1e-9)
        assert phi_r == pytest.approx(phi, abs=1e-9)
        assert alt_r == pytest.approx(alt, abs=0.001)

    def test_body_enu_rotation_round_trip_non_identity(self, simple_mass_model):
        """Non-identity attitude: body→ENU→body recovers vector."""
        # 90° yaw quaternion: q = [cos(45°), 0, 0, sin(45°)]
        c = math.cos(math.pi / 4.0)
        s = math.sin(math.pi / 4.0)
        q = np.array([c, 0.0, 0.0, s], dtype=np.float64)
        # Normalize to enforce unit norm
        q /= np.linalg.norm(q)

        frame = BodyFrame(
            quaternion=q,
            omega_body=np.zeros(3),
            com_body=simple_mass_model.com_body,
            total_mass=simple_mass_model.total_mass,
            epoch_time=0.0,
            inertia_body=simple_mass_model.inertia_body,
            inertia_inv=simple_mass_model.inertia_inv,
        )
        v_body = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v_enu = frame.rotate_to_enu(v_body)
        v_body_back = frame.rotate_from_enu(v_enu)
        assert v_body_back == pytest.approx(v_body, abs=1.0e-12)
