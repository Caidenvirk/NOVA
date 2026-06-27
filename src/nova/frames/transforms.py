"""
nova.frames.transforms
======================
All coordinate frame transformation matrices for Project NOVA.

Four reference frames are implemented, with explicit documented
transformations between every adjacent pair:

  ECI  ──T_ECI_to_ECEF──▶  ECEF  ──T_ECEF_to_ENU──▶  ENU  ──T_ENU_to_body──▶  Body

Every transformation function returns a (3, 3) float64 NumPy array
representing the rotation matrix **T** such that:

    v_B = T_A_to_B @ v_A

All inverse transformations are the matrix transpose (all rotation matrices
are orthogonal: T⁻¹ = Tᵀ), verified within TRANSFORM_IDENTITY_TOL.

Angle convention
----------------
All angles are in RADIANS. No function accepts degrees. If you have
degrees, multiply by ``nova.core.constants.DEG_TO_RAD`` before calling.

Quaternion convention
---------------------
q = (q0, q1, q2, q3) — scalar part first (Hamilton convention).
‖q‖ = 1 is assumed; no internal renormalisation is performed here.

Frame axis conventions
----------------------
ECI:   X → vernal equinox, Z → north celestial pole, Y completes RH
ECEF:  X → prime meridian/equator, Z → north pole, Y completes RH
ENU:   X → East, Y → North, Z → Up  (local tangent plane)
Body:  X → forward (longitudinal), Y → right (lateral), Z → down (normal)
       Right-hand NED-aligned body frame.

References
----------
- Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., §3
- Stevens & Lewis, "Aircraft Control and Simulation", §2
- Titterton & Weston, "Strapdown Inertial Navigation Technology", §3
"""

from __future__ import annotations

import math
import numpy as np

from nova.core.constants import (
    EARTH_OMEGA,
    TRANSFORM_IDENTITY_TOL,
)

# ---------------------------------------------------------------------------
# Type alias (for clarity in signatures)
# ---------------------------------------------------------------------------

DCM = np.ndarray   # Direction Cosine Matrix: shape (3, 3), dtype float64


# ---------------------------------------------------------------------------
# 1. ECI → ECEF   T(t)
# ---------------------------------------------------------------------------

def T_ECI_to_ECEF(time: float, omega: float = EARTH_OMEGA) -> DCM:
    """
    Rotation matrix from Earth-Centred Inertial (ECI) to
    Earth-Centred Earth-Fixed (ECEF) frame.

    The ECEF frame rotates about the ECI Z-axis with angular velocity
    ``omega`` [rad s⁻¹]. The rotation angle θ = omega · t increases
    eastward (right-hand rule about +Z).

    Parameters
    ----------
    time : float
        Elapsed simulation time [s]. t=0 assumes ECI and ECEF are aligned
        (Greenwich sidereal time = 0 at epoch; sufficient for relative
        trajectory analysis — absolute GAST alignment handled by the
        orbital initialiser).
    omega : float
        Planetary rotation rate [rad s⁻¹]. Defaults to Earth's sidereal
        rotation rate (7.292 115 × 10⁻⁵ rad s⁻¹).

    Returns
    -------
    DCM : ndarray, shape (3, 3), dtype float64

    Matrix definition
    -----------------
    θ = omega · t

    T_ECI→ECEF = ⎡  cos θ   sin θ   0 ⎤
                 ⎢ −sin θ   cos θ   0 ⎥
                 ⎣   0        0     1 ⎦

    Note: some texts define this as the *transpose* (ECI components
    expressed in ECEF). Here we follow the active rotation convention:
    apply this matrix to an ECI vector to get the ECEF vector.
    """
    theta = omega * time
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [ c,  s,  0.0],
        [-s,  c,  0.0],
        [ 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def T_ECEF_to_ECI(time: float, omega: float = EARTH_OMEGA) -> DCM:
    """
    Inverse of T_ECI_to_ECEF — rotate ECEF vector to ECI.

    Mathematically: T_ECEF→ECI = T_ECI→ECEF(t)ᵀ
    """
    return T_ECI_to_ECEF(time, omega).T


# ---------------------------------------------------------------------------
# 2. ECEF → ENU   T(λ, φ)
# ---------------------------------------------------------------------------

def T_ECEF_to_ENU(longitude_rad: float, latitude_rad: float) -> DCM:
    """
    Rotation matrix from ECEF to local East-North-Up (ENU) frame.

    The ENU frame is tangent to the planet ellipsoid at the point defined
    by (longitude, latitude). It is standard for launch-pad and
    atmospheric-flight operations.

    Parameters
    ----------
    longitude_rad : float
        Geodetic longitude λ [rad], positive east of prime meridian.
    latitude_rad : float
        Geodetic latitude φ [rad], positive north of equator.

    Returns
    -------
    DCM : ndarray, shape (3, 3), dtype float64

    Matrix definition
    -----------------
    λ = longitude,  φ = latitude

    T_ECEF→ENU = ⎡  −sin λ            cos λ            0      ⎤
                 ⎢  −sin φ cos λ      −sin φ sin λ     cos φ   ⎥
                 ⎣   cos φ cos λ       cos φ sin λ     sin φ   ⎦

    Row 0 = East unit vector expressed in ECEF
    Row 1 = North unit vector expressed in ECEF
    Row 2 = Up unit vector expressed in ECEF

    This transforms an ECEF column vector [x, y, z]ᵀ to ENU
    [East, North, Up]ᵀ.
    """
    lam = longitude_rad    # λ
    phi = latitude_rad     # φ

    sl, cl = math.sin(lam), math.cos(lam)
    sp, cp = math.sin(phi), math.cos(phi)

    return np.array([
        [      -sl,        cl,    0.0],
        [-sp * cl,  -sp * sl,    cp ],
        [ cp * cl,   cp * sl,    sp ],
    ], dtype=np.float64)


def T_ENU_to_ECEF(longitude_rad: float, latitude_rad: float) -> DCM:
    """
    Inverse of T_ECEF_to_ENU — rotate ENU vector to ECEF.

    Mathematically: T_ENU→ECEF = T_ECEF→ENU(λ, φ)ᵀ
    """
    return T_ECEF_to_ENU(longitude_rad, latitude_rad).T


# ---------------------------------------------------------------------------
# 3. ENU → Body   R(q)  via unit quaternion
# ---------------------------------------------------------------------------

def T_ENU_to_body(quaternion: np.ndarray) -> DCM:
    """
    Direction Cosine Matrix (DCM) rotating the ENU frame to the
    vehicle Body Frame, derived from a unit quaternion.

    The quaternion encodes the vehicle's attitude relative to the local
    ENU frame. At q = [1, 0, 0, 0] (identity), the body axes are aligned
    with ENU (X_body = East, Y_body = North, Z_body = Up — note that
    Z_body = Up means Z_body is inverted relative to the NED-down convention;
    the NED-to-body final flip is applied separately in full 6-DoF pipelines
    but is NOT included here to keep this function general).

    Parameters
    ----------
    quaternion : ndarray, shape (4,), dtype float64
        Unit quaternion (q0, q1, q2, q3) — scalar-first Hamilton convention.
        ‖q‖ must equal 1.0 (not enforced here; enforced by VehicleState).

    Returns
    -------
    DCM : ndarray, shape (3, 3), dtype float64

    Matrix definition
    -----------------
    q = (q0, q1, q2, q3)

    R(q) = ⎡ 1−2(q2²+q3²)    2(q1q2−q0q3)    2(q1q3+q0q2) ⎤
           ⎢ 2(q1q2+q0q3)   1−2(q1²+q3²)     2(q2q3−q0q1) ⎥
           ⎣ 2(q1q3−q0q2)    2(q2q3+q0q1)   1−2(q1²+q2²) ⎦

    This is the standard Shepperd/Diebel DCM from a Hamilton unit quaternion.
    """
    q = quaternion.astype(np.float64)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    q0q0, q1q1, q2q2, q3q3 = q0*q0, q1*q1, q2*q2, q3*q3
    q0q1, q0q2, q0q3       = q0*q1, q0*q2, q0*q3
    q1q2, q1q3, q2q3       = q1*q2, q1*q3, q2*q3

    return np.array([
        [1.0 - 2.0*(q2q2 + q3q3),  2.0*(q1q2 - q0q3),  2.0*(q1q3 + q0q2)],
        [      2.0*(q1q2 + q0q3),  1.0 - 2.0*(q1q1 + q3q3),  2.0*(q2q3 - q0q1)],
        [      2.0*(q1q3 - q0q2),  2.0*(q2q3 + q0q1),  1.0 - 2.0*(q1q1 + q2q2)],
    ], dtype=np.float64)


def T_body_to_ENU(quaternion: np.ndarray) -> DCM:
    """
    Inverse of T_ENU_to_body — rotate Body Frame vector to ENU.

    Mathematically: T_body→ENU = T_ENU→body(q)ᵀ = R(q)ᵀ = R(q⁻¹)
    """
    return T_ENU_to_body(quaternion).T


# ---------------------------------------------------------------------------
# 4. Composed transformations (ECI → Body and inverses)
# ---------------------------------------------------------------------------

def T_ECI_to_body(
    quaternion: np.ndarray,
    longitude_rad: float,
    latitude_rad: float,
    time: float,
    omega: float = EARTH_OMEGA,
) -> DCM:
    """
    Full ECI → Body frame transformation (composed product).

    T_ECI→Body = T_ENU→Body(q) @ T_ECEF→ENU(λ, φ) @ T_ECI→ECEF(t)

    Parameters
    ----------
    quaternion : ndarray, shape (4,)
        Vehicle attitude quaternion in ENU frame (scalar-first).
    longitude_rad : float
        Surface longitude of reference point [rad].
    latitude_rad : float
        Surface latitude of reference point [rad].
    time : float
        Elapsed simulation time [s].
    omega : float
        Planet rotation rate [rad s⁻¹].

    Returns
    -------
    DCM : ndarray, shape (3, 3), dtype float64
    """
    R_eci_to_ecef = T_ECI_to_ECEF(time, omega)
    R_ecef_to_enu = T_ECEF_to_ENU(longitude_rad, latitude_rad)
    R_enu_to_body = T_ENU_to_body(quaternion)
    return R_enu_to_body @ R_ecef_to_enu @ R_eci_to_ecef


def T_body_to_ECI(
    quaternion: np.ndarray,
    longitude_rad: float,
    latitude_rad: float,
    time: float,
    omega: float = EARTH_OMEGA,
) -> DCM:
    """
    Full Body → ECI frame transformation.

    T_Body→ECI = T_ECI→Body(...)ᵀ
    """
    return T_ECI_to_body(quaternion, longitude_rad, latitude_rad, time, omega).T


# ---------------------------------------------------------------------------
# 5. Quaternion kinematics matrix Ξ(q)
# ---------------------------------------------------------------------------

def xi_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    The 4×3 quaternion kinematic matrix Ξ(q) used in the attitude ODE:

        dq/dt = ½ · Ξ(q) · ω_body

    where ω_body = [p, q, r]ᵀ is the angular velocity in the Body Frame.

    Parameters
    ----------
    quaternion : ndarray, shape (4,)
        Unit quaternion (q0, q1, q2, q3), scalar-first.

    Returns
    -------
    Xi : ndarray, shape (4, 3), dtype float64

    Matrix definition
    -----------------
              ⎡ −q1   −q2   −q3 ⎤
    Ξ(q) =   ⎢  q0   −q3    q2 ⎥
              ⎢  q3    q0   −q1 ⎥
              ⎣ −q2    q1    q0 ⎦

    Reference: Diebel (2006), "Representing Attitude", §5.4
    """
    q = quaternion.astype(np.float64)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    return np.array([
        [-q1, -q2, -q3],
        [ q0, -q3,  q2],
        [ q3,  q0, -q1],
        [-q2,  q1,  q0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# 6. Euler angles ↔ quaternion (display use only — NOT used internally)
# ---------------------------------------------------------------------------

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert ZYX Tait-Bryan Euler angles to a unit quaternion.

    Rotation order: yaw (ψ) first, then pitch (θ), then roll (φ).
    This is the standard aerospace ZYX 3-2-1 sequence.

    Parameters
    ----------
    roll : float   [rad]  φ — rotation about X_body
    pitch : float  [rad]  θ — rotation about Y_body
    yaw : float    [rad]  ψ — rotation about Z_body

    Returns
    -------
    q : ndarray, shape (4,), dtype float64, ‖q‖ = 1
        (q0, q1, q2, q3) — scalar-first.

    WARNING: Euler angles suffer from gimbal lock at pitch = ±90°.
    This function is provided for initial condition setup and display
    only. The integrator operates exclusively on quaternions.
    """
    cr, sr = math.cos(roll  * 0.5), math.sin(roll  * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw   * 0.5), math.sin(yaw   * 0.5)

    q0 = cr*cp*cy + sr*sp*sy
    q1 = sr*cp*cy - cr*sp*sy
    q2 = cr*sp*cy + sr*cp*sy
    q3 = cr*cp*sy - sr*sp*cy

    return np.array([q0, q1, q2, q3], dtype=np.float64)


def quaternion_to_euler(quaternion: np.ndarray) -> tuple[float, float, float]:
    """
    Convert a unit quaternion to ZYX Tait-Bryan Euler angles.

    Returns
    -------
    (roll, pitch, yaw) : tuple of float, all in [rad]

    WARNING: Gimbal lock occurs at pitch = ±π/2. Output is undefined there.
    For display use only — never feed these back into the integrator.
    """
    q = quaternion.astype(np.float64)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    # Roll (φ)
    sinr_cosp = 2.0 * (q0*q1 + q2*q3)
    cosr_cosp = 1.0 - 2.0 * (q1*q1 + q2*q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (θ) — clamp for numerical safety at gimbal lock
    sinp = 2.0 * (q0*q2 - q3*q1)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # Yaw (ψ)
    siny_cosp = 2.0 * (q0*q3 + q1*q2)
    cosy_cosp = 1.0 - 2.0 * (q2*q2 + q3*q3)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# 7. Validation utility
# ---------------------------------------------------------------------------

def assert_dcm_orthogonal(dcm: DCM, name: str = "DCM") -> None:
    """
    Assert that a DCM is orthogonal: ‖T @ Tᵀ − I‖_F ≤ TRANSFORM_IDENTITY_TOL.

    Raises
    ------
    AssertionError if the Frobenius norm of (T @ Tᵀ − I₃) exceeds tolerance.
    """
    residual = dcm @ dcm.T - np.eye(3, dtype=np.float64)
    frob = float(np.linalg.norm(residual, "fro"))
    assert frob <= TRANSFORM_IDENTITY_TOL, (
        f"{name} orthogonality check failed: "
        f"‖T·Tᵀ − I‖_F = {frob:.3e} > {TRANSFORM_IDENTITY_TOL:.3e}"
    )n
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_transforms.py
=============================
Unit tests for nova.frames.transforms.

Tests verify (per architecture validation matrix):
  1. All DCMs are orthogonal:  ‖T · Tᵀ − I₃‖_F ≤ TRANSFORM_IDENTITY_TOL
  2. Roundtrip identity:  T_A→B · T_B→A = I₃
  3. Known-value checks: identity quaternion → I₃; specific rotations →
     analytically expected matrices.
  4. Quaternion kinematic matrix Ξ(q) dimension and structure.
  5. Euler ↔ quaternion roundtrip for non-degenerate angles.
  6. Composed ECI → Body transform inherits orthogonality.
"""

import math
import pytest
import numpy as np

from nova.core.constants import (
    EARTH_OMEGA, TRANSFORM_IDENTITY_TOL, DEG_TO_RAD,
)
from nova.frames.transforms import (
    T_ECI_to_ECEF, T_ECEF_to_ECI,
    T_ECEF_to_ENU, T_ENU_to_ECEF,
    T_ENU_to_body, T_body_to_ENU,
    T_ECI_to_body, T_body_to_ECI,
    xi_matrix,
    euler_to_quaternion, quaternion_to_euler,
    assert_dcm_orthogonal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frob_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, "fro"))


def _identity_error(T: np.ndarray) -> float:
    """‖T · Tᵀ − I₃‖_F"""
    return _frob_norm(T @ T.T - np.eye(3))


# ---------------------------------------------------------------------------
# 1. ECI → ECEF
# ---------------------------------------------------------------------------

class TestECItoECEF:

    def test_identity_at_t0(self):
        """At t=0, ECI and ECEF coincide → T = I₃."""
        T = T_ECI_to_ECEF(time=0.0)
        assert np.allclose(T, np.eye(3), atol=1.0e-15)

    def test_orthogonal_at_various_times(self):
        for t in [0.0, 100.0, 3600.0, 86164.0]:
            T = T_ECI_to_ECEF(time=t)
            assert _identity_error(T) <= TRANSFORM_IDENTITY_TOL, \
                f"Not orthogonal at t={t}s: error={_identity_error(T):.2e}"

    def test_full_rotation_at_sidereal_period(self):
        """After one sidereal period T_sidereal = 2π/Ω, the matrix returns to I₃."""
        T_sidereal = 2.0 * math.pi / EARTH_OMEGA
        T = T_ECI_to_ECEF(time=T_sidereal)
        assert np.allclose(T, np.eye(3), atol=1.0e-10), \
            f"Sidereal period roundtrip failed:\n{T}"

    def test_custom_omega(self):
        """Should accept arbitrary planet rotation rates."""
        omega_moon = 2.0 * math.pi / (27.3 * 86400.0)
        T = T_ECI_to_ECEF(time=1000.0, omega=omega_moon)
        assert _identity_error(T) <= TRANSFORM_IDENTITY_TOL

    def test_roundtrip_ECI_ECEF(self):
        """T_ECI→ECEF · T_ECEF→ECI = I₃."""
        for t in [0.0, 500.0, 43082.0]:
            T_fwd = T_ECI_to_ECEF(time=t)
            T_inv = T_ECEF_to_ECI(time=t)
            product = T_fwd @ T_inv
            error = _frob_norm(product - np.eye(3))
            assert error <= TRANSFORM_IDENTITY_TOL, \
                f"Roundtrip failed at t={t}s: error={error:.2e}"

    def test_inverse_is_transpose(self):
        """Orthogonal matrix inverse equals transpose."""
        T_fwd = T_ECI_to_ECEF(time=12345.0)
        T_inv = T_ECEF_to_ECI(time=12345.0)
        assert np.allclose(T_inv, T_fwd.T, atol=1.0e-15)

    def test_only_z_axis_rotation(self):
        """
        ECEF rotates about the ECI Z axis — the Z component of any vector
        must be unchanged by the transformation.
        """
        v_eci = np.array([0.0, 0.0, 1.0])  # pure Z vector
        T = T_ECI_to_ECEF(time=5000.0)
        v_ecef = T @ v_eci
        assert abs(v_ecef[2] - 1.0) < 1.0e-14
        assert abs(v_ecef[0]) < 1.0e-14
        assert abs(v_ecef[1]) < 1.0e-14


# ---------------------------------------------------------------------------
# 2. ECEF → ENU
# ---------------------------------------------------------------------------

class TestECEFtoENU:

    def test_orthogonal_at_prime_meridian_equator(self):
        T = T_ECEF_to_ENU(longitude_rad=0.0, latitude_rad=0.0)
        assert _identity_error(T) <= TRANSFORM_IDENTITY_TOL

    def test_orthogonal_at_arbitrary_point(self):
        # Longitude 45°E, Latitude 51.5°N (London-ish)
        lon = 0.0 * DEG_TO_RAD
        lat = 51.5 * DEG_TO_RAD
        T = T_ECEF_to_ENU(longitude_rad=lon, latitude_rad=lat)
        assert _identity_error(T) <= TRANSFORM_IDENTITY_TOL

    def test_roundtrip_ECEF_ENU(self):
        for lon, lat in [(0.0, 0.0), (1.2, 0.5), (-0.5, -0.8), (3.0, 1.0)]:
            T_fwd = T_ECEF_to_ENU(lon, lat)
            T_inv = T_ENU_to_ECEF(lon, lat)
            error = _frob_norm(T_fwd @ T_inv - np.eye(3))
            assert error <= TRANSFORM_IDENTITY_TOL, \
                f"Roundtrip ECEF-ENU failed at ({lon:.2f},{lat:.2f}): error={error:.2e}"

    def test_known_value_north_pole(self):
        """
        At the geographic North Pole (lat=π/2), the Up direction in ENU
        should point along the ECEF +Z axis (which equals [0, 0, 1] in ECEF).

        Row 2 of T_ECEF→ENU = Up unit vector in ECEF coords.
        """
        T = T_ECEF_to_ENU(longitude_rad=0.0, latitude_rad=math.pi / 2.0)
        up_in_ecef = T[2, :]
        assert np.allclose(up_in_ecef, [0.0, 0.0, 1.0], atol=1.0e-14), \
            f"North pole Up vector wrong: {up_in_ecef}"

    def test_known_value_equator_prime_meridian(self):
        """
        At (lon=0, lat=0) the ECEF origin is at the prime meridian equator.
        Up = +X_ecef, East = +Y_ecef, North = +Z_ecef.
        """
        T = T_ECEF_to_ENU(longitude_rad=0.0, latitude_rad=0.0)
        # Row 0 = East = should be [0, 1, 0] in ECEF
        east = T[0, :]
        assert np.allclose(east, [0.0, 1.0, 0.0], atol=1.0e-14), \
            f"East vector at (0,0) wrong: {east}"
        # Row 2 = Up = should be [1, 0, 0] in ECEF
        up = T[2, :]
        assert np.allclose(up, [1.0, 0.0, 0.0], atol=1.0e-14), \
            f"Up vector at (0,0) wrong: {up}"


# ---------------------------------------------------------------------------
# 3. ENU → Body (quaternion DCM)
# ---------------------------------------------------------------------------

class TestENUtoBody:

    def test_identity_quaternion_gives_identity_dcm(self):
        """q = [1, 0, 0, 0] → R(q) = I₃."""
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        R = T_ENU_to_body(q)
        assert np.allclose(R, np.eye(3), atol=1.0e-15)

    def test_orthogonal_identity_quaternion(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        R = T_ENU_to_body(q)
        assert _identity_error(R) <= TRANSFORM_IDENTITY_TOL

    def test_orthogonal_90deg_yaw(self):
        """90° rotation about Z: q = [cos45°, 0, 0, sin45°]."""
        angle = math.pi / 2.0
        q = np.array([math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)], dtype=np.float64)
        R = T_ENU_to_body(q)
        assert _identity_error(R) <= TRANSFORM_IDENTITY_TOL

    def test_90deg_yaw_known_matrix(self):
        """
        90° yaw (rotation about +Z) should map:
          X_ENU (+East)  → +Y_body (right lateral)
          Y_ENU (+North) → −X_body (aft)
        In the DCM: R @ [1,0,0] = [0,1,0], R @ [0,1,0] = [-1,0,0].
        """
        angle = math.pi / 2.0
        q = np.array([math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)], dtype=np.float64)
        R = T_ENU_to_body(q)
        x_mapped = R @ np.array([1.0, 0.0, 0.0])
        y_mapped = R @ np.array([0.0, 1.0, 0.0])
        assert np.allclose(x_mapped, [0.0, 1.0, 0.0], atol=1.0e-14), \
            f"X_ENU mapped to {x_mapped}, expected [0,1,0]"
        assert np.allclose(y_mapped, [-1.0, 0.0, 0.0], atol=1.0e-14), \
            f"Y_ENU mapped to {y_mapped}, expected [-1,0,0]"

    def test_roundtrip_ENU_body(self):
        """T_ENU→Body · T_Body→ENU = I₃ for arbitrary quaternion."""
        q = euler_to_quaternion(roll=0.3, pitch=-0.2, yaw=1.1)
        T_fwd = T_ENU_to_body(q)
        T_inv = T_body_to_ENU(q)
        error = _frob_norm(T_fwd @ T_inv - np.eye(3))
        assert error <= TRANSFORM_IDENTITY_TOL, f"ENU-Body roundtrip error={error:.2e}"

    def test_inverse_is_transpose(self):
        q = euler_to_quaternion(roll=0.1, pitch=0.3, yaw=-0.5)
        T_fwd = T_ENU_to_body(q)
        T_inv = T_body_to_ENU(q)
        assert np.allclose(T_inv, T_fwd.T, atol=1.0e-15)


# ---------------------------------------------------------------------------
# 4. Composed ECI → Body
# ---------------------------------------------------------------------------

class TestComposedTransforms:

    def test_ECI_to_body_orthogonal(self):
        q = euler_to_quaternion(roll=0.2, pitch=0.1, yaw=0.5)
        T = T_ECI_to_body(q, longitude_rad=0.3, latitude_rad=0.8, time=500.0)
        assert _identity_error(T) <= TRANSFORM_IDENTITY_TOL

    def test_ECI_to_body_roundtrip(self):
        q   = euler_to_quaternion(roll=0.0, pitch=0.2, yaw=-0.4)
        lon, lat, t = 1.0, 0.5, 3600.0
        T_fwd = T_ECI_to_body(q, lon, lat, t)
        T_inv = T_body_to_ECI(q, lon, lat, t)
        error = _frob_norm(T_fwd @ T_inv - np.eye(3))
        assert error <= TRANSFORM_IDENTITY_TOL, \
            f"ECI-Body composed roundtrip error={error:.2e}"

    def test_assert_dcm_orthogonal_passes_for_valid_dcm(self):
        T = T_ECI_to_ECEF(time=0.0)
        assert_dcm_orthogonal(T, name="T_ECI_ECEF")   # should not raise

    def test_assert_dcm_orthogonal_fails_for_corrupted_dcm(self):
        T = np.eye(3, dtype=np.float64)
        T[0, 0] = 2.0   # deliberately non-orthogonal
        with pytest.raises(AssertionError, match="orthogonality"):
            assert_dcm_orthogonal(T, name="corrupted")


# ---------------------------------------------------------------------------
# 5. Quaternion kinematic matrix Ξ(q)
# ---------------------------------------------------------------------------

class TestXiMatrix:

    def test_shape(self):
        q  = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        Xi = xi_matrix(q)
        assert Xi.shape == (4, 3)
        assert Xi.dtype == np.float64

    def test_identity_quaternion_xi(self):
        """For q=[1,0,0,0]: Ξ(q) @ ω = [0, ωx, ωy, ωz] (scalar dot product zero)."""
        q   = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        Xi  = xi_matrix(q)
        omega = np.array([0.1, 0.2, 0.3])
        qdot  = 0.5 * (Xi @ omega)
        # Scalar component (qdot[0]) = -½ (q1*p + q2*q + q3*r) = 0 for identity q
        assert abs(qdot[0]) < 1.0e-15

    def test_qdot_preserves_norm_to_first_order(self):
        """
        For a unit quaternion q with angular rate ω, the kinematic equation
        q̇ = ½ Ξ(q) ω must satisfy q · q̇ = 0 (norm-preserving to first order).

        Because ‖q + ε q̇‖² = ‖q‖² + 2ε (q·q̇) + O(ε²), the first-order
        norm change is zero iff q·q̇ = 0.
        """
        q     = euler_to_quaternion(0.3, 0.2, 0.5)
        omega = np.array([0.05, -0.03, 0.08])
        Xi    = xi_matrix(q)
        q_dot = 0.5 * (Xi @ omega)
        # q · q̇ must be zero (orthogonality of q and q̇)
        inner = float(np.dot(q, q_dot))
        assert abs(inner) < 1.0e-14, \
            f"Quaternion norm not preserved to first order: q·q̇ = {inner:.2e}"


# ---------------------------------------------------------------------------
# 6. Euler ↔ quaternion
# ---------------------------------------------------------------------------

class TestEulerQuaternion:

    @pytest.mark.parametrize("roll,pitch,yaw", [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (0.0, 0.0, 1.2),
        (0.3, -0.2, 0.8),
        (-0.1, 0.5, -1.5),
    ])
    def test_euler_roundtrip(self, roll, pitch, yaw):
        """euler→quat→euler roundtrip (below gimbal lock)."""
        q = euler_to_quaternion(roll, pitch, yaw)
        roll2, pitch2, yaw2 = quaternion_to_euler(q)
        assert abs(roll2  - roll)  < 1.0e-12, f"Roll mismatch: {roll2} vs {roll}"
        assert abs(pitch2 - pitch) < 1.0e-12, f"Pitch mismatch: {pitch2} vs {pitch}"
        assert abs(yaw2   - yaw)   < 1.0e-12, f"Yaw mismatch: {yaw2} vs {yaw}"

    def test_output_quaternion_is_unit(self):
        for args in [(0.1, 0.2, 0.3), (-0.5, 0.5, -0.5), (0.0, 0.0, math.pi)]:
            q = euler_to_quaternion(*args)
            assert abs(np.linalg.norm(q) - 1.0) < 1.0e-14

    def test_zero_euler_gives_identity_quat(self):
        q = euler_to_quaternion(0.0, 0.0, 0.0)
        assert np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1.0e-15)

    def test_identity_quat_gives_zero_euler(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        roll, pitch, yaw = quaternion_to_euler(q)
        assert abs(roll)  < 1.0e-14
        assert abs(pitch) < 1.0e-14
        assert abs(yaw)   < 1.0e-14

    def test_180_yaw(self):
        q = euler_to_quaternion(0.0, 0.0, math.pi)
        norm = np.linalg.norm(q)
        assert abs(norm - 1.0) < 1.0e-14
        roll, pitch, yaw = quaternion_to_euler(q)
        assert abs(yaw - math.pi) < 1.0e-12
