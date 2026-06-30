"""
nova.frames.ecef
================
Earth-Centred Earth-Fixed (ECEF) frame container for Project NOVA.

Architectural role
------------------
Phase 7 — Frame Definition Classes.
Pipeline stage: bridge between ECI (orbital mechanics, Stage 6) and ENU
(atmosphere and aerodynamics, Stage 3/7). Consumed by coordinate routing
in the pipeline and by geodetic ground-track computations.

I/O contract
------------
Input  : position_ecef [m], velocity_ecef [m s⁻¹], epoch_time [s]
Output : ECEFFrame instance (frozen dataclass); geodetic accessors;
         transform methods to ECI and ENU.

Physical basis
--------------
The ECEF frame co-rotates with the planet at angular velocity Ω about the
ECI +Z axis. Its origin is at the planetary barycentre:
  +X → prime meridian / equatorial plane intersection
  +Z → geographic north pole
  +Y → completes right-handed triad (≈ 90° east longitude)

The rotation angle at time t is θ = Ω · t. The inverse DCM is its transpose.

Geodetic conversion (ECEF ↔ geodetic longitude/latitude/altitude) uses
the closed-form Bowring iterative formula on the WGS-84 / IAU ellipsoid
defined by EARTH_RADIUS_EQ and EARTH_RADIUS_POLAR from constants.

References
----------
- Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., §3.3
- Zhu, "Conversion of Earth-centered Earth-fixed coordinates to geodetic
  coordinates", IEEE Trans. Aerosp. Electron. Syst., 30(3), 1994
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nova.core.constants import (
    EARTH_MU,
    EARTH_OMEGA,
    EARTH_RADIUS_EQ,
    EARTH_RADIUS_POLAR,
    TRANSFORM_IDENTITY_TOL,
)
from nova.frames.transforms import (
    T_ECI_to_ECEF,
    T_ECEF_to_ECI,
    T_ECEF_to_ENU,
    T_ENU_to_ECEF,
    assert_dcm_orthogonal,
)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

DCM = np.ndarray  # shape (3, 3), dtype float64


# ---------------------------------------------------------------------------
# Internal geodetic helper
# ---------------------------------------------------------------------------

def _ecef_to_geodetic(
    x: float,
    y: float,
    z: float,
    a: float = EARTH_RADIUS_EQ,
    b: float = EARTH_RADIUS_POLAR,
) -> tuple[float, float, float]:
    """
    Convert ECEF Cartesian (x, y, z) to geodetic (longitude, latitude, altitude).

    Uses Bowring's iterative method, accurate to millimetre level on WGS-84.

    Parameters
    ----------
    x, y, z : float
        ECEF Cartesian coordinates [m].
    a : float
        Semi-major (equatorial) axis [m]. Default EARTH_RADIUS_EQ.
    b : float
        Semi-minor (polar) axis [m]. Default EARTH_RADIUS_POLAR.

    Returns
    -------
    longitude_rad : float  [rad]  −π ≤ λ ≤ π
    latitude_rad  : float  [rad]  −π/2 ≤ φ ≤ π/2
    altitude_m    : float  [m]   above the reference ellipsoid
    """
    # Longitude — exact, no iteration needed
    lam = math.atan2(y, x)  # −π … π

    # Ellipsoid parameters
    e2 = 1.0 - (b / a) ** 2              # first eccentricity squared
    ep2 = (a / b) ** 2 - 1.0            # second eccentricity squared
    p = math.sqrt(x * x + y * y)        # distance from Z-axis

    # Special case: on the polar axis
    if p < 1.0e-10:
        phi = math.pi / 2.0 if z >= 0.0 else -math.pi / 2.0
        alt = abs(z) - b
        return lam, phi, alt

    # Bowring iterative latitude (converges in ≤ 5 iterations)
    theta = math.atan2(z * a, p * b)
    for _ in range(10):
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        phi = math.atan2(
            z + ep2 * b * sin_t ** 3,
            p - e2 * a * cos_t ** 3,
        )
        theta_new = math.atan2(math.sin(phi) * b, math.cos(phi) * a)
        if abs(theta_new - theta) < 1.0e-12:
            theta = theta_new
            break
        theta = theta_new

    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    # Radius of curvature in the meridian
    N = a / math.sqrt(1.0 - e2 * sin_phi * sin_phi)
    if abs(cos_phi) > 1.0e-10:
        alt = p / cos_phi - N
    else:
        alt = abs(z) / abs(sin_phi) - N * (1.0 - e2)

    return lam, phi, alt


# ---------------------------------------------------------------------------
# ECEFFrame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ECEFFrame:
    """
    Immutable container for a Cartesian state vector expressed in the
    Earth-Centred Earth-Fixed (ECEF) co-rotating reference frame.

    Attributes
    ----------
    position_ecef : ndarray, shape (3,), dtype float64
        Cartesian position [m] in ECEF.
    velocity_ecef : ndarray, shape (3,), dtype float64
        Velocity [m s⁻¹] in ECEF. This is the velocity as observed in
        the rotating ECEF frame (i.e. ECI velocity minus the frame
        rotation contribution ω × r_ecef).
    epoch_time : float
        Mission-elapsed time [s] when this snapshot was taken. Non-negative.
    body_name : str
        Central body identifier (e.g. "Earth"). Default "Earth".
    omega : float
        Body rotation rate [rad s⁻¹]. Default EARTH_OMEGA.
    radius_eq : float
        Equatorial radius of the reference ellipsoid [m]. Default EARTH_RADIUS_EQ.
    radius_polar : float
        Polar radius of the reference ellipsoid [m]. Default EARTH_RADIUS_POLAR.
    """

    position_ecef: np.ndarray
    velocity_ecef: np.ndarray
    epoch_time: float
    body_name: str = "Earth"
    omega: float = EARTH_OMEGA
    radius_eq: float = EARTH_RADIUS_EQ
    radius_polar: float = EARTH_RADIUS_POLAR

    def __post_init__(self) -> None:
        # --- position_ecef ---
        pos = np.asarray(self.position_ecef, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(
                f"position_ecef must have shape (3,); got {pos.shape}"
            )
        object.__setattr__(self, "position_ecef", pos)

        # --- velocity_ecef ---
        vel = np.asarray(self.velocity_ecef, dtype=np.float64)
        if vel.shape != (3,):
            raise ValueError(
                f"velocity_ecef must have shape (3,); got {vel.shape}"
            )
        object.__setattr__(self, "velocity_ecef", vel)

        # --- epoch_time ---
        t = float(self.epoch_time)
        if t < 0.0:
            raise ValueError(
                f"epoch_time must be non-negative; got {t:.6g}"
            )
        object.__setattr__(self, "epoch_time", t)

        # --- body_name ---
        if not isinstance(self.body_name, str) or not self.body_name.strip():
            raise ValueError("body_name must be a non-empty string")

        # --- omega ---
        omega = float(self.omega)
        if omega < 0.0:
            raise ValueError(
                f"omega must be non-negative; got {omega:.6g}"
            )
        object.__setattr__(self, "omega", omega)

        # --- radii ---
        req = float(self.radius_eq)
        rpolar = float(self.radius_polar)
        if req <= 0.0:
            raise ValueError(f"radius_eq must be positive; got {req:.6g}")
        if rpolar <= 0.0:
            raise ValueError(f"radius_polar must be positive; got {rpolar:.6g}")
        if rpolar > req:
            raise ValueError(
                f"radius_polar ({rpolar:.3f}) must not exceed radius_eq ({req:.3f})"
            )
        object.__setattr__(self, "radius_eq", req)
        object.__setattr__(self, "radius_polar", rpolar)

    # ------------------------------------------------------------------
    # Derived scalar properties
    # ------------------------------------------------------------------

    @property
    def radius(self) -> float:
        """Distance from the barycentre [m]. Always positive."""
        return float(np.linalg.norm(self.position_ecef))

    @property
    def rotation_angle(self) -> float:
        """
        ECEF rotation angle relative to ECI at epoch_time [rad].

        θ = Ω · t  (positive eastward, right-hand rule about +Z)
        """
        return self.omega * self.epoch_time

    # ------------------------------------------------------------------
    # Geodetic conversion
    # ------------------------------------------------------------------

    @property
    def longitude_rad(self) -> float:
        """Geodetic longitude λ [rad].  Range: −π ≤ λ ≤ π."""
        lam, _, _ = _ecef_to_geodetic(
            float(self.position_ecef[0]),
            float(self.position_ecef[1]),
            float(self.position_ecef[2]),
            self.radius_eq,
            self.radius_polar,
        )
        return lam

    @property
    def latitude_rad(self) -> float:
        """Geodetic latitude φ [rad].  Range: −π/2 ≤ φ ≤ π/2."""
        _, phi, _ = _ecef_to_geodetic(
            float(self.position_ecef[0]),
            float(self.position_ecef[1]),
            float(self.position_ecef[2]),
            self.radius_eq,
            self.radius_polar,
        )
        return phi

    @property
    def altitude_m(self) -> float:
        """Geodetic altitude above the reference ellipsoid [m]."""
        _, _, alt = _ecef_to_geodetic(
            float(self.position_ecef[0]),
            float(self.position_ecef[1]),
            float(self.position_ecef[2]),
            self.radius_eq,
            self.radius_polar,
        )
        return alt

    def geodetic(self) -> tuple[float, float, float]:
        """
        Return (longitude_rad, latitude_rad, altitude_m) in a single call.

        This avoids repeating the Bowring iteration three times.

        Returns
        -------
        tuple[float, float, float]
        """
        return _ecef_to_geodetic(
            float(self.position_ecef[0]),
            float(self.position_ecef[1]),
            float(self.position_ecef[2]),
            self.radius_eq,
            self.radius_polar,
        )

    # ------------------------------------------------------------------
    # Frame conversion helpers
    # ------------------------------------------------------------------

    def to_eci_position(self) -> np.ndarray:
        """
        Rotate position_ecef to ECI at epoch_time.

        Returns
        -------
        ndarray, shape (3,), dtype float64
        """
        T = T_ECEF_to_ECI(self.epoch_time, self.omega)
        return (T @ self.position_ecef).astype(np.float64)

    def to_eci_velocity(self) -> np.ndarray:
        """
        Rotate velocity_ecef to ECI at epoch_time.

        Note: this applies only the frame-rotation DCM. If velocity_ecef
        already subtracts ω × r (ECEF-relative velocity), add ω × r_eci
        externally to recover the full inertial velocity.

        Returns
        -------
        ndarray, shape (3,), dtype float64
        """
        T = T_ECEF_to_ECI(self.epoch_time, self.omega)
        return (T @ self.velocity_ecef).astype(np.float64)

    def to_enu_position(self) -> np.ndarray:
        """
        Express position_ecef in the local ENU frame at the geodetic
        reference point (longitude, latitude) of this ECEF position.

        Note: the ENU origin is at the surface reference point, not at the
        barycentre. This returns the rotated ECI→ENU vector, which has the
        correct direction but the origin offset must be handled externally
        for surface-relative range computations.

        Returns
        -------
        ndarray, shape (3,), dtype float64
        """
        lam, phi, _ = self.geodetic()
        T = T_ECEF_to_ENU(lam, phi)
        return (T @ self.position_ecef).astype(np.float64)

    def dcm_to_eci(self) -> DCM:
        """Direction cosine matrix T_ECEF→ECI at epoch_time."""
        T = T_ECEF_to_ECI(self.epoch_time, self.omega)
        assert_dcm_orthogonal(T, "T_ECEF_to_ECI")
        return T

    def dcm_from_eci(self) -> DCM:
        """Direction cosine matrix T_ECI→ECEF at epoch_time."""
        T = T_ECI_to_ECEF(self.epoch_time, self.omega)
        assert_dcm_orthogonal(T, "T_ECI_to_ECEF")
        return T

    def dcm_to_enu(self) -> DCM:
        """
        Direction cosine matrix T_ECEF→ENU at the geodetic reference point.
        """
        lam, phi, _ = self.geodetic()
        T = T_ECEF_to_ENU(lam, phi)
        assert_dcm_orthogonal(T, "T_ECEF_to_ENU")
        return T

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lam, phi, alt = self.geodetic()
        lam_deg = math.degrees(lam)
        phi_deg = math.degrees(phi)
        alt_km = alt / 1_000.0
        return (
            f"ECEFFrame(body={self.body_name!r}, t={self.epoch_time:.3f}s, "
            f"lon={lam_deg:.4f}°, lat={phi_deg:.4f}°, alt={alt_km:.3f} km)"
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def ecef_from_eci(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
    epoch_time: float,
    body_name: str = "Earth",
    omega: float = EARTH_OMEGA,
    radius_eq: float = EARTH_RADIUS_EQ,
    radius_polar: float = EARTH_RADIUS_POLAR,
) -> ECEFFrame:
    """
    Construct an ECEFFrame by rotating ECI state vectors to ECEF.

    The ECEF velocity returned is the rotated ECI velocity minus the
    frame velocity contribution ω × r_ecef (i.e. ECEF-relative velocity).

    Parameters
    ----------
    position_eci : array_like, shape (3,)
        ECI position [m].
    velocity_eci : array_like, shape (3,)
        ECI velocity [m s⁻¹].
    epoch_time : float
        Mission-elapsed time [s].
    body_name : str
        Central body name. Default "Earth".
    omega : float
        Rotation rate [rad s⁻¹]. Default EARTH_OMEGA.
    radius_eq : float
        Equatorial radius [m]. Default EARTH_RADIUS_EQ.
    radius_polar : float
        Polar radius [m]. Default EARTH_RADIUS_POLAR.

    Returns
    -------
    ECEFFrame
    """
    pos_eci = np.asarray(position_eci, dtype=np.float64)
    vel_eci = np.asarray(velocity_eci, dtype=np.float64)

    T = T_ECI_to_ECEF(float(epoch_time), omega)
    pos_ecef = T @ pos_eci

    # v_ecef = T @ v_eci  −  ω × r_ecef
    omega_vec = np.array([0.0, 0.0, omega], dtype=np.float64)
    vel_ecef = T @ vel_eci - np.cross(omega_vec, pos_ecef)

    return ECEFFrame(
        position_ecef=pos_ecef.astype(np.float64),
        velocity_ecef=vel_ecef.astype(np.float64),
        epoch_time=float(epoch_time),
        body_name=body_name,
        omega=omega,
        radius_eq=radius_eq,
        radius_polar=radius_polar,
    )


def ecef_from_geodetic(
    longitude_rad: float,
    latitude_rad: float,
    altitude_m: float,
    velocity_enu: np.ndarray | None = None,
    epoch_time: float = 0.0,
    body_name: str = "Earth",
    omega: float = EARTH_OMEGA,
    radius_eq: float = EARTH_RADIUS_EQ,
    radius_polar: float = EARTH_RADIUS_POLAR,
) -> ECEFFrame:
    """
    Construct an ECEFFrame from geodetic coordinates.

    Parameters
    ----------
    longitude_rad : float
        Geodetic longitude λ [rad].
    latitude_rad : float
        Geodetic latitude φ [rad].
    altitude_m : float
        Altitude above reference ellipsoid [m].
    velocity_enu : array_like, shape (3,) or None
        Velocity in ENU frame [m s⁻¹]. Zero if None.
    epoch_time : float
        Mission-elapsed time [s]. Default 0.
    body_name : str
        Central body name. Default "Earth".
    omega, radius_eq, radius_polar : float
        Ellipsoid and rotation parameters.

    Returns
    -------
    ECEFFrame
    """
    a = radius_eq
    b = radius_polar
    e2 = 1.0 - (b / a) ** 2

    lam = float(longitude_rad)
    phi = float(latitude_rad)
    h = float(altitude_m)

    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)

    N = a / math.sqrt(1.0 - e2 * sin_phi ** 2)  # radius of curvature

    x = (N + h) * cos_phi * cos_lam
    y = (N + h) * cos_phi * sin_lam
    z = (N * (1.0 - e2) + h) * sin_phi

    pos_ecef = np.array([x, y, z], dtype=np.float64)

    if velocity_enu is None:
        vel_ecef = np.zeros(3, dtype=np.float64)
    else:
        T_enu_to_ecef = T_ENU_to_ECEF(lam, phi)
        vel_ecef = (T_enu_to_ecef @ np.asarray(velocity_enu, dtype=np.float64)).astype(np.float64)

    return ECEFFrame(
        position_ecef=pos_ecef,
        velocity_ecef=vel_ecef,
        epoch_time=float(epoch_time),
        body_name=body_name,
        omega=omega,
        radius_eq=radius_eq,
        radius_polar=radius_polar,
    )
