"""
nova.frames.enu
===============
East-North-Up (ENU) local topocentric frame container for Project NOVA.

Architectural role
------------------
Phase 7 — Frame Definition Classes.
Pipeline stage: bridge between ECEF (planet-fixed, Stage geodetic) and Body
(vehicle attitude, Stage 3). The ENU frame is the natural reference for
atmospheric flight: altitude, heading, and aerodynamic angles are all
defined relative to a surface-tangent plane.

I/O contract
------------
Input  : position_enu [m], velocity_enu [m s⁻¹], reference longitude/latitude,
         epoch_time [s]
Output : ENUFrame instance (frozen dataclass); convenience accessors for
         altitude, bearing, range-from-origin; transform methods to ECEF/ECI.

Physical basis
--------------
The ENU frame is anchored to a reference geodetic point (λ_ref, φ_ref) on
the surface ellipsoid. Its axes are:
  +X → East  (tangent to the parallel at reference latitude)
  +Y → North (tangent to the meridian at reference latitude)
  +Z → Up    (outward normal to the reference ellipsoid)

This is a right-handed Cartesian system. It is non-inertial (rotates with
ECEF), so ENU vectors must be converted to ECI before feeding the integrator.

The transformation chain is:
  ECI → (T_ECI_to_ECEF(t)) → ECEF → (T_ECEF_to_ENU(λ,φ)) → ENU

References
----------
- Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., §3.4
- Titterton & Weston, "Strapdown Inertial Navigation Technology", §3.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nova.core.constants import (
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
# ENUFrame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ENUFrame:
    """
    Immutable container for a state vector expressed in the local
    East-North-Up (ENU) topocentric frame.

    The frame is anchored to a fixed geodetic reference point
    (ref_longitude_rad, ref_latitude_rad). All position and velocity
    components are expressed relative to that reference origin.

    Attributes
    ----------
    position_enu : ndarray, shape (3,), dtype float64
        Position [m] in ENU relative to the reference surface point.
        [East, North, Up].
    velocity_enu : ndarray, shape (3,), dtype float64
        Velocity [m s⁻¹] in ENU. [v_East, v_North, v_Up].
    ref_longitude_rad : float
        Geodetic longitude of the ENU origin [rad].  −π ≤ λ ≤ π.
    ref_latitude_rad : float
        Geodetic latitude of the ENU origin [rad].  −π/2 ≤ φ ≤ π/2.
    epoch_time : float
        Mission-elapsed time [s]. Non-negative.
    body_name : str
        Central body identifier. Default "Earth".
    omega : float
        Body rotation rate [rad s⁻¹]. Default EARTH_OMEGA.
    """

    position_enu: np.ndarray
    velocity_enu: np.ndarray
    ref_longitude_rad: float
    ref_latitude_rad: float
    epoch_time: float
    body_name: str = "Earth"
    omega: float = EARTH_OMEGA

    def __post_init__(self) -> None:
        # --- position_enu ---
        pos = np.asarray(self.position_enu, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(
                f"position_enu must have shape (3,); got {pos.shape}"
            )
        object.__setattr__(self, "position_enu", pos)

        # --- velocity_enu ---
        vel = np.asarray(self.velocity_enu, dtype=np.float64)
        if vel.shape != (3,):
            raise ValueError(
                f"velocity_enu must have shape (3,); got {vel.shape}"
            )
        object.__setattr__(self, "velocity_enu", vel)

        # --- ref_longitude_rad ---
        lam = float(self.ref_longitude_rad)
        if not (-math.pi - 1.0e-9 <= lam <= math.pi + 1.0e-9):
            raise ValueError(
                f"ref_longitude_rad must be in [−π, π]; got {lam:.6g}"
            )
        object.__setattr__(self, "ref_longitude_rad", lam)

        # --- ref_latitude_rad ---
        phi = float(self.ref_latitude_rad)
        if not (-math.pi / 2.0 - 1.0e-9 <= phi <= math.pi / 2.0 + 1.0e-9):
            raise ValueError(
                f"ref_latitude_rad must be in [−π/2, π/2]; got {phi:.6g}"
            )
        object.__setattr__(self, "ref_latitude_rad", phi)

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
            raise ValueError(f"omega must be non-negative; got {omega:.6g}")
        object.__setattr__(self, "omega", omega)

    # ------------------------------------------------------------------
    # Derived scalar properties
    # ------------------------------------------------------------------

    @property
    def east(self) -> float:
        """East component of position [m]."""
        return float(self.position_enu[0])

    @property
    def north(self) -> float:
        """North component of position [m]."""
        return float(self.position_enu[1])

    @property
    def up(self) -> float:
        """Up (altitude above reference surface) component [m]."""
        return float(self.position_enu[2])

    @property
    def horizontal_range(self) -> float:
        """Horizontal range from the ENU origin [m] (East-North plane only)."""
        return float(math.sqrt(self.east ** 2 + self.north ** 2))

    @property
    def slant_range(self) -> float:
        """3-D slant range from the ENU origin [m]."""
        return float(np.linalg.norm(self.position_enu))

    @property
    def bearing_rad(self) -> float:
        """
        Bearing (heading) from North to the horizontal position vector [rad].

        Measured clockwise from North: 0 → North, π/2 → East.
        Returns 0.0 if the vehicle is at the ENU origin.
        """
        e, n = self.east, self.north
        if abs(e) < 1.0e-12 and abs(n) < 1.0e-12:
            return 0.0
        return math.atan2(e, n) % (2.0 * math.pi)

    @property
    def speed(self) -> float:
        """Total speed [m s⁻¹]."""
        return float(np.linalg.norm(self.velocity_enu))

    @property
    def vertical_speed(self) -> float:
        """Vertical speed (Up component) [m s⁻¹]. Positive = ascending."""
        return float(self.velocity_enu[2])

    @property
    def horizontal_speed(self) -> float:
        """Horizontal groundspeed (East-North plane) [m s⁻¹]."""
        return float(math.sqrt(float(self.velocity_enu[0]) ** 2 + float(self.velocity_enu[1]) ** 2))

    # ------------------------------------------------------------------
    # DCM helpers
    # ------------------------------------------------------------------

    def dcm_to_ecef(self) -> DCM:
        """
        Direction cosine matrix T_ENU→ECEF for this frame's reference point.

        Returns
        -------
        ndarray, shape (3, 3), dtype float64
        """
        T = T_ENU_to_ECEF(self.ref_longitude_rad, self.ref_latitude_rad)
        assert_dcm_orthogonal(T, "T_ENU_to_ECEF")
        return T

    def dcm_from_ecef(self) -> DCM:
        """
        Direction cosine matrix T_ECEF→ENU for this frame's reference point.

        Returns
        -------
        ndarray, shape (3, 3), dtype float64
        """
        T = T_ECEF_to_ENU(self.ref_longitude_rad, self.ref_latitude_rad)
        assert_dcm_orthogonal(T, "T_ECEF_to_ENU")
        return T

    # ------------------------------------------------------------------
    # Frame conversion helpers
    # ------------------------------------------------------------------

    def to_ecef_position(self) -> np.ndarray:
        """
        Express position_enu in the ECEF frame.

        Returns the rotated position vector (origin offset not included —
        this is the direction transform only, suitable for velocity/direction
        conversion, not absolute position).

        Returns
        -------
        ndarray, shape (3,), dtype float64
        """
        T = T_ENU_to_ECEF(self.ref_longitude_rad, self.ref_latitude_rad)
        return (T @ self.position_enu).astype(np.float64)

    def to_ecef_velocity(self) -> np.ndarray:
        """
        Rotate velocity_enu to the ECEF frame.

        Returns
        -------
        ndarray, shape (3,), dtype float64
        """
        T = T_ENU_to_ECEF(self.ref_longitude_rad, self.ref_latitude_rad)
        return (T @ self.velocity_enu).astype(np.float64)

    def to_eci_position(self) -> np.ndarray:
        """
        Rotate position_enu to ECI at epoch_time.

        Chain: ENU → ECEF → ECI.

        Returns
        -------
        ndarray, shape (3,), dtype float64
        """
        T_enu_ecef = T_ENU_to_ECEF(self.ref_longitude_rad, self.ref_latitude_rad)
        T_ecef_eci = T_ECEF_to_ECI(self.epoch_time, self.omega)
        return (T_ecef_eci @ (T_enu_ecef @ self.position_enu)).astype(np.float64)

    def to_eci_velocity(self) -> np.ndarray:
        """
        Rotate velocity_enu to ECI at epoch_time.

        Returns
        -------
        ndarray, shape (3,), dtype float64
        """
        T_enu_ecef = T_ENU_to_ECEF(self.ref_longitude_rad, self.ref_latitude_rad)
        T_ecef_eci = T_ECEF_to_ECI(self.epoch_time, self.omega)
        return (T_ecef_eci @ (T_enu_ecef @ self.velocity_enu)).astype(np.float64)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lam_deg = math.degrees(self.ref_longitude_rad)
        phi_deg = math.degrees(self.ref_latitude_rad)
        return (
            f"ENUFrame(ref=({lam_deg:.4f}°, {phi_deg:.4f}°), "
            f"t={self.epoch_time:.3f}s, "
            f"E={self.east:.1f}m, N={self.north:.1f}m, U={self.up:.1f}m)"
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def enu_from_ecef(
    position_ecef: np.ndarray,
    velocity_ecef: np.ndarray,
    ref_longitude_rad: float,
    ref_latitude_rad: float,
    epoch_time: float = 0.0,
    body_name: str = "Earth",
    omega: float = EARTH_OMEGA,
) -> ENUFrame:
    """
    Construct an ENUFrame by rotating an ECEF state to the local ENU frame.

    Parameters
    ----------
    position_ecef : array_like, shape (3,)
        ECEF position [m].
    velocity_ecef : array_like, shape (3,)
        ECEF velocity [m s⁻¹].
    ref_longitude_rad : float
        ENU reference longitude λ [rad].
    ref_latitude_rad : float
        ENU reference latitude φ [rad].
    epoch_time : float
        Mission-elapsed time [s].
    body_name : str
        Central body name. Default "Earth".
    omega : float
        Rotation rate [rad s⁻¹]. Default EARTH_OMEGA.

    Returns
    -------
    ENUFrame
    """
    T = T_ECEF_to_ENU(float(ref_longitude_rad), float(ref_latitude_rad))
    pos_ecef = np.asarray(position_ecef, dtype=np.float64)
    vel_ecef = np.asarray(velocity_ecef, dtype=np.float64)

    pos_enu = (T @ pos_ecef).astype(np.float64)
    vel_enu = (T @ vel_ecef).astype(np.float64)

    return ENUFrame(
        position_enu=pos_enu,
        velocity_enu=vel_enu,
        ref_longitude_rad=float(ref_longitude_rad),
        ref_latitude_rad=float(ref_latitude_rad),
        epoch_time=float(epoch_time),
        body_name=body_name,
        omega=omega,
    )


def enu_from_eci(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
    ref_longitude_rad: float,
    ref_latitude_rad: float,
    epoch_time: float,
    body_name: str = "Earth",
    omega: float = EARTH_OMEGA,
) -> ENUFrame:
    """
    Construct an ENUFrame from an ECI state vector.

    Chain: ECI → ECEF → ENU.

    Parameters
    ----------
    position_eci : array_like, shape (3,)
        ECI position [m].
    velocity_eci : array_like, shape (3,)
        ECI velocity [m s⁻¹].
    ref_longitude_rad : float
        ENU reference longitude λ [rad].
    ref_latitude_rad : float
        ENU reference latitude φ [rad].
    epoch_time : float
        Mission-elapsed time [s].
    body_name : str
        Central body name. Default "Earth".
    omega : float
        Rotation rate [rad s⁻¹]. Default EARTH_OMEGA.

    Returns
    -------
    ENUFrame
    """
    pos_eci = np.asarray(position_eci, dtype=np.float64)
    vel_eci = np.asarray(velocity_eci, dtype=np.float64)

    T_eci_ecef = T_ECI_to_ECEF(float(epoch_time), omega)
    T_ecef_enu = T_ECEF_to_ENU(float(ref_longitude_rad), float(ref_latitude_rad))

    T_combined = T_ecef_enu @ T_eci_ecef

    pos_enu = (T_combined @ pos_eci).astype(np.float64)
    vel_enu = (T_combined @ vel_eci).astype(np.float64)

    return ENUFrame(
        position_enu=pos_enu,
        velocity_enu=vel_enu,
        ref_longitude_rad=float(ref_longitude_rad),
        ref_latitude_rad=float(ref_latitude_rad),
        epoch_time=float(epoch_time),
        body_name=body_name,
        omega=omega,
    )
