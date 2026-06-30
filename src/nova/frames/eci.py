"""
nova.frames.eci
===============
Earth-Centred Inertial (ECI) frame container for Project NOVA.

Architectural role
------------------
Phase 7 — Frame Definition Classes.
Pipeline stage: upstream of all coordinate transforms; consumed by Stages 3, 4, 6.

This module defines ECIFrame, a lightweight, frozen value object that anchors
an inertial position/velocity pair to a specific simulation epoch and body
name. It provides no physics computation — it is a typed, validated carrier
for ECI-frame Cartesian state vectors.

I/O contract
------------
Input  : position_eci [m], velocity_eci [m s⁻¹], epoch_time [s], body_name str
Output : ECIFrame instance (frozen dataclass); convenience constructors

Physical basis
--------------
The ECI frame is an inertial (non-rotating) coordinate system whose origin is
at the planetary barycentre. Axes are fixed relative to distant stars:
  +X → vernal equinox direction (J2000 epoch)
  +Z → north celestial pole
  +Y → completes right-handed triad

At epoch t = 0 the ECI and ECEF X-axes are assumed co-aligned
(Greenwich Apparent Sidereal Time = 0), which is standard for relative
trajectory analysis in this simulation framework.

References
----------
- Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., §3.2
- Curtis, "Orbital Mechanics for Engineering Students", §2.1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nova.core.constants import (
    EARTH_MU,
    EARTH_OMEGA,
    EARTH_RADIUS_EQ,
    TRANSFORM_IDENTITY_TOL,
)
from nova.frames.transforms import (
    T_ECI_to_ECEF,
    T_ECEF_to_ECI,
    assert_dcm_orthogonal,
)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

DCM = np.ndarray  # shape (3, 3), dtype float64


# ---------------------------------------------------------------------------
# ECIFrame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ECIFrame:
    """
    Immutable container for a Cartesian state vector expressed in the
    Earth-Centred Inertial (ECI) reference frame.

    All distance quantities are in metres, velocities in m s⁻¹.
    The frame is non-rotating; it is fixed to distant stars with origin
    at the planetary barycentre.

    Attributes
    ----------
    position_eci : ndarray, shape (3,), dtype float64
        Cartesian position vector [m] expressed in ECI.
    velocity_eci : ndarray, shape (3,), dtype float64
        Cartesian velocity vector [m s⁻¹] expressed in ECI.
    epoch_time : float
        Mission-elapsed time [s] at which this frame snapshot was taken.
        Must be non-negative. Used to derive ECEF rotation angle θ = Ω · t.
    body_name : str
        Identifier for the central body (e.g. "Earth", "Moon"). Default "Earth".
    omega : float
        Rotation rate of the central body [rad s⁻¹]. Defaults to EARTH_OMEGA.
    mu : float
        Gravitational parameter of the central body [m³ s⁻²]. Defaults to EARTH_MU.
    """

    position_eci: np.ndarray
    velocity_eci: np.ndarray
    epoch_time: float
    body_name: str = "Earth"
    omega: float = EARTH_OMEGA
    mu: float = EARTH_MU

    def __post_init__(self) -> None:
        # --- position_eci ---
        pos = np.asarray(self.position_eci, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(
                f"position_eci must have shape (3,); got {pos.shape}"
            )
        object.__setattr__(self, "position_eci", pos)

        # --- velocity_eci ---
        vel = np.asarray(self.velocity_eci, dtype=np.float64)
        if vel.shape != (3,):
            raise ValueError(
                f"velocity_eci must have shape (3,); got {vel.shape}"
            )
        object.__setattr__(self, "velocity_eci", vel)

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
                f"omega (rotation rate) must be non-negative; got {omega:.6g}"
            )
        object.__setattr__(self, "omega", omega)

        # --- mu ---
        mu = float(self.mu)
        if mu <= 0.0:
            raise ValueError(
                f"mu (gravitational parameter) must be positive; got {mu:.6g}"
            )
        object.__setattr__(self, "mu", mu)

    # ------------------------------------------------------------------
    # Derived scalar properties
    # ------------------------------------------------------------------

    @property
    def radius(self) -> float:
        """Scalar distance from barycentre [m]. Always positive."""
        return float(np.linalg.norm(self.position_eci))

    @property
    def speed(self) -> float:
        """Scalar orbital speed [m s⁻¹]. Always non-negative."""
        return float(np.linalg.norm(self.velocity_eci))

    @property
    def specific_orbital_energy(self) -> float:
        """
        Specific orbital energy ε = v²/2 − μ/r  [J kg⁻¹].

        Negative for bound orbits, zero for parabolic escape, positive for
        hyperbolic trajectories.
        """
        r = self.radius
        if r < 1.0:
            raise ValueError(
                "radius is below 1 m — position_eci appears to be at the barycentre"
            )
        return 0.5 * self.speed ** 2 - self.mu / r

    @property
    def position_unit(self) -> np.ndarray:
        """Unit vector in the radial direction [dimensionless], shape (3,)."""
        r = self.radius
        if r < 1.0:
            raise ValueError("Cannot compute unit vector: position_eci is zero")
        return self.position_eci / r

    # ------------------------------------------------------------------
    # Frame conversion helpers
    # ------------------------------------------------------------------

    def to_ecef_position(self) -> np.ndarray:
        """
        Rotate position_eci to the ECEF frame at epoch_time.

        Returns
        -------
        ndarray, shape (3,), dtype float64
            Position expressed in ECEF [m].
        """
        T = T_ECI_to_ECEF(self.epoch_time, self.omega)
        return (T @ self.position_eci).astype(np.float64)

    def to_ecef_velocity(self) -> np.ndarray:
        """
        Rotate velocity_eci to the ECEF frame at epoch_time.

        Note: this returns only the *rotational* contribution (the Coriolis
        term ω × r is NOT added here). For the full ECEF velocity including
        planet-surface-relative motion, add ω × r_ecef externally.

        Returns
        -------
        ndarray, shape (3,), dtype float64
            Velocity expressed in ECEF [m s⁻¹] (rotation only, no Coriolis).
        """
        T = T_ECI_to_ECEF(self.epoch_time, self.omega)
        return (T @ self.velocity_eci).astype(np.float64)

    def dcm_to_ecef(self) -> DCM:
        """
        Direction cosine matrix T_ECI→ECEF at epoch_time.

        Returns
        -------
        ndarray, shape (3, 3), dtype float64
        """
        T = T_ECI_to_ECEF(self.epoch_time, self.omega)
        assert_dcm_orthogonal(T, "T_ECI_to_ECEF")
        return T

    def dcm_from_ecef(self) -> DCM:
        """
        Direction cosine matrix T_ECEF→ECI at epoch_time.

        Returns
        -------
        ndarray, shape (3, 3), dtype float64
        """
        T = T_ECEF_to_ECI(self.epoch_time, self.omega)
        assert_dcm_orthogonal(T, "T_ECEF_to_ECI")
        return T

    # ------------------------------------------------------------------
    # Comparison and representation
    # ------------------------------------------------------------------

    def is_close(
        self,
        other: "ECIFrame",
        atol_pos: float = 1.0e-3,
        atol_vel: float = 1.0e-6,
        atol_time: float = 1.0e-9,
    ) -> bool:
        """
        Return True if this frame is numerically close to *other*.

        Parameters
        ----------
        other : ECIFrame
        atol_pos : float
            Absolute position tolerance [m]. Default 1 mm.
        atol_vel : float
            Absolute velocity tolerance [m s⁻¹]. Default 1 μm s⁻¹.
        atol_time : float
            Absolute time tolerance [s]. Default 1 ns.

        Returns
        -------
        bool
        """
        return (
            bool(np.allclose(self.position_eci, other.position_eci, atol=atol_pos, rtol=0.0))
            and bool(np.allclose(self.velocity_eci, other.velocity_eci, atol=atol_vel, rtol=0.0))
            and abs(self.epoch_time - other.epoch_time) <= atol_time
            and self.body_name == other.body_name
        )

    def __repr__(self) -> str:
        r = self.radius / 1_000.0  # km
        v = self.speed
        return (
            f"ECIFrame(body={self.body_name!r}, t={self.epoch_time:.3f}s, "
            f"r={r:.3f} km, v={v:.3f} m/s)"
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def eci_from_state(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
    epoch_time: float,
    body_name: str = "Earth",
    omega: float = EARTH_OMEGA,
    mu: float = EARTH_MU,
) -> ECIFrame:
    """
    Construct an ECIFrame from raw position/velocity arrays and epoch time.

    Parameters
    ----------
    position_eci : array_like, shape (3,)
        Cartesian ECI position [m].
    velocity_eci : array_like, shape (3,)
        Cartesian ECI velocity [m s⁻¹].
    epoch_time : float
        Mission-elapsed time [s] (non-negative).
    body_name : str
        Central body identifier. Default "Earth".
    omega : float
        Central body rotation rate [rad s⁻¹]. Default EARTH_OMEGA.
    mu : float
        Central body gravitational parameter [m³ s⁻²]. Default EARTH_MU.

    Returns
    -------
    ECIFrame
    """
    return ECIFrame(
        position_eci=np.asarray(position_eci, dtype=np.float64),
        velocity_eci=np.asarray(velocity_eci, dtype=np.float64),
        epoch_time=float(epoch_time),
        body_name=body_name,
        omega=omega,
        mu=mu,
    )


def eci_circular_orbit(
    altitude_m: float,
    inclination_rad: float = 0.0,
    epoch_time: float = 0.0,
    body_name: str = "Earth",
    radius_eq: float = EARTH_RADIUS_EQ,
    mu: float = EARTH_MU,
    omega: float = EARTH_OMEGA,
) -> ECIFrame:
    """
    Build an ECIFrame for a circular orbit at a given altitude.

    The orbit starts at the ascending node (longitude = 0, latitude = 0)
    with velocity in the +Y ECI direction (for zero inclination).
    Inclination rotates the orbit plane about the ECI X-axis.

    Parameters
    ----------
    altitude_m : float
        Orbital altitude above equatorial radius [m]. Must be > 0.
    inclination_rad : float
        Orbital inclination [rad]. 0 → equatorial prograde. Default 0.
    epoch_time : float
        Epoch time [s]. Default 0.
    body_name : str
        Central body name. Default "Earth".
    radius_eq : float
        Equatorial radius [m]. Default EARTH_RADIUS_EQ.
    mu : float
        Gravitational parameter [m³ s⁻²]. Default EARTH_MU.
    omega : float
        Rotation rate [rad s⁻¹]. Default EARTH_OMEGA.

    Returns
    -------
    ECIFrame
    """
    if altitude_m <= 0.0:
        raise ValueError(f"altitude_m must be positive; got {altitude_m:.6g}")

    r = radius_eq + altitude_m
    v_circ = np.sqrt(mu / r)

    # Position: at (r, 0, 0) in ECI — ascending node on X-axis
    pos = np.array([r, 0.0, 0.0], dtype=np.float64)

    # Velocity: at inclination i, in the direction (0, cos i, sin i)
    ci = np.cos(inclination_rad)
    si = np.sin(inclination_rad)
    vel = np.array([0.0, v_circ * ci, v_circ * si], dtype=np.float64)

    return ECIFrame(
        position_eci=pos,
        velocity_eci=vel,
        epoch_time=float(epoch_time),
        body_name=body_name,
        omega=omega,
        mu=mu,
    )
