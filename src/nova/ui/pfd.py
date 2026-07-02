"""
nova.ui.pfd
============
Primary Flight Display (PFD) and Navball for Project NOVA.

Architectural role
------------------
Phase 13 — UI Glass Cockpit.
Pipeline stage: Stage 13 (UI Engine). Consumes a RenderFrame from the
Viewport (Phase 12) and produces a PFDState frozen dataclass containing all
data needed to draw the PFD panel and 3-D navball overlay.

Design
------
The PFD displays:
  - Attitude (roll, pitch, yaw) derived from the attitude quaternion
  - Prograde / retrograde / normal / anti-normal flight vectors on the navball
  - Speed tape: orbital speed, surface speed, vertical speed
  - Altitude tape: altitude, altitude rate
  - Mach indicator
  - Angle of attack α and sideslip β
  - G-load (total acceleration / g)

The navball is a synthetic 3-D sphere showing orientation relative to the
local horizon. The ball orientation is expressed as the body-frame quaternion
which is the canonical attitude representation in NOVA (no Euler angles used
in physics — Euler angles here are display-only, as per architecture §8.4).

Flight vectors are unit vectors expressed in the body frame:
  Prograde vector:  unit(velocity_eci) rotated into body frame
  Retrograde:       -prograde
  Normal:           unit(r × v) in body frame (orbit normal)
  Anti-normal:      -normal
  Radial:           unit(r) in body frame (away from planet centre)
  Anti-radial:      -radial

I/O contract
------------
Input  : RenderFrame, optional TelemetryRegistry for derivative quantities
Output : PFDState (frozen dataclass) — all display data, no drawing

No Pygame calls. No physics calculations.

References
----------
- NOVA Engineering Handoff §7 Stage 13, §12 Phase 13
- FAA AC 25.1322-1 (Flight Crew Alerting)
- Diebel, "Representing Attitude", Stanford 2006 §5 (quaternion→euler for display)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from nova.core.constants import STD_GRAVITY
from nova.frames.transforms import T_ENU_to_body, T_body_to_ENU, quaternion_to_euler
from nova.rendering.viewport import RenderFrame

# ---------------------------------------------------------------------------
# NavballVector — a labelled direction marker on the navball sphere
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NavballVector:
    """
    A labelled flight vector projected onto the navball sphere.

    Attributes
    ----------
    label : str
        Short identifier (e.g. "PRO", "RET", "NRM", "ANM", "RAD", "ARD").
    direction_body : ndarray, shape (3,)
        Unit vector in body frame pointing in this flight direction.
    color : tuple[int,int,int]
        RGB display colour.
    is_visible : bool
        True if this vector is on the front hemisphere of the navball.
    azimuth_rad : float
        Azimuth angle on navball sphere [rad], measured from navball +Y axis.
    elevation_rad : float
        Elevation angle on navball sphere [rad] from the equatorial plane.
    """

    label: str
    direction_body: np.ndarray
    color: Tuple[int, int, int]
    is_visible: bool
    azimuth_rad: float
    elevation_rad: float

    def __post_init__(self) -> None:
        arr = np.asarray(self.direction_body, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"direction_body must have shape (3,); got {arr.shape}"
            )
        object.__setattr__(self, "direction_body", arr)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "is_visible", bool(self.is_visible))
        object.__setattr__(self, "azimuth_rad", float(self.azimuth_rad))
        object.__setattr__(self, "elevation_rad", float(self.elevation_rad))


# ---------------------------------------------------------------------------
# SpeedTapeData, AltitudeTapeData
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeedTapeData:
    """
    Data for the speed tape instruments.

    Attributes
    ----------
    orbital_speed_m_s : float
        Inertial speed |v_eci| [m s⁻¹].
    surface_speed_m_s : float
        Speed relative to rotating surface. Approximated as orbital speed
        minus EARTH_OMEGA × r × cos(lat) for equatorial orbit. For display
        purposes, same as orbital_speed_m_s (precise computation left to
        the guidance layer). This field carries the value from RenderFrame.
    vertical_speed_m_s : float
        Radial (up) speed component [m s⁻¹]. Positive = ascending.
    mach : float
        Mach number.
    """

    orbital_speed_m_s: float
    surface_speed_m_s: float
    vertical_speed_m_s: float
    mach: float

    def __post_init__(self) -> None:
        for attr in ("orbital_speed_m_s", "surface_speed_m_s",
                     "vertical_speed_m_s", "mach"):
            object.__setattr__(self, attr, float(getattr(self, attr)))


@dataclass(frozen=True)
class AltitudeTapeData:
    """
    Data for the altitude tape instrument.

    Attributes
    ----------
    altitude_m : float
        Altitude above reference ellipsoid [m].
    altitude_rate_m_s : float
        Rate of altitude change [m s⁻¹] (= vertical_speed).
    target_altitude_m : float | None
        Commanded target altitude if set, else None.
    """

    altitude_m: float
    altitude_rate_m_s: float
    target_altitude_m: Optional[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "altitude_m", float(self.altitude_m))
        object.__setattr__(self, "altitude_rate_m_s", float(self.altitude_rate_m_s))
        if self.target_altitude_m is not None:
            object.__setattr__(self, "target_altitude_m",
                               float(self.target_altitude_m))


# ---------------------------------------------------------------------------
# PFDState — complete PFD data bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PFDState:
    """
    Complete frozen data bundle for the Primary Flight Display.

    Attributes
    ----------
    mission_time : float
        Mission elapsed time [s].
    roll_rad : float
        Roll angle φ [rad] (display-only; derived from quaternion).
    pitch_rad : float
        Pitch angle θ [rad] (display-only).
    yaw_rad : float
        Yaw angle ψ [rad] (display-only).
    roll_deg : float
        Roll angle in degrees.
    pitch_deg : float
        Pitch angle in degrees.
    yaw_deg : float
        Yaw angle in degrees.
    alpha_rad : float
        Angle of attack [rad].
    beta_rad : float
        Sideslip angle [rad].
    g_load : float
        Dimensionless g-load = |a_net| / g₀. 1.0 at rest on surface.
    quaternion : ndarray, shape (4,)
        Attitude quaternion for navball rendering.
    navball_vectors : list[NavballVector]
        Flight vector markers (prograde, retrograde, normal, anti-normal,
        radial, anti-radial).
    speed_tape : SpeedTapeData
        Speed tape instrument data.
    altitude_tape : AltitudeTapeData
        Altitude tape instrument data.
    dynamic_pressure_pa : float
        Dynamic pressure q∞ [Pa].
    any_structural_failure : bool
        Master warning flag from structural health monitor.
    throttle : float
        Current throttle [0, 1].
    thrust_n : float
        Engine thrust magnitude [N].
    mass_kg : float
        Current vehicle mass [kg].
    """

    mission_time: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    alpha_rad: float
    beta_rad: float
    g_load: float
    quaternion: np.ndarray
    navball_vectors: List[NavballVector]
    speed_tape: SpeedTapeData
    altitude_tape: AltitudeTapeData
    dynamic_pressure_pa: float
    any_structural_failure: bool
    throttle: float
    thrust_n: float
    mass_kg: float

    def __post_init__(self) -> None:
        for attr in ("mission_time", "roll_rad", "pitch_rad", "yaw_rad",
                     "roll_deg", "pitch_deg", "yaw_deg", "alpha_rad",
                     "beta_rad", "g_load", "dynamic_pressure_pa",
                     "throttle", "thrust_n", "mass_kg"):
            object.__setattr__(self, attr, float(getattr(self, attr)))
        q = np.asarray(self.quaternion, dtype=np.float64)
        object.__setattr__(self, "quaternion", q)
        object.__setattr__(self, "any_structural_failure",
                           bool(self.any_structural_failure))

    def __repr__(self) -> str:
        return (
            f"PFDState(t={self.mission_time:.1f}s, "
            f"roll={self.roll_deg:.1f}°, "
            f"pitch={self.pitch_deg:.1f}°, "
            f"yaw={self.yaw_deg:.1f}°, "
            f"alt={self.altitude_tape.altitude_m/1000:.2f}km, "
            f"v={self.speed_tape.orbital_speed_m_s:.1f}m/s)"
        )


# ---------------------------------------------------------------------------
# Flight vector computation
# ---------------------------------------------------------------------------

# Navball vector colours (KSP-inspired standard)
_COLOR_PROGRADE = (255, 255, 0)
_COLOR_RETROGRADE = (255, 100, 0)
_COLOR_NORMAL = (200, 80, 255)
_COLOR_ANTINORMAL = (100, 200, 255)
_COLOR_RADIAL = (0, 220, 255)
_COLOR_ANTIRADIAL = (255, 120, 0)


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    """Return unit vector or None if near-zero."""
    n = float(np.linalg.norm(v))
    if n < 1.0e-10:
        return None
    return v / n


def _body_direction(
    vec_eci: np.ndarray,
    quaternion: np.ndarray,
) -> Optional[np.ndarray]:
    """Rotate an ECI direction into body frame via body←ENU←ECI chain."""
    u = _unit(vec_eci)
    if u is None:
        return None
    # T_ENU_to_body rotates ENU→body; we need ECI→body.
    # For the navball, we treat ECI ≈ ENU (display-only approximation;
    # precise for short timescales — navball shows attitude relative to
    # an inertial reference, which matches user expectation).
    T_e2b = T_ENU_to_body(quaternion)
    return (T_e2b @ u).astype(np.float64)


def _navball_angles(direction_body: np.ndarray) -> Tuple[float, float, bool]:
    """
    Convert a body-frame direction vector to navball azimuth/elevation.

    The navball convention:
      - +X body (forward) projects to the top of the ball
      - Ball is viewed from outside looking at the front hemisphere

    Returns
    -------
    azimuth_rad, elevation_rad, is_visible
    """
    x, y, z = float(direction_body[0]), float(direction_body[1]), float(direction_body[2])
    # Elevation = angle above XY body plane
    elevation = math.asin(max(-1.0, min(1.0, x)))
    # Azimuth = angle in YZ plane from -Z (down) axis
    azimuth = math.atan2(y, -z)
    # Visible if on the front hemisphere (x > 0 → forward direction)
    is_visible = x > 0.0
    return azimuth, elevation, is_visible


def _compute_navball_vectors(frame: RenderFrame) -> List[NavballVector]:
    """
    Compute all six standard navball flight vectors from a RenderFrame.

    Returns a list of NavballVector instances. Empty entries are skipped
    when the underlying ECI vector is near-zero.
    """
    q = frame.quaternion
    r_eci = frame.position_eci
    v_eci = frame.velocity_eci

    # Prograde = velocity direction
    pro_body = _body_direction(v_eci, q)
    # Normal = orbit normal = r × v
    h_eci = np.cross(r_eci, v_eci)
    nrm_body = _body_direction(h_eci, q)
    # Radial = outward radial
    rad_body = _body_direction(r_eci, q)

    vectors: List[NavballVector] = []

    def _add(label: str, direction: Optional[np.ndarray], color: Tuple[int, int, int]) -> None:
        if direction is None:
            return
        az, el, vis = _navball_angles(direction)
        vectors.append(NavballVector(
            label=label,
            direction_body=direction.copy(),
            color=color,
            is_visible=vis,
            azimuth_rad=az,
            elevation_rad=el,
        ))

    _add("PRO", pro_body, _COLOR_PROGRADE)
    if pro_body is not None:
        _add("RET", -pro_body, _COLOR_RETROGRADE)
    _add("NRM", nrm_body, _COLOR_NORMAL)
    if nrm_body is not None:
        _add("ANM", -nrm_body, _COLOR_ANTINORMAL)
    _add("RAD", rad_body, _COLOR_RADIAL)
    if rad_body is not None:
        _add("ARD", -rad_body, _COLOR_ANTIRADIAL)

    return vectors


# ---------------------------------------------------------------------------
# PFD builder
# ---------------------------------------------------------------------------

class PrimaryFlightDisplay:
    """
    Produces PFDState from a RenderFrame each display tick.

    Parameters
    ----------
    target_altitude_m : float | None
        Commanded target altitude [m] for the altitude tape. None if unset.
    """

    def __init__(self, target_altitude_m: Optional[float] = None) -> None:
        self._target_alt = (
            float(target_altitude_m) if target_altitude_m is not None else None
        )

    @property
    def target_altitude_m(self) -> Optional[float]:
        return self._target_alt

    @target_altitude_m.setter
    def target_altitude_m(self, value: Optional[float]) -> None:
        self._target_alt = float(value) if value is not None else None

    def build(self, frame: RenderFrame) -> PFDState:
        """
        Build a PFDState from the current RenderFrame.

        Parameters
        ----------
        frame : RenderFrame
            Interpolated render state from Viewport.

        Returns
        -------
        PFDState
        """
        if not isinstance(frame, RenderFrame):
            raise TypeError(
                f"frame must be a RenderFrame; got {type(frame).__name__}"
            )

        # Attitude (display-only Euler angles from quaternion)
        roll, pitch, yaw = quaternion_to_euler(frame.quaternion)

        # Estimate vertical speed as radial component of velocity
        r_vec = frame.position_eci
        v_vec = frame.velocity_eci
        r_norm = float(np.linalg.norm(r_vec))
        if r_norm > 1.0:
            r_hat = r_vec / r_norm
            v_radial = float(np.dot(v_vec, r_hat))
        else:
            v_radial = 0.0

        # G-load: approximate from net force (not available in RenderFrame)
        # Use thrust and gravity estimate as a conservative approximation
        # g_load = 1 at rest; here we estimate from thrust/weight
        g_load = 1.0  # default (no net force data in RenderFrame)
        if frame.mass > 0.0 and frame.thrust_magnitude >= 0.0:
            # Rough: g_load = T/(m*g0) + 1 as a display approximation
            g_load = frame.thrust_magnitude / (frame.mass * STD_GRAVITY) + 1.0
            g_load = max(0.0, g_load)

        # Speed tape
        speed_tape = SpeedTapeData(
            orbital_speed_m_s=frame.speed,
            surface_speed_m_s=frame.speed,   # display approximation
            vertical_speed_m_s=v_radial,
            mach=frame.mach,
        )

        # Altitude tape
        alt_tape = AltitudeTapeData(
            altitude_m=frame.altitude,
            altitude_rate_m_s=v_radial,
            target_altitude_m=self._target_alt,
        )

        # Navball vectors
        navball = _compute_navball_vectors(frame)

        return PFDState(
            mission_time=frame.mission_time,
            roll_rad=roll,
            pitch_rad=pitch,
            yaw_rad=yaw,
            roll_deg=math.degrees(roll),
            pitch_deg=math.degrees(pitch),
            yaw_deg=math.degrees(yaw),
            alpha_rad=frame.alpha,
            beta_rad=0.0,           # beta not in RenderFrame; default 0
            g_load=g_load,
            quaternion=frame.quaternion.copy(),
            navball_vectors=navball,
            speed_tape=speed_tape,
            altitude_tape=alt_tape,
            dynamic_pressure_pa=frame.dynamic_pressure,
            any_structural_failure=frame.any_structural_failure,
            throttle=frame.throttle,
            thrust_n=frame.thrust_magnitude,
            mass_kg=frame.mass,
        )

    def __repr__(self) -> str:
        return f"PrimaryFlightDisplay(target_alt={self._target_alt})"
