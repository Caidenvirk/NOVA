"""
nova.rendering.viewport
========================
Variable-rate render loop manager for Project NOVA.

Architectural role
------------------
Phase 12 — Rendering.
Pipeline stage: Stage 12 (Renderer). Manages the decoupled display update
loop, using interpolation across sequential telemetry frames to produce
smooth rendering at any frame rate independent of the physics timestep.

Design
------
The physics simulation runs at a fixed timestep dt (e.g. 0.01 s = 100 Hz).
The renderer may run at any frame rate (e.g. 60 Hz). Rather than locking
the two together, the viewport interpolates between the two most recent
TelemetrySnapshots to produce a RenderFrame at any requested display time.

Interpolation is linear (LERP) for scalar telemetry values and SLERP for
the attitude quaternion to avoid gimbal lock and maintain unit-norm.

The Viewport class is intentionally display-backend-agnostic. It owns no
Pygame surface or window. The caller (HUD compositor or integration test)
provides a surface if needed. This allows full unit testing without a
display environment.

Pygame import is guarded: if Pygame is unavailable the module degrades
gracefully to headless mode. All geometry and interpolation logic is
testable in both modes.

I/O contract
------------
Input  : TelemetryRegistry (read-only, append-only history)
Output : RenderFrame (frozen dataclass) — interpolated state snapshot
         ready for drawing, plus ViewportConfig metadata

Architectural invariants respected
-----------------------------------
- No wall-clock time in physics. The renderer reads wall_clock from
  TelemetrySnapshot (set by the pipeline at publish time) — not from
  Python time.time().
- The TelemetryRegistry is never written to by this module.
- All physics quantities come from frozen TelemetrySnapshot instances.

References
----------
- NOVA Engineering Handoff §7 Stage 12, §12 Phase 12
- Shoemake, K. "Animating Rotation with Quaternion Curves." SIGGRAPH 1985.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from nova.core.telemetry_registry import TelemetryRegistry, TelemetrySnapshot

# ---------------------------------------------------------------------------
# Pygame availability guard
# ---------------------------------------------------------------------------

try:
    import pygame as _pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _pygame = None          # type: ignore[assignment]
    _PYGAME_AVAILABLE = False


def pygame_available() -> bool:
    """Return True if Pygame is importable in this environment."""
    return _PYGAME_AVAILABLE


# ---------------------------------------------------------------------------
# ViewportConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ViewportConfig:
    """
    Display configuration for the render viewport.

    Attributes
    ----------
    width_px : int
        Viewport width in pixels. Must be positive. Default 1280.
    height_px : int
        Viewport height in pixels. Must be positive. Default 720.
    target_fps : float
        Target display frame rate [Hz]. Must be positive. Default 60.0.
    title : str
        Window title string. Default "NOVA Flight Simulation".
    background_color : tuple[int,int,int]
        RGB background fill colour. Each channel in [0, 255]. Default (5,5,15).
    show_hud : bool
        Whether the HUD overlay should be drawn this frame. Default True.
    near_clip : float
        Near clipping distance for 3-D projection [m]. Default 0.1.
    far_clip : float
        Far clipping distance [m]. Default 1.0e9.
    fov_deg : float
        Vertical field of view [degrees]. Must be in (0, 180). Default 60.0.
    """

    width_px: int = 1280
    height_px: int = 720
    target_fps: float = 60.0
    title: str = "NOVA Flight Simulation"
    background_color: Tuple[int, int, int] = (5, 5, 15)
    show_hud: bool = True
    near_clip: float = 0.1
    far_clip: float = 1.0e9
    fov_deg: float = 60.0

    def __post_init__(self) -> None:
        if self.width_px <= 0:
            raise ValueError(f"width_px must be positive; got {self.width_px}")
        if self.height_px <= 0:
            raise ValueError(f"height_px must be positive; got {self.height_px}")
        if self.target_fps <= 0.0:
            raise ValueError(f"target_fps must be positive; got {self.target_fps:.6g}")
        r, g, b = self.background_color
        for ch, name in ((r, 'R'), (g, 'G'), (b, 'B')):
            if not (0 <= ch <= 255):
                raise ValueError(f"background_color {name} must be in [0,255]; got {ch}")
        if self.near_clip <= 0.0:
            raise ValueError(f"near_clip must be positive; got {self.near_clip:.6g}")
        if self.far_clip <= self.near_clip:
            raise ValueError(
                f"far_clip ({self.far_clip:.6g}) must exceed near_clip ({self.near_clip:.6g})"
            )
        if not (0.0 < self.fov_deg < 180.0):
            raise ValueError(f"fov_deg must be in (0, 180); got {self.fov_deg:.6g}")

    @property
    def aspect_ratio(self) -> float:
        """Width / height aspect ratio."""
        return self.width_px / self.height_px

    @property
    def frame_budget_s(self) -> float:
        """Target time budget per display frame [s] = 1 / target_fps."""
        return 1.0 / self.target_fps

    @property
    def center_px(self) -> Tuple[int, int]:
        """Screen centre in pixels (x, y)."""
        return (self.width_px // 2, self.height_px // 2)


# ---------------------------------------------------------------------------
# RenderFrame — interpolated state for one display frame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RenderFrame:
    """
    Interpolated telemetry snapshot ready for a single display frame.

    Produced by :meth:`Viewport.get_render_frame`. All values are linearly
    interpolated between the two bounding TelemetrySnapshots, except the
    attitude quaternion which is SLERP-interpolated.

    Attributes
    ----------
    position_eci : ndarray, shape (3,), dtype float64
        Interpolated ECI position [m].
    velocity_eci : ndarray, shape (3,), dtype float64
        Interpolated ECI velocity [m s⁻¹].
    quaternion : ndarray, shape (4,), dtype float64
        SLERP-interpolated attitude quaternion (unit, scalar-first).
    omega_body : ndarray, shape (3,), dtype float64
        Interpolated body angular rate [rad s⁻¹].
    mass : float
        Interpolated vehicle mass [kg].
    mission_time : float
        Interpolated mission elapsed time [s].
    altitude : float
        Interpolated altitude [m].
    speed : float
        Interpolated speed [m s⁻¹].
    mach : float
        Interpolated Mach number.
    throttle : float
        Interpolated throttle [0, 1].
    thrust_magnitude : float
        Interpolated thrust magnitude [N].
    alpha : float
        Interpolated angle of attack [rad].
    dynamic_pressure : float
        Interpolated dynamic pressure [Pa].
    semi_major_axis : float
        Interpolated semi-major axis [m].
    eccentricity : float
        Interpolated eccentricity.
    inclination : float
        Interpolated inclination [rad].
    apoapsis : float
        Interpolated apoapsis altitude [m].
    periapsis : float
        Interpolated periapsis altitude [m].
    any_structural_failure : bool
        True if either bounding snapshot reported structural failure.
    alpha_blend : float
        Interpolation parameter t ∈ [0, 1] used to produce this frame.
        0 = earlier snapshot, 1 = later snapshot.
    earlier_snap_time : float
        Mission time of the earlier bounding snapshot [s].
    later_snap_time : float
        Mission time of the later bounding snapshot [s].
    """

    position_eci: np.ndarray
    velocity_eci: np.ndarray
    quaternion: np.ndarray
    omega_body: np.ndarray
    mass: float
    mission_time: float
    altitude: float
    speed: float
    mach: float
    throttle: float
    thrust_magnitude: float
    alpha: float
    dynamic_pressure: float
    semi_major_axis: float
    eccentricity: float
    inclination: float
    apoapsis: float
    periapsis: float
    any_structural_failure: bool
    alpha_blend: float
    earlier_snap_time: float
    later_snap_time: float

    def __post_init__(self) -> None:
        for attr in ("position_eci", "velocity_eci", "quaternion", "omega_body"):
            arr = np.asarray(getattr(self, attr), dtype=np.float64)
            object.__setattr__(self, attr, arr)
        for attr in ("mass", "mission_time", "altitude", "speed", "mach",
                     "throttle", "thrust_magnitude", "alpha", "dynamic_pressure",
                     "semi_major_axis", "eccentricity", "inclination",
                     "apoapsis", "periapsis", "alpha_blend",
                     "earlier_snap_time", "later_snap_time"):
            object.__setattr__(self, attr, float(getattr(self, attr)))
        object.__setattr__(self, "any_structural_failure",
                           bool(self.any_structural_failure))

    @property
    def altitude_km(self) -> float:
        """Altitude in kilometres."""
        return self.altitude / 1_000.0

    @property
    def speed_km_s(self) -> float:
        """Speed in km s⁻¹."""
        return self.speed / 1_000.0

    def __repr__(self) -> str:
        return (
            f"RenderFrame(t={self.mission_time:.3f}s, "
            f"alt={self.altitude_km:.2f}km, "
            f"v={self.speed:.1f}m/s, "
            f"α={self.alpha_blend:.3f})"
        )


# ---------------------------------------------------------------------------
# Pure interpolation helpers
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a + t*(b - a), t ∈ [0, 1]."""
    return a + t * (b - a)


def _lerp_vec(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Element-wise LERP for numpy arrays."""
    return a + t * (b - a)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two unit quaternions.

    Parameters
    ----------
    q0, q1 : ndarray, shape (4,)
        Unit quaternions (scalar-first). Norms must be ≈ 1.
    t : float
        Interpolation parameter ∈ [0, 1].

    Returns
    -------
    ndarray, shape (4,), dtype float64
        Interpolated unit quaternion.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    # Normalise inputs
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))

    # Ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = min(1.0, dot)

    # Fall back to LERP if quaternions are nearly identical
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0 + s1 * q1
    norm = float(np.linalg.norm(result))
    if norm < 1.0e-12:
        return q0.copy()
    return result / norm


def interpolate_snapshots(
    snap_a: TelemetrySnapshot,
    snap_b: TelemetrySnapshot,
    t: float,
) -> RenderFrame:
    """
    Produce an interpolated RenderFrame between two TelemetrySnapshots.

    Parameters
    ----------
    snap_a : TelemetrySnapshot
        Earlier snapshot (t=0 end).
    snap_b : TelemetrySnapshot
        Later snapshot (t=1 end).
    t : float
        Interpolation parameter ∈ [0, 1]. Clamped internally.

    Returns
    -------
    RenderFrame
    """
    t = float(max(0.0, min(1.0, t)))

    def L(a: float, b: float) -> float:
        return _lerp(a, b, t)

    va = snap_a.vehicle_state
    vb = snap_b.vehicle_state

    pos = _lerp_vec(va.position_eci, vb.position_eci, t)
    vel = _lerp_vec(va.velocity_eci, vb.velocity_eci, t)
    quat = _slerp(va.quaternion, vb.quaternion, t)
    omega = _lerp_vec(va.omega_body, vb.omega_body, t)

    return RenderFrame(
        position_eci=pos,
        velocity_eci=vel,
        quaternion=quat,
        omega_body=omega,
        mass=L(va.mass, vb.mass),
        mission_time=L(snap_a.mission_time, snap_b.mission_time),
        altitude=L(snap_a.altitude, snap_b.altitude),
        speed=L(snap_a.speed, snap_b.speed),
        mach=L(snap_a.mach, snap_b.mach),
        throttle=L(snap_a.throttle, snap_b.throttle),
        thrust_magnitude=L(snap_a.thrust_magnitude, snap_b.thrust_magnitude),
        alpha=L(snap_a.alpha, snap_b.alpha),
        dynamic_pressure=L(snap_a.dynamic_pressure, snap_b.dynamic_pressure),
        semi_major_axis=L(snap_a.semi_major_axis, snap_b.semi_major_axis),
        eccentricity=L(snap_a.eccentricity, snap_b.eccentricity),
        inclination=L(snap_a.inclination, snap_b.inclination),
        apoapsis=L(snap_a.apoapsis, snap_b.apoapsis),
        periapsis=L(snap_a.periapsis, snap_b.periapsis),
        any_structural_failure=(
            snap_a.any_structural_failure or snap_b.any_structural_failure
        ),
        alpha_blend=t,
        earlier_snap_time=float(snap_a.mission_time),
        later_snap_time=float(snap_b.mission_time),
    )


# ---------------------------------------------------------------------------
# Viewport — render loop manager
# ---------------------------------------------------------------------------

class Viewport:
    """
    Variable-rate render loop manager.

    Reads from a :class:`TelemetryRegistry` and produces interpolated
    :class:`RenderFrame` instances at any requested display time, completely
    decoupled from the physics simulation frequency.

    The Viewport owns no display surface. Surface creation (Pygame window
    init) is delegated to :meth:`open_display` and is a no-op in headless
    mode.

    Parameters
    ----------
    registry : TelemetryRegistry
        Simulation telemetry source. Never written to by this class.
    config : ViewportConfig | None
        Display configuration. Defaults to ViewportConfig().
    """

    def __init__(
        self,
        registry: TelemetryRegistry,
        config: Optional[ViewportConfig] = None,
    ) -> None:
        if not isinstance(registry, TelemetryRegistry):
            raise TypeError("registry must be a TelemetryRegistry")
        self._registry = registry
        self._config = config if config is not None else ViewportConfig()
        if not isinstance(self._config, ViewportConfig):
            raise TypeError("config must be a ViewportConfig")

        self._surface = None          # Pygame surface (None in headless)
        self._clock = None            # Pygame clock (None in headless)
        self._display_open = False
        self._frame_count: int = 0
        self._last_render_time: float = 0.0   # mission time of last frame

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ViewportConfig:
        return self._config

    @property
    def registry(self) -> TelemetryRegistry:
        return self._registry

    @property
    def frame_count(self) -> int:
        """Total number of frames rendered since construction."""
        return self._frame_count

    @property
    def display_open(self) -> bool:
        """True if a Pygame display window is currently open."""
        return self._display_open

    # ------------------------------------------------------------------
    # Display lifecycle (Pygame-optional)
    # ------------------------------------------------------------------

    def open_display(self) -> bool:
        """
        Open a Pygame display window using ViewportConfig dimensions.

        Returns
        -------
        bool
            True if the window was successfully opened; False in headless mode.
        """
        if not _PYGAME_AVAILABLE:
            return False
        if self._display_open:
            return True
        _pygame.init()
        self._surface = _pygame.display.set_mode(
            (self._config.width_px, self._config.height_px)
        )
        _pygame.display.set_caption(self._config.title)
        self._clock = _pygame.time.Clock()
        self._display_open = True
        return True

    def close_display(self) -> None:
        """Close the Pygame display window if open."""
        if _PYGAME_AVAILABLE and self._display_open:
            _pygame.quit()
        self._surface = None
        self._clock = None
        self._display_open = False

    # ------------------------------------------------------------------
    # Core: get_render_frame
    # ------------------------------------------------------------------

    def get_render_frame(
        self,
        display_time: Optional[float] = None,
    ) -> Optional[RenderFrame]:
        """
        Produce an interpolated RenderFrame for *display_time*.

        If the registry is empty, returns None. If only one snapshot is
        available, returns a RenderFrame at t=1 (that snapshot, no interp).
        Otherwise interpolates between the two bounding snapshots.

        Parameters
        ----------
        display_time : float | None
            Mission elapsed time [s] to render. If None, uses the latest
            published snapshot time (i.e. no look-ahead).

        Returns
        -------
        RenderFrame | None
            Interpolated render state, or None if no data is available.
        """
        history = self._registry.history
        if not history:
            return None

        if len(history) == 1:
            snap = history[0]
            return interpolate_snapshots(snap, snap, 1.0)

        # Default: render at latest time
        if display_time is None:
            snap_b = history[-1]
            snap_a = history[-2]
            t = 1.0
        else:
            # Find bounding pair
            snap_a, snap_b, t = _find_bounding_pair(history, display_time)

        self._frame_count += 1
        self._last_render_time = float(
            snap_a.mission_time + t * (snap_b.mission_time - snap_a.mission_time)
        )
        return interpolate_snapshots(snap_a, snap_b, t)

    def tick_display(self) -> None:
        """
        Advance the Pygame clock by one display frame.

        Caps the frame rate to target_fps. No-op in headless mode.
        """
        if _PYGAME_AVAILABLE and self._clock is not None:
            self._clock.tick(int(self._config.target_fps))

    def fill_background(self) -> None:
        """Fill the display surface with the background colour. No-op headless."""
        if self._surface is not None and _PYGAME_AVAILABLE:
            self._surface.fill(self._config.background_color)

    def flip(self) -> None:
        """Flip the Pygame display buffer. No-op in headless mode."""
        if _PYGAME_AVAILABLE and self._display_open:
            _pygame.display.flip()

    def __repr__(self) -> str:
        return (
            f"Viewport({self._config.width_px}×{self._config.height_px}, "
            f"fps={self._config.target_fps:.0f}, "
            f"frames={self._frame_count}, "
            f"headless={not self._display_open})"
        )


# ---------------------------------------------------------------------------
# Internal helper: find bounding snapshot pair
# ---------------------------------------------------------------------------

def _find_bounding_pair(
    history: List[TelemetrySnapshot],
    display_time: float,
) -> Tuple[TelemetrySnapshot, TelemetrySnapshot, float]:
    """
    Find the two snapshots that bracket *display_time* and compute t ∈ [0,1].

    Parameters
    ----------
    history : list[TelemetrySnapshot]
        Ordered list of snapshots (ascending mission time).
    display_time : float
        Target mission time [s].

    Returns
    -------
    snap_a, snap_b : TelemetrySnapshot
        Bounding pair (snap_a.time ≤ display_time ≤ snap_b.time).
    t : float
        Interpolation parameter ∈ [0, 1].
    """
    times = [s.mission_time for s in history]

    # Clamp to available range
    if display_time <= times[0]:
        return history[0], history[0], 0.0
    if display_time >= times[-1]:
        return history[-2], history[-1], 1.0

    # Binary search for lower bound
    lo, hi = 0, len(times) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if times[mid] <= display_time:
            lo = mid
        else:
            hi = mid

    snap_a = history[lo]
    snap_b = history[hi]
    dt_span = times[hi] - times[lo]
    if dt_span < 1.0e-12:
        t = 0.0
    else:
        t = (display_time - times[lo]) / dt_span

    return snap_a, snap_b, float(t)
