"""
nova.core.telemetry_registry
============================
Read-only telemetry snapshot bus for Project NOVA.

Architecture role — Pipeline Stage 11
--------------------------------------
After every simulation tick the pipeline serialises the complete vehicle
state into a TelemetrySnapshot and publishes it to the TelemetryRegistry.
Stages 12 and 13 (Renderer, UI Engine) and Stage 10 (AI Monitor) consume
the registry without ever writing to the vehicle state.

This separation enforces the deterministic pipeline contract:
  - No downstream stage can corrupt upstream physics state.
  - The AI monitor reads derivatives of immutable snapshots.
  - The renderer interpolates between adjacent frozen snapshots.

Design
------
TelemetrySnapshot is a frozen dataclass carrying:
  - The full VehicleState (position, velocity, quaternion, ω, mass, time)
  - Derived atmospheric quantities (density, pressure, Mach, q_inf)
  - Orbital elements (a, e, i, Ω, ω, ν)
  - Force/torque breakdown (per named contributor)
  - Structural health summary (worst margin per joint)
  - Propulsion state (thrust, Isp, mass flow, throttle)
  - Aerodynamic state (α, β, CL, CD, dynamic pressure)

TelemetryRegistry holds a rolling buffer of the last N snapshots. This
supports:
  - AI Monitor derivative estimation (dX/dt across consecutive ticks)
  - HUD trend arrows and rate-of-change displays
  - Post-flight data export

All access is read-only after publish. The registry itself is thread-safe
for single-producer / multiple-consumer use (GIL-protected in CPython).

Units — strict SI throughout, angles in radians.
"""

from __future__ import annotations

import time as _time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nova.core.state_vector import VehicleState


# ---------------------------------------------------------------------------
# Telemetry snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TelemetrySnapshot:
    """
    Complete read-only vehicle telemetry at one simulation tick.

    All physical quantities are in strict SI units. Angles in radians.

    Parameters
    ----------
    vehicle_state : VehicleState
        Core 13-element ODE state (position, velocity, quaternion, ω, mass, time).
    wall_clock : float
        Wall-clock time at publication [s since epoch]. For logging only.

    Atmospheric telemetry
    ---------------------
    altitude : float
        Geodetic altitude above MSL [m].
    density : float
        Atmospheric density ρ [kg m⁻³].
    pressure : float
        Atmospheric static pressure [Pa].
    mach : float
        Freestream Mach number [-].
    dynamic_pressure : float
        q_∞ = ½ρv² [Pa].
    speed_of_sound : float
        Local speed of sound [m s⁻¹].

    Aerodynamic telemetry
    ---------------------
    alpha : float
        Angle of attack [rad].
    beta : float
        Sideslip angle [rad].
    CL : float
        Lift coefficient [-].
    CD : float
        Drag coefficient [-].
    lift_force : float
        Lift magnitude [N].
    drag_force : float
        Drag magnitude [N].

    Propulsion telemetry
    --------------------
    thrust_magnitude : float
        Total engine thrust [N].
    mass_flow_rate : float
        Propellant consumption rate ṁ [kg s⁻¹].
    isp_effective : float
        Effective specific impulse [s].
    throttle : float
        Engine throttle setting [0–1].

    Orbital telemetry
    -----------------
    semi_major_axis : float       a [m]
    eccentricity : float          e [-]
    inclination : float           i [rad]
    raan : float                  Ω [rad]
    argument_of_periapsis : float ω [rad]
    true_anomaly : float          ν [rad]
    orbital_period : float        T [s]
    apoapsis : float              r_a [m]
    periapsis : float             r_p [m]

    Forces [N] in ECI frame
    -----------------------
    force_gravity : ndarray (3,)
    force_thrust : ndarray (3,)
    force_aero : ndarray (3,)
    force_net : ndarray (3,)

    Moments [N·m] in Body frame
    ---------------------------
    torque_aero : ndarray (3,)
    torque_gimbal : ndarray (3,)
    torque_net : ndarray (3,)

    Structural telemetry
    --------------------
    worst_structural_margin : float
        Minimum safety margin across all joints. Negative = failure.
    critical_joint_id : str
        ID of the joint with the lowest safety margin. Empty string if none.
    any_structural_failure : bool
        True if any joint has failed this tick.

    Navigation
    ----------
    speed : float
        ECI speed magnitude [m s⁻¹].
    vertical_speed : float
        Rate of change of altitude [m s⁻¹].  Positive = ascending.
    downrange_distance : float
        Cumulative ground-track distance from launch [m].
    """

    # Core state
    vehicle_state:   VehicleState
    wall_clock:      float

    # Atmospheric
    altitude:        float = 0.0
    density:         float = 0.0
    pressure:        float = 0.0
    mach:            float = 0.0
    dynamic_pressure: float = 0.0
    speed_of_sound:  float = 340.294

    # Aerodynamic
    alpha:           float = 0.0
    beta:            float = 0.0
    CL:              float = 0.0
    CD:              float = 0.0
    lift_force:      float = 0.0
    drag_force:      float = 0.0

    # Propulsion
    thrust_magnitude: float = 0.0
    mass_flow_rate:   float = 0.0
    isp_effective:    float = 0.0
    throttle:         float = 0.0

    # Orbital elements
    semi_major_axis:           float = 0.0
    eccentricity:              float = 0.0
    inclination:               float = 0.0
    raan:                      float = 0.0
    argument_of_periapsis:     float = 0.0
    true_anomaly:              float = 0.0
    orbital_period:            float = 0.0
    apoapsis:                  float = 0.0
    periapsis:                 float = 0.0

    # Forces (ECI) — default to zero vectors
    force_gravity:   np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    force_thrust:    np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    force_aero:      np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    force_net:       np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    # Torques (Body) — default to zero vectors
    torque_aero:     np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    torque_gimbal:   np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    torque_net:      np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    # Structural
    worst_structural_margin: float = 1.0
    critical_joint_id:       str   = ""
    any_structural_failure:  bool  = False

    # Navigation
    speed:              float = 0.0
    vertical_speed:     float = 0.0
    downrange_distance: float = 0.0

    # ----------------------------------------------------------------
    # Convenience properties
    # ----------------------------------------------------------------

    @property
    def mission_time(self) -> float:
        """Mission elapsed time (MET) [s]."""
        return self.vehicle_state.time

    @property
    def mass(self) -> float:
        """Current vehicle mass [kg]."""
        return self.vehicle_state.mass

    @property
    def position_eci(self) -> np.ndarray:
        """ECI position [m]."""
        return self.vehicle_state.position_eci

    @property
    def velocity_eci(self) -> np.ndarray:
        """ECI velocity [m s⁻¹]."""
        return self.vehicle_state.velocity_eci

    @property
    def quaternion(self) -> np.ndarray:
        """Attitude quaternion (scalar-first)."""
        return self.vehicle_state.quaternion

    @property
    def omega_body(self) -> np.ndarray:
        """Angular velocity in Body frame [rad s⁻¹]."""
        return self.vehicle_state.omega_body

    @property
    def twr(self) -> float:
        """Thrust-to-weight ratio (dimensionless). 0 if engine off."""
        weight = self.mass * 9.80665
        return self.thrust_magnitude / weight if weight > 0.0 else 0.0

    def __repr__(self) -> str:
        return (
            f"TelemetrySnapshot("
            f"t={self.mission_time:.3f}s, "
            f"alt={self.altitude/1000:.2f}km, "
            f"v={self.speed:.1f}m/s, "
            f"M={self.mach:.3f}, "
            f"m={self.mass:.1f}kg)"
        )


# ---------------------------------------------------------------------------
# Telemetry registry
# ---------------------------------------------------------------------------

class TelemetryRegistry:
    """
    Rolling buffer of immutable TelemetrySnapshots.

    The pipeline calls ``publish()`` once per tick. Downstream consumers
    (AI monitor, renderer, HUD) call ``latest`` or ``history`` for read-only
    access.

    Parameters
    ----------
    buffer_size : int
        Maximum number of snapshots to retain. Older snapshots are
        discarded when the buffer is full. Default 1000 (~16.7 minutes
        at dt=1s, or ~100 seconds at dt=0.01s default).
    """

    def __init__(self, buffer_size: int = 1000) -> None:
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be ≥ 1, got {buffer_size!r}")
        self._buffer: deque[TelemetrySnapshot] = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        self._publish_count: int = 0

    # ------------------------------------------------------------------
    # Write interface (pipeline only)
    # ------------------------------------------------------------------

    def publish(self, snapshot: TelemetrySnapshot) -> None:
        """
        Publish a new snapshot to the registry.

        Called exactly once per simulation tick by the pipeline (Stage 11).
        Snapshots must be published in monotonically increasing mission time.

        Parameters
        ----------
        snapshot : TelemetrySnapshot
            The frozen snapshot to publish.

        Raises
        ------
        TypeError
            If snapshot is not a TelemetrySnapshot instance.
        ValueError
            If mission time is not monotonically increasing (guard against
            out-of-order publishes from a broken pipeline).
        """
        if not isinstance(snapshot, TelemetrySnapshot):
            raise TypeError(
                f"TelemetryRegistry.publish expects TelemetrySnapshot, "
                f"got {type(snapshot).__name__}"
            )
        if self._buffer and snapshot.mission_time < self._buffer[-1].mission_time:
            raise ValueError(
                f"Telemetry time must be monotonically non-decreasing. "
                f"Got t={snapshot.mission_time:.6f} after "
                f"t={self._buffer[-1].mission_time:.6f}"
            )
        self._buffer.append(snapshot)
        self._publish_count += 1

    # ------------------------------------------------------------------
    # Read interface (AI monitor, renderer, HUD)
    # ------------------------------------------------------------------

    @property
    def latest(self) -> Optional[TelemetrySnapshot]:
        """Most recent snapshot, or None if registry is empty."""
        return self._buffer[-1] if self._buffer else None

    @property
    def history(self) -> List[TelemetrySnapshot]:
        """
        All buffered snapshots in chronological order (oldest first).
        Returns a copy of the buffer as a list.
        """
        return list(self._buffer)

    def at_time(self, mission_time: float) -> Optional[TelemetrySnapshot]:
        """
        Return the snapshot closest to ``mission_time`` [s].

        Uses binary search for O(log n) lookup on the time-ordered buffer.

        Parameters
        ----------
        mission_time : float
            Target mission elapsed time [s].

        Returns
        -------
        TelemetrySnapshot or None
            Closest snapshot, or None if the registry is empty.
        """
        if not self._buffer:
            return None
        buf = list(self._buffer)
        # Binary search for closest time
        lo, hi = 0, len(buf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if buf[mid].mission_time < mission_time:
                lo = mid + 1
            else:
                hi = mid
        # lo is the first index ≥ mission_time; compare with lo-1
        if lo > 0:
            before = buf[lo - 1]
            after  = buf[lo]
            if abs(before.mission_time - mission_time) <= abs(after.mission_time - mission_time):
                return before
        return buf[lo]

    def last_n(self, n: int) -> List[TelemetrySnapshot]:
        """
        Return the ``n`` most recent snapshots (newest last).

        Parameters
        ----------
        n : int
            Number of snapshots to return. Clamped to buffer length.
        """
        if n <= 0:
            return []
        buf = list(self._buffer)
        return buf[-n:]

    def derivative(
        self,
        attr: str,
        n_points: int = 2,
    ) -> Optional[float]:
        """
        Estimate the time derivative of a scalar telemetry attribute using
        finite differences on the last ``n_points`` snapshots.

        dX/dt ≈ (X_n − X_{n-1}) / (t_n − t_{n-1})  for n_points=2

        Used by the AI Monitor to compute rate-of-change warnings
        (e.g. dq_inf/dt to predict max-Q exceedance).

        Parameters
        ----------
        attr : str
            Name of a scalar float attribute on TelemetrySnapshot.
            Must be accessible via ``getattr(snapshot, attr)``.
        n_points : int
            Number of snapshots to use. 2 = first-order backward difference.
            3 = second-order central difference on last 3 snapshots.

        Returns
        -------
        float or None
            Estimated derivative [units/s], or None if insufficient data
            or zero time difference.
        """
        if len(self._buffer) < n_points:
            return None

        snapshots = list(self._buffer)[-n_points:]

        try:
            values = [float(getattr(s, attr)) for s in snapshots]
            times  = [s.mission_time for s in snapshots]
        except AttributeError:
            return None

        dt = times[-1] - times[0]
        if abs(dt) < 1.0e-12:
            return None

        if n_points == 2:
            return (values[-1] - values[0]) / dt
        elif n_points == 3:
            # Second-order central difference
            dt1 = times[1] - times[0]
            dt2 = times[2] - times[1]
            if abs(dt1) < 1.0e-12 or abs(dt2) < 1.0e-12:
                return None
            dv1 = (values[1] - values[0]) / dt1
            dv2 = (values[2] - values[1]) / dt2
            return 0.5 * (dv1 + dv2)
        else:
            # Fall back to simple endpoint difference
            return (values[-1] - values[0]) / dt

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def publish_count(self) -> int:
        """Total number of snapshots ever published (not capped by buffer)."""
        return self._publish_count

    @property
    def buffer_size(self) -> int:
        """Maximum buffer capacity."""
        return self._buffer_size

    @property
    def buffer_length(self) -> int:
        """Current number of snapshots in the buffer."""
        return len(self._buffer)

    def clear(self) -> None:
        """Discard all buffered snapshots. Publish count is preserved."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        latest_t = self._buffer[-1].mission_time if self._buffer else 0.0
        return (
            f"TelemetryRegistry("
            f"buffer={self.buffer_length}/{self._buffer_size}, "
            f"published={self._publish_count}, "
            f"latest_t={latest_t:.3f}s)"
        )


# ---------------------------------------------------------------------------
# Snapshot builder helper
# ---------------------------------------------------------------------------

def build_snapshot(
    vehicle_state:    VehicleState,
    *,
    altitude:         float = 0.0,
    density:          float = 0.0,
    pressure:         float = 0.0,
    mach:             float = 0.0,
    dynamic_pressure: float = 0.0,
    speed_of_sound:   float = 340.294,
    alpha:            float = 0.0,
    beta:             float = 0.0,
    CL:               float = 0.0,
    CD:               float = 0.0,
    lift_force:       float = 0.0,
    drag_force:       float = 0.0,
    thrust_magnitude: float = 0.0,
    mass_flow_rate:   float = 0.0,
    isp_effective:    float = 0.0,
    throttle:         float = 0.0,
    semi_major_axis:  float = 0.0,
    eccentricity:     float = 0.0,
    inclination:      float = 0.0,
    raan:             float = 0.0,
    argument_of_periapsis: float = 0.0,
    true_anomaly:     float = 0.0,
    orbital_period:   float = 0.0,
    apoapsis:         float = 0.0,
    periapsis:        float = 0.0,
    force_gravity:    Optional[np.ndarray] = None,
    force_thrust:     Optional[np.ndarray] = None,
    force_aero:       Optional[np.ndarray] = None,
    force_net:        Optional[np.ndarray] = None,
    torque_aero:      Optional[np.ndarray] = None,
    torque_gimbal:    Optional[np.ndarray] = None,
    torque_net:       Optional[np.ndarray] = None,
    worst_structural_margin: float = 1.0,
    critical_joint_id: str  = "",
    any_structural_failure: bool = False,
    speed:            float = 0.0,
    vertical_speed:   float = 0.0,
    downrange_distance: float = 0.0,
) -> TelemetrySnapshot:
    """
    Convenience constructor for TelemetrySnapshot.

    All ndarray fields default to zero vectors if not supplied.
    Computes wall_clock automatically.

    Parameters mirror TelemetrySnapshot fields exactly; see that class
    for full documentation.
    """
    _z3 = np.zeros(3, dtype=np.float64)
    return TelemetrySnapshot(
        vehicle_state=vehicle_state,
        wall_clock=_time.time(),
        altitude=altitude,
        density=density,
        pressure=pressure,
        mach=mach,
        dynamic_pressure=dynamic_pressure,
        speed_of_sound=speed_of_sound,
        alpha=alpha,
        beta=beta,
        CL=CL,
        CD=CD,
        lift_force=lift_force,
        drag_force=drag_force,
        thrust_magnitude=thrust_magnitude,
        mass_flow_rate=mass_flow_rate,
        isp_effective=isp_effective,
        throttle=throttle,
        semi_major_axis=semi_major_axis,
        eccentricity=eccentricity,
        inclination=inclination,
        raan=raan,
        argument_of_periapsis=argument_of_periapsis,
        true_anomaly=true_anomaly,
        orbital_period=orbital_period,
        apoapsis=apoapsis,
        periapsis=periapsis,
        force_gravity=force_gravity  if force_gravity  is not None else _z3.copy(),
        force_thrust =force_thrust   if force_thrust   is not None else _z3.copy(),
        force_aero   =force_aero     if force_aero     is not None else _z3.copy(),
        force_net    =force_net      if force_net       is not None else _z3.copy(),
        torque_aero  =torque_aero    if torque_aero    is not None else _z3.copy(),
        torque_gimbal=torque_gimbal  if torque_gimbal  is not None else _z3.copy(),
        torque_net   =torque_net     if torque_net     is not None else _z3.copy(),
        worst_structural_margin=worst_structural_margin,
        critical_joint_id=critical_joint_id,
        any_structural_failure=any_structural_failure,
        speed=speed,
        vertical_speed=vertical_speed,
        downrange_distance=downrange_distance,
    )
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_telemetry_registry.py
======================================
Unit tests for nova.core.telemetry_registry.

Tests verify:
  1. TelemetrySnapshot construction with defaults and explicit values.
  2. Frozen dataclass — no mutation allowed.
  3. Convenience properties (mission_time, mass, twr).
  4. TelemetryRegistry publish — single, multiple, monotonic time.
  5. Non-monotonic time raises ValueError.
  6. Wrong type raises TypeError.
  7. latest, history, last_n, at_time queries.
  8. derivative() — backward difference on scalar attributes.
  9. buffer_size limiting discards oldest snapshots.
  10. clear() resets buffer but preserves publish_count.
  11. build_snapshot helper populates all fields.
  12. repr strings for both classes.
"""

import math
import time
import pytest
import numpy as np

from nova.core.state_vector import make_state
from nova.core.telemetry_registry import (
    TelemetrySnapshot,
    TelemetryRegistry,
    build_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vehicle_state(t: float = 0.0, mass: float = 1000.0) -> "VehicleState":
    from nova.core.state_vector import make_state
    return make_state(
        position_eci=[6_771_000.0, 0.0, 0.0],
        velocity_eci=[0.0, 7_672.0, 0.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        omega_body=[0.0, 0.0, 0.0],
        mass=mass,
        time=t,
    )


def _snap(t: float = 0.0, mass: float = 1000.0, **kwargs) -> TelemetrySnapshot:
    return build_snapshot(_vehicle_state(t, mass), **kwargs)


# ---------------------------------------------------------------------------
# 1. TelemetrySnapshot construction
# ---------------------------------------------------------------------------

class TestTelemetrySnapshotConstruction:

    def test_default_fields_are_zero(self):
        s = _snap(0.0)
        assert s.altitude == 0.0
        assert s.mach     == 0.0
        assert s.throttle == 0.0
        assert s.eccentricity == 0.0

    def test_explicit_scalar_fields(self):
        s = _snap(1.0, altitude=400_000.0, mach=7.8, throttle=0.9)
        assert s.altitude == 400_000.0
        assert s.mach     == 7.8
        assert s.throttle == 0.9

    def test_vector_fields_default_zero(self):
        s = _snap(0.0)
        assert np.allclose(s.force_gravity, [0.0, 0.0, 0.0])
        assert np.allclose(s.force_net,     [0.0, 0.0, 0.0])
        assert np.allclose(s.torque_net,    [0.0, 0.0, 0.0])

    def test_explicit_vector_fields(self):
        F = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        s = _snap(0.0, force_gravity=F)
        assert np.allclose(s.force_gravity, F)

    def test_wall_clock_set(self):
        before = time.time()
        s      = _snap(0.0)
        after  = time.time()
        assert before <= s.wall_clock <= after

    def test_mission_time_property(self):
        s = _snap(42.5)
        assert s.mission_time == 42.5

    def test_mass_property(self):
        s = _snap(0.0, mass=500.0)
        assert s.mass == 500.0

    def test_speed_property_from_vehicle_state(self):
        s = _snap(0.0)
        assert abs(s.vehicle_state.speed - 7_672.0) < 0.01

    def test_twr_with_thrust(self):
        """TWR = F_thrust / (m · g₀)."""
        mass   = 1000.0
        thrust = 20_000.0
        s = _snap(0.0, mass=mass, thrust_magnitude=thrust)
        expected = thrust / (mass * 9.80665)
        assert abs(s.twr - expected) < 1.0e-6

    def test_twr_zero_when_no_thrust(self):
        s = _snap(0.0, thrust_magnitude=0.0)
        assert s.twr == 0.0


# ---------------------------------------------------------------------------
# 2. Immutability
# ---------------------------------------------------------------------------

class TestTelemetrySnapshotImmutability:

    def test_cannot_set_scalar_field(self):
        s = _snap(0.0)
        with pytest.raises(Exception):
            s.altitude = 999.0

    def test_cannot_set_vector_field(self):
        s = _snap(0.0)
        with pytest.raises(Exception):
            s.force_net = np.ones(3)

    def test_cannot_set_vehicle_state(self):
        s = _snap(0.0)
        with pytest.raises(Exception):
            s.vehicle_state = _vehicle_state(1.0)

    def test_repr_contains_time(self):
        s = _snap(5.5)
        assert "5.500s" in repr(s)


# ---------------------------------------------------------------------------
# 3. TelemetryRegistry publish
# ---------------------------------------------------------------------------

class TestTelemetryRegistryPublish:

    def test_publish_single(self):
        reg = TelemetryRegistry()
        reg.publish(_snap(0.0))
        assert len(reg) == 1

    def test_publish_multiple(self):
        reg = TelemetryRegistry()
        for i in range(5):
            reg.publish(_snap(float(i)))
        assert len(reg) == 5

    def test_publish_count_tracks_all(self):
        reg = TelemetryRegistry(buffer_size=3)
        for i in range(10):
            reg.publish(_snap(float(i)))
        assert reg.publish_count == 10
        assert len(reg) == 3   # buffer capped at 3

    def test_wrong_type_raises(self):
        reg = TelemetryRegistry()
        with pytest.raises(TypeError, match="TelemetrySnapshot"):
            reg.publish("not a snapshot")  # type: ignore

    def test_non_monotonic_time_raises(self):
        reg = TelemetryRegistry()
        reg.publish(_snap(10.0))
        with pytest.raises(ValueError, match="monotonically"):
            reg.publish(_snap(9.0))

    def test_equal_time_allowed(self):
        """Two snapshots at the same time are allowed (same-tick republish)."""
        reg = TelemetryRegistry()
        reg.publish(_snap(5.0))
        reg.publish(_snap(5.0))   # should not raise
        assert len(reg) == 2

    def test_buffer_size_one(self):
        reg = TelemetryRegistry(buffer_size=1)
        reg.publish(_snap(0.0))
        reg.publish(_snap(1.0))
        assert len(reg) == 1
        assert reg.latest.mission_time == 1.0

    def test_invalid_buffer_size_raises(self):
        with pytest.raises(ValueError, match="buffer_size"):
            TelemetryRegistry(buffer_size=0)


# ---------------------------------------------------------------------------
# 4. Read queries
# ---------------------------------------------------------------------------

class TestTelemetryRegistryQueries:

    @pytest.fixture
    def filled_registry(self):
        reg = TelemetryRegistry(buffer_size=20)
        for i in range(10):
            reg.publish(_snap(float(i) * 0.1))
        return reg

    def test_latest_returns_newest(self, filled_registry):
        assert abs(filled_registry.latest.mission_time - 0.9) < 1.0e-10

    def test_latest_none_when_empty(self):
        reg = TelemetryRegistry()
        assert reg.latest is None

    def test_history_ordered_oldest_first(self, filled_registry):
        h = filled_registry.history
        for i in range(1, len(h)):
            assert h[i].mission_time >= h[i-1].mission_time

    def test_history_length(self, filled_registry):
        assert len(filled_registry.history) == 10

    def test_last_n(self, filled_registry):
        last3 = filled_registry.last_n(3)
        assert len(last3) == 3
        # Newest is last
        assert abs(last3[-1].mission_time - 0.9) < 1.0e-10

    def test_last_n_zero_returns_empty(self, filled_registry):
        assert filled_registry.last_n(0) == []

    def test_last_n_larger_than_buffer(self, filled_registry):
        result = filled_registry.last_n(100)
        assert len(result) == 10

    def test_at_time_exact(self, filled_registry):
        s = filled_registry.at_time(0.5)
        assert s is not None
        assert abs(s.mission_time - 0.5) < 1.0e-10

    def test_at_time_nearest(self, filled_registry):
        """t=0.35 → nearest is 0.3 or 0.4."""
        s = filled_registry.at_time(0.35)
        assert s is not None
        assert abs(s.mission_time - 0.3) < 0.15 or abs(s.mission_time - 0.4) < 0.15

    def test_at_time_empty_returns_none(self):
        reg = TelemetryRegistry()
        assert reg.at_time(1.0) is None

    def test_at_time_before_buffer_start(self, filled_registry):
        """Query before any snapshot returns the earliest."""
        s = filled_registry.at_time(-5.0)
        assert s is not None
        assert abs(s.mission_time - 0.0) < 1.0e-10

    def test_at_time_after_buffer_end(self, filled_registry):
        """Query after all snapshots returns the latest."""
        s = filled_registry.at_time(999.0)
        assert s is not None
        assert abs(s.mission_time - 0.9) < 1.0e-10


# ---------------------------------------------------------------------------
# 5. derivative()
# ---------------------------------------------------------------------------

class TestTelemetryDerivative:

    def test_derivative_altitude_constant(self):
        """Constant altitude → dalt/dt = 0."""
        reg = TelemetryRegistry()
        for i in range(5):
            reg.publish(_snap(float(i), altitude=5000.0))
        d = reg.derivative("altitude", n_points=2)
        assert d is not None
        assert abs(d) < 1.0e-6

    def test_derivative_altitude_linear_increase(self):
        """altitude = 1000 * t → dalt/dt = 1000."""
        reg = TelemetryRegistry()
        for i in range(5):
            t = float(i) * 0.1
            reg.publish(_snap(t, altitude=1000.0 * t))
        d = reg.derivative("altitude", n_points=2)
        assert d is not None
        assert abs(d - 1000.0) < 1.0, f"dalt/dt = {d:.4f}, expected 1000"

    def test_derivative_dynamic_pressure_increasing(self):
        """q increases with time — derivative should be positive."""
        reg = TelemetryRegistry()
        for i in range(3):
            t = float(i) * 0.1
            reg.publish(_snap(t, dynamic_pressure=t * 10_000.0))
        d = reg.derivative("dynamic_pressure", n_points=2)
        assert d is not None
        assert d > 0.0

    def test_derivative_returns_none_insufficient_data(self):
        reg = TelemetryRegistry()
        reg.publish(_snap(0.0))
        d = reg.derivative("altitude", n_points=2)
        assert d is None

    def test_derivative_invalid_attribute_returns_none(self):
        reg = TelemetryRegistry()
        reg.publish(_snap(0.0))
        reg.publish(_snap(0.1))
        d = reg.derivative("nonexistent_field", n_points=2)
        assert d is None

    def test_derivative_three_point(self):
        """3-point central difference on linearly increasing field."""
        reg = TelemetryRegistry()
        for i in range(3):
            t = float(i) * 0.1
            reg.publish(_snap(t, mach=t * 2.0))
        d = reg.derivative("mach", n_points=3)
        assert d is not None
        assert abs(d - 2.0) < 0.1


# ---------------------------------------------------------------------------
# 6. Buffer management
# ---------------------------------------------------------------------------

class TestBufferManagement:

    def test_buffer_caps_at_max_size(self):
        reg = TelemetryRegistry(buffer_size=5)
        for i in range(10):
            reg.publish(_snap(float(i)))
        assert len(reg) == 5

    def test_oldest_discarded_first(self):
        reg = TelemetryRegistry(buffer_size=3)
        for i in range(5):
            reg.publish(_snap(float(i)))
        h = reg.history
        # Should contain t=2, 3, 4
        times = [s.mission_time for s in h]
        assert 2.0 in times
        assert 0.0 not in times

    def test_clear_empties_buffer(self):
        reg = TelemetryRegistry()
        for i in range(5):
            reg.publish(_snap(float(i)))
        count_before = reg.publish_count
        reg.clear()
        assert len(reg) == 0
        assert reg.publish_count == count_before   # preserved

    def test_clear_then_republish(self):
        """After clear, can publish with non-monotonic time (fresh start)."""
        reg = TelemetryRegistry()
        reg.publish(_snap(10.0))
        reg.clear()
        reg.publish(_snap(0.0))   # should not raise
        assert reg.latest.mission_time == 0.0


# ---------------------------------------------------------------------------
# 7. build_snapshot helper
# ---------------------------------------------------------------------------

class TestBuildSnapshot:

    def test_all_scalar_fields_set(self):
        vs = _vehicle_state(5.0)
        s  = build_snapshot(
            vs,
            altitude=400_000.0,
            density=8.0e-4,
            mach=7.8,
            dynamic_pressure=2_500.0,
            thrust_magnitude=50_000.0,
            semi_major_axis=6_771_000.0,
            eccentricity=0.001,
        )
        assert s.altitude         == 400_000.0
        assert s.density          == 8.0e-4
        assert s.mach             == 7.8
        assert s.dynamic_pressure == 2_500.0
        assert s.thrust_magnitude == 50_000.0
        assert abs(s.semi_major_axis - 6_771_000.0) < 1.0
        assert abs(s.eccentricity - 0.001) < 1.0e-10

    def test_force_vectors_set(self):
        vs = _vehicle_state()
        F  = np.array([0.0, 0.0, -9810.0], dtype=np.float64)
        s  = build_snapshot(vs, force_gravity=F)
        assert np.allclose(s.force_gravity, F)

    def test_vehicle_state_preserved(self):
        vs = _vehicle_state(3.14)
        s  = build_snapshot(vs)
        assert s.vehicle_state is vs

    def test_structural_fields(self):
        vs = _vehicle_state()
        s  = build_snapshot(vs, worst_structural_margin=0.15,
                            critical_joint_id="interstage",
                            any_structural_failure=False)
        assert s.worst_structural_margin == 0.15
        assert s.critical_joint_id == "interstage"
        assert s.any_structural_failure is False


# ---------------------------------------------------------------------------
# 8. Registry repr
# ---------------------------------------------------------------------------

class TestRegistryRepr:

    def test_repr_contains_buffer_info(self):
        reg = TelemetryRegistry(buffer_size=100)
        reg.publish(_snap(1.5))
        r = repr(reg)
        assert "1/100" in r
        assert "published=1" in r
        assert "1.500" in r

    def test_buffer_length_property(self):
        reg = TelemetryRegistry()
        assert reg.buffer_length == 0
        reg.publish(_snap(0.0))
        assert reg.buffer_length == 1
