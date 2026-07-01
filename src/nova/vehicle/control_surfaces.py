"""
nova.vehicle.control_surfaces
==============================
Aerodynamic control surface actuator model for Project NOVA.

Architectural role
------------------
Phase 8 — Vehicle Resource Models.
Pipeline stage: Stage 2 (Vehicle Controller). Receives raw ControlInput
commands and converts them to physical ControlDeflections, accounting for
hardware velocity limits, deadband, and authority limits.

I/O contract
------------
Input  : ControlInput(elevator, aileron, rudder) raw commands [-1, 1]
         ActuatorConfig per surface (rate limit, deadband, max deflection)
         dt [s] — simulation timestep
         current deflection state (previous tick's output)
Output : ControlDeflections(δ_e, δ_a, δ_r) [rad] — frozen dataclass
         ControlSurfaceState snapshot — frozen, with per-surface positions

Physical basis
--------------
Each aerodynamic control surface is modelled as a rate-limited, deadband-
filtered, authority-limited actuator:

  1. Command normalisation: raw input ∈ [-1, 1] → target angle ∈ [-δ_max, δ_max]
  2. Deadband: |target| < deadband → target = 0 (prevents hunting near neutral)
  3. Rate limiting: the achieved deflection cannot change faster than
     δ_dot_max [rad s⁻¹] per timestep:
       δ_new = δ_prev + clip(δ_target − δ_prev, −δ_dot_max·dt, +δ_dot_max·dt)
  4. Authority clipping: δ_new = clip(δ_new, −δ_max, +δ_max)

Surfaces modelled:
  - Elevator  (δ_e): pitch control, positive trailing-edge-up
  - Aileron   (δ_a): roll control, positive right-aileron-down (left-up)
  - Rudder    (δ_r): yaw control, positive trailing-edge-left

Deflection sign conventions follow the NOVA aerodynamics stability-derivative
model in nova.physics.aerodynamics. All angles in radians internally.

References
----------
- Stevens & Lewis, "Aircraft Control and Simulation", 3rd ed., §2.5
- Etkin & Reid, "Dynamics of Flight", §6.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from nova.physics.aerodynamics import ControlDeflections

# ---------------------------------------------------------------------------
# ActuatorConfig — per-surface hardware limits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActuatorConfig:
    """
    Hardware parameters for a single control surface actuator.

    Attributes
    ----------
    name : str
        Human-readable surface name (e.g. "elevator", "aileron", "rudder").
    max_deflection_rad : float
        Maximum physical deflection magnitude [rad]. Must be positive.
        Symmetric: deflection range is [−max, +max].
    rate_limit_rad_s : float
        Maximum angular rate of change [rad s⁻¹]. Must be positive.
        Set to math.inf to disable rate limiting.
    deadband_rad : float
        Deflection magnitude below which the command is treated as zero [rad].
        Must be non-negative. Default 0.0 (no deadband).
    """

    name: str
    max_deflection_rad: float
    rate_limit_rad_s: float
    deadband_rad: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ActuatorConfig.name must be a non-empty string")
        max_d = float(self.max_deflection_rad)
        if max_d <= 0.0:
            raise ValueError(
                f"max_deflection_rad must be positive; got {max_d:.6g}"
            )
        object.__setattr__(self, "max_deflection_rad", max_d)

        rate = float(self.rate_limit_rad_s)
        if rate <= 0.0:
            raise ValueError(
                f"rate_limit_rad_s must be positive; got {rate:.6g}"
            )
        object.__setattr__(self, "rate_limit_rad_s", rate)

        db = float(self.deadband_rad)
        if db < 0.0:
            raise ValueError(
                f"deadband_rad must be non-negative; got {db:.6g}"
            )
        if db >= max_d:
            raise ValueError(
                f"deadband_rad ({db:.6g}) must be less than "
                f"max_deflection_rad ({max_d:.6g})"
            )
        object.__setattr__(self, "deadband_rad", db)


# ---------------------------------------------------------------------------
# ControlSurfaceState — per-tick actuator snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControlSurfaceState:
    """
    Immutable snapshot of all three control surface positions at one tick.

    Attributes
    ----------
    elevator_rad : float
        Elevator deflection δ_e [rad]. Positive = trailing-edge-up (pitch-up).
    aileron_rad : float
        Aileron deflection δ_a [rad]. Positive = right-aileron-down (roll-right).
    rudder_rad : float
        Rudder deflection δ_r [rad]. Positive = trailing-edge-left (yaw-left).
    elevator_rate : float
        Elevator rate of change this tick [rad s⁻¹].
    aileron_rate : float
        Aileron rate of change this tick [rad s⁻¹].
    rudder_rate : float
        Rudder rate of change this tick [rad s⁻¹].
    """

    elevator_rad: float
    aileron_rad: float
    rudder_rad: float
    elevator_rate: float
    aileron_rate: float
    rudder_rate: float

    def __post_init__(self) -> None:
        for field in ("elevator_rad", "aileron_rad", "rudder_rad",
                      "elevator_rate", "aileron_rate", "rudder_rate"):
            object.__setattr__(self, field, float(getattr(self, field)))

    def to_deflections(self) -> ControlDeflections:
        """
        Convert to a ControlDeflections instance for the aerodynamics model.

        Returns
        -------
        ControlDeflections
        """
        return ControlDeflections(
            elevator=self.elevator_rad,
            aileron=self.aileron_rad,
            rudder=self.rudder_rad,
        )

    def __repr__(self) -> str:
        return (
            f"ControlSurfaceState("
            f"δ_e={math.degrees(self.elevator_rad):.2f}°, "
            f"δ_a={math.degrees(self.aileron_rad):.2f}°, "
            f"δ_r={math.degrees(self.rudder_rad):.2f}°)"
        )


# ---------------------------------------------------------------------------
# ControlSurfaceConfig — configuration bundle for all three surfaces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControlSurfaceConfig:
    """
    Actuator configuration for all three primary control surfaces.

    Attributes
    ----------
    elevator : ActuatorConfig
        Elevator actuator parameters.
    aileron : ActuatorConfig
        Aileron actuator parameters.
    rudder : ActuatorConfig
        Rudder actuator parameters.
    """

    elevator: ActuatorConfig
    aileron: ActuatorConfig
    rudder: ActuatorConfig

    def __post_init__(self) -> None:
        for field in ("elevator", "aileron", "rudder"):
            if not isinstance(getattr(self, field), ActuatorConfig):
                raise TypeError(
                    f"ControlSurfaceConfig.{field} must be an ActuatorConfig"
                )


# ---------------------------------------------------------------------------
# Pure actuator step function
# ---------------------------------------------------------------------------

def _actuator_step(
    command_norm: float,
    current_rad: float,
    config: ActuatorConfig,
    dt: float,
) -> tuple[float, float]:
    """
    Advance a single actuator by one timestep.

    Parameters
    ----------
    command_norm : float
        Normalised command input in [-1, 1]. Values outside this range are
        clamped to [-1, 1].
    current_rad : float
        Current actuator position [rad] from the previous tick.
    config : ActuatorConfig
        Hardware parameters for this surface.
    dt : float
        Timestep [s]. Must be positive.

    Returns
    -------
    new_position_rad : float
        Updated actuator position [rad].
    rate_rad_s : float
        Achieved rate of change this step [rad s⁻¹].
    """
    # 1. Clamp normalised command to [-1, 1]
    cmd = float(np.clip(command_norm, -1.0, 1.0))

    # 2. Convert to target angle
    target = cmd * config.max_deflection_rad

    # 3. Apply deadband
    if abs(target) < config.deadband_rad:
        target = 0.0

    # 4. Rate limiting
    delta = target - current_rad
    max_delta = config.rate_limit_rad_s * dt
    delta_clamped = float(np.clip(delta, -max_delta, max_delta))
    new_pos = current_rad + delta_clamped

    # 5. Authority clipping
    new_pos = float(np.clip(new_pos, -config.max_deflection_rad, config.max_deflection_rad))

    # 6. Rate achieved (after clipping)
    rate = (new_pos - current_rad) / dt if dt > 0.0 else 0.0

    return new_pos, rate


# ---------------------------------------------------------------------------
# ControlSurfaceActuator — stateful actuator model
# ---------------------------------------------------------------------------

class ControlSurfaceActuator:
    """
    Stateful actuator model for all three aerodynamic control surfaces.

    Holds the current surface positions and advances them each tick using
    rate limiting, deadband filtering, and authority clamping.

    This class is intentionally NOT frozen — it tracks mutable actuator
    positions across ticks. It is the pipeline's responsibility to call
    step() exactly once per tick and extract the resulting
    ControlSurfaceState.

    Parameters
    ----------
    config : ControlSurfaceConfig
        Hardware limits for all three surfaces.
    initial_state : ControlSurfaceState, optional
        Starting surface positions. Defaults to all-zero (neutral).
    """

    def __init__(
        self,
        config: ControlSurfaceConfig,
        initial_state: Optional[ControlSurfaceState] = None,
    ) -> None:
        if not isinstance(config, ControlSurfaceConfig):
            raise TypeError("config must be a ControlSurfaceConfig")
        self._config = config

        if initial_state is not None:
            if not isinstance(initial_state, ControlSurfaceState):
                raise TypeError("initial_state must be a ControlSurfaceState")
            self._elevator = float(initial_state.elevator_rad)
            self._aileron = float(initial_state.aileron_rad)
            self._rudder = float(initial_state.rudder_rad)
        else:
            self._elevator = 0.0
            self._aileron = 0.0
            self._rudder = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ControlSurfaceConfig:
        """Read-only reference to the actuator configuration."""
        return self._config

    @property
    def current_state(self) -> ControlSurfaceState:
        """
        Current surface positions as a frozen snapshot (rates = 0).

        Use step() to advance and retrieve the updated snapshot with rates.
        """
        return ControlSurfaceState(
            elevator_rad=self._elevator,
            aileron_rad=self._aileron,
            rudder_rad=self._rudder,
            elevator_rate=0.0,
            aileron_rate=0.0,
            rudder_rate=0.0,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        elevator_cmd: float,
        aileron_cmd: float,
        rudder_cmd: float,
        dt: float,
    ) -> ControlSurfaceState:
        """
        Advance all three actuators by one simulation timestep.

        Parameters
        ----------
        elevator_cmd : float
            Normalised elevator command [-1, 1]. +1 = full trailing-edge-up.
        aileron_cmd : float
            Normalised aileron command [-1, 1]. +1 = full right-aileron-down.
        rudder_cmd : float
            Normalised rudder command [-1, 1]. +1 = full trailing-edge-left.
        dt : float
            Timestep [s]. Must be positive.

        Returns
        -------
        ControlSurfaceState
            Frozen snapshot of the new surface positions and achieved rates.

        Raises
        ------
        ValueError
            If dt ≤ 0.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive; got {dt:.6g}")

        new_e, rate_e = _actuator_step(elevator_cmd, self._elevator,
                                       self._config.elevator, dt)
        new_a, rate_a = _actuator_step(aileron_cmd, self._aileron,
                                       self._config.aileron, dt)
        new_r, rate_r = _actuator_step(rudder_cmd, self._rudder,
                                       self._config.rudder, dt)

        self._elevator = new_e
        self._aileron = new_a
        self._rudder = new_r

        return ControlSurfaceState(
            elevator_rad=new_e,
            aileron_rad=new_a,
            rudder_rad=new_r,
            elevator_rate=rate_e,
            aileron_rate=rate_a,
            rudder_rate=rate_r,
        )

    def reset(self) -> None:
        """Return all surfaces to neutral (zero deflection)."""
        self._elevator = 0.0
        self._aileron = 0.0
        self._rudder = 0.0

    def __repr__(self) -> str:
        return (
            f"ControlSurfaceActuator("
            f"δ_e={math.degrees(self._elevator):.2f}°, "
            f"δ_a={math.degrees(self._aileron):.2f}°, "
            f"δ_r={math.degrees(self._rudder):.2f}°)"
        )


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def default_control_surface_config(
    max_elevator_rad: float = math.radians(25.0),
    max_aileron_rad: float = math.radians(20.0),
    max_rudder_rad: float = math.radians(30.0),
    elevator_rate_rad_s: float = math.radians(60.0),
    aileron_rate_rad_s: float = math.radians(80.0),
    rudder_rate_rad_s: float = math.radians(60.0),
    deadband_rad: float = math.radians(0.1),
) -> ControlSurfaceConfig:
    """
    Build a ControlSurfaceConfig with typical general-aviation style limits.

    Default values are representative of a medium-sized subsonic aircraft.
    All angles in radians.

    Parameters
    ----------
    max_elevator_rad : float
        Elevator authority [rad]. Default 25°.
    max_aileron_rad : float
        Aileron authority [rad]. Default 20°.
    max_rudder_rad : float
        Rudder authority [rad]. Default 30°.
    elevator_rate_rad_s : float
        Elevator slew rate [rad s⁻¹]. Default 60°/s.
    aileron_rate_rad_s : float
        Aileron slew rate [rad s⁻¹]. Default 80°/s.
    rudder_rate_rad_s : float
        Rudder slew rate [rad s⁻¹]. Default 60°/s.
    deadband_rad : float
        Deadband for all surfaces [rad]. Default 0.1°.

    Returns
    -------
    ControlSurfaceConfig
    """
    return ControlSurfaceConfig(
        elevator=ActuatorConfig(
            name="elevator",
            max_deflection_rad=max_elevator_rad,
            rate_limit_rad_s=elevator_rate_rad_s,
            deadband_rad=deadband_rad,
        ),
        aileron=ActuatorConfig(
            name="aileron",
            max_deflection_rad=max_aileron_rad,
            rate_limit_rad_s=aileron_rate_rad_s,
            deadband_rad=deadband_rad,
        ),
        rudder=ActuatorConfig(
            name="rudder",
            max_deflection_rad=max_rudder_rad,
            rate_limit_rad_s=rudder_rate_rad_s,
            deadband_rad=deadband_rad,
        ),
    )
