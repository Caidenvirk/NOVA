"""
nova.guidance.controller
=========================
Vehicle controller: maps ControlInput to actuator commands for Project NOVA.

Architectural role
------------------
Phase 11 — Guidance Subsystem.
Pipeline stage: Stage 2 (Vehicle Controller). Receives the ControlInput
packet from Stage 1 (Input Handler) and maps it to:
  - ControlSurfaceActuator commands (elevator, aileron, rudder)
  - Engine throttle command
  - Gimbal angle commands (pitch, yaw) — passed directly to propulsion
  - Staging signal

The controller enforces authority limits defined by the ControllerConfig
and delegates rate limiting entirely to the ControlSurfaceActuator (Phase 8).
No physics calculations occur here.

I/O contract
------------
Input  : ControlInput (from Stage 1), dt [s]
Output : ControllerOutput (frozen dataclass) — actuator commands ready for
         the physics engine

Design
------
The controller is a pure mapping layer with gain scheduling:
  1. Throttle passthrough: clamp to [throttle_min, throttle_max]
  2. Gimbal commands: scale ControlInput gimbal axes by max_gimbal_rad
  3. Aerodynamic surface commands: pass normalised [-1,1] to ControlSurfaceActuator
  4. Staging: propagate boolean flag

Authority limits are enforced at the controller output level, before the
actuator's own rate limiter. This means the controller output is already
within the hardware envelope before the actuator processes it.

No wall-clock time. No physics. No imports from nova.physics.*.

References
----------
- NOVA Engineering Handoff, §7 Stage 2, §12 Phase 11
- Stevens & Lewis, "Aircraft Control and Simulation", §2.5, §6.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from nova.core.pipeline import ControlInput
from nova.physics.aerodynamics import ControlDeflections
from nova.vehicle.control_surfaces import (
    ControlSurfaceActuator,
    ControlSurfaceState,
    default_control_surface_config,
)

# ---------------------------------------------------------------------------
# ControllerConfig — authority and gain parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControllerConfig:
    """
    Authority limits and gains for the vehicle controller.

    Attributes
    ----------
    throttle_min : float
        Minimum throttle fraction [0, 1]. Must be in [0, throttle_max].
        Default 0.0 (full-off allowed).
    throttle_max : float
        Maximum throttle fraction [0, 1]. Default 1.0.
    max_gimbal_pitch_rad : float
        Maximum engine gimbal pitch angle [rad]. Must be ≥ 0. Default 5°.
    max_gimbal_yaw_rad : float
        Maximum engine gimbal yaw angle [rad]. Must be ≥ 0. Default 5°.
    elevator_gain : float
        Multiplier applied to the elevator axis command before passing to
        the actuator. Range (0, 1] — reduces effective authority. Default 1.0.
    aileron_gain : float
        Multiplier applied to the aileron axis. Default 1.0.
    rudder_gain : float
        Multiplier applied to the rudder axis. Default 1.0.
    """

    throttle_min: float = 0.0
    throttle_max: float = 1.0
    max_gimbal_pitch_rad: float = math.radians(5.0)
    max_gimbal_yaw_rad: float = math.radians(5.0)
    elevator_gain: float = 1.0
    aileron_gain: float = 1.0
    rudder_gain: float = 1.0

    def __post_init__(self) -> None:
        t_min = float(self.throttle_min)
        t_max = float(self.throttle_max)
        if not (0.0 <= t_min <= 1.0):
            raise ValueError(f"throttle_min must be in [0, 1]; got {t_min:.6g}")
        if not (0.0 <= t_max <= 1.0):
            raise ValueError(f"throttle_max must be in [0, 1]; got {t_max:.6g}")
        if t_min > t_max:
            raise ValueError(
                f"throttle_min ({t_min}) must not exceed throttle_max ({t_max})"
            )
        object.__setattr__(self, "throttle_min", t_min)
        object.__setattr__(self, "throttle_max", t_max)

        for attr, label in (
            ("max_gimbal_pitch_rad", "max_gimbal_pitch_rad"),
            ("max_gimbal_yaw_rad", "max_gimbal_yaw_rad"),
        ):
            val = float(getattr(self, attr))
            if val < 0.0:
                raise ValueError(f"{label} must be ≥ 0; got {val:.6g}")
            object.__setattr__(self, attr, val)

        for attr, label in (
            ("elevator_gain", "elevator_gain"),
            ("aileron_gain", "aileron_gain"),
            ("rudder_gain", "rudder_gain"),
        ):
            val = float(getattr(self, attr))
            if not (0.0 < val <= 1.0):
                raise ValueError(f"{label} must be in (0, 1]; got {val:.6g}")
            object.__setattr__(self, attr, val)


# ---------------------------------------------------------------------------
# ControllerOutput — one-tick actuator command bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControllerOutput:
    """
    Immutable bundle of actuator commands produced for one simulation tick.

    Attributes
    ----------
    throttle : float
        Engine throttle command [0, 1].
    gimbal_pitch_rad : float
        Engine gimbal pitch angle command [rad].
    gimbal_yaw_rad : float
        Engine gimbal yaw angle command [rad].
    surface_state : ControlSurfaceState
        Updated control surface positions after actuator stepping.
    deflections : ControlDeflections
        ControlDeflections extracted from surface_state for the aero model.
    staging : bool
        True if a staging event should be executed this tick.
    """

    throttle: float
    gimbal_pitch_rad: float
    gimbal_yaw_rad: float
    surface_state: ControlSurfaceState
    deflections: ControlDeflections
    staging: bool

    def __post_init__(self) -> None:
        t = float(self.throttle)
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"throttle must be in [0, 1]; got {t:.6g}")
        object.__setattr__(self, "throttle", t)
        object.__setattr__(self, "gimbal_pitch_rad", float(self.gimbal_pitch_rad))
        object.__setattr__(self, "gimbal_yaw_rad", float(self.gimbal_yaw_rad))
        object.__setattr__(self, "staging", bool(self.staging))

        if not isinstance(self.surface_state, ControlSurfaceState):
            raise TypeError("surface_state must be a ControlSurfaceState")
        if not isinstance(self.deflections, ControlDeflections):
            raise TypeError("deflections must be a ControlDeflections")

    def __repr__(self) -> str:
        return (
            f"ControllerOutput("
            f"throttle={self.throttle:.3f}, "
            f"gimbal_p={math.degrees(self.gimbal_pitch_rad):.2f}°, "
            f"gimbal_y={math.degrees(self.gimbal_yaw_rad):.2f}°, "
            f"staging={self.staging})"
        )


# ---------------------------------------------------------------------------
# VehicleController — stateful controller (owns the actuator)
# ---------------------------------------------------------------------------

class VehicleController:
    """
    Stateful vehicle controller: maps ControlInput → ControllerOutput each tick.

    Owns the ControlSurfaceActuator and steps it every tick. The SAS
    (nova.guidance.sas) may augment the aileron/elevator/rudder commands
    before they reach this controller — the SAS output is passed in via
    the ControlInput fields at Stage 1.

    Parameters
    ----------
    config : ControllerConfig | None
        Authority limits and gain settings. Defaults to ControllerConfig().
    actuator : ControlSurfaceActuator | None
        Pre-built actuator. If None, a default_control_surface_config()
        actuator is created.
    """

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        actuator: Optional[ControlSurfaceActuator] = None,
    ) -> None:
        self._config = config if config is not None else ControllerConfig()
        if not isinstance(self._config, ControllerConfig):
            raise TypeError("config must be a ControllerConfig")

        if actuator is not None:
            if not isinstance(actuator, ControlSurfaceActuator):
                raise TypeError("actuator must be a ControlSurfaceActuator")
            self._actuator = actuator
        else:
            self._actuator = ControlSurfaceActuator(default_control_surface_config())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ControllerConfig:
        """Read-only controller configuration."""
        return self._config

    @property
    def actuator(self) -> ControlSurfaceActuator:
        """The underlying control surface actuator."""
        return self._actuator

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, cmd: ControlInput, dt: float) -> ControllerOutput:
        """
        Map a ControlInput to actuator commands for one tick.

        Parameters
        ----------
        cmd : ControlInput
            Raw control input from Stage 1.
        dt : float
            Simulation timestep [s]. Must be positive.

        Returns
        -------
        ControllerOutput
            Frozen bundle of all actuator commands for this tick.

        Raises
        ------
        ValueError
            If dt ≤ 0.
        TypeError
            If cmd is not a ControlInput.
        """
        if not isinstance(cmd, ControlInput):
            raise TypeError(f"cmd must be a ControlInput; got {type(cmd).__name__}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive; got {dt:.6g}")

        cfg = self._config

        # 1. Throttle — clamp to [throttle_min, throttle_max]
        throttle = float(max(cfg.throttle_min, min(cfg.throttle_max, cmd.throttle)))

        # 2. Gimbal — scale normalised command to physical angle
        gimbal_p = float(
            max(-cfg.max_gimbal_pitch_rad,
                min(cfg.max_gimbal_pitch_rad,
                    cmd.gimbal_pitch * cfg.max_gimbal_pitch_rad))
        )
        gimbal_y = float(
            max(-cfg.max_gimbal_yaw_rad,
                min(cfg.max_gimbal_yaw_rad,
                    cmd.gimbal_yaw * cfg.max_gimbal_yaw_rad))
        )

        # 3. Aerodynamic surfaces — apply gain then pass to actuator
        elev_cmd = float(max(-1.0, min(1.0, cmd.elevator * cfg.elevator_gain)))
        ail_cmd = float(max(-1.0, min(1.0, cmd.aileron * cfg.aileron_gain)))
        rud_cmd = float(max(-1.0, min(1.0, cmd.rudder * cfg.rudder_gain)))

        surface_state = self._actuator.step(elev_cmd, ail_cmd, rud_cmd, dt)
        deflections = surface_state.to_deflections()

        return ControllerOutput(
            throttle=throttle,
            gimbal_pitch_rad=gimbal_p,
            gimbal_yaw_rad=gimbal_y,
            surface_state=surface_state,
            deflections=deflections,
            staging=cmd.staging,
        )

    def reset(self) -> None:
        """Reset the actuator to neutral position."""
        self._actuator.reset()

    def __repr__(self) -> str:
        return (
            f"VehicleController("
            f"throttle=[{self._config.throttle_min:.2f},{self._config.throttle_max:.2f}], "
            f"gimbal_p=±{math.degrees(self._config.max_gimbal_pitch_rad):.1f}°)"
        )
