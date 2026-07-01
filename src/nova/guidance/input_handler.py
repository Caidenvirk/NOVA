"""
nova.guidance.input_handler
============================
Hardware interface translation layer for Project NOVA.

Architectural role
------------------
Phase 11 — Guidance Subsystem.
Pipeline stage: Stage 1 (Input Handler). Translates raw hardware interface
signals (keyboard events, joystick axes, programmatic commands) into a
standardised ControlInput packet consumed by Stage 2 (Vehicle Controller).

I/O contract
------------
Input  : Raw hardware signals — axis values ∈ [-1, 1], button booleans,
         or programmatic AxisCommand / ButtonCommand structures
Output : ControlInput frozen dataclass — all fields validated and clamped

Design
------
The input handler is intentionally stateless with respect to physics: it
transforms raw input into a ControlInput every tick. It imposes no rate
limiting (that is the actuator's job in control_surfaces.py) and performs
no aerodynamic computations.

Three input sources are supported:
  1. Direct construction — caller builds ControlInput directly (headless /
     scripted flight).
  2. AxisMap — maps named axes to ControlInput fields via a configurable
     dead-zone and gain, returning a ControlInput from a dict of raw values.
  3. InputHandler class — stateful handler accumulating axis and button
     state, producing a ControlInput each tick via .poll().

Axis convention (all normalised to [-1, 1] before passing here):
  throttle      +1 = full throttle,   0 = idle,   not signed (clamped [0,1])
  gimbal_pitch  +1 = pitch-up command
  gimbal_yaw    +1 = yaw-left command
  elevator      +1 = trailing-edge-up (pitch-up)
  aileron       +1 = right-aileron-down (roll-right)
  rudder        +1 = trailing-edge-left (yaw-left)
  staging       True  = staging event requested this tick

No physics calculation may appear in this module.

References
----------
- NOVA Engineering Handoff, §7 Stage 1, §12 Phase 11
- Stevens & Lewis, "Aircraft Control and Simulation", §2.6
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from nova.core.pipeline import ControlInput

# ---------------------------------------------------------------------------
# Axis names — canonical string keys
# ---------------------------------------------------------------------------

AXIS_THROTTLE: str = "throttle"
AXIS_GIMBAL_PITCH: str = "gimbal_pitch"
AXIS_GIMBAL_YAW: str = "gimbal_yaw"
AXIS_ELEVATOR: str = "elevator"
AXIS_AILERON: str = "aileron"
AXIS_RUDDER: str = "rudder"
BUTTON_STAGING: str = "staging"

_ALL_AXES = (
    AXIS_THROTTLE,
    AXIS_GIMBAL_PITCH,
    AXIS_GIMBAL_YAW,
    AXIS_ELEVATOR,
    AXIS_AILERON,
    AXIS_RUDDER,
)

# ---------------------------------------------------------------------------
# AxisMapConfig — dead-zone and gain per axis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisConfig:
    """
    Configuration for a single input axis.

    Attributes
    ----------
    dead_zone : float
        Raw value magnitude below which output is forced to 0.0.
        Must be in [0, 1). Default 0.05.
    gain : float
        Scaling factor applied after dead-zone removal. Default 1.0.
        Output is clamped to [-1, 1] after gain (or [0, 1] for throttle).
    invert : bool
        If True, the raw value is negated before dead-zone/gain. Default False.
    """

    dead_zone: float = 0.05
    gain: float = 1.0
    invert: bool = False

    def __post_init__(self) -> None:
        dz = float(self.dead_zone)
        if not (0.0 <= dz < 1.0):
            raise ValueError(
                f"dead_zone must be in [0, 1); got {dz:.6g}"
            )
        object.__setattr__(self, "dead_zone", dz)

        g = float(self.gain)
        if g <= 0.0:
            raise ValueError(f"gain must be positive; got {g:.6g}")
        object.__setattr__(self, "gain", g)


@dataclass(frozen=True)
class AxisMapConfig:
    """
    Complete axis mapping configuration for all ControlInput fields.

    Attributes
    ----------
    throttle, gimbal_pitch, gimbal_yaw, elevator, aileron, rudder : AxisConfig
        Per-axis configuration. Defaults to AxisConfig() (5% dead-zone, gain 1).
    """

    throttle: AxisConfig = field(default_factory=AxisConfig)
    gimbal_pitch: AxisConfig = field(default_factory=AxisConfig)
    gimbal_yaw: AxisConfig = field(default_factory=AxisConfig)
    elevator: AxisConfig = field(default_factory=AxisConfig)
    aileron: AxisConfig = field(default_factory=AxisConfig)
    rudder: AxisConfig = field(default_factory=AxisConfig)


# ---------------------------------------------------------------------------
# Pure axis-processing function
# ---------------------------------------------------------------------------

def _process_axis(raw: float, cfg: AxisConfig, clamp_positive: bool = False) -> float:
    """
    Apply invert → dead-zone → gain → clamp to a raw axis value.

    Parameters
    ----------
    raw : float
        Raw input in [-1, 1].
    cfg : AxisConfig
        Axis processing configuration.
    clamp_positive : bool
        If True, output is clamped to [0, 1] (for throttle). Default False.

    Returns
    -------
    float
        Processed value in [-1, 1] or [0, 1].
    """
    v = float(raw)
    if cfg.invert:
        v = -v

    # Dead-zone: map [-1,-dz] ∪ [dz,1] → [-1,1]
    dz = cfg.dead_zone
    if abs(v) < dz:
        v = 0.0
    elif v > 0.0:
        v = (v - dz) / (1.0 - dz)
    else:
        v = (v + dz) / (1.0 - dz)

    v *= cfg.gain

    if clamp_positive:
        return float(max(0.0, min(1.0, v)))
    return float(max(-1.0, min(1.0, v)))


# ---------------------------------------------------------------------------
# Raw-dict → ControlInput conversion
# ---------------------------------------------------------------------------

def axes_to_control_input(
    axes: Dict[str, float],
    cfg: Optional[AxisMapConfig] = None,
    staging: bool = False,
) -> ControlInput:
    """
    Convert a dictionary of raw axis values to a ControlInput.

    Missing axes default to 0.0. All values are dead-zone filtered,
    gain-scaled, and clamped per the AxisMapConfig.

    Parameters
    ----------
    axes : dict[str, float]
        Raw axis values keyed by axis name constant (e.g. AXIS_THROTTLE).
        Values should be in [-1, 1]; clamped internally.
    cfg : AxisMapConfig | None
        Axis processing configuration. Defaults to AxisMapConfig() (5% dz).
    staging : bool
        Staging button state this tick. Default False.

    Returns
    -------
    ControlInput
        Validated, clamped control packet.
    """
    if cfg is None:
        cfg = AxisMapConfig()

    throttle = _process_axis(
        axes.get(AXIS_THROTTLE, 0.0), cfg.throttle, clamp_positive=True
    )
    gimbal_pitch = _process_axis(axes.get(AXIS_GIMBAL_PITCH, 0.0), cfg.gimbal_pitch)
    gimbal_yaw = _process_axis(axes.get(AXIS_GIMBAL_YAW, 0.0), cfg.gimbal_yaw)
    elevator = _process_axis(axes.get(AXIS_ELEVATOR, 0.0), cfg.elevator)
    aileron = _process_axis(axes.get(AXIS_AILERON, 0.0), cfg.aileron)
    rudder = _process_axis(axes.get(AXIS_RUDDER, 0.0), cfg.rudder)

    return ControlInput(
        throttle=throttle,
        gimbal_pitch=gimbal_pitch,
        gimbal_yaw=gimbal_yaw,
        elevator=elevator,
        aileron=aileron,
        rudder=rudder,
        staging=bool(staging),
    )


# ---------------------------------------------------------------------------
# InputHandler — stateful tick-by-tick input accumulator
# ---------------------------------------------------------------------------

class InputHandler:
    """
    Stateful hardware input handler producing a ControlInput each tick.

    Maintains the current axis and button state. Callers update individual
    axes/buttons via set_axis() / set_button() as hardware events arrive,
    then call poll() once per simulation tick to obtain the ControlInput.

    Parameters
    ----------
    cfg : AxisMapConfig | None
        Axis processing configuration. Defaults to AxisMapConfig().
    """

    def __init__(self, cfg: Optional[AxisMapConfig] = None) -> None:
        self._cfg = cfg if cfg is not None else AxisMapConfig()
        self._axes: Dict[str, float] = {ax: 0.0 for ax in _ALL_AXES}
        self._staging: bool = False
        self._staging_edge: bool = False  # True only on the tick staging first fires

    # ------------------------------------------------------------------
    # State setters
    # ------------------------------------------------------------------

    def set_axis(self, axis_name: str, value: float) -> None:
        """
        Set a raw axis value.

        Parameters
        ----------
        axis_name : str
            One of the AXIS_* constants.
        value : float
            Raw value in [-1, 1]. Clamped internally to [-1, 1].

        Raises
        ------
        KeyError
            If axis_name is not a recognised axis.
        """
        if axis_name not in self._axes:
            raise KeyError(
                f"Unknown axis '{axis_name}'. "
                f"Valid axes: {sorted(self._axes.keys())}"
            )
        self._axes[axis_name] = float(max(-1.0, min(1.0, value)))

    def set_button(self, button_name: str, state: bool) -> None:
        """
        Set a button state.

        Currently only BUTTON_STAGING is supported.

        Parameters
        ----------
        button_name : str
            One of the BUTTON_* constants.
        state : bool
            True = pressed, False = released.

        Raises
        ------
        KeyError
            If button_name is not recognised.
        """
        if button_name == BUTTON_STAGING:
            self._staging = bool(state)
        else:
            raise KeyError(
                f"Unknown button '{button_name}'. Valid buttons: ['{BUTTON_STAGING}']"
            )

    def set_throttle(self, value: float) -> None:
        """Convenience: set throttle axis value [0, 1] (clamped)."""
        self._axes[AXIS_THROTTLE] = float(max(0.0, min(1.0, value)))

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    def poll(self) -> ControlInput:
        """
        Produce a ControlInput from the current axis and button state.

        The staging flag is True only on the single tick where the staging
        button is pressed; it auto-clears after poll() is called.

        Returns
        -------
        ControlInput
        """
        staging_this_tick = self._staging
        # Edge-trigger: staging fires once per press
        if staging_this_tick:
            self._staging = False

        return axes_to_control_input(
            axes=dict(self._axes),
            cfg=self._cfg,
            staging=staging_this_tick,
        )

    def reset(self) -> None:
        """Return all axes to zero and clear staging."""
        for ax in self._axes:
            self._axes[ax] = 0.0
        self._staging = False

    def __repr__(self) -> str:
        thr = self._axes.get(AXIS_THROTTLE, 0.0)
        return (
            f"InputHandler(throttle={thr:.2f}, staging={self._staging})"
        )


# ---------------------------------------------------------------------------
# NullInputHandler — always returns neutral ControlInput
# ---------------------------------------------------------------------------

class NullInputHandler:
    """
    Headless input source that always returns a neutral ControlInput.

    Used by the pipeline when no hardware interface is present.

    Parameters
    ----------
    throttle : float
        Fixed throttle setting [0, 1]. Default 0.0.
    """

    def __init__(self, throttle: float = 0.0) -> None:
        t = float(throttle)
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"throttle must be in [0, 1]; got {t:.6g}")
        self._throttle = t

    def poll(self) -> ControlInput:
        """Return a neutral ControlInput with the configured throttle."""
        return ControlInput(
            throttle=self._throttle,
            gimbal_pitch=0.0,
            gimbal_yaw=0.0,
            elevator=0.0,
            aileron=0.0,
            rudder=0.0,
            staging=False,
        )

    def __repr__(self) -> str:
        return f"NullInputHandler(throttle={self._throttle:.2f})"
