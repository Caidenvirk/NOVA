"""
nova.guidance.sas
==================
Stability Augmentation System (SAS) for Project NOVA.

Architectural role
------------------
Phase 11 — Guidance Subsystem.
Pipeline stage: Stage 1 augmentation (pre-controller). The SAS reads the
vehicle's angular rates (p, q, r) from the current VehicleState and
computes rate-damping corrections that are blended with the pilot's control
input before the combined signal reaches the Vehicle Controller (Stage 2).

I/O contract
------------
Input  : VehicleState (omega_body = [p, q, r] [rad s⁻¹]),
         ControlInput (pilot commands, normalised [-1, 1]),
         SASConfig, dt [s]
Output : ControlInput — pilot command augmented by SAS damping corrections

Physical basis
--------------
The SAS implements proportional-integral-derivative (PID) rate feedback to
damp angular oscillations:

    error(t) = ω_target − ω_measured   [rad s⁻¹]
    P term : K_p · error(t)
    I term : K_i · ∫ error dt          (with anti-windup clamping)
    D term : K_d · d(error)/dt         (finite-difference approximation)

    correction = -(P + I + D)           [normalised, clamped to [-1, 1]]

The target angular rate for each axis is zero when the pilot is not
commanding a rotation (rate-command zero-rate-hold). When the pilot is
commanding, the SAS authority is blended with the pilot authority so that
full stick deflection overrides the SAS correction.

Blending rule:
    augmented_cmd = pilot_cmd + sas_correction · (1 − |pilot_cmd|)

The (1 − |pilot_cmd|) factor ensures the SAS has full authority at neutral
stick and zero authority at full deflection, providing smooth washout.

Authority limit:
    The total augmented command is clamped to [-1, 1] after blending.

Axes:
    Roll  (p) → aileron correction
    Pitch (q) → elevator correction
    Yaw   (r) → rudder correction

References
----------
- Stevens & Lewis, "Aircraft Control and Simulation", §6.3 (rate damper)
- McLean, "Automatic Flight Control Systems", §3.2 (PID rate feedback)
- NOVA Engineering Handoff, §12 Phase 11
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from nova.core.pipeline import ControlInput
from nova.core.state_vector import VehicleState

# ---------------------------------------------------------------------------
# SASAxisConfig — per-axis PID gains and limits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SASAxisConfig:
    """
    PID gains and authority limits for one SAS axis.

    Attributes
    ----------
    kp : float
        Proportional gain [normalised / (rad s⁻¹)]. Must be ≥ 0.
    ki : float
        Integral gain [normalised / rad]. Must be ≥ 0.
    kd : float
        Derivative gain [normalised / (rad s⁻²)]. Must be ≥ 0.
    max_authority : float
        Maximum SAS correction magnitude [0, 1]. Clamps the SAS output
        before blending with pilot command. Default 0.3.
    integrator_limit : float
        Anti-windup clamp for the integrator state [rad]. Must be > 0.
        Default 1.0 rad·s (≈ 57°·s accumulated error).
    """

    kp: float = 0.1
    ki: float = 0.01
    kd: float = 0.005
    max_authority: float = 0.3
    integrator_limit: float = 1.0

    def __post_init__(self) -> None:
        for attr in ("kp", "ki", "kd"):
            val = float(getattr(self, attr))
            if val < 0.0:
                raise ValueError(f"{attr} must be ≥ 0; got {val:.6g}")
            object.__setattr__(self, attr, val)

        auth = float(self.max_authority)
        if not (0.0 <= auth <= 1.0):
            raise ValueError(
                f"max_authority must be in [0, 1]; got {auth:.6g}"
            )
        object.__setattr__(self, "max_authority", auth)

        ilim = float(self.integrator_limit)
        if ilim <= 0.0:
            raise ValueError(
                f"integrator_limit must be positive; got {ilim:.6g}"
            )
        object.__setattr__(self, "integrator_limit", ilim)


@dataclass(frozen=True)
class SASConfig:
    """
    Full SAS configuration for all three axes.

    Attributes
    ----------
    roll : SASAxisConfig
        Roll-rate (p) damper configuration.
    pitch : SASAxisConfig
        Pitch-rate (q) damper configuration.
    yaw : SASAxisConfig
        Yaw-rate (r) damper configuration.
    enabled : bool
        Global SAS enable switch. If False, SAS returns pilot command
        unchanged. Default True.
    """

    roll: SASAxisConfig = None    # type: ignore[assignment]
    pitch: SASAxisConfig = None   # type: ignore[assignment]
    yaw: SASAxisConfig = None     # type: ignore[assignment]
    enabled: bool = True

    def __post_init__(self) -> None:
        roll = self.roll if self.roll is not None else SASAxisConfig()
        pitch = self.pitch if self.pitch is not None else SASAxisConfig()
        yaw = self.yaw if self.yaw is not None else SASAxisConfig()

        if not isinstance(roll, SASAxisConfig):
            raise TypeError("roll must be a SASAxisConfig")
        if not isinstance(pitch, SASAxisConfig):
            raise TypeError("pitch must be a SASAxisConfig")
        if not isinstance(yaw, SASAxisConfig):
            raise TypeError("yaw must be a SASAxisConfig")

        object.__setattr__(self, "roll", roll)
        object.__setattr__(self, "pitch", pitch)
        object.__setattr__(self, "yaw", yaw)
        object.__setattr__(self, "enabled", bool(self.enabled))


# ---------------------------------------------------------------------------
# SASAxisState — per-axis PID integrator state
# ---------------------------------------------------------------------------

@dataclass
class _SASAxisState:
    """Mutable PID state for one SAS axis."""
    integrator: float = 0.0
    prev_error: float = 0.0
    initialized: bool = False


# ---------------------------------------------------------------------------
# SASDiagnostic — per-tick SAS output record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SASDiagnostic:
    """
    Diagnostic record of SAS activity for one tick.

    Attributes
    ----------
    roll_correction : float
        SAS aileron correction applied [-1, 1].
    pitch_correction : float
        SAS elevator correction applied [-1, 1].
    yaw_correction : float
        SAS rudder correction applied [-1, 1].
    roll_rate_error : float
        Roll rate error p_error [rad s⁻¹] (target − measured).
    pitch_rate_error : float
        Pitch rate error q_error [rad s⁻¹].
    yaw_rate_error : float
        Yaw rate error r_error [rad s⁻¹].
    sas_active : bool
        True if SAS is enabled and corrections were non-zero.
    """

    roll_correction: float
    pitch_correction: float
    yaw_correction: float
    roll_rate_error: float
    pitch_rate_error: float
    yaw_rate_error: float
    sas_active: bool

    def __post_init__(self) -> None:
        for attr in ("roll_correction", "pitch_correction", "yaw_correction",
                     "roll_rate_error", "pitch_rate_error", "yaw_rate_error"):
            object.__setattr__(self, attr, float(getattr(self, attr)))
        object.__setattr__(self, "sas_active", bool(self.sas_active))

    def __repr__(self) -> str:
        return (
            f"SASDiagnostic("
            f"Δail={self.roll_correction:+.4f}, "
            f"Δelv={self.pitch_correction:+.4f}, "
            f"Δrud={self.yaw_correction:+.4f}, "
            f"active={self.sas_active})"
        )


# ---------------------------------------------------------------------------
# Pure PID correction function
# ---------------------------------------------------------------------------

def _pid_correction(
    measured_rate: float,
    target_rate: float,
    axis_state: _SASAxisState,
    cfg: SASAxisConfig,
    dt: float,
) -> tuple[float, float]:
    """
    Compute a single-axis PID rate-damping correction.

    Parameters
    ----------
    measured_rate : float
        Current angular rate [rad s⁻¹].
    target_rate : float
        Desired angular rate [rad s⁻¹]. Typically 0.
    axis_state : _SASAxisState
        Mutable PID integrator/derivative state (modified in-place).
    cfg : SASAxisConfig
        PID gains and limits.
    dt : float
        Timestep [s]. Must be positive.

    Returns
    -------
    correction : float
        Raw PID output in normalised units [-max_authority, +max_authority].
    error : float
        Current rate error [rad s⁻¹].
    """
    error = target_rate - measured_rate

    # Proportional
    p_term = cfg.kp * error

    # Integral with anti-windup clamp
    axis_state.integrator += error * dt
    axis_state.integrator = float(
        max(-cfg.integrator_limit, min(cfg.integrator_limit, axis_state.integrator))
    )
    i_term = cfg.ki * axis_state.integrator

    # Derivative (backward difference; skip first tick to avoid spike)
    if axis_state.initialized:
        d_error = (error - axis_state.prev_error) / dt
    else:
        d_error = 0.0
        axis_state.initialized = True

    d_term = cfg.kd * d_error
    axis_state.prev_error = error

    # Sum and clamp to authority limit
    raw = p_term + i_term + d_term
    clamped = float(max(-cfg.max_authority, min(cfg.max_authority, raw)))
    return clamped, error


# ---------------------------------------------------------------------------
# StabilityAugmentationSystem
# ---------------------------------------------------------------------------

class StabilityAugmentationSystem:
    """
    Proportional-integral-derivative angular-rate damper for all three axes.

    Reads omega_body = [p, q, r] from the VehicleState, computes corrections
    for roll (aileron), pitch (elevator), and yaw (rudder), then blends them
    with the pilot's ControlInput to produce an augmented ControlInput.

    Parameters
    ----------
    config : SASConfig | None
        SAS configuration. Defaults to SASConfig() with default gains.
    """

    def __init__(self, config: Optional[SASConfig] = None) -> None:
        if config is None:
            config = SASConfig()
        if not isinstance(config, SASConfig):
            raise TypeError("config must be a SASConfig")
        self._config = config

        # Per-axis mutable PID states
        self._roll_state = _SASAxisState()
        self._pitch_state = _SASAxisState()
        self._yaw_state = _SASAxisState()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SASConfig:
        """Read-only SAS configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """True when the SAS is globally enabled."""
        return self._config.enabled

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        state: VehicleState,
        pilot_cmd: ControlInput,
        dt: float,
        target_rates: Optional[np.ndarray] = None,
    ) -> tuple[ControlInput, SASDiagnostic]:
        """
        Compute SAS corrections and return augmented ControlInput.

        Parameters
        ----------
        state : VehicleState
            Current vehicle state. Uses omega_body = [p, q, r].
        pilot_cmd : ControlInput
            Raw pilot commands from Stage 1.
        dt : float
            Timestep [s]. Must be positive.
        target_rates : ndarray, shape (3,) | None
            Target angular rates [p_tgt, q_tgt, r_tgt] [rad s⁻¹].
            If None, defaults to [0, 0, 0] (rate-hold at zero).

        Returns
        -------
        augmented_cmd : ControlInput
            Pilot command augmented with SAS corrections.
        diagnostic : SASDiagnostic
            Per-tick SAS activity record.

        Raises
        ------
        TypeError
            If state is not a VehicleState or pilot_cmd is not a ControlInput.
        ValueError
            If dt ≤ 0.
        """
        if not isinstance(state, VehicleState):
            raise TypeError(
                f"state must be a VehicleState; got {type(state).__name__}"
            )
        if not isinstance(pilot_cmd, ControlInput):
            raise TypeError(
                f"pilot_cmd must be a ControlInput; got {type(pilot_cmd).__name__}"
            )
        if dt <= 0.0:
            raise ValueError(f"dt must be positive; got {dt:.6g}")

        # If SAS disabled: return pilot command unchanged
        if not self._config.enabled:
            diag = SASDiagnostic(
                roll_correction=0.0, pitch_correction=0.0, yaw_correction=0.0,
                roll_rate_error=0.0, pitch_rate_error=0.0, yaw_rate_error=0.0,
                sas_active=False,
            )
            return pilot_cmd, diag

        # Parse measured rates
        omega = state.omega_body.astype(np.float64)
        p_meas, q_meas, r_meas = float(omega[0]), float(omega[1]), float(omega[2])

        # Target rates
        if target_rates is None:
            p_tgt = q_tgt = r_tgt = 0.0
        else:
            tgt = np.asarray(target_rates, dtype=np.float64)
            if tgt.shape != (3,):
                raise ValueError(
                    f"target_rates must have shape (3,); got {tgt.shape}"
                )
            p_tgt, q_tgt, r_tgt = float(tgt[0]), float(tgt[1]), float(tgt[2])

        # Compute PID corrections (note: negate sign — correction opposes error)
        roll_corr, roll_err = _pid_correction(
            p_meas, p_tgt, self._roll_state, self._config.roll, dt
        )
        pitch_corr, pitch_err = _pid_correction(
            q_meas, q_tgt, self._pitch_state, self._config.pitch, dt
        )
        yaw_corr, yaw_err = _pid_correction(
            r_meas, r_tgt, self._yaw_state, self._config.yaw, dt
        )

        # Blend with pilot commands using washout: α = (1 - |pilot|)
        def _blend(pilot: float, correction: float) -> float:
            washout = 1.0 - abs(pilot)
            blended = pilot + correction * washout
            return float(max(-1.0, min(1.0, blended)))

        aug_elevator = _blend(pilot_cmd.elevator, pitch_corr)
        aug_aileron = _blend(pilot_cmd.aileron, roll_corr)
        aug_rudder = _blend(pilot_cmd.rudder, yaw_corr)

        augmented = ControlInput(
            throttle=pilot_cmd.throttle,
            gimbal_pitch=pilot_cmd.gimbal_pitch,
            gimbal_yaw=pilot_cmd.gimbal_yaw,
            elevator=aug_elevator,
            aileron=aug_aileron,
            rudder=aug_rudder,
            staging=pilot_cmd.staging,
        )

        any_active = (
            abs(roll_corr) > 1e-10
            or abs(pitch_corr) > 1e-10
            or abs(yaw_corr) > 1e-10
        )
        diag = SASDiagnostic(
            roll_correction=roll_corr,
            pitch_correction=pitch_corr,
            yaw_correction=yaw_corr,
            roll_rate_error=roll_err,
            pitch_rate_error=pitch_err,
            yaw_rate_error=yaw_err,
            sas_active=any_active,
        )
        return augmented, diag

    def reset(self) -> None:
        """Reset all PID integrators and derivative states."""
        self._roll_state = _SASAxisState()
        self._pitch_state = _SASAxisState()
        self._yaw_state = _SASAxisState()

    def __repr__(self) -> str:
        return (
            f"StabilityAugmentationSystem("
            f"enabled={self._config.enabled}, "
            f"kp_roll={self._config.roll.kp:.4f})"
        )
