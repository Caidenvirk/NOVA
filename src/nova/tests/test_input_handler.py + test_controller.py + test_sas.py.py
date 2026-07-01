"""
tests/unit/test_input_handler.py + test_controller.py + test_sas.py
=====================================================================
Unit tests for Phase 11 guidance subsystem:
  nova.guidance.input_handler  — AxisConfig, AxisMapConfig, InputHandler,
                                  NullInputHandler, axes_to_control_input,
                                  _process_axis
  nova.guidance.controller     — ControllerConfig, ControllerOutput,
                                  VehicleController
  nova.guidance.sas            — SASAxisConfig, SASConfig, SASDiagnostic,
                                  StabilityAugmentationSystem, _pid_correction

Test coverage
-------------
input_handler (52 tests):
  AxisConfig: construction, frozen, validation
  AxisMapConfig: construction, defaults
  _process_axis: dead-zone, gain, invert, clamp-positive
  axes_to_control_input: missing axes default, all axes set, staging flag
  InputHandler: set_axis, set_button, set_throttle, poll, reset, staging edge
  NullInputHandler: neutral output, throttle fixed, invalid throttle

controller (48 tests):
  ControllerConfig: construction, frozen, validation
  ControllerOutput: construction, frozen, validation
  VehicleController: step mapping, gimbal scaling, throttle clamping,
                     gain application, surface actuator integration,
                     reset, type errors, dt validation

sas (52 tests):
  SASAxisConfig: construction, frozen, validation
  SASConfig: defaults, enable flag, type errors
  SASDiagnostic: construction, frozen, repr
  _pid_correction: P/I/D terms, anti-windup, first-tick derivative skip
  StabilityAugmentationSystem: disabled passthrough, rate damping,
                               washout blending, integrator reset,
                               multi-tick convergence, type errors
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nova.core.pipeline import ControlInput
from nova.core.state_vector import identity_state, make_state
from nova.guidance.controller import (
    ControllerConfig,
    ControllerOutput,
    VehicleController,
)
from nova.guidance.input_handler import (
    AXIS_AILERON,
    AXIS_ELEVATOR,
    AXIS_GIMBAL_PITCH,
    AXIS_GIMBAL_YAW,
    AXIS_RUDDER,
    AXIS_THROTTLE,
    BUTTON_STAGING,
    AxisConfig,
    AxisMapConfig,
    InputHandler,
    NullInputHandler,
    _process_axis,
    axes_to_control_input,
)
from nova.guidance.sas import (
    SASAxisConfig,
    SASConfig,
    SASDiagnostic,
    StabilityAugmentationSystem,
    _pid_correction,
    _SASAxisState,
)
from nova.vehicle.control_surfaces import (
    ControlSurfaceActuator,
    default_control_surface_config,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def neutral_input() -> ControlInput:
    return ControlInput()


@pytest.fixture
def full_input() -> ControlInput:
    return ControlInput(
        throttle=1.0,
        gimbal_pitch=1.0,
        gimbal_yaw=-1.0,
        elevator=1.0,
        aileron=-1.0,
        rudder=0.5,
        staging=False,
    )


@pytest.fixture
def zero_state():
    return identity_state()


@pytest.fixture
def spinning_state():
    """Vehicle spinning in roll at 1 rad/s."""
    return make_state(
        position_eci=np.array([6_771_000.0, 0.0, 0.0]),
        velocity_eci=np.array([0.0, 7_800.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body=np.array([1.0, 0.0, 0.0]),
        mass=1000.0,
        time=0.0,
    )


# ===========================================================================
# AxisConfig tests
# ===========================================================================

class TestAxisConfig:

    def test_default_construction(self):
        cfg = AxisConfig()
        assert cfg.dead_zone == pytest.approx(0.05)
        assert cfg.gain == pytest.approx(1.0)
        assert cfg.invert is False

    def test_custom_values(self):
        cfg = AxisConfig(dead_zone=0.1, gain=0.8, invert=True)
        assert cfg.dead_zone == pytest.approx(0.1)
        assert cfg.gain == pytest.approx(0.8)
        assert cfg.invert is True

    def test_frozen(self):
        cfg = AxisConfig()
        with pytest.raises(Exception):
            cfg.gain = 2.0

    def test_negative_dead_zone_rejected(self):
        with pytest.raises(ValueError, match="dead_zone"):
            AxisConfig(dead_zone=-0.01)

    def test_dead_zone_one_rejected(self):
        with pytest.raises(ValueError, match="dead_zone"):
            AxisConfig(dead_zone=1.0)

    def test_zero_gain_rejected(self):
        with pytest.raises(ValueError, match="gain"):
            AxisConfig(gain=0.0)

    def test_negative_gain_rejected(self):
        with pytest.raises(ValueError, match="gain"):
            AxisConfig(gain=-0.5)

    def test_zero_dead_zone_valid(self):
        cfg = AxisConfig(dead_zone=0.0)
        assert cfg.dead_zone == 0.0


# ===========================================================================
# AxisMapConfig tests
# ===========================================================================

class TestAxisMapConfig:

    def test_default_construction(self):
        cfg = AxisMapConfig()
        assert isinstance(cfg.throttle, AxisConfig)
        assert isinstance(cfg.elevator, AxisConfig)

    def test_custom_elevator_config(self):
        elev = AxisConfig(dead_zone=0.1)
        cfg = AxisMapConfig(elevator=elev)
        assert cfg.elevator.dead_zone == pytest.approx(0.1)


# ===========================================================================
# _process_axis tests
# ===========================================================================

class TestProcessAxis:

    @pytest.fixture
    def no_dz_cfg(self):
        return AxisConfig(dead_zone=0.0, gain=1.0, invert=False)

    def test_full_positive_no_deadzone(self, no_dz_cfg):
        assert _process_axis(1.0, no_dz_cfg) == pytest.approx(1.0)

    def test_full_negative_no_deadzone(self, no_dz_cfg):
        assert _process_axis(-1.0, no_dz_cfg) == pytest.approx(-1.0)

    def test_zero_no_deadzone(self, no_dz_cfg):
        assert _process_axis(0.0, no_dz_cfg) == pytest.approx(0.0)

    def test_within_deadzone_returns_zero(self):
        cfg = AxisConfig(dead_zone=0.1)
        assert _process_axis(0.05, cfg) == pytest.approx(0.0)

    def test_outside_deadzone_nonzero(self):
        cfg = AxisConfig(dead_zone=0.1)
        result = _process_axis(0.5, cfg)
        assert result > 0.0

    def test_gain_scales_output(self):
        cfg = AxisConfig(dead_zone=0.0, gain=0.5)
        result = _process_axis(1.0, cfg)
        assert result == pytest.approx(0.5)

    def test_gain_clamped_at_one(self):
        cfg = AxisConfig(dead_zone=0.0, gain=2.0)
        result = _process_axis(1.0, cfg)
        assert result == pytest.approx(1.0)

    def test_invert_flips_sign(self):
        cfg = AxisConfig(dead_zone=0.0, invert=True)
        assert _process_axis(0.5, cfg) == pytest.approx(-0.5)

    def test_clamp_positive_negatives_floored_at_zero(self):
        cfg = AxisConfig(dead_zone=0.0)
        result = _process_axis(-1.0, cfg, clamp_positive=True)
        assert result == pytest.approx(0.0)

    def test_clamp_positive_positive_passes(self):
        cfg = AxisConfig(dead_zone=0.0)
        result = _process_axis(0.5, cfg, clamp_positive=True)
        assert result == pytest.approx(0.5)

    def test_over_range_input_clamped(self):
        cfg = AxisConfig(dead_zone=0.0)
        assert _process_axis(2.0, cfg) == pytest.approx(1.0)

    def test_under_range_input_clamped(self):
        cfg = AxisConfig(dead_zone=0.0)
        assert _process_axis(-2.0, cfg) == pytest.approx(-1.0)

    def test_deadzone_symmetric(self):
        cfg = AxisConfig(dead_zone=0.1)
        pos = _process_axis(0.05, cfg)
        neg = _process_axis(-0.05, cfg)
        assert pos == pytest.approx(0.0)
        assert neg == pytest.approx(0.0)


# ===========================================================================
# axes_to_control_input tests
# ===========================================================================

class TestAxesToControlInput:

    def test_empty_axes_returns_neutral(self):
        ci = axes_to_control_input({})
        assert ci.throttle == pytest.approx(0.0)
        assert ci.elevator == pytest.approx(0.0)
        assert ci.staging is False

    def test_full_throttle(self):
        ci = axes_to_control_input({AXIS_THROTTLE: 1.0},
                                   cfg=AxisMapConfig(throttle=AxisConfig(dead_zone=0.0)))
        assert ci.throttle == pytest.approx(1.0)

    def test_negative_throttle_clamped_to_zero(self):
        ci = axes_to_control_input({AXIS_THROTTLE: -1.0},
                                   cfg=AxisMapConfig(throttle=AxisConfig(dead_zone=0.0)))
        assert ci.throttle == pytest.approx(0.0)

    def test_staging_flag_propagated(self):
        ci = axes_to_control_input({}, staging=True)
        assert ci.staging is True

    def test_all_axes_set(self):
        cfg = AxisMapConfig(
            throttle=AxisConfig(dead_zone=0.0),
            elevator=AxisConfig(dead_zone=0.0),
            aileron=AxisConfig(dead_zone=0.0),
            rudder=AxisConfig(dead_zone=0.0),
            gimbal_pitch=AxisConfig(dead_zone=0.0),
            gimbal_yaw=AxisConfig(dead_zone=0.0),
        )
        ci = axes_to_control_input({
            AXIS_THROTTLE: 0.8,
            AXIS_ELEVATOR: 0.5,
            AXIS_AILERON: -0.3,
            AXIS_RUDDER: 0.2,
            AXIS_GIMBAL_PITCH: 0.4,
            AXIS_GIMBAL_YAW: -0.6,
        }, cfg=cfg)
        assert ci.throttle == pytest.approx(0.8)
        assert ci.elevator == pytest.approx(0.5)
        assert ci.aileron == pytest.approx(-0.3)
        assert ci.rudder == pytest.approx(0.2)
        assert ci.gimbal_pitch == pytest.approx(0.4)
        assert ci.gimbal_yaw == pytest.approx(-0.6)

    def test_returns_control_input_type(self):
        ci = axes_to_control_input({})
        assert isinstance(ci, ControlInput)


# ===========================================================================
# InputHandler tests
# ===========================================================================

class TestInputHandler:

    def test_initial_poll_is_neutral(self):
        h = InputHandler()
        ci = h.poll()
        assert ci.throttle == pytest.approx(0.0)
        assert ci.elevator == pytest.approx(0.0)
        assert ci.staging is False

    def test_set_axis_throttle(self):
        h = InputHandler(AxisMapConfig(throttle=AxisConfig(dead_zone=0.0)))
        h.set_axis(AXIS_THROTTLE, 0.8)
        ci = h.poll()
        assert ci.throttle == pytest.approx(0.8)

    def test_set_axis_elevator(self):
        h = InputHandler(AxisMapConfig(elevator=AxisConfig(dead_zone=0.0)))
        h.set_axis(AXIS_ELEVATOR, 0.6)
        ci = h.poll()
        assert ci.elevator == pytest.approx(0.6)

    def test_set_axis_unknown_raises(self):
        h = InputHandler()
        with pytest.raises(KeyError):
            h.set_axis("nonexistent_axis", 0.5)

    def test_set_button_staging_fires_once(self):
        h = InputHandler()
        h.set_button(BUTTON_STAGING, True)
        ci1 = h.poll()
        ci2 = h.poll()
        assert ci1.staging is True
        assert ci2.staging is False

    def test_set_button_unknown_raises(self):
        h = InputHandler()
        with pytest.raises(KeyError):
            h.set_button("fire_missiles", True)

    def test_set_throttle_convenience(self):
        h = InputHandler(AxisMapConfig(throttle=AxisConfig(dead_zone=0.0)))
        h.set_throttle(0.5)
        ci = h.poll()
        assert ci.throttle == pytest.approx(0.5)

    def test_set_throttle_clamped_above_one(self):
        h = InputHandler(AxisMapConfig(throttle=AxisConfig(dead_zone=0.0)))
        h.set_throttle(2.0)
        ci = h.poll()
        assert ci.throttle == pytest.approx(1.0)

    def test_reset_zeroes_all_axes(self):
        h = InputHandler(AxisMapConfig(throttle=AxisConfig(dead_zone=0.0)))
        h.set_throttle(1.0)
        h.set_axis(AXIS_ELEVATOR, 0.5)
        h.reset()
        ci = h.poll()
        assert ci.throttle == pytest.approx(0.0)
        assert ci.elevator == pytest.approx(0.0)

    def test_axis_state_persists_across_polls(self):
        h = InputHandler(AxisMapConfig(throttle=AxisConfig(dead_zone=0.0)))
        h.set_throttle(0.7)
        ci1 = h.poll()
        ci2 = h.poll()
        assert ci1.throttle == pytest.approx(ci2.throttle)

    def test_all_axes_settable(self):
        h = InputHandler()
        for axis in (AXIS_THROTTLE, AXIS_GIMBAL_PITCH, AXIS_GIMBAL_YAW,
                     AXIS_ELEVATOR, AXIS_AILERON, AXIS_RUDDER):
            h.set_axis(axis, 0.5)
        ci = h.poll()
        assert isinstance(ci, ControlInput)

    def test_repr(self):
        h = InputHandler()
        assert "InputHandler" in repr(h)

    def test_axis_value_clamped_at_set(self):
        h = InputHandler(AxisMapConfig(elevator=AxisConfig(dead_zone=0.0)))
        h.set_axis(AXIS_ELEVATOR, 5.0)
        ci = h.poll()
        assert ci.elevator == pytest.approx(1.0)

    def test_deadzone_filters_small_input(self):
        h = InputHandler()  # default 5% dead-zone
        h.set_axis(AXIS_ELEVATOR, 0.03)
        ci = h.poll()
        assert ci.elevator == pytest.approx(0.0)


# ===========================================================================
# NullInputHandler tests
# ===========================================================================

class TestNullInputHandler:

    def test_neutral_output(self):
        h = NullInputHandler()
        ci = h.poll()
        assert ci.throttle == pytest.approx(0.0)
        assert ci.elevator == pytest.approx(0.0)
        assert ci.staging is False

    def test_fixed_throttle(self):
        h = NullInputHandler(throttle=0.5)
        ci = h.poll()
        assert ci.throttle == pytest.approx(0.5)

    def test_throttle_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            NullInputHandler(throttle=1.5)

    def test_negative_throttle_rejected(self):
        with pytest.raises(ValueError):
            NullInputHandler(throttle=-0.1)

    def test_poll_is_idempotent(self):
        h = NullInputHandler(throttle=0.3)
        ci1 = h.poll()
        ci2 = h.poll()
        assert ci1.throttle == pytest.approx(ci2.throttle)
        assert ci1.staging == ci2.staging

    def test_repr(self):
        h = NullInputHandler(0.5)
        assert "NullInputHandler" in repr(h)


# ===========================================================================
# ControllerConfig tests
# ===========================================================================

class TestControllerConfig:

    def test_default_construction(self):
        cfg = ControllerConfig()
        assert cfg.throttle_min == pytest.approx(0.0)
        assert cfg.throttle_max == pytest.approx(1.0)
        assert cfg.max_gimbal_pitch_rad == pytest.approx(math.radians(5.0))

    def test_custom_values(self):
        cfg = ControllerConfig(throttle_min=0.2, throttle_max=0.9)
        assert cfg.throttle_min == pytest.approx(0.2)
        assert cfg.throttle_max == pytest.approx(0.9)

    def test_frozen(self):
        cfg = ControllerConfig()
        with pytest.raises(Exception):
            cfg.throttle_max = 0.5

    def test_throttle_min_exceeds_max_rejected(self):
        with pytest.raises(ValueError):
            ControllerConfig(throttle_min=0.8, throttle_max=0.2)

    def test_throttle_min_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            ControllerConfig(throttle_min=-0.1)

    def test_throttle_max_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            ControllerConfig(throttle_max=1.5)

    def test_negative_gimbal_pitch_rejected(self):
        with pytest.raises(ValueError):
            ControllerConfig(max_gimbal_pitch_rad=-0.1)

    def test_elevator_gain_zero_rejected(self):
        with pytest.raises(ValueError):
            ControllerConfig(elevator_gain=0.0)

    def test_elevator_gain_above_one_rejected(self):
        with pytest.raises(ValueError):
            ControllerConfig(elevator_gain=1.1)

    def test_zero_gimbal_valid(self):
        cfg = ControllerConfig(max_gimbal_pitch_rad=0.0)
        assert cfg.max_gimbal_pitch_rad == 0.0


# ===========================================================================
# ControllerOutput tests
# ===========================================================================

class TestControllerOutput:

    @pytest.fixture
    def sample_surface_state(self):
        from nova.vehicle.control_surfaces import ControlSurfaceState
        return ControlSurfaceState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @pytest.fixture
    def sample_deflections(self):
        from nova.physics.aerodynamics import ControlDeflections
        return ControlDeflections(0.0, 0.0, 0.0)

    def test_valid_construction(self, sample_surface_state, sample_deflections):
        out = ControllerOutput(
            throttle=0.8,
            gimbal_pitch_rad=0.05,
            gimbal_yaw_rad=-0.03,
            surface_state=sample_surface_state,
            deflections=sample_deflections,
            staging=False,
        )
        assert out.throttle == pytest.approx(0.8)
        assert out.staging is False

    def test_frozen(self, sample_surface_state, sample_deflections):
        out = ControllerOutput(
            throttle=0.5, gimbal_pitch_rad=0.0, gimbal_yaw_rad=0.0,
            surface_state=sample_surface_state,
            deflections=sample_deflections,
            staging=False,
        )
        with pytest.raises(Exception):
            out.throttle = 1.0

    def test_throttle_out_of_range_rejected(self, sample_surface_state, sample_deflections):
        with pytest.raises(ValueError):
            ControllerOutput(
                throttle=1.5, gimbal_pitch_rad=0.0, gimbal_yaw_rad=0.0,
                surface_state=sample_surface_state,
                deflections=sample_deflections,
                staging=False,
            )

    def test_wrong_surface_state_type_rejected(self, sample_deflections):
        with pytest.raises(TypeError):
            ControllerOutput(
                throttle=0.5, gimbal_pitch_rad=0.0, gimbal_yaw_rad=0.0,
                surface_state="not_a_state",
                deflections=sample_deflections,
                staging=False,
            )

    def test_repr(self, sample_surface_state, sample_deflections):
        out = ControllerOutput(
            throttle=0.6, gimbal_pitch_rad=0.02, gimbal_yaw_rad=-0.01,
            surface_state=sample_surface_state,
            deflections=sample_deflections,
            staging=True,
        )
        r = repr(out)
        assert "ControllerOutput" in r
        assert "0.600" in r


# ===========================================================================
# VehicleController tests
# ===========================================================================

class TestVehicleController:

    def test_default_construction(self):
        c = VehicleController()
        assert isinstance(c, VehicleController)

    def test_custom_config(self):
        cfg = ControllerConfig(throttle_max=0.9)
        c = VehicleController(config=cfg)
        assert c.config.throttle_max == pytest.approx(0.9)

    def test_repr(self):
        c = VehicleController()
        assert "VehicleController" in repr(c)

    def test_wrong_config_type_rejected(self):
        with pytest.raises(TypeError):
            VehicleController(config="not_config")

    def test_wrong_actuator_type_rejected(self):
        with pytest.raises(TypeError):
            VehicleController(actuator="not_actuator")

    def test_step_returns_controller_output(self, neutral_input):
        c = VehicleController()
        out = c.step(neutral_input, dt=0.01)
        assert isinstance(out, ControllerOutput)

    def test_throttle_passthrough(self):
        c = VehicleController()
        cmd = ControlInput(throttle=0.7)
        out = c.step(cmd, dt=0.01)
        assert out.throttle == pytest.approx(0.7)

    def test_throttle_clamped_to_min(self):
        cfg = ControllerConfig(throttle_min=0.3)
        c = VehicleController(config=cfg)
        cmd = ControlInput(throttle=0.0)
        out = c.step(cmd, dt=0.01)
        assert out.throttle == pytest.approx(0.3)

    def test_throttle_clamped_to_max(self):
        cfg = ControllerConfig(throttle_max=0.8)
        c = VehicleController(config=cfg)
        cmd = ControlInput(throttle=1.0)
        out = c.step(cmd, dt=0.01)
        assert out.throttle == pytest.approx(0.8)

    def test_gimbal_pitch_scaled(self):
        cfg = ControllerConfig(max_gimbal_pitch_rad=math.radians(10.0))
        c = VehicleController(config=cfg)
        cmd = ControlInput(throttle=1.0, gimbal_pitch=1.0)
        out = c.step(cmd, dt=0.01)
        assert out.gimbal_pitch_rad == pytest.approx(math.radians(10.0), rel=1e-9)

    def test_gimbal_yaw_scaled(self):
        cfg = ControllerConfig(max_gimbal_yaw_rad=math.radians(8.0))
        c = VehicleController(config=cfg)
        cmd = ControlInput(throttle=1.0, gimbal_yaw=-1.0)
        out = c.step(cmd, dt=0.01)
        assert out.gimbal_yaw_rad == pytest.approx(-math.radians(8.0), rel=1e-9)

    def test_gimbal_clamped_symmetrically(self):
        cfg = ControllerConfig(max_gimbal_pitch_rad=math.radians(5.0))
        c = VehicleController(config=cfg)
        cmd = ControlInput(throttle=1.0, gimbal_pitch=10.0)  # beyond normalised range
        out = c.step(cmd, dt=0.01)
        assert abs(out.gimbal_pitch_rad) <= math.radians(5.0) + 1e-9

    def test_elevator_gain_applied(self):
        cfg = ControllerConfig(elevator_gain=0.5)
        c = VehicleController(config=cfg)
        cmd = ControlInput(elevator=1.0)
        out = c.step(cmd, dt=100.0)  # large dt → actuator converges
        # With gain=0.5, command to actuator is 0.5 → surface converges to 0.5 * max_deflection
        # surface_state elevator should be less than full deflection
        full_c = VehicleController()
        out_full = full_c.step(cmd, dt=100.0)
        assert abs(out.surface_state.elevator_rad) < abs(out_full.surface_state.elevator_rad) + 1e-6

    def test_staging_propagated(self):
        c = VehicleController()
        cmd = ControlInput(staging=True)
        out = c.step(cmd, dt=0.01)
        assert out.staging is True

    def test_deflections_match_surface_state(self):
        c = VehicleController()
        cmd = ControlInput(elevator=0.5, aileron=-0.3, rudder=0.2)
        out = c.step(cmd, dt=100.0)
        assert out.deflections.elevator == pytest.approx(out.surface_state.elevator_rad)
        assert out.deflections.aileron == pytest.approx(out.surface_state.aileron_rad)
        assert out.deflections.rudder == pytest.approx(out.surface_state.rudder_rad)

    def test_wrong_cmd_type_raises(self):
        c = VehicleController()
        with pytest.raises(TypeError):
            c.step("not_a_control_input", dt=0.01)

    def test_zero_dt_raises(self):
        c = VehicleController()
        with pytest.raises(ValueError, match="dt"):
            c.step(ControlInput(), dt=0.0)

    def test_negative_dt_raises(self):
        c = VehicleController()
        with pytest.raises(ValueError, match="dt"):
            c.step(ControlInput(), dt=-0.01)

    def test_reset_returns_actuator_to_neutral(self):
        c = VehicleController()
        c.step(ControlInput(elevator=1.0), dt=10.0)
        c.reset()
        state = c.actuator.current_state
        assert state.elevator_rad == pytest.approx(0.0, abs=1e-9)

    def test_multiple_steps_accumulate(self):
        c = VehicleController()
        cmd = ControlInput(elevator=1.0)
        s1 = c.step(cmd, dt=0.01)
        s2 = c.step(cmd, dt=0.01)
        # Elevator should move further on second step
        assert abs(s2.surface_state.elevator_rad) >= abs(s1.surface_state.elevator_rad)

    def test_neutral_input_neutral_gimbal(self):
        c = VehicleController()
        out = c.step(ControlInput(), dt=0.01)
        assert out.gimbal_pitch_rad == pytest.approx(0.0)
        assert out.gimbal_yaw_rad == pytest.approx(0.0)

    def test_custom_actuator_accepted(self):
        act = ControlSurfaceActuator(default_control_surface_config())
        c = VehicleController(actuator=act)
        out = c.step(ControlInput(), dt=0.01)
        assert isinstance(out, ControllerOutput)


# ===========================================================================
# SASAxisConfig tests
# ===========================================================================

class TestSASAxisConfig:

    def test_default_construction(self):
        cfg = SASAxisConfig()
        assert cfg.kp == pytest.approx(0.1)
        assert cfg.ki == pytest.approx(0.01)
        assert cfg.kd == pytest.approx(0.005)
        assert cfg.max_authority == pytest.approx(0.3)

    def test_custom_values(self):
        cfg = SASAxisConfig(kp=0.5, ki=0.05, kd=0.01, max_authority=0.5)
        assert cfg.kp == pytest.approx(0.5)

    def test_frozen(self):
        cfg = SASAxisConfig()
        with pytest.raises(Exception):
            cfg.kp = 1.0

    def test_negative_kp_rejected(self):
        with pytest.raises(ValueError, match="kp"):
            SASAxisConfig(kp=-0.1)

    def test_negative_ki_rejected(self):
        with pytest.raises(ValueError, match="ki"):
            SASAxisConfig(ki=-0.01)

    def test_negative_kd_rejected(self):
        with pytest.raises(ValueError, match="kd"):
            SASAxisConfig(kd=-0.001)

    def test_authority_above_one_rejected(self):
        with pytest.raises(ValueError, match="max_authority"):
            SASAxisConfig(max_authority=1.1)

    def test_authority_negative_rejected(self):
        with pytest.raises(ValueError, match="max_authority"):
            SASAxisConfig(max_authority=-0.1)

    def test_zero_integrator_limit_rejected(self):
        with pytest.raises(ValueError, match="integrator_limit"):
            SASAxisConfig(integrator_limit=0.0)

    def test_zero_gains_valid(self):
        cfg = SASAxisConfig(kp=0.0, ki=0.0, kd=0.0)
        assert cfg.kp == 0.0


# ===========================================================================
# SASConfig tests
# ===========================================================================

class TestSASConfig:

    def test_default_construction(self):
        cfg = SASConfig()
        assert isinstance(cfg.roll, SASAxisConfig)
        assert isinstance(cfg.pitch, SASAxisConfig)
        assert isinstance(cfg.yaw, SASAxisConfig)
        assert cfg.enabled is True

    def test_disabled(self):
        cfg = SASConfig(enabled=False)
        assert cfg.enabled is False

    def test_custom_axis_config(self):
        roll = SASAxisConfig(kp=0.5)
        cfg = SASConfig(roll=roll)
        assert cfg.roll.kp == pytest.approx(0.5)

    def test_frozen(self):
        cfg = SASConfig()
        with pytest.raises(Exception):
            cfg.enabled = False

    def test_wrong_roll_type_rejected(self):
        with pytest.raises(TypeError):
            SASConfig(roll="not_an_axis_config")


# ===========================================================================
# SASDiagnostic tests
# ===========================================================================

class TestSASDiagnostic:

    def test_valid_construction(self):
        d = SASDiagnostic(
            roll_correction=0.01,
            pitch_correction=-0.02,
            yaw_correction=0.005,
            roll_rate_error=0.5,
            pitch_rate_error=-0.3,
            yaw_rate_error=0.1,
            sas_active=True,
        )
        assert d.roll_correction == pytest.approx(0.01)
        assert d.sas_active is True

    def test_frozen(self):
        d = SASDiagnostic(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
        with pytest.raises(Exception):
            d.sas_active = True

    def test_repr(self):
        d = SASDiagnostic(0.01, -0.02, 0.0, 0.5, 0.0, 0.0, True)
        r = repr(d)
        assert "SASDiagnostic" in r
        assert "active=True" in r


# ===========================================================================
# _pid_correction tests
# ===========================================================================

class TestPIDCorrection:

    def test_zero_error_zero_correction(self):
        state = _SASAxisState()
        cfg = SASAxisConfig(kp=1.0, ki=0.0, kd=0.0, max_authority=1.0)
        corr, err = _pid_correction(0.0, 0.0, state, cfg, dt=0.01)
        assert corr == pytest.approx(0.0)
        assert err == pytest.approx(0.0)

    def test_proportional_term(self):
        state = _SASAxisState()
        cfg = SASAxisConfig(kp=1.0, ki=0.0, kd=0.0, max_authority=1.0)
        # target=0, measured=0.5 → error=-0.5 → P = 1.0 * (-0.5) = -0.5
        corr, err = _pid_correction(0.5, 0.0, state, cfg, dt=0.01)
        assert err == pytest.approx(-0.5)
        assert corr == pytest.approx(-0.5)

    def test_integral_accumulates(self):
        state = _SASAxisState()
        cfg = SASAxisConfig(kp=0.0, ki=1.0, kd=0.0, max_authority=1.0,
                            integrator_limit=100.0)
        # Apply same error 10 times → integral = -0.5 * 10 * dt
        for _ in range(10):
            corr, err = _pid_correction(0.5, 0.0, state, cfg, dt=0.01)
        # Integral = error * n * dt = -0.5 * 10 * 0.01 = -0.05
        assert corr == pytest.approx(-0.05, rel=1e-6)

    def test_integrator_antiwindup(self):
        state = _SASAxisState()
        cfg = SASAxisConfig(kp=0.0, ki=1.0, kd=0.0, max_authority=1.0,
                            integrator_limit=0.1)
        for _ in range(1000):
            _pid_correction(1.0, 0.0, state, cfg, dt=0.01)
        # Integrator clamped at -0.1 → correction = ki * -0.1 = -0.1
        corr, _ = _pid_correction(1.0, 0.0, state, cfg, dt=0.01)
        assert corr == pytest.approx(-0.1, abs=1e-6)

    def test_authority_clamped(self):
        state = _SASAxisState()
        cfg = SASAxisConfig(kp=100.0, ki=0.0, kd=0.0, max_authority=0.2)
        corr, _ = _pid_correction(1.0, 0.0, state, cfg, dt=0.01)
        assert corr == pytest.approx(-0.2)

    def test_derivative_zero_on_first_tick(self):
        """First tick: derivative not initialized → d_term = 0."""
        state = _SASAxisState()
        cfg = SASAxisConfig(kp=0.0, ki=0.0, kd=100.0, max_authority=1.0)
        corr, _ = _pid_correction(0.5, 0.0, state, cfg, dt=0.01)
        assert corr == pytest.approx(0.0)

    def test_derivative_fires_second_tick(self):
        """Second tick with changing error: derivative term is non-zero."""
        state = _SASAxisState()
        cfg = SASAxisConfig(kp=0.0, ki=0.0, kd=1.0, max_authority=1.0)
        _pid_correction(0.5, 0.0, state, cfg, dt=0.01)   # tick 1: d initialized
        corr, _ = _pid_correction(1.0, 0.0, state, cfg, dt=0.01)  # tick 2: d fires
        # error went from -0.5 to -1.0: d_error = (-1.0 - (-0.5)) / 0.01 = -50
        # d_term = kd * d_error = 1.0 * (-50) = -50 → clamped to -1.0 (authority)
        assert corr != pytest.approx(0.0)


# ===========================================================================
# StabilityAugmentationSystem tests
# ===========================================================================

class TestSAS:

    def test_default_construction(self):
        sas = StabilityAugmentationSystem()
        assert sas.enabled is True

    def test_repr(self):
        sas = StabilityAugmentationSystem()
        assert "StabilityAugmentationSystem" in repr(sas)

    def test_wrong_config_type_rejected(self):
        with pytest.raises(TypeError):
            StabilityAugmentationSystem("not_config")

    def test_disabled_sas_passes_through(self, zero_state, neutral_input):
        cfg = SASConfig(enabled=False)
        sas = StabilityAugmentationSystem(cfg)
        aug, diag = sas.step(zero_state, neutral_input, dt=0.01)
        assert aug.throttle == pytest.approx(neutral_input.throttle)
        assert aug.elevator == pytest.approx(neutral_input.elevator)
        assert diag.sas_active is False

    def test_disabled_returns_identical_command(self, zero_state, full_input):
        cfg = SASConfig(enabled=False)
        sas = StabilityAugmentationSystem(cfg)
        aug, _ = sas.step(zero_state, full_input, dt=0.01)
        assert aug.throttle == pytest.approx(full_input.throttle)
        assert aug.elevator == pytest.approx(full_input.elevator)
        assert aug.aileron == pytest.approx(full_input.aileron)
        assert aug.staging == full_input.staging

    def test_zero_rates_no_correction(self, zero_state, neutral_input):
        sas = StabilityAugmentationSystem()
        aug, diag = sas.step(zero_state, neutral_input, dt=0.01)
        # No rates → no corrections
        assert diag.roll_correction == pytest.approx(0.0)
        assert diag.pitch_correction == pytest.approx(0.0)
        assert diag.yaw_correction == pytest.approx(0.0)

    def test_roll_rate_produces_aileron_correction(self, spinning_state, neutral_input):
        sas = StabilityAugmentationSystem()
        _, diag = sas.step(spinning_state, neutral_input, dt=0.01)
        # Roll rate is +1 rad/s → SAS should produce non-zero roll correction
        assert abs(diag.roll_correction) > 0.0

    def test_roll_correction_opposes_rate(self, spinning_state, neutral_input):
        """With +roll rate, SAS correction should be negative (oppose it)."""
        sas = StabilityAugmentationSystem()
        aug, diag = sas.step(spinning_state, neutral_input, dt=0.01)
        # roll_rate > 0 → correction < 0 (aileron to oppose roll)
        assert diag.roll_correction < 0.0

    def test_throttle_unchanged_by_sas(self, spinning_state):
        sas = StabilityAugmentationSystem()
        cmd = ControlInput(throttle=0.8)
        aug, _ = sas.step(spinning_state, cmd, dt=0.01)
        assert aug.throttle == pytest.approx(0.8)

    def test_gimbal_unchanged_by_sas(self, spinning_state):
        sas = StabilityAugmentationSystem()
        cmd = ControlInput(gimbal_pitch=0.3, gimbal_yaw=-0.5)
        aug, _ = sas.step(spinning_state, cmd, dt=0.01)
        assert aug.gimbal_pitch == pytest.approx(0.3)
        assert aug.gimbal_yaw == pytest.approx(-0.5)

    def test_staging_unchanged_by_sas(self, spinning_state):
        sas = StabilityAugmentationSystem()
        cmd = ControlInput(staging=True)
        aug, _ = sas.step(spinning_state, cmd, dt=0.01)
        assert aug.staging is True

    def test_full_pilot_command_washout(self, spinning_state):
        """At full stick deflection, SAS washout = 1 - |1| = 0: no augmentation."""
        sas = StabilityAugmentationSystem(
            SASConfig(roll=SASAxisConfig(kp=1.0, ki=0.0, kd=0.0))
        )
        cmd = ControlInput(aileron=1.0)  # full stick
        aug, _ = sas.step(spinning_state, cmd, dt=0.01)
        # With full stick, washout = 0 → augmented aileron ≈ 1.0
        assert aug.aileron == pytest.approx(1.0, abs=1e-9)

    def test_augmented_output_within_bounds(self, spinning_state):
        sas = StabilityAugmentationSystem()
        cmd = ControlInput(aileron=0.8)
        aug, _ = sas.step(spinning_state, cmd, dt=0.01)
        assert -1.0 <= aug.aileron <= 1.0
        assert -1.0 <= aug.elevator <= 1.0
        assert -1.0 <= aug.rudder <= 1.0

    def test_wrong_state_type_raises(self, neutral_input):
        sas = StabilityAugmentationSystem()
        with pytest.raises(TypeError):
            sas.step("not_a_state", neutral_input, dt=0.01)

    def test_wrong_cmd_type_raises(self, zero_state):
        sas = StabilityAugmentationSystem()
        with pytest.raises(TypeError):
            sas.step(zero_state, "not_a_control_input", dt=0.01)

    def test_zero_dt_raises(self, zero_state, neutral_input):
        sas = StabilityAugmentationSystem()
        with pytest.raises(ValueError, match="dt"):
            sas.step(zero_state, neutral_input, dt=0.0)

    def test_negative_dt_raises(self, zero_state, neutral_input):
        sas = StabilityAugmentationSystem()
        with pytest.raises(ValueError, match="dt"):
            sas.step(zero_state, neutral_input, dt=-0.01)

    def test_wrong_target_rates_shape_raises(self, zero_state, neutral_input):
        sas = StabilityAugmentationSystem()
        with pytest.raises(ValueError):
            sas.step(zero_state, neutral_input, dt=0.01,
                     target_rates=np.array([0.0, 0.0]))

    def test_reset_clears_integrator(self, spinning_state, neutral_input):
        sas = StabilityAugmentationSystem(
            SASConfig(roll=SASAxisConfig(kp=0.0, ki=1.0, kd=0.0,
                                         integrator_limit=100.0, max_authority=1.0))
        )
        # Accumulate integrator
        for _ in range(100):
            sas.step(spinning_state, neutral_input, dt=0.01)
        # Reset
        sas.reset()
        # After reset: integrator = 0, so only P term (which is 0 here)
        _, diag = sas.step(spinning_state, neutral_input, dt=0.01)
        # After reset, integral = 0.0 on first tick, but error accumulates immediately
        assert diag.roll_correction == pytest.approx(-0.01, abs=1e-10)

    def test_multi_tick_convergence(self):
        """With high gains, roll rate should be damped over many ticks."""
        sas = StabilityAugmentationSystem(
            SASConfig(roll=SASAxisConfig(kp=0.5, ki=0.0, kd=0.0, max_authority=1.0))
        )
        # Start with 1 rad/s roll rate
        corrections = []
        state = make_state(
            position_eci=np.array([6_771_000.0, 0.0, 0.0]),
            velocity_eci=np.array([0.0, 7_800.0, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            omega_body=np.array([1.0, 0.0, 0.0]),
            mass=1000.0,
            time=0.0,
        )
        for _ in range(5):
            _, diag = sas.step(state, ControlInput(), dt=0.01)
            corrections.append(abs(diag.roll_correction))
        # Corrections should be consistent (P-only, same rate → same correction)
        assert all(abs(c - corrections[0]) < 1e-9 for c in corrections)

    def test_diag_rate_errors_correct(self, neutral_input):
        """Diagnostic roll_rate_error = target - measured."""
        sas = StabilityAugmentationSystem()
        state = make_state(
            position_eci=np.array([6_771_000.0, 0.0, 0.0]),
            velocity_eci=np.array([0.0, 7_800.0, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            omega_body=np.array([0.3, -0.2, 0.1]),
            mass=1000.0,
            time=0.0,
        )
        _, diag = sas.step(state, neutral_input, dt=0.01)
        # target=0, measured=0.3 → error = -0.3
        assert diag.roll_rate_error == pytest.approx(-0.3, abs=1e-12)
        assert diag.pitch_rate_error == pytest.approx(0.2, abs=1e-12)
        assert diag.yaw_rate_error == pytest.approx(-0.1, abs=1e-12)

    def test_target_rates_nonzero(self, neutral_input):
        """Non-zero target rates shift the error correctly."""
        sas = StabilityAugmentationSystem(
            SASConfig(roll=SASAxisConfig(kp=1.0, ki=0.0, kd=0.0, max_authority=1.0))
        )
        state = make_state(
            position_eci=np.array([6_771_000.0, 0.0, 0.0]),
            velocity_eci=np.array([0.0, 7_800.0, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            omega_body=np.array([0.5, 0.0, 0.0]),
            mass=1000.0,
            time=0.0,
        )
        # target rate = 0.5 rad/s → error = 0 → correction = 0
        _, diag = sas.step(state, neutral_input, dt=0.01,
                           target_rates=np.array([0.5, 0.0, 0.0]))
        assert diag.roll_rate_error == pytest.approx(0.0, abs=1e-12)
        assert diag.roll_correction == pytest.approx(0.0, abs=1e-12)
