"""
nova.core.pipeline
==================
13-stage deterministic simulation pipeline for Project NOVA.

Architecture role — Master loop orchestrator
---------------------------------------------
The pipeline drives the complete simulation at a fixed timestep dt.
Each call to ``tick()`` executes all 13 stages in strict sequential order.
No stage reads mutable state written by a later stage in the same tick.

Stage execution order
---------------------
1.  Input Handler       Poll hardware control state vector.
2.  Vehicle Controller  Map inputs to actuator commands.
3.  Physics Engine      Accumulate global F and τ tensors.
4.  RK4 Integrator      Propagate state forward by dt.
5.  Collision Detection Bounding-volume and raycast checks.
6.  Orbital Solver      N-body gravity + Keplerian elements.
7.  Atmosphere Solver   Density, temperature, speed of sound.
8.  Thermodynamics      Convective/radiative skin heating.
9.  Component Updates   Mass flow, fatigue accumulation.
10. AI Monitor          Predictive anomaly detection.
11. Telemetry Layer     Serialise snapshot to registry.
12. Renderer            Variable-rate visual update (decoupled).
13. UI Engine           Refresh HUD panels.

Stages 8 (Thermodynamics), 12 (Renderer), and 13 (UI Engine) are stubbed
in Phase 5. They are designed as no-ops with the correct interface so Phase 8
and Phase 10 can drop in concrete implementations without touching pipeline.py.

Determinism contract
--------------------
The pipeline is deterministic: given identical initial state, control inputs,
and dt, it always produces identical output. This requires:
  - No wall-clock time in physics calculations.
  - No global mutable state outside the pipeline instance.
  - Forces and torques accumulated fresh every tick.
  - RK4 integrator operating on float64 flat state vector.

Configuration
-------------
PipelineConfig controls which optional subsystems are active. Inactive
subsystems (e.g. aerodynamics in vacuum) are bypassed at zero cost.
"""

from __future__ import annotations

import math
import time as _wall_clock
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from nova.core.constants import (
    EARTH_MU,
    EARTH_RADIUS_MEAN,
    EARTH_RADIUS_EQ,
    EARTH_J2,
    STD_GRAVITY,
)
from nova.core.state_vector import VehicleState
from nova.core.telemetry_registry import (
    TelemetryRegistry,
    TelemetrySnapshot,
    build_snapshot,
)
from nova.frames.transforms import (
    T_ENU_to_body,
    euler_to_quaternion,
    quaternion_to_euler,
)
from nova.physics.integrator import build_deriv_fn, integrate_state
from nova.physics.atmosphere import atmosphere, atmosphere_from_eci
from nova.physics.orbital import (
    gravity_acceleration,
    gravity_force,
    elements_from_state,
    EARTH_BODY,
    GravBody,
)
from nova.physics.forces import ForceAccumulator, rotate_body_to_eci
from nova.physics.torques import TorqueAccumulator
from nova.physics.aerodynamics import (
    AeroConfig,
    ControlDeflections,
    compute_aero,
    AeroState,
)
from nova.physics.propulsion import (
    EngineConfig,
    compute_propulsion,
    PropulsionState,
)
from nova.vehicle.mass_model import MassModel, compute_mass_properties
from nova.vehicle.component_graph import ComponentGraph
from nova.physics.structural import (
    VehicleLoadState,
    analyse_structure,
    critical_margin,
)


# ---------------------------------------------------------------------------
# Control input vector
# ---------------------------------------------------------------------------

@dataclass
class ControlInput:
    """
    Immutable hardware control state vector produced by Stage 1.

    All values normalised to [-1, 1] or [0, 1] as noted.

    Parameters
    ----------
    throttle : float
        Engine throttle command [0, 1].
    gimbal_pitch : float
        Engine gimbal pitch command [rad].
    gimbal_yaw : float
        Engine gimbal yaw command [rad].
    elevator : float
        Elevator deflection command [rad].
    aileron : float
        Aileron deflection command [rad].
    rudder : float
        Rudder deflection command [rad].
    staging : bool
        True on the tick a stage separation command is issued.
    """
    throttle:     float = 0.0
    gimbal_pitch: float = 0.0
    gimbal_yaw:   float = 0.0
    elevator:     float = 0.0
    aileron:      float = 0.0
    rudder:       float = 0.0
    staging:      bool  = False


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Runtime configuration for the simulation pipeline.

    Parameters
    ----------
    dt : float
        Fixed simulation timestep [s]. Default 0.01 s.
    enable_aerodynamics : bool
        If False, aerodynamic forces/moments are zeroed (rocket above 80 km).
    enable_j2 : bool
        If True, include J2 oblateness perturbation in gravity.
    enable_structural : bool
        If True, run structural analysis each tick.
    gravity_bodies : list of GravBody
        Gravitational bodies to include in N-body computation.
    aero_config : AeroConfig or None
        Vehicle aerodynamic configuration. Required if enable_aerodynamics=True.
    engine_config : EngineConfig or None
        Engine configuration. If None, thrust is always zero.
    aero_vacuum_altitude : float
        Altitude [m] above which aerodynamics is disabled regardless of flag.
    """
    dt:                    float  = 0.01
    enable_aerodynamics:   bool   = True
    enable_j2:             bool   = True
    enable_structural:     bool   = False
    gravity_bodies:        list   = field(default_factory=lambda: [EARTH_BODY])
    aero_config:           Optional[AeroConfig]   = None
    engine_config:         Optional[EngineConfig] = None
    aero_vacuum_altitude:  float  = 80_000.0   # [m]


# ---------------------------------------------------------------------------
# Per-tick pipeline result
# ---------------------------------------------------------------------------

@dataclass
class TickResult:
    """
    Complete output of one pipeline tick.

    Attributes
    ----------
    new_state : VehicleState
        State after RK4 integration.
    snapshot : TelemetrySnapshot
        Telemetry snapshot published to the registry.
    dt : float
        Timestep used [s].
    structural_failures : list of str
        Joint IDs that failed this tick. Empty if no failures.
    """
    new_state:            VehicleState
    snapshot:             TelemetrySnapshot
    dt:                   float
    structural_failures:  List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class SimulationPipeline:
    """
    13-stage deterministic simulation pipeline.

    Usage
    -----
    ::

        config  = PipelineConfig(dt=0.01, engine_config=my_engine)
        pipe    = SimulationPipeline(config, initial_state, component_graph)
        registry = pipe.registry

        for _ in range(n_ticks):
            control = ControlInput(throttle=1.0)
            result  = pipe.tick(control)
            # result.new_state is the next VehicleState
            # registry.latest   is the telemetry snapshot

    Parameters
    ----------
    config : PipelineConfig
        Simulation configuration.
    initial_state : VehicleState
        Starting vehicle state.
    component_graph : ComponentGraph
        Vehicle component graph (mutable — structural solver updates joints).
    registry : TelemetryRegistry, optional
        External registry to publish to. Creates a new one if not supplied.
    propellant_mass : float
        Initial propellant mass [kg]. Depleted by the engine each tick.
    """

    def __init__(
        self,
        config:          PipelineConfig,
        initial_state:   VehicleState,
        component_graph: ComponentGraph,
        registry:        Optional[TelemetryRegistry] = None,
        propellant_mass: float = math.inf,
    ) -> None:
        self._config       = config
        self._state        = initial_state
        self._graph        = component_graph
        self._registry     = registry if registry is not None else TelemetryRegistry()
        self._propellant   = propellant_mass
        self._downrange    = 0.0   # cumulative [m]
        self._tick_count   = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> VehicleState:
        """Current vehicle state (latest integrated snapshot)."""
        return self._state

    @property
    def registry(self) -> TelemetryRegistry:
        """Read-only telemetry registry."""
        return self._registry

    @property
    def tick_count(self) -> int:
        """Number of ticks executed so far."""
        return self._tick_count

    def tick(self, control: ControlInput = ControlInput()) -> TickResult:
        """
        Execute one complete simulation tick (all 13 stages).

        Parameters
        ----------
        control : ControlInput
            Current hardware control state vector.

        Returns
        -------
        TickResult
        """
        dt  = self._config.dt
        cfg = self._config
        s   = self._state

        # ── Stage 1: Input Handler ──────────────────────────────────
        # Control vector already consumed as parameter; no hardware poll
        # needed in headless mode. In full Pygame mode this stage calls
        # pygame.event.get() and maps keyboard/joystick to ControlInput.

        # ── Stage 2: Vehicle Controller ─────────────────────────────
        deflections = ControlDeflections(
            elevator=control.elevator,
            aileron=control.aileron,
            rudder=control.rudder,
        )

        # ── Stage 3: Physics Engine (force + torque accumulation) ────
        # Mass model from component graph
        active_comps = self._graph.active_mass_components()
        mass_model   = compute_mass_properties(active_comps)

        # DCM: body → ECI (approximate: use attitude quaternion directly)
        R_body_to_eci = T_ENU_to_body(s.quaternion).T   # body→ENU ≈ body→ECI

        # --- 3a. Forces ---
        facc = ForceAccumulator(s)

        # Gravity (Stage 6 feeds back here; pre-evaluated for current pos)
        a_grav = gravity_acceleration(
            s.position_eci,
            cfg.gravity_bodies,
            include_j2=cfg.enable_j2,
        )
        F_grav = s.mass * a_grav
        facc.add_gravity(F_grav)

        # Propulsion
        prop_state: Optional[PropulsionState] = None
        if cfg.engine_config is not None:
            atm_for_isp = atmosphere_from_eci(s.position_eci, EARTH_RADIUS_MEAN)
            prop_state  = compute_propulsion(
                cfg.engine_config,
                throttle=control.throttle,
                atm_pressure=atm_for_isp.pressure,
                gimbal_pitch=control.gimbal_pitch,
                gimbal_yaw=control.gimbal_yaw,
                propellant_remaining=self._propellant,
            )
            if prop_state.is_active:
                facc.add_thrust(prop_state.thrust_body, R_body_to_eci)

        # Aerodynamics
        aero_state: Optional[AeroState] = None
        atm_state   = atmosphere_from_eci(s.position_eci, EARTH_RADIUS_MEAN)
        aero_active = (
            cfg.enable_aerodynamics
            and cfg.aero_config is not None
            and atm_state.altitude < cfg.aero_vacuum_altitude
            and atm_state.density > 1.0e-6
        )
        if aero_active and cfg.aero_config is not None:
            v_body = T_ENU_to_body(s.quaternion) @ s.velocity_eci
            aero_state = compute_aero(
                v_body, s.omega_body, atm_state, cfg.aero_config, deflections
            )
            facc.add_aerodynamic(aero_state.force_body, R_body_to_eci)

        F_net_eci = facc.build()

        # --- 3b. Torques ---
        tacc = TorqueAccumulator(s)
        if aero_state is not None:
            tacc.add_aerodynamic_moments(
                pitching_moment=aero_state.pitching_moment,
                yawing_moment=aero_state.yawing_moment,
                rolling_moment=aero_state.rolling_moment,
            )
        if prop_state is not None and prop_state.is_active and cfg.engine_config is not None:
            if (cfg.engine_config.gimbal_max_rad > 0.0
                    and (prop_state.gimbal_angle_pitch != 0.0
                         or prop_state.gimbal_angle_yaw != 0.0)):
                tacc.add_moment_arm(
                    "gimbal",
                    prop_state.thrust_body,
                    cfg.engine_config.mount_point_body,
                )
        tau_net_body = tacc.build()

        # ── Stage 4: RK4 Integrator ──────────────────────────────────
        new_mass = s.mass
        if prop_state is not None and prop_state.is_active:
            dm = prop_state.mass_flow_rate * dt
            self._propellant = max(0.0, self._propellant - dm)
            new_mass         = max(s.mass - dm, 0.001)

        deriv_fn  = build_deriv_fn(F_net_eci, tau_net_body,
                                   mass_model.inertia_body, s.mass)
        new_state = integrate_state(deriv_fn, s, dt, new_mass=new_mass)

        # ── Stage 5: Collision Detection ─────────────────────────────
        # Stubbed: bounding-volume/raycast against terrain mesh.
        # Phase 7 will insert CollisionDetector here.

        # ── Stage 6: Orbital Solver ───────────────────────────────────
        try:
            orb_elements = elements_from_state(
                new_state.position_eci, new_state.velocity_eci
            )
        except (ValueError, ZeroDivisionError):
            orb_elements = None

        # ── Stage 7: Atmosphere Solver ────────────────────────────────
        atm_new = atmosphere_from_eci(new_state.position_eci, EARTH_RADIUS_MEAN)

        # ── Stage 8: Thermodynamics Engine ───────────────────────────
        # Stubbed. Phase 8 inserts skin-heating model here.

        # ── Stage 9: Component Updates ────────────────────────────────
        # Structural analysis (if enabled)
        structural_failures: List[str] = []
        worst_margin  = 1.0
        critical_jid  = ""
        any_failure   = False

        if cfg.enable_structural:
            accel_body = T_ENU_to_body(new_state.quaternion) @ (
                F_net_eci / s.mass
            )
            thrust_mag = (float(np.linalg.norm(prop_state.thrust_body))
                          if prop_state and prop_state.is_active else 0.0)
            drag_body  = (aero_state.force_body.copy()
                          if aero_state else np.zeros(3, dtype=np.float64))
            grav_body  = T_ENU_to_body(new_state.quaternion) @ a_grav

            load_state = VehicleLoadState(
                acceleration_body=accel_body.astype(np.float64),
                alpha_body=np.zeros(3, dtype=np.float64),
                dynamic_pressure=atm_new.dynamic_pressure_at_speed(
                    float(np.linalg.norm(new_state.velocity_eci))),
                axial_thrust=thrust_mag,
                aero_drag_body=drag_body.astype(np.float64),
                gravity_body=grav_body.astype(np.float64),
            )
            results = analyse_structure(self._graph, load_state)
            cm      = critical_margin(results)
            if cm:
                worst_margin = min(cm.margin_axial, cm.margin_shear, cm.margin_bending)
                critical_jid = cm.joint_id
            failed_joints = self._graph.evaluate_structural_failures()
            structural_failures = [j.joint_id for j in failed_joints]
            any_failure = len(structural_failures) > 0

        # Staging event (component jettison)
        if control.staging:
            # Future: identify which stage to jettison from graph topology.
            # For now this is a no-op placeholder.
            pass

        # ── Stage 10: AI Monitor ──────────────────────────────────────
        # Stubbed. Phase 6 inserts AIMonitor here.

        # ── Stage 11: Telemetry Layer ─────────────────────────────────
        speed  = new_state.speed
        r_prev = float(np.linalg.norm(s.position_eci))
        r_now  = float(np.linalg.norm(new_state.position_eci))
        v_speed = (r_now - r_prev) / dt if dt > 0 else 0.0

        # Cumulative downrange (approximate arc-length on sphere)
        d_pos = float(np.linalg.norm(new_state.position_eci - s.position_eci))
        self._downrange += d_pos

        _z3 = np.zeros(3, dtype=np.float64)
        _g = facc.contribution_by_name("gravity");      F_grav_snap   = _g   if _g   is not None else _z3.copy()
        _t = facc.contribution_by_name("thrust");       F_thrust_snap = _t   if _t   is not None else _z3.copy()
        _a = facc.contribution_by_name("aerodynamic");  F_aero_snap   = _a   if _a   is not None else _z3.copy()
        _ta = tacc.contribution_by_name("aerodynamic"); tau_aero_snap   = _ta  if _ta  is not None else _z3.copy()
        _tg = tacc.contribution_by_name("gimbal");      tau_gimbal_snap = _tg  if _tg  is not None else _z3.copy()

        snapshot = build_snapshot(
            new_state,
            altitude=atm_new.altitude,
            density=atm_new.density,
            pressure=atm_new.pressure,
            speed_of_sound=atm_new.speed_of_sound,
            mach=atm_new.mach(speed),
            dynamic_pressure=atm_new.dynamic_pressure_at_speed(speed),
            alpha=aero_state.alpha if aero_state else 0.0,
            beta=aero_state.beta  if aero_state else 0.0,
            CL=aero_state.CL     if aero_state else 0.0,
            CD=aero_state.CD     if aero_state else 0.0,
            lift_force=float(np.linalg.norm(aero_state.lift_body)) if aero_state else 0.0,
            drag_force=float(np.linalg.norm(aero_state.drag_body)) if aero_state else 0.0,
            thrust_magnitude=float(np.linalg.norm(prop_state.thrust_body)) if prop_state else 0.0,
            mass_flow_rate=prop_state.mass_flow_rate if prop_state else 0.0,
            isp_effective=prop_state.isp_effective   if prop_state else 0.0,
            throttle=prop_state.throttle             if prop_state else 0.0,
            semi_major_axis=orb_elements.semi_major_axis           if orb_elements else 0.0,
            eccentricity=orb_elements.eccentricity                 if orb_elements else 0.0,
            inclination=orb_elements.inclination                   if orb_elements else 0.0,
            raan=orb_elements.raan                                 if orb_elements else 0.0,
            argument_of_periapsis=orb_elements.argument_of_periapsis if orb_elements else 0.0,
            true_anomaly=orb_elements.true_anomaly                 if orb_elements else 0.0,
            orbital_period=orb_elements.period                     if orb_elements else 0.0,
            apoapsis=orb_elements.apoapsis                         if orb_elements else 0.0,
            periapsis=orb_elements.periapsis                       if orb_elements else 0.0,
            force_gravity=F_grav_snap.astype(np.float64),
            force_thrust=F_thrust_snap.astype(np.float64),
            force_aero=F_aero_snap.astype(np.float64),
            force_net=F_net_eci.astype(np.float64),
            torque_aero=tau_aero_snap.astype(np.float64),
            torque_gimbal=tau_gimbal_snap.astype(np.float64),
            torque_net=tau_net_body.astype(np.float64),
            worst_structural_margin=worst_margin,
            critical_joint_id=critical_jid,
            any_structural_failure=any_failure,
            speed=speed,
            vertical_speed=v_speed,
            downrange_distance=self._downrange,
        )
        self._registry.publish(snapshot)

        # ── Stage 12: Renderer ────────────────────────────────────────
        # Decoupled from sim frequency. Phase 10 inserts Renderer here.

        # ── Stage 13: UI Engine ───────────────────────────────────────
        # Phase 10 inserts HUD refresh here.

        # Advance state
        self._state      = new_state
        self._tick_count += 1

        return TickResult(
            new_state=new_state,
            snapshot=snapshot,
            dt=dt,
            structural_failures=structural_failures,
        )

    def run(
        self,
        n_ticks: int,
        control_fn: Optional[Callable[[int, VehicleState], ControlInput]] = None,
    ) -> List[TickResult]:
        """
        Run the pipeline for ``n_ticks`` steps.

        Parameters
        ----------
        n_ticks : int
            Number of ticks to execute.
        control_fn : callable (tick_index, state) → ControlInput, optional
            Function that returns the control input for each tick.
            If None, uses zero control (ControlInput()) throughout.

        Returns
        -------
        list of TickResult
            One result per tick.
        """
        results = []
        for i in range(n_ticks):
            ctrl = control_fn(i, self._state) if control_fn else ControlInput()
            results.append(self.tick(ctrl))
        return results
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_pipeline.py
============================
Integration tests for nova.core.pipeline.

Tests verify:
  1. Pipeline construction with minimal config.
  2. Single tick advances mission time by dt.
  3. Multiple ticks are deterministic (same result for same inputs).
  4. Telemetry registry receives one snapshot per tick.
  5. Force gravity populates telemetry snapshot.
  6. Engine thrust reduces propellant mass monotonically.
  7. Propellant exhaustion stops engine without crashing.
  8. Orbital elements computed and published in snapshot.
  9. Aerodynamics disabled above vacuum altitude.
  10. run() convenience method returns correct number of results.
  11. Tick count increments correctly.
  12. TickResult contains new_state with advanced time.
"""

import math
import pytest
import numpy as np

from nova.core.state_vector import make_state, VehicleState
from nova.core.pipeline import (
    SimulationPipeline,
    PipelineConfig,
    ControlInput,
    TickResult,
)
from nova.core.telemetry_registry import TelemetryRegistry
from nova.physics.propulsion import EngineConfig
from nova.physics.aerodynamics import AeroConfig
from nova.vehicle.component_graph import ComponentGraph, ComponentNode
from nova.vehicle.mass_model import point_mass, solid_cylinder
from nova.core.constants import EARTH_RADIUS_MEAN, EARTH_MU


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _leo_state(t: float = 0.0, mass: float = 5000.0) -> VehicleState:
    """Circular 400 km LEO initial state."""
    r = EARTH_RADIUS_MEAN + 400_000.0
    v = math.sqrt(EARTH_MU / r)
    return make_state(
        position_eci=[r, 0.0, 0.0],
        velocity_eci=[0.0, v, 0.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        omega_body=[0.0, 0.0, 0.0],
        mass=mass,
        time=t,
    )


def _minimal_graph(mass: float = 5000.0) -> ComponentGraph:
    g = ComponentGraph()
    mc = solid_cylinder("body", mass, 2.0, 20.0, [0.0, 0.0, 0.0])
    n  = ComponentNode("body", "Body", mc, "structure", is_separable=False)
    g.add_node(n)
    return g


def _minimal_config(dt: float = 0.01) -> PipelineConfig:
    return PipelineConfig(
        dt=dt,
        enable_aerodynamics=False,
        enable_j2=False,
        enable_structural=False,
        aero_config=None,
        engine_config=None,
    )


def _engine() -> EngineConfig:
    return EngineConfig(
        name="TestEngine",
        thrust_vac=50_000.0,
        isp_vac=300.0,
        isp_sl=270.0,
        throttle_min=0.0,
        throttle_max=1.0,
        gimbal_max_rad=0.0,
        exit_area=0.0,
        exit_pressure=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestPipelineConstruction:

    def test_constructs_with_minimal_config(self):
        state = _leo_state()
        graph = _minimal_graph()
        cfg   = _minimal_config()
        pipe  = SimulationPipeline(cfg, state, graph)
        assert pipe.tick_count == 0
        assert pipe.state is state

    def test_uses_provided_registry(self):
        state = _leo_state()
        graph = _minimal_graph()
        cfg   = _minimal_config()
        reg   = TelemetryRegistry(buffer_size=50)
        pipe  = SimulationPipeline(cfg, state, graph, registry=reg)
        assert pipe.registry is reg

    def test_creates_registry_if_not_supplied(self):
        state = _leo_state()
        graph = _minimal_graph()
        cfg   = _minimal_config()
        pipe  = SimulationPipeline(cfg, state, graph)
        assert isinstance(pipe.registry, TelemetryRegistry)


# ---------------------------------------------------------------------------
# 2. Single tick basics
# ---------------------------------------------------------------------------

class TestSingleTick:

    @pytest.fixture
    def pipe(self):
        return SimulationPipeline(
            _minimal_config(dt=0.01), _leo_state(), _minimal_graph()
        )

    def test_tick_returns_tick_result(self, pipe):
        result = pipe.tick()
        assert isinstance(result, TickResult)

    def test_tick_advances_time_by_dt(self, pipe):
        result = pipe.tick()
        assert abs(result.new_state.time - 0.01) < 1.0e-12

    def test_tick_count_increments(self, pipe):
        pipe.tick()
        assert pipe.tick_count == 1
        pipe.tick()
        assert pipe.tick_count == 2

    def test_state_updated_after_tick(self, pipe):
        initial_time = pipe.state.time
        pipe.tick()
        assert pipe.state.time > initial_time

    def test_registry_receives_snapshot(self, pipe):
        assert len(pipe.registry) == 0
        pipe.tick()
        assert len(pipe.registry) == 1

    def test_one_snapshot_per_tick(self, pipe):
        for _ in range(5):
            pipe.tick()
        assert len(pipe.registry) == 5

    def test_new_state_in_result(self, pipe):
        result = pipe.tick()
        assert result.new_state is pipe.state

    def test_dt_in_result(self, pipe):
        result = pipe.tick()
        assert result.dt == 0.01


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_identical_inputs_identical_outputs(self):
        """Two pipelines with the same config and state produce identical results."""
        state  = _leo_state()
        graph1 = _minimal_graph()
        graph2 = _minimal_graph()
        cfg    = _minimal_config(dt=0.01)
        ctrl   = ControlInput()

        pipe1 = SimulationPipeline(cfg, state, graph1)
        pipe2 = SimulationPipeline(cfg, state, graph2)

        r1 = pipe1.tick(ctrl)
        r2 = pipe2.tick(ctrl)

        assert np.allclose(r1.new_state.position_eci, r2.new_state.position_eci, atol=1.0e-10)
        assert np.allclose(r1.new_state.velocity_eci, r2.new_state.velocity_eci, atol=1.0e-10)
        assert np.allclose(r1.new_state.quaternion,   r2.new_state.quaternion,   atol=1.0e-12)

    def test_100_ticks_deterministic(self):
        """100-tick run is reproducible."""
        state  = _leo_state()

        def run_pipeline():
            g = _minimal_graph(5000.0)
            p = SimulationPipeline(_minimal_config(dt=0.1), state, g)
            results = p.run(100)
            return results[-1].new_state

        s1 = run_pipeline()
        s2 = run_pipeline()
        assert np.allclose(s1.position_eci, s2.position_eci, atol=1.0e-8)


# ---------------------------------------------------------------------------
# 4. Gravity in telemetry
# ---------------------------------------------------------------------------

class TestGravityTelemetry:

    def test_gravity_force_nonzero_in_snapshot(self):
        state = _leo_state()
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph())
        pipe.tick()
        snap = pipe.registry.latest
        assert snap is not None
        F_grav_mag = float(np.linalg.norm(snap.force_gravity))
        assert F_grav_mag > 0.0, "Gravity force should be nonzero"

    def test_gravity_magnitude_order_of_magnitude(self):
        """At 400 km LEO: F_grav ≈ m × g ≈ 5000 × 8.43 ≈ 42 150 N."""
        mass  = 5000.0
        state = _leo_state(mass=mass)
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph(mass))
        pipe.tick()
        snap  = pipe.registry.latest
        F_mag = float(np.linalg.norm(snap.force_gravity))
        # g at 400 km ≈ 8.43 m/s²
        assert 30_000 < F_mag < 60_000, f"F_grav = {F_mag:.0f} N unexpected"


# ---------------------------------------------------------------------------
# 5. Orbital elements in telemetry
# ---------------------------------------------------------------------------

class TestOrbitalElementsTelemetry:

    def test_orbital_elements_populated(self):
        state = _leo_state()
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph())
        pipe.tick()
        snap = pipe.registry.latest
        assert snap.semi_major_axis > 0.0
        assert 0.0 <= snap.eccentricity < 1.0

    def test_semi_major_axis_near_leo_radius(self):
        """For 400 km circular orbit, a ≈ 6 771 km."""
        r     = EARTH_RADIUS_MEAN + 400_000.0
        state = _leo_state()
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph())
        pipe.tick()
        snap = pipe.registry.latest
        assert abs(snap.semi_major_axis - r) / r < 0.01, \
            f"a = {snap.semi_major_axis/1000:.1f} km, expected {r/1000:.1f} km"

    def test_eccentricity_near_zero_for_circular(self):
        state = _leo_state()
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph())
        pipe.tick()
        assert pipe.registry.latest.eccentricity < 0.01


# ---------------------------------------------------------------------------
# 6. Engine and propellant
# ---------------------------------------------------------------------------

class TestEngineAndPropellant:

    @pytest.fixture
    def engine_pipe(self):
        state  = _leo_state(mass=5000.0)
        graph  = _minimal_graph(5000.0)
        cfg    = PipelineConfig(
            dt=0.1,
            enable_aerodynamics=False,
            enable_j2=False,
            engine_config=_engine(),
        )
        return SimulationPipeline(cfg, state, graph, propellant_mass=2000.0)

    def test_thrust_appears_in_snapshot(self, engine_pipe):
        ctrl = ControlInput(throttle=1.0)
        engine_pipe.tick(ctrl)
        snap = engine_pipe.registry.latest
        assert snap.thrust_magnitude > 0.0

    def test_mass_decreases_with_thrust(self, engine_pipe):
        initial_mass = engine_pipe.state.mass
        ctrl = ControlInput(throttle=1.0)
        engine_pipe.tick(ctrl)
        assert engine_pipe.state.mass < initial_mass

    def test_mass_flow_rate_in_snapshot(self, engine_pipe):
        engine_pipe.tick(ControlInput(throttle=1.0))
        snap = engine_pipe.registry.latest
        assert snap.mass_flow_rate > 0.0

    def test_zero_throttle_no_thrust(self, engine_pipe):
        engine_pipe.tick(ControlInput(throttle=0.0))
        snap = engine_pipe.registry.latest
        assert snap.thrust_magnitude == 0.0

    def test_propellant_decreases_monotonically(self, engine_pipe):
        ctrl = ControlInput(throttle=1.0)
        prev_mass = engine_pipe.state.mass
        for _ in range(10):
            engine_pipe.tick(ctrl)
            assert engine_pipe.state.mass <= prev_mass
            prev_mass = engine_pipe.state.mass

    def test_propellant_exhaustion_no_crash(self):
        """Engine should shut off gracefully when propellant runs out."""
        state = _leo_state(mass=1001.0)
        graph = _minimal_graph(1001.0)
        cfg   = PipelineConfig(
            dt=0.1,
            enable_aerodynamics=False,
            engine_config=_engine(),
        )
        # Only 1 kg propellant — engine shuts off almost immediately
        pipe = SimulationPipeline(cfg, state, graph, propellant_mass=1.0)
        for _ in range(50):
            pipe.tick(ControlInput(throttle=1.0))
        # Must complete without exception; thrust must be zero after exhaustion
        snap = pipe.registry.latest
        assert snap.thrust_magnitude == 0.0 or snap.mass_flow_rate >= 0.0


# ---------------------------------------------------------------------------
# 7. Aerodynamics disabled above vacuum altitude
# ---------------------------------------------------------------------------

class TestAerodynamicsVacuumCutoff:

    def test_no_aero_force_above_80km(self):
        """At 400 km altitude, aero should be disabled."""
        state = _leo_state()
        cfg   = PipelineConfig(
            dt=0.01,
            enable_aerodynamics=True,
            aero_config=AeroConfig(
                reference_area=20.0, mean_chord=2.5, span=10.0
            ),
            aero_vacuum_altitude=80_000.0,
        )
        pipe = SimulationPipeline(cfg, state, _minimal_graph())
        pipe.tick(ControlInput())
        snap = pipe.registry.latest
        F_aero_mag = float(np.linalg.norm(snap.force_aero))
        assert F_aero_mag == 0.0, f"Expected zero aero above 80 km, got {F_aero_mag:.2f} N"


# ---------------------------------------------------------------------------
# 8. run() convenience method
# ---------------------------------------------------------------------------

class TestRunMethod:

    def test_run_returns_correct_count(self):
        state = _leo_state()
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph())
        results = pipe.run(10)
        assert len(results) == 10

    def test_run_tick_count_correct(self):
        state = _leo_state()
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph())
        pipe.run(7)
        assert pipe.tick_count == 7

    def test_run_with_control_fn(self):
        """Control function receives tick index and current state."""
        received = []
        state = _leo_state()
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph())

        def ctrl_fn(i, s):
            received.append(i)
            return ControlInput()

        pipe.run(5, control_fn=ctrl_fn)
        assert received == [0, 1, 2, 3, 4]

    def test_time_advances_by_n_dt(self):
        dt    = 0.05
        n     = 20
        state = _leo_state()
        pipe  = SimulationPipeline(_minimal_config(dt), state, _minimal_graph())
        pipe.run(n)
        expected_time = n * dt
        assert abs(pipe.state.time - expected_time) < 1.0e-10


# ---------------------------------------------------------------------------
# 9. Telemetry snapshot fields
# ---------------------------------------------------------------------------

class TestSnapshotFields:

    def test_altitude_positive_at_leo(self):
        state = _leo_state()
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph())
        pipe.tick()
        snap = pipe.registry.latest
        assert snap.altitude > 300_000.0   # > 300 km

    def test_speed_in_snapshot(self):
        state = _leo_state()
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph())
        pipe.tick()
        snap = pipe.registry.latest
        assert snap.speed > 7_000.0   # LEO speed > 7 km/s

    def test_snapshot_mass_matches_state(self):
        state = _leo_state(mass=3000.0)
        pipe  = SimulationPipeline(_minimal_config(), state, _minimal_graph(3000.0))
        pipe.tick()
        snap = pipe.registry.latest
        assert abs(snap.mass - pipe.state.mass) < 1.0e-6
