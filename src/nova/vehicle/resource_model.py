"""
nova.vehicle.resource_model
============================
Onboard consumable resource tracker for Project NOVA.

Architectural role
------------------
Phase 8 — Vehicle Resource Models.
Pipeline stage: Stage 9 (Component Updates). Called after RK4 integration
to deplete propellant from the active tank, accumulate structural fatigue
from dynamic pressure exposure, and track electrical power draw.

I/O contract
------------
Input  : ResourceState (current totals), PropulsionState (mass_flow_rate),
         dt [s], dynamic_pressure [Pa], power_draw_W [W per consumer]
Output : ResourceState (updated totals, frozen dataclass)
         ResourceStatus (warning flags, frozen dataclass)

Physical basis
--------------
Propellant depletion:
    dm/dt = −ṁ  (kg s⁻¹ from PropulsionState.mass_flow_rate)
    m_prop(t+dt) = max(0, m_prop(t) − ṁ · dt)

Fatigue accumulation:
    Structural fatigue is proportional to dynamic-pressure exposure.
    A simple linear damage model is used:
        D(t+dt) = D(t) + (q_∞ / q_ref) · dt
    where q_ref is a reference dynamic pressure for the vehicle design.
    D is dimensionless damage [0, ∞). Failure is flagged when D ≥ D_limit.
    This is a simplified Miner's-rule-inspired accumulator — not a full
    S-N curve model. Suitable for simulation alerting and phase gating.

Electrical power budget:
    Power is modelled as a fixed draw per consumer, checked against the
    total generation capacity. No energy storage is modelled (all-or-nothing
    per tick). Net power margin = P_gen − Σ P_consumers [W].

All resource states are frozen dataclasses — the model is purely functional
(old state in, new state out). No side effects.

References
----------
- Sutton & Biblarz, "Rocket Propulsion Elements", 8th ed., §2.1
- Miner, M.A. "Cumulative Damage in Fatigue", J. Appl. Mech. (1945)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from nova.physics.propulsion import PropulsionState

# ---------------------------------------------------------------------------
# ResourceConfig — vehicle-level resource parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TankConfig:
    """
    Configuration for a single propellant tank.

    Attributes
    ----------
    tank_id : str
        Unique identifier (e.g. "lox", "kerosene", "monoprop").
    capacity_kg : float
        Maximum propellant capacity [kg]. Must be positive.
    initial_mass_kg : float
        Propellant mass at mission start [kg]. Must be in [0, capacity_kg].
    is_pressurant : bool
        If True, this tank holds pressurant gas rather than propellant.
        Depletion uses a different (slower) rate model when True.
        Default False.
    """

    tank_id: str
    capacity_kg: float
    initial_mass_kg: float
    is_pressurant: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.tank_id, str) or not self.tank_id.strip():
            raise ValueError("tank_id must be a non-empty string")
        cap = float(self.capacity_kg)
        if cap <= 0.0:
            raise ValueError(f"capacity_kg must be positive; got {cap:.6g}")
        object.__setattr__(self, "capacity_kg", cap)

        init = float(self.initial_mass_kg)
        if init < 0.0:
            raise ValueError(
                f"initial_mass_kg must be non-negative; got {init:.6g}"
            )
        if init > cap:
            raise ValueError(
                f"initial_mass_kg ({init:.6g}) exceeds capacity_kg ({cap:.6g})"
            )
        object.__setattr__(self, "initial_mass_kg", init)


@dataclass(frozen=True)
class FatigueConfig:
    """
    Parameters for the structural fatigue accumulation model.

    Attributes
    ----------
    reference_dynamic_pressure_pa : float
        Reference dynamic pressure q_ref [Pa] at which D accumulates at 1/s.
        Must be positive.
    damage_limit : float
        Dimensionless damage threshold D_limit. When the accumulated damage
        D ≥ D_limit, a structural fatigue warning is raised. Must be positive.
    """

    reference_dynamic_pressure_pa: float
    damage_limit: float

    def __post_init__(self) -> None:
        q_ref = float(self.reference_dynamic_pressure_pa)
        if q_ref <= 0.0:
            raise ValueError(
                f"reference_dynamic_pressure_pa must be positive; got {q_ref:.6g}"
            )
        object.__setattr__(self, "reference_dynamic_pressure_pa", q_ref)
        d_lim = float(self.damage_limit)
        if d_lim <= 0.0:
            raise ValueError(
                f"damage_limit must be positive; got {d_lim:.6g}"
            )
        object.__setattr__(self, "damage_limit", d_lim)


@dataclass(frozen=True)
class PowerConfig:
    """
    Electrical power system configuration.

    Attributes
    ----------
    generation_capacity_w : float
        Total electrical power generation capacity [W]. Must be positive.
    consumers : dict[str, float]
        Named power consumers and their draw [W]. Values must be non-negative.
    """

    generation_capacity_w: float
    consumers: Dict[str, float]

    def __post_init__(self) -> None:
        gen = float(self.generation_capacity_w)
        if gen <= 0.0:
            raise ValueError(
                f"generation_capacity_w must be positive; got {gen:.6g}"
            )
        object.__setattr__(self, "generation_capacity_w", gen)

        consumers = dict(self.consumers)
        for name, draw in consumers.items():
            if not isinstance(name, str):
                raise TypeError("consumer names must be strings")
            draw_f = float(draw)
            if draw_f < 0.0:
                raise ValueError(
                    f"consumer '{name}' draw must be non-negative; got {draw_f:.6g}"
                )
            consumers[name] = draw_f
        object.__setattr__(self, "consumers", consumers)

    @property
    def total_draw_w(self) -> float:
        """Total power draw from all consumers [W]."""
        return sum(self.consumers.values())

    @property
    def power_margin_w(self) -> float:
        """Available power margin [W]. Negative = deficit."""
        return self.generation_capacity_w - self.total_draw_w


@dataclass(frozen=True)
class ResourceConfig:
    """
    Complete vehicle resource system configuration.

    Attributes
    ----------
    tanks : list[TankConfig]
        All propellant tanks aboard the vehicle. Must be non-empty.
    fatigue : FatigueConfig
        Structural fatigue accumulation parameters.
    power : PowerConfig
        Electrical power system parameters.
    primary_tank_id : str
        ID of the tank that feeds the main engine. Must match a tank_id
        in tanks. The propulsion mass flow is drawn from this tank first;
        if exhausted, the engine is considered out of propellant.
    """

    tanks: List[TankConfig]
    fatigue: FatigueConfig
    power: PowerConfig
    primary_tank_id: str

    def __post_init__(self) -> None:
        tanks = list(self.tanks)
        if not tanks:
            raise ValueError("tanks must contain at least one TankConfig")
        for t in tanks:
            if not isinstance(t, TankConfig):
                raise TypeError("All entries in tanks must be TankConfig instances")
        object.__setattr__(self, "tanks", tanks)

        if not isinstance(self.fatigue, FatigueConfig):
            raise TypeError("fatigue must be a FatigueConfig")
        if not isinstance(self.power, PowerConfig):
            raise TypeError("power must be a PowerConfig")

        pid = str(self.primary_tank_id)
        tank_ids = {t.tank_id for t in tanks}
        if pid not in tank_ids:
            raise ValueError(
                f"primary_tank_id '{pid}' not found in tanks "
                f"(available: {sorted(tank_ids)})"
            )
        object.__setattr__(self, "primary_tank_id", pid)

    def get_tank(self, tank_id: str) -> TankConfig:
        """Return the TankConfig with the given tank_id."""
        for t in self.tanks:
            if t.tank_id == tank_id:
                return t
        raise KeyError(f"tank_id '{tank_id}' not found in ResourceConfig")


# ---------------------------------------------------------------------------
# ResourceState — per-tick consumable state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceState:
    """
    Immutable snapshot of all onboard resource levels at one simulation tick.

    Attributes
    ----------
    tank_masses_kg : dict[str, float]
        Current propellant mass per tank [kg]. Keys match TankConfig.tank_id.
    fatigue_damage : float
        Accumulated structural fatigue damage [dimensionless]. Range [0, ∞).
    power_draw_w : float
        Total electrical power draw this tick [W].
    power_generation_w : float
        Available electrical generation capacity [W].
    mission_elapsed_time : float
        Mission elapsed time [s]. Non-negative.
    """

    tank_masses_kg: Dict[str, float]
    fatigue_damage: float
    power_draw_w: float
    power_generation_w: float
    mission_elapsed_time: float

    def __post_init__(self) -> None:
        tanks = dict(self.tank_masses_kg)
        for tid, mass in tanks.items():
            m = float(mass)
            if m < 0.0:
                raise ValueError(
                    f"tank '{tid}' mass must be non-negative; got {m:.6g}"
                )
            tanks[tid] = m
        object.__setattr__(self, "tank_masses_kg", tanks)

        fd = float(self.fatigue_damage)
        if fd < 0.0:
            raise ValueError(
                f"fatigue_damage must be non-negative; got {fd:.6g}"
            )
        object.__setattr__(self, "fatigue_damage", fd)

        pd = float(self.power_draw_w)
        if pd < 0.0:
            raise ValueError(
                f"power_draw_w must be non-negative; got {pd:.6g}"
            )
        object.__setattr__(self, "power_draw_w", pd)

        pg = float(self.power_generation_w)
        if pg < 0.0:
            raise ValueError(
                f"power_generation_w must be non-negative; got {pg:.6g}"
            )
        object.__setattr__(self, "power_generation_w", pg)

        t = float(self.mission_elapsed_time)
        if t < 0.0:
            raise ValueError(
                f"mission_elapsed_time must be non-negative; got {t:.6g}"
            )
        object.__setattr__(self, "mission_elapsed_time", t)

    @property
    def total_propellant_kg(self) -> float:
        """Sum of all tank masses [kg]."""
        return sum(self.tank_masses_kg.values())

    @property
    def power_margin_w(self) -> float:
        """Net power margin = generation − draw [W]. Negative = deficit."""
        return self.power_generation_w - self.power_draw_w

    @property
    def power_deficit(self) -> bool:
        """True when draw exceeds generation capacity."""
        return self.power_draw_w > self.power_generation_w

    def primary_tank_mass(self, primary_tank_id: str) -> float:
        """Return the mass [kg] in the primary propellant tank."""
        return self.tank_masses_kg.get(primary_tank_id, 0.0)

    def __repr__(self) -> str:
        prop = self.total_propellant_kg
        fd = self.fatigue_damage
        pm = self.power_margin_w
        t = self.mission_elapsed_time
        return (
            f"ResourceState(t={t:.1f}s, prop={prop:.2f}kg, "
            f"fatigue={fd:.4f}, P_margin={pm:.1f}W)"
        )


# ---------------------------------------------------------------------------
# ResourceStatus — warning / alert flags
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceStatus:
    """
    Flags summarising resource health at the current tick.

    Attributes
    ----------
    propellant_low : bool
        True when primary tank mass < 10% of its capacity.
    propellant_exhausted : bool
        True when primary tank mass == 0.
    fatigue_warning : bool
        True when fatigue_damage ≥ 50% of damage_limit.
    fatigue_critical : bool
        True when fatigue_damage ≥ damage_limit.
    power_deficit : bool
        True when electrical draw exceeds generation capacity.
    any_warning : bool
        True if any of the above flags is set.
    """

    propellant_low: bool
    propellant_exhausted: bool
    fatigue_warning: bool
    fatigue_critical: bool
    power_deficit: bool

    def __post_init__(self) -> None:
        for f in ("propellant_low", "propellant_exhausted",
                  "fatigue_warning", "fatigue_critical", "power_deficit"):
            object.__setattr__(self, f, bool(getattr(self, f)))

    @property
    def any_warning(self) -> bool:
        """True if any resource warning or critical flag is active."""
        return (
            self.propellant_low
            or self.propellant_exhausted
            or self.fatigue_warning
            or self.fatigue_critical
            or self.power_deficit
        )

    def __repr__(self) -> str:
        flags = []
        if self.propellant_exhausted:
            flags.append("PROP_EXHAUSTED")
        elif self.propellant_low:
            flags.append("PROP_LOW")
        if self.fatigue_critical:
            flags.append("FATIGUE_CRITICAL")
        elif self.fatigue_warning:
            flags.append("FATIGUE_WARNING")
        if self.power_deficit:
            flags.append("POWER_DEFICIT")
        status = ", ".join(flags) if flags else "OK"
        return f"ResourceStatus({status})"


# ---------------------------------------------------------------------------
# Core update functions (pure, stateless)
# ---------------------------------------------------------------------------

def update_propellant(
    tank_masses_kg: Dict[str, float],
    primary_tank_id: str,
    mass_flow_rate_kg_s: float,
    dt: float,
) -> Dict[str, float]:
    """
    Deplete propellant from the primary tank by one timestep.

    Parameters
    ----------
    tank_masses_kg : dict[str, float]
        Current masses in all tanks [kg].
    primary_tank_id : str
        Tank ID that feeds the main engine.
    mass_flow_rate_kg_s : float
        Mass flow rate [kg s⁻¹]. Must be non-negative.
    dt : float
        Timestep [s]. Must be positive.

    Returns
    -------
    dict[str, float]
        Updated tank masses. The primary tank is reduced by
        mass_flow_rate_kg_s × dt, floored at 0.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt:.6g}")
    if mass_flow_rate_kg_s < 0.0:
        raise ValueError(
            f"mass_flow_rate_kg_s must be non-negative; got {mass_flow_rate_kg_s:.6g}"
        )

    updated = dict(tank_masses_kg)
    current = updated.get(primary_tank_id, 0.0)
    depleted = mass_flow_rate_kg_s * dt
    updated[primary_tank_id] = max(0.0, current - depleted)
    return updated


def update_fatigue(
    current_damage: float,
    dynamic_pressure_pa: float,
    config: FatigueConfig,
    dt: float,
) -> float:
    """
    Accumulate structural fatigue damage over one timestep.

    Parameters
    ----------
    current_damage : float
        Accumulated damage at start of tick [dimensionless].
    dynamic_pressure_pa : float
        Current dynamic pressure q_∞ [Pa]. Clamped to [0, ∞).
    config : FatigueConfig
        Reference parameters.
    dt : float
        Timestep [s]. Must be positive.

    Returns
    -------
    float
        Updated damage [dimensionless].
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt:.6g}")
    q = max(0.0, float(dynamic_pressure_pa))
    increment = (q / config.reference_dynamic_pressure_pa) * dt
    return current_damage + increment


def evaluate_status(
    state: ResourceState,
    config: ResourceConfig,
) -> ResourceStatus:
    """
    Evaluate resource health flags from a ResourceState.

    Parameters
    ----------
    state : ResourceState
        Current resource levels.
    config : ResourceConfig
        Vehicle resource configuration (for thresholds).

    Returns
    -------
    ResourceStatus
    """
    primary_tank = config.get_tank(config.primary_tank_id)
    primary_mass = state.primary_tank_mass(config.primary_tank_id)
    capacity = primary_tank.capacity_kg

    prop_exhausted = primary_mass <= 0.0
    prop_low = (not prop_exhausted) and (primary_mass < 0.10 * capacity)

    fatigue_warning = state.fatigue_damage >= 0.5 * config.fatigue.damage_limit
    fatigue_critical = state.fatigue_damage >= config.fatigue.damage_limit

    power_def = state.power_deficit

    return ResourceStatus(
        propellant_low=prop_low,
        propellant_exhausted=prop_exhausted,
        fatigue_warning=fatigue_warning,
        fatigue_critical=fatigue_critical,
        power_deficit=power_def,
    )


# ---------------------------------------------------------------------------
# ResourceModel — high-level stateful tracker
# ---------------------------------------------------------------------------

class ResourceModel:
    """
    Stateful resource tracker for propellant, fatigue, and electrical power.

    Wraps the pure update functions and holds the current ResourceState.
    Called once per tick by the pipeline's Stage 9 (Component Updates).

    Parameters
    ----------
    config : ResourceConfig
        Vehicle resource configuration.
    """

    def __init__(self, config: ResourceConfig) -> None:
        if not isinstance(config, ResourceConfig):
            raise TypeError("config must be a ResourceConfig")
        self._config = config

        # Build initial state from config
        tank_masses = {t.tank_id: t.initial_mass_kg for t in config.tanks}
        self._state = ResourceState(
            tank_masses_kg=tank_masses,
            fatigue_damage=0.0,
            power_draw_w=config.power.total_draw_w,
            power_generation_w=config.power.generation_capacity_w,
            mission_elapsed_time=0.0,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ResourceConfig:
        """Read-only resource configuration."""
        return self._config

    @property
    def state(self) -> ResourceState:
        """Current ResourceState (frozen snapshot)."""
        return self._state

    @property
    def primary_propellant_kg(self) -> float:
        """Remaining propellant mass in the primary tank [kg]."""
        return self._state.primary_tank_mass(self._config.primary_tank_id)

    @property
    def total_propellant_kg(self) -> float:
        """Total propellant mass across all tanks [kg]."""
        return self._state.total_propellant_kg

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        propulsion_state: PropulsionState,
        dynamic_pressure_pa: float,
        dt: float,
    ) -> tuple[ResourceState, ResourceStatus]:
        """
        Advance the resource model by one simulation timestep.

        Parameters
        ----------
        propulsion_state : PropulsionState
            Output of compute_propulsion() for this tick. Provides
            mass_flow_rate [kg s⁻¹].
        dynamic_pressure_pa : float
            Dynamic pressure q_∞ [Pa] from the atmosphere model.
        dt : float
            Timestep [s]. Must be positive.

        Returns
        -------
        new_state : ResourceState
            Updated resource levels (frozen).
        status : ResourceStatus
            Health flags evaluated against the new state.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive; got {dt:.6g}")

        # 1. Deplete propellant
        new_tanks = update_propellant(
            tank_masses_kg=self._state.tank_masses_kg,
            primary_tank_id=self._config.primary_tank_id,
            mass_flow_rate_kg_s=propulsion_state.mass_flow_rate,
            dt=dt,
        )

        # 2. Accumulate fatigue
        new_damage = update_fatigue(
            current_damage=self._state.fatigue_damage,
            dynamic_pressure_pa=dynamic_pressure_pa,
            config=self._config.fatigue,
            dt=dt,
        )

        # 3. Evaluate power (static per tick — consumers are fixed in config)
        new_draw = self._config.power.total_draw_w
        new_gen = self._config.power.generation_capacity_w

        # 4. Advance time
        new_t = self._state.mission_elapsed_time + dt

        new_state = ResourceState(
            tank_masses_kg=new_tanks,
            fatigue_damage=new_damage,
            power_draw_w=new_draw,
            power_generation_w=new_gen,
            mission_elapsed_time=new_t,
        )

        self._state = new_state
        status = evaluate_status(new_state, self._config)
        return new_state, status

    def __repr__(self) -> str:
        return (
            f"ResourceModel(prop={self.primary_propellant_kg:.2f}kg, "
            f"fatigue={self._state.fatigue_damage:.4f}, "
            f"t={self._state.mission_elapsed_time:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def simple_resource_config(
    propellant_kg: float,
    primary_tank_id: str = "main",
    reference_q_pa: float = 50_000.0,
    damage_limit: float = 3_600.0,
    power_generation_w: float = 5_000.0,
    power_consumers: Optional[Dict[str, float]] = None,
) -> ResourceConfig:
    """
    Build a minimal single-tank ResourceConfig for common use cases.

    Parameters
    ----------
    propellant_kg : float
        Initial (and maximum) propellant mass [kg]. Must be positive.
    primary_tank_id : str
        ID for the single propellant tank. Default "main".
    reference_q_pa : float
        Fatigue reference dynamic pressure [Pa]. Default 50 kPa
        (representative of max-q for a sounding rocket).
    damage_limit : float
        Fatigue damage limit [dimensionless]. Default 3600 (1 hour at q_ref).
    power_generation_w : float
        Power generation capacity [W]. Default 5 kW.
    power_consumers : dict[str, float] or None
        Named power consumers and their draws [W]. If None, uses a
        default set (avionics 200 W, sensors 100 W, actuators 150 W).

    Returns
    -------
    ResourceConfig
    """
    if power_consumers is None:
        power_consumers = {
            "avionics": 200.0,
            "sensors": 100.0,
            "actuators": 150.0,
        }

    return ResourceConfig(
        tanks=[
            TankConfig(
                tank_id=primary_tank_id,
                capacity_kg=propellant_kg,
                initial_mass_kg=propellant_kg,
            )
        ],
        fatigue=FatigueConfig(
            reference_dynamic_pressure_pa=reference_q_pa,
            damage_limit=damage_limit,
        ),
        power=PowerConfig(
            generation_capacity_w=power_generation_w,
            consumers=power_consumers,
        ),
        primary_tank_id=primary_tank_id,
    )
