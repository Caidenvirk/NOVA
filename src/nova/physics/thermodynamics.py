"""
nova.physics.thermodynamics
============================
Aerodynamic skin heating and radiative cooling model for Project NOVA.

Architectural role
------------------
Phase 10 — Thermodynamics Engine.
Pipeline stage: Stage 8. Receives AtmosphericState (density, temperature)
and vehicle airspeed from Stage 7, plus surface geometry, and returns the
net heat flux per panel [W m⁻²] and updated panel wall temperatures.

I/O contract
------------
Input  : AtmosphericState (ρ, T_∞), airspeed [m s⁻¹], SurfacePanel list,
         dt [s]
Output : ThermalState (frozen dataclass) — heat flux and wall temperature
         per panel; ThermalSnapshot for telemetry

Physical basis
--------------
Convective heating (Stanton-number model):
    Q̇_conv = St · ρ · v³ / 2   [W m⁻²]

where St is the Stanton number (dimensionless heat transfer coefficient),
ρ is the free-stream density [kg m⁻³], and v is the airspeed [m s⁻¹].

This is the standard engineering approximation for stagnation-point heating
and is used in early trajectory analysis for reentry vehicles:
    St_stag ≈ 1.83 × 10⁻⁴ / √R_nose  (Chapman's formula)
where R_nose is the nose radius [m].

Radiative cooling (Stefan-Boltzmann):
    Q̇_rad = ε · σ · T_wall⁴   [W m⁻²]

where ε is the surface emissivity (dimensionless), σ is the Stefan-Boltzmann
constant, and T_wall is the wall temperature [K].

Solar absorption (optional):
    Q̇_solar = α · G_solar · cos θ   [W m⁻²]

where α is the solar absorptivity, G_solar is the solar irradiance (~1361
W m⁻² at 1 AU), and θ is the angle of incidence relative to the surface
normal.

Net heat flux balance:
    Q̇_net = Q̇_conv + Q̇_solar − Q̇_rad   [W m⁻²]

Wall temperature evolution (lumped thermal mass):
    dT_wall/dt = Q̇_net · A / (m_panel · cp)   [K s⁻¹]

where A is the panel area [m²], m_panel is the panel mass [kg], and cp is
the specific heat capacity [J kg⁻¹ K⁻¹].

Forward Euler integration is used for T_wall (accuracy adequate at dt ≤ 0.1 s
for thermal timescales of minutes to hours).

References
----------
- Chapman, D.R. "An Approximate Analytical Method for Studying Entry into
  Planetary Atmospheres." NACA TN 4276, 1958.
- Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer", 7th ed., §7.
- Tauber, M.E. "A Review of High-Speed Convective Heat Transfer Computation
  Methods." NASA TP-2914, 1989.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from nova.core.constants import SIGMA_SB
from nova.physics.atmosphere import AtmosphericState

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_SOLAR_IRRADIANCE_W_M2: float = 1361.0   # Solar constant at 1 AU [W m⁻²]
_T_SPACE_K: float = 2.7                  # Cosmic background temperature [K]
_MIN_WALL_TEMP_K: float = 2.7            # Floor: cannot drop below space temp


# ---------------------------------------------------------------------------
# SurfacePanel — geometry and material for one thermal panel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceConfig:
    """
    Thermal material and geometry properties for one surface panel.

    Attributes
    ----------
    panel_id : str
        Unique identifier (e.g. "nose_cap", "windward", "leeward").
    area_m2 : float
        Exposed surface area [m²]. Must be positive.
    mass_kg : float
        Thermal mass of the panel [kg]. Must be positive.
    specific_heat_j_kg_k : float
        Specific heat capacity cp [J kg⁻¹ K⁻¹]. Must be positive.
    emissivity : float
        Surface emissivity ε ∈ (0, 1]. Dimensionless.
    absorptivity : float
        Solar absorptivity α ∈ [0, 1]. Dimensionless. Default 0.9.
    stanton_number : float
        Stanton number St for convective heating. Must be positive.
        Default 1.0e-4 (typical for a blunt reentry body).
    initial_temp_k : float
        Initial wall temperature [K]. Default 293.15 K (20°C).
    """

    panel_id: str
    area_m2: float
    mass_kg: float
        
    specific_heat_j_kg_k: float
    emissivity: float
    absorptivity: float = 0.9
    stanton_number: float = 1.0e-4
    initial_temp_k: float = 293.15

    def __post_init__(self) -> None:
        if not isinstance(self.panel_id, str) or not self.panel_id.strip():
            raise ValueError("panel_id must be a non-empty string")

        for attr, label in (
            ("area_m2", "area_m2"),
            ("mass_kg", "mass_kg"),
            ("specific_heat_j_kg_k", "specific_heat_j_kg_k"),
            ("stanton_number", "stanton_number"),
        ):
            val = float(getattr(self, attr))
            if val <= 0.0:
                raise ValueError(f"{label} must be positive; got {val:.6g}")
            object.__setattr__(self, attr, val)

        eps = float(self.emissivity)
        if not (0.0 < eps <= 1.0):
            raise ValueError(
                f"emissivity must be in (0, 1]; got {eps:.6g}"
            )
        object.__setattr__(self, "emissivity", eps)

        alp = float(self.absorptivity)
        if not (0.0 <= alp <= 1.0):
            raise ValueError(
                f"absorptivity must be in [0, 1]; got {alp:.6g}"
            )
        object.__setattr__(self, "absorptivity", alp)

        t0 = float(self.initial_temp_k)
        if t0 < _MIN_WALL_TEMP_K:
            raise ValueError(
                f"initial_temp_k must be ≥ {_MIN_WALL_TEMP_K} K; got {t0:.6g}"
            )
        object.__setattr__(self, "initial_temp_k", t0)

    @property
    def thermal_mass(self) -> float:
        """Thermal mass = m · cp [J K⁻¹]."""
        return self.mass_kg * self.specific_heat_j_kg_k


# ---------------------------------------------------------------------------
# PanelThermalState — per-panel thermal state (one tick)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PanelThermalState:
    """
    Immutable per-panel thermal state at one simulation tick.

    Attributes
    ----------
    panel_id : str
        Matches SurfaceConfig.panel_id.
    wall_temp_k : float
        Current wall temperature [K].
    conv_flux_w_m2 : float
        Convective heat flux [W m⁻²] (always ≥ 0).
    rad_flux_w_m2 : float
        Radiative cooling flux [W m⁻²] (always ≥ 0).
    solar_flux_w_m2 : float
        Absorbed solar flux [W m⁻²] (always ≥ 0).
    net_flux_w_m2 : float
        Net heat flux = conv + solar − rad [W m⁻²]. May be negative.
    temp_rate_k_s : float
        Rate of temperature change dT/dt [K s⁻¹].
    """

    panel_id: str
    wall_temp_k: float
    conv_flux_w_m2: float
    rad_flux_w_m2: float
    solar_flux_w_m2: float
    net_flux_w_m2: float
    temp_rate_k_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", str(self.panel_id))
        for attr in ("wall_temp_k", "conv_flux_w_m2", "rad_flux_w_m2",
                     "solar_flux_w_m2", "net_flux_w_m2", "temp_rate_k_s"):
            object.__setattr__(self, attr, float(getattr(self, attr)))

        if self.wall_temp_k < _MIN_WALL_TEMP_K:
            raise ValueError(
                f"wall_temp_k must be ≥ {_MIN_WALL_TEMP_K} K; "
                f"got {self.wall_temp_k:.6g}"
            )
        if self.conv_flux_w_m2 < 0.0:
            raise ValueError(
                f"conv_flux_w_m2 must be non-negative; got {self.conv_flux_w_m2:.6g}"
            )
        if self.rad_flux_w_m2 < 0.0:
            raise ValueError(
                f"rad_flux_w_m2 must be non-negative; got {self.rad_flux_w_m2:.6g}"
            )
        if self.solar_flux_w_m2 < 0.0:
            raise ValueError(
                f"solar_flux_w_m2 must be non-negative; got {self.solar_flux_w_m2:.6g}"
            )

    def __repr__(self) -> str:
        return (
            f"PanelThermalState(id={self.panel_id!r}, "
            f"T={self.wall_temp_k:.1f}K, "
            f"Q_net={self.net_flux_w_m2:.1f} W/m²)"
        )


# ---------------------------------------------------------------------------
# ThermalSnapshot — vehicle-level thermal state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThermalSnapshot:
    """
    Immutable vehicle-level thermal summary at one simulation tick.

    Attributes
    ----------
    panels : list[PanelThermalState]
        Thermal state of each surface panel.
    max_wall_temp_k : float
        Maximum wall temperature across all panels [K].
    total_conv_power_w : float
        Total convective heating power = Σ Q̇_conv · A [W].
    total_rad_power_w : float
        Total radiative cooling power = Σ Q̇_rad · A [W].
    total_net_power_w : float
        Net thermal power deposited in vehicle [W].
    mission_time : float
        Mission elapsed time [s].
    """

    panels: List[PanelThermalState]
    max_wall_temp_k: float
    total_conv_power_w: float
    total_rad_power_w: float
    total_net_power_w: float
    mission_time: float

    def __post_init__(self) -> None:
        panels = list(self.panels)
        object.__setattr__(self, "panels", panels)
        object.__setattr__(self, "max_wall_temp_k", float(self.max_wall_temp_k))
        object.__setattr__(self, "total_conv_power_w", float(self.total_conv_power_w))
        object.__setattr__(self, "total_rad_power_w", float(self.total_rad_power_w))
        object.__setattr__(self, "total_net_power_w", float(self.total_net_power_w))
        t = float(self.mission_time)
        if t < 0.0:
            raise ValueError(f"mission_time must be non-negative; got {t:.6g}")
        object.__setattr__(self, "mission_time", t)

    def get_panel(self, panel_id: str) -> PanelThermalState:
        """Return the PanelThermalState for *panel_id*."""
        for p in self.panels:
            if p.panel_id == panel_id:
                return p
        raise KeyError(f"panel_id '{panel_id}' not found in ThermalSnapshot")

    def __repr__(self) -> str:
        return (
            f"ThermalSnapshot(t={self.mission_time:.2f}s, "
            f"T_max={self.max_wall_temp_k:.1f}K, "
            f"Q_net={self.total_net_power_w:.1f}W, "
            f"panels={len(self.panels)})"
        )


# ---------------------------------------------------------------------------
# Pure heat-flux functions
# ---------------------------------------------------------------------------

def convective_flux(
    density_kg_m3: float,
    airspeed_m_s: float,
    stanton_number: float,
) -> float:
    """
    Compute the convective heat flux using the Stanton-number model.

    Q̇_conv = St · ρ · v³ / 2   [W m⁻²]

    Parameters
    ----------
    density_kg_m3 : float
        Free-stream air density ρ [kg m⁻³]. Must be non-negative.
    airspeed_m_s : float
        Vehicle airspeed v [m s⁻¹]. Must be non-negative.
    stanton_number : float
        Stanton number St (dimensionless). Must be positive.

    Returns
    -------
    float
        Convective heat flux [W m⁻²]. Always ≥ 0.
    """
    if density_kg_m3 < 0.0:
        raise ValueError(
            f"density_kg_m3 must be non-negative; got {density_kg_m3:.6g}"
        )
    if airspeed_m_s < 0.0:
        raise ValueError(
            f"airspeed_m_s must be non-negative; got {airspeed_m_s:.6g}"
        )
    if stanton_number <= 0.0:
        raise ValueError(
            f"stanton_number must be positive; got {stanton_number:.6g}"
        )
    return stanton_number * density_kg_m3 * (airspeed_m_s ** 3) / 2.0


def radiative_flux(
    wall_temp_k: float,
    emissivity: float,
) -> float:
    """
    Compute the radiative heat flux from a surface.

    Q̇_rad = ε · σ · T_wall⁴   [W m⁻²]

    Parameters
    ----------
    wall_temp_k : float
        Wall temperature [K]. Must be ≥ 2.7 K.
    emissivity : float
        Surface emissivity ε ∈ (0, 1].

    Returns
    -------
    float
        Radiative flux [W m⁻²]. Always ≥ 0.
    """
    if wall_temp_k < _MIN_WALL_TEMP_K:
        raise ValueError(
            f"wall_temp_k must be ≥ {_MIN_WALL_TEMP_K} K; got {wall_temp_k:.6g}"
        )
    if not (0.0 < emissivity <= 1.0):
        raise ValueError(
            f"emissivity must be in (0, 1]; got {emissivity:.6g}"
        )
    return emissivity * SIGMA_SB * (wall_temp_k ** 4)


def solar_flux(
    absorptivity: float,
    cos_incidence: float = 1.0,
    irradiance_w_m2: float = _SOLAR_IRRADIANCE_W_M2,
) -> float:
    """
    Compute the absorbed solar heat flux.

    Q̇_solar = α · G_solar · max(0, cos θ)   [W m⁻²]

    Parameters
    ----------
    absorptivity : float
        Solar absorptivity α ∈ [0, 1].
    cos_incidence : float
        Cosine of angle of incidence θ (between sun vector and surface normal).
        Clamped to [0, 1]: negative means sun is behind the panel.
        Default 1.0 (normal incidence).
    irradiance_w_m2 : float
        Incident solar irradiance [W m⁻²]. Default 1361 W m⁻² (1 AU).

    Returns
    -------
    float
        Absorbed solar flux [W m⁻²]. Always ≥ 0.
    """
    if not (0.0 <= absorptivity <= 1.0):
        raise ValueError(
            f"absorptivity must be in [0, 1]; got {absorptivity:.6g}"
        )
    if irradiance_w_m2 < 0.0:
        raise ValueError(
            f"irradiance_w_m2 must be non-negative; got {irradiance_w_m2:.6g}"
        )
    cos_i = max(0.0, float(cos_incidence))
    return absorptivity * irradiance_w_m2 * cos_i


def net_flux(
    q_conv: float,
    q_rad: float,
    q_solar: float = 0.0,
) -> float:
    """
    Net heat flux deposited into the surface.

    Q̇_net = Q̇_conv + Q̇_solar − Q̇_rad   [W m⁻²]

    Parameters
    ----------
    q_conv : float
        Convective flux [W m⁻²]. Must be ≥ 0.
    q_rad : float
        Radiative flux [W m⁻²]. Must be ≥ 0.
    q_solar : float
        Solar flux [W m⁻²]. Must be ≥ 0. Default 0.

    Returns
    -------
    float
        Net flux [W m⁻²]. May be negative (net cooling).
    """
    if q_conv < 0.0:
        raise ValueError(f"q_conv must be non-negative; got {q_conv:.6g}")
    if q_rad < 0.0:
        raise ValueError(f"q_rad must be non-negative; got {q_rad:.6g}")
    if q_solar < 0.0:
        raise ValueError(f"q_solar must be non-negative; got {q_solar:.6g}")
    return q_conv + q_solar - q_rad


def temperature_rate(
    net_flux_w_m2: float,
    area_m2: float,
    thermal_mass_j_k: float,
) -> float:
    """
    Rate of temperature change for a lumped thermal mass.

    dT/dt = Q̇_net · A / (m · cp) = Q̇_net · A / C_th   [K s⁻¹]

    Parameters
    ----------
    net_flux_w_m2 : float
        Net heat flux [W m⁻²].
    area_m2 : float
        Surface area [m²]. Must be positive.
    thermal_mass_j_k : float
        Thermal mass C_th = m · cp [J K⁻¹]. Must be positive.

    Returns
    -------
    float
        Temperature rate [K s⁻¹]. May be negative.
    """
    if area_m2 <= 0.0:
        raise ValueError(f"area_m2 must be positive; got {area_m2:.6g}")
    if thermal_mass_j_k <= 0.0:
        raise ValueError(
            f"thermal_mass_j_k must be positive; got {thermal_mass_j_k:.6g}"
        )
    return (net_flux_w_m2 * area_m2) / thermal_mass_j_k


def equilibrium_temperature(
    q_conv: float,
    q_solar: float,
    emissivity: float,
) -> float:
    """
    Compute the radiative equilibrium wall temperature.

    At equilibrium Q̇_conv + Q̇_solar = Q̇_rad = ε · σ · T_eq⁴

    T_eq = ((Q̇_conv + Q̇_solar) / (ε · σ))^(1/4)   [K]

    If the combined input flux is zero, returns _T_SPACE_K (2.7 K).

    Parameters
    ----------
    q_conv : float
        Convective flux [W m⁻²]. Must be ≥ 0.
    q_solar : float
        Solar flux [W m⁻²]. Must be ≥ 0.
    emissivity : float
        Surface emissivity ε ∈ (0, 1].

    Returns
    -------
    float
        Equilibrium temperature [K].
    """
    if q_conv < 0.0:
        raise ValueError(f"q_conv must be non-negative; got {q_conv:.6g}")
    if q_solar < 0.0:
        raise ValueError(f"q_solar must be non-negative; got {q_solar:.6g}")
    if not (0.0 < emissivity <= 1.0):
        raise ValueError(f"emissivity must be in (0, 1]; got {emissivity:.6g}")

    total_in = q_conv + q_solar
    if total_in <= 0.0:
        return _T_SPACE_K
    return max(_T_SPACE_K, (total_in / (emissivity * SIGMA_SB)) ** 0.25)


# ---------------------------------------------------------------------------
# Single-panel step function (pure)
# ---------------------------------------------------------------------------

def step_panel(
    config: SurfaceConfig,
    wall_temp_k: float,
    atm: AtmosphericState,
    airspeed_m_s: float,
    dt: float,
    cos_solar_incidence: float = 0.0,
    solar_irradiance_w_m2: float = _SOLAR_IRRADIANCE_W_M2,
) -> tuple[float, PanelThermalState]:
    """
    Advance the thermal state of one surface panel by one timestep.

    Parameters
    ----------
    config : SurfaceConfig
        Panel material and geometry configuration.
    wall_temp_k : float
        Current wall temperature [K].
    atm : AtmosphericState
        Atmospheric state (density, temperature, etc.) at this altitude.
    airspeed_m_s : float
        Vehicle airspeed [m s⁻¹]. Must be ≥ 0.
    dt : float
        Timestep [s]. Must be positive.
    cos_solar_incidence : float
        Cosine of solar angle of incidence for this panel. Default 0
        (sun behind/perpendicular).
    solar_irradiance_w_m2 : float
        Solar irradiance [W m⁻²]. Default 1361.

    Returns
    -------
    new_wall_temp_k : float
        Updated wall temperature after dt [K].
    panel_state : PanelThermalState
        Frozen snapshot of the thermal state this tick.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt:.6g}")
    if airspeed_m_s < 0.0:
        raise ValueError(f"airspeed_m_s must be non-negative; got {airspeed_m_s:.6g}")

    q_conv = convective_flux(atm.density, airspeed_m_s, config.stanton_number)
    q_rad = radiative_flux(wall_temp_k, config.emissivity)
    q_sol = solar_flux(config.absorptivity, cos_solar_incidence, solar_irradiance_w_m2)
    q_net = net_flux(q_conv, q_rad, q_sol)

    dT_dt = temperature_rate(q_net, config.area_m2, config.thermal_mass)

    # Forward Euler integration
    new_temp = wall_temp_k + dT_dt * dt
    # Floor: cannot drop below space background temperature
    new_temp = max(_MIN_WALL_TEMP_K, new_temp)

    panel_state = PanelThermalState(
        panel_id=config.panel_id,
        wall_temp_k=new_temp,
        conv_flux_w_m2=q_conv,
        rad_flux_w_m2=q_rad,
        solar_flux_w_m2=q_sol,
        net_flux_w_m2=q_net,
        temp_rate_k_s=dT_dt,
    )
    return new_temp, panel_state


# ---------------------------------------------------------------------------
# ThermalModel — stateful multi-panel model
# ---------------------------------------------------------------------------

class ThermalModel:
    """
    Stateful thermal model tracking wall temperatures for all surface panels.

    Pipeline Stage 8 entry point. Called once per tick after the atmosphere
    solver (Stage 7) and before Component Updates (Stage 9).

    Parameters
    ----------
    panels : list[SurfaceConfig]
        All surface panels with their material properties.
    """

    def __init__(self, panels: List[SurfaceConfig]) -> None:
        if not panels:
            raise ValueError("panels must be a non-empty list of SurfaceConfig")
        for p in panels:
            if not isinstance(p, SurfaceConfig):
                raise TypeError(
                    f"All entries in panels must be SurfaceConfig; "
                    f"got {type(p).__name__}"
                )
        self._configs: List[SurfaceConfig] = list(panels)
        # Initialize wall temperatures from config
        self._wall_temps: List[float] = [p.initial_temp_k for p in panels]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def panel_count(self) -> int:
        """Number of surface panels."""
        return len(self._configs)

    @property
    def wall_temperatures(self) -> List[float]:
        """Current wall temperatures [K], one per panel."""
        return list(self._wall_temps)

    @property
    def max_wall_temp_k(self) -> float:
        """Maximum wall temperature across all panels [K]."""
        return max(self._wall_temps)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        atm: AtmosphericState,
        airspeed_m_s: float,
        mission_time: float,
        dt: float,
        cos_solar_incidences: Optional[List[float]] = None,
        solar_irradiance_w_m2: float = _SOLAR_IRRADIANCE_W_M2,
    ) -> ThermalSnapshot:
        """
        Advance all panels by one simulation timestep.

        Parameters
        ----------
        atm : AtmosphericState
            Atmospheric state at the vehicle's current altitude.
        airspeed_m_s : float
            Vehicle airspeed [m s⁻¹]. Must be ≥ 0.
        mission_time : float
            Current mission elapsed time [s]. Used for telemetry.
        dt : float
            Timestep [s]. Must be positive.
        cos_solar_incidences : list[float] | None
            Per-panel cosine of solar angle of incidence. If None, defaults
            to 0.0 for all panels (no solar heating).
        solar_irradiance_w_m2 : float
            Solar irradiance [W m⁻²]. Default 1361 W m⁻².

        Returns
        -------
        ThermalSnapshot
            Vehicle-level frozen thermal summary.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive; got {dt:.6g}")
        if airspeed_m_s < 0.0:
            raise ValueError(
                f"airspeed_m_s must be non-negative; got {airspeed_m_s:.6g}"
            )
        if not isinstance(atm, AtmosphericState):
            raise TypeError("atm must be an AtmosphericState")

        n = len(self._configs)
        if cos_solar_incidences is None:
            cos_list = [0.0] * n
        else:
            if len(cos_solar_incidences) != n:
                raise ValueError(
                    f"cos_solar_incidences must have {n} entries; "
                    f"got {len(cos_solar_incidences)}"
                )
            cos_list = [float(c) for c in cos_solar_incidences]

        panel_states: List[PanelThermalState] = []
        total_conv = 0.0
        total_rad = 0.0
        total_net = 0.0

        for i, (cfg, T_wall) in enumerate(zip(self._configs, self._wall_temps)):
            new_T, ps = step_panel(
                config=cfg,
                wall_temp_k=T_wall,
                atm=atm,
                airspeed_m_s=airspeed_m_s,
                dt=dt,
                cos_solar_incidence=cos_list[i],
                solar_irradiance_w_m2=solar_irradiance_w_m2,
            )
            self._wall_temps[i] = new_T
            panel_states.append(ps)
            total_conv += ps.conv_flux_w_m2 * cfg.area_m2
            total_rad += ps.rad_flux_w_m2 * cfg.area_m2
            total_net += ps.net_flux_w_m2 * cfg.area_m2

        return ThermalSnapshot(
            panels=panel_states,
            max_wall_temp_k=max(self._wall_temps),
            total_conv_power_w=total_conv,
            total_rad_power_w=total_rad,
            total_net_power_w=total_net,
            mission_time=mission_time,
        )

    def reset(self) -> None:
        """Reset all wall temperatures to their initial values from config."""
        self._wall_temps = [p.initial_temp_k for p in self._configs]

    def __repr__(self) -> str:
        t_max = self.max_wall_temp_k
        return (
            f"ThermalModel(panels={self.panel_count}, "
            f"T_max={t_max:.1f}K)"
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def nose_cap_config(
    nose_radius_m: float = 0.15,
    thickness_m: float = 0.02,
    material_density_kg_m3: float = 1800.0,
    specific_heat_j_kg_k: float = 1200.0,
    emissivity: float = 0.85,
    absorptivity: float = 0.9,
) -> SurfaceConfig:
    """
    Build a SurfaceConfig for a hemispherical nose cap.

    Uses Chapman's formula for the Stanton number:
        St = 1.83e-4 / √R_nose

    Parameters
    ----------
    nose_radius_m : float
        Nose radius of curvature [m]. Default 0.15 m.
    thickness_m : float
        Cap wall thickness [m]. Default 20 mm (for mass estimation).
    material_density_kg_m3 : float
        Cap material density [kg m⁻³]. Default 1800 (carbon composite).
    specific_heat_j_kg_k : float
        Specific heat [J kg⁻¹ K⁻¹]. Default 1200 (carbon composite).
    emissivity : float
        Surface emissivity. Default 0.85.
    absorptivity : float
        Solar absorptivity. Default 0.9.

    Returns
    -------
    SurfaceConfig
    """
    if nose_radius_m <= 0.0:
        raise ValueError(f"nose_radius_m must be positive; got {nose_radius_m:.6g}")
    if thickness_m <= 0.0:
        raise ValueError(f"thickness_m must be positive; got {thickness_m:.6g}")

    # Hemisphere area: 2πR²
    area = 2.0 * math.pi * nose_radius_m ** 2
    # Hemisphere shell mass: area × thickness × density
    mass = area * thickness_m * material_density_kg_m3
    # Chapman's stagnation-point Stanton number
    st = 1.83e-4 / math.sqrt(nose_radius_m)

    return SurfaceConfig(
        panel_id="nose_cap",
        area_m2=area,
        mass_kg=mass,
        specific_heat_j_kg_k=specific_heat_j_kg_k,
        emissivity=emissivity,
        absorptivity=absorptivity,
        stanton_number=st,
        initial_temp_k=293.15,
    )


def simple_thermal_model(
    n_panels: int = 1,
    area_m2: float = 1.0,
    mass_kg: float = 10.0,
    specific_heat_j_kg_k: float = 900.0,
    emissivity: float = 0.8,
    stanton_number: float = 1.0e-4,
    initial_temp_k: float = 293.15,
) -> ThermalModel:
    """
    Build a ThermalModel with *n_panels* identical panels.

    Useful for unit tests and simple trajectory analyses.

    Parameters
    ----------
    n_panels : int
        Number of panels. Default 1.
    area_m2 : float
        Area per panel [m²]. Default 1.0.
    mass_kg : float
        Mass per panel [kg]. Default 10.0.
    specific_heat_j_kg_k : float
        Specific heat [J kg⁻¹ K⁻¹]. Default 900 (aluminium).
    emissivity : float
        Surface emissivity. Default 0.8.
    stanton_number : float
        Stanton number. Default 1e-4.
    initial_temp_k : float
        Initial wall temperature [K]. Default 293.15 K.

    Returns
    -------
    ThermalModel
    """
    if n_panels < 1:
        raise ValueError(f"n_panels must be ≥ 1; got {n_panels}")

    panels = [
        SurfaceConfig(
            panel_id=f"panel_{i}",
            area_m2=area_m2,
            mass_kg=mass_kg,
            specific_heat_j_kg_k=specific_heat_j_kg_k,
            emissivity=emissivity,
            stanton_number=stanton_number,
            initial_temp_k=initial_temp_k,
        )
        for i in range(n_panels)
    ]
    return ThermalModel(panels)

# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


# tests/unit/test_thermodynamics.py
import math
import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

from nova.core.constants import SIGMA_SB
from nova.physics.atmosphere import AtmosphericState
from nova.physics.thermodynamics import (
    SurfaceConfig,
    PanelThermalState,
    ThermalSnapshot,
    convective_flux,
    radiative_flux,
    solar_flux,
    net_flux,
    temperature_rate,
    equilibrium_temperature,
    step_panel,
    ThermalModel,
    nose_cap_config,
    simple_thermal_model,
    _MIN_WALL_TEMP_K,
    _SOLAR_IRRADIANCE_W_M2,
    _T_SPACE_K,
)


@pytest.fixture
def mock_atm() -> AtmosphericState:
    atm = MagicMock(spec=AtmosphericState)
    atm.density = 1.225
    return atm


class TestSurfaceConfig:
    def test_valid_initialization(self):
        cfg = SurfaceConfig(
            panel_id="windward",
            area_m2=2.5,
            mass_kg=15.0,
            specific_heat_j_kg_k=1000.0,
            emissivity=0.85,
            absorptivity=0.92,
            stanton_number=1.5e-4,
            initial_temp_k=300.0,
        )
        assert cfg.panel_id == "windward"
        assert cfg.area_m2 == 2.5
        assert cfg.thermal_mass == 15000.0
        assert cfg.emissivity == 0.85

    def test_immutability(self):
        cfg = SurfaceConfig("test", 1.0, 1.0, 1.0, 1.0)
        with pytest.raises(FrozenInstanceError):
            cfg.area_m2 = 2.0

    @pytest.mark.parametrize(
        "kwargs, expected_error",
        [
            ({"panel_id": ""}, "non-empty string"),
            ({"area_m2": 0.0}, "must be positive"),
            ({"mass_kg": -5.0}, "must be positive"),
            ({"emissivity": 1.5}, "must be in \\(0, 1\\]"),
            ({"emissivity": 0.0}, "must be in \\(0, 1\\]"),
            ({"absorptivity": -0.1}, "must be in \\[0, 1\\]"),
            ({"initial_temp_k": 1.0}, "must be ≥ 2.7"),
        ],
    )
    def test_invalid_parameters(self, kwargs, expected_error):
        base_args = {
            "panel_id": "test",
            "area_m2": 1.0,
            "mass_kg": 1.0,
            "specific_heat_j_kg_k": 1000.0,
            "emissivity": 0.5,
        }
        base_args.update(kwargs)
        with pytest.raises(ValueError, match=expected_error):
            SurfaceConfig(**base_args)


class TestPurePhysicsFunctions:
    def test_convective_flux(self):
        flux = convective_flux(density_kg_m3=1.225, airspeed_m_s=3000.0, stanton_number=1.0e-4)
        expected = 1.0e-4 * 1.225 * (3000.0**3) / 2.0
        assert math.isclose(flux, expected)

    def test_convective_flux_invalid(self):
        with pytest.raises(ValueError, match="density_kg_m3 must be non-negative"):
            convective_flux(-1.0, 1000.0, 1e-4)
        with pytest.raises(ValueError, match="stanton_number must be positive"):
            convective_flux(1.0, 1000.0, 0.0)

    def test_radiative_flux(self):
        flux = radiative_flux(wall_temp_k=1000.0, emissivity=0.8)
        expected = 0.8 * SIGMA_SB * (1000.0**4)
        assert math.isclose(flux, expected)

    def test_radiative_flux_invalid(self):
        with pytest.raises(ValueError, match="wall_temp_k must be ≥ 2.7"):
            radiative_flux(1.0, 0.8)

    def test_solar_flux(self):
        flux = solar_flux(absorptivity=0.9, cos_incidence=0.5, irradiance_w_m2=1361.0)
        assert math.isclose(flux, 0.9 * 1361.0 * 0.5)

    def test_solar_flux_behind_panel(self):
        flux = solar_flux(absorptivity=0.9, cos_incidence=-0.5)
        assert flux == 0.0

    def test_net_flux(self):
        flux = net_flux(q_conv=1000.0, q_solar=500.0, q_rad=1200.0)
        assert math.isclose(flux, 300.0)

    def test_temperature_rate(self):
        rate = temperature_rate(net_flux_w_m2=500.0, area_m2=2.0, thermal_mass_j_k=10000.0)
        assert math.isclose(rate, (500.0 * 2.0) / 10000.0)

    def test_equilibrium_temperature(self):
        t_eq = equilibrium_temperature(q_conv=10000.0, q_solar=0.0, emissivity=0.8)
        expected = (10000.0 / (0.8 * SIGMA_SB)) ** 0.25
        assert math.isclose(t_eq, expected)

    def test_equilibrium_temperature_zero_input(self):
        t_eq = equilibrium_temperature(0.0, 0.0, 0.8)
        assert t_eq == _T_SPACE_K


class TestStepPanel:
    def test_step_panel_integration(self, mock_atm):
        cfg = SurfaceConfig("test", area_m2=1.0, mass_kg=10.0, specific_heat_j_kg_k=100.0, emissivity=1.0, stanton_number=1e-4)
        new_t, state = step_panel(cfg, 300.0, mock_atm, 2000.0, 0.1)
        
        q_conv = 1e-4 * 1.225 * (2000.0**3) / 2.0
        q_rad = 1.0 * SIGMA_SB * (300.0**4)
        q_net = q_conv - q_rad
        dt_dt = (q_net * 1.0) / 1000.0
        expected_t = 300.0 + dt_dt * 0.1

        assert math.isclose(new_t, expected_t)
        assert state.wall_temp_k == new_t
        assert state.conv_flux_w_m2 == q_conv
        assert state.rad_flux_w_m2 == q_rad

    def test_step_panel_floor_temperature(self, mock_atm):
        cfg = SurfaceConfig("test", area_m2=1.0, mass_kg=1.0, specific_heat_j_kg_k=1.0, emissivity=1.0)
        mock_atm.density = 0.0 
        new_t, state = step_panel(cfg, 3.0, mock_atm, 0.0, 1000.0)
        assert new_t == _MIN_WALL_TEMP_K
        assert state.wall_temp_k == _MIN_WALL_TEMP_K


class TestThermalModel:
    def test_thermal_model_initialization(self):
        model = simple_thermal_model(n_panels=2, initial_temp_k=300.0)
        assert model.panel_count == 2
        assert model.wall_temperatures == [300.0, 300.0]
        assert model.max_wall_temp_k == 300.0

    def test_thermal_model_step(self, mock_atm):
        model = simple_thermal_model(n_panels=2, area_m2=1.0, mass_kg=10.0, emissivity=0.8)
        snapshot = model.step(mock_atm, airspeed_m_s=1000.0, mission_time=5.0, dt=0.1, cos_solar_incidences=[1.0, 0.0])
        
        assert snapshot.mission_time == 5.0
        assert len(snapshot.panels) == 2
        assert snapshot.panels[0].solar_flux_w_m2 > 0.0
        assert snapshot.panels[1].solar_flux_w_m2 == 0.0
        assert model.wall_temperatures[0] > model.wall_temperatures[1]

    def test_thermal_model_reset(self, mock_atm):
        model = simple_thermal_model(initial_temp_k=300.0)
        model.step(mock_atm, 5000.0, 1.0, 0.1)
        assert model.wall_temperatures[0] > 300.0
        model.reset()
        assert model.wall_temperatures[0] == 300.0

    def test_step_invalid_cos_list(self, mock_atm):
        model = simple_thermal_model(n_panels=2)
        with pytest.raises(ValueError, match="cos_solar_incidences must have 2 entries"):
            model.step(mock_atm, 1000.0, 1.0, 0.1, cos_solar_incidences=[1.0])


class TestConstructors:
    def test_nose_cap_config(self):
        cfg = nose_cap_config(nose_radius_m=0.15)
        assert cfg.panel_id == "nose_cap"
        expected_st = 1.83e-4 / math.sqrt(0.15)
        assert math.isclose(cfg.stanton_number, expected_st)

    def test_nose_cap_invalid(self):
        with pytest.raises(ValueError, match="nose_radius_m must be positive"):
            nose_cap_config(nose_radius_m=-0.1)
