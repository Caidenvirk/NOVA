"""
nova.ui.orbital_deck
=====================
Orbital elements panel for Project NOVA glass cockpit.

Architectural role
------------------
Phase 13 — UI Glass Cockpit.
Pipeline stage: Stage 13 (UI Engine). Consumes a RenderFrame and produces
an OrbitalDeckState frozen dataclass containing all data needed to render
the orbital elements readout panel.

Design
------
The orbital deck displays:
  - Apoapsis / Periapsis altitudes [km]
  - Semi-major axis [km]
  - Eccentricity (dimensionless)
  - Inclination [deg]
  - RAAN — Right Ascension of Ascending Node [deg]
  - Argument of Periapsis ω [deg]
  - True Anomaly ν [deg]
  - Orbital period [s] / [min]
  - Time to apoapsis / periapsis [s]
  - Hohmann transfer delta-v budget (if a target orbit altitude is set)

All quantities are derived from the RenderFrame's pre-computed orbital
elements (populated from Phase 2 orbital.py via the telemetry pipeline).
No orbital mechanics is re-computed here.

Manoeuvre node support:
  If a target_apoapsis_m and/or target_periapsis_m is set on the deck,
  the OrbitalDeckState includes a simplified hohmann_dv estimate:
    Δv_total ≈ Δv1 (at periapsis to raise apoapsis) +
                Δv2 (at new apoapsis to circularise)
  Using the vis-viva equation: v = √(μ(2/r − 1/a))

I/O contract
------------
Input  : RenderFrame, optional target orbit parameters
Output : OrbitalDeckState (frozen dataclass)

No Pygame calls. No physics side-effects.

References
----------
- NOVA Engineering Handoff §12 Phase 13
- Bate, Mueller & White "Fundamentals of Astrodynamics" §6.3 (Hohmann)
- Vallado "Fundamentals of Astrodynamics" 4th ed. §6.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from nova.core.constants import EARTH_MU, EARTH_RADIUS_EQ
from nova.rendering.viewport import RenderFrame

# ---------------------------------------------------------------------------
# HohmannBudget — optional manoeuvre node data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HohmannBudget:
    """
    Simplified Hohmann transfer delta-v budget.

    Attributes
    ----------
    current_periapsis_m : float
        Current periapsis altitude [m].
    current_apoapsis_m : float
        Current apoapsis altitude [m].
    target_periapsis_m : float
        Target periapsis altitude [m].
    target_apoapsis_m : float
        Target apoapsis altitude [m].
    dv1_m_s : float
        First burn delta-v [m s⁻¹] (at periapsis/apoapsis of current orbit).
    dv2_m_s : float
        Second burn delta-v [m s⁻¹] (circularise at target).
    total_dv_m_s : float
        Total Δv = |dv1| + |dv2| [m s⁻¹].
    is_valid : bool
        True when both orbits are valid elliptic orbits.
    """

    current_periapsis_m: float
    current_apoapsis_m: float
    target_periapsis_m: float
    target_apoapsis_m: float
    dv1_m_s: float
    dv2_m_s: float
    total_dv_m_s: float
    is_valid: bool

    def __post_init__(self) -> None:
        for attr in ("current_periapsis_m", "current_apoapsis_m",
                     "target_periapsis_m", "target_apoapsis_m",
                     "dv1_m_s", "dv2_m_s", "total_dv_m_s"):
            object.__setattr__(self, attr, float(getattr(self, attr)))
        object.__setattr__(self, "is_valid", bool(self.is_valid))

    def __repr__(self) -> str:
        return (
            f"HohmannBudget(Δv1={self.dv1_m_s:.1f}m/s, "
            f"Δv2={self.dv2_m_s:.1f}m/s, "
            f"total={self.total_dv_m_s:.1f}m/s)"
        )


# ---------------------------------------------------------------------------
# OrbitalDeckState — complete orbital panel data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitalDeckState:
    """
    Complete frozen data bundle for the orbital elements panel.

    All altitudes are above the equatorial reference ellipsoid.
    All angles are stored in both radians (physics) and degrees (display).

    Attributes
    ----------
    mission_time : float
        Mission elapsed time [s].
    apoapsis_m : float
        Apoapsis altitude [m].
    periapsis_m : float
        Periapsis altitude [m].
    apoapsis_km : float
        Apoapsis altitude [km].
    periapsis_km : float
        Periapsis altitude [km].
    semi_major_axis_m : float
        Semi-major axis [m].
    semi_major_axis_km : float
        Semi-major axis [km].
    eccentricity : float
        Orbital eccentricity (dimensionless).
    inclination_rad : float
        Inclination [rad].
    inclination_deg : float
        Inclination [degrees].
    raan_rad : float
        Right ascension of ascending node [rad].
    raan_deg : float
        RAAN [degrees].
    arg_of_periapsis_rad : float
        Argument of periapsis ω [rad].
    arg_of_periapsis_deg : float
        Argument of periapsis [degrees].
    true_anomaly_rad : float
        True anomaly ν [rad].
    true_anomaly_deg : float
        True anomaly [degrees].
    orbital_period_s : float
        Orbital period [s]. 0 if orbit is not closed.
    orbital_period_min : float
        Orbital period [minutes].
    time_to_apoapsis_s : float
        Estimated time to next apoapsis [s]. NaN if not computable.
    time_to_periapsis_s : float
        Estimated time to next periapsis [s]. NaN if not computable.
    is_orbit_closed : bool
        True when eccentricity < 1 (elliptic orbit).
    is_suborbital : bool
        True when periapsis is below atmosphere threshold (120 km).
    hohmann : HohmannBudget | None
        Manoeuvre budget if target orbit is set, else None.
    """

    mission_time: float
    apoapsis_m: float
    periapsis_m: float
    apoapsis_km: float
    periapsis_km: float
    semi_major_axis_m: float
    semi_major_axis_km: float
    eccentricity: float
    inclination_rad: float
    inclination_deg: float
    raan_rad: float
    raan_deg: float
    arg_of_periapsis_rad: float
    arg_of_periapsis_deg: float
    true_anomaly_rad: float
    true_anomaly_deg: float
    orbital_period_s: float
    orbital_period_min: float
    time_to_apoapsis_s: float
    time_to_periapsis_s: float
    is_orbit_closed: bool
    is_suborbital: bool
    hohmann: Optional[HohmannBudget]

    def __post_init__(self) -> None:
        for attr in ("mission_time", "apoapsis_m", "periapsis_m",
                     "apoapsis_km", "periapsis_km", "semi_major_axis_m",
                     "semi_major_axis_km", "eccentricity",
                     "inclination_rad", "inclination_deg",
                     "raan_rad", "raan_deg",
                     "arg_of_periapsis_rad", "arg_of_periapsis_deg",
                     "true_anomaly_rad", "true_anomaly_deg",
                     "orbital_period_s", "orbital_period_min",
                     "time_to_apoapsis_s", "time_to_periapsis_s"):
            object.__setattr__(self, attr, float(getattr(self, attr)))
        object.__setattr__(self, "is_orbit_closed", bool(self.is_orbit_closed))
        object.__setattr__(self, "is_suborbital", bool(self.is_suborbital))

    def __repr__(self) -> str:
        return (
            f"OrbitalDeckState(t={self.mission_time:.1f}s, "
            f"Ap={self.apoapsis_km:.1f}km, "
            f"Pe={self.periapsis_km:.1f}km, "
            f"e={self.eccentricity:.4f}, "
            f"i={self.inclination_deg:.2f}°)"
        )


# ---------------------------------------------------------------------------
# Pure Hohmann delta-v computation
# ---------------------------------------------------------------------------

_ATMO_THRESHOLD_M = 120_000.0   # 120 km — below this is "suborbital" / reentry

def _vis_viva(r: float, a: float, mu: float) -> float:
    """Orbital speed from vis-viva equation: v = √(μ(2/r − 1/a)) [m s⁻¹]."""
    arg = mu * (2.0 / r - 1.0 / a)
    if arg < 0.0:
        return 0.0
    return math.sqrt(arg)


def compute_hohmann_budget(
    current_periapsis_m: float,
    current_apoapsis_m: float,
    target_periapsis_m: float,
    target_apoapsis_m: float,
    planet_radius_m: float = EARTH_RADIUS_EQ,
    mu: float = EARTH_MU,
) -> HohmannBudget:
    """
    Compute a simplified Hohmann-like transfer delta-v budget.

    Assumes both orbits are coplanar and circular at their respective radii.
    For elliptic transfers, uses the vis-viva equation at the burn points.

    Parameters
    ----------
    current_periapsis_m, current_apoapsis_m : float
        Current orbit periapsis/apoapsis altitudes [m].
    target_periapsis_m, target_apoapsis_m : float
        Target orbit altitudes [m].
    planet_radius_m : float
        Planet equatorial radius [m]. Default EARTH_RADIUS_EQ.
    mu : float
        Gravitational parameter [m³ s⁻²]. Default EARTH_MU.

    Returns
    -------
    HohmannBudget
    """
    R = planet_radius_m

    r_pe1 = R + current_periapsis_m
    r_ap1 = R + current_apoapsis_m
    r_pe2 = R + target_periapsis_m
    r_ap2 = R + target_apoapsis_m

    # Validate: need positive radii and elliptic orbits
    valid = (
        current_periapsis_m >= 0.0 and r_ap1 >= r_pe1
        and target_periapsis_m >= 0.0 and r_ap2 >= r_pe2
    )

    if not valid:
        return HohmannBudget(
            current_periapsis_m=current_periapsis_m,
            current_apoapsis_m=current_apoapsis_m,
            target_periapsis_m=target_periapsis_m,
            target_apoapsis_m=target_apoapsis_m,
            dv1_m_s=0.0,
            dv2_m_s=0.0,
            total_dv_m_s=0.0,
            is_valid=False,
        )

    a1 = 0.5 * (r_pe1 + r_ap1)   # current orbit SMA
    a2 = 0.5 * (r_pe2 + r_ap2)   # target orbit SMA

    # Transfer ellipse: periapsis at current orbit, apoapsis at target orbit
    # (assumes raising; adjust for lowering by swapping)
    r_burn1 = r_pe1    # first burn at current periapsis
    r_burn2 = r_ap2    # second burn at target apoapsis
    a_transfer = 0.5 * (r_burn1 + r_burn2)

    v1_current = _vis_viva(r_burn1, a1, mu)
    v1_transfer = _vis_viva(r_burn1, a_transfer, mu)
    dv1 = abs(v1_transfer - v1_current)

    v2_transfer = _vis_viva(r_burn2, a_transfer, mu)
    v2_target = _vis_viva(r_burn2, a2, mu)
    dv2 = abs(v2_target - v2_transfer)

    return HohmannBudget(
        current_periapsis_m=current_periapsis_m,
        current_apoapsis_m=current_apoapsis_m,
        target_periapsis_m=target_periapsis_m,
        target_apoapsis_m=target_apoapsis_m,
        dv1_m_s=dv1,
        dv2_m_s=dv2,
        total_dv_m_s=dv1 + dv2,
        is_valid=True,
    )


# ---------------------------------------------------------------------------
# OrbitalDeck builder
# ---------------------------------------------------------------------------

class OrbitalDeck:
    """
    Produces OrbitalDeckState from a RenderFrame each display tick.

    Parameters
    ----------
    target_apoapsis_m : float | None
        Target apoapsis altitude [m] for Hohmann budget. None = no manoeuvre node.
    target_periapsis_m : float | None
        Target periapsis altitude [m]. If only target_apoapsis_m is set,
        target_periapsis_m defaults to current periapsis (circularisation).
    planet_radius_m : float
        Planet equatorial radius [m]. Default EARTH_RADIUS_EQ.
    mu : float
        Gravitational parameter [m³ s⁻²]. Default EARTH_MU.
    """

    def __init__(
        self,
        target_apoapsis_m: Optional[float] = None,
        target_periapsis_m: Optional[float] = None,
        planet_radius_m: float = EARTH_RADIUS_EQ,
        mu: float = EARTH_MU,
    ) -> None:
        self._target_ap = (
            float(target_apoapsis_m) if target_apoapsis_m is not None else None
        )
        self._target_pe = (
            float(target_periapsis_m) if target_periapsis_m is not None else None
        )
        self._planet_radius = float(planet_radius_m)
        self._mu = float(mu)

    def set_manoeuvre_node(
        self,
        target_apoapsis_m: Optional[float],
        target_periapsis_m: Optional[float] = None,
    ) -> None:
        """Update the target orbit for the Hohmann budget."""
        self._target_ap = (
            float(target_apoapsis_m) if target_apoapsis_m is not None else None
        )
        self._target_pe = (
            float(target_periapsis_m) if target_periapsis_m is not None else None
        )

    def clear_manoeuvre_node(self) -> None:
        """Remove the manoeuvre node."""
        self._target_ap = None
        self._target_pe = None

    def build(self, frame: RenderFrame) -> OrbitalDeckState:
        """
        Build an OrbitalDeckState from the current RenderFrame.

        Parameters
        ----------
        frame : RenderFrame
            Interpolated render state from Viewport.

        Returns
        -------
        OrbitalDeckState
        """
        if not isinstance(frame, RenderFrame):
            raise TypeError(
                f"frame must be a RenderFrame; got {type(frame).__name__}"
            )

        a = frame.semi_major_axis
        e = frame.eccentricity
        inc = frame.inclination
        ap_alt = frame.apoapsis
        pe_alt = frame.periapsis

        is_closed = (e < 1.0) and (a > 0.0)
        is_sub = pe_alt < _ATMO_THRESHOLD_M

        # Period
        if is_closed and a > 0.0:
            period_s = 2.0 * math.pi * math.sqrt(a ** 3 / self._mu)
        else:
            period_s = 0.0

        period_min = period_s / 60.0

        # Time to apoapsis / periapsis from true anomaly
        # Using mean anomaly: M = E − e*sin(E), t = M/(2π) * T
        nu = 0.0  # true_anomaly not in RenderFrame; computed below from r/v
        # Access true_anomaly from frame; not directly a field — use inclination proxy
        # RenderFrame doesn't carry true_anomaly directly, so we use a positional estimate
        # We approximate via the angle from periapsis direction using position/velocity
        tta_s = float("nan")
        ttp_s = float("nan")

        if is_closed and period_s > 0.0:
            # Estimate from angular position in orbit
            r_vec = frame.position_eci
            v_vec = frame.velocity_eci
            r_norm = float(np.linalg.norm(r_vec))
            if r_norm > 1.0:
                h_vec = np.cross(r_vec, v_vec)
                h_norm = float(np.linalg.norm(h_vec))
                if h_norm > 1.0:
                    e_vec = (
                        np.cross(v_vec, h_vec) / self._mu
                        - r_vec / r_norm
                    )
                    e_norm = float(np.linalg.norm(e_vec))
                    if e_norm > 1.0e-10:
                        cos_nu = float(np.dot(e_vec / e_norm, r_vec / r_norm))
                        cos_nu = max(-1.0, min(1.0, cos_nu))
                        true_nu = math.acos(cos_nu)
                        # Sign: if r·v > 0, vehicle is moving away from Pe
                        if float(np.dot(r_vec, v_vec)) < 0.0:
                            true_nu = 2.0 * math.pi - true_nu
                        # Mean anomaly via eccentric anomaly
                        cos_E = (e + math.cos(true_nu)) / (1.0 + e * math.cos(true_nu))
                        cos_E = max(-1.0, min(1.0, cos_E))
                        ecc_an = math.acos(cos_E)
                        if true_nu > math.pi:
                            ecc_an = 2.0 * math.pi - ecc_an
                        mean_an = ecc_an - e * math.sin(ecc_an)
                        # Time since periapsis
                        t_since_pe = mean_an / (2.0 * math.pi) * period_s
                        ttp_s = period_s - t_since_pe  # time to next periapsis
                        tta_s = (0.5 * period_s - t_since_pe) % period_s

        # Hohmann budget
        hohmann: Optional[HohmannBudget] = None
        if self._target_ap is not None:
            tgt_pe = (
                self._target_pe
                if self._target_pe is not None
                else pe_alt
            )
            hohmann = compute_hohmann_budget(
                current_periapsis_m=pe_alt,
                current_apoapsis_m=ap_alt,
                target_periapsis_m=tgt_pe,
                target_apoapsis_m=self._target_ap,
                planet_radius_m=self._planet_radius,
                mu=self._mu,
            )

        # Extract RAAN and arg_of_periapsis from RenderFrame — not present,
        # so we compute them from position/velocity if possible
        raan = 0.0
        arg_peri = 0.0

        r_vec2 = frame.position_eci
        v_vec2 = frame.velocity_eci
        r2 = float(np.linalg.norm(r_vec2))
        if r2 > 1.0:
            h2 = np.cross(r_vec2, v_vec2)
            h2_n = float(np.linalg.norm(h2))
            if h2_n > 1.0:
                # Node vector n = K × h  (K = [0,0,1])
                K = np.array([0.0, 0.0, 1.0])
                node = np.cross(K, h2)
                node_n = float(np.linalg.norm(node))
                if node_n > 1.0e-10:
                    raan = math.acos(
                        max(-1.0, min(1.0, node[0] / node_n))
                    )
                    if node[1] < 0.0:
                        raan = 2.0 * math.pi - raan
                # Eccentricity vector
                e_vec2 = np.cross(v_vec2, h2) / self._mu - r_vec2 / r2
                e2_n = float(np.linalg.norm(e_vec2))
                if e2_n > 1.0e-10 and node_n > 1.0e-10:
                    arg_peri = math.acos(
                        max(-1.0, min(1.0,
                            float(np.dot(node / node_n, e_vec2 / e2_n))
                        ))
                    )
                    if e_vec2[2] < 0.0:
                        arg_peri = 2.0 * math.pi - arg_peri

        true_anomaly_display = 0.0
        r3 = float(np.linalg.norm(frame.position_eci))
        if r3 > 1.0:
            h3 = np.cross(frame.position_eci, frame.velocity_eci)
            h3_n = float(np.linalg.norm(h3))
            if h3_n > 1.0:
                e3 = np.cross(frame.velocity_eci, h3) / self._mu - frame.position_eci / r3
                e3_n = float(np.linalg.norm(e3))
                if e3_n > 1.0e-10:
                    cos_nu3 = float(np.dot(e3 / e3_n, frame.position_eci / r3))
                    cos_nu3 = max(-1.0, min(1.0, cos_nu3))
                    true_anomaly_display = math.acos(cos_nu3)
                    if float(np.dot(frame.position_eci, frame.velocity_eci)) < 0.0:
                        true_anomaly_display = 2.0 * math.pi - true_anomaly_display

        return OrbitalDeckState(
            mission_time=frame.mission_time,
            apoapsis_m=ap_alt,
            periapsis_m=pe_alt,
            apoapsis_km=ap_alt / 1_000.0,
            periapsis_km=pe_alt / 1_000.0,
            semi_major_axis_m=a,
            semi_major_axis_km=a / 1_000.0,
            eccentricity=e,
            inclination_rad=inc,
            inclination_deg=math.degrees(inc),
            raan_rad=raan,
            raan_deg=math.degrees(raan),
            arg_of_periapsis_rad=arg_peri,
            arg_of_periapsis_deg=math.degrees(arg_peri),
            true_anomaly_rad=true_anomaly_display,
            true_anomaly_deg=math.degrees(true_anomaly_display),
            orbital_period_s=period_s,
            orbital_period_min=period_min,
            time_to_apoapsis_s=tta_s,
            time_to_periapsis_s=ttp_s,
            is_orbit_closed=is_closed,
            is_suborbital=is_sub,
            hohmann=hohmann,
        )

    def __repr__(self) -> str:
        return (
            f"OrbitalDeck(target_ap={self._target_ap}, "
            f"target_pe={self._target_pe})"
        )
