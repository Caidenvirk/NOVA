"""
nova.physics.orbital
====================
Orbital mechanics solver for Project NOVA.

Provides two independent subsystems:

1. **Keplerian orbital elements** — extract and use the six classical orbital
   elements (a, e, i, Ω, ω, ν) from an ECI state vector, and compute
   closed-form Keplerian trajectories for single-body motion.

2. **N-body gravitational acceleration** — compute the gravitational
   acceleration on the vehicle from an arbitrary number of point-mass bodies
   (Earth, Moon, Sun, custom). This is the quantity injected into the force
   accumulator at pipeline Stage 6.

Physics contracts
-----------------
* Newtonian gravity only — no GR corrections.
* J2 oblateness perturbation is included for Earth (switchable).
* No atmospheric drag in this module — that belongs in aerodynamics.py.
* All vectors in ECI frame, SI units.

The orbital solver does NOT modify VehicleState. It returns force vectors
and orbital elements that are consumed by the force accumulator and the
orbital mechanics HUD panel respectively.

References
----------
- Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
- Bate, Mueller & White, "Fundamentals of Astrodynamics", Dover
- Montenbruck & Gill, "Satellite Orbits", Springer
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from nova.core.constants import (
    EARTH_MU,
    EARTH_RADIUS_EQ,
    EARTH_J2,
    G,
)


# ---------------------------------------------------------------------------
# Gravitational body descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GravBody:
    """
    Point-mass gravitational body for N-body computation.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "Earth", "Moon").
    mu : float
        Standard gravitational parameter μ = GM [m³ s⁻²].
    position_eci : ndarray, shape (3,)
        Position of the body's centre of mass in ECI [m].
        For the primary body (Earth), this is typically [0, 0, 0].
    """
    name: str
    mu: float
    position_eci: np.ndarray   # shape (3,), float64

    def __post_init__(self) -> None:
        pos = self.position_eci
        if not isinstance(pos, np.ndarray):
            object.__setattr__(
                self, "position_eci",
                np.asarray(pos, dtype=np.float64)
            )
        if self.mu <= 0.0:
            raise ValueError(f"GravBody.mu must be > 0, got {self.mu!r}")


# Pre-built Earth body centred at ECI origin
EARTH_BODY = GravBody(
    name="Earth",
    mu=EARTH_MU,
    position_eci=np.zeros(3, dtype=np.float64),
)


# ---------------------------------------------------------------------------
# Keplerian orbital elements
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeplerianElements:
    """
    Classical six Keplerian orbital elements computed from an ECI state.

    All angles in radians. Distances in metres.

    Attributes
    ----------
    semi_major_axis : float
        a [m]. Negative for hyperbolic trajectories (e > 1).
    eccentricity : float
        e [-]. 0 = circular, 0 < e < 1 = elliptic,
               e = 1 = parabolic, e > 1 = hyperbolic.
    inclination : float
        i [rad]. Angle between orbit plane and equatorial plane. [0, π].
    raan : float
        Ω [rad]. Right Ascension of the Ascending Node. [0, 2π).
    argument_of_periapsis : float
        ω [rad]. Angle from ascending node to periapsis. [0, 2π).
    true_anomaly : float
        ν [rad]. Current angular position from periapsis. [0, 2π).
    specific_angular_momentum : float
        h [m² s⁻¹]. ‖r × v‖.
    specific_orbital_energy : float
        ε [J kg⁻¹]. v²/2 − μ/r. Negative for bound orbits.
    period : float
        T [s]. Orbital period. Positive for elliptic; inf for parabolic/hyperbolic.
    apoapsis : float
        r_a [m]. Radius at apoapsis = a(1 + e). Inf if e ≥ 1.
    periapsis : float
        r_p [m]. Radius at periapsis = a(1 − e).
    """
    semi_major_axis:            float   # a [m]
    eccentricity:               float   # e [-]
    inclination:                float   # i [rad]
    raan:                       float   # Ω [rad]
    argument_of_periapsis:      float   # ω [rad]
    true_anomaly:               float   # ν [rad]
    specific_angular_momentum:  float   # h [m² s⁻¹]
    specific_orbital_energy:    float   # ε [J kg⁻¹]
    period:                     float   # T [s]
    apoapsis:                   float   # r_a [m]
    periapsis:                  float   # r_p [m]


def elements_from_state(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
    mu: float = EARTH_MU,
) -> KeplerianElements:
    """
    Convert an ECI state vector (r, v) to classical Keplerian orbital elements.

    Algorithm follows Bate, Mueller & White §2.4 ("The Determination of an
    Orbit from r and v"). All edge cases (circular, equatorial, hyperbolic)
    are handled numerically; true anomaly is always measured from periapsis.

    Parameters
    ----------
    position_eci : ndarray, shape (3,)
        ECI position [m].
    velocity_eci : ndarray, shape (3,)
        ECI velocity [m s⁻¹].
    mu : float
        Gravitational parameter [m³ s⁻²]. Default = EARTH_MU.

    Returns
    -------
    KeplerianElements

    Raises
    ------
    ValueError
        If the position vector is zero (spacecraft at body centre).
    """
    r_vec = np.asarray(position_eci, dtype=np.float64)
    v_vec = np.asarray(velocity_eci, dtype=np.float64)

    r_mag = float(np.linalg.norm(r_vec))
    v_mag = float(np.linalg.norm(v_vec))

    if r_mag < 1.0:
        raise ValueError(
            f"Position vector magnitude {r_mag:.3f} m is too small. "
            "Vehicle may be inside the planet."
        )

    # --- Specific angular momentum h = r × v ---
    h_vec = np.cross(r_vec, v_vec)
    h_mag = float(np.linalg.norm(h_vec))

    # --- Node vector N = k̂ × h (k̂ = ECI Z-axis) ---
    k_hat = np.array([0.0, 0.0, 1.0])
    N_vec = np.cross(k_hat, h_vec)
    N_mag = float(np.linalg.norm(N_vec))

    # --- Eccentricity vector e = (1/μ)[(v²−μ/r)r − (r·v)v] ---
    rdotv    = float(np.dot(r_vec, v_vec))
    e_vec    = ((v_mag**2 - mu / r_mag) * r_vec - rdotv * v_vec) / mu
    ecc      = float(np.linalg.norm(e_vec))

    # --- Specific orbital energy ε = v²/2 − μ/r ---
    energy = 0.5 * v_mag**2 - mu / r_mag

    # --- Semi-major axis a ---
    if abs(ecc - 1.0) < 1.0e-10:
        # Parabolic — semi-major axis undefined (use convention: ∞)
        a = math.inf
    else:
        a = -mu / (2.0 * energy)

    # --- Inclination i = arccos(h_z / |h|) ---
    inclination = math.acos(max(-1.0, min(1.0, h_vec[2] / h_mag))) if h_mag > 0.0 else 0.0

    # --- RAAN Ω = arccos(N_x / |N|) ---
    if N_mag < 1.0e-10:
        # Equatorial orbit — RAAN undefined; set to 0
        raan = 0.0
    else:
        raan = math.acos(max(-1.0, min(1.0, N_vec[0] / N_mag)))
        if N_vec[1] < 0.0:
            raan = 2.0 * math.pi - raan

    # --- Argument of periapsis ω = arccos((N · e) / (|N||e|)) ---
    if N_mag < 1.0e-10 or ecc < 1.0e-10:
        # Circular or equatorial — ω undefined; set to 0
        arg_pe = 0.0
    else:
        cos_w = float(np.dot(N_vec, e_vec)) / (N_mag * ecc)
        arg_pe = math.acos(max(-1.0, min(1.0, cos_w)))
        if e_vec[2] < 0.0:
            arg_pe = 2.0 * math.pi - arg_pe

    # --- True anomaly ν = arccos((e · r) / (|e||r|)) ---
    if ecc < 1.0e-10:
        # Circular — measure ν from ascending node
        if N_mag < 1.0e-10:
            # Circular equatorial — measure from +X
            cos_nu = r_vec[0] / r_mag
            nu = math.acos(max(-1.0, min(1.0, cos_nu)))
            if r_vec[1] < 0.0:
                nu = 2.0 * math.pi - nu
        else:
            cos_nu = float(np.dot(N_vec, r_vec)) / (N_mag * r_mag)
            nu = math.acos(max(-1.0, min(1.0, cos_nu)))
            if float(np.dot(N_vec, v_vec)) > 0.0:
                nu = 2.0 * math.pi - nu
    else:
        cos_nu = float(np.dot(e_vec, r_vec)) / (ecc * r_mag)
        nu = math.acos(max(-1.0, min(1.0, cos_nu)))
        if rdotv < 0.0:
            nu = 2.0 * math.pi - nu

    # --- Period T = 2π √(a³/μ) ---
    if a > 0.0 and ecc < 1.0:
        period = 2.0 * math.pi * math.sqrt(a**3 / mu)
    else:
        period = math.inf

    # --- Apoapsis and periapsis radii ---
    if ecc < 1.0:
        apoapsis  = a * (1.0 + ecc)
        periapsis = a * (1.0 - ecc)
    else:
        apoapsis  = math.inf
        periapsis = a * (1.0 - ecc) if not math.isinf(a) else math.inf

    return KeplerianElements(
        semi_major_axis=a,
        eccentricity=ecc,
        inclination=inclination,
        raan=raan,
        argument_of_periapsis=arg_pe,
        true_anomaly=nu,
        specific_angular_momentum=h_mag,
        specific_orbital_energy=energy,
        period=period,
        apoapsis=apoapsis,
        periapsis=periapsis,
    )


# ---------------------------------------------------------------------------
# Analytical Keplerian propagation (for validation only)
# ---------------------------------------------------------------------------

def kepler_period(semi_major_axis: float, mu: float = EARTH_MU) -> float:
    """
    Orbital period via Kepler's 3rd law.

    T = 2π √(a³ / μ)

    Parameters
    ----------
    semi_major_axis : float
        Semi-major axis [m]. Must be > 0 (elliptic orbit).
    mu : float
        Gravitational parameter [m³ s⁻²].

    Returns
    -------
    float
        Period [s].
    """
    if semi_major_axis <= 0.0:
        raise ValueError(f"semi_major_axis must be > 0, got {semi_major_axis!r}")
    return 2.0 * math.pi * math.sqrt(semi_major_axis**3 / mu)


def vis_viva_speed(radius: float, semi_major_axis: float, mu: float = EARTH_MU) -> float:
    """
    Orbital speed at a given radius via the vis-viva equation.

    v = √(μ · (2/r − 1/a))

    Parameters
    ----------
    radius : float
        Current orbital radius [m].
    semi_major_axis : float
        Semi-major axis [m]. Use r for circular orbit.
    mu : float
        Gravitational parameter [m³ s⁻²].

    Returns
    -------
    float
        Speed [m s⁻¹].
    """
    val = mu * (2.0 / radius - 1.0 / semi_major_axis)
    if val < 0.0:
        raise ValueError(
            f"Vis-viva argument is negative ({val:.3e}): "
            f"r={radius:.1f} m, a={semi_major_axis:.1f} m — "
            "check that r ≤ 2a for a bound orbit."
        )
    return math.sqrt(val)


def circular_orbit_speed(radius: float, mu: float = EARTH_MU) -> float:
    """
    Speed for a circular orbit at the given radius.

    v_c = √(μ / r)

    Parameters
    ----------
    radius : float
        Orbital radius from planet centre [m].
    mu : float
        Gravitational parameter [m³ s⁻²].

    Returns
    -------
    float
        Circular speed [m s⁻¹].
    """
    return math.sqrt(mu / radius)


def hohmann_delta_v(r1: float, r2: float, mu: float = EARTH_MU) -> tuple[float, float]:
    """
    Compute the two Δv burns for a Hohmann transfer between two circular orbits.

    Parameters
    ----------
    r1 : float
        Initial circular orbit radius [m].
    r2 : float
        Target circular orbit radius [m].
    mu : float
        Gravitational parameter [m³ s⁻²].

    Returns
    -------
    (dv1, dv2) : tuple of float
        First and second Δv burns [m s⁻¹]. Both positive (prograde).

    Derivation
    ----------
    Transfer ellipse semi-major axis: a_t = (r1 + r2) / 2

    v1_circ = √(μ/r1)         circular speed at r1
    v_pe    = √(μ(2/r1−1/aₜ)) transfer orbit speed at periapsis
    Δv₁     = v_pe − v1_circ  (prograde burn to enter transfer)

    v2_circ = √(μ/r2)
    v_ap    = √(μ(2/r2−1/aₜ)) transfer orbit speed at apoapsis
    Δv₂     = v2_circ − v_ap  (prograde burn to circularise at r2)
    """
    a_t   = 0.5 * (r1 + r2)
    v1c   = circular_orbit_speed(r1, mu)
    v2c   = circular_orbit_speed(r2, mu)
    v_pe  = vis_viva_speed(r1, a_t, mu)
    v_ap  = vis_viva_speed(r2, a_t, mu)
    dv1   = v_pe - v1c
    dv2   = v2c - v_ap
    return dv1, dv2


# ---------------------------------------------------------------------------
# N-body gravitational acceleration
# ---------------------------------------------------------------------------

def gravity_acceleration(
    position_eci: np.ndarray,
    bodies: Sequence[GravBody],
    include_j2: bool = True,
    planet_radius_eq: float = EARTH_RADIUS_EQ,
    j2_coefficient: float = EARTH_J2,
    primary_mu: float = EARTH_MU,
) -> np.ndarray:
    """
    Compute the total gravitational acceleration on the vehicle.

    Sums point-mass contributions from all provided bodies, plus an optional
    J2 oblateness perturbation from the primary body (Earth).

    Parameters
    ----------
    position_eci : ndarray, shape (3,)
        Vehicle position in ECI [m].
    bodies : sequence of GravBody
        List of gravitational bodies to include. Must contain at least one.
    include_j2 : bool
        If True, add the J2 zonal harmonic perturbation from the primary body.
        Only valid when the primary body is centred at ECI origin.
    planet_radius_eq : float
        Equatorial radius of the primary body [m]. Used in J2 term.
    j2_coefficient : float
        J2 coefficient (dimensionless). Default = EARTH_J2.
    primary_mu : float
        μ of the primary body [m³ s⁻²]. Used in J2 term.

    Returns
    -------
    ndarray, shape (3,), float64
        Total gravitational acceleration [m s⁻²] in ECI frame.

    Notes
    -----
    The J2 perturbation accelerations in ECI are:

        a_x = −(3/2) J2 (μ/r²)(R_eq/r)² (1 − 5(z/r)²) (x/r)
        a_y = −(3/2) J2 (μ/r²)(R_eq/r)² (1 − 5(z/r)²) (y/r)
        a_z = −(3/2) J2 (μ/r²)(R_eq/r)² (3 − 5(z/r)²) (z/r)

    Reference: Vallado §8.7.1
    """
    r_veh = np.asarray(position_eci, dtype=np.float64)
    a_total = np.zeros(3, dtype=np.float64)

    for body in bodies:
        r_body = np.asarray(body.position_eci, dtype=np.float64)
        r_rel  = r_veh - r_body           # vector from body to vehicle
        r_mag  = float(np.linalg.norm(r_rel))

        if r_mag < 1.0:
            # Vehicle is inside body — physically undefined; skip
            continue

        # Point-mass term: a = −μ / r³ · r_rel
        a_total += (-body.mu / r_mag**3) * r_rel

    if include_j2 and len(bodies) > 0:
        # J2 perturbation (primary body assumed at ECI origin)
        r_mag = float(np.linalg.norm(r_veh))
        if r_mag > 1.0:
            x, y, z = r_veh[0], r_veh[1], r_veh[2]
            ratio   = planet_radius_eq / r_mag
            coeff   = -1.5 * j2_coefficient * (primary_mu / r_mag**2) * ratio**2
            z_r     = z / r_mag
            five_z2 = 5.0 * z_r**2

            a_total[0] += coeff * (1.0 - five_z2) * (x / r_mag)
            a_total[1] += coeff * (1.0 - five_z2) * (y / r_mag)
            a_total[2] += coeff * (3.0 - five_z2) * (z / r_mag)

    return a_total


def gravity_force(
    position_eci: np.ndarray,
    mass: float,
    bodies: Sequence[GravBody] = (EARTH_BODY,),
    include_j2: bool = True,
) -> np.ndarray:
    """
    Gravitational force on a vehicle of given mass.

    F_grav = m · a_grav

    Parameters
    ----------
    position_eci : ndarray, shape (3,)
        Vehicle ECI position [m].
    mass : float
        Vehicle mass [kg].
    bodies : sequence of GravBody
        Gravitational bodies. Default: Earth only.
    include_j2 : bool
        Include J2 oblateness perturbation.

    Returns
    -------
    ndarray, shape (3,), float64
        Gravitational force [N] in ECI frame.
    """
    a_grav = gravity_acceleration(
        position_eci, bodies, include_j2=include_j2
    )
    return float(mass) * a_grav


# ---------------------------------------------------------------------------
# Maneuver Δv helper (HUD maneuver node support)
# ---------------------------------------------------------------------------

def delta_v_budget(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
    target_velocity_eci: np.ndarray,
) -> float:
    """
    Compute the Δv magnitude required to go from the current velocity
    to a target velocity at the same position (instantaneous burn assumption).

    Δv = ‖v_target − v_current‖

    Parameters
    ----------
    position_eci, velocity_eci : ndarray, shape (3,)
        Current state in ECI.
    target_velocity_eci : ndarray, shape (3,)
        Desired velocity after burn in ECI [m s⁻¹].

    Returns
    -------
    float
        Δv magnitude [m s⁻¹].
    """
    dv_vec = np.asarray(target_velocity_eci) - np.asarray(velocity_eci)
    return float(np.linalg.norm(dv_vec))
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_orbital.py
==========================
Unit tests for nova.physics.orbital.

Validation criteria (per architecture spec §7):
  - Kepler period vs vis-viva: T = 2π√(a³/μ) matches state-derived period.
  - elements_from_state → period → compare against direct formula.
  - Circular orbit: e ≈ 0, a = r, T = 2π√(r³/μ).
  - Hohmann Δv budget vs analytical closed form.
  - N-body: two equal masses at symmetric positions cancel.
  - J2: perturbation adds non-zero acceleration in polar orbit.
  - gravity_force = m · gravity_acceleration.
"""

import math
import pytest
import numpy as np

from nova.physics.orbital import (
    elements_from_state,
    KeplerianElements,
    GravBody,
    EARTH_BODY,
    kepler_period,
    vis_viva_speed,
    circular_orbit_speed,
    hohmann_delta_v,
    gravity_acceleration,
    gravity_force,
    delta_v_budget,
)
from nova.core.constants import (
    EARTH_MU,
    EARTH_RADIUS_EQ,
    EARTH_RADIUS_MEAN,
    EARTH_J2,
    DEG_TO_RAD,
)


# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
ELEMENT_TOL   = 1.0e-6   # relative tolerance for orbital elements
ANGLE_TOL_RAD = 1.0e-8   # absolute tolerance for angular elements [rad]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circular_state(
    altitude_m: float,
    inclination_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct an ECI (r, v) for a circular orbit at given altitude.

    Orbit is in the equatorial plane if inclination = 0.
    For nonzero inclination, the spacecraft is placed at the ascending node.
    """
    r_mag = EARTH_RADIUS_MEAN + altitude_m
    v_mag = circular_orbit_speed(r_mag)

    inc   = inclination_deg * DEG_TO_RAD
    # Position: on +X axis
    r_vec = np.array([r_mag, 0.0, 0.0], dtype=np.float64)
    # Velocity: in the Y-Z plane at the specified inclination
    v_vec = np.array([0.0, v_mag * math.cos(inc), v_mag * math.sin(inc)], dtype=np.float64)
    return r_vec, v_vec


# ---------------------------------------------------------------------------
# 1. GravBody validation
# ---------------------------------------------------------------------------

class TestGravBody:

    def test_earth_body_sentinel(self):
        assert EARTH_BODY.name == "Earth"
        assert abs(EARTH_BODY.mu - EARTH_MU) < 1.0

    def test_negative_mu_raises(self):
        with pytest.raises(ValueError, match="mu"):
            GravBody("Bad", mu=-1.0, position_eci=np.zeros(3))

    def test_zero_mu_raises(self):
        with pytest.raises(ValueError, match="mu"):
            GravBody("Bad", mu=0.0, position_eci=np.zeros(3))

    def test_list_coerced_to_ndarray(self):
        b = GravBody("Test", mu=1e14, position_eci=[0.0, 0.0, 0.0])
        assert isinstance(b.position_eci, np.ndarray)


# ---------------------------------------------------------------------------
# 2. Keplerian elements — circular equatorial orbit
# ---------------------------------------------------------------------------

class TestCircularEquatorialOrbit:
    """A circular equatorial orbit is the simplest case: e≈0, i≈0."""

    @pytest.fixture
    def leo_elements(self) -> KeplerianElements:
        r, v = _circular_state(400_000.0, inclination_deg=0.0)
        return elements_from_state(r, v)

    def test_eccentricity_near_zero(self, leo_elements):
        assert leo_elements.eccentricity < 1.0e-8, \
            f"Circular orbit eccentricity = {leo_elements.eccentricity:.2e}"

    def test_semi_major_axis_equals_radius(self, leo_elements):
        r_mag = EARTH_RADIUS_MEAN + 400_000.0
        rel_err = abs(leo_elements.semi_major_axis - r_mag) / r_mag
        assert rel_err < 1.0e-6, \
            f"a={leo_elements.semi_major_axis:.1f} m, expected r={r_mag:.1f} m"

    def test_inclination_near_zero(self, leo_elements):
        assert abs(leo_elements.inclination) < 1.0e-8

    def test_period_matches_kepler_formula(self, leo_elements):
        r_mag = EARTH_RADIUS_MEAN + 400_000.0
        T_expected = kepler_period(r_mag)
        rel_err = abs(leo_elements.period - T_expected) / T_expected
        assert rel_err < 1.0e-6, \
            f"Period: {leo_elements.period:.3f} s vs expected {T_expected:.3f} s"

    def test_specific_energy_negative(self, leo_elements):
        """Bound orbit → specific orbital energy must be negative."""
        assert leo_elements.specific_orbital_energy < 0.0

    def test_apoapsis_periapsis_equal_for_circular(self, leo_elements):
        """e≈0 → apoapsis ≈ periapsis ≈ a."""
        diff = abs(leo_elements.apoapsis - leo_elements.periapsis)
        assert diff < 1.0, f"Apoapsis-periapsis gap for circular orbit: {diff:.3f} m"


# ---------------------------------------------------------------------------
# 3. Keplerian elements — inclined orbit
# ---------------------------------------------------------------------------

class TestInclinedOrbit:

    @pytest.fixture
    def iss_like(self) -> KeplerianElements:
        """ISS-like: ~400 km, 51.6° inclination."""
        r, v = _circular_state(400_000.0, inclination_deg=51.6)
        return elements_from_state(r, v)

    def test_inclination_correct(self, iss_like):
        expected = 51.6 * DEG_TO_RAD
        assert abs(iss_like.inclination - expected) < ANGLE_TOL_RAD * 100, \
            f"Inclination: {math.degrees(iss_like.inclination):.4f}° expected 51.6°"

    def test_still_circular(self, iss_like):
        assert iss_like.eccentricity < 1.0e-7

    def test_period_independent_of_inclination(self, iss_like):
        """Period depends only on a, not inclination."""
        r_mag = EARTH_RADIUS_MEAN + 400_000.0
        T_expected = kepler_period(r_mag)
        rel_err = abs(iss_like.period - T_expected) / T_expected
        assert rel_err < 1.0e-6


# ---------------------------------------------------------------------------
# 4. Vis-viva equation
# ---------------------------------------------------------------------------

class TestVisViva:

    def test_circular_orbit_speed(self):
        """vis_viva with a=r gives circular speed: v = √(μ/r)."""
        r = EARTH_RADIUS_MEAN + 400_000.0
        v_vv  = vis_viva_speed(r, r)
        v_circ = circular_orbit_speed(r)
        assert abs(v_vv - v_circ) < 1.0e-6, \
            f"Vis-viva circular: {v_vv:.6f} vs {v_circ:.6f}"

    def test_vis_viva_at_periapsis(self):
        """
        For an elliptic orbit (r_p=6800 km, r_a=8000 km):
          a = (r_p + r_a) / 2 = 7400 km
          v_p = √(μ(2/r_p − 1/a))
        """
        r_p = 6_800_000.0
        r_a = 8_000_000.0
        a   = (r_p + r_a) / 2.0
        v_p = vis_viva_speed(r_p, a)
        # Cross-check with energy: v² = μ(2/r − 1/a)
        v_expected = math.sqrt(EARTH_MU * (2.0/r_p - 1.0/a))
        assert abs(v_p - v_expected) < 1.0e-6

    def test_invalid_vis_viva_raises(self):
        """r > 2a for a bound orbit is physically impossible."""
        r = 10_000_000.0
        a =  3_000_000.0   # a < r/2 → negative argument
        with pytest.raises(ValueError, match="Vis-viva argument"):
            vis_viva_speed(r, a)

    def test_kepler_period_formula(self):
        """T = 2π√(a³/μ) — known value at 400 km LEO."""
        a = EARTH_RADIUS_MEAN + 400_000.0
        T = kepler_period(a)
        # Cross-check: T should be ~5559 s at this altitude
        T_expected = 2.0 * math.pi * math.sqrt(a**3 / EARTH_MU)
        assert abs(T - T_expected) < 0.001

    def test_kepler_period_raises_for_negative_a(self):
        with pytest.raises(ValueError, match="semi_major_axis"):
            kepler_period(-1_000_000.0)


# ---------------------------------------------------------------------------
# 5. Hohmann transfer
# ---------------------------------------------------------------------------

class TestHohmannTransfer:

    def test_delta_v_values(self):
        """
        Known Hohmann: 400 km LEO → 800 km LEO.
        Analytical Δv₁ and Δv₂ computed from vis-viva.
        """
        r1 = EARTH_RADIUS_MEAN + 400_000.0
        r2 = EARTH_RADIUS_MEAN + 800_000.0
        dv1, dv2 = hohmann_delta_v(r1, r2)

        # Cross-check with analytical formula
        a_t   = 0.5 * (r1 + r2)
        v1c   = circular_orbit_speed(r1)
        v2c   = circular_orbit_speed(r2)
        v_pe  = vis_viva_speed(r1, a_t)
        v_ap  = vis_viva_speed(r2, a_t)
        dv1_expected = v_pe - v1c
        dv2_expected = v2c - v_ap

        assert abs(dv1 - dv1_expected) < 1.0e-6
        assert abs(dv2 - dv2_expected) < 1.0e-6

    def test_both_burns_positive(self):
        """Both Δv burns in a prograde Hohmann must be positive."""
        r1 = EARTH_RADIUS_MEAN + 200_000.0
        r2 = EARTH_RADIUS_MEAN + 35_786_000.0   # GEO
        dv1, dv2 = hohmann_delta_v(r1, r2)
        assert dv1 > 0.0
        assert dv2 > 0.0

    def test_same_orbit_zero_delta_v(self):
        """Transfer from orbit to itself should require zero Δv."""
        r = EARTH_RADIUS_MEAN + 400_000.0
        dv1, dv2 = hohmann_delta_v(r, r)
        assert abs(dv1) < 1.0e-6
        assert abs(dv2) < 1.0e-6


# ---------------------------------------------------------------------------
# 6. N-body gravity
# ---------------------------------------------------------------------------

class TestGravityAcceleration:

    def test_single_body_magnitude(self):
        """
        At 400 km altitude on the +X axis, gravity ≈ μ/r² directed toward −X.
        """
        r_mag = EARTH_RADIUS_MEAN + 400_000.0
        r_vec = np.array([r_mag, 0.0, 0.0], dtype=np.float64)
        a_vec = gravity_acceleration(r_vec, [EARTH_BODY], include_j2=False)
        a_mag = float(np.linalg.norm(a_vec))
        a_expected = EARTH_MU / r_mag**2
        rel_err = abs(a_mag - a_expected) / a_expected
        assert rel_err < 1.0e-10

    def test_single_body_direction(self):
        """Gravity on +X axis must point in −X direction."""
        r_vec = np.array([7_000_000.0, 0.0, 0.0], dtype=np.float64)
        a_vec = gravity_acceleration(r_vec, [EARTH_BODY], include_j2=False)
        assert a_vec[0] < 0.0
        assert abs(a_vec[1]) < 1.0e-20
        assert abs(a_vec[2]) < 1.0e-20

    def test_two_symmetric_bodies_cancel(self):
        """
        Two identical masses at +X and −X should produce zero net gravity
        when the vehicle is at the origin.

        Note: NOVA convention is vehicle at origin → both bodies at ±r.
        With vehicle at ECI origin and bodies symmetrically placed, forces cancel.
        """
        mu_test = 1.0e12
        body_pos = 1_000_000.0
        b1 = GravBody("B1", mu=mu_test, position_eci=np.array([ body_pos, 0.0, 0.0]))
        b2 = GravBody("B2", mu=mu_test, position_eci=np.array([-body_pos, 0.0, 0.0]))
        r_veh = np.zeros(3, dtype=np.float64)
        a_vec = gravity_acceleration(r_veh, [b1, b2], include_j2=False)
        assert np.linalg.norm(a_vec) < 1.0e-20, \
            f"Symmetric bodies did not cancel: {a_vec}"

    def test_gravity_force_equals_mass_times_accel(self):
        """gravity_force = m · gravity_acceleration must hold exactly."""
        r_vec = np.array([7_000_000.0, 0.0, 0.0], dtype=np.float64)
        mass  = 1500.0
        a_vec = gravity_acceleration(r_vec, [EARTH_BODY], include_j2=False)
        F_vec = gravity_force(r_vec, mass, [EARTH_BODY], include_j2=False)
        assert np.allclose(F_vec, mass * a_vec, rtol=1.0e-12)

    def test_inverse_square_law(self):
        """
        Doubling the distance should reduce gravity magnitude by factor of 4.
        """
        r1 = 7_000_000.0
        r2 = 14_000_000.0
        r1_vec = np.array([r1, 0.0, 0.0])
        r2_vec = np.array([r2, 0.0, 0.0])
        a1 = float(np.linalg.norm(gravity_acceleration(r1_vec, [EARTH_BODY], include_j2=False)))
        a2 = float(np.linalg.norm(gravity_acceleration(r2_vec, [EARTH_BODY], include_j2=False)))
        ratio = a1 / a2
        assert abs(ratio - 4.0) < 1.0e-10, f"Inverse-square ratio: {ratio:.8f}, expected 4.0"

    def test_empty_bodies_returns_zero(self):
        r_vec = np.array([7_000_000.0, 0.0, 0.0])
        a_vec = gravity_acceleration(r_vec, [], include_j2=False)
        assert np.allclose(a_vec, 0.0)


# ---------------------------------------------------------------------------
# 7. J2 perturbation
# ---------------------------------------------------------------------------

class TestJ2Perturbation:

    def test_j2_adds_nonzero_acceleration(self):
        """J2 must produce a non-zero perturbation in a polar orbit."""
        r_vec = np.array([0.0, 0.0, 7_000_000.0], dtype=np.float64)  # over pole
        a_no_j2 = gravity_acceleration(r_vec, [EARTH_BODY], include_j2=False)
        a_j2    = gravity_acceleration(r_vec, [EARTH_BODY], include_j2=True)
        delta_a = float(np.linalg.norm(a_j2 - a_no_j2))
        assert delta_a > 1.0e-5, \
            f"J2 perturbation at polar position too small: {delta_a:.2e} m/s²"

    def test_j2_magnitude_order(self):
        """
        At r=7000 km, J2 perturbation should be ~10⁻⁴ of central body term.
        J2 ≈ 1.083e-3, (R_eq/r)² ≈ (6378/7000)² ≈ 0.83 → correction ~3e-4 of g.
        """
        r_vec = np.array([7_000_000.0, 0.0, 0.0], dtype=np.float64)
        a_no_j2 = gravity_acceleration(r_vec, [EARTH_BODY], include_j2=False)
        a_j2    = gravity_acceleration(r_vec, [EARTH_BODY], include_j2=True)
        ratio = float(np.linalg.norm(a_j2 - a_no_j2)) / float(np.linalg.norm(a_no_j2))
        assert 1.0e-4 < ratio < 1.0e-2, \
            f"J2/g0 ratio unexpected: {ratio:.4e}"

    def test_j2_equatorial_radial_only(self):
        """
        On the equatorial plane (z=0), the J2 perturbation should be purely
        radial (no Z component). J2 term z-component = coeff * (3 − 5*(0)²) * 0 = 0.
        """
        r_vec = np.array([7_000_000.0, 0.0, 0.0], dtype=np.float64)
        a_j2  = gravity_acceleration(r_vec, [EARTH_BODY], include_j2=True)
        a_no  = gravity_acceleration(r_vec, [EARTH_BODY], include_j2=False)
        delta = a_j2 - a_no
        assert abs(delta[1]) < 1.0e-15 and abs(delta[2]) < 1.0e-15, \
            f"J2 at equator has unexpected off-radial components: {delta}"


# ---------------------------------------------------------------------------
# 8. Orbital elements from non-circular state
# ---------------------------------------------------------------------------

class TestEllipticOrbit:
    """
    Construct an elliptic orbit analytically and verify element extraction.
    Periapsis: 300 km altitude, apoapsis: 1000 km altitude.
    """

    @pytest.fixture
    def elliptic_state(self):
        r_p = EARTH_RADIUS_MEAN + 300_000.0
        r_a = EARTH_RADIUS_MEAN + 1_000_000.0
        a   = 0.5 * (r_p + r_a)
        v_p = vis_viva_speed(r_p, a)
        r_vec = np.array([r_p, 0.0, 0.0], dtype=np.float64)
        v_vec = np.array([0.0, v_p, 0.0], dtype=np.float64)
        return r_vec, v_vec, r_p, r_a, a

    def test_periapsis_radius(self, elliptic_state):
        r_vec, v_vec, r_p, r_a, a = elliptic_state
        el = elements_from_state(r_vec, v_vec)
        assert abs(el.periapsis - r_p) < 10.0, \
            f"Periapsis: {el.periapsis:.1f} m, expected {r_p:.1f} m"

    def test_apoapsis_radius(self, elliptic_state):
        r_vec, v_vec, r_p, r_a, a = elliptic_state
        el = elements_from_state(r_vec, v_vec)
        assert abs(el.apoapsis - r_a) < 10.0, \
            f"Apoapsis: {el.apoapsis:.1f} m, expected {r_a:.1f} m"

    def test_semi_major_axis(self, elliptic_state):
        r_vec, v_vec, r_p, r_a, a = elliptic_state
        el = elements_from_state(r_vec, v_vec)
        rel_err = abs(el.semi_major_axis - a) / a
        assert rel_err < 1.0e-8

    def test_eccentricity(self, elliptic_state):
        r_vec, v_vec, r_p, r_a, a = elliptic_state
        el = elements_from_state(r_vec, v_vec)
        e_expected = (r_a - r_p) / (r_a + r_p)
        assert abs(el.eccentricity - e_expected) < 1.0e-8


# ---------------------------------------------------------------------------
# 9. delta_v_budget
# ---------------------------------------------------------------------------

class TestDeltaVBudget:

    def test_zero_for_same_velocity(self):
        r = np.array([7_000_000.0, 0.0, 0.0])
        v = np.array([0.0, 7500.0, 0.0])
        dv = delta_v_budget(r, v, v)
        assert abs(dv) < 1.0e-12

    def test_known_delta_v(self):
        r = np.array([7_000_000.0, 0.0, 0.0])
        v = np.array([0.0, 7500.0, 0.0])
        v_target = np.array([0.0, 7600.0, 0.0])
        dv = delta_v_budget(r, v, v_target)
        assert abs(dv - 100.0) < 1.0e-10
