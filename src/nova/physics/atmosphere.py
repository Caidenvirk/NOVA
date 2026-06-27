"""
nova.physics.atmosphere
=======================
Atmospheric state solver for Project NOVA.

Implements the International Standard Atmosphere (ISA) 1976 / US Standard
Atmosphere 1976 with the correct multi-layer temperature profile:

  Layer 0: Troposphere        0 – 11 000 m   L = −6.5  K/km
  Layer 1: Lower stratosphere 11 000 – 20 000 m  isothermal (216.65 K)
  Layer 2: Middle stratosphere 20 000 – 32 000 m  L = +1.0  K/km
  Layer 3: Upper stratosphere  32 000 – 47 000 m  L = +2.8  K/km
  Layer 4: Above 47 000 m     isothermal (270.65 K) + exponential decay

Each layer uses the correct hydrostatic pressure formula for its lapse rate:
  Constant lapse:    P = P_base * (T / T_base)^(g₀ / (-L · R_air))
  Isothermal (L=0):  P = P_base * exp(-g₀ · Δh / (R_air · T))

All base pressures and temperatures are ISA 1976 tabulated values, NOT
derived from lower layers, to prevent error accumulation.

Output quantities (strict SI):
  ρ   [kg m⁻³]   atmospheric density
  T   [K]         static temperature
  P   [Pa]        static pressure
  a   [m s⁻¹]    speed of sound
  μ   [Pa·s]      dynamic viscosity (Sutherland's law)

References
----------
- US Standard Atmosphere, 1976 (NOAA-S/T 76-1562)
- ICAO Doc 7488, Manual of the ICAO Standard Atmosphere, 3rd ed.
- Anderson, "Introduction to Flight", 8th ed., Appendix A
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nova.core.constants import (
    EARTH_T_SL,
    EARTH_P_SL,
    EARTH_RHO_SL,
    EARTH_A_SL,
    EARTH_RADIUS_MEAN,
    R_AIR,
    GAMMA_AIR,
    STD_GRAVITY,
    ISA_TROPOPAUSE_ALT,
    ISA_TROPOPAUSE_TEMP,
)


# ---------------------------------------------------------------------------
# ISA 1976 layer definitions
# ---------------------------------------------------------------------------
# Each tuple: (base_altitude_m, base_temp_K, base_pressure_Pa, lapse_K_per_m)
# Base pressures are ISA 1976 authoritative tabulated values — NOT derived.
# Lapse rate sign convention: negative = cooling with altitude.

_ISA_LAYERS = [
    (    0.0,  288.15, 101_325.0,   -6.5e-3),   # Layer 0: troposphere
    (11_000.0, 216.65,  22_632.1,    0.0   ),   # Layer 1: lower stratosphere (isothermal)
    (20_000.0, 216.65,   5_474.9,   +1.0e-3),   # Layer 2: middle stratosphere
    (32_000.0, 228.65,     868.019, +2.8e-3),   # Layer 3: upper stratosphere
    (47_000.0, 270.65,     110.906,  0.0   ),   # Layer 4: stratopause (isothermal)
]


def _isa_layer_state(altitude: float) -> tuple[float, float]:
    """
    Compute ISA temperature and pressure at the given altitude using the
    appropriate layer formula.

    Parameters
    ----------
    altitude : float
        Geodetic altitude [m]. Extrapolation below 0 m uses Layer 0 formula.

    Returns
    -------
    (T, P) : tuple of float
        Temperature [K] and pressure [Pa].
    """
    # Find the applicable layer (scan from top down)
    h_base, T_base, P_base, L = _ISA_LAYERS[0]
    for (h_b, T_b, P_b, lapse) in reversed(_ISA_LAYERS):
        if altitude >= h_b:
            h_base, T_base, P_base, L = h_b, T_b, P_b, lapse
            break

    dh = altitude - h_base

    if abs(L) < 1.0e-12:
        # Isothermal layer: P = P_base * exp(-g₀·Δh / (R·T))
        T = T_base
        P = P_base * math.exp(-STD_GRAVITY * dh / (R_AIR * T_base))
    else:
        # Linear lapse: T = T_base + L·Δh
        T = T_base + L * dh
        T = max(T, 1.0)   # numerical guard
        exponent = STD_GRAVITY / (-L * R_AIR)
        P = P_base * (T / T_base) ** exponent

    P = max(P, 0.0)
    return T, P


# ---------------------------------------------------------------------------
# Sutherland's law for dynamic viscosity
# ---------------------------------------------------------------------------

_SUTHERLAND_C1: float = 1.458e-6   # kg m⁻¹ s⁻¹ K⁻⁰·⁵
_SUTHERLAND_S:  float = 110.4      # K


def sutherland_viscosity(temperature: float) -> float:
    """
    Dynamic viscosity of air [Pa·s] via Sutherland's law.

    μ = C₁ T^(3/2) / (T + S)

    Valid for 170 K < T < 1900 K.
    """
    T = max(temperature, 1.0)
    return _SUTHERLAND_C1 * (T ** 1.5) / (T + _SUTHERLAND_S)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtmosphericState:
    """
    Complete thermodynamic state of the atmosphere at a given altitude.

    All fields are in strict SI units.
    """
    altitude:          float   # [m]
    density:           float   # [kg m⁻³]
    temperature:       float   # [K]
    pressure:          float   # [Pa]
    speed_of_sound:    float   # [m s⁻¹]
    dynamic_viscosity: float   # [Pa·s]

    def mach(self, airspeed: float) -> float:
        """Mach number M = v / a for a given airspeed [m s⁻¹]."""
        if self.speed_of_sound < 1.0e-12:
            return 0.0
        return airspeed / self.speed_of_sound

    @property
    def dynamic_pressure_at_speed(self):
        """Return callable q(v) = ½ρv² [Pa] for a given airspeed v [m s⁻¹]."""
        rho = self.density
        return lambda v: 0.5 * rho * v * v


# ---------------------------------------------------------------------------
# Core atmosphere function
# ---------------------------------------------------------------------------

def atmosphere(altitude: float) -> AtmosphericState:
    """
    Compute the ISA 1976 atmospheric state at a given geodetic altitude.

    Parameters
    ----------
    altitude : float
        Geodetic altitude above mean sea level [m].
        Accepts negative values (extrapolation below MSL via Layer 0).

    Returns
    -------
    AtmosphericState
        Frozen snapshot of all atmospheric quantities at this altitude.
    """
    T, P  = _isa_layer_state(altitude)
    rho   = P / (R_AIR * T)
    a     = math.sqrt(GAMMA_AIR * R_AIR * T)
    mu    = sutherland_viscosity(T)

    return AtmosphericState(
        altitude=altitude,
        density=rho,
        temperature=T,
        pressure=P,
        speed_of_sound=a,
        dynamic_viscosity=mu,
    )


# ---------------------------------------------------------------------------
# Vector form — accepts ECI position
# ---------------------------------------------------------------------------

def atmosphere_from_eci(
    position_eci: np.ndarray,
    planet_radius: float,
) -> AtmosphericState:
    """
    Compute atmospheric state from an ECI position vector.

    Uses spherical-Earth approximation: altitude = ‖r_eci‖ − R_planet.

    Parameters
    ----------
    position_eci : ndarray, shape (3,)
        ECI position [m].
    planet_radius : float
        Reference planet radius [m].
    """
    r_mag    = float(np.linalg.norm(position_eci))
    altitude = r_mag - planet_radius
    return atmosphere(altitude)


# ---------------------------------------------------------------------------
# Standalone aerodynamic helpers
# ---------------------------------------------------------------------------

def dynamic_pressure(rho: float, airspeed: float) -> float:
    """q = ½ ρ v²  [Pa]."""
    return 0.5 * rho * airspeed * airspeed


def mach_number(airspeed: float, speed_of_sound: float) -> float:
    """
    M = v / a  (dimensionless).

    Raises
    ------
    ValueError if speed_of_sound ≤ 0.
    """
    if speed_of_sound <= 0.0:
        raise ValueError(
            f"speed_of_sound must be > 0, got {speed_of_sound!r}"
        )
    return airspeed / speed_of_sound


def prandtl_glauert_factor(mach: float) -> float:
    """
    Prandtl-Glauert compressibility correction β = 1 / √(1 − M²).

    Valid for 0 ≤ M < 1.

    Raises
    ------
    ValueError if mach not in [0, 1).
    """
    if not (0.0 <= mach < 1.0):
        raise ValueError(
            f"Prandtl-Glauert correction requires 0 ≤ M < 1, got M={mach:.4f}"
        )
    return 1.0 / math.sqrt(1.0 - mach * mach)
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_atmosphere.py
=============================
Unit tests for nova.physics.atmosphere.

Validation criteria (per architecture spec §7):
  - ρ(0), ρ(11 000), ρ(25 000) m match ISA 1976 table within 0.1%.
  - Prandtl-Glauert at M=0.5 → β = 1/√(1−0.25) = 1.15470...
  - AtmosphericState frozen; all fields positive above MSL.
  - Speed of sound = √(γ R T) at every altitude.
  - Mach number computation and dynamic pressure.

ISA reference values are taken from the US Standard Atmosphere 1976
(NOAA-S/T 76-1562) multi-layer analytical formulation, not from rounded
summary tables.  The model implements five ISA layers to match the standard
within 0.1% from 0 to 47 km.
"""

import math
import pytest
import numpy as np

from nova.physics.atmosphere import (
    atmosphere,
    atmosphere_from_eci,
    dynamic_pressure,
    mach_number,
    prandtl_glauert_factor,
    sutherland_viscosity,
    AtmosphericState,
)
from nova.core.constants import (
    EARTH_T_SL, EARTH_P_SL, EARTH_RHO_SL,
    EARTH_RADIUS_MEAN,
    R_AIR, GAMMA_AIR,
    ISA_TROPOPAUSE_TEMP,
)

# ---------------------------------------------------------------------------
# ISA 1976 authoritative reference table
# (altitude_m, rho_kg_m3, T_K, P_Pa, a_m_s)
# Density and speed of sound are derived from T and P via ideal gas law and
# a = sqrt(gamma*R*T); reference values here are the analytically-derived
# ISA values, not rounded single-source printouts.
# ---------------------------------------------------------------------------
ISA_TABLE = [
    (    0.0,  1.22500,  288.15, 101_325.0, 340.294),
    ( 1_000.0, 1.11170,  281.65,  89_874.6, 336.434),
    ( 5_000.0, 0.73640,  255.65,  54_048.3, 320.529),
    (11_000.0, 0.36391,  216.65,  22_632.1, 295.070),
    (15_000.0, 0.19367,  216.65,  12_044.5, 295.070),
    (20_000.0, 0.08803,  216.65,   5_474.9, 295.070),
    (25_000.0, 0.03947,  221.65,   2_511.0, 298.389),
    (30_000.0, 0.01801,  226.65,   1_172.0, 301.709),
]

_ISA_TOL = 0.001   # 0.1% relative tolerance


# ---------------------------------------------------------------------------
# 1. ISA table validation — architecture spec §7 requirement
# ---------------------------------------------------------------------------

class TestISAValues:
    """Validate atmosphere() against ISA 1976 authoritative values."""

    @pytest.mark.parametrize("alt,rho_ref,T_ref,P_ref,a_ref", ISA_TABLE)
    def test_density_within_0p1_percent(self, alt, rho_ref, T_ref, P_ref, a_ref):
        state   = atmosphere(alt)
        rel_err = abs(state.density - rho_ref) / rho_ref
        assert rel_err <= _ISA_TOL, (
            f"Density at h={alt:.0f} m: got {state.density:.6f}, "
            f"ref {rho_ref:.6f}, rel_err={rel_err*100:.4f}%"
        )

    @pytest.mark.parametrize("alt,rho_ref,T_ref,P_ref,a_ref", ISA_TABLE)
    def test_temperature_within_0p1_percent(self, alt, rho_ref, T_ref, P_ref, a_ref):
        state   = atmosphere(alt)
        rel_err = abs(state.temperature - T_ref) / T_ref
        assert rel_err <= _ISA_TOL, (
            f"Temperature at h={alt:.0f} m: got {state.temperature:.4f}, "
            f"ref {T_ref:.4f}, rel_err={rel_err*100:.4f}%"
        )

    @pytest.mark.parametrize("alt,rho_ref,T_ref,P_ref,a_ref", ISA_TABLE)
    def test_pressure_within_0p1_percent(self, alt, rho_ref, T_ref, P_ref, a_ref):
        state   = atmosphere(alt)
        rel_err = abs(state.pressure - P_ref) / P_ref
        assert rel_err <= _ISA_TOL, (
            f"Pressure at h={alt:.0f} m: got {state.pressure:.2f}, "
            f"ref {P_ref:.2f}, rel_err={rel_err*100:.4f}%"
        )

    @pytest.mark.parametrize("alt,rho_ref,T_ref,P_ref,a_ref", ISA_TABLE)
    def test_speed_of_sound_within_0p1_percent(self, alt, rho_ref, T_ref, P_ref, a_ref):
        state   = atmosphere(alt)
        rel_err = abs(state.speed_of_sound - a_ref) / a_ref
        assert rel_err <= _ISA_TOL, (
            f"Speed of sound at h={alt:.0f} m: got {state.speed_of_sound:.4f}, "
            f"ref {a_ref:.4f}, rel_err={rel_err*100:.4f}%"
        )


# ---------------------------------------------------------------------------
# 2. Speed of sound self-consistency: a = √(γ R T) at every altitude
# ---------------------------------------------------------------------------

class TestSpeedOfSoundConsistency:

    @pytest.mark.parametrize("alt", [0, 1000, 5000, 11000, 20000, 32000, 47000, 60000])
    def test_a_equals_sqrt_gamma_R_T(self, alt):
        state     = atmosphere(alt)
        a_derived = math.sqrt(GAMMA_AIR * R_AIR * state.temperature)
        assert abs(state.speed_of_sound - a_derived) < 1.0e-6, (
            f"a at {alt} m: got {state.speed_of_sound:.8f}, "
            f"derived {a_derived:.8f}"
        )


# ---------------------------------------------------------------------------
# 3. AtmosphericState contract
# ---------------------------------------------------------------------------

class TestAtmosphericStateContract:

    def test_returns_atmospheric_state(self):
        assert isinstance(atmosphere(0.0), AtmosphericState)

    def test_frozen_cannot_mutate(self):
        state = atmosphere(0.0)
        with pytest.raises(Exception):
            state.density = 999.0

    def test_altitude_stored_correctly(self):
        for h in [0.0, 5000.0, 11000.0, 25000.0]:
            assert atmosphere(h).altitude == h

    def test_all_fields_positive_above_msl(self):
        for h in [0, 100, 1000, 11000, 25000, 47000, 60000]:
            s = atmosphere(h)
            assert s.density        > 0.0, f"density ≤ 0 at h={h}"
            assert s.temperature    > 0.0
            assert s.pressure       > 0.0
            assert s.speed_of_sound > 0.0
            assert s.dynamic_viscosity > 0.0

    def test_density_decreases_monotonically(self):
        densities = [atmosphere(h).density for h in range(0, 80_001, 1000)]
        for i in range(1, len(densities)):
            assert densities[i] <= densities[i-1], (
                f"Density increased from h={i-1} to h={i} km"
            )

    def test_sea_level_matches_constants(self):
        s = atmosphere(0.0)
        assert abs(s.temperature - EARTH_T_SL) < 0.01
        assert abs(s.pressure    - EARTH_P_SL) < 1.0
        assert abs(s.density     - EARTH_RHO_SL) < 0.001

    def test_tropopause_temperature_isothermal_layer1(self):
        """11 000 – 20 000 m must be isothermal at 216.65 K."""
        for h in [11_001, 15_000, 19_999]:
            s = atmosphere(h)
            assert abs(s.temperature - ISA_TROPOPAUSE_TEMP) < 0.01, (
                f"T at {h} m = {s.temperature:.4f} K, expected 216.65 K"
            )

    def test_temperature_increases_in_layer2(self):
        """20 000 – 32 000 m: temperature must increase with altitude."""
        T20 = atmosphere(20_000.0).temperature
        T30 = atmosphere(30_000.0).temperature
        assert T30 > T20, f"T not increasing in layer 2: T20={T20:.2f}, T30={T30:.2f}"

    def test_mach_property(self):
        s = atmosphere(0.0)
        M = s.mach(s.speed_of_sound)
        assert abs(M - 1.0) < 1.0e-10

    def test_dynamic_pressure_callable(self):
        s   = atmosphere(0.0)
        q   = s.dynamic_pressure_at_speed(100.0)
        assert abs(q - 0.5 * s.density * 100.0**2) < 0.01


# ---------------------------------------------------------------------------
# 4. Below-MSL extrapolation
# ---------------------------------------------------------------------------

class TestNegativeAltitude:

    def test_density_above_sea_level_at_negative_alt(self):
        assert atmosphere(-500.0).density > atmosphere(0.0).density

    def test_negative_altitude_all_fields_positive(self):
        s = atmosphere(-500.0)
        assert s.density > 0.0
        assert s.temperature > 0.0
        assert s.pressure > 0.0


# ---------------------------------------------------------------------------
# 5. atmosphere_from_eci
# ---------------------------------------------------------------------------

class TestAtmosphereFromECI:

    def test_sea_level_eci(self):
        r_eci = np.array([EARTH_RADIUS_MEAN, 0.0, 0.0])
        s     = atmosphere_from_eci(r_eci, EARTH_RADIUS_MEAN)
        assert abs(s.altitude) < 1.0
        assert abs(s.density - EARTH_RHO_SL) < 0.001

    def test_400km_altitude(self):
        alt   = 400_000.0
        r_eci = np.array([EARTH_RADIUS_MEAN + alt, 0.0, 0.0])
        s     = atmosphere_from_eci(r_eci, EARTH_RADIUS_MEAN)
        assert abs(s.altitude - alt) < 1.0


# ---------------------------------------------------------------------------
# 6. Dynamic pressure helper
# ---------------------------------------------------------------------------

class TestDynamicPressure:

    def test_zero_speed(self):
        assert dynamic_pressure(1.225, 0.0) == 0.0

    def test_known_value(self):
        assert abs(dynamic_pressure(1.225, 100.0) - 6125.0) < 0.1

    def test_quadratic_in_speed(self):
        q1 = dynamic_pressure(1.0, 50.0)
        q2 = dynamic_pressure(1.0, 100.0)
        assert abs(q2 / q1 - 4.0) < 1.0e-10


# ---------------------------------------------------------------------------
# 7. Mach number helper
# ---------------------------------------------------------------------------

class TestMachNumber:

    def test_mach_one(self):
        a = atmosphere(0.0).speed_of_sound
        assert abs(mach_number(a, a) - 1.0) < 1.0e-12

    def test_mach_zero(self):
        assert mach_number(0.0, 340.0) == 0.0

    def test_supersonic(self):
        assert abs(mach_number(680.0, 340.0) - 2.0) < 1.0e-10

    def test_zero_speed_of_sound_raises(self):
        with pytest.raises(ValueError, match="speed_of_sound"):
            mach_number(100.0, 0.0)

    def test_negative_speed_of_sound_raises(self):
        with pytest.raises(ValueError, match="speed_of_sound"):
            mach_number(100.0, -340.0)


# ---------------------------------------------------------------------------
# 8. Prandtl-Glauert (architecture spec §7 explicit criterion)
# ---------------------------------------------------------------------------

class TestPrandtlGlauert:

    def test_mach_half_spec_value(self):
        """
        Architecture spec §7: M=0.5 → β = 1/√(1−0.25) = 1.15470...
        """
        beta     = prandtl_glauert_factor(0.5)
        expected = 1.0 / math.sqrt(1.0 - 0.25)
        assert abs(beta - expected) < 1.0e-12, (
            f"PG at M=0.5: got {beta:.10f}, expected {expected:.10f}"
        )

    def test_mach_zero_gives_one(self):
        assert abs(prandtl_glauert_factor(0.0) - 1.0) < 1.0e-15

    def test_mach_point_eight(self):
        beta     = prandtl_glauert_factor(0.8)
        expected = 1.0 / math.sqrt(1.0 - 0.64)
        assert abs(beta - expected) < 1.0e-10

    def test_mach_one_raises(self):
        with pytest.raises(ValueError, match="0 ≤ M < 1"):
            prandtl_glauert_factor(1.0)

    def test_mach_above_one_raises(self):
        with pytest.raises(ValueError, match="0 ≤ M < 1"):
            prandtl_glauert_factor(1.5)

    def test_negative_mach_raises(self):
        with pytest.raises(ValueError, match="0 ≤ M < 1"):
            prandtl_glauert_factor(-0.1)

    def test_beta_monotonically_increases_with_mach(self):
        machs = [0.0, 0.2, 0.4, 0.6, 0.79]
        betas = [prandtl_glauert_factor(m) for m in machs]
        for i in range(1, len(betas)):
            assert betas[i] > betas[i-1], f"β not increasing at M={machs[i]}"


# ---------------------------------------------------------------------------
# 9. Sutherland viscosity
# ---------------------------------------------------------------------------

class TestSutherlandViscosity:

    def test_sea_level_viscosity(self):
        """ISA 1976: μ ≈ 1.789 × 10⁻⁵ Pa·s at 288.15 K."""
        mu = sutherland_viscosity(288.15)
        assert abs(mu - 1.789e-5) < 2.0e-7

    def test_viscosity_positive_at_all_temps(self):
        for T in [150.0, 216.65, 288.15, 500.0, 1000.0]:
            assert sutherland_viscosity(T) > 0.0

    def test_viscosity_increases_with_temperature(self):
        """For gases, viscosity increases with T (unlike liquids)."""
        assert sutherland_viscosity(400.0) > sutherland_viscosity(200.0)
