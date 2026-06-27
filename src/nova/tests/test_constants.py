"""
tests/unit/test_constants.py
============================
Unit tests for nova.core.constants.

Tests verify:
  1. Fundamental constant values against authoritative sources (NIST CODATA 2018).
  2. Derived relationships (μ = G·M, Ω·T_sidereal = 2π).
  3. All angles stored in radians.
  4. Simulation numerical tolerances are non-zero positive floats.
"""

import math
import pytest

from nova.core.constants import (
    G, STD_GRAVITY, EARTH_MU, EARTH_MASS, EARTH_OMEGA,
    EARTH_RADIUS_EQ, EARTH_RADIUS_POLAR,
    EARTH_RHO_SL, EARTH_P_SL, EARTH_T_SL, EARTH_A_SL, EARTH_SCALE_HEIGHT,
    R_AIR, GAMMA_AIR, ISA_LAPSE_RATE, ISA_TROPOPAUSE_ALT, ISA_TROPOPAUSE_TEMP,
    MOON_MU, MOON_MASS, MOON_RADIUS_MEAN,
    SUN_MU, SUN_RADIUS_MEAN, AU,
    PI, TWO_PI, DEG_TO_RAD, RAD_TO_DEG,
    ENERGY_CONSERVATION_TOL, ANGULAR_MOMENTUM_TOL,
    QUATERNION_NORM_TOL, TRANSFORM_IDENTITY_TOL,
    DEFAULT_DT, MACH_PG_UPPER, MACH_DRAG_DIVERGENCE,
    SIGMA_SB, R_UNIVERSAL, K_BOLTZMANN, SPEED_OF_LIGHT,
)


class TestFundamentalConstants:
    """NIST CODATA 2018 reference values."""

    def test_G_value(self):
        """G = 6.67430 × 10⁻¹¹ m³ kg⁻¹ s⁻²  (CODATA 2018)."""
        assert abs(G - 6.674_30e-11) < 1.0e-15

    def test_std_gravity(self):
        """g₀ = 9.80665 m s⁻² exactly (ISO 80000-3)."""
        assert abs(STD_GRAVITY - 9.806_65) < 1.0e-10

    def test_speed_of_light(self):
        """c = 2.99792458 × 10⁸ m s⁻¹ (exact by definition)."""
        assert abs(SPEED_OF_LIGHT - 2.997_924_58e8) < 1.0

    def test_stefan_boltzmann(self):
        """σ = 5.670374419 × 10⁻⁸ W m⁻² K⁻⁴ (CODATA 2018)."""
        assert abs(SIGMA_SB - 5.670_374_419e-8) < 1.0e-16

    def test_universal_gas_constant(self):
        """R = 8.314462618 J mol⁻¹ K⁻¹ (CODATA 2018)."""
        assert abs(R_UNIVERSAL - 8.314_462_618) < 1.0e-8

    def test_boltzmann(self):
        """k_B = 1.380649 × 10⁻²³ J K⁻¹ (exact by SI redefinition)."""
        assert abs(K_BOLTZMANN - 1.380_649e-23) < 1.0e-31


class TestMathematicalConstants:

    def test_pi_value(self):
        assert abs(PI - math.pi) < 1.0e-15

    def test_two_pi(self):
        assert abs(TWO_PI - 2.0 * math.pi) < 1.0e-15

    def test_deg_rad_roundtrip(self):
        """90° × DEG_TO_RAD × RAD_TO_DEG should return 90°."""
        angle_deg = 90.0
        result = angle_deg * DEG_TO_RAD * RAD_TO_DEG
        assert abs(result - angle_deg) < 1.0e-12

    def test_deg_to_rad_is_pi_over_180(self):
        assert abs(DEG_TO_RAD - math.pi / 180.0) < 1.0e-15

    def test_rad_to_deg_is_180_over_pi(self):
        assert abs(RAD_TO_DEG - 180.0 / math.pi) < 1.0e-12


class TestEarthParameters:

    def test_earth_mu_value(self):
        """μ⊕ = 3.986004418 × 10¹⁴ m³ s⁻²  (IAU 2015)."""
        assert abs(EARTH_MU - 3.986_004_418e14) < 1.0e5

    def test_earth_mu_equals_gm(self):
        """μ = G · M_earth must be consistent within rounding."""
        mu_derived = G * EARTH_MASS
        # Tolerance: G has 5 sig-figs, M derived from mu/G — roundtrip
        # should agree to better than 1 part in 10⁶
        relative_error = abs(mu_derived - EARTH_MU) / EARTH_MU
        assert relative_error < 1.0e-6

    def test_earth_radius_oblateness(self):
        """Equatorial radius must exceed polar radius (oblate spheroid)."""
        assert EARTH_RADIUS_EQ > EARTH_RADIUS_POLAR

    def test_earth_radius_eq_wgs84(self):
        """WGS-84: a = 6 378 137.0 m (exact)."""
        assert abs(EARTH_RADIUS_EQ - 6_378_137.0) < 0.1

    def test_earth_omega_sidereal(self):
        """Ω⊕ ≈ 7.2921 × 10⁻⁵ rad/s; sidereal period ≈ 86164 s."""
        T_sidereal = 2.0 * math.pi / EARTH_OMEGA
        # IAU sidereal day: 86164.09054 s
        assert abs(T_sidereal - 86164.09) < 1.0

    def test_isa_sea_level_density(self):
        """US Std Atm 1976: ρ₀ = 1.225 kg m⁻³."""
        assert abs(EARTH_RHO_SL - 1.225) < 0.001

    def test_isa_sea_level_pressure(self):
        """P₀ = 101325 Pa (standard atmosphere, exact)."""
        assert abs(EARTH_P_SL - 101_325.0) < 0.1

    def test_isa_sea_level_temperature(self):
        """T₀ = 288.15 K = 15°C."""
        assert abs(EARTH_T_SL - 288.15) < 0.01

    def test_isa_sea_level_speed_of_sound(self):
        """a₀ = √(γ R_air T₀) = √(1.4 × 287.058 × 288.15) ≈ 340.294 m/s."""
        a_derived = math.sqrt(GAMMA_AIR * R_AIR * EARTH_T_SL)
        assert abs(a_derived - EARTH_A_SL) < 0.01

    def test_scale_height_derived(self):
        """H = R_air · T₀ / g₀  (exponential atmosphere model)."""
        H_derived = R_AIR * EARTH_T_SL / STD_GRAVITY
        assert abs(H_derived - EARTH_SCALE_HEIGHT) < 1.0

    def test_isa_lapse_rate_negative(self):
        """Lapse rate must be negative (temperature decreases with altitude)."""
        assert ISA_LAPSE_RATE < 0.0

    def test_isa_tropopause_consistency(self):
        """Tropopause temp = T₀ + lapse × alt_tropo."""
        T_tropo = EARTH_T_SL + ISA_LAPSE_RATE * ISA_TROPOPAUSE_ALT
        assert abs(T_tropo - ISA_TROPOPAUSE_TEMP) < 0.1


class TestMoonParameters:

    def test_moon_mu_gm_consistency(self):
        """μ_moon = G · M_moon within 1 ppm."""
        mu_check = G * MOON_MASS
        relative_error = abs(mu_check - MOON_MU) / MOON_MU
        assert relative_error < 1.0e-6

    def test_moon_radius_order_of_magnitude(self):
        """Moon radius ~1737 km."""
        assert 1_700_000 < MOON_RADIUS_MEAN < 1_800_000


class TestSunParameters:

    def test_sun_mu_order(self):
        """μ_sun ~ 1.327 × 10²⁰."""
        assert 1.3e20 < SUN_MU < 1.4e20

    def test_au_value(self):
        """1 AU = 1.495978707 × 10¹¹ m."""
        assert abs(AU - 1.495_978_707e11) < 1.0e3


class TestSimulationNumericalConstants:

    def test_tolerances_positive(self):
        """All conservation tolerances must be strictly positive floats."""
        for name, val in [
            ("ENERGY_CONSERVATION_TOL", ENERGY_CONSERVATION_TOL),
            ("ANGULAR_MOMENTUM_TOL",    ANGULAR_MOMENTUM_TOL),
            ("QUATERNION_NORM_TOL",     QUATERNION_NORM_TOL),
            ("TRANSFORM_IDENTITY_TOL",  TRANSFORM_IDENTITY_TOL),
        ]:
            assert val > 0.0, f"{name} must be > 0"

    def test_tolerances_hierarchy(self):
        """
        Transform identity tolerance must be tighter than quaternion norm
        tolerance, which must be tighter than physics conservation tolerances.
        """
        assert TRANSFORM_IDENTITY_TOL < QUATERNION_NORM_TOL < ENERGY_CONSERVATION_TOL

    def test_default_dt_positive(self):
        assert DEFAULT_DT > 0.0

    def test_mach_ordering(self):
        """Drag divergence Mach must exceed Prandtl-Glauert upper limit."""
        assert MACH_PG_UPPER < MACH_DRAG_DIVERGENCE
