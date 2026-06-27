"""
nova.core.constants
===================
Authoritative SI physical and planetary constants for Project NOVA.

All values are stored in strict SI base units:
  distance  → metres      (m)
  mass      → kilograms   (kg)
  time      → seconds     (s)
  force     → Newtons     (N)
  pressure  → Pascals     (Pa)
  angle     → radians     (rad)   [never degrees internally]
  temp      → Kelvin      (K)

Constants are module-level floats — intentionally not wrapped in a class
so they can be imported directly without instantiation overhead:

    from nova.core.constants import G, EARTH_MU, STD_GRAVITY

Do NOT mutate these after import. They are global simulation constants,
not configuration values.

Sources
-------
- NIST CODATA 2018 for fundamental constants
- IAU 2015 nominal solar system constants
- US Standard Atmosphere 1976 for atmospheric parameters
"""

# ---------------------------------------------------------------------------
# Fundamental physical constants
# ---------------------------------------------------------------------------

#: Newtonian gravitational constant [m³ kg⁻¹ s⁻²]
G: float = 6.674_30e-11

#: Standard gravitational acceleration at Earth's surface [m s⁻²]
#: Defined by ISO 80000-3; used for Isp calculations.
STD_GRAVITY: float = 9.806_65

#: Speed of light in vacuum [m s⁻¹]
SPEED_OF_LIGHT: float = 2.997_924_58e8

#: Stefan-Boltzmann constant [W m⁻² K⁻⁴]
SIGMA_SB: float = 5.670_374_419e-8

#: Universal gas constant [J mol⁻¹ K⁻¹]
R_UNIVERSAL: float = 8.314_462_618

#: Boltzmann constant [J K⁻¹]
K_BOLTZMANN: float = 1.380_649e-23

# ---------------------------------------------------------------------------
# Mathematical constants (here for locality — avoid importing math globally)
# ---------------------------------------------------------------------------

import math as _math

#: π to full float64 precision
PI: float = _math.pi

#: 2π
TWO_PI: float = 2.0 * _math.pi

#: Degrees → radians conversion factor
DEG_TO_RAD: float = _math.pi / 180.0

#: Radians → degrees conversion factor
RAD_TO_DEG: float = 180.0 / _math.pi

# ---------------------------------------------------------------------------
# Earth parameters
# ---------------------------------------------------------------------------

#: Earth gravitational parameter μ = GM [m³ s⁻²]
#: Using the TT-compatible value (IAU 2015).
EARTH_MU: float = 3.986_004_418e14

#: Earth mean equatorial radius [m]
EARTH_RADIUS_EQ: float = 6_378_137.0

#: Earth polar radius [m]
EARTH_RADIUS_POLAR: float = 6_356_752.314_245

#: Earth mean radius (volumetric) [m]
EARTH_RADIUS_MEAN: float = 6_371_000.0

#: Earth sidereal rotation rate [rad s⁻¹]
#: Ω = 7.292_115_085_7 × 10⁻⁵ rad/s (IAU 2012)
EARTH_OMEGA: float = 7.292_115_085_7e-5

#: Earth J2 oblateness coefficient (dimensionless)
EARTH_J2: float = 1.082_626_68e-3

#: Earth mass [kg]
EARTH_MASS: float = EARTH_MU / G

#: Earth atmosphere: sea-level density [kg m⁻³]  (US Std Atm 1976)
EARTH_RHO_SL: float = 1.225

#: Earth atmosphere: sea-level pressure [Pa]
EARTH_P_SL: float = 101_325.0

#: Earth atmosphere: sea-level temperature [K]
EARTH_T_SL: float = 288.15

#: Earth atmosphere: sea-level speed of sound [m s⁻¹]
EARTH_A_SL: float = 340.294

#: Earth atmosphere: scale height (exponential model) [m]
#: H = R_specific * T_SL / g₀  ≈  8434.5 m
EARTH_SCALE_HEIGHT: float = 8_434.5

#: Specific gas constant for dry air [J kg⁻¹ K⁻¹]
R_AIR: float = 287.058

#: Heat capacity ratio for dry air (γ = Cp/Cv, dimensionless)
GAMMA_AIR: float = 1.400

#: ISA temperature lapse rate (troposphere) [K m⁻¹]
ISA_LAPSE_RATE: float = -6.5e-3          # negative → cooling with altitude

#: ISA tropopause altitude [m]
ISA_TROPOPAUSE_ALT: float = 11_000.0

#: ISA tropopause temperature [K]
ISA_TROPOPAUSE_TEMP: float = 216.65

# ---------------------------------------------------------------------------
# Moon parameters
# ---------------------------------------------------------------------------

#: Moon gravitational parameter μ [m³ s⁻²]
MOON_MU: float = 4.902_800_066e12

#: Moon mean radius [m]
MOON_RADIUS_MEAN: float = 1_737_400.0

#: Moon mass [kg]
MOON_MASS: float = MOON_MU / G

# ---------------------------------------------------------------------------
# Sun parameters
# ---------------------------------------------------------------------------

#: Sun gravitational parameter μ [m³ s⁻²]  (IAU 2015 nominal)
SUN_MU: float = 1.327_124_400_41e20

#: Sun mean radius [m]
SUN_RADIUS_MEAN: float = 6.957e8

#: Solar luminosity [W]
SUN_LUMINOSITY: float = 3.828e26

#: 1 Astronomical Unit [m]
AU: float = 1.495_978_707e11

# ---------------------------------------------------------------------------
# Simulation numerical constants
# ---------------------------------------------------------------------------

#: Maximum allowable energy conservation drift per RK4 step [J]
ENERGY_CONSERVATION_TOL: float = 1.0e-6

#: Maximum allowable angular momentum drift per RK4 step [kg m² s⁻¹]
ANGULAR_MOMENTUM_TOL: float = 1.0e-6

#: Quaternion normalisation tolerance — re-normalise if ‖q‖ departs by more
QUATERNION_NORM_TOL: float = 1.0e-9

#: Coordinate transform identity tolerance (‖T·Tᵀ − I‖_F)
TRANSFORM_IDENTITY_TOL: float = 1.0e-12

#: Default simulation fixed timestep [s]
DEFAULT_DT: float = 0.01

#: Mach number below which Prandtl-Glauert correction is linear (< 0.8)
MACH_PG_UPPER: float = 0.80

#: Mach number above which wave drag divergence modelling activates
MACH_DRAG_DIVERGENCE: float = 0.85
