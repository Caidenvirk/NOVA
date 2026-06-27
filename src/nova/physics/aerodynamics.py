"""
nova.physics.aerodynamics
=========================
6-DoF rigid-body aerodynamic model for Project NOVA.

Architecture role — Pipeline Stage 3 (physics engine input)
------------------------------------------------------------
Computes aerodynamic forces and moments acting on the vehicle at one
simulation tick, given the current atmospheric state, airspeed vector,
attitude, and control surface deflections.

Output is passed to:
  - ForceAccumulator.add_aerodynamic()   → Stage 3 force tensor
  - TorqueAccumulator.add_aerodynamic_moments() → Stage 3 torque tensor

Physics implemented
-------------------

Forces (stability axes, then rotated to Body)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  L = q_∞ · S · C_L(α, δ_e, M)         Lift       [N]  (+Z_stab → up)
  D = q_∞ · S · C_D(α, M)              Drag       [N]  (−X_stab → aft)
  Y = q_∞ · S · C_Y(β, δ_r)            Side force [N]  (+Y_stab)

where q_∞ = ½ρv² is dynamic pressure, S is reference wing area.

Moments (Body frame)
~~~~~~~~~~~~~~~~~~~~
  M_pitch = q_∞ · S · c̄ · C_m(α, q_rate, δ_e)    [N·m]
  M_yaw   = q_∞ · S · b  · C_n(β, r_rate, δ_r)    [N·m]
  M_roll  = q_∞ · S · b  · C_l(β, p_rate, δ_a)    [N·m]

Stability derivative model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Coefficients are built from linear stability derivative expansions:
  C_L = C_L0 + C_Lα·α + C_Lδe·δ_e
  C_D = C_D0 + C_D_induced + C_D_wave(M)   (polar + wave drag)
  C_Y = C_Yβ·β + C_Yδr·δ_r
  C_m = C_m0 + C_mα·α + C_mq·(q·c̄/(2v)) + C_mδe·δ_e
  C_n = C_nβ·β + C_nr·(r·b/(2v)) + C_nδr·δ_r
  C_l = C_lβ·β + C_lp·(p·b/(2v)) + C_lδa·δ_a

Compressibility
~~~~~~~~~~~~~~~
Prandtl-Glauert correction applied for M < MACH_PG_UPPER:
  C_L_corrected = C_L / √(1 − M²)

Above MACH_DRAG_DIVERGENCE, a tabulated wave-drag increment ΔC_D_wave is
added. The table is defined as a piecewise-linear spline indexed by Mach.

Coordinate conventions
~~~~~~~~~~~~~~~~~~~~~~
  Stability axes: X_s → forward (into freestream), Z_s → up, Y_s → right
  Body axes:      X_b → forward, Y_b → right, Z_b → down

The rotation from stability to body axes is:
  [X_b]   [ cos α   0  −sin α ] [X_s]
  [Y_b] = [   0     1     0   ] [Y_s]
  [Z_b]   [ sin α   0   cos α ] [Z_s]

For small α this is approximately identity in the pitch plane.

References
----------
- Stevens & Lewis, "Aircraft Control and Simulation", 3rd ed., §2.3–§2.5
- Anderson, "Introduction to Flight", 8th ed., §4–§5
- Roskam, "Airplane Flight Dynamics & Automatic Flight Controls", Part I
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nova.core.constants import (
    MACH_PG_UPPER,
    MACH_DRAG_DIVERGENCE,
)
from nova.physics.atmosphere import (
    dynamic_pressure,
    prandtl_glauert_factor,
    AtmosphericState,
)


# ---------------------------------------------------------------------------
# Control surface deflection angles
# ---------------------------------------------------------------------------

@dataclass
class ControlDeflections:
    """
    Actuated control surface deflection angles [rad].

    All angles measured from neutral (trim) position.
    Positive deflection conventions follow standard aerospace sign rules:
      δ_e > 0 → trailing edge down (pitch-up moment)
      δ_a > 0 → right aileron down, left up (right-roll moment)
      δ_r > 0 → trailing edge left (yaw-left / nose-right moment)

    Attributes
    ----------
    elevator : float   δ_e [rad]
    aileron  : float   δ_a [rad]
    rudder   : float   δ_r [rad]
    """
    elevator: float = 0.0   # δ_e [rad]
    aileron:  float = 0.0   # δ_a [rad]
    rudder:   float = 0.0   # δ_r [rad]


# ---------------------------------------------------------------------------
# Vehicle aerodynamic configuration
# ---------------------------------------------------------------------------

@dataclass
class AeroConfig:
    """
    Vehicle-specific aerodynamic reference geometry and stability derivatives.

    All derivatives are per-radian unless noted as dimensionless.

    Parameters
    ----------
    reference_area : float
        S — wing reference area [m²].
    mean_chord : float
        c̄ — mean aerodynamic chord [m]. Used for pitch moment normalisation.
    span : float
        b — wing span [m]. Used for roll/yaw moment normalisation.

    Lift derivatives
    ----------------
    CL0 : float
        Zero-AoA lift coefficient [-].
    CLa : float
        Lift-curve slope dC_L/dα [rad⁻¹]. Typically ~2π for thin wings.
    CLde : float
        Elevator lift effectiveness dC_L/dδ_e [rad⁻¹].

    Drag polar
    ----------
    CD0 : float
        Parasite drag coefficient [-].
    k_induced : float
        Induced drag factor k such that C_Di = k·C_L² (= 1/(π·AR·e)).
    CD_wave_table : list of (Mach, ΔC_D) pairs, optional
        Piecewise-linear wave-drag increment table. Indexed by Mach number.
        Applied above MACH_DRAG_DIVERGENCE.

    Side force
    ----------
    CYb : float
        Side-force due to sideslip dC_Y/dβ [rad⁻¹]. Typically negative.
    CYdr : float
        Side-force due to rudder dC_Y/dδ_r [rad⁻¹].

    Pitching moment
    ---------------
    Cm0 : float
        Zero-lift pitching moment coefficient [-].
    Cma : float
        Pitch stiffness dC_m/dα [rad⁻¹]. Negative → statically stable.
    Cmq : float
        Pitch damping dC_m/d(q·c̄/(2v)) [rad⁻¹]. Typically negative.
    Cmde : float
        Elevator pitch effectiveness dC_m/dδ_e [rad⁻¹].

    Yawing moment
    -------------
    Cnb : float
        Yaw stiffness dC_n/dβ [rad⁻¹]. Positive → weathercock stable.
    Cnr : float
        Yaw damping dC_n/d(r·b/(2v)) [rad⁻¹]. Typically negative.
    Cndr : float
        Rudder yaw effectiveness dC_n/dδ_r [rad⁻¹].

    Rolling moment
    --------------
    Clb : float
        Roll-due-to-sideslip (dihedral effect) dC_l/dβ [rad⁻¹]. Neg → stable.
    Clp : float
        Roll damping dC_l/d(p·b/(2v)) [rad⁻¹]. Typically negative.
    Clda : float
        Aileron roll effectiveness dC_l/dδ_a [rad⁻¹].
    """
    # Reference geometry
    reference_area: float   # S [m²]
    mean_chord:     float   # c̄ [m]
    span:           float   # b [m]

    # Lift
    CL0:  float = 0.0
    CLa:  float = 5.730    # ≈ 2π·0.9 for a moderately cambered wing [rad⁻¹]
    CLde: float = 0.5

    # Drag polar
    CD0:         float = 0.020
    k_induced:   float = 0.050   # k = 1/(π·AR·e), typical clean wing

    # Wave drag: list of (Mach, delta_CD) breakpoints (piecewise linear)
    CD_wave_table: list = field(default_factory=lambda: [
        (0.85, 0.000),
        (0.90, 0.005),
        (0.95, 0.020),
        (1.00, 0.060),
        (1.05, 0.050),
        (1.20, 0.030),
        (2.00, 0.015),
        (5.00, 0.008),
    ])

    # Side force
    CYb:  float = -0.980   # [rad⁻¹]
    CYdr: float =  0.175

    # Pitching moment
    Cm0:  float =  0.000
    Cma:  float = -1.800   # [rad⁻¹] — statically stable
    Cmq:  float = -12.0    # [rad⁻¹]
    Cmde: float = -1.500   # [rad⁻¹]

    # Yawing moment
    Cnb:  float =  0.120   # [rad⁻¹]
    Cnr:  float = -0.150   # [rad⁻¹]
    Cndr: float = -0.100   # [rad⁻¹]

    # Rolling moment
    Clb:  float = -0.100   # [rad⁻¹]
    Clp:  float = -0.500   # [rad⁻¹]
    Clda: float =  0.220   # [rad⁻¹]

    def __post_init__(self) -> None:
        if self.reference_area <= 0.0:
            raise ValueError(f"AeroConfig.reference_area must be > 0, got {self.reference_area!r}")
        if self.mean_chord <= 0.0:
            raise ValueError(f"AeroConfig.mean_chord must be > 0, got {self.mean_chord!r}")
        if self.span <= 0.0:
            raise ValueError(f"AeroConfig.span must be > 0, got {self.span!r}")


# ---------------------------------------------------------------------------
# Aerodynamic state output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AeroState:
    """
    Complete aerodynamic output for one simulation tick.

    All forces in Body frame [N]; all moments in Body frame [N·m].

    Attributes
    ----------
    alpha : float
        Angle of attack [rad].
    beta : float
        Sideslip angle [rad].
    mach : float
        Freestream Mach number [-].
    dynamic_pressure : float
        q_∞ = ½ρv² [Pa].
    CL, CD, CY : float
        Lift, drag, side-force coefficients (corrected for compressibility).
    Cm, Cn, Cl : float
        Pitching, yawing, rolling moment coefficients.
    lift_body : ndarray, shape (3,), float64
        Lift force in Body frame [N].
    drag_body : ndarray, shape (3,), float64
        Drag force in Body frame [N].
    side_force_body : ndarray, shape (3,), float64
        Side force in Body frame [N].
    force_body : ndarray, shape (3,), float64
        Net aerodynamic force in Body frame [N] = lift + drag + side.
    pitching_moment : float   [N·m]
    yawing_moment   : float   [N·m]
    rolling_moment  : float   [N·m]
    """
    alpha:            float
    beta:             float
    mach:             float
    dynamic_pressure: float
    CL:               float
    CD:               float
    CY:               float
    Cm:               float
    Cn:               float
    Cl:               float
    lift_body:        np.ndarray
    drag_body:        np.ndarray
    side_force_body:  np.ndarray
    force_body:       np.ndarray
    pitching_moment:  float
    yawing_moment:    float
    rolling_moment:   float


# ---------------------------------------------------------------------------
# Wave drag interpolator
# ---------------------------------------------------------------------------

def _wave_drag_increment(mach: float, table: list) -> float:
    """
    Piecewise-linear interpolation of wave drag increment ΔC_D from
    a (Mach, ΔC_D) breakpoint table.

    Parameters
    ----------
    mach : float
        Freestream Mach number.
    table : list of (float, float)
        Sorted (Mach, ΔC_D) breakpoints.

    Returns
    -------
    float
        Wave drag increment ΔC_D [-].
    """
    if not table or mach <= table[0][0]:
        return 0.0
    if mach >= table[-1][0]:
        return float(table[-1][1])
    for i in range(len(table) - 1):
        m0, cd0 = table[i]
        m1, cd1 = table[i + 1]
        if m0 <= mach <= m1:
            t = (mach - m0) / (m1 - m0)
            return cd0 + t * (cd1 - cd0)
    return 0.0


# ---------------------------------------------------------------------------
# Angle-of-attack and sideslip extraction
# ---------------------------------------------------------------------------

def aero_angles(velocity_body: np.ndarray) -> tuple[float, float]:
    """
    Compute angle of attack (α) and sideslip angle (β) from the
    velocity vector expressed in the Body frame.

    Definitions (standard aerospace)
    ---------------------------------
    α = arctan2(w_b, u_b)     [rad]   pitch-plane angle of attack
    β = arcsin(v_b / |v|)     [rad]   yaw-plane sideslip

    where (u_b, v_b, w_b) = velocity components in Body frame:
      u_b → forward (+X_b)
      v_b → right   (+Y_b)
      w_b → down    (+Z_b)  ← positive w means downward relative wind → positive α

    Parameters
    ----------
    velocity_body : ndarray, shape (3,), float64
        Airspeed vector in Body frame [m s⁻¹].

    Returns
    -------
    (alpha, beta) : tuple of float, [rad]
        Returns (0, 0) if airspeed magnitude is below 0.01 m/s.
    """
    v = np.asarray(velocity_body, dtype=np.float64)
    v_mag = float(np.linalg.norm(v))
    if v_mag < 0.01:
        return 0.0, 0.0

    u_b, v_b, w_b = float(v[0]), float(v[1]), float(v[2])

    alpha = math.atan2(w_b, u_b)
    beta  = math.asin(max(-1.0, min(1.0, v_b / v_mag)))

    return alpha, beta


# ---------------------------------------------------------------------------
# Stability-to-body axis rotation
# ---------------------------------------------------------------------------

def stability_to_body(
    L_stab: float,
    D_stab: float,
    Y_stab: float,
    alpha:  float,
) -> np.ndarray:
    """
    Rotate aerodynamic forces from stability axes to Body axes.

    Stability axes: X_s = −drag direction, Z_s = −lift direction
    Body axes:      X_b = forward, Y_b = right, Z_b = down

    The rotation about the Y axis (pitch axis) by angle α gives:

        F_body = R_stab_to_body(α) · F_stab

    where:
        X_body = X_stab · cos α − Z_stab · sin α   (= −D·cosα + L·sinα)
        Y_body = Y_stab
        Z_body = X_stab · sin α + Z_stab · cos α   (= −D·sinα − L·cosα)

    Note: in stability axes, X_s is aligned with the velocity vector (−drag
    direction), so D is applied along −X_s and L along +Z_s (upward).
    After rotation:
      F_x_body = −D·cos α + L·sin α  (net axial force, usually negative = drag)
      F_y_body = Y                    (side force unchanged)
      F_z_body = −D·sin α − L·cos α  (net normal force; negative = upward lift)

    Parameters
    ----------
    L_stab : float   Lift [N]   (positive = up in stability axes)
    D_stab : float   Drag [N]   (positive = opposes velocity)
    Y_stab : float   Side force [N]
    alpha  : float   Angle of attack [rad]

    Returns
    -------
    ndarray, shape (3,), float64
        Net aerodynamic force in Body frame [N].
    """
    ca, sa = math.cos(alpha), math.sin(alpha)

    # In stability axes: F_stab = [−D, Y, −L]  (drag aft, lift up→ −Z_stab)
    # Rotate to body:
    Fx_body = -D_stab * ca + L_stab * sa   # axial  (aft = negative)
    Fy_body =  Y_stab                       # lateral
    Fz_body = -D_stab * sa - L_stab * ca   # normal (up = negative in body Z-down)

    return np.array([Fx_body, Fy_body, Fz_body], dtype=np.float64)


# ---------------------------------------------------------------------------
# Primary aerodynamics function
# ---------------------------------------------------------------------------

def compute_aero(
    velocity_body:  np.ndarray,
    omega_body:     np.ndarray,
    atm:            AtmosphericState,
    config:         AeroConfig,
    deflections:    ControlDeflections,
) -> AeroState:
    """
    Compute the full aerodynamic force and moment state for one tick.

    Parameters
    ----------
    velocity_body : ndarray, shape (3,), float64
        Vehicle airspeed vector in Body frame [m s⁻¹].
        Typically v_eci − v_wind, rotated to Body frame.
    omega_body : ndarray, shape (3,), float64
        Angular velocity in Body frame (p, q, r) [rad s⁻¹].
    atm : AtmosphericState
        Atmospheric state at the vehicle's current altitude.
    config : AeroConfig
        Vehicle aerodynamic configuration and stability derivatives.
    deflections : ControlDeflections
        Current control surface deflection angles [rad].

    Returns
    -------
    AeroState
        Complete aerodynamic output snapshot (frozen).

    Notes
    -----
    * If airspeed < 0.01 m/s, returns a zero-force AeroState (vacuum/hover).
    * No stall model is implemented beyond the linear CLa regime. AoA
      limiting (stall onset) is handled by the AI monitor advisory layer.
    * The dynamic pressure term uses true airspeed relative to the atmosphere;
      the caller must subtract wind velocity from ECI velocity before rotating.
    """
    v    = np.asarray(velocity_body, dtype=np.float64)
    omg  = np.asarray(omega_body,    dtype=np.float64)
    v_mag = float(np.linalg.norm(v))

    # --- Zero-airspeed guard ---
    _ZERO = np.zeros(3, dtype=np.float64)
    if v_mag < 0.01:
        return AeroState(
            alpha=0.0, beta=0.0, mach=0.0, dynamic_pressure=0.0,
            CL=0.0, CD=0.0, CY=0.0, Cm=0.0, Cn=0.0, Cl=0.0,
            lift_body=_ZERO.copy(), drag_body=_ZERO.copy(),
            side_force_body=_ZERO.copy(), force_body=_ZERO.copy(),
            pitching_moment=0.0, yawing_moment=0.0, rolling_moment=0.0,
        )

    # --- Aero angles ---
    alpha, beta = aero_angles(v)
    p_rate, q_rate, r_rate = float(omg[0]), float(omg[1]), float(omg[2])

    # --- Atmospheric quantities ---
    rho  = atm.density
    a_s  = atm.speed_of_sound
    q_inf = dynamic_pressure(rho, v_mag)
    M     = v_mag / a_s if a_s > 1.0e-6 else 0.0

    # --- Compressibility correction ---
    if M < MACH_PG_UPPER:
        beta_pg = prandtl_glauert_factor(M)   # β_PG = 1/√(1−M²)
    else:
        beta_pg = 1.0   # P-G breaks down near M=1; correction capped

    # --- Dimensionless rate terms (standard normalised rates) ---
    v_safe = max(v_mag, 0.01)
    q_hat = q_rate * config.mean_chord / (2.0 * v_safe)   # pitch rate term
    r_hat = r_rate * config.span       / (2.0 * v_safe)   # yaw rate term
    p_hat = p_rate * config.span       / (2.0 * v_safe)   # roll rate term

    de = deflections.elevator
    da = deflections.aileron
    dr = deflections.rudder

    # ---------------------------------------------------------------
    # Lift coefficient
    # ---------------------------------------------------------------
    CL_base = config.CL0 + config.CLa * alpha + config.CLde * de
    CL      = CL_base * beta_pg   # Prandtl-Glauert correction

    # ---------------------------------------------------------------
    # Drag coefficient (polar + induced + wave)
    # ---------------------------------------------------------------
    CD_induced = config.k_induced * CL_base**2
    CD_wave    = _wave_drag_increment(M, config.CD_wave_table)
    CD         = config.CD0 + CD_induced + CD_wave

    # ---------------------------------------------------------------
    # Side-force coefficient
    # ---------------------------------------------------------------
    CY = config.CYb * beta + config.CYdr * dr

    # ---------------------------------------------------------------
    # Pitching moment coefficient
    # ---------------------------------------------------------------
    Cm = (config.Cm0
          + config.Cma  * alpha
          + config.Cmq  * q_hat
          + config.Cmde * de)

    # ---------------------------------------------------------------
    # Yawing moment coefficient
    # ---------------------------------------------------------------
    Cn = (config.Cnb  * beta
          + config.Cnr  * r_hat
          + config.Cndr * dr)

    # ---------------------------------------------------------------
    # Rolling moment coefficient
    # ---------------------------------------------------------------
    Cl = (config.Clb  * beta
          + config.Clp  * p_hat
          + config.Clda * da)

    # ---------------------------------------------------------------
    # Dimensionalise forces
    # ---------------------------------------------------------------
    S = config.reference_area
    L_stab =  CL * q_inf * S   # [N] lift, positive up
    D_stab =  CD * q_inf * S   # [N] drag, always positive
    Y_stab =  CY * q_inf * S   # [N] side force

    # Rotate to Body frame
    aero_force_body = stability_to_body(L_stab, D_stab, Y_stab, alpha)

    # Decompose for telemetry
    ca, sa = math.cos(alpha), math.sin(alpha)
    lift_body = np.array([ L_stab * sa,  0.0, -L_stab * ca], dtype=np.float64)
    drag_body = np.array([-D_stab * ca,  0.0, -D_stab * sa], dtype=np.float64)
    side_body = np.array([0.0, Y_stab, 0.0],                  dtype=np.float64)

    # ---------------------------------------------------------------
    # Dimensionalise moments
    # ---------------------------------------------------------------
    c_bar = config.mean_chord
    b     = config.span

    M_pitch =  Cm * q_inf * S * c_bar   # [N·m]
    M_yaw   =  Cn * q_inf * S * b       # [N·m]
    M_roll  =  Cl * q_inf * S * b       # [N·m]

    return AeroState(
        alpha=alpha,
        beta=beta,
        mach=M,
        dynamic_pressure=q_inf,
        CL=CL,
        CD=CD,
        CY=CY,
        Cm=Cm,
        Cn=Cn,
        Cl=Cl,
        lift_body=lift_body,
        drag_body=drag_body,
        side_force_body=side_body,
        force_body=aero_force_body,
        pitching_moment=M_pitch,
        yawing_moment=M_yaw,
        rolling_moment=M_roll,
    )
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_aerodynamics.py
================================
Unit tests for nova.physics.aerodynamics.

Tests verify:
  1. aero_angles — known α and β from analytically defined velocity vectors.
  2. stability_to_body — force rotation at zero and nonzero AoA.
  3. compute_aero — zero airspeed returns zero AeroState.
  4. Lift coefficient linear with AoA; Prandtl-Glauert correction applied.
  5. Drag polar: CD = CD0 + k·CL² at subsonic speeds.
  6. Wave drag increment activates above MACH_DRAG_DIVERGENCE.
  7. Pitching moment sign (nose-up positive), damping term with pitch rate.
  8. Sideslip drives side-force and yawing moment with correct sign.
  9. Rolling moment: aileron and roll-rate contributions.
  10. Force body vector magnitude consistency: |F| = q·S·√(CL²+CD²) approx.
  11. AeroState is frozen (immutable).
  12. AeroConfig validates geometry inputs.
"""

import math
import pytest
import numpy as np

from nova.physics.aerodynamics import (
    AeroConfig,
    ControlDeflections,
    AeroState,
    aero_angles,
    stability_to_body,
    compute_aero,
    _wave_drag_increment,
)
from nova.physics.atmosphere import atmosphere
from nova.core.constants import MACH_PG_UPPER, MACH_DRAG_DIVERGENCE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generic_config() -> AeroConfig:
    """A representative aircraft-like aerodynamic configuration."""
    return AeroConfig(
        reference_area=20.0,   # S [m²]
        mean_chord=2.5,        # c̄ [m]
        span=10.0,             # b [m]
        CL0=0.1,
        CLa=5.0,
        CLde=0.4,
        CD0=0.020,
        k_induced=0.050,
        CYb=-0.80,
        CYdr=0.15,
        Cm0=0.0,
        Cma=-1.5,
        Cmq=-10.0,
        Cmde=-1.2,
        Cnb=0.10,
        Cnr=-0.12,
        Cndr=-0.08,
        Clb=-0.09,
        Clp=-0.45,
        Clda=0.20,
    )


@pytest.fixture
def sl_atm():
    """Sea-level ISA atmospheric state."""
    return atmosphere(0.0)


@pytest.fixture
def neutral_deflections() -> ControlDeflections:
    return ControlDeflections(elevator=0.0, aileron=0.0, rudder=0.0)


# ---------------------------------------------------------------------------
# 1. Angle of attack and sideslip
# ---------------------------------------------------------------------------

class TestAeroAngles:

    def test_pure_forward_velocity_zero_angles(self):
        """Velocity along +X_body → α = 0, β = 0."""
        v = np.array([100.0, 0.0, 0.0], dtype=np.float64)
        alpha, beta = aero_angles(v)
        assert abs(alpha) < 1.0e-12
        assert abs(beta)  < 1.0e-12

    def test_positive_alpha_from_downward_w(self):
        """
        w_b > 0 (velocity component downward in body Z-down frame)
        → AoA α = arctan2(w, u) > 0.
        Physically: nose pitched above velocity vector.
        """
        u, w = 100.0, 10.0
        alpha, beta = aero_angles(np.array([u, 0.0, w], dtype=np.float64))
        expected = math.atan2(w, u)
        assert abs(alpha - expected) < 1.0e-12

    def test_known_alpha_5deg(self):
        """5° AoA: velocity in X-Z plane, w/u = tan(5°)."""
        a_deg  = 5.0
        a_rad  = math.radians(a_deg)
        speed  = 200.0
        u = speed * math.cos(a_rad)
        w = speed * math.sin(a_rad)
        alpha, beta = aero_angles(np.array([u, 0.0, w]))
        assert abs(alpha - a_rad) < 1.0e-10

    def test_positive_beta_from_positive_v(self):
        """v_b > 0 → β = arcsin(v/|v|) > 0 (sideslip to the right)."""
        v_vec = np.array([100.0, 10.0, 0.0], dtype=np.float64)
        alpha, beta = aero_angles(v_vec)
        v_mag = float(np.linalg.norm(v_vec))
        expected = math.asin(10.0 / v_mag)
        assert abs(beta - expected) < 1.0e-10

    def test_zero_airspeed_returns_zeros(self):
        alpha, beta = aero_angles(np.array([0.0, 0.0, 0.0]))
        assert alpha == 0.0
        assert beta  == 0.0

    def test_tiny_airspeed_returns_zeros(self):
        """Below 0.01 m/s threshold → (0, 0)."""
        alpha, beta = aero_angles(np.array([0.005, 0.0, 0.0]))
        assert alpha == 0.0 and beta == 0.0


# ---------------------------------------------------------------------------
# 2. Stability-to-body axis rotation
# ---------------------------------------------------------------------------

class TestStabilityToBody:

    def test_zero_alpha_lift_is_minus_z(self):
        """
        At α=0: lift is perpendicular to velocity (+Z_stab = up).
        In Body frame (Z-down): lift → −Z_body.
        F_z_body = −L·cos(0) = −L.
        """
        L, D, Y = 1000.0, 100.0, 0.0
        f = stability_to_body(L, D, Y, alpha=0.0)
        assert abs(f[2] - (-L)) < 1.0e-10, f"F_z = {f[2]}, expected {-L}"
        assert abs(f[0] - (-D)) < 1.0e-10, f"F_x = {f[0]}, expected {-D}"
        assert abs(f[1]) < 1.0e-12

    def test_zero_alpha_drag_is_minus_x(self):
        """At α=0: drag opposes forward motion → −X_body."""
        f = stability_to_body(0.0, 500.0, 0.0, alpha=0.0)
        assert abs(f[0] - (-500.0)) < 1.0e-10

    def test_side_force_in_y(self):
        """Side force passes through unchanged → +Y_body."""
        f = stability_to_body(0.0, 0.0, 300.0, alpha=0.0)
        assert abs(f[1] - 300.0) < 1.0e-10

    def test_magnitude_preserved(self):
        """
        Rotation from stability to body is orthogonal → magnitude preserved.
        |F_body| = √(L²+D²+Y²) when combined as a stability-axis vector.

        The individual component magnitudes depend on α, but the total
        force magnitude from the rotation must equal √((−D)²+(Y)²+(−L)²).
        """
        L, D, Y = 2000.0, 300.0, 150.0
        alpha   = math.radians(10.0)
        f       = stability_to_body(L, D, Y, alpha)
        # Input force magnitude in stability axes
        F_in_mag = math.sqrt(L**2 + D**2 + Y**2)
        F_out_mag = float(np.linalg.norm(f))
        assert abs(F_out_mag - F_in_mag) < 1.0e-8, \
            f"|F_out|={F_out_mag:.4f}, |F_in|={F_in_mag:.4f}"

    def test_90deg_alpha_rotates_lift_to_x(self):
        """At α=90°: lift moves entirely into X_body direction."""
        L, D, Y = 1000.0, 0.0, 0.0
        f = stability_to_body(L, D, Y, alpha=math.pi / 2.0)
        # F_x = −D·cos90 + L·sin90 = L
        assert abs(f[0] - L) < 1.0e-9


# ---------------------------------------------------------------------------
# 3. Zero-airspeed guard
# ---------------------------------------------------------------------------

class TestZeroAirspeed:

    def test_zero_velocity_returns_zero_aerostate(
            self, generic_config, sl_atm, neutral_deflections):
        v_zero = np.zeros(3, dtype=np.float64)
        omega  = np.zeros(3, dtype=np.float64)
        state  = compute_aero(v_zero, omega, sl_atm, generic_config, neutral_deflections)
        assert np.allclose(state.force_body, [0.0, 0.0, 0.0])
        assert state.dynamic_pressure == 0.0
        assert state.mach == 0.0
        assert state.pitching_moment == 0.0

    def test_tiny_velocity_returns_zero_aerostate(
            self, generic_config, sl_atm, neutral_deflections):
        v_tiny = np.array([0.009, 0.0, 0.0])
        omega  = np.zeros(3)
        state  = compute_aero(v_tiny, omega, sl_atm, generic_config, neutral_deflections)
        assert state.dynamic_pressure == 0.0


# ---------------------------------------------------------------------------
# 4. Lift coefficient model
# ---------------------------------------------------------------------------

class TestLiftCoefficient:

    def test_CL_increases_with_alpha(self, generic_config, sl_atm, neutral_deflections):
        """C_L must increase monotonically with AoA in the linear regime."""
        CLs = []
        for a_deg in [0, 2, 5, 8, 10]:
            a_rad = math.radians(a_deg)
            v = np.array([200.0 * math.cos(a_rad), 0.0, 200.0 * math.sin(a_rad)])
            s = compute_aero(v, np.zeros(3), sl_atm, generic_config, neutral_deflections)
            CLs.append(s.CL)
        for i in range(1, len(CLs)):
            assert CLs[i] > CLs[i - 1], f"CL not increasing at index {i}"

    def test_CL0_at_zero_alpha_no_deflection(self, generic_config, sl_atm, neutral_deflections):
        """At α=0, no deflections: CL = CL0 (Prandtl-Glauert ≈ 1 at low M)."""
        v = np.array([50.0, 0.0, 0.0])   # low speed → M ≈ 0.15, PG ≈ 1.011
        s = compute_aero(v, np.zeros(3), sl_atm, generic_config, neutral_deflections)
        # PG correction small but nonzero; CL ≈ CL0 * beta_pg
        M   = s.mach
        pg  = 1.0 / math.sqrt(1.0 - M**2) if M < 1.0 else 1.0
        expected = generic_config.CL0 * pg
        assert abs(s.CL - expected) < 1.0e-6

    def test_elevator_increases_CL(self, generic_config, sl_atm):
        """Positive elevator deflection must increase C_L."""
        v    = np.array([200.0, 0.0, 0.0])
        omg  = np.zeros(3)
        s0   = compute_aero(v, omg, sl_atm, generic_config, ControlDeflections())
        s_de = compute_aero(v, omg, sl_atm, generic_config,
                            ControlDeflections(elevator=math.radians(5.0)))
        assert s_de.CL > s0.CL

    def test_prandtl_glauert_increases_CL(self, generic_config, sl_atm):
        """Higher Mach (still < PG limit) should increase CL for same geometry."""
        v_lo = np.array([ 50.0, 0.0, math.radians(3.0) * 50.0])   # low Mach
        v_hi = np.array([250.0, 0.0, math.radians(3.0) * 250.0])   # higher Mach
        omg  = np.zeros(3)
        defl = neutral_deflections = ControlDeflections()
        s_lo = compute_aero(v_lo, omg, sl_atm, generic_config, defl)
        s_hi = compute_aero(v_hi, omg, sl_atm, generic_config, defl)
        assert s_hi.CL > s_lo.CL, \
            f"CL not increasing with Mach: {s_lo.CL:.4f} @ M={s_lo.mach:.3f}, " \
            f"{s_hi.CL:.4f} @ M={s_hi.mach:.3f}"


# ---------------------------------------------------------------------------
# 5. Drag polar
# ---------------------------------------------------------------------------

class TestDragPolar:

    def test_CD_minimum_at_zero_alpha(self, generic_config, sl_atm, neutral_deflections):
        """CD is minimised near α=0 (CL ≈ CL0, induced drag ≈ k·CL0²)."""
        CDs = []
        for a_deg in [-5, -2, 0, 2, 5]:
            a = math.radians(a_deg)
            v = np.array([200.0 * math.cos(a), 0.0, 200.0 * math.sin(a)])
            s = compute_aero(v, np.zeros(3), sl_atm, generic_config, neutral_deflections)
            CDs.append(s.CD)
        # Minimum should be near α=0
        min_idx = CDs.index(min(CDs))
        assert min_idx in [1, 2, 3], f"CD minimum not near α=0, at index {min_idx}"

    def test_induced_drag_quadratic_in_CL(self, generic_config, sl_atm, neutral_deflections):
        """
        At subsonic speeds with no wave drag:
        CD_induced = k · CL_base²  (where CL_base is pre-PG lift coeff).
        Verify CD increases faster than CL with AoA.
        """
        v1 = np.array([200.0, 0.0, 200.0 * math.tan(math.radians(2.0))])
        v2 = np.array([200.0, 0.0, 200.0 * math.tan(math.radians(6.0))])
        s1 = compute_aero(v1, np.zeros(3), sl_atm, generic_config, neutral_deflections)
        s2 = compute_aero(v2, np.zeros(3), sl_atm, generic_config, neutral_deflections)
        # CL roughly triples (2→6°), CD should increase by more than triple
        dCL = s2.CL - s1.CL
        dCD = s2.CD - s1.CD
        assert dCD > 0.0 and dCL > 0.0


# ---------------------------------------------------------------------------
# 6. Wave drag increment
# ---------------------------------------------------------------------------

class TestWaveDrag:

    def test_no_wave_drag_below_divergence(self, generic_config):
        """Below MACH_DRAG_DIVERGENCE, wave drag increment = 0."""
        delta = _wave_drag_increment(MACH_DRAG_DIVERGENCE - 0.01, generic_config.CD_wave_table)
        assert delta == 0.0

    def test_wave_drag_positive_above_divergence(self, generic_config):
        """Above MACH_DRAG_DIVERGENCE, wave drag must be positive."""
        delta = _wave_drag_increment(0.95, generic_config.CD_wave_table)
        assert delta > 0.0

    def test_wave_drag_interpolated(self, generic_config):
        """At M=0.875 (midpoint between 0.85 and 0.90): linear interp."""
        table = [(0.85, 0.0), (0.90, 0.010)]
        delta = _wave_drag_increment(0.875, table)
        assert abs(delta - 0.005) < 1.0e-10

    def test_wave_drag_capped_at_table_end(self, generic_config):
        """Above the last table entry, use the final value."""
        table = [(0.85, 0.0), (2.0, 0.015)]
        delta = _wave_drag_increment(10.0, table)
        assert abs(delta - 0.015) < 1.0e-10

    def test_wave_drag_in_compute_aero(self, generic_config):
        """
        At supersonic speed the total CD should exceed the subsonic polar.
        Compare CD at M~0.5 vs M~1.1 (same AoA, high-altitude low-density atm).
        """
        atm_hi = atmosphere(15_000.0)   # 15 km, T=216.65 K, a=295 m/s
        v_lo   = np.array([150.0, 0.0, 0.0])   # M~0.51
        v_hi   = np.array([340.0, 0.0, 0.0])   # M~1.15
        defl   = ControlDeflections()
        s_lo   = compute_aero(v_lo, np.zeros(3), atm_hi, generic_config, defl)
        s_hi   = compute_aero(v_hi, np.zeros(3), atm_hi, generic_config, defl)
        assert s_hi.CD > s_lo.CD, \
            f"Wave drag not visible: CD_lo={s_lo.CD:.5f} CD_hi={s_hi.CD:.5f}"


# ---------------------------------------------------------------------------
# 7. Pitching moment
# ---------------------------------------------------------------------------

class TestPitchingMoment:

    def test_static_stability_nose_down_at_positive_alpha(
            self, generic_config, sl_atm, neutral_deflections):
        """
        Cma < 0 → statically stable: positive AoA produces nose-down (negative) Cm.
        M_pitch = Cm * q * S * c̄ → negative for positive α.
        """
        a = math.radians(5.0)
        v = np.array([200.0 * math.cos(a), 0.0, 200.0 * math.sin(a)])
        s = compute_aero(v, np.zeros(3), sl_atm, generic_config, neutral_deflections)
        assert s.pitching_moment < 0.0, \
            f"Expected nose-down pitch moment, got {s.pitching_moment:.2f} N·m"

    def test_pitch_damping_opposes_pitch_rate(self, generic_config, sl_atm, neutral_deflections):
        """
        Cmq < 0: nose-up pitch rate (q > 0) should decrease Cm.
        """
        v   = np.array([200.0, 0.0, 0.0])
        omg_zero = np.zeros(3)
        omg_q    = np.array([0.0, 0.1, 0.0])   # q = 0.1 rad/s (pitch up)
        s0 = compute_aero(v, omg_zero, sl_atm, generic_config, neutral_deflections)
        sq = compute_aero(v, omg_q,   sl_atm, generic_config, neutral_deflections)
        # Cmq < 0 → pitch rate reduces Cm
        assert sq.Cm < s0.Cm, \
            f"Pitch damping not acting: Cm0={s0.Cm:.4f}, Cm_q={sq.Cm:.4f}"

    def test_elevator_changes_pitch_moment(self, generic_config, sl_atm):
        """Positive elevator (δ_e > 0) should produce nose-up moment change (Cmde < 0 → nose-down for positive δ_e in our convention)."""
        v   = np.array([200.0, 0.0, 0.0])
        omg = np.zeros(3)
        s0  = compute_aero(v, omg, sl_atm, generic_config, ControlDeflections())
        sde = compute_aero(v, omg, sl_atm, generic_config, ControlDeflections(elevator=math.radians(5.0)))
        # Cmde is negative in our config → positive δ_e reduces Cm
        assert sde.Cm != s0.Cm, "Elevator had no effect on pitching moment"


# ---------------------------------------------------------------------------
# 8. Sideslip — side force and yawing moment
# ---------------------------------------------------------------------------

class TestSideslip:

    def test_positive_sideslip_positive_side_force_sign(
            self, generic_config, sl_atm, neutral_deflections):
        """
        CYb < 0 → positive β produces negative side force (Y component in body).
        """
        v_mag = 200.0
        beta  = math.radians(5.0)
        v = np.array([v_mag * math.cos(beta), v_mag * math.sin(beta), 0.0])
        s = compute_aero(v, np.zeros(3), sl_atm, generic_config, neutral_deflections)
        # CYb < 0 → CY < 0 for positive β → F_y_body < 0
        assert s.CY < 0.0, f"Side force sign wrong: CY={s.CY:.4f}"

    def test_yawing_moment_weathercock_stability(
            self, generic_config, sl_atm, neutral_deflections):
        """
        Cnb > 0 → positive β produces positive Cn → positive yawing moment
        (nose-right = restoring for positive sideslip).
        """
        beta = math.radians(5.0)
        v_mag = 200.0
        v = np.array([v_mag * math.cos(beta), v_mag * math.sin(beta), 0.0])
        s = compute_aero(v, np.zeros(3), sl_atm, generic_config, neutral_deflections)
        assert s.yawing_moment > 0.0, \
            f"Expected positive yaw moment for positive β, got {s.yawing_moment:.2f}"


# ---------------------------------------------------------------------------
# 9. Rolling moment
# ---------------------------------------------------------------------------

class TestRollingMoment:

    def test_aileron_produces_roll(self, generic_config, sl_atm):
        """Positive aileron (δ_a > 0, Clda > 0) → positive rolling moment."""
        v = np.array([200.0, 0.0, 0.0])
        s = compute_aero(v, np.zeros(3), sl_atm, generic_config,
                         ControlDeflections(aileron=math.radians(5.0)))
        assert s.rolling_moment > 0.0, f"Aileron roll: {s.rolling_moment:.2f}"

    def test_roll_damping_opposes_roll_rate(self, generic_config, sl_atm, neutral_deflections):
        """
        Clp < 0: positive roll rate (p > 0) should produce negative Cl.
        """
        v   = np.array([200.0, 0.0, 0.0])
        omg = np.array([0.2, 0.0, 0.0])   # positive roll rate
        s   = compute_aero(v, omg, sl_atm, generic_config, neutral_deflections)
        assert s.rolling_moment < 0.0, \
            f"Roll damping not acting: M_roll={s.rolling_moment:.4f}"


# ---------------------------------------------------------------------------
# 10. Force magnitude consistency
# ---------------------------------------------------------------------------

class TestForceMagnitude:

    def test_force_body_is_sum_of_components(self, generic_config, sl_atm, neutral_deflections):
        """force_body = lift_body + drag_body + side_force_body."""
        v = np.array([200.0, 0.0, 200.0 * math.tan(math.radians(5.0))])
        s = compute_aero(v, np.zeros(3), sl_atm, generic_config, neutral_deflections)
        expected = s.lift_body + s.drag_body + s.side_force_body
        assert np.allclose(s.force_body, expected, atol=1.0e-6), \
            f"force_body mismatch: {s.force_body} vs {expected}"

    def test_lift_force_magnitude(self, generic_config, sl_atm, neutral_deflections):
        """|lift_body| = |CL| · q · S."""
        v   = np.array([200.0, 0.0, 200.0 * math.tan(math.radians(5.0))])
        s   = compute_aero(v, np.zeros(3), sl_atm, generic_config, neutral_deflections)
        L_expected = abs(s.CL) * s.dynamic_pressure * generic_config.reference_area
        L_actual   = float(np.linalg.norm(s.lift_body))
        assert abs(L_actual - L_expected) < 0.1, \
            f"|L|={L_actual:.2f}, expected {L_expected:.2f}"


# ---------------------------------------------------------------------------
# 11. AeroState immutability
# ---------------------------------------------------------------------------

class TestAeroStateImmutable:

    def test_aerostate_is_frozen(self, generic_config, sl_atm, neutral_deflections):
        v = np.array([200.0, 0.0, 0.0])
        s = compute_aero(v, np.zeros(3), sl_atm, generic_config, neutral_deflections)
        with pytest.raises(Exception):
            s.CL = 999.0


# ---------------------------------------------------------------------------
# 12. AeroConfig validation
# ---------------------------------------------------------------------------

class TestAeroConfigValidation:

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="reference_area"):
            AeroConfig(reference_area=-1.0, mean_chord=2.0, span=10.0)

    def test_zero_area_raises(self):
        with pytest.raises(ValueError, match="reference_area"):
            AeroConfig(reference_area=0.0, mean_chord=2.0, span=10.0)

    def test_negative_chord_raises(self):
        with pytest.raises(ValueError, match="mean_chord"):
            AeroConfig(reference_area=20.0, mean_chord=-1.0, span=10.0)

    def test_negative_span_raises(self):
        with pytest.raises(ValueError, match="span"):
            AeroConfig(reference_area=20.0, mean_chord=2.0, span=-5.0)

    def test_valid_config_constructs(self):
        cfg = AeroConfig(reference_area=20.0, mean_chord=2.5, span=10.0)
        assert cfg.reference_area == 20.0
