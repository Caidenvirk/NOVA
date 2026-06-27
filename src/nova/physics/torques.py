"""
nova.physics.torques
====================
Torque tensor accumulator for Project NOVA.

Architecture role — Pipeline Stage 3 (partial)
-----------------------------------------------
The TorqueAccumulator mirrors the ForceAccumulator but operates on torques
(moments), which drive the rotational equations of motion in the RK4
integrator via Euler's rotation equations:

    I ω̇ = τ_net − ω × (I ω)

All torques are accumulated in the **Body Frame** because the inertia
tensor I is most naturally expressed in body-fixed coordinates. The RK4
integrator receives the Body-frame τ_net directly.

Torque taxonomy
---------------
  Category A — Aerodynamic moments  : pitching, yawing, rolling moments
                                       applied at the Aerodynamic Centre (AC)
                                       relative to the Centre of Mass (CoM).
  Category B — Thrust gimbal        : moment arm × thrust force for gimballed
                                       engines or offset RCS thrusters.
  Category C — Gravity gradient     : tidal torque for large structures in LEO.
  Category D — Custom               : any additional Body-frame torque vector.

All moments are in Newton-metres [N·m], vectors float64.

Sign convention
---------------
Right-hand rule in the Body Frame:
  +τ_x  → rolling  moment (right-wing down for positive roll)
  +τ_y  → pitching moment (nose up for positive pitch)
  +τ_z  → yawing   moment (nose right for positive yaw)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Torque contribution record
# ---------------------------------------------------------------------------

@dataclass
class TorqueContribution:
    """
    A single named torque contribution in the Body frame [N·m].

    Attributes
    ----------
    name : str
        Human-readable label (e.g. ``"aero_pitch"``, ``"gimbal"``).
    vector_body : ndarray, shape (3,), float64
        Torque vector in Body frame [N·m]. Components: (roll, pitch, yaw).
    """
    name:        str
    vector_body: np.ndarray   # (3,) float64, [N·m]


# ---------------------------------------------------------------------------
# Torque accumulator
# ---------------------------------------------------------------------------

class TorqueAccumulator:
    """
    Mutable per-tick accumulator that collects all external torques acting on
    the vehicle and produces a net Body-frame torque vector for the RK4
    integrator.

    Usage pattern (called by the pipeline each tick)::

        tacc = TorqueAccumulator(state)
        tacc.add_body_torque("aero_pitch", np.array([0, M_pitch, 0]))
        tacc.add_moment_arm("gimbal", thrust_body, moment_arm_body)
        tau_net_body = tacc.build()

    Parameters
    ----------
    state : VehicleState
        Current immutable state snapshot (stored for reference).
    """

    def __init__(self, state) -> None:
        self._state = state
        self._contributions: List[TorqueContribution] = []
        self._total_body: np.ndarray = np.zeros(3, dtype=np.float64)
        self._built: bool = False

    # ------------------------------------------------------------------
    # Registration methods
    # ------------------------------------------------------------------

    def add_body_torque(self, name: str, vector_body: np.ndarray) -> None:
        """
        Register a torque already expressed in the Body frame.

        Parameters
        ----------
        name : str
            Contribution label for telemetry.
        vector_body : ndarray, shape (3,), float64
            Torque vector in Body frame [N·m].
            Components: (τ_roll, τ_pitch, τ_yaw).
        """
        self._check_open()
        v = np.asarray(vector_body, dtype=np.float64)
        _assert_vec3(v, name)
        self._contributions.append(TorqueContribution(name, v.copy()))
        self._total_body += v

    def add_moment_arm(
        self,
        name: str,
        force_body: np.ndarray,
        moment_arm_body: np.ndarray,
    ) -> None:
        """
        Register a torque from a force applied at an offset from the CoM.

        τ = r × F   (cross product in Body frame)

        Used for:
          - Gimballed engine thrust offset from CoM
          - RCS thruster moment arms
          - Aerodynamic force at AC offset from CoM

        Parameters
        ----------
        name : str
            Contribution label.
        force_body : ndarray, shape (3,), float64
            Applied force in Body frame [N].
        moment_arm_body : ndarray, shape (3,), float64
            Vector from CoM to the point of force application,
            expressed in Body frame [m].
            τ = moment_arm × force
        """
        self._check_open()
        F = np.asarray(force_body,      dtype=np.float64)
        r = np.asarray(moment_arm_body, dtype=np.float64)
        _assert_vec3(F, f"{name}/force")
        _assert_vec3(r, f"{name}/moment_arm")
        tau = np.cross(r, F)
        self._contributions.append(TorqueContribution(name, tau.copy()))
        self._total_body += tau

    def add_aerodynamic_moments(
        self,
        pitching_moment: float,
        yawing_moment:   float,
        rolling_moment:  float,
    ) -> None:
        """
        Register the three aerodynamic moment components.

        These are the dimensionalised moments (already multiplied by
        dynamic pressure, reference area, and reference length) computed
        by the aerodynamics module.

        Parameters
        ----------
        pitching_moment : float
            M_pitch [N·m] — positive nose-up.
        yawing_moment : float
            M_yaw [N·m] — positive nose-right.
        rolling_moment : float
            M_roll [N·m] — positive right-wing-down.
        """
        tau = np.array([rolling_moment, pitching_moment, yawing_moment],
                       dtype=np.float64)
        self.add_body_torque("aerodynamic", tau)

    def add_gravity_gradient(
        self,
        position_body: np.ndarray,
        inertia_tensor: np.ndarray,
        mu: float,
    ) -> None:
        """
        Gravity gradient torque for an extended body in a non-uniform
        gravitational field.

        τ_gg = (3μ/r⁵) · r_body × (I · r_body)

        where r_body is the position vector in Body frame and I is the
        inertia tensor.

        This effect is significant for large spacecraft (e.g. space stations,
        solar arrays) in LEO but negligible for compact launch vehicles.
        Include only when the vehicle's largest dimension exceeds ~10 m.

        Parameters
        ----------
        position_body : ndarray, shape (3,), float64
            Vehicle position vector expressed in the Body frame [m].
        inertia_tensor : ndarray, shape (3, 3), float64
            Inertia tensor in Body frame [kg·m²].
        mu : float
            Gravitational parameter [m³ s⁻²].
        """
        self._check_open()
        r = np.asarray(position_body,  dtype=np.float64)
        I = np.asarray(inertia_tensor, dtype=np.float64)
        r_mag = float(np.linalg.norm(r))
        if r_mag < 1.0:
            return   # degenerate — skip
        coeff = 3.0 * mu / r_mag**5
        tau   = coeff * np.cross(r, I @ r)
        self._contributions.append(TorqueContribution("gravity_gradient", tau.copy()))
        self._total_body += tau

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> np.ndarray:
        """
        Finalise and return the net torque vector in Body frame [N·m].

        Marks the accumulator as closed.

        Returns
        -------
        ndarray, shape (3,), float64
            Net external torque in Body frame [N·m].
            Components: (τ_roll, τ_pitch, τ_yaw).
        """
        self._built = True
        return self._total_body.copy()

    # ------------------------------------------------------------------
    # Telemetry inspection
    # ------------------------------------------------------------------

    @property
    def contributions(self) -> List[TorqueContribution]:
        """Read-only list of all registered torque contributions."""
        return list(self._contributions)

    @property
    def total_magnitude(self) -> float:
        """‖τ_net‖ [N·m]."""
        return float(np.linalg.norm(self._total_body))

    def contribution_by_name(self, name: str) -> Optional[np.ndarray]:
        """
        Return the Body-frame torque vector for a named contribution, or None.
        """
        for c in self._contributions:
            if c.name == name:
                return c.vector_body.copy()
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._built:
            raise RuntimeError(
                "TorqueAccumulator.build() has already been called. "
                "Create a new accumulator for the next tick."
            )

    def __repr__(self) -> str:
        n = len(self._contributions)
        return (
            f"TorqueAccumulator("
            f"contributions={n}, "
            f"|τ_net|={self.total_magnitude:.3f} N·m, "
            f"built={self._built})"
        )


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _assert_vec3(v: np.ndarray, name: str) -> None:
    if v.shape != (3,):
        raise ValueError(
            f"Torque vector '{name}' must be shape (3,), got {v.shape}"
        )
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_torques.py
==========================
Unit tests for nova.physics.torques.

Tests verify:
  1. TorqueAccumulator accumulates Body-frame torques correctly.
  2. add_moment_arm computes τ = r × F exactly.
  3. add_aerodynamic_moments maps (pitch, yaw, roll) to correct Body indices.
  4. add_gravity_gradient produces a physically reasonable torque.
  5. Named contributions retrievable via contribution_by_name.
  6. build() locks the accumulator; further add_* raise RuntimeError.
  7. Zero-torque accumulator returns zero vector.
"""

import math
import pytest
import numpy as np

from nova.core.state_vector import make_state
from nova.physics.torques import TorqueAccumulator, TorqueContribution
from nova.core.constants import EARTH_MU


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_state():
    return make_state(
        position_eci=[6_771_000.0, 0.0, 0.0],
        velocity_eci=[0.0, 7_672.0, 0.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        omega_body=[0.0, 0.0, 0.0],
        mass=1000.0,
    )


# ---------------------------------------------------------------------------
# 1. Empty accumulator
# ---------------------------------------------------------------------------

class TestEmptyTorqueAccumulator:

    def test_build_returns_zero(self, default_state):
        tacc = TorqueAccumulator(default_state)
        tau  = tacc.build()
        assert np.allclose(tau, [0.0, 0.0, 0.0])

    def test_shape_dtype(self, default_state):
        tacc = TorqueAccumulator(default_state)
        tau  = tacc.build()
        assert tau.shape == (3,)
        assert tau.dtype == np.float64

    def test_total_magnitude_zero(self, default_state):
        tacc = TorqueAccumulator(default_state)
        assert tacc.total_magnitude == 0.0

    def test_no_contributions(self, default_state):
        tacc = TorqueAccumulator(default_state)
        assert tacc.contributions == []


# ---------------------------------------------------------------------------
# 2. Body-torque accumulation
# ---------------------------------------------------------------------------

class TestBodyTorqueAccumulation:

    def test_single_torque(self, default_state):
        tau_in = np.array([10.0, -5.0, 2.0], dtype=np.float64)
        tacc   = TorqueAccumulator(default_state)
        tacc.add_body_torque("test", tau_in)
        assert np.allclose(tacc.build(), tau_in)

    def test_two_torques_sum(self, default_state):
        t1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        t2 = np.array([4.0, -1.0, 0.0], dtype=np.float64)
        tacc = TorqueAccumulator(default_state)
        tacc.add_body_torque("A", t1)
        tacc.add_body_torque("B", t2)
        assert np.allclose(tacc.build(), t1 + t2)

    def test_opposing_torques_cancel(self, default_state):
        t = np.array([5.0, -3.0, 8.0], dtype=np.float64)
        tacc = TorqueAccumulator(default_state)
        tacc.add_body_torque("fwd", t)
        tacc.add_body_torque("rev", -t)
        assert np.allclose(tacc.build(), [0.0, 0.0, 0.0], atol=1.0e-14)

    def test_wrong_shape_raises(self, default_state):
        tacc = TorqueAccumulator(default_state)
        with pytest.raises(ValueError, match="shape"):
            tacc.add_body_torque("bad", np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# 3. Moment arm: τ = r × F
# ---------------------------------------------------------------------------

class TestMomentArm:

    def test_axial_force_lateral_offset_pitching_moment(self, default_state):
        """
        Engine at [−5, 0, 0] from CoM, force along +X → zero torque
        (force along moment arm).
        """
        r_arm = np.array([-5.0, 0.0, 0.0], dtype=np.float64)
        F     = np.array([1000.0, 0.0, 0.0], dtype=np.float64)
        tacc  = TorqueAccumulator(default_state)
        tacc.add_moment_arm("engine", F, r_arm)
        tau = tacc.build()
        # r × F = [−5,0,0] × [1000,0,0] = [0,0,0]
        assert np.allclose(tau, [0.0, 0.0, 0.0], atol=1.0e-12)

    def test_vertical_offset_lateral_force(self, default_state):
        """
        Force along +Y at r = [0, 0, −2] (below CoM):
        τ = [0,0,−2] × [0,F,0] = [−2·F·ẑ×ŷ] = [2F, 0, 0] → roll.
        """
        r_arm = np.array([0.0, 0.0, -2.0], dtype=np.float64)
        F     = np.array([0.0, 500.0, 0.0], dtype=np.float64)
        tacc  = TorqueAccumulator(default_state)
        tacc.add_moment_arm("rcs", F, r_arm)
        tau   = tacc.build()
        # [0,0,−2] × [0,500,0] = (0·0−(−2)·500, (−2)·0−0·0, 0·500−0·0)
        #                        = (1000, 0, 0)
        expected = np.cross(r_arm, F)
        assert np.allclose(tau, expected, atol=1.0e-10)

    def test_known_cross_product(self, default_state):
        """
        r = [1, 0, 0], F = [0, 1, 0] → τ = r × F = [0, 0, 1].
        """
        r_arm = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        F     = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tacc  = TorqueAccumulator(default_state)
        tacc.add_moment_arm("unit", F, r_arm)
        tau   = tacc.build()
        assert np.allclose(tau, [0.0, 0.0, 1.0], atol=1.0e-14)

    def test_gimbal_moment_arm(self, default_state):
        """
        Engine at [−10, 0, 0], thrust 50 000 N along +X with 2° yaw gimbal.
        Gimbal deflects thrust into Y by F·sin(2°).
        τ_z = r_x × F_y = −10 × (−50000·sin2°) ... handled by the moment arm.
        """
        import math
        gim   = math.radians(2.0)
        F_y   = 50_000.0 * math.sin(gim)
        r_arm = np.array([-10.0, 0.0, 0.0], dtype=np.float64)
        F     = np.array([50_000.0 * math.cos(gim), F_y, 0.0])
        tacc  = TorqueAccumulator(default_state)
        tacc.add_moment_arm("gimbal", F, r_arm)
        tau   = tacc.build()
        expected = np.cross(r_arm, F)
        assert np.allclose(tau, expected, atol=1.0e-6)

    def test_wrong_force_shape_raises(self, default_state):
        tacc = TorqueAccumulator(default_state)
        with pytest.raises(ValueError, match="shape"):
            tacc.add_moment_arm("bad", np.array([1.0, 0.0]), np.array([0.0, 0.0, 1.0]))

    def test_wrong_arm_shape_raises(self, default_state):
        tacc = TorqueAccumulator(default_state)
        with pytest.raises(ValueError, match="shape"):
            tacc.add_moment_arm("bad", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# 4. Aerodynamic moment convenience wrapper
# ---------------------------------------------------------------------------

class TestAerodynamicMoments:

    def test_components_assigned_to_correct_body_axes(self, default_state):
        """
        Body convention: index 0 = roll, 1 = pitch, 2 = yaw.
        add_aerodynamic_moments(pitch=P, yaw=Y, roll=R) → [R, P, Y].
        """
        tacc = TorqueAccumulator(default_state)
        tacc.add_aerodynamic_moments(
            pitching_moment=1000.0,
            yawing_moment=500.0,
            rolling_moment=200.0,
        )
        tau = tacc.build()
        assert abs(tau[0] - 200.0)  < 1.0e-10, f"roll: {tau[0]}"
        assert abs(tau[1] - 1000.0) < 1.0e-10, f"pitch: {tau[1]}"
        assert abs(tau[2] - 500.0)  < 1.0e-10, f"yaw: {tau[2]}"

    def test_named_aerodynamic(self, default_state):
        tacc = TorqueAccumulator(default_state)
        tacc.add_aerodynamic_moments(100.0, 200.0, 300.0)
        _ = tacc.build()
        assert tacc.contributions[0].name == "aerodynamic"

    def test_zero_moments(self, default_state):
        tacc = TorqueAccumulator(default_state)
        tacc.add_aerodynamic_moments(0.0, 0.0, 0.0)
        assert np.allclose(tacc.build(), [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# 5. Gravity gradient torque
# ---------------------------------------------------------------------------

class TestGravityGradient:

    def test_nonzero_for_asymmetric_inertia_off_axis(self, default_state):
        """
        Gravity gradient torque is non-zero when I is non-spherical and
        position vector is not aligned with a principal axis.
        """
        I     = np.diag([200.0, 400.0, 300.0])   # asymmetric
        r_b   = np.array([7_000_000.0, 100_000.0, 50_000.0])
        tacc  = TorqueAccumulator(default_state)
        tacc.add_gravity_gradient(r_b, I, EARTH_MU)
        tau   = tacc.build()
        assert float(np.linalg.norm(tau)) > 1.0e-6, \
            f"Expected nonzero gravity gradient, got {tau}"

    def test_zero_for_spherical_inertia_along_axis(self, default_state):
        """
        For a spherical inertia tensor (I = kI₃), r × (I·r) = k·r × r = 0.
        """
        I   = np.diag([100.0, 100.0, 100.0])   # spherical
        r_b = np.array([7_000_000.0, 0.0, 0.0])
        tacc = TorqueAccumulator(default_state)
        tacc.add_gravity_gradient(r_b, I, EARTH_MU)
        tau  = tacc.build()
        assert np.allclose(tau, [0.0, 0.0, 0.0], atol=1.0e-6)

    def test_degenerate_zero_position_skipped(self, default_state):
        """Zero position vector → gravity gradient skipped, no exception."""
        I   = np.diag([200.0, 300.0, 150.0])
        r_b = np.zeros(3, dtype=np.float64)
        tacc = TorqueAccumulator(default_state)
        tacc.add_gravity_gradient(r_b, I, EARTH_MU)
        assert np.allclose(tacc.build(), [0.0, 0.0, 0.0])
        assert len(tacc.contributions) == 0   # nothing registered

    def test_named_gravity_gradient(self, default_state):
        I   = np.diag([200.0, 400.0, 300.0])
        r_b = np.array([7_000_000.0, 100_000.0, 0.0])
        tacc = TorqueAccumulator(default_state)
        tacc.add_gravity_gradient(r_b, I, EARTH_MU)
        _ = tacc.build()
        assert tacc.contributions[0].name == "gravity_gradient"


# ---------------------------------------------------------------------------
# 6. Telemetry
# ---------------------------------------------------------------------------

class TestTorqueTelemetry:

    def test_contribution_by_name_found(self, default_state):
        tau_in = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        tacc   = TorqueAccumulator(default_state)
        tacc.add_body_torque("aero_pitch", tau_in)
        _ = tacc.build()
        v = tacc.contribution_by_name("aero_pitch")
        assert v is not None
        assert np.allclose(v, tau_in)

    def test_contribution_by_name_missing(self, default_state):
        tacc = TorqueAccumulator(default_state)
        _ = tacc.build()
        assert tacc.contribution_by_name("nonexistent") is None

    def test_contribution_vector_is_copy(self, default_state):
        tau_in = np.array([5.0, 0.0, 0.0], dtype=np.float64)
        tacc   = TorqueAccumulator(default_state)
        tacc.add_body_torque("roll", tau_in)
        v = tacc.contribution_by_name("roll")
        if v is not None:
            v[0] = -9999.0
        v2 = tacc.contribution_by_name("roll")
        assert v2 is not None and abs(v2[0] - 5.0) < 1.0e-10


# ---------------------------------------------------------------------------
# 7. build() locking
# ---------------------------------------------------------------------------

class TestTorqueLocking:

    def test_add_after_build_raises(self, default_state):
        tacc = TorqueAccumulator(default_state)
        tacc.build()
        with pytest.raises(RuntimeError, match="build"):
            tacc.add_body_torque("late", np.zeros(3))

    def test_build_returns_copy(self, default_state):
        tacc = TorqueAccumulator(default_state)
        tacc.add_body_torque("t", np.array([1.0, 2.0, 3.0]))
        tau1 = tacc.build()
        tau1[0] = -999.0   # mutate return value
        # contribution should be unchanged
        c = tacc.contribution_by_name("t")
        assert c is not None and abs(c[0] - 1.0) < 1.0e-10

    def test_repr_shows_built_status(self, default_state):
        tacc = TorqueAccumulator(default_state)
        tacc.add_body_torque("x", np.array([10.0, 0.0, 0.0]))
        r1 = repr(tacc)
        tacc.build()
        r2 = repr(tacc)
        assert "built=True" in r2
        assert "built=False" in 
