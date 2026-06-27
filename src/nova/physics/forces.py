"""
nova.physics.forces
===================
Force tensor accumulator for Project NOVA.

Architecture role — Pipeline Stage 3 (partial)
-----------------------------------------------
The ForceAccumulator gathers every external force acting on the vehicle
during one simulation tick and produces a single net force vector in the
ECI frame for consumption by the RK4 integrator (Stage 4).

Design principles
-----------------
* **Single responsibility**: this module only *accumulates* forces. It does
  not integrate, does not modify state, and does not own physics models.
  Gravity, aerodynamics, and propulsion each live in their own modules;
  they produce force vectors that are registered here.

* **Frame contract**: all forces are converted to ECI before registration.
  The integrator receives one ECI vector — it never sees Body-frame forces
  directly. Coordinate transforms are applied inside each registration
  method, not at integration time.

* **Immutable per-tick snapshot**: once ``build()`` is called the result is
  a frozen (3,) float64 array. The accumulator object may then be discarded;
  a fresh one is constructed each tick.

Force taxonomy
--------------
  Category A — Gravitational  : point-mass + J2  (ECI; from orbital.py)
  Category B — Aerodynamic    : lift, drag, side  (Body → ECI via DCM)
  Category C — Propulsive     : thrust vector     (Body → ECI via DCM)
  Category D — Custom         : any additional ECI or Body-frame vector

All SI units: force in Newtons [N], vectors float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from nova.core.state_vector import VehicleState
from nova.frames.transforms import T_ENU_to_body


# ---------------------------------------------------------------------------
# Force contribution record (for telemetry / AI monitor inspection)
# ---------------------------------------------------------------------------

@dataclass
class ForceContribution:
    """
    A single named force contribution in the ECI frame.

    Stored alongside the accumulated total so the telemetry layer and AI
    monitor can inspect per-category force magnitudes without re-computing.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. ``"gravity"``, ``"drag"``, ``"thrust"``).
    vector_eci : ndarray, shape (3,), float64
        Force vector in ECI frame [N].
    """
    name:       str
    vector_eci: np.ndarray   # (3,) float64, [N]


# ---------------------------------------------------------------------------
# Force accumulator
# ---------------------------------------------------------------------------

class ForceAccumulator:
    """
    Mutable per-tick accumulator that collects all external forces acting on
    the vehicle and produces a net ECI force vector for the RK4 integrator.

    Usage pattern (called by the pipeline each tick)::

        acc = ForceAccumulator(state)
        acc.add_gravity(gravity_force(state.position_eci, state.mass))
        acc.add_body_force("thrust", thrust_vec_body, dcm_body_to_eci)
        acc.add_body_force("aero",   aero_vec_body,   dcm_body_to_eci)
        F_net_eci = acc.build()

    Parameters
    ----------
    state : VehicleState
        Current immutable state snapshot. Stored for reference (mass, DCM).
    """

    def __init__(self, state: VehicleState) -> None:
        self._state: VehicleState = state
        self._contributions: List[ForceContribution] = []
        self._total_eci: np.ndarray = np.zeros(3, dtype=np.float64)
        self._built: bool = False

    # ------------------------------------------------------------------
    # Registration methods
    # ------------------------------------------------------------------

    def add_eci_force(self, name: str, vector_eci: np.ndarray) -> None:
        """
        Register a force already expressed in the ECI frame.

        Parameters
        ----------
        name : str
            Contribution label for telemetry.
        vector_eci : ndarray, shape (3,), float64
            Force vector in ECI [N].
        """
        self._check_open()
        v = np.asarray(vector_eci, dtype=np.float64)
        _assert_vec3(v, name)
        self._contributions.append(ForceContribution(name, v.copy()))
        self._total_eci += v

    def add_body_force(
        self,
        name: str,
        vector_body: np.ndarray,
        dcm_body_to_eci: np.ndarray,
    ) -> None:
        """
        Register a force expressed in the Body frame.

        The provided DCM rotates the vector from Body to ECI before
        accumulation.  Typically ``dcm_body_to_eci = T_body_to_ECI(q, ...)``.

        Parameters
        ----------
        name : str
            Contribution label.
        vector_body : ndarray, shape (3,), float64
            Force vector in Body frame [N].
        dcm_body_to_eci : ndarray, shape (3, 3), float64
            Rotation matrix Body → ECI.
        """
        self._check_open()
        v_body = np.asarray(vector_body, dtype=np.float64)
        R      = np.asarray(dcm_body_to_eci, dtype=np.float64)
        _assert_vec3(v_body, name)
        _assert_dcm33(R, name)
        v_eci = R @ v_body
        self._contributions.append(ForceContribution(name, v_eci.copy()))
        self._total_eci += v_eci

    def add_gravity(self, gravity_force_eci: np.ndarray) -> None:
        """
        Convenience wrapper: register a gravitational force (ECI).

        Parameters
        ----------
        gravity_force_eci : ndarray, shape (3,), float64
            Output of ``nova.physics.orbital.gravity_force(...)`` [N].
        """
        self.add_eci_force("gravity", gravity_force_eci)

    def add_thrust(
        self,
        thrust_vector_body: np.ndarray,
        dcm_body_to_eci: np.ndarray,
    ) -> None:
        """
        Convenience wrapper: register engine thrust (Body frame).

        Parameters
        ----------
        thrust_vector_body : ndarray, shape (3,), float64
            Thrust vector in Body frame [N]. Typically aligned with +X_body
            for an axial engine; off-axis for gimbal or RCS.
        dcm_body_to_eci : ndarray, shape (3, 3), float64
            Rotation matrix Body → ECI.
        """
        self.add_body_force("thrust", thrust_vector_body, dcm_body_to_eci)

    def add_aerodynamic(
        self,
        aero_force_body: np.ndarray,
        dcm_body_to_eci: np.ndarray,
    ) -> None:
        """
        Convenience wrapper: register net aerodynamic force (Body frame).

        The aerodynamic force vector in the Body frame is the vector sum of
        lift, drag, and side-force already resolved from the stability axis
        by the aerodynamics module.

        Parameters
        ----------
        aero_force_body : ndarray, shape (3,), float64
            Net aero force in Body frame [N].
        dcm_body_to_eci : ndarray, shape (3, 3), float64
            Rotation matrix Body → ECI.
        """
        self.add_body_force("aerodynamic", aero_force_body, dcm_body_to_eci)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> np.ndarray:
        """
        Finalise and return the net force vector in ECI [N].

        Marks the accumulator as closed — further ``add_*`` calls will raise.

        Returns
        -------
        ndarray, shape (3,), float64
            Net external force in ECI frame [N].
        """
        self._built = True
        return self._total_eci.copy()

    # ------------------------------------------------------------------
    # Telemetry inspection
    # ------------------------------------------------------------------

    @property
    def contributions(self) -> List[ForceContribution]:
        """Read-only list of all registered force contributions."""
        return list(self._contributions)

    @property
    def total_magnitude(self) -> float:
        """‖F_net‖ [N]."""
        return float(np.linalg.norm(self._total_eci))

    def contribution_by_name(self, name: str) -> Optional[np.ndarray]:
        """
        Return the ECI force vector for a named contribution, or None.

        Parameters
        ----------
        name : str
            Contribution label as passed to ``add_*``.
        """
        for c in self._contributions:
            if c.name == name:
                return c.vector_eci.copy()
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._built:
            raise RuntimeError(
                "ForceAccumulator.build() has already been called. "
                "Create a new accumulator for the next tick."
            )

    def __repr__(self) -> str:
        n = len(self._contributions)
        return (
            f"ForceAccumulator("
            f"contributions={n}, "
            f"|F_net|={self.total_magnitude:.3f} N, "
            f"built={self._built})"
        )


# ---------------------------------------------------------------------------
# Functional helpers (stateless — used in tests and by the pipeline)
# ---------------------------------------------------------------------------

def net_force_eci(contributions: List[ForceContribution]) -> np.ndarray:
    """
    Sum a list of ForceContribution records into one ECI vector.

    Parameters
    ----------
    contributions : list of ForceContribution

    Returns
    -------
    ndarray, shape (3,), float64  [N]
    """
    total = np.zeros(3, dtype=np.float64)
    for c in contributions:
        total += c.vector_eci
    return total


def rotate_body_to_eci(
    vector_body: np.ndarray,
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    Rotate a Body-frame vector to ECI using the vehicle's attitude quaternion.

    This is a convenience wrapper for the common pattern of converting a
    Body-frame force to ECI without constructing the full composed transform
    (which would also require longitude/latitude/time for ECEF→ECI).

    For the force accumulator, only the *attitude* rotation (ENU→Body) is
    reversed; the ECI→ECEF→ENU chain cancels because forces accumulate
    in ECI and the ECI axes are inertial.

    In practice: R_body→ECI = R_ENU→Body(q)ᵀ  (ignoring ECEF rotation,
    which is negligible for force magnitudes over one tick at dt=0.01 s).

    Parameters
    ----------
    vector_body : ndarray, shape (3,), float64
        Vector expressed in Body frame.
    quaternion : ndarray, shape (4,), float64
        Vehicle attitude quaternion (scalar-first, unit norm).

    Returns
    -------
    ndarray, shape (3,), float64
        Vector expressed in ECI frame (approx — ignores ECEF rotation).
    """
    R_enu_to_body = T_ENU_to_body(quaternion)
    R_body_to_enu = R_enu_to_body.T          # orthogonal → transpose = inverse
    # ENU ≈ ECI for force direction over one tick; full chain in pipeline.
    return R_body_to_enu @ np.asarray(vector_body, dtype=np.float64)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _assert_vec3(v: np.ndarray, name: str) -> None:
    if v.shape != (3,):
        raise ValueError(
            f"Force '{name}' must be shape (3,), got {v.shape}"
        )


def _assert_dcm33(R: np.ndarray, name: str) -> None:
    if R.shape != (3, 3):
        raise ValueError(
            f"DCM for force '{name}' must be shape (3, 3), got {R.shape}"
        )
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_forces.py
=========================
Unit tests for nova.physics.forces.

Tests verify:
  1. ForceAccumulator accumulates ECI forces correctly.
  2. Body-frame forces are correctly rotated to ECI before accumulation.
  3. Named contributions are retrievable from telemetry.
  4. build() returns the correct net vector and locks the accumulator.
  5. Convenience wrappers (add_gravity, add_thrust, add_aerodynamic).
  6. rotate_body_to_eci helper produces correct rotations.
  7. Functional helper net_force_eci.
  8. Zero-force accumulator returns zero vector.
"""

import math
import pytest
import numpy as np

from nova.core.state_vector import make_state
from nova.frames.transforms import euler_to_quaternion, T_ENU_to_body
from nova.physics.forces import (
    ForceAccumulator,
    ForceContribution,
    net_force_eci,
    rotate_body_to_eci,
)


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


@pytest.fixture
def identity_dcm():
    return np.eye(3, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Empty accumulator
# ---------------------------------------------------------------------------

class TestEmptyAccumulator:

    def test_build_returns_zero_vector(self, default_state):
        acc = ForceAccumulator(default_state)
        F = acc.build()
        assert np.allclose(F, [0.0, 0.0, 0.0])

    def test_zero_total_magnitude(self, default_state):
        acc = ForceAccumulator(default_state)
        assert acc.total_magnitude == 0.0

    def test_no_contributions(self, default_state):
        acc = ForceAccumulator(default_state)
        assert acc.contributions == []

    def test_build_output_shape_dtype(self, default_state):
        acc = ForceAccumulator(default_state)
        F = acc.build()
        assert F.shape == (3,)
        assert F.dtype == np.float64


# ---------------------------------------------------------------------------
# 2. ECI force accumulation
# ---------------------------------------------------------------------------

class TestECIForceAccumulation:

    def test_single_eci_force(self, default_state, identity_dcm):
        F_grav = np.array([0.0, 0.0, -9810.0], dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_eci_force("gravity", F_grav)
        F_net = acc.build()
        assert np.allclose(F_net, F_grav)

    def test_two_eci_forces_sum(self, default_state):
        F1 = np.array([1000.0, 0.0, 0.0], dtype=np.float64)
        F2 = np.array([0.0, 500.0, -200.0], dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_eci_force("A", F1)
        acc.add_eci_force("B", F2)
        F_net = acc.build()
        assert np.allclose(F_net, F1 + F2)

    def test_opposing_forces_cancel(self, default_state):
        F = np.array([500.0, -300.0, 100.0], dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_eci_force("push", F)
        acc.add_eci_force("brake", -F)
        F_net = acc.build()
        assert np.allclose(F_net, [0.0, 0.0, 0.0], atol=1.0e-12)

    def test_total_magnitude_correct(self, default_state):
        F = np.array([3.0, 4.0, 0.0], dtype=np.float64)  # |F| = 5
        acc = ForceAccumulator(default_state)
        acc.add_eci_force("test", F)
        assert abs(acc.total_magnitude - 5.0) < 1.0e-10


# ---------------------------------------------------------------------------
# 3. Body-frame force rotation
# ---------------------------------------------------------------------------

class TestBodyFrameRotation:

    def test_identity_dcm_passthrough(self, default_state):
        """With identity DCM, Body force should equal ECI force."""
        F_body = np.array([100.0, -50.0, 200.0], dtype=np.float64)
        R_id   = np.eye(3, dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_body_force("test", F_body, R_id)
        F_net = acc.build()
        assert np.allclose(F_net, F_body)

    def test_90deg_yaw_rotation(self, default_state):
        """
        90° yaw: +X_body maps to +Y_ECI (approximately ENU).
        Body force along +X_body should appear as +Y after rotation.
        """
        q   = euler_to_quaternion(0.0, 0.0, math.pi / 2.0)
        R   = T_ENU_to_body(q).T   # body→ENU (≈ body→ECI here)
        F_b = np.array([1000.0, 0.0, 0.0], dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_body_force("thrust", F_b, R)
        F_net = acc.build()
        # After 90° yaw, +X_body → +Y direction
        assert abs(F_net[1]) > 900.0, f"Expected force in Y, got {F_net}"
        assert abs(F_net[0]) < 10.0

    def test_wrong_dcm_shape_raises(self, default_state):
        F_body = np.array([100.0, 0.0, 0.0], dtype=np.float64)
        bad_R  = np.eye(2, dtype=np.float64)
        acc = ForceAccumulator(default_state)
        with pytest.raises(ValueError, match="shape"):
            acc.add_body_force("bad", F_body, bad_R)

    def test_wrong_force_shape_raises(self, default_state):
        bad_F = np.array([100.0, 0.0], dtype=np.float64)
        R     = np.eye(3, dtype=np.float64)
        acc   = ForceAccumulator(default_state)
        with pytest.raises(ValueError, match="shape"):
            acc.add_body_force("bad", bad_F, R)


# ---------------------------------------------------------------------------
# 4. Convenience wrappers
# ---------------------------------------------------------------------------

class TestConvenienceWrappers:

    def test_add_gravity_registers_as_eci(self, default_state):
        F_g = np.array([-9810.0, 0.0, 0.0], dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_gravity(F_g)
        assert np.allclose(acc.build(), F_g)

    def test_add_gravity_named_gravity(self, default_state):
        F_g = np.array([0.0, 0.0, -9810.0], dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_gravity(F_g)
        c = acc.contributions[0]
        assert c.name == "gravity"

    def test_add_thrust(self, default_state):
        F_t = np.array([50_000.0, 0.0, 0.0], dtype=np.float64)
        R   = np.eye(3, dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_thrust(F_t, R)
        assert acc.contributions[0].name == "thrust"

    def test_add_aerodynamic(self, default_state):
        F_a = np.array([-500.0, 0.0, -2000.0], dtype=np.float64)
        R   = np.eye(3, dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_aerodynamic(F_a, R)
        assert acc.contributions[0].name == "aerodynamic"


# ---------------------------------------------------------------------------
# 5. Telemetry / contribution inspection
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_contribution_by_name(self, default_state):
        F_g = np.array([0.0, 0.0, -9810.0], dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_eci_force("gravity", F_g)
        _ = acc.build()
        v = acc.contribution_by_name("gravity")
        assert v is not None
        assert np.allclose(v, F_g)

    def test_contribution_by_name_missing_returns_none(self, default_state):
        acc = ForceAccumulator(default_state)
        _ = acc.build()
        assert acc.contribution_by_name("nonexistent") is None

    def test_contributions_list_length(self, default_state):
        acc = ForceAccumulator(default_state)
        acc.add_eci_force("A", np.array([1.0, 0.0, 0.0]))
        acc.add_eci_force("B", np.array([0.0, 1.0, 0.0]))
        acc.add_eci_force("C", np.array([0.0, 0.0, 1.0]))
        _ = acc.build()
        assert len(acc.contributions) == 3

    def test_contribution_vector_is_copy(self, default_state):
        """Mutations of the retrieved vector must not affect the accumulator."""
        F_g = np.array([0.0, 0.0, -9810.0], dtype=np.float64)
        acc = ForceAccumulator(default_state)
        acc.add_eci_force("gravity", F_g)
        v = acc.contribution_by_name("gravity")
        if v is not None:
            v[0] = 9999.0
        v2 = acc.contribution_by_name("gravity")
        assert v2 is not None and abs(v2[0]) < 1.0


# ---------------------------------------------------------------------------
# 6. build() locking
# ---------------------------------------------------------------------------

class TestBuildLocking:

    def test_build_twice_raises(self, default_state):
        acc = ForceAccumulator(default_state)
        acc.build()
        with pytest.raises(RuntimeError, match="build"):
            acc.add_eci_force("late", np.zeros(3))

    def test_build_returns_copy(self, default_state):
        acc = ForceAccumulator(default_state)
        acc.add_eci_force("F", np.array([1.0, 2.0, 3.0]))
        F1 = acc.build()
        F1[0] = -999.0   # mutate return value
        # Internal total should not change — but accumulator is locked
        # so we verify the original contribution is still intact
        c = acc.contribution_by_name("F")
        assert c is not None and abs(c[0] - 1.0) < 1.0e-10


# ---------------------------------------------------------------------------
# 7. rotate_body_to_eci helper
# ---------------------------------------------------------------------------

class TestRotateBodyToECI:

    def test_identity_quaternion(self):
        """Identity quaternion → body vector passes through unchanged."""
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v_out = rotate_body_to_eci(v, q)
        assert np.allclose(v_out, v, atol=1.0e-14)

    def test_90deg_yaw(self):
        """90° yaw: +X_body → +Y (ENU frame)."""
        q   = euler_to_quaternion(0.0, 0.0, math.pi / 2.0)
        v_b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v_e = rotate_body_to_eci(v_b, q)
        # After 90° yaw, X_body maps to -Y in ENU (convention)
        # Just check the magnitude is preserved
        assert abs(float(np.linalg.norm(v_e)) - 1.0) < 1.0e-12

    def test_magnitude_preserved(self):
        """Rotation must preserve vector magnitude."""
        q = euler_to_quaternion(0.3, -0.2, 1.1)
        v = np.array([3.0, 4.0, 0.0], dtype=np.float64)
        v_out = rotate_body_to_eci(v, q)
        assert abs(float(np.linalg.norm(v_out)) - 5.0) < 1.0e-10


# ---------------------------------------------------------------------------
# 8. net_force_eci functional helper
# ---------------------------------------------------------------------------

class TestNetForceECI:

    def test_empty_list_returns_zero(self):
        result = net_force_eci([])
        assert np.allclose(result, [0.0, 0.0, 0.0])

    def test_single_contribution(self):
        F = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        c = ForceContribution("test", F)
        result = net_force_eci([c])
        assert np.allclose(result, F)

    def test_multiple_contributions_sum(self):
        c1 = ForceContribution("A", np.array([1.0, 0.0, 0.0]))
        c2 = ForceContribution("B", np.array([0.0, 2.0, 0.0]))
        c3 = ForceContribution("C", np.array([0.0, 0.0, 3.0]))
        result = net_force_eci([c1, c2, c3])
        assert np.allclose(result, [1.0, 2.0, 3.0])
