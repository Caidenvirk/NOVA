"""
nova.core.state_vector
======================
Defines the canonical, immutable VehicleState — the complete 13-element
state vector propagated by the RK4 integrator at every simulation tick.

Design constraints
------------------
* Frozen dataclass: no field may be mutated after construction. The
  integrator produces a *new* VehicleState each tick; it never modifies
  an existing one. This enforces the deterministic, race-condition-free
  pipeline defined in the architecture.

* All vectors are NumPy arrays with dtype=np.float64 (64-bit IEEE 754).
  This is non-negotiable: float32 accumulates ~1 ULP/step error that
  violates the 1×10⁻⁶ conservation tolerances at simulation timescales.

* Angles are stored in radians. The quaternion `q` encodes orientation
  as (q₀, q₁, q₂, q₃) = (scalar, i, j, k).  ‖q‖ = 1 is maintained
  by the integrator after every RK4 step.

* The state is frame-tagged. `position_eci` and `velocity_eci` live in
  the Earth-Centred Inertial frame. Body-frame quantities are derivable
  via the coordinate transforms in nova.frames.transforms.

State vector layout (13 scalars → stored as named array fields)
---------------------------------------------------------------
  [0:3]   position_eci     [m]          ECI Cartesian position
  [3:6]   velocity_eci     [m s⁻¹]      ECI Cartesian velocity
  [6:10]  quaternion       [-]          q = (q0, q1, q2, q3), ‖q‖ = 1
  [10:13] omega_body       [rad s⁻¹]    Angular velocity in Body Frame (p, q, r)

Additional scalar fields (not part of the ODE state, but carried in the
snapshot for telemetry):
  mass                     [kg]         Current total vehicle mass
  time                     [s]          Mission elapsed time
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nova.core.constants import QUATERNION_NORM_TOL


# ---------------------------------------------------------------------------
# VehicleState — frozen, immutable simulation state snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleState:
    """
    Complete rigid-body state of the simulated vehicle at one instant in time.

    All vectors are stored as 1-D NumPy float64 arrays. The dataclass is
    frozen so any attempted mutation raises ``FrozenInstanceError`` at
    runtime — enforcing the immutable-snapshot contract required by the
    deterministic pipeline.

    Parameters
    ----------
    position_eci : ndarray, shape (3,), dtype float64
        Cartesian position in the ECI frame [m].
    velocity_eci : ndarray, shape (3,), dtype float64
        Cartesian velocity in the ECI frame [m s⁻¹].
    quaternion : ndarray, shape (4,), dtype float64
        Orientation quaternion (q0, q1, q2, q3) — scalar part first.
        Must satisfy ‖q‖ ≈ 1.0 within QUATERNION_NORM_TOL.
    omega_body : ndarray, shape (3,), dtype float64
        Angular velocity vector expressed in the Body Frame [rad s⁻¹].
        Components are (p, q, r) — roll rate, pitch rate, yaw rate.
    mass : float
        Total vehicle mass at this instant [kg]. Must be > 0.
    time : float
        Mission elapsed time (MET) at this state [s]. Must be ≥ 0.
    """

    position_eci: np.ndarray   # shape (3,), float64, [m]
    velocity_eci: np.ndarray   # shape (3,), float64, [m s⁻¹]
    quaternion:   np.ndarray   # shape (4,), float64, unit quaternion
    omega_body:   np.ndarray   # shape (3,), float64, [rad s⁻¹]
    mass:         float        # [kg], > 0
    time:         float        # [s], MET ≥ 0

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate shapes, dtypes, and physical constraints on construction."""

        # Enforce numpy array types and shapes
        _assert_vector(self.position_eci, shape=(3,), name="position_eci")
        _assert_vector(self.velocity_eci, shape=(3,), name="velocity_eci")
        _assert_vector(self.quaternion,   shape=(4,), name="quaternion")
        _assert_vector(self.omega_body,   shape=(3,), name="omega_body")

        # Physical constraints
        if self.mass <= 0.0:
            raise ValueError(
                f"VehicleState.mass must be > 0 kg, got {self.mass!r}"
            )
        if self.time < 0.0:
            raise ValueError(
                f"VehicleState.time must be ≥ 0 s (MET), got {self.time!r}"
            )

        # Quaternion unit-norm check
        q_norm = float(np.linalg.norm(self.quaternion))
        if abs(q_norm - 1.0) > QUATERNION_NORM_TOL:
            raise ValueError(
                f"VehicleState.quaternion must be a unit quaternion "
                f"(‖q‖ = 1.0 ± {QUATERNION_NORM_TOL}), "
                f"got ‖q‖ = {q_norm:.15f}"
            )

    # ------------------------------------------------------------------
    # Convenience properties (derived, zero-allocation)
    # ------------------------------------------------------------------

    @property
    def speed(self) -> float:
        """Magnitude of ECI velocity vector [m s⁻¹]."""
        return float(np.linalg.norm(self.velocity_eci))

    @property
    def altitude_eci(self) -> float:
        """
        Distance from ECI origin (planetary barycenter) [m].

        Note: this is the ECI radius, NOT geodetic altitude above the
        surface. Use the frame transforms module to obtain geodetic altitude.
        """
        return float(np.linalg.norm(self.position_eci))

    @property
    def kinetic_energy(self) -> float:
        """
        Translational kinetic energy in the ECI frame [J].

        E_k = ½ m v²

        Does not include rotational kinetic energy (requires inertia tensor
        from the vehicle component graph — not stored in the base state).
        """
        v_sq = float(np.dot(self.velocity_eci, self.velocity_eci))
        return 0.5 * self.mass * v_sq

    @property
    def angular_momentum_body(self) -> np.ndarray:
        """
        Angular momentum direction in Body Frame [rad s⁻¹].

        Returns omega_body directly. The full angular momentum vector
        H = I · ω requires the inertia tensor I, which lives in the
        mass model — this property returns the ω component only.
        """
        return self.omega_body.copy()

    # ------------------------------------------------------------------
    # Flat state array interface (for RK4 integrator)
    # ------------------------------------------------------------------

    def to_flat(self) -> np.ndarray:
        """
        Serialise to a flat 13-element float64 array for ODE integration.

        Layout: [pos(3), vel(3), quat(4), omega(3)]

        The integrator operates on this flat representation; the result is
        deserialised back to a VehicleState via ``VehicleState.from_flat``.
        """
        return np.concatenate([
            self.position_eci,
            self.velocity_eci,
            self.quaternion,
            self.omega_body,
        ], dtype=np.float64)

    @staticmethod
    def from_flat(
        flat: np.ndarray,
        mass: float,
        time: float,
        normalize_quaternion: bool = True,
    ) -> "VehicleState":
        """
        Deserialise from a flat 13-element array produced by ``to_flat``.

        Parameters
        ----------
        flat : ndarray, shape (13,)
            Flat state vector in the canonical layout.
        mass : float
            Vehicle mass at this instant [kg]. Must be supplied externally
            because mass is not part of the ODE state (it changes via the
            propulsion model, not by integration).
        time : float
            Mission elapsed time [s].
        normalize_quaternion : bool
            If True (default), re-normalise the quaternion before
            constructing the state. Always True during integration to
            prevent norm drift across RK4 steps.

        Returns
        -------
        VehicleState
        """
        if flat.shape != (13,):
            raise ValueError(
                f"from_flat expects shape (13,), got {flat.shape}"
            )

        pos  = flat[0:3].astype(np.float64)
        vel  = flat[3:6].astype(np.float64)
        quat = flat[6:10].astype(np.float64)
        omg  = flat[10:13].astype(np.float64)

        if normalize_quaternion:
            q_norm = np.linalg.norm(quat)
            if q_norm < 1.0e-15:
                raise ValueError(
                    "Quaternion norm collapsed to zero — state is degenerate."
                )
            quat = quat / q_norm

        return VehicleState(
            position_eci=pos,
            velocity_eci=vel,
            quaternion=quat,
            omega_body=omg,
            mass=mass,
            time=time,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def with_time(self, new_time: float) -> "VehicleState":
        """Return a new VehicleState identical to this one but at ``new_time``."""
        return VehicleState(
            position_eci=self.position_eci.copy(),
            velocity_eci=self.velocity_eci.copy(),
            quaternion=self.quaternion.copy(),
            omega_body=self.omega_body.copy(),
            mass=self.mass,
            time=new_time,
        )

    def __repr__(self) -> str:
        r = float(self.altitude_eci)
        v = self.speed
        return (
            f"VehicleState("
            f"t={self.time:.3f}s, "
            f"|r|={r:.1f}m, "
            f"|v|={v:.3f}m/s, "
            f"m={self.mass:.3f}kg)"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_state(
    *,
    position_eci: "ArrayLike",
    velocity_eci: "ArrayLike",
    quaternion:   "ArrayLike",
    omega_body:   "ArrayLike",
    mass:         float,
    time:         float = 0.0,
) -> VehicleState:
    """
    Convenience constructor that coerces inputs to float64 arrays.

    All positional arrays may be Python lists, tuples, or ndarrays;
    they are coerced to float64 and validated on construction.

    Example
    -------
    >>> from nova.core.state_vector import make_state
    >>> import numpy as np
    >>> state = make_state(
    ...     position_eci=[6_571_000.0, 0.0, 0.0],   # 200 km LEO
    ...     velocity_eci=[0.0, 7_784.0, 0.0],
    ...     quaternion=[1.0, 0.0, 0.0, 0.0],         # identity
    ...     omega_body=[0.0, 0.0, 0.0],
    ...     mass=1000.0,
    ... )
    """
    return VehicleState(
        position_eci=np.asarray(position_eci, dtype=np.float64),
        velocity_eci=np.asarray(velocity_eci, dtype=np.float64),
        quaternion=np.asarray(quaternion,   dtype=np.float64),
        omega_body=np.asarray(omega_body,   dtype=np.float64),
        mass=float(mass),
        time=float(time),
    )


def identity_state(
    position_eci: "ArrayLike" = (6_571_000.0, 0.0, 0.0),
    velocity_eci: "ArrayLike" = (0.0, 7_784.0, 0.0),
    mass: float = 1000.0,
) -> VehicleState:
    """
    Return a zero-rotation, zero-angular-velocity state at the given
    position and velocity. Useful for unit tests that only need to exercise
    one dimension of the state.

    Default position is a nominal 200 km LEO pass-through (not a valid
    circular orbit at these numbers — use OrbitalSolver for precise ICs).
    """
    return make_state(
        position_eci=position_eci,
        velocity_eci=velocity_eci,
        quaternion=[1.0, 0.0, 0.0, 0.0],
        omega_body=[0.0, 0.0, 0.0],
        mass=mass,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_vector(arr: object, shape: tuple, name: str) -> None:
    """Raise TypeError/ValueError if ``arr`` is not a float64 array of ``shape``."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"VehicleState.{name} must be a numpy.ndarray, "
            f"got {type(arr).__name__}"
        )
    if arr.dtype != np.float64:
        raise TypeError(
            f"VehicleState.{name} must have dtype float64, "
            f"got {arr.dtype}"
        )
    if arr.shape != shape:
        raise ValueError(
            f"VehicleState.{name} must have shape {shape}, "
            f"got {arr.shape}"
        )
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_state_vector.py
===============================
Unit tests for nova.core.state_vector.

Tests verify:
  1. Valid construction with all dtype and shape constraints.
  2. Validation rejection of malformed inputs (wrong dtype, shape, norm).
  3. Flat serialisation roundtrip: to_flat() → from_flat() → identity.
  4. Immutability: frozen dataclass raises on attempted mutation.
  5. Derived properties (speed, altitude, kinetic energy).
  6. make_state and identity_state factories.
"""

import math
import pytest
import numpy as np

from nova.core.state_vector import VehicleState, make_state, identity_state
from nova.core.constants import QUATERNION_NORM_TOL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def leo_state() -> VehicleState:
    """A nominal 200 km circular LEO state (not analytically exact — for testing)."""
    return make_state(
        position_eci=[6_571_000.0, 0.0, 0.0],
        velocity_eci=[0.0, 7_784.0, 0.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        omega_body=[0.0, 0.0, 0.0],
        mass=500.0,
        time=0.0,
    )


@pytest.fixture
def tumbling_state() -> VehicleState:
    """State with non-trivial quaternion and angular rate."""
    angle = math.pi / 4.0   # 45° rotation about Z
    q = np.array([
        math.cos(angle / 2.0),
        0.0,
        0.0,
        math.sin(angle / 2.0),
    ], dtype=np.float64)
    return make_state(
        position_eci=[7_000_000.0, 100_000.0, 50_000.0],
        velocity_eci=[100.0, 7_500.0, 200.0],
        quaternion=q,
        omega_body=[0.1, 0.05, 0.02],
        mass=1200.0,
        time=300.0,
    )


# ---------------------------------------------------------------------------
# 1. Valid construction
# ---------------------------------------------------------------------------

class TestValidConstruction:

    def test_basic_construction(self, leo_state):
        assert leo_state.mass == 500.0
        assert leo_state.time == 0.0
        assert leo_state.position_eci.shape == (3,)
        assert leo_state.velocity_eci.shape == (3,)
        assert leo_state.quaternion.shape == (4,)
        assert leo_state.omega_body.shape == (3,)

    def test_all_arrays_are_float64(self, leo_state):
        assert leo_state.position_eci.dtype == np.float64
        assert leo_state.velocity_eci.dtype == np.float64
        assert leo_state.quaternion.dtype == np.float64
        assert leo_state.omega_body.dtype == np.float64

    def test_identity_quaternion_norm(self, leo_state):
        norm = np.linalg.norm(leo_state.quaternion)
        assert abs(norm - 1.0) < QUATERNION_NORM_TOL

    def test_non_trivial_quaternion_norm(self, tumbling_state):
        norm = np.linalg.norm(tumbling_state.quaternion)
        assert abs(norm - 1.0) < QUATERNION_NORM_TOL

    def test_nonzero_time(self, tumbling_state):
        assert tumbling_state.time == 300.0

    def test_make_state_coerces_lists(self):
        """make_state must accept Python lists and coerce to float64."""
        s = make_state(
            position_eci=[1.0, 2.0, 3.0],
            velocity_eci=[4.0, 5.0, 6.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            omega_body=[0.0, 0.0, 0.0],
            mass=100.0,
        )
        assert s.position_eci.dtype == np.float64

    def test_identity_state_factory(self):
        s = identity_state()
        assert np.allclose(s.quaternion, [1.0, 0.0, 0.0, 0.0])
        assert np.allclose(s.omega_body, [0.0, 0.0, 0.0])
        assert s.time == 0.0


# ---------------------------------------------------------------------------
# 2. Validation — malformed inputs must raise
# ---------------------------------------------------------------------------

class TestValidationRejects:

    def _base_kwargs(self) -> dict:
        return dict(
            position_eci=np.zeros(3, dtype=np.float64),
            velocity_eci=np.zeros(3, dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            omega_body=np.zeros(3, dtype=np.float64),
            mass=100.0,
            time=0.0,
        )

    def test_wrong_dtype_raises(self):
        kw = self._base_kwargs()
        kw["position_eci"] = np.zeros(3, dtype=np.float32)
        with pytest.raises(TypeError, match="float64"):
            VehicleState(**kw)

    def test_wrong_shape_raises(self):
        kw = self._base_kwargs()
        kw["velocity_eci"] = np.zeros(4, dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            VehicleState(**kw)

    def test_wrong_quaternion_shape_raises(self):
        kw = self._base_kwargs()
        kw["quaternion"] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            VehicleState(**kw)

    def test_non_unit_quaternion_raises(self):
        kw = self._base_kwargs()
        kw["quaternion"] = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="unit quaternion"):
            VehicleState(**kw)

    def test_near_zero_quaternion_raises(self):
        kw = self._base_kwargs()
        kw["quaternion"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="unit quaternion"):
            VehicleState(**kw)

    def test_negative_mass_raises(self):
        kw = self._base_kwargs()
        kw["mass"] = -1.0
        with pytest.raises(ValueError, match="mass"):
            VehicleState(**kw)

    def test_zero_mass_raises(self):
        kw = self._base_kwargs()
        kw["mass"] = 0.0
        with pytest.raises(ValueError, match="mass"):
            VehicleState(**kw)

    def test_negative_time_raises(self):
        kw = self._base_kwargs()
        kw["time"] = -0.001
        with pytest.raises(ValueError, match="time"):
            VehicleState(**kw)

    def test_list_input_to_vehiclestate_raises(self):
        """VehicleState itself requires ndarray; use make_state for list coercion."""
        kw = self._base_kwargs()
        kw["position_eci"] = [0.0, 0.0, 0.0]   # Python list, not ndarray
        with pytest.raises(TypeError, match="numpy.ndarray"):
            VehicleState(**kw)


# ---------------------------------------------------------------------------
# 3. Immutability
# ---------------------------------------------------------------------------

class TestImmutability:

    def test_cannot_set_field(self, leo_state):
        with pytest.raises(Exception):   # FrozenInstanceError
            leo_state.mass = 999.0

    def test_cannot_set_array_reference(self, leo_state):
        with pytest.raises(Exception):
            leo_state.position_eci = np.zeros(3, dtype=np.float64)

    def test_array_in_place_mutation_does_not_affect_copy(self, leo_state):
        """
        NumPy arrays inside a frozen dataclass are still mutable in-place
        (Python cannot prevent __setitem__ on ndarray). This test documents
        that to_flat() and from_flat() produce independent copies, not views.
        """
        flat = leo_state.to_flat()
        flat[0] = -999_999.0   # mutate the flat copy
        # Original state should be unchanged
        assert leo_state.position_eci[0] == 6_571_000.0


# ---------------------------------------------------------------------------
# 4. Flat serialisation roundtrip
# ---------------------------------------------------------------------------

class TestFlatSerialisation:

    def test_to_flat_shape(self, leo_state):
        flat = leo_state.to_flat()
        assert flat.shape == (13,)
        assert flat.dtype == np.float64

    def test_to_flat_layout(self, leo_state):
        flat = leo_state.to_flat()
        assert np.allclose(flat[0:3],  leo_state.position_eci)
        assert np.allclose(flat[3:6],  leo_state.velocity_eci)
        assert np.allclose(flat[6:10], leo_state.quaternion)
        assert np.allclose(flat[10:13],leo_state.omega_body)

    def test_roundtrip_identity(self, leo_state):
        flat      = leo_state.to_flat()
        recovered = VehicleState.from_flat(flat, mass=leo_state.mass, time=leo_state.time)
        assert np.allclose(recovered.position_eci, leo_state.position_eci, atol=1e-12)
        assert np.allclose(recovered.velocity_eci, leo_state.velocity_eci, atol=1e-12)
        assert np.allclose(recovered.quaternion,   leo_state.quaternion,   atol=1e-12)
        assert np.allclose(recovered.omega_body,   leo_state.omega_body,   atol=1e-12)
        assert recovered.mass == leo_state.mass
        assert recovered.time == leo_state.time

    def test_roundtrip_tumbling(self, tumbling_state):
        flat      = tumbling_state.to_flat()
        recovered = VehicleState.from_flat(
            flat, mass=tumbling_state.mass, time=tumbling_state.time
        )
        assert np.allclose(recovered.quaternion, tumbling_state.quaternion, atol=1e-12)
        assert np.allclose(recovered.omega_body, tumbling_state.omega_body, atol=1e-12)

    def test_from_flat_normalises_quaternion(self):
        """from_flat should renormalise a slightly-drifted quaternion."""
        flat = np.zeros(13, dtype=np.float64)
        flat[0:3]  = [6_571_000.0, 0.0, 0.0]
        flat[3:6]  = [0.0, 7_784.0, 0.0]
        flat[6:10] = [1.001, 0.0, 0.0, 0.0]   # norm ≠ 1 exactly
        flat[10:13]= [0.0, 0.0, 0.0]
        state = VehicleState.from_flat(flat, mass=500.0, time=0.0, normalize_quaternion=True)
        assert abs(np.linalg.norm(state.quaternion) - 1.0) < QUATERNION_NORM_TOL

    def test_from_flat_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            VehicleState.from_flat(np.zeros(12, dtype=np.float64), mass=100.0, time=0.0)

    def test_from_flat_collapsed_quaternion_raises(self):
        flat = np.zeros(13, dtype=np.float64)
        flat[6:10] = [0.0, 0.0, 0.0, 0.0]   # zero quaternion → undefined orientation
        flat[3:6]  = [0.0, 7_784.0, 0.0]
        with pytest.raises(ValueError, match="norm"):
            VehicleState.from_flat(flat, mass=500.0, time=0.0, normalize_quaternion=True)


# ---------------------------------------------------------------------------
# 5. Derived properties
# ---------------------------------------------------------------------------

class TestDerivedProperties:

    def test_speed(self, leo_state):
        """‖v‖ = 7784 m/s for the LEO fixture."""
        assert abs(leo_state.speed - 7_784.0) < 1.0e-8

    def test_altitude_eci(self, leo_state):
        """‖r‖ = 6 571 000 m (ECI radius, not geodetic altitude)."""
        assert abs(leo_state.altitude_eci - 6_571_000.0) < 1.0e-4

    def test_kinetic_energy(self, leo_state):
        """KE = ½ m v² = ½ × 500 × 7784² ≈ 1.513 × 10¹⁰ J."""
        ke_expected = 0.5 * 500.0 * 7_784.0**2
        assert abs(leo_state.kinetic_energy - ke_expected) < 1.0

    def test_with_time(self, leo_state):
        new_state = leo_state.with_time(123.456)
        assert new_state.time == 123.456
        assert np.allclose(new_state.position_eci, leo_state.position_eci)
        assert new_state is not leo_state   # must be a new object

    def test_repr_contains_key_fields(self, leo_state):
        r = repr(leo_state)
        assert "VehicleState" in r
        assert "t=0.000s" in r
