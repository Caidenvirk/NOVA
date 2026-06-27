"""
nova.vehicle.mass_model
=======================
Dynamic mass and inertia model for Project NOVA.

Architecture role — Pipeline Stage 9 (Component Updates) + Stage 3 input
-------------------------------------------------------------------------
Tracks the vehicle's total mass, centre of mass (CoM) position in the Body
frame, and inertia tensor as propellant is depleted and structural components
change state.

The inertia tensor is required by the RK4 integrator (via ``build_deriv_fn``)
and by the torque accumulator (gravity gradient). The CoM position is required
by the force/torque accumulators to compute moment arms correctly.

Physics
-------
For N point-mass or solid-body components, each with mass mᵢ and CoM
position rᵢ in a shared reference frame:

  m_total = Σ mᵢ

  r_CoM   = (1/m_total) · Σ mᵢ rᵢ         (in Body Frame)

  I_total = Σ [Iᵢ_own + mᵢ · D(rᵢ − r_CoM)]

where D(r) is the parallel-axis (Huygens-Steiner) displacement tensor:
  D(r)ₙₙ = |r|² − rₙ²    (diagonal)
  D(r)ₘₙ = −rₘ rₙ        (off-diagonal)

This gives the full symmetric 3×3 inertia tensor with parallel-axis
correction applied component by component.

Design notes
------------
* MassComponent is a lightweight dataclass — no external imports needed.
* MassModel is rebuilt from scratch each tick by the pipeline (Component
  Updates, Stage 9) after applying propellant depletion. This avoids
  accumulated floating-point state drift.
* All positions are in the Body Frame, which is CoM-centred by definition.
  Therefore r_CoM should be near zero in a well-designed vehicle; any
  residual offset is a CoM shift due to asymmetric propellant depletion.

References
----------
- Goldstein, "Classical Mechanics", 3rd ed., §5.3
- Kane & Levinson, "Dynamics: Theory and Applications", §4.4
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Individual mass component
# ---------------------------------------------------------------------------

@dataclass
class MassComponent:
    """
    A single rigid-body mass component contributing to the vehicle mass model.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"LOX tank"``, ``"payload fairing"``).
    mass : float
        Current mass of this component [kg]. Must be ≥ 0.
    position_body : ndarray, shape (3,), float64
        Position of this component's own CoM in the Body Frame [m].
        Measured relative to the vehicle's nominal Body Frame origin
        (which coincides with the full-stack CoM at a reference configuration).
    inertia_own : ndarray, shape (3, 3), float64
        Inertia tensor of this component about its OWN CoM [kg·m²].
        Must be symmetric positive semi-definite. For a point mass, use zeros.
    is_active : bool
        If False, component has been jettisoned/separated — excluded from
        mass totals and inertia computation.
    """
    name:          str
    mass:          float
    position_body: np.ndarray   # (3,) float64 [m]
    inertia_own:   np.ndarray   # (3,3) float64 [kg·m²]
    is_active:     bool = True

    def __post_init__(self) -> None:
        if self.mass < 0.0:
            raise ValueError(f"MassComponent '{self.name}': mass must be ≥ 0, got {self.mass!r}")
        self.position_body = np.asarray(self.position_body, dtype=np.float64)
        self.inertia_own   = np.asarray(self.inertia_own,   dtype=np.float64)
        if self.position_body.shape != (3,):
            raise ValueError(
                f"MassComponent '{self.name}': position_body must be shape (3,), "
                f"got {self.position_body.shape}"
            )
        if self.inertia_own.shape != (3, 3):
            raise ValueError(
                f"MassComponent '{self.name}': inertia_own must be shape (3,3), "
                f"got {self.inertia_own.shape}"
            )


# ---------------------------------------------------------------------------
# Parallel-axis (Steiner) displacement tensor
# ---------------------------------------------------------------------------

def steiner_tensor(r: np.ndarray) -> np.ndarray:
    """
    Compute the parallel-axis displacement tensor for offset r.

    D(r)ᵢⱼ = |r|²·δᵢⱼ − rᵢ·rⱼ

    This is added to a component's own inertia tensor to shift it from
    the component's CoM to the vehicle's full CoM:

        I_about_CoM = I_own + m · D(r)

    where r = component_position − vehicle_CoM_position.

    Parameters
    ----------
    r : ndarray, shape (3,)
        Offset vector [m] from original CoM to new reference point.

    Returns
    -------
    ndarray, shape (3,3), float64
    """
    r = np.asarray(r, dtype=np.float64)
    r_sq = float(np.dot(r, r))
    D = r_sq * np.eye(3, dtype=np.float64) - np.outer(r, r)
    return D


# ---------------------------------------------------------------------------
# Full vehicle mass model
# ---------------------------------------------------------------------------

@dataclass
class MassModel:
    """
    Computed mass properties of the complete vehicle assembly.

    Built by ``compute_mass_properties`` from a list of MassComponents.
    This object is frozen after construction.

    Attributes
    ----------
    total_mass : float
        Sum of all active component masses [kg].
    com_body : ndarray, shape (3,), float64
        Centre of mass position in the Body Frame [m].
        For a well-trimmed vehicle this should be near [0, 0, 0].
    inertia_body : ndarray, shape (3,3), float64
        Full inertia tensor about the vehicle CoM in the Body Frame [kg·m²].
        Symmetric positive semi-definite.
    inertia_inv : ndarray, shape (3,3), float64
        Inverse of the inertia tensor (pre-computed for the integrator).
    """
    total_mass:   float
    com_body:     np.ndarray   # (3,) float64 [m]
    inertia_body: np.ndarray   # (3,3) float64 [kg·m²]
    inertia_inv:  np.ndarray   # (3,3) float64 [(kg·m²)⁻¹]

    @property
    def Ixx(self) -> float:
        """Roll inertia [kg·m²]."""
        return float(self.inertia_body[0, 0])

    @property
    def Iyy(self) -> float:
        """Pitch inertia [kg·m²]."""
        return float(self.inertia_body[1, 1])

    @property
    def Izz(self) -> float:
        """Yaw inertia [kg·m²]."""
        return float(self.inertia_body[2, 2])

    @property
    def is_symmetric(self) -> bool:
        """True if inertia tensor is symmetric (as it always should be)."""
        return bool(np.allclose(self.inertia_body, self.inertia_body.T, atol=1.0e-6))


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def compute_mass_properties(
    components: List[MassComponent],
    enforce_positive_definite: bool = True,
) -> MassModel:
    """
    Compute total mass, CoM, and inertia tensor from a list of components.

    Parameters
    ----------
    components : list of MassComponent
        All vehicle components. Inactive (jettisoned) ones are skipped.
    enforce_positive_definite : bool
        If True (default), add a small regularisation (1e-6·I₃) to the
        inertia tensor to prevent singular I in degenerate cases (e.g.
        a single point mass with no extent). This keeps the integrator stable.

    Returns
    -------
    MassModel

    Raises
    ------
    ValueError
        If total active mass is zero or negative (unphysical state).
    """
    active = [c for c in components if c.is_active]

    if not active:
        raise ValueError("compute_mass_properties: no active components — total mass is zero.")

    # --- Total mass ---
    m_total = sum(c.mass for c in active)
    if m_total <= 0.0:
        raise ValueError(
            f"compute_mass_properties: total mass = {m_total:.6f} kg ≤ 0. "
            "All active components have zero mass?"
        )

    # --- Centre of mass (Body Frame) ---
    r_com = np.zeros(3, dtype=np.float64)
    for c in active:
        r_com += c.mass * c.position_body
    r_com /= m_total

    # --- Inertia tensor about vehicle CoM (parallel-axis theorem) ---
    I_total = np.zeros((3, 3), dtype=np.float64)
    for c in active:
        if c.mass <= 0.0:
            continue   # zero-mass component contributes nothing
        r_offset = c.position_body - r_com   # vector from vehicle CoM to component CoM
        D        = steiner_tensor(r_offset)
        I_total += c.inertia_own + c.mass * D

    # --- Symmetrise (eliminate floating-point asymmetry) ---
    I_total = 0.5 * (I_total + I_total.T)

    # --- Regularisation ---
    if enforce_positive_definite:
        I_total += 1.0e-6 * np.eye(3, dtype=np.float64)

    # --- Inertia inverse ---
    I_inv = np.linalg.inv(I_total)

    return MassModel(
        total_mass=m_total,
        com_body=r_com,
        inertia_body=I_total,
        inertia_inv=I_inv,
    )


# ---------------------------------------------------------------------------
# Convenience constructors for common component shapes
# ---------------------------------------------------------------------------

def point_mass(name: str, mass: float, position_body: "ArrayLike") -> MassComponent:
    """
    Create a point-mass component (zero own inertia).

    Parameters
    ----------
    name : str
    mass : float   [kg]
    position_body : array-like, shape (3,)   [m]
    """
    return MassComponent(
        name=name,
        mass=mass,
        position_body=np.asarray(position_body, dtype=np.float64),
        inertia_own=np.zeros((3, 3), dtype=np.float64),
    )


def solid_cylinder(
    name:          str,
    mass:          float,
    radius:        float,
    length:        float,
    position_body: "ArrayLike",
    axis:          int = 0,
) -> MassComponent:
    """
    Create a solid uniform cylinder component.

    Parameters
    ----------
    name : str
    mass : float    [kg]
    radius : float  [m]  Cylinder radius.
    length : float  [m]  Cylinder length along ``axis``.
    position_body : array-like, shape (3,)   CoM position in Body frame [m].
    axis : int
        Body-frame axis along which the cylinder is oriented.
        0 = X (longitudinal), 1 = Y, 2 = Z.

    Returns
    -------
    MassComponent

    Inertia
    -------
    Axial moment:       I_axial     = (1/2) m r²
    Transverse moments: I_transverse = (1/12) m (3r² + L²)
    """
    I_axial     = 0.5  * mass * radius**2
    I_transverse = (1.0 / 12.0) * mass * (3.0 * radius**2 + length**2)

    # Place axial inertia along the correct diagonal
    diag = [I_transverse, I_transverse, I_transverse]
    diag[axis] = I_axial
    I_own = np.diag(diag).astype(np.float64)

    return MassComponent(
        name=name,
        mass=mass,
        position_body=np.asarray(position_body, dtype=np.float64),
        inertia_own=I_own,
    )


def solid_sphere(
    name:          str,
    mass:          float,
    radius:        float,
    position_body: "ArrayLike",
) -> MassComponent:
    """
    Create a solid uniform sphere component.

    I = (2/5) m r²  (isotropic — same for all axes)
    """
    I_val = (2.0 / 5.0) * mass * radius**2
    I_own = I_val * np.eye(3, dtype=np.float64)
    return MassComponent(
        name=name,
        mass=mass,
        position_body=np.asarray(position_body, dtype=np.float64),
        inertia_own=I_own,
    )


def hollow_cylinder(
    name:          str,
    mass:          float,
    inner_radius:  float,
    outer_radius:  float,
    length:        float,
    position_body: "ArrayLike",
    axis:          int = 0,
) -> MassComponent:
    """
    Create a hollow cylinder (thin-walled tank) component.

    Axial:      I_axial     = (1/2) m (r_i² + r_o²)
    Transverse: I_transverse = (1/12) m (3(r_i² + r_o²) + L²)
    """
    r_sq_sum    = inner_radius**2 + outer_radius**2
    I_axial     = 0.5 * mass * r_sq_sum
    I_transverse = (1.0 / 12.0) * mass * (3.0 * r_sq_sum + length**2)

    diag = [I_transverse, I_transverse, I_transverse]
    diag[axis] = I_axial
    I_own = np.diag(diag).astype(np.float64)

    return MassComponent(
        name=name,
        mass=mass,
        position_body=np.asarray(position_body, dtype=np.float64),
        inertia_own=I_own,
    )
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_mass_model.py
==============================
Unit tests for nova.vehicle.mass_model.

Tests verify:
  1. Single point-mass: CoM = component position, I = 0 (+ regularisation).
  2. Two equal masses symmetric about origin: CoM = [0,0,0].
  3. Parallel-axis theorem: steiner_tensor produces correct D(r) matrix.
  4. Inertia tensor is symmetric and positive-definite.
  5. Total mass = sum of active component masses.
  6. Inactive components excluded from mass totals.
  7. solid_cylinder inertia formula: I_axial = ½mr², I_trans = (1/12)m(3r²+L²).
  8. solid_sphere inertia formula: I = (2/5)mr².
  9. hollow_cylinder inertia formula.
  10. Zero active components raises ValueError.
  11. Negative mass raises ValueError.
  12. MassModel.Ixx/Iyy/Izz diagonal properties.
"""

import math
import pytest
import numpy as np

from nova.vehicle.mass_model import (
    MassComponent,
    MassModel,
    compute_mass_properties,
    steiner_tensor,
    point_mass,
    solid_cylinder,
    solid_sphere,
    hollow_cylinder,
)


# ---------------------------------------------------------------------------
# 1. steiner_tensor
# ---------------------------------------------------------------------------

class TestSteinerTensor:

    def test_zero_offset_gives_zero_tensor(self):
        r = np.zeros(3, dtype=np.float64)
        D = steiner_tensor(r)
        assert np.allclose(D, np.zeros((3, 3)), atol=1.0e-14)

    def test_unit_x_offset(self):
        """
        r = [1, 0, 0]:
        D = |r|²·I − r⊗r = diag(0, 1, 1)
        """
        r = np.array([1.0, 0.0, 0.0])
        D = steiner_tensor(r)
        expected = np.diag([0.0, 1.0, 1.0])
        assert np.allclose(D, expected, atol=1.0e-14)

    def test_unit_y_offset(self):
        """r = [0, 1, 0] → D = diag(1, 0, 1)."""
        D = steiner_tensor(np.array([0.0, 1.0, 0.0]))
        assert np.allclose(D, np.diag([1.0, 0.0, 1.0]), atol=1.0e-14)

    def test_unit_z_offset(self):
        """r = [0, 0, 1] → D = diag(1, 1, 0)."""
        D = steiner_tensor(np.array([0.0, 0.0, 1.0]))
        assert np.allclose(D, np.diag([1.0, 1.0, 0.0]), atol=1.0e-14)

    def test_off_diagonal_terms(self):
        """r = [1, 1, 0]: D[0,1] = −r_x·r_y = −1."""
        r = np.array([1.0, 1.0, 0.0])
        D = steiner_tensor(r)
        assert abs(D[0, 1] - (-1.0)) < 1.0e-14
        assert abs(D[1, 0] - (-1.0)) < 1.0e-14

    def test_symmetry(self):
        r = np.array([2.0, -3.0, 1.5])
        D = steiner_tensor(r)
        assert np.allclose(D, D.T, atol=1.0e-14)

    def test_trace_equals_2_r_squared(self):
        """tr(D) = 2|r|² for any r."""
        r = np.array([3.0, 4.0, 0.0])
        D = steiner_tensor(r)
        assert abs(np.trace(D) - 2.0 * float(np.dot(r, r))) < 1.0e-10


# ---------------------------------------------------------------------------
# 2. Single point-mass
# ---------------------------------------------------------------------------

class TestSinglePointMass:

    @pytest.fixture
    def single_mass(self):
        return [point_mass("body", 100.0, [5.0, 0.0, 0.0])]

    def test_total_mass(self, single_mass):
        model = compute_mass_properties(single_mass)
        assert abs(model.total_mass - 100.0) < 1.0e-10

    def test_com_equals_component_position(self, single_mass):
        model = compute_mass_properties(single_mass)
        assert np.allclose(model.com_body, [5.0, 0.0, 0.0], atol=1.0e-10)

    def test_inertia_symmetric(self, single_mass):
        model = compute_mass_properties(single_mass)
        assert model.is_symmetric

    def test_inertia_positive_definite(self, single_mass):
        """All eigenvalues of I must be positive (PD after regularisation)."""
        model  = compute_mass_properties(single_mass)
        eigvals = np.linalg.eigvalsh(model.inertia_body)
        assert all(e > 0.0 for e in eigvals), f"Non-positive eigenvalues: {eigvals}"

    def test_inertia_inv_is_inverse(self, single_mass):
        """I · I⁻¹ = I₃ within floating-point tolerance."""
        model = compute_mass_properties(single_mass)
        product = model.inertia_body @ model.inertia_inv
        assert np.allclose(product, np.eye(3), atol=1.0e-10)


# ---------------------------------------------------------------------------
# 3. Two symmetric masses — CoM at origin
# ---------------------------------------------------------------------------

class TestSymmetricMasses:

    @pytest.fixture
    def symmetric_pair(self):
        return [
            point_mass("left",  500.0, [-3.0, 0.0, 0.0]),
            point_mass("right", 500.0, [ 3.0, 0.0, 0.0]),
        ]

    def test_total_mass(self, symmetric_pair):
        model = compute_mass_properties(symmetric_pair)
        assert abs(model.total_mass - 1000.0) < 1.0e-10

    def test_com_at_origin(self, symmetric_pair):
        model = compute_mass_properties(symmetric_pair)
        assert np.allclose(model.com_body, [0.0, 0.0, 0.0], atol=1.0e-10)

    def test_ixx_zero_for_point_masses_on_x_axis(self, symmetric_pair):
        """
        Two point masses on X-axis: I_xx = Σ mᵢ(yᵢ²+zᵢ²) = 0.
        (regularisation adds 1e-6.)
        """
        model = compute_mass_properties(symmetric_pair)
        assert abs(model.Ixx) < 1.0e-4   # only regularisation term

    def test_iyy_correct(self, symmetric_pair):
        """I_yy = Σ mᵢ(xᵢ²+zᵢ²) = 2 × 500 × 9 = 9000 kg·m²."""
        model = compute_mass_properties(symmetric_pair)
        assert abs(model.Iyy - 9000.0) < 0.01

    def test_izz_correct(self, symmetric_pair):
        """I_zz = Σ mᵢ(xᵢ²+yᵢ²) = 9000 kg·m²."""
        model = compute_mass_properties(symmetric_pair)
        assert abs(model.Izz - 9000.0) < 0.01


# ---------------------------------------------------------------------------
# 4. Asymmetric masses — known CoM
# ---------------------------------------------------------------------------

class TestAsymmetricMasses:

    @pytest.fixture
    def asym_pair(self):
        # m1=100 at x=0, m2=300 at x=4 → CoM at x=3
        return [
            point_mass("light", 100.0, [0.0, 0.0, 0.0]),
            point_mass("heavy", 300.0, [4.0, 0.0, 0.0]),
        ]

    def test_com_x(self, asym_pair):
        model = compute_mass_properties(asym_pair)
        x_com = (100.0 * 0.0 + 300.0 * 4.0) / 400.0   # = 3.0
        assert abs(model.com_body[0] - x_com) < 1.0e-10

    def test_total_mass(self, asym_pair):
        model = compute_mass_properties(asym_pair)
        assert abs(model.total_mass - 400.0) < 1.0e-10

    def test_parallel_axis_applied(self, asym_pair):
        """
        After CoM shift, I_yy = Σ mᵢ·(xᵢ − x_com)²:
          m1·(0−3)² + m2·(4−3)² = 100·9 + 300·1 = 900+300 = 1200 kg·m²
        """
        model  = compute_mass_properties(asym_pair)
        I_yy_expected = 100.0 * 3.0**2 + 300.0 * 1.0**2
        assert abs(model.Iyy - I_yy_expected) < 0.01, \
            f"I_yy={model.Iyy:.4f}, expected {I_yy_expected}"


# ---------------------------------------------------------------------------
# 5. Inactive component excluded
# ---------------------------------------------------------------------------

class TestInactiveComponent:

    def test_inactive_excluded_from_mass(self):
        c1 = point_mass("active",   500.0, [0.0, 0.0, 0.0])
        c2 = point_mass("inactive", 200.0, [10.0, 0.0, 0.0])
        c2.is_active = False
        model = compute_mass_properties([c1, c2])
        assert abs(model.total_mass - 500.0) < 1.0e-10

    def test_inactive_excluded_from_com(self):
        c1 = point_mass("active",   500.0, [0.0, 0.0, 0.0])
        c2 = point_mass("inactive", 200.0, [100.0, 0.0, 0.0])
        c2.is_active = False
        model = compute_mass_properties([c1, c2])
        assert np.allclose(model.com_body, [0.0, 0.0, 0.0], atol=1.0e-10)

    def test_all_inactive_raises(self):
        c = point_mass("x", 100.0, [0.0, 0.0, 0.0])
        c.is_active = False
        with pytest.raises(ValueError, match="no active"):
            compute_mass_properties([c])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="no active"):
            compute_mass_properties([])


# ---------------------------------------------------------------------------
# 6. MassComponent validation
# ---------------------------------------------------------------------------

class TestMassComponentValidation:

    def test_negative_mass_raises(self):
        with pytest.raises(ValueError, match="mass"):
            MassComponent(
                name="bad",
                mass=-1.0,
                position_body=np.zeros(3, dtype=np.float64),
                inertia_own=np.zeros((3, 3), dtype=np.float64),
            )

    def test_wrong_position_shape_raises(self):
        with pytest.raises(ValueError, match="position_body"):
            MassComponent(
                name="bad",
                mass=10.0,
                position_body=np.zeros(4, dtype=np.float64),
                inertia_own=np.zeros((3, 3), dtype=np.float64),
            )

    def test_wrong_inertia_shape_raises(self):
        with pytest.raises(ValueError, match="inertia_own"):
            MassComponent(
                name="bad",
                mass=10.0,
                position_body=np.zeros(3, dtype=np.float64),
                inertia_own=np.zeros((4, 4), dtype=np.float64),
            )

    def test_zero_mass_allowed(self):
        """Zero-mass components are permitted (e.g. empty tanks)."""
        c = point_mass("empty_tank", 0.0, [0.0, 0.0, 0.0])
        assert c.mass == 0.0


# ---------------------------------------------------------------------------
# 7. solid_cylinder inertia
# ---------------------------------------------------------------------------

class TestSolidCylinder:

    def test_axial_inertia(self):
        """I_axial = (1/2) m r²."""
        m, r, L = 100.0, 2.0, 10.0
        cyl = solid_cylinder("cyl", m, r, L, [0.0, 0.0, 0.0], axis=0)
        I_axial_expected = 0.5 * m * r**2
        assert abs(cyl.inertia_own[0, 0] - I_axial_expected) < 1.0e-8

    def test_transverse_inertia(self):
        """I_transverse = (1/12) m (3r² + L²)."""
        m, r, L = 100.0, 2.0, 10.0
        cyl = solid_cylinder("cyl", m, r, L, [0.0, 0.0, 0.0], axis=0)
        I_trans_expected = (1.0/12.0) * m * (3.0*r**2 + L**2)
        assert abs(cyl.inertia_own[1, 1] - I_trans_expected) < 1.0e-8
        assert abs(cyl.inertia_own[2, 2] - I_trans_expected) < 1.0e-8

    def test_symmetric_inertia(self):
        cyl = solid_cylinder("cyl", 100.0, 1.5, 8.0, [0.0, 0.0, 0.0])
        assert np.allclose(cyl.inertia_own, cyl.inertia_own.T, atol=1.0e-14)

    def test_y_axis_orientation(self):
        """axis=1: axial inertia is on [1,1] diagonal."""
        m, r, L = 50.0, 1.0, 5.0
        cyl = solid_cylinder("cyl", m, r, L, [0.0, 0.0, 0.0], axis=1)
        assert abs(cyl.inertia_own[1, 1] - 0.5*m*r**2) < 1.0e-8


# ---------------------------------------------------------------------------
# 8. solid_sphere inertia
# ---------------------------------------------------------------------------

class TestSolidSphere:

    def test_isotropic_inertia(self):
        """I = (2/5) m r² for all three diagonal entries."""
        m, r = 200.0, 0.5
        sph  = solid_sphere("sphere", m, r, [0.0, 0.0, 0.0])
        I_expected = (2.0/5.0) * m * r**2
        for i in range(3):
            assert abs(sph.inertia_own[i, i] - I_expected) < 1.0e-10

    def test_off_diagonal_zero(self):
        sph = solid_sphere("sphere", 100.0, 1.0, [0.0, 0.0, 0.0])
        offdiag = sph.inertia_own.copy()
        np.fill_diagonal(offdiag, 0.0)
        assert np.allclose(offdiag, 0.0, atol=1.0e-14)


# ---------------------------------------------------------------------------
# 9. hollow_cylinder inertia
# ---------------------------------------------------------------------------

class TestHollowCylinder:

    def test_axial_inertia(self):
        """I_axial = (1/2) m (r_i² + r_o²)."""
        m, ri, ro, L = 80.0, 1.0, 1.5, 6.0
        hc = hollow_cylinder("tank", m, ri, ro, L, [0.0, 0.0, 0.0], axis=0)
        I_axial_expected = 0.5 * m * (ri**2 + ro**2)
        assert abs(hc.inertia_own[0, 0] - I_axial_expected) < 1.0e-8

    def test_transverse_inertia(self):
        """I_transverse = (1/12) m (3(r_i²+r_o²) + L²)."""
        m, ri, ro, L = 80.0, 1.0, 1.5, 6.0
        hc = hollow_cylinder("tank", m, ri, ro, L, [0.0, 0.0, 0.0], axis=0)
        I_trans = (1.0/12.0) * m * (3.0*(ri**2+ro**2) + L**2)
        assert abs(hc.inertia_own[1, 1] - I_trans) < 1.0e-8

    def test_hollow_greater_than_solid_axial(self):
        """Hollow cylinder has higher axial inertia than solid at same mass."""
        m, ri, ro, L = 80.0, 1.0, 1.5, 6.0
        hc  = hollow_cylinder("h", m, ri, ro, L, [0.0,0.0,0.0])
        sol = solid_cylinder("s",  m, ro,    L, [0.0,0.0,0.0])
        assert hc.inertia_own[0, 0] > sol.inertia_own[0, 0]


# ---------------------------------------------------------------------------
# 10. Multi-component vehicle model
# ---------------------------------------------------------------------------

class TestMultiComponentVehicle:

    @pytest.fixture
    def rocket_stack(self):
        """Simple two-stage model: fuel tank + payload."""
        fuel  = solid_cylinder("fuel",    8000.0, 2.0, 20.0, [-10.0, 0.0, 0.0], axis=0)
        pay   = solid_cylinder("payload",  500.0, 1.0,  3.0, [ 12.0, 0.0, 0.0], axis=0)
        eng   = solid_sphere  ("engine",   300.0, 0.5,         [-22.0, 0.0, 0.0])
        return [fuel, pay, eng]

    def test_total_mass(self, rocket_stack):
        model = compute_mass_properties(rocket_stack)
        assert abs(model.total_mass - 8800.0) < 1.0e-6

    def test_inertia_symmetric(self, rocket_stack):
        model = compute_mass_properties(rocket_stack)
        assert model.is_symmetric

    def test_inertia_positive_definite(self, rocket_stack):
        model   = compute_mass_properties(rocket_stack)
        eigvals = np.linalg.eigvalsh(model.inertia_body)
        assert all(e > 0.0 for e in eigvals)

    def test_diagonal_properties(self, rocket_stack):
        model = compute_mass_properties(rocket_stack)
        assert abs(model.Ixx - model.inertia_body[0, 0]) < 1.0e-10
        assert abs(model.Iyy - model.inertia_body[1, 1]) < 1.0e-10
        assert abs(model.Izz - model.inertia_body[2, 2]) < 1.0e-10
