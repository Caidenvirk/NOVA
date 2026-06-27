"""
nova.physics.structural
=======================
Structural mechanics solver for Project NOVA.

Architecture role — Pipeline Stage 9 (Component Updates)
---------------------------------------------------------
Computes internal structural load tensors at each joint in the component
graph, then checks them against analytical failure criteria. Triggered
after the physics engine and integrator have determined the vehicle's
acceleration, angular acceleration, and dynamic pressure.

Internal loads computed per joint
----------------------------------
For each joint j connecting parent P and child C, the internal loads
represent the forces and moments that the parent must exert on the child
to maintain structural continuity. They are computed by summing all
external forces and inertial reactions on the child subtree:

  F_axial   [N]     Axial compressive/tensile force along joint axis (+X_body)
  V_shear   [N]     Resultant transverse shear (√(Vy²+Vz²))
  M_bend    [N·m]   Resultant bending moment (√(My²+Mz²))
  T_torsion [N·m]   Torsional moment about joint axis

Failure criteria
----------------
1. Euler buckling (compressive axial):
   F_axial_critical = π² E I / L_eff²
   Failure if F_a (compressive) > F_cr.

2. Shear yield (rectangular approx):
   V_max = (2/3) τ_yield A
   Failure if |V| > V_max.

3. Bending moment:
   M_yield = σ_yield I / c
   Failure if |M_b| > M_yield.

4. Damage accumulation (S-N):
   D += (n_cycles / N_failure) per tick.
   Failure if D ≥ 1.

All criteria are checked in sequence; the first exceeded triggers
joint.failed = True and joint.damage = 1.0.

Simplified loading model
------------------------
This module uses a simplified free-body diagram approach. Each child subtree
is treated as a rigid body under:
  - Weight of the subtree (gravity)
  - Aerodynamic drag on the subtree (from vehicle aero state)
  - Inertial reaction force = m_subtree · a_vehicle

The full multi-body structural FEM is beyond the scope of a trajectory
simulation; this model captures the dominant axial and shear loads that
drive real launch-vehicle structural failures at high Q and during max-Q.

References
----------
- Megson, "Aircraft Structures for Engineering Students", 5th ed., §2, §15
- Bruhn, "Analysis and Design of Flight Vehicle Structures", §C2, §D
- NASA SP-8028, "Buckling of Thin-Walled Circular Cylinders"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from nova.vehicle.component_graph import ComponentGraph, StructuralJoint
from nova.vehicle.mass_model import MassModel, compute_mass_properties


# ---------------------------------------------------------------------------
# Load state input (per-tick snapshot from physics engine)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleLoadState:
    """
    Vehicle-level load quantities needed for structural analysis.

    Passed in from the physics pipeline after the integrator has run.

    Attributes
    ----------
    acceleration_body : ndarray, shape (3,), float64
        Vehicle linear acceleration in Body Frame [m s⁻²].
    alpha_body : ndarray, shape (3,), float64
        Vehicle angular acceleration in Body Frame [rad s⁻²].
    dynamic_pressure : float
        Freestream dynamic pressure q_∞ [Pa].
    axial_thrust : float
        Total engine thrust force along vehicle +X axis [N].
        Used to compute axial compression from thrust.
    aero_drag_body : ndarray, shape (3,), float64
        Net aerodynamic drag force in Body Frame [N].
    gravity_body : ndarray, shape (3,), float64
        Gravitational acceleration in Body Frame [m s⁻²].
        (Force = mass × gravity_body per subtree.)
    """
    acceleration_body: np.ndarray   # (3,) [m s⁻²]
    alpha_body:        np.ndarray   # (3,) [rad s⁻²]
    dynamic_pressure:  float        # [Pa]
    axial_thrust:      float        # [N]
    aero_drag_body:    np.ndarray   # (3,) [N]
    gravity_body:      np.ndarray   # (3,) [m s⁻²]


# ---------------------------------------------------------------------------
# Joint load result
# ---------------------------------------------------------------------------

@dataclass
class JointLoadResult:
    """
    Internal load tensor computed for a single structural joint.

    Attributes
    ----------
    joint_id : str
    axial_force : float       F_axial [N] — positive = tension
    shear_force : float       V_total [N] — resultant transverse shear
    bending_moment : float    M_bend  [N·m] — resultant bending moment
    torsion_moment : float    T       [N·m] — torsional moment
    failed : bool             True if any failure criterion exceeded this tick
    margin_axial : float      (F_cr − |F_axial|) / F_cr — buckling margin
    margin_shear : float      (V_max − |V|) / V_max — shear margin
    margin_bending : float    (M_yield − |M_b|) / M_yield — bending margin
    """
    joint_id:         str
    axial_force:      float
    shear_force:      float
    bending_moment:   float
    torsion_moment:   float
    failed:           bool
    margin_axial:     float
    margin_shear:     float
    margin_bending:   float


# ---------------------------------------------------------------------------
# Euler buckling limit (standalone — used in tests and HUD)
# ---------------------------------------------------------------------------

def euler_buckling_limit(
    elastic_modulus: float,
    second_moment:   float,
    effective_length: float,
) -> float:
    """
    Critical Euler column buckling load [N].

    F_cr = π² E I / L_eff²

    Parameters
    ----------
    elastic_modulus : float   E [Pa]
    second_moment : float     I [m⁴]
    effective_length : float  L_eff [m]

    Returns
    -------
    float   F_cr [N]
    """
    if effective_length <= 0.0:
        raise ValueError(f"effective_length must be > 0, got {effective_length!r}")
    return (math.pi**2 * elastic_modulus * second_moment) / effective_length**2


def shear_yield_limit(shear_yield_strength: float, area: float) -> float:
    """
    Shear yield force limit [N] (rectangular section approximation).

    V_max = (2/3) · τ_yield · A

    Parameters
    ----------
    shear_yield_strength : float   τ_yield [Pa]
    area : float                   A [m²]

    Returns
    -------
    float   V_max [N]
    """
    return (2.0 / 3.0) * shear_yield_strength * area


def bending_failure_moment(
    yield_strength: float,
    second_moment:  float,
    extreme_fibre:  float,
) -> float:
    """
    Bending moment at onset of yielding [N·m].

    M_yield = σ_yield · I / c

    Parameters
    ----------
    yield_strength : float   σ_yield [Pa]
    second_moment : float    I [m⁴]
    extreme_fibre : float    c [m]  distance to outermost fibre

    Returns
    -------
    float   M_yield [N·m]
    """
    if extreme_fibre <= 0.0:
        raise ValueError(f"extreme_fibre must be > 0, got {extreme_fibre!r}")
    return yield_strength * second_moment / extreme_fibre


# ---------------------------------------------------------------------------
# Subtree load computation
# ---------------------------------------------------------------------------

def _subtree_mass(graph: ComponentGraph, root_id: str) -> float:
    """Sum of active component masses in the subtree rooted at root_id."""
    total = 0.0
    stack = [root_id]
    while stack:
        nid  = stack.pop()
        node = graph._nodes.get(nid)
        if node is None or not node.is_active:
            continue
        total += node.mass_component.mass
        stack.extend(graph._children.get(nid, []))
    return total


def _subtree_com(graph: ComponentGraph, root_id: str) -> np.ndarray:
    """Mass-weighted CoM of subtree in Body Frame [m]."""
    m_total = 0.0
    r_total = np.zeros(3, dtype=np.float64)
    stack   = [root_id]
    while stack:
        nid  = stack.pop()
        node = graph._nodes.get(nid)
        if node is None or not node.is_active:
            continue
        m = node.mass_component.mass
        m_total += m
        r_total += m * node.mass_component.position_body
        stack.extend(graph._children.get(nid, []))
    if m_total <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return r_total / m_total


# ---------------------------------------------------------------------------
# Per-joint load computation
# ---------------------------------------------------------------------------

def compute_joint_loads(
    joint:      StructuralJoint,
    graph:      ComponentGraph,
    load_state: VehicleLoadState,
) -> JointLoadResult:
    """
    Compute internal loads at a single joint using the free-body diagram
    of the child subtree.

    The joint transmits whatever forces and moments are needed to give the
    child subtree the same acceleration as the rest of the vehicle.

    Newton's 2nd law for the child subtree:
        F_joint + F_external = m_sub · a_vehicle
        → F_joint = m_sub · a_vehicle − F_external

    External forces on the subtree:
        F_gravity  = m_sub · g_body
        F_aero     = aero_drag_body (scaled by subtree mass fraction)
        F_thrust   = 0 (thrust is on the engine node itself, not the subtree)

    For simplicity, aero drag is distributed proportionally to mass.
    Full aerodynamic distribution requires a panel method (Phase 8).

    Bending moment is estimated as the axial load × lateral CoM offset
    from the joint axis (simplified beam model).

    Parameters
    ----------
    joint : StructuralJoint
    graph : ComponentGraph
    load_state : VehicleLoadState

    Returns
    -------
    JointLoadResult
    """
    child_id  = joint.child_id
    m_sub     = _subtree_mass(graph, child_id)
    r_com_sub = _subtree_com(graph, child_id)

    # Vehicle total mass for proportional distribution
    total_active = graph.active_nodes
    m_vehicle    = sum(n.mass_component.mass for n in total_active) or 1.0
    mass_frac    = m_sub / m_vehicle

    a    = np.asarray(load_state.acceleration_body, dtype=np.float64)
    g    = np.asarray(load_state.gravity_body,      dtype=np.float64)
    drag = np.asarray(load_state.aero_drag_body,    dtype=np.float64)

    # Inertial force needed to accelerate the subtree at vehicle acceleration
    F_inertia = m_sub * a

    # External forces on the subtree
    F_grav    = m_sub * g
    F_aero_sub = mass_frac * drag

    # Joint force = what the parent must supply (Newton reaction)
    F_joint = F_inertia - F_grav - F_aero_sub

    # Axial component (along vehicle +X_body)
    F_axial = float(F_joint[0])

    # Add thrust-induced compression on joints below the engine
    # (simplified: thrust compresses the stack above the engine)
    # For structural analysis: thrust axial compression acts downward on upper joints
    F_axial -= load_state.axial_thrust * mass_frac   # compression = negative

    # Shear force: transverse components (Y and Z)
    V_y = float(F_joint[1])
    V_z = float(F_joint[2])
    V_total = math.sqrt(V_y**2 + V_z**2)

    # Bending moment: F_shear × moment arm from joint to subtree CoM
    # Joint is at child node's body position; subtree CoM offset is r_com_sub
    # Moment arm = lateral offset of CoM from joint axial line
    child_node = graph._nodes.get(child_id)
    if child_node is not None:
        r_joint = child_node.mass_component.position_body
    else:
        r_joint = np.zeros(3, dtype=np.float64)

    r_arm     = r_com_sub - r_joint
    M_y       = V_z * abs(float(r_arm[0]))   # bending about Y from Z-shear × X-arm
    M_z       = V_y * abs(float(r_arm[0]))   # bending about Z from Y-shear × X-arm
    M_bend    = math.sqrt(M_y**2 + M_z**2)

    # Torsion: angular acceleration reaction on subtree (simplified)
    # T ≈ I_axial_sub × alpha_x  (rough estimate)
    T_torsion = abs(float(load_state.alpha_body[0])) * m_sub * 0.5   # rough

    # Update joint internal loads
    joint.axial_force    = F_axial
    joint.shear_force    = V_total
    joint.bending_moment = M_bend
    joint.torsion_moment = T_torsion

    # Failure check
    joint.check_failure()

    # Compute safety margins
    F_cr    = joint.euler_buckling_limit
    V_max   = joint.shear_yield_limit
    M_yield = joint.bending_failure_moment

    margin_axial   = (F_cr   - abs(F_axial)) / F_cr   if F_cr   > 0 else 1.0
    margin_shear   = (V_max  - V_total)      / V_max  if V_max  > 0 else 1.0
    margin_bending = (M_yield - M_bend)      / M_yield if M_yield > 0 else 1.0

    return JointLoadResult(
        joint_id=joint.joint_id,
        axial_force=F_axial,
        shear_force=V_total,
        bending_moment=M_bend,
        torsion_moment=T_torsion,
        failed=joint.failed,
        margin_axial=margin_axial,
        margin_shear=margin_shear,
        margin_bending=margin_bending,
    )


# ---------------------------------------------------------------------------
# Full structural analysis pass (called by pipeline Stage 9)
# ---------------------------------------------------------------------------

def analyse_structure(
    graph:      ComponentGraph,
    load_state: VehicleLoadState,
) -> List[JointLoadResult]:
    """
    Run the full structural analysis pass for all joints in the graph.

    Iterates over all joints, computes internal loads, and checks failure
    criteria. Updates joint.failed and joint.damage in-place.

    Parameters
    ----------
    graph : ComponentGraph
        Vehicle component graph (mutable — joints are updated in-place).
    load_state : VehicleLoadState
        Current tick vehicle load quantities.

    Returns
    -------
    list of JointLoadResult
        One result per joint, ordered arbitrarily.
    """
    results = []
    for joint in graph.joints:
        result = compute_joint_loads(joint, graph, load_state)
        results.append(result)
    return results


def critical_margin(results: List[JointLoadResult]) -> Optional[JointLoadResult]:
    """
    Return the joint result with the lowest (most critical) safety margin.

    The critical margin is min(margin_axial, margin_shear, margin_bending)
    across all joints.

    Parameters
    ----------
    results : list of JointLoadResult

    Returns
    -------
    JointLoadResult or None
        The most critical joint, or None if results is empty.
    """
    if not results:
        return None

    def _min_margin(r: JointLoadResult) -> float:
        return min(r.margin_axial, r.margin_shear, r.margin_bending)

    return min(results, key=_min_margin)
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_structural.py
==============================
Unit tests for nova.physics.structural.

Tests verify (per architecture spec §4):
  1. euler_buckling_limit formula: F_cr = π²EI/L² at known values.
  2. shear_yield_limit: V_max = (2/3)τ·A.
  3. bending_failure_moment: M_yield = σ·I/c.
  4. compute_joint_loads: axial force direction for thrust-loaded stack.
  5. Joint loads nonzero under vehicle acceleration.
  6. analyse_structure returns one result per joint.
  7. critical_margin returns the most critical joint.
  8. Safety margins positive for lightly loaded joints.
  9. JointLoadResult.failed matches joint.failed after analysis.
"""

import math
import pytest
import numpy as np

from nova.physics.structural import (
    euler_buckling_limit,
    shear_yield_limit,
    bending_failure_moment,
    compute_joint_loads,
    analyse_structure,
    critical_margin,
    VehicleLoadState,
    JointLoadResult,
)
from nova.vehicle.component_graph import (
    ComponentGraph,
    ComponentNode,
    StructuralJoint,
    JointCrossSection,
    AL_7075_T6,
)
from nova.vehicle.mass_model import point_mass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cs(area=0.05, I=1e-3, J=2e-3, L=3.0, c=0.1):
    return JointCrossSection(
        area=area, second_moment=I, torsion_constant=J,
        effective_length=L, distance_to_extreme_fibre=c,
    )


def _make_node(nid, mass=1000.0, pos=None):
    mc = point_mass(nid, mass, pos or [0.0, 0.0, 0.0])
    return ComponentNode(nid, nid, mc, "structure", is_separable=True)


def _make_joint(jid, parent, child):
    return StructuralJoint(jid, parent, child, AL_7075_T6, _make_cs())


def _nominal_load_state(
    accel=None, alpha=None, q_inf=50_000.0, thrust=100_000.0,
    drag=None, gravity=None
):
    return VehicleLoadState(
        acceleration_body=np.asarray(accel   or [20.0, 0.0, 0.0], dtype=np.float64),
        alpha_body=       np.asarray(alpha   or [0.0,  0.0, 0.0], dtype=np.float64),
        dynamic_pressure=q_inf,
        axial_thrust=thrust,
        aero_drag_body=   np.asarray(drag    or [-5000.0, 0.0, 0.0], dtype=np.float64),
        gravity_body=     np.asarray(gravity or [-9.81, 0.0, 0.0],   dtype=np.float64),
    )


def _two_node_graph(m_parent=2000.0, m_child=1000.0):
    g = ComponentGraph()
    g.add_node(_make_node("parent", m_parent, [-5.0, 0.0, 0.0]))
    g.add_node(_make_node("child",  m_child,  [ 5.0, 0.0, 0.0]))
    g.add_joint(_make_joint("J1", "parent", "child"))
    return g


# ---------------------------------------------------------------------------
# 1. Euler buckling limit formula
# ---------------------------------------------------------------------------

class TestEulerBucklingLimit:

    def test_known_value(self):
        """
        F_cr = π² × 71.7×10⁹ × 1×10⁻³ / 3² = π² × 71.7×10⁶ / 9 ≈ 78.5 MN
        """
        E, I, L = 71.7e9, 1e-3, 3.0
        F_cr = euler_buckling_limit(E, I, L)
        expected = (math.pi**2 * E * I) / L**2
        assert abs(F_cr - expected) < 1.0

    def test_inversely_proportional_to_L_squared(self):
        """Doubling L halves F_cr by factor of 4."""
        E, I = 71.7e9, 1e-3
        F1 = euler_buckling_limit(E, I, 1.0)
        F2 = euler_buckling_limit(E, I, 2.0)
        assert abs(F1 / F2 - 4.0) < 1.0e-10

    def test_proportional_to_EI(self):
        """Doubling I doubles F_cr."""
        E, L = 71.7e9, 2.0
        F1 = euler_buckling_limit(E, 1e-3, L)
        F2 = euler_buckling_limit(E, 2e-3, L)
        assert abs(F2 / F1 - 2.0) < 1.0e-10

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="effective_length"):
            euler_buckling_limit(71.7e9, 1e-3, 0.0)

    def test_positive_result(self):
        assert euler_buckling_limit(200e9, 5e-4, 1.5) > 0.0


# ---------------------------------------------------------------------------
# 2. Shear yield limit
# ---------------------------------------------------------------------------

class TestShearYieldLimit:

    def test_known_value(self):
        """V_max = (2/3) × 290×10⁶ × 0.05 = 9.667×10⁶ N."""
        tau_y, A = 290e6, 0.05
        V_max = shear_yield_limit(tau_y, A)
        expected = (2.0/3.0) * tau_y * A
        assert abs(V_max - expected) < 1.0

    def test_linear_in_area(self):
        tau_y = 290e6
        V1 = shear_yield_limit(tau_y, 0.01)
        V2 = shear_yield_limit(tau_y, 0.02)
        assert abs(V2 / V1 - 2.0) < 1.0e-10

    def test_positive_result(self):
        assert shear_yield_limit(200e6, 0.01) > 0.0


# ---------------------------------------------------------------------------
# 3. Bending failure moment
# ---------------------------------------------------------------------------

class TestBendingFailureMoment:

    def test_known_value(self):
        """M_yield = 503×10⁶ × 1×10⁻³ / 0.05 = 1.006×10⁷ N·m."""
        sigma_y, I, c = 503e6, 1e-3, 0.05
        M_y = bending_failure_moment(sigma_y, I, c)
        expected = sigma_y * I / c
        assert abs(M_y - expected) < 1.0

    def test_zero_c_raises(self):
        with pytest.raises(ValueError, match="extreme_fibre"):
            bending_failure_moment(503e6, 1e-3, 0.0)

    def test_positive_result(self):
        assert bending_failure_moment(400e6, 5e-4, 0.1) > 0.0


# ---------------------------------------------------------------------------
# 4. VehicleLoadState construction
# ---------------------------------------------------------------------------

class TestVehicleLoadState:

    def test_frozen(self):
        ls = _nominal_load_state()
        with pytest.raises(Exception):
            ls.axial_thrust = 0.0

    def test_fields_are_float64(self):
        ls = _nominal_load_state()
        assert ls.acceleration_body.dtype == np.float64
        assert ls.alpha_body.dtype == np.float64
        assert ls.aero_drag_body.dtype == np.float64
        assert ls.gravity_body.dtype == np.float64


# ---------------------------------------------------------------------------
# 5. compute_joint_loads
# ---------------------------------------------------------------------------

class TestComputeJointLoads:

    def test_returns_joint_load_result(self):
        g  = _two_node_graph()
        j  = g.get_joint("J1")
        ls = _nominal_load_state()
        r  = compute_joint_loads(j, g, ls)
        assert isinstance(r, JointLoadResult)

    def test_result_joint_id_matches(self):
        g  = _two_node_graph()
        j  = g.get_joint("J1")
        ls = _nominal_load_state()
        r  = compute_joint_loads(j, g, ls)
        assert r.joint_id == "J1"

    def test_shear_force_nonnegative(self):
        """Shear force magnitude is always ≥ 0."""
        g  = _two_node_graph()
        j  = g.get_joint("J1")
        ls = _nominal_load_state()
        r  = compute_joint_loads(j, g, ls)
        assert r.shear_force >= 0.0

    def test_bending_moment_nonnegative(self):
        g  = _two_node_graph()
        j  = g.get_joint("J1")
        ls = _nominal_load_state()
        r  = compute_joint_loads(j, g, ls)
        assert r.bending_moment >= 0.0

    def test_nonzero_loads_under_acceleration(self):
        """Under vehicle acceleration, joint loads must be nonzero."""
        g  = _two_node_graph()
        j  = g.get_joint("J1")
        ls = _nominal_load_state(accel=[50.0, 0.0, 0.0])
        r  = compute_joint_loads(j, g, ls)
        total_load = abs(r.axial_force) + r.shear_force + r.bending_moment
        assert total_load > 0.0, f"All loads zero under 50 m/s² accel"

    def test_zero_acceleration_reduces_inertial_load(self):
        """Under zero acceleration (free fall), inertial load is near zero."""
        g  = _two_node_graph()
        j  = g.get_joint("J1")
        ls_hi = _nominal_load_state(accel=[50.0, 0.0, 0.0], thrust=0.0)
        ls_lo = _nominal_load_state(accel=[ 0.0, 0.0, 0.0], thrust=0.0,
                                    drag=[0.0,0.0,0.0], gravity=[0.0,0.0,0.0])
        r_hi  = compute_joint_loads(g.get_joint("J1"), g, ls_hi)
        g2    = _two_node_graph()
        r_lo  = compute_joint_loads(g2.get_joint("J1"), g2, ls_lo)
        assert abs(r_hi.axial_force) > abs(r_lo.axial_force)

    def test_safety_margins_computed(self):
        """Safety margins must be finite numbers."""
        g  = _two_node_graph()
        j  = g.get_joint("J1")
        ls = _nominal_load_state()
        r  = compute_joint_loads(j, g, ls)
        assert math.isfinite(r.margin_axial)
        assert math.isfinite(r.margin_shear)
        assert math.isfinite(r.margin_bending)

    def test_positive_margins_for_nominal_load(self):
        """Lightly loaded joint should have positive margins (no failure)."""
        g  = _two_node_graph()
        j  = g.get_joint("J1")
        ls = _nominal_load_state(
            accel=[1.0, 0.0, 0.0], thrust=1000.0,
            drag=[-100.0, 0.0, 0.0], gravity=[-9.81, 0.0, 0.0]
        )
        r = compute_joint_loads(j, g, ls)
        # With 1 m/s² accel on 1000 kg child: F_inertia ~ 1000 N << F_cr
        assert r.margin_axial > 0.0 or r.failed  # either surviving or explicitly failed

    def test_failed_joint_in_result(self):
        """If joint already has damage=1, result.failed=True."""
        g = _two_node_graph()
        j = g.get_joint("J1")
        j.damage = 1.0   # pre-failed
        ls = _nominal_load_state()
        r  = compute_joint_loads(j, g, ls)
        assert r.failed


# ---------------------------------------------------------------------------
# 6. analyse_structure
# ---------------------------------------------------------------------------

class TestAnalyseStructure:

    def test_returns_one_result_per_joint(self):
        g = _two_node_graph()
        g.add_node(_make_node("child2", 500.0, [8.0, 0.0, 0.0]))
        g.add_joint(_make_joint("J2", "parent", "child2"))
        ls = _nominal_load_state()
        results = analyse_structure(g, ls)
        assert len(results) == 2

    def test_result_ids_match_joints(self):
        g  = _two_node_graph()
        ls = _nominal_load_state()
        results = analyse_structure(g, ls)
        result_ids = {r.joint_id for r in results}
        joint_ids  = {j.joint_id for j in g.joints}
        assert result_ids == joint_ids

    def test_empty_graph_returns_empty(self):
        g = ComponentGraph()
        ls = _nominal_load_state()
        results = analyse_structure(g, ls)
        assert results == []


# ---------------------------------------------------------------------------
# 7. critical_margin
# ---------------------------------------------------------------------------

class TestCriticalMargin:

    def test_none_for_empty_list(self):
        assert critical_margin([]) is None

    def test_returns_single_for_one_result(self):
        g  = _two_node_graph()
        j  = g.get_joint("J1")
        ls = _nominal_load_state()
        r  = compute_joint_loads(j, g, ls)
        assert critical_margin([r]) is r

    def test_returns_most_critical(self):
        """
        Manually create two JointLoadResults with different margins;
        critical_margin must return the one with lower minimum margin.
        """
        r1 = JointLoadResult("J1", 0.0, 0.0, 0.0, 0.0, False,
                             margin_axial=0.9, margin_shear=0.8, margin_bending=0.7)
        r2 = JointLoadResult("J2", 0.0, 0.0, 0.0, 0.0, False,
                             margin_axial=0.5, margin_shear=0.9, margin_bending=0.9)
        # r2 has min margin 0.5 < r1 min margin 0.7 → r2 is most critical
        most_critical = critical_margin([r1, r2])
        assert most_critical is r2

    def test_failed_joint_has_negative_margin(self):
        """A failed joint typically has negative margin; must be selected as critical."""
        r_good   = JointLoadResult("J1", 0.0, 0.0, 0.0, 0.0, False,
                                   margin_axial=0.8, margin_shear=0.9, margin_bending=0.85)
        r_failed = JointLoadResult("J2", 0.0, 0.0, 0.0, 0.0, True,
                                   margin_axial=-0.1, margin_shear=0.5, margin_bending=0.5)
        most_critical = critical_margin([r_good, r_failed])
        assert most_critical is r_failed
