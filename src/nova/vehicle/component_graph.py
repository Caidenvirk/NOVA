"""
nova.vehicle.component_graph
============================
Directed graph of vehicle components connected by structural joints.

Architecture role — Pipeline Stage 9 (Component Updates)
---------------------------------------------------------
The ComponentGraph is the authoritative registry of all vehicle parts and
their inter-connections. It is consumed by:
  - MassModel computation (active component list)
  - Structural solver (joint load paths)
  - AI Monitor (component health flags)
  - Propulsion model (propellant mass per tank)
  - Avionics HUD (component status display)

Graph structure
---------------
Each node is a ``ComponentNode`` (a vehicle part: tank, engine, fairing…).
Each directed edge is a ``StructuralJoint`` (a physical connection between
two parts with defined material properties and failure limits).

The graph is a directed acyclic graph (DAG) rooted at the primary structure
(e.g. the interstage or payload adapter). The direction of edges represents
the load path: parent → child = structural load flows from parent to child.

Joint failure model
-------------------
Each joint tracks a scalar damage parameter D ∈ [0, 1]:
  D = 0  → undamaged
  D = 1  → failure (instantaneous separation)

Damage accumulates via cycle counting against material S-N curve data.
Structural limit checks (Euler buckling, shear yield, bending) are evaluated
by the structural solver per tick and trigger instantaneous D = 1 if exceeded.

References
----------
- Megson, "Aircraft Structures for Engineering Students", 5th ed., §15
- Bruhn, "Analysis and Design of Flight Vehicle Structures", §C2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from nova.vehicle.mass_model import MassComponent


# ---------------------------------------------------------------------------
# Structural material properties
# ---------------------------------------------------------------------------

@dataclass
class MaterialProperties:
    """
    Structural material properties for a joint cross-section.

    Parameters
    ----------
    elastic_modulus : float
        Young's modulus E [Pa].
    shear_modulus : float
        Shear modulus G [Pa].
    yield_strength : float
        Tensile/compressive yield strength σ_yield [Pa].
    shear_yield_strength : float
        Shear yield strength τ_yield [Pa].  Typically ≈ σ_yield / √3.
    ultimate_strength : float
        Ultimate tensile strength σ_ult [Pa].
    density : float
        Material density [kg m⁻³].  Used for structural mass estimation.
    """
    elastic_modulus:      float   # E [Pa]
    shear_modulus:        float   # G [Pa]
    yield_strength:       float   # σ_yield [Pa]
    shear_yield_strength: float   # τ_yield [Pa]
    ultimate_strength:    float   # σ_ult [Pa]
    density:              float   # ρ [kg m⁻³]

    def __post_init__(self) -> None:
        for attr in ("elastic_modulus", "shear_modulus", "yield_strength",
                     "shear_yield_strength", "ultimate_strength", "density"):
            if getattr(self, attr) <= 0.0:
                raise ValueError(
                    f"MaterialProperties.{attr} must be > 0, "
                    f"got {getattr(self, attr)!r}"
                )


# Pre-defined materials
AL_7075_T6 = MaterialProperties(
    elastic_modulus=71.7e9,
    shear_modulus=26.9e9,
    yield_strength=503e6,
    shear_yield_strength=290e6,
    ultimate_strength=572e6,
    density=2810.0,
)

STEEL_4340 = MaterialProperties(
    elastic_modulus=200e9,
    shear_modulus=76.9e9,
    yield_strength=470e6,
    shear_yield_strength=271e6,
    ultimate_strength=745e6,
    density=7850.0,
)

CARBON_FIBER_T300 = MaterialProperties(
    elastic_modulus=70e9,
    shear_modulus=5e9,
    yield_strength=600e6,
    shear_yield_strength=90e6,
    ultimate_strength=3530e6,
    density=1760.0,
)


# ---------------------------------------------------------------------------
# Structural joint cross-section
# ---------------------------------------------------------------------------

@dataclass
class JointCrossSection:
    """
    Cross-sectional geometry of a structural joint.

    Used by the structural solver to compute internal loads per unit area
    and apply failure criteria.

    Parameters
    ----------
    area : float
        Cross-sectional area A [m²].
    second_moment : float
        Second moment of area I [m⁴] about the bending neutral axis.
    torsion_constant : float
        Torsional constant J [m⁴]. Equal to polar moment for circular sections.
    effective_length : float
        Effective column length L_eff [m] for Euler buckling: L_eff = K·L.
        K = 1.0 for pin-pin, 0.5 for fixed-fixed, etc.
    distance_to_extreme_fibre : float
        c [m] — distance from neutral axis to outermost fibre.
        Used in M/I = σ/c for bending stress.
    """
    area:                      float   # A [m²]
    second_moment:             float   # I [m⁴]
    torsion_constant:          float   # J [m⁴]
    effective_length:          float   # L_eff [m]
    distance_to_extreme_fibre: float   # c [m]

    def __post_init__(self) -> None:
        for attr in ("area", "second_moment", "torsion_constant",
                     "effective_length", "distance_to_extreme_fibre"):
            if getattr(self, attr) <= 0.0:
                raise ValueError(
                    f"JointCrossSection.{attr} must be > 0, "
                    f"got {getattr(self, attr)!r}"
                )


# ---------------------------------------------------------------------------
# Structural joint
# ---------------------------------------------------------------------------

@dataclass
class StructuralJoint:
    """
    Structural connection between two ComponentNodes.

    Tracks internal load state and damage accumulation.

    Parameters
    ----------
    joint_id : str
        Unique identifier for this joint.
    parent_id : str
        Node ID of the parent (load-path upstream) component.
    child_id : str
        Node ID of the child (load-path downstream) component.
    material : MaterialProperties
        Material properties of the joint cross-section.
    cross_section : JointCrossSection
        Geometric properties of the joint cross-section.
    """
    joint_id:      str
    parent_id:     str
    child_id:      str
    material:      MaterialProperties
    cross_section: JointCrossSection

    # Internal load state — updated each tick by the structural solver
    axial_force:     float = 0.0   # F_a [N], positive = tension
    shear_force:     float = 0.0   # V   [N]
    bending_moment:  float = 0.0   # M_b [N·m]
    torsion_moment:  float = 0.0   # T   [N·m]
    damage:          float = 0.0   # D ∈ [0, 1]
    failed:          bool  = False

    @property
    def euler_buckling_limit(self) -> float:
        """
        Critical Euler buckling load [N].

        F_cr = π² E I / L_eff²

        Applies to compressive axial loads (F_a < 0).
        """
        cs = self.cross_section
        mat = self.material
        return (math.pi**2 * mat.elastic_modulus * cs.second_moment
                / cs.effective_length**2)

    @property
    def shear_yield_limit(self) -> float:
        """
        Shear yield force limit [N].

        V_max = (2/3) · τ_yield · A    (rectangular cross-section approximation)
        """
        return (2.0 / 3.0) * self.material.shear_yield_strength * self.cross_section.area

    @property
    def bending_failure_moment(self) -> float:
        """
        Bending moment at which yielding initiates [N·m].

        M_yield = σ_yield · I / c
        """
        cs = self.cross_section
        return self.material.yield_strength * cs.second_moment / cs.distance_to_extreme_fibre

    def check_failure(self) -> bool:
        """
        Evaluate all failure criteria and set self.failed = True if any is exceeded.

        Returns
        -------
        bool
            True if this joint has failed (newly or previously).
        """
        if self.failed:
            return True

        # Euler buckling (compressive axial)
        if self.axial_force < 0.0 and abs(self.axial_force) > self.euler_buckling_limit:
            self.failed = True
            self.damage = 1.0
            return True

        # Shear yield
        if abs(self.shear_force) > self.shear_yield_limit:
            self.failed = True
            self.damage = 1.0
            return True

        # Bending failure
        if abs(self.bending_moment) > self.bending_failure_moment:
            self.failed = True
            self.damage = 1.0
            return True

        # Damage accumulation threshold
        if self.damage >= 1.0:
            self.failed = True
            return True

        return False


# ---------------------------------------------------------------------------
# Component node
# ---------------------------------------------------------------------------

@dataclass
class ComponentNode:
    """
    A single vehicle component in the structural graph.

    Parameters
    ----------
    node_id : str
        Unique identifier (e.g. ``"stage1_tank"``, ``"engine_merlin"``).
    display_name : str
        Human-readable name for HUD / telemetry display.
    mass_component : MassComponent
        Mass and inertia data for this component.
    component_type : str
        Category string: ``"tank"``, ``"engine"``, ``"structure"``,
        ``"fairing"``, ``"payload"``, ``"rcs"``, ``"battery"``.
    is_separable : bool
        If True, this component can be jettisoned (stage separation, fairing).
    """
    node_id:         str
    display_name:    str
    mass_component:  MassComponent
    component_type:  str
    is_separable:    bool = False

    @property
    def is_active(self) -> bool:
        """Delegate to the underlying MassComponent active flag."""
        return self.mass_component.is_active

    def jettison(self) -> None:
        """Mark this component as jettisoned (inactive)."""
        if not self.is_separable:
            raise RuntimeError(
                f"ComponentNode '{self.node_id}' is not separable — "
                "cannot jettison."
            )
        self.mass_component.is_active = False


# ---------------------------------------------------------------------------
# Component graph
# ---------------------------------------------------------------------------

class ComponentGraph:
    """
    Directed acyclic graph (DAG) of vehicle components and structural joints.

    Usage
    -----
    Build the graph at vehicle initialisation::

        graph = ComponentGraph()
        graph.add_node(ComponentNode("tank_lox", ...))
        graph.add_node(ComponentNode("engine", ...))
        graph.add_joint(StructuralJoint("interstage", "tank_lox", "engine", ...))

    During the simulation (Stage 9)::

        mass_props = graph.compute_mass_model()
        failed_joints = graph.evaluate_structural_failures()
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ComponentNode] = {}
        self._joints: Dict[str, StructuralJoint] = {}
        self._children: Dict[str, List[str]] = {}    # node_id → [child node_ids]
        self._parents:  Dict[str, List[str]] = {}    # node_id → [parent node_ids]

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: ComponentNode) -> None:
        """
        Add a component node to the graph.

        Raises
        ------
        ValueError if a node with the same ID already exists.
        """
        if node.node_id in self._nodes:
            raise ValueError(
                f"ComponentGraph: node '{node.node_id}' already exists."
            )
        self._nodes[node.node_id]    = node
        self._children[node.node_id] = []
        self._parents[node.node_id]  = []

    def add_joint(self, joint: StructuralJoint) -> None:
        """
        Add a structural joint connecting two existing nodes.

        Raises
        ------
        ValueError if either node ID is missing or joint ID is duplicate.
        """
        if joint.joint_id in self._joints:
            raise ValueError(
                f"ComponentGraph: joint '{joint.joint_id}' already exists."
            )
        if joint.parent_id not in self._nodes:
            raise ValueError(
                f"ComponentGraph: parent node '{joint.parent_id}' not found."
            )
        if joint.child_id not in self._nodes:
            raise ValueError(
                f"ComponentGraph: child node '{joint.child_id}' not found."
            )
        self._joints[joint.joint_id] = joint
        self._children[joint.parent_id].append(joint.child_id)
        self._parents[joint.child_id].append(joint.parent_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> ComponentNode:
        if node_id not in self._nodes:
            raise KeyError(f"ComponentGraph: no node '{node_id}'.")
        return self._nodes[node_id]

    def get_joint(self, joint_id: str) -> StructuralJoint:
        if joint_id not in self._joints:
            raise KeyError(f"ComponentGraph: no joint '{joint_id}'.")
        return self._joints[joint_id]

    @property
    def nodes(self) -> List[ComponentNode]:
        """All nodes (active and inactive)."""
        return list(self._nodes.values())

    @property
    def active_nodes(self) -> List[ComponentNode]:
        """Only active (non-jettisoned) nodes."""
        return [n for n in self._nodes.values() if n.is_active]

    @property
    def joints(self) -> List[StructuralJoint]:
        """All structural joints."""
        return list(self._joints.values())

    @property
    def failed_joints(self) -> List[StructuralJoint]:
        """Joints that have failed (D = 1)."""
        return [j for j in self._joints.values() if j.failed]

    def children_of(self, node_id: str) -> List[ComponentNode]:
        """Return direct children of the given node."""
        return [self._nodes[nid] for nid in self._children.get(node_id, [])]

    def parents_of(self, node_id: str) -> List[ComponentNode]:
        """Return direct parents of the given node."""
        return [self._nodes[nid] for nid in self._parents.get(node_id, [])]

    # ------------------------------------------------------------------
    # Mass model integration
    # ------------------------------------------------------------------

    def active_mass_components(self) -> List[MassComponent]:
        """Return MassComponent objects for all active nodes."""
        return [n.mass_component for n in self.active_nodes]

    # ------------------------------------------------------------------
    # Structural failure evaluation (Stage 9 call)
    # ------------------------------------------------------------------

    def evaluate_structural_failures(self) -> List[StructuralJoint]:
        """
        Evaluate all joints for structural failure this tick.

        Returns
        -------
        list of StructuralJoint
            Joints that newly failed or were already failed.
        """
        failed = []
        for joint in self._joints.values():
            if joint.check_failure():
                failed.append(joint)
        return failed

    def jettison_subtree(self, node_id: str) -> List[str]:
        """
        Jettison a node and all its descendants (recursive stage separation).

        Parameters
        ----------
        node_id : str
            Root of the sub-tree to jettison.

        Returns
        -------
        list of str
            Node IDs that were jettisoned.
        """
        jettisoned = []
        stack = [node_id]
        while stack:
            nid = stack.pop()
            node = self._nodes.get(nid)
            if node is None:
                continue
            node.mass_component.is_active = False
            jettisoned.append(nid)
            stack.extend(self._children.get(nid, []))
        return jettisoned

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a one-line summary string for logging."""
        n_active = len(self.active_nodes)
        n_total  = len(self._nodes)
        n_failed = len(self.failed_joints)
        return (
            f"ComponentGraph: {n_active}/{n_total} active nodes, "
            f"{len(self._joints)} joints, {n_failed} failed"
        )

    def __repr__(self) -> str:
        return self.summary()
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_component_graph.py
===================================
Unit tests for nova.vehicle.component_graph.

Tests verify:
  1. Nodes added correctly; duplicate ID raises.
  2. Joints added correctly; missing node ID raises.
  3. children_of / parents_of navigation.
  4. active_nodes excludes jettisoned components.
  5. jettison_subtree deactivates a node and all its descendants.
  6. evaluate_structural_failures returns failed joints.
  7. MaterialProperties and JointCrossSection validate inputs.
  8. StructuralJoint failure criteria: Euler buckling, shear yield, bending.
  9. summary() string format.
"""

import math
import pytest
import numpy as np

from nova.vehicle.component_graph import (
    ComponentGraph,
    ComponentNode,
    StructuralJoint,
    MaterialProperties,
    JointCrossSection,
    AL_7075_T6,
    STEEL_4340,
    CARBON_FIBER_T300,
)
from nova.vehicle.mass_model import point_mass


# Re-export analytical helpers for test use
def euler_buckling_limit(E, I, L):
    return (math.pi**2 * E * I) / L**2

def shear_yield_limit(tau_y, A):
    return (2.0/3.0) * tau_y * A

def bending_failure_moment(sigma_y, I, c):
    return sigma_y * I / c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cross_section(area=0.01, I=1e-4, J=2e-4, L=2.0, c=0.05):
    return JointCrossSection(
        area=area,
        second_moment=I,
        torsion_constant=J,
        effective_length=L,
        distance_to_extreme_fibre=c,
    )


def _make_joint(jid, parent, child, mat=None, cs=None):
    return StructuralJoint(
        joint_id=jid,
        parent_id=parent,
        child_id=child,
        material=mat or AL_7075_T6,
        cross_section=cs or _make_cross_section(),
    )


def _make_node(nid, mass=500.0, pos=None):
    mc = point_mass(nid, mass, pos or [0.0, 0.0, 0.0])
    return ComponentNode(
        node_id=nid,
        display_name=nid.replace("_", " ").title(),
        mass_component=mc,
        component_type="structure",
        is_separable=True,
    )


# ---------------------------------------------------------------------------
# 1. Graph construction
# ---------------------------------------------------------------------------

class TestGraphConstruction:

    def test_add_node(self):
        g = ComponentGraph()
        g.add_node(_make_node("A"))
        assert "A" in [n.node_id for n in g.nodes]

    def test_duplicate_node_raises(self):
        g = ComponentGraph()
        g.add_node(_make_node("A"))
        with pytest.raises(ValueError, match="already exists"):
            g.add_node(_make_node("A"))

    def test_add_joint(self):
        g = ComponentGraph()
        g.add_node(_make_node("P"))
        g.add_node(_make_node("C"))
        g.add_joint(_make_joint("J1", "P", "C"))
        assert "J1" in [j.joint_id for j in g.joints]

    def test_duplicate_joint_raises(self):
        g = ComponentGraph()
        g.add_node(_make_node("P"))
        g.add_node(_make_node("C"))
        g.add_joint(_make_joint("J1", "P", "C"))
        with pytest.raises(ValueError, match="already exists"):
            g.add_joint(_make_joint("J1", "P", "C"))

    def test_joint_missing_parent_raises(self):
        g = ComponentGraph()
        g.add_node(_make_node("C"))
        with pytest.raises(ValueError, match="parent node"):
            g.add_joint(_make_joint("J1", "MISSING", "C"))

    def test_joint_missing_child_raises(self):
        g = ComponentGraph()
        g.add_node(_make_node("P"))
        with pytest.raises(ValueError, match="child node"):
            g.add_joint(_make_joint("J1", "P", "MISSING"))

    def test_get_node(self):
        g = ComponentGraph()
        g.add_node(_make_node("A", mass=123.0))
        n = g.get_node("A")
        assert n.mass_component.mass == 123.0

    def test_get_node_missing_raises(self):
        g = ComponentGraph()
        with pytest.raises(KeyError):
            g.get_node("NOPE")

    def test_get_joint(self):
        g = ComponentGraph()
        g.add_node(_make_node("P"))
        g.add_node(_make_node("C"))
        g.add_joint(_make_joint("J1", "P", "C"))
        j = g.get_joint("J1")
        assert j.joint_id == "J1"

    def test_get_joint_missing_raises(self):
        g = ComponentGraph()
        with pytest.raises(KeyError):
            g.get_joint("NOPE")


# ---------------------------------------------------------------------------
# 2. Graph navigation
# ---------------------------------------------------------------------------

class TestGraphNavigation:

    @pytest.fixture
    def three_node_graph(self):
        g = ComponentGraph()
        for nid in ["root", "child1", "child2", "grandchild"]:
            g.add_node(_make_node(nid))
        g.add_joint(_make_joint("j_rc1", "root", "child1"))
        g.add_joint(_make_joint("j_rc2", "root", "child2"))
        g.add_joint(_make_joint("j_c1g", "child1", "grandchild"))
        return g

    def test_children_of_root(self, three_node_graph):
        children = three_node_graph.children_of("root")
        ids = {n.node_id for n in children}
        assert ids == {"child1", "child2"}

    def test_children_of_leaf(self, three_node_graph):
        children = three_node_graph.children_of("grandchild")
        assert children == []

    def test_parents_of_grandchild(self, three_node_graph):
        parents = three_node_graph.parents_of("grandchild")
        assert [n.node_id for n in parents] == ["child1"]

    def test_parents_of_root(self, three_node_graph):
        parents = three_node_graph.parents_of("root")
        assert parents == []


# ---------------------------------------------------------------------------
# 3. Active nodes and jettison
# ---------------------------------------------------------------------------

class TestActiveAndJettison:

    @pytest.fixture
    def simple_graph(self):
        g = ComponentGraph()
        g.add_node(_make_node("stage1"))
        g.add_node(_make_node("stage2"))
        g.add_node(_make_node("payload"))
        g.add_joint(_make_joint("j12", "stage1", "stage2"))
        g.add_joint(_make_joint("j2p", "stage2", "payload"))
        return g

    def test_all_active_initially(self, simple_graph):
        assert len(simple_graph.active_nodes) == 3

    def test_jettison_single_node(self, simple_graph):
        simple_graph.jettison_subtree("stage1")
        active_ids = {n.node_id for n in simple_graph.active_nodes}
        assert "stage1" not in active_ids

    def test_jettison_subtree_deactivates_descendants(self, simple_graph):
        simple_graph.jettison_subtree("stage2")
        active_ids = {n.node_id for n in simple_graph.active_nodes}
        assert "stage2" not in active_ids
        assert "payload" not in active_ids

    def test_jettison_subtree_returns_list(self, simple_graph):
        jettisoned = simple_graph.jettison_subtree("stage2")
        assert set(jettisoned) == {"stage2", "payload"}

    def test_non_separable_node_raises(self):
        g = ComponentGraph()
        mc = point_mass("body", 100.0, [0.0, 0.0, 0.0])
        n  = ComponentNode("body", "Body", mc, "structure", is_separable=False)
        g.add_node(n)
        with pytest.raises(RuntimeError, match="not separable"):
            n.jettison()

    def test_active_mass_components(self, simple_graph):
        comps = simple_graph.active_mass_components()
        assert len(comps) == 3
        # Jettison the leaf (payload) only — stage1 and stage2 stay active
        simple_graph.jettison_subtree("payload")
        comps2 = simple_graph.active_mass_components()
        assert len(comps2) == 2


# ---------------------------------------------------------------------------
# 4. Structural joint failure criteria
# ---------------------------------------------------------------------------

class TestStructuralJointFailure:

    def _joint_with_cs(self, area, I, J, L, c, mat=None):
        cs  = JointCrossSection(area=area, second_moment=I, torsion_constant=J,
                                effective_length=L, distance_to_extreme_fibre=c)
        mat = mat or AL_7075_T6
        j   = StructuralJoint("test", "P", "C", mat, cs)
        return j

    def test_no_load_no_failure(self):
        j = self._joint_with_cs(0.01, 1e-4, 2e-4, 2.0, 0.05)
        assert not j.check_failure()
        assert not j.failed

    def test_euler_buckling_just_below_limit(self):
        E, I, L = AL_7075_T6.elastic_modulus, 1e-4, 2.0
        F_cr = euler_buckling_limit(E, I, L)
        j = self._joint_with_cs(0.01, I, 2e-4, L, 0.05)
        j.axial_force = -(F_cr * 0.99)   # just below limit
        assert not j.check_failure()

    def test_euler_buckling_just_above_limit(self):
        E, I, L = AL_7075_T6.elastic_modulus, 1e-4, 2.0
        F_cr = euler_buckling_limit(E, I, L)
        j = self._joint_with_cs(0.01, I, 2e-4, L, 0.05)
        j.axial_force = -(F_cr * 1.01)   # just above limit
        assert j.check_failure()
        assert j.failed
        assert j.damage == 1.0

    def test_tensile_load_does_not_trigger_buckling(self):
        """Euler buckling only applies to compressive (negative) axial loads."""
        E, I, L = AL_7075_T6.elastic_modulus, 1e-4, 2.0
        F_cr = euler_buckling_limit(E, I, L)
        j = self._joint_with_cs(0.01, I, 2e-4, L, 0.05)
        j.axial_force = F_cr * 10.0   # large tension — should not buckle
        # May fail other criteria at very high tension, but not buckling
        j.check_failure()
        # Specifically: buckling criterion should not be the trigger
        # (Check: tensile force alone should not cause buckling failure
        #  unless shear/bending also exceeded — here they're zero)
        assert not j.failed

    def test_shear_yield_just_below(self):
        tau_y, A = AL_7075_T6.shear_yield_strength, 0.01
        V_max = shear_yield_limit(tau_y, A)
        j = self._joint_with_cs(A, 1e-4, 2e-4, 2.0, 0.05)
        j.shear_force = V_max * 0.99
        assert not j.check_failure()

    def test_shear_yield_just_above(self):
        tau_y, A = AL_7075_T6.shear_yield_strength, 0.01
        V_max = shear_yield_limit(tau_y, A)
        j = self._joint_with_cs(A, 1e-4, 2e-4, 2.0, 0.05)
        j.shear_force = V_max * 1.01
        assert j.check_failure()
        assert j.failed

    def test_bending_failure_just_below(self):
        sigma_y, I, c = AL_7075_T6.yield_strength, 1e-4, 0.05
        M_y = bending_failure_moment(sigma_y, I, c)
        j = self._joint_with_cs(0.01, I, 2e-4, 2.0, c)
        j.bending_moment = M_y * 0.99
        assert not j.check_failure()

    def test_bending_failure_just_above(self):
        sigma_y, I, c = AL_7075_T6.yield_strength, 1e-4, 0.05
        M_y = bending_failure_moment(sigma_y, I, c)
        j = self._joint_with_cs(0.01, I, 2e-4, 2.0, c)
        j.bending_moment = M_y * 1.01
        assert j.check_failure()
        assert j.failed

    def test_damage_threshold_triggers_failure(self):
        j = self._joint_with_cs(0.01, 1e-4, 2e-4, 2.0, 0.05)
        j.damage = 1.0
        assert j.check_failure()
        assert j.failed

    def test_already_failed_stays_failed(self):
        j = self._joint_with_cs(0.01, 1e-4, 2e-4, 2.0, 0.05)
        j.failed = True
        assert j.check_failure()

    def test_euler_buckling_limit_property(self):
        E, I, L = AL_7075_T6.elastic_modulus, 1e-4, 2.0
        j   = self._joint_with_cs(0.01, I, 2e-4, L, 0.05)
        F_cr_expected = euler_buckling_limit(E, I, L)
        assert abs(j.euler_buckling_limit - F_cr_expected) < 1.0

    def test_shear_yield_limit_property(self):
        tau_y, A = AL_7075_T6.shear_yield_strength, 0.01
        j        = self._joint_with_cs(A, 1e-4, 2e-4, 2.0, 0.05)
        V_max    = shear_yield_limit(tau_y, A)
        assert abs(j.shear_yield_limit - V_max) < 1.0

    def test_bending_failure_moment_property(self):
        sigma_y, I, c = AL_7075_T6.yield_strength, 1e-4, 0.05
        j  = self._joint_with_cs(0.01, I, 2e-4, 2.0, c)
        My = bending_failure_moment(sigma_y, I, c)
        assert abs(j.bending_failure_moment - My) < 1.0


# ---------------------------------------------------------------------------
# 5. evaluate_structural_failures
# ---------------------------------------------------------------------------

class TestEvaluateFailures:

    def test_no_failures_returns_empty(self):
        g = ComponentGraph()
        g.add_node(_make_node("P"))
        g.add_node(_make_node("C"))
        g.add_joint(_make_joint("J1", "P", "C"))
        # No loads applied → no failure
        failed = g.evaluate_structural_failures()
        assert len(failed) == 0

    def test_pre_failed_joint_returned(self):
        g = ComponentGraph()
        g.add_node(_make_node("P"))
        g.add_node(_make_node("C"))
        j = _make_joint("J1", "P", "C")
        j.failed = True
        g.add_joint(j)
        failed = g.evaluate_structural_failures()
        assert len(failed) == 1
        assert failed[0].joint_id == "J1"


# ---------------------------------------------------------------------------
# 6. Material properties validation
# ---------------------------------------------------------------------------

class TestMaterialProperties:

    def test_valid_material(self):
        m = AL_7075_T6
        assert m.elastic_modulus > 0.0

    def test_zero_elastic_modulus_raises(self):
        with pytest.raises(ValueError, match="elastic_modulus"):
            MaterialProperties(
                elastic_modulus=0.0,
                shear_modulus=26.9e9,
                yield_strength=503e6,
                shear_yield_strength=290e6,
                ultimate_strength=572e6,
                density=2810.0,
            )

    def test_negative_density_raises(self):
        with pytest.raises(ValueError, match="density"):
            MaterialProperties(
                elastic_modulus=71.7e9,
                shear_modulus=26.9e9,
                yield_strength=503e6,
                shear_yield_strength=290e6,
                ultimate_strength=572e6,
                density=-1.0,
            )


# ---------------------------------------------------------------------------
# 7. JointCrossSection validation
# ---------------------------------------------------------------------------

class TestJointCrossSection:

    def test_valid_cross_section(self):
        cs = _make_cross_section()
        assert cs.area > 0.0

    def test_zero_area_raises(self):
        with pytest.raises(ValueError, match="area"):
            JointCrossSection(area=0.0, second_moment=1e-4,
                              torsion_constant=2e-4, effective_length=2.0,
                              distance_to_extreme_fibre=0.05)

    def test_zero_effective_length_raises(self):
        with pytest.raises(ValueError, match="effective_length"):
            JointCrossSection(area=0.01, second_moment=1e-4,
                              torsion_constant=2e-4, effective_length=0.0,
                              distance_to_extreme_fibre=0.05)


# ---------------------------------------------------------------------------
# 8. Summary string
# ---------------------------------------------------------------------------

class TestSummaryString:

    def test_summary_contains_node_count(self):
        g = ComponentGraph()
        g.add_node(_make_node("A"))
        g.add_node(_make_node("B"))
        s = g.summary()
        assert "2/2" in s

    def test_repr_is_summary(self):
        g = ComponentGraph()
        assert repr(g) == g.summary()

    def test_summary_after_jettison(self):
        g = ComponentGraph()
        g.add_node(_make_node("A"))
        g.add_node(_make_node("B"))
        g.add_joint(_make_joint("J", "A", "B"))
        g.jettison_subtree("B")
        s = g.summary()
        assert "1/2" in s
