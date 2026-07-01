"""
nova.collision.detector
========================
Spatial collision detection engine for Project NOVA.

Architectural role
------------------
Phase 9 — Collision Detection.
Pipeline stage: Stage 5. Receives the new VehicleState produced by the RK4
integrator and checks whether the vehicle's bounding volume intersects a
terrain mesh. Returns a ContactEvent if a collision is detected, or None if
the vehicle is clear.

I/O contract
------------
Input  : VehicleState (position_eci), TerrainMesh (triangle soup),
         vehicle bounding sphere radius [m]
Output : ContactEvent (frozen dataclass) | None

Algorithm
---------
Two-phase detection:

1. Broad phase — Axis-Aligned Bounding Box (AABB) BVH:
   The terrain mesh is partitioned into a binary BVH tree whose leaves each
   hold a small number of triangles. At query time the vehicle's bounding
   sphere is tested against node AABBs top-down; branches whose AABBs do not
   overlap the sphere are culled. This reduces the narrow-phase triangle set
   from O(N) to O(log N) on average.

2. Narrow phase — Möller-Trumbore ray-triangle intersection:
   The vehicle's velocity direction is cast as a ray from its position. Each
   candidate triangle from the BVH is tested using the Möller-Trumbore
   algorithm (1997), which operates entirely in barycentric coordinates and
   requires no precomputed plane equations. A hit is confirmed when:
     a. The intersection parameter t ∈ [0, look-ahead distance]
     b. Barycentric coordinates (u, v) satisfy u ≥ 0, v ≥ 0, u + v ≤ 1

   Additionally, sphere-triangle intersection is tested for the current
   position (zero-velocity case / grazing contact).

All geometry is in a consistent local Cartesian frame. For this simulation
the terrain mesh is expressed in the ECI frame; the caller is responsible for
frame alignment before passing the mesh.

Numerical constants
-------------------
EPSILON : float = 1e-9
    Near-zero threshold for Möller-Trumbore determinant check (parallel-ray
    guard). Chosen to be well above float64 machine epsilon (~2.2e-16) but
    small enough to catch all non-parallel rays.

References
----------
- Möller, T. & Trumbore, B. "Fast, Minimum Storage Ray/Triangle
  Intersection." J. Graphics Tools 2(1):21-28, 1997.
- Ericson, C. "Real-Time Collision Detection." Morgan Kaufmann, 2005,
  Chapter 6 (BVH) and Chapter 5 (sphere-triangle).
- Shirley, P. & Morley, R.K. "Realistic Ray Tracing." AK Peters, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from nova.core.constants import EARTH_RADIUS_EQ
from nova.core.state_vector import VehicleState

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

_MT_EPSILON: float = 1.0e-9   # Möller-Trumbore parallel-ray guard
_SPHERE_EPSILON: float = 1.0e-9  # Sphere-triangle near-contact tolerance

# ---------------------------------------------------------------------------
# Triangle primitive
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Triangle:
    """
    A single triangle in the terrain mesh.

    Attributes
    ----------
    v0, v1, v2 : ndarray, shape (3,), dtype float64
        Vertices in a consistent Cartesian frame [m].
    normal : ndarray, shape (3,), dtype float64
        Outward-facing unit normal. Computed automatically if not supplied;
        direction follows the right-hand rule from (v1-v0) × (v2-v0).
    triangle_id : int
        Optional integer identifier for diagnostics. Default -1.
    """

    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    normal: np.ndarray
    triangle_id: int = -1

    def __post_init__(self) -> None:
        for attr in ("v0", "v1", "v2", "normal"):
            arr = np.asarray(getattr(self, attr), dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError(
                    f"Triangle.{attr} must have shape (3,); got {arr.shape}"
                )
            object.__setattr__(self, attr, arr)
        tid = int(self.triangle_id)
        object.__setattr__(self, "triangle_id", tid)

    @property
    def centroid(self) -> np.ndarray:
        """Centroid of the triangle [m]."""
        return (self.v0 + self.v1 + self.v2) / 3.0

    @property
    def aabb(self) -> "AABB":
        """Tight axis-aligned bounding box of this triangle."""
        mins = np.minimum(self.v0, np.minimum(self.v1, self.v2))
        maxs = np.maximum(self.v0, np.maximum(self.v1, self.v2))
        return AABB(mins, maxs)


def make_triangle(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    triangle_id: int = -1,
) -> Triangle:
    """
    Construct a Triangle, computing the outward normal automatically.

    The normal direction follows the right-hand rule: (v1−v0) × (v2−v0).
    If the triangle is degenerate (zero area), the normal defaults to
    [0, 0, 1].

    Parameters
    ----------
    v0, v1, v2 : array_like, shape (3,)
        Triangle vertices [m].
    triangle_id : int
        Optional identifier. Default -1.

    Returns
    -------
    Triangle
    """
    a = np.asarray(v0, dtype=np.float64)
    b = np.asarray(v1, dtype=np.float64)
    c = np.asarray(v2, dtype=np.float64)
    edge1 = b - a
    edge2 = c - a
    n = np.cross(edge1, edge2)
    n_len = float(np.linalg.norm(n))
    if n_len < 1.0e-15:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = n / n_len
    return Triangle(v0=a, v1=b, v2=c, normal=n, triangle_id=triangle_id)


# ---------------------------------------------------------------------------
# AABB — Axis-Aligned Bounding Box
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AABB:
    """
    Axis-aligned bounding box defined by its minimum and maximum corners.

    Attributes
    ----------
    min_corner : ndarray, shape (3,), dtype float64
        Minimum x, y, z coordinates [m].
    max_corner : ndarray, shape (3,), dtype float64
        Maximum x, y, z coordinates [m].
    """

    min_corner: np.ndarray
    max_corner: np.ndarray

    def __post_init__(self) -> None:
        mn = np.asarray(self.min_corner, dtype=np.float64)
        mx = np.asarray(self.max_corner, dtype=np.float64)
        if mn.shape != (3,) or mx.shape != (3,):
            raise ValueError("AABB corners must have shape (3,)")
        if np.any(mn > mx + _SPHERE_EPSILON):
            raise ValueError(
                f"AABB min_corner must be ≤ max_corner; "
                f"got min={mn}, max={mx}"
            )
        object.__setattr__(self, "min_corner", mn)
        object.__setattr__(self, "max_corner", mx)

    @property
    def center(self) -> np.ndarray:
        """Geometric centre of the AABB [m]."""
        return 0.5 * (self.min_corner + self.max_corner)

    @property
    def half_extents(self) -> np.ndarray:
        """Half-size along each axis [m]."""
        return 0.5 * (self.max_corner - self.min_corner)

    def overlaps_sphere(self, center: np.ndarray, radius: float) -> bool:
        """
        Test whether this AABB overlaps a sphere.

        Uses the nearest-point-on-AABB method: the squared distance from the
        sphere centre to the closest point on the AABB is compared against
        radius².

        Parameters
        ----------
        center : ndarray, shape (3,)
            Sphere centre [m].
        radius : float
            Sphere radius [m]. Must be non-negative.

        Returns
        -------
        bool
        """
        # Clamp sphere centre to AABB extents
        closest = np.clip(center, self.min_corner, self.max_corner)
        diff = center - closest
        dist_sq = float(np.dot(diff, diff))
        return dist_sq <= radius * radius + _SPHERE_EPSILON

    def expanded(self, margin: float) -> "AABB":
        """Return a new AABB expanded by *margin* on all sides."""
        m = float(margin)
        return AABB(self.min_corner - m, self.max_corner + m)

    @staticmethod
    def union(a: "AABB", b: "AABB") -> "AABB":
        """Return the smallest AABB that contains both *a* and *b*."""
        return AABB(
            np.minimum(a.min_corner, b.min_corner),
            np.maximum(a.max_corner, b.max_corner),
        )


def _aabb_of_triangles(triangles: List[Triangle]) -> AABB:
    """Compute the tight AABB enclosing all vertices of *triangles*."""
    if not triangles:
        raise ValueError("Cannot compute AABB of empty triangle list")
    vertices = []
    for tri in triangles:
        vertices.extend([tri.v0, tri.v1, tri.v2])
    verts = np.array(vertices, dtype=np.float64)
    return AABB(np.min(verts, axis=0), np.max(verts, axis=0))


# ---------------------------------------------------------------------------
# BVH node
# ---------------------------------------------------------------------------

@dataclass
class _BVHNode:
    """
    Internal node of the Bounding Volume Hierarchy.

    Leaf nodes have triangles and no children.
    Internal nodes have children and no triangles.
    """

    aabb: AABB
    triangles: List[Triangle] = field(default_factory=list)
    left: Optional["_BVHNode"] = None
    right: Optional["_BVHNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


_BVH_LEAF_SIZE: int = 4   # max triangles per leaf node


def _build_bvh(triangles: List[Triangle], depth: int = 0) -> _BVHNode:
    """
    Recursively build a BVH tree from a list of triangles.

    Splitting heuristic: median split along the longest AABB axis.
    Leaves are created when len(triangles) ≤ _BVH_LEAF_SIZE or max depth
    (64) is reached.

    Parameters
    ----------
    triangles : list[Triangle]
    depth : int
        Current recursion depth.

    Returns
    -------
    _BVHNode
    """
    aabb = _aabb_of_triangles(triangles)
    node = _BVHNode(aabb=aabb)

    if len(triangles) <= _BVH_LEAF_SIZE or depth >= 64:
        node.triangles = list(triangles)
        return node

    # Choose longest axis
    extents = aabb.max_corner - aabb.min_corner
    axis = int(np.argmax(extents))

    # Sort by centroid along chosen axis
    sorted_tris = sorted(triangles, key=lambda t: t.centroid[axis])
    mid = len(sorted_tris) // 2

    left_tris = sorted_tris[:mid]
    right_tris = sorted_tris[mid:]

    # Guard: if all centroids coincide, force leaf
    if not left_tris or not right_tris:
        node.triangles = list(triangles)
        return node

    node.left = _build_bvh(left_tris, depth + 1)
    node.right = _build_bvh(right_tris, depth + 1)
    return node


def _query_bvh(
    node: _BVHNode,
    sphere_center: np.ndarray,
    sphere_radius: float,
    result: List[Triangle],
) -> None:
    """
    Collect all triangles in the BVH whose AABB overlaps *sphere*.

    Results are appended to *result* (in-place).
    """
    if not node.aabb.overlaps_sphere(sphere_center, sphere_radius):
        return
    if node.is_leaf:
        result.extend(node.triangles)
        return
    if node.left is not None:
        _query_bvh(node.left, sphere_center, sphere_radius, result)
    if node.right is not None:
        _query_bvh(node.right, sphere_center, sphere_radius, result)


# ---------------------------------------------------------------------------
# Möller-Trumbore ray-triangle intersection
# ---------------------------------------------------------------------------

def _moller_trumbore(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    tri: Triangle,
    t_min: float = 0.0,
    t_max: float = math.inf,
) -> Optional[float]:
    """
    Möller-Trumbore ray-triangle intersection test.

    Tests whether the ray defined by (origin + t·direction) intersects *tri*
    for t ∈ [t_min, t_max].

    Parameters
    ----------
    ray_origin : ndarray, shape (3,)
        Ray origin [m].
    ray_dir : ndarray, shape (3,)
        Ray direction (need not be normalised).
    tri : Triangle
        Triangle to test against.
    t_min : float
        Minimum valid intersection parameter. Default 0.
    t_max : float
        Maximum valid intersection parameter. Default inf.

    Returns
    -------
    float | None
        Intersection parameter t if hit; None otherwise.
    """
    edge1 = tri.v1 - tri.v0
    edge2 = tri.v2 - tri.v0
    h = np.cross(ray_dir, edge2)
    det = float(np.dot(edge1, h))

    # Parallel ray check
    if abs(det) < _MT_EPSILON:
        return None

    inv_det = 1.0 / det
    s = ray_origin - tri.v0
    u = float(np.dot(s, h)) * inv_det
    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = float(np.dot(ray_dir, q)) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None

    t = float(np.dot(edge2, q)) * inv_det
    if t_min <= t <= t_max:
        return t
    return None


# ---------------------------------------------------------------------------
# Sphere-triangle closest-point test
# ---------------------------------------------------------------------------

def _sphere_intersects_triangle(
    center: np.ndarray,
    radius: float,
    tri: Triangle,
) -> bool:
    """
    Test whether a sphere overlaps a triangle using closest-point clamping.

    Computes the closest point on the triangle to the sphere centre using
    Ericson's barycentric clamp method, then checks distance against radius.

    Parameters
    ----------
    center : ndarray, shape (3,)
        Sphere centre [m].
    radius : float
        Sphere radius [m].
    tri : Triangle

    Returns
    -------
    bool
    """
    ab = tri.v1 - tri.v0
    ac = tri.v2 - tri.v0
    ap = center - tri.v0

    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))

    if d1 <= 0.0 and d2 <= 0.0:
        closest = tri.v0
    else:
        bp = center - tri.v1
        d3 = float(np.dot(ab, bp))
        d4 = float(np.dot(ac, bp))
        if d3 >= 0.0 and d4 <= d3:
            closest = tri.v1
        else:
            cp = center - tri.v2
            d5 = float(np.dot(ab, cp))
            d6 = float(np.dot(ac, cp))
            if d6 >= 0.0 and d5 <= d6:
                closest = tri.v2
            else:
                vc = d1 * d4 - d3 * d2
                if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    w = d1 / (d1 - d3)
                    closest = tri.v0 + w * ab
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        closest = tri.v0 + w * ac
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            closest = tri.v1 + w * (tri.v2 - tri.v1)
                        else:
                            denom = 1.0 / (vc + vb + va) if (vc + vb + va) != 0.0 else 0.0
                            t_b = vb * denom
                            t_c = vc * denom
                            closest = tri.v0 + ab * t_b + ac * t_c

    diff = center - closest
    dist_sq = float(np.dot(diff, diff))
    return dist_sq <= radius * radius + _SPHERE_EPSILON


# ---------------------------------------------------------------------------
# ContactEvent — collision result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContactEvent:
    """
    Immutable record of a detected collision between the vehicle and terrain.

    Attributes
    ----------
    contact_point : ndarray, shape (3,), dtype float64
        Estimated contact point in the simulation frame [m].
    contact_normal : ndarray, shape (3,), dtype float64
        Surface normal at the contact point (outward from terrain), unit vector.
    penetration_depth : float
        Estimated penetration depth [m]. Zero for ray-intersection detections
        (first contact); positive for sphere-overlap detections.
    triangle_id : int
        Identifier of the intersected triangle. -1 if unknown.
    vehicle_speed : float
        Vehicle speed at time of detection [m s⁻¹].
    mission_time : float
        Mission elapsed time at detection [s].
    detection_method : str
        "ray" for Möller-Trumbore hit, "sphere" for sphere-triangle overlap.
    """

    contact_point: np.ndarray
    contact_normal: np.ndarray
    penetration_depth: float
    triangle_id: int
    vehicle_speed: float
    mission_time: float
    detection_method: str

    def __post_init__(self) -> None:
        for attr in ("contact_point", "contact_normal"):
            arr = np.asarray(getattr(self, attr), dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError(
                    f"ContactEvent.{attr} must have shape (3,); got {arr.shape}"
                )
            object.__setattr__(self, attr, arr)

        pd = float(self.penetration_depth)
        if pd < 0.0:
            raise ValueError(
                f"penetration_depth must be non-negative; got {pd:.6g}"
            )
        object.__setattr__(self, "penetration_depth", pd)
        object.__setattr__(self, "triangle_id", int(self.triangle_id))
        object.__setattr__(self, "vehicle_speed", float(self.vehicle_speed))
        object.__setattr__(self, "mission_time", float(self.mission_time))
        object.__setattr__(self, "detection_method", str(self.detection_method))

    def __repr__(self) -> str:
        pt = self.contact_point
        return (
            f"ContactEvent(method={self.detection_method!r}, "
            f"t={self.mission_time:.3f}s, "
            f"depth={self.penetration_depth:.3f}m, "
            f"pt=[{pt[0]:.1f},{pt[1]:.1f},{pt[2]:.1f}])"
        )


# ---------------------------------------------------------------------------
# TerrainMesh — input mesh container
# ---------------------------------------------------------------------------

@dataclass
class TerrainMesh:
    """
    Triangle-soup terrain mesh with a pre-built BVH.

    Construct via :func:`build_terrain_mesh` to ensure the BVH is populated.

    Attributes
    ----------
    triangles : list[Triangle]
        All triangles in the mesh.
    bvh_root : _BVHNode | None
        Root of the BVH acceleration structure. None for empty meshes.
    """

    triangles: List[Triangle]
    bvh_root: Optional[_BVHNode] = None

    @property
    def triangle_count(self) -> int:
        """Total number of triangles in the mesh."""
        return len(self.triangles)

    @property
    def is_empty(self) -> bool:
        """True when the mesh has no triangles."""
        return len(self.triangles) == 0

    def __repr__(self) -> str:
        return f"TerrainMesh(triangles={self.triangle_count}, bvh={'yes' if self.bvh_root else 'no'})"


def build_terrain_mesh(triangles: List[Triangle]) -> TerrainMesh:
    """
    Construct a TerrainMesh and build its BVH acceleration structure.

    Parameters
    ----------
    triangles : list[Triangle]
        Triangle soup in a consistent Cartesian frame [m].

    Returns
    -------
    TerrainMesh
    """
    if not triangles:
        return TerrainMesh(triangles=[], bvh_root=None)
    bvh = _build_bvh(list(triangles))
    return TerrainMesh(triangles=list(triangles), bvh_root=bvh)


# ---------------------------------------------------------------------------
# CollisionDetector — main detection class
# ---------------------------------------------------------------------------

class CollisionDetector:
    """
    Pipeline Stage 5 collision detector.

    Wraps a :class:`TerrainMesh` and provides a single entry-point
    :meth:`check` that runs broad-phase BVH + narrow-phase
    Möller-Trumbore / sphere-triangle tests.

    Parameters
    ----------
    mesh : TerrainMesh
        Terrain mesh with pre-built BVH.
    vehicle_radius : float
        Bounding sphere radius of the vehicle [m]. Must be positive.
        Used for both broad-phase culling and sphere-triangle tests.
    look_ahead_factor : float
        Ray look-ahead distance multiplier. The ray length is set to
        ``look_ahead_factor × ‖velocity_eci‖ × dt`` where dt defaults to
        DEFAULT_DT. Default 2.0 (look two timesteps ahead).
    """

    def __init__(
        self,
        mesh: TerrainMesh,
        vehicle_radius: float,
        look_ahead_factor: float = 2.0,
    ) -> None:
        if not isinstance(mesh, TerrainMesh):
            raise TypeError("mesh must be a TerrainMesh")
        r = float(vehicle_radius)
        if r <= 0.0:
            raise ValueError(f"vehicle_radius must be positive; got {r:.6g}")
        laf = float(look_ahead_factor)
        if laf <= 0.0:
            raise ValueError(f"look_ahead_factor must be positive; got {laf:.6g}")

        self._mesh = mesh
        self._vehicle_radius = r
        self._look_ahead_factor = laf

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mesh(self) -> TerrainMesh:
        """The terrain mesh used for detection."""
        return self._mesh

    @property
    def vehicle_radius(self) -> float:
        """Vehicle bounding sphere radius [m]."""
        return self._vehicle_radius

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def check(
        self,
        state: VehicleState,
        look_ahead_distance: Optional[float] = None,
    ) -> Optional[ContactEvent]:
        """
        Check the current VehicleState for terrain collision.

        Two sub-tests are run in order:

        1. **Ray test**: casts a ray from position_eci in the direction of
           velocity_eci. Hit distance limited to look_ahead_distance.
        2. **Sphere test**: tests the vehicle's bounding sphere against all
           BVH-candidate triangles for current-tick overlap.

        The ray test is skipped when speed < 0.1 m/s (stationary vehicle).

        Parameters
        ----------
        state : VehicleState
            Current simulation state.
        look_ahead_distance : float | None
            Maximum ray intersection distance [m]. If None, defaults to
            ``look_ahead_factor × speed × DEFAULT_DT`` (with a 1 m minimum).

        Returns
        -------
        ContactEvent | None
            ContactEvent on first detected hit; None if clear.
        """
        if not isinstance(state, VehicleState):
            raise TypeError("state must be a VehicleState")
        if self._mesh.is_empty:
            return None

        pos = state.position_eci.astype(np.float64)
        vel = state.velocity_eci.astype(np.float64)
        speed = float(np.linalg.norm(vel))

        # Broad phase: gather candidate triangles within bounding sphere
        # Inflate sphere by look-ahead distance for the broad phase
        if look_ahead_distance is None:
            from nova.core.constants import DEFAULT_DT
            lad = max(1.0, self._look_ahead_factor * speed * DEFAULT_DT)
        else:
            lad = max(1.0, float(look_ahead_distance))

        broad_radius = self._vehicle_radius + lad
        candidates: List[Triangle] = []
        _query_bvh(self._mesh.bvh_root, pos, broad_radius, candidates)

        if not candidates:
            return None

        # Narrow phase 1: Möller-Trumbore ray test (skip if nearly stationary)
        if speed >= 0.1:
            ray_dir = vel / speed  # unit direction
            for tri in candidates:
                t_hit = _moller_trumbore(pos, ray_dir, tri, t_min=0.0, t_max=lad)
                if t_hit is not None:
                    contact_pt = pos + t_hit * ray_dir
                    return ContactEvent(
                        contact_point=contact_pt,
                        contact_normal=tri.normal.copy(),
                        penetration_depth=0.0,
                        triangle_id=tri.triangle_id,
                        vehicle_speed=speed,
                        mission_time=state.time,
                        detection_method="ray",
                    )

        # Narrow phase 2: sphere-triangle overlap test
        for tri in candidates:
            if _sphere_intersects_triangle(pos, self._vehicle_radius, tri):
                # Estimate penetration depth as (radius − dist_to_closest_pt)
                ab = tri.v1 - tri.v0
                ac = tri.v2 - tri.v0
                projected = pos - tri.v0
                # Use dot product with normal for signed distance to plane
                dist_to_plane = abs(float(np.dot(projected, tri.normal)))
                depth = max(0.0, self._vehicle_radius - dist_to_plane)
                # Contact point: project sphere centre onto triangle plane
                contact_pt = pos - dist_to_plane * tri.normal
                return ContactEvent(
                    contact_point=contact_pt,
                    contact_normal=tri.normal.copy(),
                    penetration_depth=depth,
                    triangle_id=tri.triangle_id,
                    vehicle_speed=speed,
                    mission_time=state.time,
                    detection_method="sphere",
                )

        return None

    def __repr__(self) -> str:
        return (
            f"CollisionDetector(triangles={self._mesh.triangle_count}, "
            f"radius={self._vehicle_radius:.2f}m)"
        )


# ---------------------------------------------------------------------------
# Convenience: flat terrain patch
# ---------------------------------------------------------------------------

def flat_terrain_patch(
    center: np.ndarray,
    half_width: float,
    altitude: float = 0.0,
    n_subdivisions: int = 1,
) -> TerrainMesh:
    """
    Build a flat square terrain patch as a triangle mesh.

    The patch is axis-aligned in the XY plane at z = *altitude*. Useful for
    unit tests and simple launch-site terrain.

    Parameters
    ----------
    center : array_like, shape (3,) [m]
        Centre of the patch (z component is overridden by *altitude*).
    half_width : float
        Half-size of the patch along X and Y [m]. Must be positive.
    altitude : float
        Z coordinate of the patch [m]. Default 0.
    n_subdivisions : int
        Number of grid divisions along each axis. Default 1 (2 triangles).
        Total triangles = 2 × n_subdivisions².

    Returns
    -------
    TerrainMesh
    """
    cx, cy = float(center[0]), float(center[1])
    hw = float(half_width)
    if hw <= 0.0:
        raise ValueError(f"half_width must be positive; got {hw:.6g}")
    n = max(1, int(n_subdivisions))
    z = float(altitude)

    triangles: List[Triangle] = []
    step = 2.0 * hw / n
    tid = 0

    for i in range(n):
        for j in range(n):
            x0 = cx - hw + i * step
            x1 = x0 + step
            y0 = cy - hw + j * step
            y1 = y0 + step

            # Two triangles per grid cell
            v00 = np.array([x0, y0, z])
            v10 = np.array([x1, y0, z])
            v01 = np.array([x0, y1, z])
            v11 = np.array([x1, y1, z])

            triangles.append(make_triangle(v00, v10, v01, triangle_id=tid))
            tid += 1
            triangles.append(make_triangle(v10, v11, v01, triangle_id=tid))
            tid += 1

    return build_terrain_mesh(triangles)
  
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

"""
tests/unit/test_detector.py
============================
Unit tests for Phase 9: nova.collision.detector

Test coverage
-------------
Triangle / make_triangle:
  1. Valid construction, dtype enforcement, field shapes
  2. Normal auto-computation via right-hand rule
  3. Degenerate triangle normal defaults to [0,0,1]
  4. Centroid property
  5. AABB property of triangle

AABB:
  6. Construction and frozen
  7. Validation rejects min > max
  8. overlaps_sphere: contained, touching, outside
  9. union of two AABBs
 10. expanded AABB

Möller-Trumbore (_moller_trumbore):
 11. Ray hits triangle centre
 12. Ray misses (outside barycentric bounds)
 13. Ray parallel to triangle returns None
 14. t outside [t_min, t_max] returns None
 15. Back-face hit (negative t) rejected by t_min=0
 16. Non-unit ray direction still gives correct hit point

Sphere-triangle (_sphere_intersects_triangle):
 17. Sphere centred on triangle vertex — hit
 18. Sphere centred above triangle — hit (within radius)
 19. Sphere far above triangle — miss
 20. Sphere beside triangle (outside) — miss

BVH (_build_bvh / _query_bvh):
 21. Single triangle leaf
 22. Multiple triangles split correctly
 23. Query returns correct candidates
 24. Query with far sphere returns empty

ContactEvent:
 25. Valid construction and frozen
 26. Wrong-shape contact_point rejected
 27. Negative penetration_depth rejected
 28. repr contains expected fields

TerrainMesh / build_terrain_mesh:
 29. Empty mesh construction
 30. Non-empty mesh has BVH
 31. triangle_count and is_empty properties

CollisionDetector construction:
 32. Valid construction
 33. Non-positive radius rejected
 34. Non-positive look-ahead factor rejected
 35. Wrong mesh type rejected

CollisionDetector.check — ray hits:
 36. Vehicle moving directly toward flat patch — ray hit
 37. Vehicle moving away from patch — no hit
 38. Vehicle moving parallel to patch — no hit
 39. Hit returns correct detection_method = "ray"
 40. Hit contact_point on the triangle plane
 41. mission_time recorded correctly
 42. vehicle_speed recorded correctly

CollisionDetector.check — sphere hits:
 43. Stationary vehicle overlapping terrain — sphere hit
 44. Sphere hit returns detection_method = "sphere"
 45. Sphere hit penetration_depth ≥ 0

CollisionDetector.check — edge cases:
 46. Empty mesh always returns None
 47. Vehicle high above terrain — no hit
 48. look_ahead_distance=0 equivalent forces sphere-only test
 49. Wrong state type raises TypeError

flat_terrain_patch:
 50. Returns TerrainMesh
 51. Default n_subdivisions=1 → 2 triangles
 52. n_subdivisions=2 → 8 triangles
 53. All normals point upward (+Z)
 54. Patch covers expected XY extent
 55. Zero half_width rejected
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from nova.collision.detector import (
    AABB,
    CollisionDetector,
    ContactEvent,
    TerrainMesh,
    Triangle,
    _BVHNode,
    _build_bvh,
    _moller_trumbore,
    _query_bvh,
    _sphere_intersects_triangle,
    build_terrain_mesh,
    flat_terrain_patch,
    make_triangle,
)
from nova.core.state_vector import make_state


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_triangle() -> Triangle:
    """Flat triangle in XY plane with vertices at (0,0,0),(1,0,0),(0,1,0)."""
    return make_triangle(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )


@pytest.fixture
def xy_patch() -> TerrainMesh:
    """1000 m × 1000 m flat patch centred at origin in XY plane, z=0."""
    return flat_terrain_patch(np.zeros(3), half_width=500.0, altitude=0.0)


@pytest.fixture
def state_above_patch() -> "VehicleState":
    """Vehicle 100 m above origin, moving downward at 200 m/s."""
    return make_state(
        position_eci=np.array([0.0, 0.0, 100.0]),
        velocity_eci=np.array([0.0, 0.0, -200.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body=np.zeros(3),
        mass=1000.0,
        time=5.0,
    )


@pytest.fixture
def detector(xy_patch) -> CollisionDetector:
    return CollisionDetector(xy_patch, vehicle_radius=2.0)


# ===========================================================================
# Triangle / make_triangle
# ===========================================================================

class TestTriangleConstruction:

    def test_basic_fields(self, unit_triangle):
        assert unit_triangle.v0.shape == (3,)
        assert unit_triangle.v1.shape == (3,)
        assert unit_triangle.v2.shape == (3,)
        assert unit_triangle.normal.shape == (3,)

    def test_dtype_float64(self, unit_triangle):
        assert unit_triangle.v0.dtype == np.float64
        assert unit_triangle.normal.dtype == np.float64

    def test_frozen(self, unit_triangle):
        with pytest.raises(Exception):
            unit_triangle.v0 = np.zeros(3)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError):
            Triangle(
                v0=np.zeros(2),
                v1=np.zeros(3),
                v2=np.zeros(3),
                normal=np.zeros(3),
            )

    def test_normal_unit_length(self, unit_triangle):
        assert abs(float(np.linalg.norm(unit_triangle.normal)) - 1.0) < 1e-12

    def test_normal_points_upward_for_xy_triangle(self, unit_triangle):
        """(v1-v0) × (v2-v0) for XY triangle points in +Z."""
        assert unit_triangle.normal[2] == pytest.approx(1.0, abs=1e-12)
        assert abs(unit_triangle.normal[0]) < 1e-12
        assert abs(unit_triangle.normal[1]) < 1e-12

    def test_degenerate_triangle_normal_default(self):
        """Zero-area triangle → normal defaults to [0,0,1]."""
        p = np.array([1.0, 0.0, 0.0])
        tri = make_triangle(p, p, p)
        assert tri.normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)

    def test_centroid(self):
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([3.0, 0.0, 0.0])
        v2 = np.array([0.0, 3.0, 0.0])
        tri = make_triangle(v0, v1, v2)
        expected = np.array([1.0, 1.0, 0.0])
        assert tri.centroid == pytest.approx(expected, abs=1e-12)

    def test_aabb_tight(self):
        tri = make_triangle(
            np.array([-1.0, -2.0, 0.0]),
            np.array([3.0, 0.0, 0.0]),
            np.array([0.0, 4.0, 5.0]),
        )
        aabb = tri.aabb
        assert aabb.min_corner == pytest.approx([-1.0, -2.0, 0.0])
        assert aabb.max_corner == pytest.approx([3.0, 4.0, 5.0])

    def test_triangle_id_stored(self):
        tri = make_triangle(np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                            triangle_id=42)
        assert tri.triangle_id == 42


# ===========================================================================
# AABB
# ===========================================================================

class TestAABB:

    def test_valid_construction(self):
        aabb = AABB(np.zeros(3), np.ones(3))
        assert aabb.center == pytest.approx([0.5, 0.5, 0.5])

    def test_frozen(self):
        aabb = AABB(np.zeros(3), np.ones(3))
        with pytest.raises(Exception):
            aabb.min_corner = np.ones(3)

    def test_min_exceeds_max_rejected(self):
        with pytest.raises(ValueError):
            AABB(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0]))

    def test_half_extents(self):
        aabb = AABB(np.zeros(3), np.array([4.0, 6.0, 8.0]))
        assert aabb.half_extents == pytest.approx([2.0, 3.0, 4.0])

    def test_sphere_inside_overlaps(self):
        aabb = AABB(np.zeros(3), np.ones(3) * 10.0)
        assert aabb.overlaps_sphere(np.array([5.0, 5.0, 5.0]), 0.5) is True

    def test_sphere_outside_no_overlap(self):
        aabb = AABB(np.zeros(3), np.ones(3))
        assert aabb.overlaps_sphere(np.array([5.0, 5.0, 5.0]), 0.5) is False

    def test_sphere_touching_face(self):
        """Sphere tangent to a face should overlap."""
        aabb = AABB(np.zeros(3), np.ones(3))
        # sphere at (1.5, 0.5, 0.5) with radius 0.5 → touches face at x=1
        assert aabb.overlaps_sphere(np.array([1.5, 0.5, 0.5]), 0.5) is True

    def test_sphere_just_outside_corner(self):
        aabb = AABB(np.zeros(3), np.ones(3))
        # corner at (1,1,1); sphere at (2,2,2) with radius < sqrt(3)
        dist = math.sqrt(3.0)
        assert aabb.overlaps_sphere(np.array([2.0, 2.0, 2.0]), dist - 0.1) is False

    def test_union(self):
        a = AABB(np.zeros(3), np.ones(3))
        b = AABB(np.array([0.5, 0.5, 0.5]), np.array([2.0, 2.0, 2.0]))
        u = AABB.union(a, b)
        assert u.min_corner == pytest.approx([0.0, 0.0, 0.0])
        assert u.max_corner == pytest.approx([2.0, 2.0, 2.0])

    def test_expanded(self):
        aabb = AABB(np.ones(3), np.ones(3) * 3.0)
        exp = aabb.expanded(1.0)
        assert exp.min_corner == pytest.approx([0.0, 0.0, 0.0])
        assert exp.max_corner == pytest.approx([4.0, 4.0, 4.0])


# ===========================================================================
# Möller-Trumbore
# ===========================================================================

class TestMollerTrumbore:

    def test_ray_hits_centre_of_triangle(self, unit_triangle):
        """Ray from above hits centroid of unit_triangle."""
        origin = np.array([0.3, 0.3, 10.0])
        direction = np.array([0.0, 0.0, -1.0])
        t = _moller_trumbore(origin, direction, unit_triangle)
        assert t is not None
        assert t == pytest.approx(10.0, rel=1e-9)

    def test_hit_point_on_plane(self, unit_triangle):
        origin = np.array([0.2, 0.2, 5.0])
        direction = np.array([0.0, 0.0, -1.0])
        t = _moller_trumbore(origin, direction, unit_triangle)
        assert t is not None
        pt = origin + t * direction
        assert pt[2] == pytest.approx(0.0, abs=1e-9)

    def test_ray_misses_outside_triangle(self, unit_triangle):
        """Ray aimed at (0.8, 0.8, 0) which is outside the triangle."""
        origin = np.array([0.8, 0.8, 5.0])
        direction = np.array([0.0, 0.0, -1.0])
        t = _moller_trumbore(origin, direction, unit_triangle)
        assert t is None

    def test_ray_parallel_to_triangle(self, unit_triangle):
        """Horizontal ray parallel to the XY triangle returns None."""
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([1.0, 0.0, 0.0])
        t = _moller_trumbore(origin, direction, unit_triangle)
        assert t is None

    def test_t_outside_t_max_rejected(self, unit_triangle):
        origin = np.array([0.3, 0.3, 10.0])
        direction = np.array([0.0, 0.0, -1.0])
        t = _moller_trumbore(origin, direction, unit_triangle, t_min=0.0, t_max=5.0)
        assert t is None  # t=10 > t_max=5

    def test_t_outside_t_min_rejected(self, unit_triangle):
        """Ray shooting away (negative t) is rejected by t_min=0."""
        origin = np.array([0.3, 0.3, -5.0])
        direction = np.array([0.0, 0.0, -1.0])
        t = _moller_trumbore(origin, direction, unit_triangle, t_min=0.0)
        assert t is None

    def test_non_unit_direction_gives_correct_t(self, unit_triangle):
        """Non-unit direction: t encodes parameterisation along that direction."""
        origin = np.array([0.3, 0.3, 20.0])
        # Direction with magnitude 2 → t should be 10 (20/2)
        direction = np.array([0.0, 0.0, -2.0])
        t = _moller_trumbore(origin, direction, unit_triangle)
        assert t is not None
        pt = origin + t * direction
        assert pt[2] == pytest.approx(0.0, abs=1e-9)

    def test_edge_hit(self):
        """Ray aimed exactly at edge midpoint."""
        tri = make_triangle(
            np.array([0.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
        )
        origin = np.array([1.0, 0.0, 5.0])
        direction = np.array([0.0, 0.0, -1.0])
        t = _moller_trumbore(origin, direction, tri)
        assert t is not None
        assert t == pytest.approx(5.0, rel=1e-9)

    def test_vertex_hit(self):
        """Ray aimed at v0."""
        tri = make_triangle(
            np.array([0.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
        )
        origin = np.array([0.0, 0.0, 3.0])
        direction = np.array([0.0, 0.0, -1.0])
        t = _moller_trumbore(origin, direction, tri)
        assert t is not None
        assert t == pytest.approx(3.0, rel=1e-9)


# ===========================================================================
# Sphere-triangle intersection
# ===========================================================================

class TestSphereTriangle:

    def test_sphere_at_vertex_hits(self, unit_triangle):
        """Sphere centred exactly on v0 with small radius — hits."""
        assert _sphere_intersects_triangle(
            unit_triangle.v0.copy(), 0.1, unit_triangle
        ) is True

    def test_sphere_directly_above_centre_hits(self, unit_triangle):
        """Sphere directly above centroid within radius — hits."""
        centroid = unit_triangle.centroid
        center = centroid + np.array([0.0, 0.0, 0.05])
        assert _sphere_intersects_triangle(center, 0.1, unit_triangle) is True

    def test_sphere_far_above_misses(self, unit_triangle):
        centroid = unit_triangle.centroid
        center = centroid + np.array([0.0, 0.0, 10.0])
        assert _sphere_intersects_triangle(center, 1.0, unit_triangle) is False

    def test_sphere_beside_triangle_misses(self, unit_triangle):
        """Sphere far to the side, outside triangle extents."""
        center = np.array([5.0, 5.0, 0.0])
        assert _sphere_intersects_triangle(center, 0.5, unit_triangle) is False

    def test_sphere_touching_edge(self):
        """Sphere tangent to the midpoint of edge v0v1."""
        tri = make_triangle(
            np.array([0.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 0.0]),
        )
        # Edge midpoint: (1, 0, 0); sphere centred at (1, -0.5, 0) radius 0.5
        center = np.array([1.0, -0.5, 0.0])
        assert _sphere_intersects_triangle(center, 0.5, tri) is True


# ===========================================================================
# BVH
# ===========================================================================

class TestBVH:

    def test_single_triangle_leaf(self, unit_triangle):
        root = _build_bvh([unit_triangle])
        assert root.is_leaf
        assert len(root.triangles) == 1

    def test_multiple_triangles_split(self):
        """10 triangles in a row should split into a tree, not a leaf."""
        tris = []
        for i in range(10):
            tris.append(make_triangle(
                np.array([float(i), 0.0, 0.0]),
                np.array([float(i) + 1.0, 0.0, 0.0]),
                np.array([float(i) + 0.5, 1.0, 0.0]),
                triangle_id=i,
            ))
        root = _build_bvh(tris)
        assert not root.is_leaf or len(root.triangles) <= 4

    def test_query_hits_nearby_triangle(self, unit_triangle):
        root = _build_bvh([unit_triangle])
        result: list = []
        _query_bvh(root, np.array([0.5, 0.3, 0.0]), 0.5, result)
        assert len(result) == 1

    def test_query_misses_far_sphere(self, unit_triangle):
        root = _build_bvh([unit_triangle])
        result: list = []
        _query_bvh(root, np.array([100.0, 100.0, 0.0]), 0.5, result)
        assert len(result) == 0

    def test_aabb_covers_all_triangles(self):
        tris = [
            make_triangle(np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.])),
            make_triangle(np.array([5., 5., 0.]), np.array([6., 5., 0.]), np.array([5., 6., 0.])),
        ]
        root = _build_bvh(tris)
        assert root.aabb.min_corner[0] == pytest.approx(0.0)
        assert root.aabb.max_corner[0] == pytest.approx(6.0)


# ===========================================================================
# ContactEvent
# ===========================================================================

class TestContactEvent:

    def test_valid_construction(self):
        evt = ContactEvent(
            contact_point=np.array([1.0, 2.0, 0.0]),
            contact_normal=np.array([0.0, 0.0, 1.0]),
            penetration_depth=0.5,
            triangle_id=7,
            vehicle_speed=100.0,
            mission_time=42.0,
            detection_method="ray",
        )
        assert evt.penetration_depth == pytest.approx(0.5)
        assert evt.triangle_id == 7
        assert evt.detection_method == "ray"

    def test_frozen(self):
        evt = ContactEvent(
            contact_point=np.zeros(3),
            contact_normal=np.array([0., 0., 1.]),
            penetration_depth=0.0,
            triangle_id=0,
            vehicle_speed=0.0,
            mission_time=0.0,
            detection_method="ray",
        )
        with pytest.raises(Exception):
            evt.penetration_depth = 1.0

    def test_wrong_shape_contact_point_rejected(self):
        with pytest.raises(ValueError, match="contact_point"):
            ContactEvent(
                contact_point=np.zeros(2),
                contact_normal=np.zeros(3),
                penetration_depth=0.0,
                triangle_id=0,
                vehicle_speed=0.0,
                mission_time=0.0,
                detection_method="sphere",
            )

    def test_negative_penetration_rejected(self):
        with pytest.raises(ValueError, match="penetration_depth"):
            ContactEvent(
                contact_point=np.zeros(3),
                contact_normal=np.zeros(3),
                penetration_depth=-0.1,
                triangle_id=0,
                vehicle_speed=0.0,
                mission_time=0.0,
                detection_method="ray",
            )

    def test_repr_contains_expected_fields(self):
        evt = ContactEvent(
            contact_point=np.array([1.0, 2.0, 3.0]),
            contact_normal=np.array([0., 0., 1.]),
            penetration_depth=0.0,
            triangle_id=3,
            vehicle_speed=150.0,
            mission_time=7.5,
            detection_method="ray",
        )
        r = repr(evt)
        assert "ContactEvent" in r
        assert "ray" in r
        assert "7.500" in r


# ===========================================================================
# TerrainMesh / build_terrain_mesh
# ===========================================================================

class TestTerrainMesh:

    def test_empty_mesh(self):
        mesh = build_terrain_mesh([])
        assert mesh.is_empty
        assert mesh.triangle_count == 0
        assert mesh.bvh_root is None

    def test_non_empty_has_bvh(self, unit_triangle):
        mesh = build_terrain_mesh([unit_triangle])
        assert not mesh.is_empty
        assert mesh.bvh_root is not None

    def test_triangle_count(self, unit_triangle):
        mesh = build_terrain_mesh([unit_triangle, unit_triangle])
        assert mesh.triangle_count == 2

    def test_repr(self, unit_triangle):
        mesh = build_terrain_mesh([unit_triangle])
        r = repr(mesh)
        assert "TerrainMesh" in r


# ===========================================================================
# CollisionDetector construction
# ===========================================================================

class TestCollisionDetectorConstruction:

    def test_valid_construction(self, xy_patch):
        det = CollisionDetector(xy_patch, vehicle_radius=5.0)
        assert det.vehicle_radius == pytest.approx(5.0)

    def test_repr(self, xy_patch):
        det = CollisionDetector(xy_patch, vehicle_radius=3.0)
        r = repr(det)
        assert "CollisionDetector" in r

    def test_zero_radius_rejected(self, xy_patch):
        with pytest.raises(ValueError, match="vehicle_radius"):
            CollisionDetector(xy_patch, vehicle_radius=0.0)

    def test_negative_radius_rejected(self, xy_patch):
        with pytest.raises(ValueError, match="vehicle_radius"):
            CollisionDetector(xy_patch, vehicle_radius=-1.0)

    def test_negative_look_ahead_factor_rejected(self, xy_patch):
        with pytest.raises(ValueError, match="look_ahead_factor"):
            CollisionDetector(xy_patch, vehicle_radius=1.0, look_ahead_factor=-1.0)

    def test_wrong_mesh_type_rejected(self):
        with pytest.raises(TypeError):
            CollisionDetector("not_a_mesh", vehicle_radius=1.0)


# ===========================================================================
# CollisionDetector.check — ray detections
# ===========================================================================

class TestCollisionDetectorRay:

    def test_vehicle_moving_toward_patch_detects_hit(
        self, detector, state_above_patch
    ):
        event = detector.check(state_above_patch, look_ahead_distance=500.0)
        assert event is not None

    def test_detection_method_is_ray(self, detector, state_above_patch):
        event = detector.check(state_above_patch, look_ahead_distance=500.0)
        assert event is not None
        assert event.detection_method == "ray"

    def test_contact_point_on_ground_plane(self, detector, state_above_patch):
        """Contact point z-coordinate should be ≈ 0 (terrain altitude)."""
        event = detector.check(state_above_patch, look_ahead_distance=500.0)
        assert event is not None
        assert event.contact_point[2] == pytest.approx(0.0, abs=0.1)

    def test_mission_time_recorded(self, detector, state_above_patch):
        event = detector.check(state_above_patch, look_ahead_distance=500.0)
        assert event is not None
        assert event.mission_time == pytest.approx(state_above_patch.time)

    def test_vehicle_speed_recorded(self, detector, state_above_patch):
        event = detector.check(state_above_patch, look_ahead_distance=500.0)
        assert event is not None
        expected_speed = float(np.linalg.norm(state_above_patch.velocity_eci))
        assert event.vehicle_speed == pytest.approx(expected_speed, rel=1e-9)

    def test_vehicle_moving_away_no_hit(self, xy_patch):
        """Vehicle below the terrain moving further down — no hit upward."""
        state = make_state(
            position_eci=np.array([0.0, 0.0, 100.0]),
            velocity_eci=np.array([0.0, 0.0, 200.0]),   # moving UP
            quaternion=np.array([1., 0., 0., 0.]),
            omega_body=np.zeros(3),
            mass=1000.0,
            time=0.0,
        )
        det = CollisionDetector(xy_patch, vehicle_radius=0.5)
        event = det.check(state, look_ahead_distance=50.0)
        assert event is None

    def test_vehicle_moving_parallel_no_ray_hit(self, xy_patch):
        """Horizontal motion above the patch — ray misses the XY plane."""
        state = make_state(
            position_eci=np.array([0.0, 0.0, 100.0]),
            velocity_eci=np.array([300.0, 0.0, 0.0]),  # horizontal
            quaternion=np.array([1., 0., 0., 0.]),
            omega_body=np.zeros(3),
            mass=1000.0,
            time=0.0,
        )
        det = CollisionDetector(xy_patch, vehicle_radius=0.5)
        event = det.check(state, look_ahead_distance=10.0)
        # Ray is horizontal, cannot hit flat horizontal patch
        assert event is None or event.detection_method != "ray"

    def test_look_ahead_too_short_no_hit(self, detector):
        """Vehicle 100 m above patch but look-ahead only 0.01 m → no ray hit."""
        state = make_state(
            position_eci=np.array([0.0, 0.0, 100.0]),
            velocity_eci=np.array([0.0, 0.0, -200.0]),
            quaternion=np.array([1., 0., 0., 0.]),
            omega_body=np.zeros(3),
            mass=1000.0,
            time=0.0,
        )
        event = detector.check(state, look_ahead_distance=0.01)
        # With tiny look-ahead the ray doesn't reach the terrain
        # and sphere (radius=2) at z=100 doesn't overlap z=0 terrain
        assert event is None

    def test_oblique_approach_still_detects(self, xy_patch):
        """Diagonal approach toward the patch centre."""
        state = make_state(
            position_eci=np.array([-50.0, 0.0, 50.0]),
            velocity_eci=np.array([100.0, 0.0, -100.0]),
            quaternion=np.array([1., 0., 0., 0.]),
            omega_body=np.zeros(3),
            mass=1000.0,
            time=0.0,
        )
        det = CollisionDetector(xy_patch, vehicle_radius=1.0)
        event = det.check(state, look_ahead_distance=200.0)
        assert event is not None


# ===========================================================================
# CollisionDetector.check — sphere detections
# ===========================================================================

class TestCollisionDetectorSphere:

    def test_stationary_vehicle_overlapping_terrain(self, xy_patch):
        """Vehicle at z=1 with radius=2 → sphere overlaps z=0 terrain."""
        state = make_state(
            position_eci=np.array([0.0, 0.0, 1.0]),   # 1 m above terrain
            velocity_eci=np.zeros(3),                   # stationary
            quaternion=np.array([1., 0., 0., 0.]),
            omega_body=np.zeros(3),
            mass=1000.0,
            time=1.0,
        )
        det = CollisionDetector(xy_patch, vehicle_radius=2.0)
        event = det.check(state)
        assert event is not None

    def test_sphere_detection_method(self, xy_patch):
        state = make_state(
            position_eci=np.array([0.0, 0.0, 1.0]),
            velocity_eci=np.zeros(3),
            quaternion=np.array([1., 0., 0., 0.]),
            omega_body=np.zeros(3),
            mass=1000.0,
            time=0.0,
        )
        det = CollisionDetector(xy_patch, vehicle_radius=2.0)
        event = det.check(state)
        assert event is not None
        assert event.detection_method == "sphere"

    def test_sphere_penetration_depth_non_negative(self, xy_patch):
        state = make_state(
            position_eci=np.array([0.0, 0.0, 0.5]),
            velocity_eci=np.zeros(3),
            quaternion=np.array([1., 0., 0., 0.]),
            omega_body=np.zeros(3),
            mass=1000.0,
            time=0.0,
        )
        det = CollisionDetector(xy_patch, vehicle_radius=2.0)
        event = det.check(state)
        assert event is not None
        assert event.penetration_depth >= 0.0

    def test_sphere_just_above_terrain_no_hit(self, xy_patch):
        """Vehicle 10 m above with radius 0.5 → no sphere overlap."""
        state = make_state(
            position_eci=np.array([0.0, 0.0, 10.0]),
            velocity_eci=np.zeros(3),
            quaternion=np.array([1., 0., 0., 0.]),
            omega_body=np.zeros(3),
            mass=1000.0,
            time=0.0,
        )
        det = CollisionDetector(xy_patch, vehicle_radius=0.5)
        event = det.check(state)
        assert event is None


# ===========================================================================
# CollisionDetector.check — edge cases
# ===========================================================================

class TestCollisionDetectorEdgeCases:

    def test_empty_mesh_always_returns_none(self):
        empty = build_terrain_mesh([])
        det = CollisionDetector(empty, vehicle_radius=1.0)
        state = make_state(
            position_eci=np.zeros(3),
            velocity_eci=np.array([0., 0., -100.]),
            quaternion=np.array([1., 0., 0., 0.]),
            omega_body=np.zeros(3),
            mass=100.0,
            time=0.0,
        )
        assert det.check(state) is None

    def test_vehicle_high_above_terrain_no_hit(self, xy_patch):
        state = make_state(
            position_eci=np.array([0.0, 0.0, 1_000_000.0]),
            velocity_eci=np.array([0.0, 0.0, -100.0]),
            quaternion=np.array([1., 0., 0., 0.]),
            omega_body=np.zeros(3),
            mass=1000.0,
            time=0.0,
        )
        det = CollisionDetector(xy_patch, vehicle_radius=2.0)
        event = det.check(state, look_ahead_distance=100.0)
        assert event is None

    def test_wrong_state_type_raises(self, detector):
        with pytest.raises(TypeError):
            detector.check("not_a_state")

    def test_contact_normal_is_unit_vector(self, detector, state_above_patch):
        event = detector.check(state_above_patch, look_ahead_distance=500.0)
        assert event is not None
        norm = float(np.linalg.norm(event.contact_normal))
        assert abs(norm - 1.0) < 1e-9

    def test_multiple_ticks_same_result(self, detector, state_above_patch):
        """Deterministic: same state produces same ContactEvent."""
        e1 = detector.check(state_above_patch, look_ahead_distance=500.0)
        e2 = detector.check(state_above_patch, look_ahead_distance=500.0)
        assert (e1 is None) == (e2 is None)
        if e1 is not None and e2 is not None:
            assert e1.contact_point == pytest.approx(e2.contact_point, abs=1e-9)


# ===========================================================================
# flat_terrain_patch
# ===========================================================================

class TestFlatTerrainPatch:

    def test_returns_terrain_mesh(self):
        mesh = flat_terrain_patch(np.zeros(3), 500.0)
        assert isinstance(mesh, TerrainMesh)

    def test_default_one_subdivision_two_triangles(self):
        mesh = flat_terrain_patch(np.zeros(3), 500.0)
        assert mesh.triangle_count == 2

    def test_two_subdivisions_eight_triangles(self):
        mesh = flat_terrain_patch(np.zeros(3), 500.0, n_subdivisions=2)
        assert mesh.triangle_count == 8

    def test_three_subdivisions_eighteen_triangles(self):
        mesh = flat_terrain_patch(np.zeros(3), 500.0, n_subdivisions=3)
        assert mesh.triangle_count == 18

    def test_all_normals_point_upward(self):
        mesh = flat_terrain_patch(np.zeros(3), 500.0, n_subdivisions=2)
        for tri in mesh.triangles:
            assert tri.normal[2] == pytest.approx(1.0, abs=1e-9)

    def test_patch_at_specified_altitude(self):
        mesh = flat_terrain_patch(np.zeros(3), 100.0, altitude=500.0)
        for tri in mesh.triangles:
            for v in (tri.v0, tri.v1, tri.v2):
                assert v[2] == pytest.approx(500.0, abs=1e-9)

    def test_patch_covers_expected_extent(self):
        hw = 250.0
        mesh = flat_terrain_patch(np.zeros(3), hw, n_subdivisions=2)
        all_x = [v[0] for tri in mesh.triangles for v in (tri.v0, tri.v1, tri.v2)]
        all_y = [v[1] for tri in mesh.triangles for v in (tri.v0, tri.v1, tri.v2)]
        assert min(all_x) == pytest.approx(-hw, abs=1e-9)
        assert max(all_x) == pytest.approx(hw, abs=1e-9)
        assert min(all_y) == pytest.approx(-hw, abs=1e-9)
        assert max(all_y) == pytest.approx(hw, abs=1e-9)

    def test_zero_half_width_rejected(self):
        with pytest.raises(ValueError):
            flat_terrain_patch(np.zeros(3), 0.0)

    def test_has_bvh(self):
        mesh = flat_terrain_patch(np.zeros(3), 500.0)
        assert mesh.bvh_root is not None

    def test_custom_centre_offset(self):
        center = np.array([1000.0, 2000.0, 0.0])
        hw = 100.0
        mesh = flat_terrain_patch(center, hw)
        all_x = [v[0] for tri in mesh.triangles for v in (tri.v0, tri.v1, tri.v2)]
        assert min(all_x) == pytest.approx(1000.0 - hw, abs=1e-9)
        assert max(all_x) == pytest.approx(1000.0 + hw, abs=1e-9)
