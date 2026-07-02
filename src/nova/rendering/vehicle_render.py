"""
nova.rendering.vehicle_render
==============================
Close-up body-frame vehicle visualizer for Project NOVA.

Architectural role
------------------
Phase 12 — Rendering.
Pipeline stage: Stage 12 (Renderer). Manages the close-up, body-frame
3-D vehicle visualization: rocket body segments, engine nozzle geometry,
exhaust plume, and structural failure overlays. All geometry is produced
as data — no Pygame drawing calls are made here.

Design
------
The vehicle is represented as a hierarchical list of VehicleSegment objects,
each describing a body-frame cylinder, cone, or sphere. The renderer
projects these into screen space using the same perspective camera as the
celestial viewport but at body-frame scale.

Engine plume geometry is parameterised by throttle and Mach number:
  - Length: plume_length = plume_max_length × throttle × plume_mach_factor(M)
  - Width: tapers from nozzle_radius to ~0 at the tip

Structural failure overlays highlight any segment that has exceeded its
load margin. Failed segments are drawn with a distinct warning colour.

Key outputs (frozen dataclasses):
  SegmentGeometry  — 2-D projected outline for one body segment
  PlumeGeometry    — engine plume cone vertices
  VehicleScene     — complete body-frame render bundle

I/O contract
------------
Input  : RenderFrame (from Viewport), VehicleRenderConfig
Output : VehicleScene (frozen dataclass)

References
----------
- NOVA Engineering Handoff §12 Phase 12
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from nova.frames.transforms import T_ENU_to_body, T_body_to_ENU
from nova.rendering.viewport import RenderFrame

# ---------------------------------------------------------------------------
# Segment shape types
# ---------------------------------------------------------------------------

SHAPE_CYLINDER = "cylinder"
SHAPE_CONE = "cone"
SHAPE_SPHERE = "sphere"
SHAPE_NOSECONE = "nosecone"

_VALID_SHAPES = {SHAPE_CYLINDER, SHAPE_CONE, SHAPE_SPHERE, SHAPE_NOSECONE}


# ---------------------------------------------------------------------------
# VehicleSegmentConfig — one physical body segment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleSegmentConfig:
    """
    Geometric description of one vehicle body segment.

    Coordinates are in the body frame (+X forward, +Y right, +Z down).
    The segment's longitudinal axis runs along the body +X axis.

    Attributes
    ----------
    segment_id : str
        Unique identifier (e.g. "payload_fairing", "first_stage").
    shape : str
        One of: "cylinder", "cone", "nosecone", "sphere".
    x_start : float
        Forward extent of the segment in the body frame [m].
        For a rocket: x_start of the nose is the largest x value.
    x_end : float
        Aft extent [m]. Must satisfy x_end < x_start for cylinders/cones.
    radius_start : float
        Radius at x_start [m]. Must be ≥ 0.
    radius_end : float
        Radius at x_end [m]. Must be ≥ 0.
    color : tuple[int,int,int]
        RGB base colour for this segment.
    failure_color : tuple[int,int,int]
        RGB colour used when the segment is flagged as structurally failed.
    n_sides : int
        Number of polygon sides for rendering. Default 16.
    is_engine : bool
        True if this segment contains the engine nozzle (plume origin here).
        Default False.
    """

    segment_id: str
    shape: str
    x_start: float
    x_end: float
    radius_start: float
    radius_end: float
    color: Tuple[int, int, int]
    failure_color: Tuple[int, int, int] = (255, 60, 60)
    n_sides: int = 16
    is_engine: bool = False

    def __post_init__(self) -> None:
        if not self.segment_id.strip():
            raise ValueError("segment_id must be a non-empty string")
        if self.shape not in _VALID_SHAPES:
            raise ValueError(
                f"shape must be one of {sorted(_VALID_SHAPES)}; got '{self.shape}'"
            )
        if self.radius_start < 0.0:
            raise ValueError(f"radius_start must be ≥ 0; got {self.radius_start:.6g}")
        if self.radius_end < 0.0:
            raise ValueError(f"radius_end must be ≥ 0; got {self.radius_end:.6g}")
        if self.n_sides < 3:
            raise ValueError(f"n_sides must be ≥ 3; got {self.n_sides}")
        for rgb_attr in ("color", "failure_color"):
            r, g, b = getattr(self, rgb_attr)
            for ch, name in ((r, 'R'), (g, 'G'), (b, 'B')):
                if not (0 <= ch <= 255):
                    raise ValueError(f"{rgb_attr}.{name} must be in [0,255]; got {ch}")

    @property
    def length(self) -> float:
        """Segment length |x_start − x_end| [m]."""
        return abs(self.x_start - self.x_end)

    @property
    def mean_radius(self) -> float:
        """Mean radius (average of start and end radii) [m]."""
        return 0.5 * (self.radius_start + self.radius_end)


# ---------------------------------------------------------------------------
# VehicleRenderConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleRenderConfig:
    """
    Configuration for the body-frame vehicle renderer.

    Attributes
    ----------
    segments : list[VehicleSegmentConfig]
        Ordered list of body segments (nose first). Must be non-empty.
    camera_distance_m : float
        Camera distance from vehicle CoM in body frame [m]. Default 50.
    camera_azimuth_rad : float
        Camera azimuth in body frame [rad]. Default π/6 (30°).
    camera_elevation_rad : float
        Camera elevation [rad]. Default π/8 (22.5°).
    plume_max_length_m : float
        Maximum engine plume length at full throttle [m]. Default 20.
    plume_color : tuple[int,int,int]
        Engine plume colour. Default (255, 180, 60).
    plume_core_color : tuple[int,int,int]
        Inner plume / shock diamond colour. Default (255, 255, 200).
    failed_joint_ids : set[str]
        Set of segment IDs currently flagged as failed. Default empty.
    show_reference_axes : bool
        If True, draw body-frame reference axes (+X, +Y, +Z). Default True.
    """

    segments: List[VehicleSegmentConfig]
    camera_distance_m: float = 50.0
    camera_azimuth_rad: float = math.pi / 6.0
    camera_elevation_rad: float = math.pi / 8.0
    plume_max_length_m: float = 20.0
    plume_color: Tuple[int, int, int] = (255, 180, 60)
    plume_core_color: Tuple[int, int, int] = (255, 255, 200)
    failed_joint_ids: frozenset = field(default_factory=frozenset)
    show_reference_axes: bool = True

    def __post_init__(self) -> None:
        segs = list(self.segments)
        if not segs:
            raise ValueError("segments must be a non-empty list")
        for s in segs:
            if not isinstance(s, VehicleSegmentConfig):
                raise TypeError(f"All segments must be VehicleSegmentConfig; got {type(s)}")
        object.__setattr__(self, "segments", segs)

        if self.camera_distance_m <= 0.0:
            raise ValueError(
                f"camera_distance_m must be positive; got {self.camera_distance_m:.6g}"
            )
        if self.plume_max_length_m < 0.0:
            raise ValueError(
                f"plume_max_length_m must be ≥ 0; got {self.plume_max_length_m:.6g}"
            )
        object.__setattr__(self, "failed_joint_ids", frozenset(self.failed_joint_ids))


# ---------------------------------------------------------------------------
# Geometry output dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentGeometry:
    """
    Projected 2-D geometry for one body segment in screen space.

    Attributes
    ----------
    segment_id : str
        Segment identifier.
    outline_px : list[tuple[float, float]]
        Screen-space polygon vertices (x, y) in pixels. Empty if behind cam.
    color : tuple[int,int,int]
        Effective display colour (may be failure_color).
    is_failed : bool
        True if this segment is structurally failed.
    visible : bool
        True if any part of the segment projects into the screen.
    """

    segment_id: str
    outline_px: List[Tuple[float, float]]
    color: Tuple[int, int, int]
    is_failed: bool
    visible: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_id", str(self.segment_id))
        object.__setattr__(self, "is_failed", bool(self.is_failed))
        object.__setattr__(self, "visible", bool(self.visible))


@dataclass(frozen=True)
class PlumeGeometry:
    """
    Engine exhaust plume geometry in screen space.

    Attributes
    ----------
    active : bool
        True if the engine is firing (throttle > 0).
    length_m : float
        Plume length in body-frame metres.
    outline_px : list[tuple[float, float]]
        Outer plume cone polygon vertices in screen pixels.
    core_px : list[tuple[float, float]]
        Inner core / shock diamond polygon vertices.
    nozzle_pos_body : ndarray, shape (3,)
        Nozzle exit position in body frame [m].
    """

    active: bool
    length_m: float
    outline_px: List[Tuple[float, float]]
    core_px: List[Tuple[float, float]]
    nozzle_pos_body: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "active", bool(self.active))
        object.__setattr__(self, "length_m", float(self.length_m))
        arr = np.asarray(self.nozzle_pos_body, dtype=np.float64)
        object.__setattr__(self, "nozzle_pos_body", arr)


@dataclass(frozen=True)
class ReferenceAxis:
    """One reference axis line in screen space."""
    label: str
    start_px: Tuple[float, float]
    end_px: Tuple[float, float]
    color: Tuple[int, int, int]


@dataclass(frozen=True)
class VehicleScene:
    """
    Complete body-frame render geometry bundle for one display frame.

    Attributes
    ----------
    segments : list[SegmentGeometry]
        Projected geometry for every body segment.
    plume : PlumeGeometry
        Engine plume geometry.
    reference_axes : list[ReferenceAxis]
        Body-frame reference axis lines (if enabled).
    camera_pos_body : ndarray, shape (3,)
        Camera position in body frame [m].
    render_time : float
        Mission time [s] this scene was computed for.
    throttle : float
        Throttle value used to compute plume.
    any_structural_failure : bool
        True if any segment is flagged as failed.
    """

    segments: List[SegmentGeometry]
    plume: PlumeGeometry
    reference_axes: List[ReferenceAxis]
    camera_pos_body: np.ndarray
    render_time: float
    throttle: float
    any_structural_failure: bool

    def __post_init__(self) -> None:
        arr = np.asarray(self.camera_pos_body, dtype=np.float64)
        object.__setattr__(self, "camera_pos_body", arr)
        object.__setattr__(self, "render_time", float(self.render_time))
        object.__setattr__(self, "throttle", float(self.throttle))
        object.__setattr__(self, "any_structural_failure", bool(self.any_structural_failure))

    @property
    def visible_segments(self) -> List[SegmentGeometry]:
        """Return only the segments that project into screen space."""
        return [s for s in self.segments if s.visible]

    @property
    def failed_segments(self) -> List[SegmentGeometry]:
        """Return only the segments marked as structurally failed."""
        return [s for s in self.segments if s.is_failed]

    def __repr__(self) -> str:
        return (
            f"VehicleScene(t={self.render_time:.2f}s, "
            f"segments={len(self.segments)}, "
            f"visible={len(self.visible_segments)}, "
            f"plume_active={self.plume.active})"
        )


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------

def _body_camera_basis(azimuth: float, elevation: float) -> np.ndarray:
    """
    Compute a body-frame camera basis (same algorithm as celestial camera).

    Returns
    -------
    ndarray, shape (3, 3) — columns: [right, up, forward]
    """
    sa, ca = math.sin(azimuth), math.cos(azimuth)
    se, ce = math.sin(elevation), math.cos(elevation)
    forward = np.array([-ca * ce, -sa * ce, -se], dtype=np.float64)
    world_up = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # body +Z is down; up = -Z
    right = np.cross(forward, world_up)
    rn = float(np.linalg.norm(right))
    if rn < 1.0e-10:
        right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        right /= rn
    up = np.cross(right, forward)
    up /= float(np.linalg.norm(up))
    return np.column_stack([right, up, forward])


def _project_body(
    point_body: np.ndarray,
    cam_pos_body: np.ndarray,
    basis: np.ndarray,
    focal_length: float,
    screen_cx: float,
    screen_cy: float,
    scale: float,
) -> Optional[Tuple[float, float]]:
    """Perspective-project a body-frame point to screen pixels."""
    delta = np.asarray(point_body, dtype=np.float64) - cam_pos_body
    x_cam = float(np.dot(delta, basis[:, 0]))
    y_cam = float(np.dot(delta, basis[:, 1]))
    z_cam = float(np.dot(delta, basis[:, 2]))
    if z_cam <= 0.01:
        return None
    px = x_cam / z_cam * focal_length * scale + screen_cx
    py = -y_cam / z_cam * focal_length * scale + screen_cy
    return (px, py)


def _segment_outline(
    seg: VehicleSegmentConfig,
    cam_pos: np.ndarray,
    basis: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
    scale: float,
) -> List[Tuple[float, float]]:
    """
    Compute the 2-D projected outline polygon for one segment.

    Samples n_sides points around the two end circles and returns the
    convex hull-like outline (left side of front ring + right side of back ring).
    """
    n = seg.n_sides
    outline: List[Tuple[float, float]] = []

    # Build rings at x_start and x_end
    for x, r in ((seg.x_start, seg.radius_start), (seg.x_end, seg.radius_end)):
        ring = []
        for i in range(n):
            angle = 2.0 * math.pi * i / n
            # Body frame: Y is right, Z is down → circle in YZ plane
            y = r * math.cos(angle)
            z = r * math.sin(angle)
            pt = np.array([x, y, z], dtype=np.float64)
            px = _project_body(pt, cam_pos, basis, focal_length, cx, cy, scale)
            if px is not None:
                ring.append(px)
        outline.extend(ring)

    return outline


def _plume_outline(
    nozzle_x: float,
    nozzle_radius: float,
    plume_length: float,
    cam_pos: np.ndarray,
    basis: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
    scale: float,
    n_sides: int = 12,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Compute outer plume cone and inner core polygon vertices.

    Returns
    -------
    outer_pts, inner_pts
    """
    outer: List[Tuple[float, float]] = []
    inner: List[Tuple[float, float]] = []

    # Outer plume (aft nozzle → tip)
    for i in range(n_sides + 1):
        frac = i / n_sides
        x = nozzle_x - frac * plume_length  # plume goes aft (-X direction)
        r = nozzle_radius * (1.0 - frac)
        angle = 2.0 * math.pi * i / n_sides
        y = r * math.cos(angle)
        z = r * math.sin(angle)
        pt = np.array([x, y, z], dtype=np.float64)
        px = _project_body(pt, cam_pos, basis, focal_length, cx, cy, scale)
        if px is not None:
            outer.append(px)

    # Inner core (bright centre, 1/3 of outer radius)
    core_len = plume_length * 0.4
    for i in range(n_sides + 1):
        frac = i / n_sides
        x = nozzle_x - frac * core_len
        r = nozzle_radius * 0.35 * (1.0 - frac)
        angle = 2.0 * math.pi * i / n_sides
        y = r * math.cos(angle)
        z = r * math.sin(angle)
        pt = np.array([x, y, z], dtype=np.float64)
        px = _project_body(pt, cam_pos, basis, focal_length, cx, cy, scale)
        if px is not None:
            inner.append(px)

    return outer, inner


# ---------------------------------------------------------------------------
# VehicleRenderer
# ---------------------------------------------------------------------------

class VehicleRenderer:
    """
    Body-frame vehicle renderer: converts RenderFrame → VehicleScene.

    Parameters
    ----------
    config : VehicleRenderConfig
        Vehicle geometry and camera configuration.
    screen_width : int
        Viewport width in pixels. Default 1280.
    screen_height : int
        Viewport height in pixels. Default 720.
    """

    def __init__(
        self,
        config: VehicleRenderConfig,
        screen_width: int = 1280,
        screen_height: int = 720,
    ) -> None:
        if not isinstance(config, VehicleRenderConfig):
            raise TypeError("config must be a VehicleRenderConfig")
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("screen dimensions must be positive")
        self._config = config
        self._sw = screen_width
        self._sh = screen_height

    @property
    def config(self) -> VehicleRenderConfig:
        return self._config

    # ------------------------------------------------------------------

    def build(self, frame: RenderFrame) -> VehicleScene:
        """
        Build a VehicleScene from the current RenderFrame.

        Parameters
        ----------
        frame : RenderFrame
            Interpolated render state from Viewport.get_render_frame().

        Returns
        -------
        VehicleScene
        """
        if not isinstance(frame, RenderFrame):
            raise TypeError(f"frame must be a RenderFrame; got {type(frame).__name__}")

        cfg = self._config
        cx, cy = self._sw / 2.0, self._sh / 2.0

        # Camera in body frame
        az = cfg.camera_azimuth_rad
        el = cfg.camera_elevation_rad
        dist = cfg.camera_distance_m
        cam_pos = np.array([
            dist * math.cos(el) * math.cos(az),
            dist * math.sin(el),
            -dist * math.cos(el) * math.sin(az),
        ], dtype=np.float64)
        basis = _body_camera_basis(az, el)

        # Focal length and scale derived from screen size
        total_length = max(
            (abs(s.x_start) + abs(s.x_end) for s in cfg.segments),
            default=10.0,
        )
        focal_length = dist * 0.8
        scale = min(self._sw, self._sh) / max(total_length * 2.0, 1.0)

        # Build segment geometries
        failed_ids = cfg.failed_joint_ids
        any_fail = False
        segment_geoms: List[SegmentGeometry] = []

        for seg in cfg.segments:
            is_failed = seg.segment_id in failed_ids
            if is_failed:
                any_fail = True
            color = seg.failure_color if is_failed else seg.color
            outline = _segment_outline(seg, cam_pos, basis, focal_length, cx, cy, scale)
            visible = len(outline) > 0
            segment_geoms.append(SegmentGeometry(
                segment_id=seg.segment_id,
                outline_px=outline,
                color=color,
                is_failed=is_failed,
                visible=visible,
            ))

        # Engine plume
        engine_seg = next((s for s in cfg.segments if s.is_engine), None)
        throttle = float(frame.throttle)
        plume_active = throttle > 1.0e-3 and engine_seg is not None
        plume_length = 0.0
        outer_pts: List[Tuple[float, float]] = []
        inner_pts: List[Tuple[float, float]] = []
        nozzle_pos = np.zeros(3, dtype=np.float64)

        if engine_seg is not None:
            nozzle_x = engine_seg.x_end  # aft face of engine segment
            nozzle_r = engine_seg.radius_end
            nozzle_pos = np.array([nozzle_x, 0.0, 0.0], dtype=np.float64)

            if plume_active:
                # Plume length: scale by throttle; vary slightly with Mach
                mach_factor = max(0.1, 1.0 - 0.05 * max(0.0, frame.mach))
                plume_length = cfg.plume_max_length_m * throttle * mach_factor
                outer_pts, inner_pts = _plume_outline(
                    nozzle_x, nozzle_r, plume_length,
                    cam_pos, basis, focal_length, cx, cy, scale,
                )

        plume = PlumeGeometry(
            active=plume_active,
            length_m=plume_length,
            outline_px=outer_pts,
            core_px=inner_pts,
            nozzle_pos_body=nozzle_pos,
        )

        # Reference axes
        ref_axes: List[ReferenceAxis] = []
        if cfg.show_reference_axes:
            axis_len = total_length * 0.25
            origin = np.zeros(3, dtype=np.float64)
            for direction, label, color in (
                (np.array([1, 0, 0], dtype=np.float64), "+X", (255, 80, 80)),
                (np.array([0, 1, 0], dtype=np.float64), "+Y", (80, 255, 80)),
                (np.array([0, 0, 1], dtype=np.float64), "+Z", (80, 80, 255)),
            ):
                tip = direction * axis_len
                p0 = _project_body(origin, cam_pos, basis, focal_length, cx, cy, scale)
                p1 = _project_body(tip, cam_pos, basis, focal_length, cx, cy, scale)
                if p0 is not None and p1 is not None:
                    ref_axes.append(ReferenceAxis(
                        label=label,
                        start_px=p0,
                        end_px=p1,
                        color=color,
                    ))

        return VehicleScene(
            segments=segment_geoms,
            plume=plume,
            reference_axes=ref_axes,
            camera_pos_body=cam_pos,
            render_time=frame.mission_time,
            throttle=throttle,
            any_structural_failure=any_fail,
        )

    def __repr__(self) -> str:
        return (
            f"VehicleRenderer(segments={len(self._config.segments)}, "
            f"{self._sw}×{self._sh}px)"
        )


# ---------------------------------------------------------------------------
# Convenience: default rocket geometry
# ---------------------------------------------------------------------------

def default_rocket_config(
    total_length_m: float = 40.0,
    max_radius_m: float = 1.85,
) -> VehicleRenderConfig:
    """
    Build a VehicleRenderConfig representing a two-stage rocket.

    Parameters
    ----------
    total_length_m : float
        Total vehicle length [m]. Default 40.
    max_radius_m : float
        Maximum body radius [m]. Default 1.85.

    Returns
    -------
    VehicleRenderConfig
    """
    L = total_length_m
    R = max_radius_m

    segments = [
        # Nose cone
        VehicleSegmentConfig(
            segment_id="nose_cone",
            shape=SHAPE_NOSECONE,
            x_start=L,
            x_end=L * 0.85,
            radius_start=0.0,
            radius_end=R * 0.6,
            color=(220, 220, 220),
        ),
        # Payload fairing
        VehicleSegmentConfig(
            segment_id="payload_fairing",
            shape=SHAPE_CYLINDER,
            x_start=L * 0.85,
            x_end=L * 0.65,
            radius_start=R * 0.6,
            radius_end=R * 0.6,
            color=(200, 200, 200),
        ),
        # Inter-stage
        VehicleSegmentConfig(
            segment_id="inter_stage",
            shape=SHAPE_CONE,
            x_start=L * 0.65,
            x_end=L * 0.60,
            radius_start=R * 0.6,
            radius_end=R,
            color=(160, 160, 160),
        ),
        # First stage body
        VehicleSegmentConfig(
            segment_id="first_stage_body",
            shape=SHAPE_CYLINDER,
            x_start=L * 0.60,
            x_end=L * 0.10,
            radius_start=R,
            radius_end=R,
            color=(240, 240, 245),
        ),
        # Engine section / nozzle
        VehicleSegmentConfig(
            segment_id="engine_section",
            shape=SHAPE_CONE,
            x_start=L * 0.10,
            x_end=0.0,
            radius_start=R,
            radius_end=R * 0.55,
            color=(80, 80, 80),
            is_engine=True,
        ),
    ]

    return VehicleRenderConfig(
        segments=segments,
        camera_distance_m=total_length_m * 2.0,
        plume_max_length_m=total_length_m * 0.5,
    )
