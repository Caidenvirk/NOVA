"""
nova.rendering.celestial
=========================
Global-space orbital viewport geometry for Project NOVA.

Architectural role
------------------
Phase 12 — Rendering.
Pipeline stage: Stage 12 (Renderer). Computes the geometry needed to draw
the long-range orbital view: planet sphere, orbital trajectory, ground track,
and celestial body markers. All computations are pure geometry — no physics,
no simulation state is modified.

Design
------
The celestial renderer works in two coordinate spaces:
  1. ECI [m] — all physics positions
  2. Screen pixels — for 2-D projection

Projection uses a simple orthographic or perspective camera centred on the
planet barycentre. The camera is parameterised by azimuth/elevation angles
and a zoom distance, producing a 3-D → 2-D projection matrix.

Key outputs (all frozen dataclasses):
  PlanetGeometry    — planet sphere wireframe circles and disc parameters
  OrbitalTrack      — list of 2-D screen points tracing the current orbit
  GroundTrack       — list of lon/lat pairs for the sub-vehicle ground track
  CelestialMarker   — a labelled point (e.g. apoapsis, periapsis markers)

No Pygame drawing calls are made here. The caller (HUD or integration)
passes the geometry to its drawing backend.

I/O contract
------------
Input  : RenderFrame (from Viewport), CelestialConfig
Output : CelestialScene (frozen dataclass containing all geometry)

References
----------
- NOVA Engineering Handoff §12 Phase 12
- Bate, Mueller & White "Fundamentals of Astrodynamics" §2 (orbit geometry)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from nova.core.constants import EARTH_RADIUS_EQ, EARTH_MU, EARTH_OMEGA
from nova.frames.ecef import _ecef_to_geodetic
from nova.rendering.viewport import RenderFrame

# ---------------------------------------------------------------------------
# CelestialConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CelestialConfig:
    """
    Configuration for the global orbital viewport.

    Attributes
    ----------
    planet_radius_m : float
        Radius of the central body [m]. Default EARTH_RADIUS_EQ.
    camera_distance_m : float
        Distance of the camera from the barycentre [m]. Must exceed
        planet_radius_m. Default 3× Earth radius.
    camera_azimuth_rad : float
        Camera azimuth angle [rad]. Default 0.0.
    camera_elevation_rad : float
        Camera elevation angle above equatorial plane [rad].
        Range [−π/2, π/2]. Default π/6 (30°).
    orbit_n_points : int
        Number of points used to trace the orbital ellipse. Default 360.
    ground_track_n_points : int
        Number of past positions recorded for ground track. Default 500.
    planet_wire_n_meridians : int
        Number of meridian lines on the planet wireframe. Default 12.
    planet_wire_n_parallels : int
        Number of parallel (latitude) lines. Default 6.
    planet_color : tuple[int,int,int]
        RGB colour for planet disc. Default (30, 80, 160).
    orbit_color : tuple[int,int,int]
        RGB colour for orbital track. Default (80, 200, 80).
    ground_track_color : tuple[int,int,int]
        RGB colour for ground track. Default (200, 140, 40).
    vehicle_color : tuple[int,int,int]
        RGB colour for vehicle marker. Default (255, 255, 255).
    apoapsis_color : tuple[int,int,int]
        RGB colour for apoapsis marker. Default (255, 100, 100).
    periapsis_color : tuple[int,int,int]
        RGB colour for periapsis marker. Default (100, 200, 255).
    """

    planet_radius_m: float = EARTH_RADIUS_EQ
    camera_distance_m: float = EARTH_RADIUS_EQ * 3.0
    camera_azimuth_rad: float = 0.0
    camera_elevation_rad: float = math.pi / 6.0
    orbit_n_points: int = 360
    ground_track_n_points: int = 500
    planet_wire_n_meridians: int = 12
    planet_wire_n_parallels: int = 6
    planet_color: Tuple[int, int, int] = (30, 80, 160)
    orbit_color: Tuple[int, int, int] = (80, 200, 80)
    ground_track_color: Tuple[int, int, int] = (200, 140, 40)
    vehicle_color: Tuple[int, int, int] = (255, 255, 255)
    apoapsis_color: Tuple[int, int, int] = (255, 100, 100)
    periapsis_color: Tuple[int, int, int] = (100, 200, 255)

    def __post_init__(self) -> None:
        if self.planet_radius_m <= 0.0:
            raise ValueError(f"planet_radius_m must be positive; got {self.planet_radius_m:.6g}")
        if self.camera_distance_m <= self.planet_radius_m:
            raise ValueError(
                f"camera_distance_m ({self.camera_distance_m:.3g}) must exceed "
                f"planet_radius_m ({self.planet_radius_m:.3g})"
            )
        if self.orbit_n_points < 3:
            raise ValueError(f"orbit_n_points must be ≥ 3; got {self.orbit_n_points}")
        if self.ground_track_n_points < 1:
            raise ValueError(f"ground_track_n_points must be ≥ 1; got {self.ground_track_n_points}")
        elev = float(self.camera_elevation_rad)
        if not (-math.pi / 2.0 <= elev <= math.pi / 2.0):
            raise ValueError(f"camera_elevation_rad must be in [−π/2, π/2]; got {elev:.6g}")
        for rgb_attr in ("planet_color", "orbit_color", "ground_track_color",
                         "vehicle_color", "apoapsis_color", "periapsis_color"):
            r, g, b = getattr(self, rgb_attr)
            for ch, name in ((r, 'R'), (g, 'G'), (b, 'B')):
                if not (0 <= ch <= 255):
                    raise ValueError(f"{rgb_attr}.{name} must be in [0,255]; got {ch}")


# ---------------------------------------------------------------------------
# Camera / projection utilities
# ---------------------------------------------------------------------------

def _camera_basis(azimuth: float, elevation: float) -> np.ndarray:
    """
    Compute a 3×3 camera-to-world rotation matrix from azimuth/elevation.

    Camera convention:
      +Z_cam → looking direction (into scene, toward origin)
      +Y_cam → up direction
      +X_cam → right direction

    Parameters
    ----------
    azimuth : float
        Azimuth angle [rad], eastward from +X.
    elevation : float
        Elevation angle [rad] above equatorial plane.

    Returns
    -------
    ndarray, shape (3, 3)
        Columns are [right, up, forward] in world (ECI) frame.
    """
    sa, ca = math.sin(azimuth), math.cos(azimuth)
    se, ce = math.sin(elevation), math.cos(elevation)

    # forward = from camera toward origin (pointing inward)
    forward = np.array([-ca * ce, -sa * ce, -se], dtype=np.float64)
    # world up = +Z (north pole)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # right = forward × world_up (normalised)
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1.0e-10:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= rn
    # recompute up = right × forward
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    return np.column_stack([right, up, forward])


def project_eci_to_screen(
    point_eci: np.ndarray,
    camera_pos_eci: np.ndarray,
    basis: np.ndarray,
    focal_length: float,
    screen_center: Tuple[float, float],
    scale: float = 1.0,
) -> Optional[Tuple[float, float]]:
    """
    Project an ECI point to screen coordinates using perspective projection.

    Returns None if the point is behind the camera (z_cam ≤ 0).

    Parameters
    ----------
    point_eci : ndarray, shape (3,)
        Point in ECI frame [m].
    camera_pos_eci : ndarray, shape (3,)
        Camera position in ECI frame [m].
    basis : ndarray, shape (3, 3)
        Camera basis matrix (columns: right, up, forward).
    focal_length : float
        Perspective focal length [m].
    screen_center : tuple[float, float]
        Screen centre (cx, cy) in pixels.
    scale : float
        Pixel-per-metre scaling factor.

    Returns
    -------
    tuple[float, float] | None
        Screen (x, y) pixel coordinates, or None if behind camera.
    """
    delta = np.asarray(point_eci, dtype=np.float64) - camera_pos_eci
    # Project into camera space
    x_cam = float(np.dot(delta, basis[:, 0]))
    y_cam = float(np.dot(delta, basis[:, 1]))
    z_cam = float(np.dot(delta, basis[:, 2]))

    # Points behind the camera (z_cam ≤ 0) are invisible
    if z_cam <= 1.0:
        return None

    # Perspective divide
    px = (x_cam / z_cam) * focal_length * scale + screen_center[0]
    py = -(y_cam / z_cam) * focal_length * scale + screen_center[1]  # flip Y
    return (px, py)


# ---------------------------------------------------------------------------
# Orbital geometry: ellipse sampling
# ---------------------------------------------------------------------------

def _orbit_eci_points(
    frame: RenderFrame,
    n_points: int,
    mu: float = EARTH_MU,
) -> List[np.ndarray]:
    """
    Sample *n_points* ECI positions along the current Keplerian orbit.

    Uses vis-viva and Keplerian geometry from the RenderFrame orbital
    elements. Returns an empty list if the orbit is hyperbolic (e ≥ 1).

    Parameters
    ----------
    frame : RenderFrame
        Current render state (must have valid orbital elements).
    n_points : int
        Number of sample points around the full orbit.
    mu : float
        Gravitational parameter [m³ s⁻²]. Default EARTH_MU.

    Returns
    -------
    list[ndarray, shape (3,)]
        ECI positions [m] sampled uniformly in true anomaly.
    """
    a = frame.semi_major_axis
    e = frame.eccentricity
    inc = frame.inclination

    # Only closed (elliptic) orbits
    if e >= 1.0 or a <= 0.0:
        return []

    # Build perifocal frame unit vectors from RenderFrame position/velocity
    r_vec = frame.position_eci
    v_vec = frame.velocity_eci
    r_norm = float(np.linalg.norm(r_vec))
    if r_norm < 1.0:
        return []

    h_vec = np.cross(r_vec, v_vec)
    h_norm = float(np.linalg.norm(h_vec))
    if h_norm < 1.0:
        return []

    # Eccentricity vector (points toward periapsis)
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r_norm
    e_norm = float(np.linalg.norm(e_vec))
    if e_norm < 1.0e-10:
        # Circular orbit: pick arbitrary periapsis direction
        e_hat = r_vec / r_norm
    else:
        e_hat = e_vec / e_norm

    h_hat = h_vec / h_norm
    q_hat = np.cross(h_hat, e_hat)  # completes perifocal frame

    points = []
    for i in range(n_points):
        nu = 2.0 * math.pi * i / n_points
        cos_nu = math.cos(nu)
        sin_nu = math.sin(nu)

        # Perifocal distance: p / (1 + e*cos(nu))
        p = a * (1.0 - e * e)
        r_pf = p / (1.0 + e * cos_nu)

        # ECI position
        pos = r_pf * (cos_nu * e_hat + sin_nu * q_hat)
        points.append(pos.copy())

    return points


# ---------------------------------------------------------------------------
# Planet wireframe geometry
# ---------------------------------------------------------------------------

def _planet_wireframe_eci(
    radius: float,
    n_meridians: int,
    n_parallels: int,
) -> List[List[np.ndarray]]:
    """
    Generate planet wireframe lines as lists of ECI points.

    Returns
    -------
    list of polylines, each polyline = list[ndarray, shape (3,)]
    """
    lines: List[List[np.ndarray]] = []

    # Meridians (lines of constant longitude)
    for i in range(n_meridians):
        lam = 2.0 * math.pi * i / n_meridians
        pts = []
        for j in range(37):  # 0→360°
            phi = -math.pi / 2.0 + math.pi * j / 36.0
            x = radius * math.cos(phi) * math.cos(lam)
            y = radius * math.cos(phi) * math.sin(lam)
            z = radius * math.sin(phi)
            pts.append(np.array([x, y, z], dtype=np.float64))
        lines.append(pts)

    # Parallels (lines of constant latitude, excluding poles)
    for j in range(1, n_parallels):
        phi = -math.pi / 2.0 + math.pi * j / n_parallels
        pts = []
        for i in range(37):
            lam = 2.0 * math.pi * i / 36.0
            x = radius * math.cos(phi) * math.cos(lam)
            y = radius * math.cos(phi) * math.sin(lam)
            z = radius * math.sin(phi)
            pts.append(np.array([x, y, z], dtype=np.float64))
        lines.append(pts)

    return lines


# ---------------------------------------------------------------------------
# Geometry result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CelestialMarker:
    """
    A labelled point marker in the orbital view.

    Attributes
    ----------
    label : str
        Display label (e.g. "Ap", "Pe", "Vehicle").
    position_eci : ndarray, shape (3,)
        ECI position of the marker [m].
    color : tuple[int,int,int]
        RGB display colour.
    radius_px : int
        Marker radius in pixels. Default 4.
    """

    label: str
    position_eci: np.ndarray
    color: Tuple[int, int, int]
    radius_px: int = 4

    def __post_init__(self) -> None:
        arr = np.asarray(self.position_eci, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"position_eci must have shape (3,); got {arr.shape}")
        object.__setattr__(self, "position_eci", arr)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "radius_px", int(self.radius_px))


@dataclass(frozen=True)
class GroundTrackPoint:
    """A single ground track point."""
    longitude_rad: float
    latitude_rad: float
    mission_time: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "longitude_rad", float(self.longitude_rad))
        object.__setattr__(self, "latitude_rad", float(self.latitude_rad))
        object.__setattr__(self, "mission_time", float(self.mission_time))


@dataclass(frozen=True)
class CelestialScene:
    """
    Complete geometry bundle for one celestial-view render frame.

    Attributes
    ----------
    orbit_points_eci : list[ndarray]
        ECI positions tracing the current Keplerian orbit.
    planet_wireframe : list[list[ndarray]]
        Groups of ECI polyline points for planet wireframe.
    markers : list[CelestialMarker]
        Labelled markers (vehicle, Ap, Pe).
    ground_track : list[GroundTrackPoint]
        Recent ground track geodetic positions.
    camera_pos_eci : ndarray, shape (3,)
        Camera position in ECI for this scene.
    camera_basis : ndarray, shape (3, 3)
        Camera orientation basis matrix.
    config : CelestialConfig
        Configuration used to generate this scene.
    render_time : float
        Mission time [s] this scene was computed for.
    """

    orbit_points_eci: List[np.ndarray]
    planet_wireframe: List[List[np.ndarray]]
    markers: List[CelestialMarker]
    ground_track: List[GroundTrackPoint]
    camera_pos_eci: np.ndarray
    camera_basis: np.ndarray
    config: CelestialConfig
    render_time: float

    def __post_init__(self) -> None:
        cam = np.asarray(self.camera_pos_eci, dtype=np.float64)
        object.__setattr__(self, "camera_pos_eci", cam)
        basis = np.asarray(self.camera_basis, dtype=np.float64)
        object.__setattr__(self, "camera_basis", basis)
        object.__setattr__(self, "render_time", float(self.render_time))

    def __repr__(self) -> str:
        return (
            f"CelestialScene(t={self.render_time:.2f}s, "
            f"orbit_pts={len(self.orbit_points_eci)}, "
            f"markers={len(self.markers)})"
        )


# ---------------------------------------------------------------------------
# CelestialRenderer — stateful ground-track accumulator + scene builder
# ---------------------------------------------------------------------------

class CelestialRenderer:
    """
    Builds CelestialScene geometry from a RenderFrame each display tick.

    Accumulates a rolling ground-track history and computes fresh orbital
    geometry from the interpolated RenderFrame on every call to :meth:`build`.

    Parameters
    ----------
    config : CelestialConfig | None
        Celestial view configuration. Defaults to CelestialConfig().
    """

    def __init__(self, config: Optional[CelestialConfig] = None) -> None:
        self._config = config if config is not None else CelestialConfig()
        if not isinstance(self._config, CelestialConfig):
            raise TypeError("config must be a CelestialConfig")
        self._ground_track: List[GroundTrackPoint] = []

    @property
    def config(self) -> CelestialConfig:
        return self._config

    @property
    def ground_track(self) -> List[GroundTrackPoint]:
        """Current ground track history (read-only view)."""
        return list(self._ground_track)

    def clear_ground_track(self) -> None:
        """Reset the ground track history."""
        self._ground_track = []

    def _update_ground_track(self, frame: RenderFrame) -> None:
        """Append the current sub-vehicle point to the ground track."""
        pos = frame.position_eci
        r = float(np.linalg.norm(pos))
        if r < 1.0:
            return
        # ECEF at t=0 (ground track uses ECEF, approximate epoch=0 here
        # since the celestial view is not sensitive to the few-second ECEF drift)
        from nova.frames.transforms import T_ECI_to_ECEF
        T = T_ECI_to_ECEF(frame.mission_time, EARTH_OMEGA)
        pos_ecef = T @ pos
        lam, phi, _ = _ecef_to_geodetic(
            float(pos_ecef[0]), float(pos_ecef[1]), float(pos_ecef[2])
        )
        pt = GroundTrackPoint(
            longitude_rad=lam,
            latitude_rad=phi,
            mission_time=frame.mission_time,
        )
        self._ground_track.append(pt)
        # Rolling window
        max_pts = self._config.ground_track_n_points
        if len(self._ground_track) > max_pts:
            self._ground_track = self._ground_track[-max_pts:]

    def _apoapsis_eci(self, frame: RenderFrame) -> Optional[np.ndarray]:
        """Compute ECI position of apoapsis from orbital elements."""
        a = frame.semi_major_axis
        e = frame.eccentricity
        if e >= 1.0 or a <= 0.0:
            return None
        r_ap = a * (1.0 + e)
        r_vec = frame.position_eci
        v_vec = frame.velocity_eci
        r_norm = float(np.linalg.norm(r_vec))
        if r_norm < 1.0:
            return None
        h_vec = np.cross(r_vec, v_vec)
        h_norm = float(np.linalg.norm(h_vec))
        if h_norm < 1.0:
            return None
        e_vec = np.cross(v_vec, h_vec) / EARTH_MU - r_vec / r_norm
        e_norm = float(np.linalg.norm(e_vec))
        if e_norm < 1.0e-10:
            e_hat = r_vec / r_norm
        else:
            e_hat = e_vec / e_norm
        # Apoapsis is in the anti-periapsis direction
        return -r_ap * e_hat

    def _periapsis_eci(self, frame: RenderFrame) -> Optional[np.ndarray]:
        """Compute ECI position of periapsis from orbital elements."""
        a = frame.semi_major_axis
        e = frame.eccentricity
        if e >= 1.0 or a <= 0.0:
            return None
        r_pe = a * (1.0 - e)
        r_vec = frame.position_eci
        v_vec = frame.velocity_eci
        r_norm = float(np.linalg.norm(r_vec))
        if r_norm < 1.0:
            return None
        h_vec = np.cross(r_vec, v_vec)
        h_norm = float(np.linalg.norm(h_vec))
        if h_norm < 1.0:
            return None
        e_vec = np.cross(v_vec, h_vec) / EARTH_MU - r_vec / r_norm
        e_norm = float(np.linalg.norm(e_vec))
        if e_norm < 1.0e-10:
            e_hat = r_vec / r_norm
        else:
            e_hat = e_vec / e_norm
        return r_pe * e_hat

    def build(self, frame: RenderFrame) -> CelestialScene:
        """
        Build a CelestialScene from the current RenderFrame.

        Parameters
        ----------
        frame : RenderFrame
            Interpolated render state from Viewport.get_render_frame().

        Returns
        -------
        CelestialScene
        """
        if not isinstance(frame, RenderFrame):
            raise TypeError(f"frame must be a RenderFrame; got {type(frame).__name__}")

        cfg = self._config

        # Update ground track
        self._update_ground_track(frame)

        # Camera position in ECI
        az = cfg.camera_azimuth_rad
        el = cfg.camera_elevation_rad
        dist = cfg.camera_distance_m
        cam_pos = np.array([
            dist * math.cos(el) * math.cos(az),
            dist * math.cos(el) * math.sin(az),
            dist * math.sin(el),
        ], dtype=np.float64)
        basis = _camera_basis(az, el)

        # Orbital track
        orbit_pts = _orbit_eci_points(frame, cfg.orbit_n_points)

        # Planet wireframe
        wireframe = _planet_wireframe_eci(
            cfg.planet_radius_m,
            cfg.planet_wire_n_meridians,
            cfg.planet_wire_n_parallels,
        )

        # Markers
        markers: List[CelestialMarker] = [
            CelestialMarker(
                label="Vehicle",
                position_eci=frame.position_eci.copy(),
                color=cfg.vehicle_color,
                radius_px=5,
            )
        ]
        ap_pos = self._apoapsis_eci(frame)
        if ap_pos is not None:
            markers.append(CelestialMarker(
                label="Ap",
                position_eci=ap_pos,
                color=cfg.apoapsis_color,
                radius_px=4,
            ))
        pe_pos = self._periapsis_eci(frame)
        if pe_pos is not None:
            markers.append(CelestialMarker(
                label="Pe",
                position_eci=pe_pos,
                color=cfg.periapsis_color,
                radius_px=4,
            ))

        return CelestialScene(
            orbit_points_eci=orbit_pts,
            planet_wireframe=wireframe,
            markers=markers,
            ground_track=list(self._ground_track),
            camera_pos_eci=cam_pos,
            camera_basis=basis,
            config=cfg,
            render_time=frame.mission_time,
        )

    def __repr__(self) -> str:
        return (
            f"CelestialRenderer(track_pts={len(self._ground_track)}, "
            f"cam_dist={self._config.camera_distance_m:.3g}m)"
        )
