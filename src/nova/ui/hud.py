"""
nova.ui.hud
============
Master HUD compositor for Project NOVA glass cockpit.

Architectural role
------------------
Phase 13 — UI Glass Cockpit.
Pipeline stage: Stage 13 (UI Engine). Orchestrates all UI sub-panels
(PFD, OrbitalDeck, Avionics) and combines them with the rendering layer
(Viewport, CelestialRenderer, VehicleRenderer) into a single HUDFrame
per display tick.

Design
------
The HUD compositor is the top-level Stage 13 object. Each tick it:
  1. Gets a RenderFrame from the Viewport (interpolated physics state)
  2. Builds PFDState from the PrimaryFlightDisplay
  3. Builds OrbitalDeckState from the OrbitalDeck
  4. Gets the latest TelemetrySnapshot from the registry
  5. Runs the AI Monitor to get current alerts
  6. Builds AvionicsState from the AvionicsPanel
  7. Builds CelestialScene from the CelestialRenderer
  8. Builds VehicleScene from the VehicleRenderer
  9. Assembles everything into a HUDFrame

HUDFrame is a frozen dataclass containing all panel data. No drawing
occurs here — the caller (Pygame draw loop or test) consumes HUDFrame.

The compositor respects ViewportConfig.show_hud: if False, sub-panel
builds are skipped and the HUDFrame carries None for UI panels (saving CPU).

I/O contract
------------
Input  : Viewport, TelemetryRegistry, CelestialRenderer, VehicleRenderer,
         PrimaryFlightDisplay, OrbitalDeck, AvionicsPanel, HUDConfig
Output : HUDFrame (frozen dataclass) — complete tick render bundle

No Pygame calls. No physics. Reads registry (never writes).

References
----------
- NOVA Engineering Handoff §7 Stage 13, §12 Phase 13
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from nova.ai.monitor import AlertMessage, MonitorConfig, assess
from nova.core.telemetry_registry import TelemetryRegistry, TelemetrySnapshot
from nova.rendering.celestial import CelestialRenderer, CelestialScene
from nova.rendering.vehicle_render import VehicleRenderer, VehicleScene
from nova.rendering.viewport import RenderFrame, Viewport
from nova.ui.avionics import AvionicsPanel, AvionicsState
from nova.ui.orbital_deck import OrbitalDeck, OrbitalDeckState
from nova.ui.pfd import PFDState, PrimaryFlightDisplay

# ---------------------------------------------------------------------------
# HUDConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HUDConfig:
    """
    Configuration for the master HUD compositor.

    Attributes
    ----------
    show_pfd : bool
        Render the Primary Flight Display panel. Default True.
    show_orbital_deck : bool
        Render the orbital elements panel. Default True.
    show_avionics : bool
        Render the avionics/engine panel. Default True.
    show_celestial : bool
        Build celestial scene (orbital globe view). Default True.
    show_vehicle : bool
        Build vehicle scene (close-up body view). Default True.
    monitor_config : MonitorConfig | None
        Configuration for the AI anomaly monitor. None uses defaults.
    show_alerts : bool
        Include alert messages in the HUDFrame. Default True.
    alert_max_display : int
        Maximum number of alerts shown on the HUD at once. Default 5.
    """

    show_pfd: bool = True
    show_orbital_deck: bool = True
    show_avionics: bool = True
    show_celestial: bool = True
    show_vehicle: bool = True
    monitor_config: Optional[MonitorConfig] = None
    show_alerts: bool = True
    alert_max_display: int = 5

    def __post_init__(self) -> None:
        for attr in ("show_pfd", "show_orbital_deck", "show_avionics",
                     "show_celestial", "show_vehicle", "show_alerts"):
            object.__setattr__(self, attr, bool(getattr(self, attr)))
        object.__setattr__(self, "alert_max_display", int(self.alert_max_display))
        if self.alert_max_display < 1:
            raise ValueError(
                f"alert_max_display must be ≥ 1; got {self.alert_max_display}"
            )


# ---------------------------------------------------------------------------
# HUDFrame — one complete tick display bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HUDFrame:
    """
    Immutable bundle of all UI data for one display tick.

    Attributes
    ----------
    render_frame : RenderFrame | None
        Interpolated physics state from the Viewport. None if no data.
    pfd : PFDState | None
        Primary flight display data. None if show_pfd=False or no data.
    orbital_deck : OrbitalDeckState | None
        Orbital elements panel data.
    avionics : AvionicsState | None
        Avionics panel data.
    celestial_scene : CelestialScene | None
        Celestial (orbital globe) geometry.
    vehicle_scene : VehicleScene | None
        Vehicle close-up geometry.
    alerts : list[AlertMessage]
        Active alert messages (already included in avionics, also here
        for convenient top-level access).
    tick_number : int
        Monotonically increasing HUD tick counter.
    mission_time : float
        Mission elapsed time [s] this frame represents. 0.0 if no data.
    has_data : bool
        True if a RenderFrame was successfully produced this tick.
    """

    render_frame: Optional[RenderFrame]
    pfd: Optional[PFDState]
    orbital_deck: Optional[OrbitalDeckState]
    avionics: Optional[AvionicsState]
    celestial_scene: Optional[CelestialScene]
    vehicle_scene: Optional[VehicleScene]
    alerts: List[AlertMessage]
    tick_number: int
    mission_time: float
    has_data: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "alerts", list(self.alerts))
        object.__setattr__(self, "tick_number", int(self.tick_number))
        object.__setattr__(self, "mission_time", float(self.mission_time))
        object.__setattr__(self, "has_data", bool(self.has_data))

    @property
    def master_warning(self) -> bool:
        """True if any CRITICAL alert is active."""
        if self.avionics is not None:
            return self.avionics.alerts.master_warning
        return False

    @property
    def master_caution(self) -> bool:
        """True if any WARNING or CAUTION alert is active."""
        if self.avionics is not None:
            return self.avionics.alerts.master_caution
        return False

    @property
    def any_structural_failure(self) -> bool:
        """True if a structural failure is flagged."""
        if self.render_frame is not None:
            return self.render_frame.any_structural_failure
        return False

    def __repr__(self) -> str:
        return (
            f"HUDFrame(tick={self.tick_number}, "
            f"t={self.mission_time:.2f}s, "
            f"has_data={self.has_data}, "
            f"MW={self.master_warning})"
        )


# ---------------------------------------------------------------------------
# HUDCompositor — master Stage 13 orchestrator
# ---------------------------------------------------------------------------

class HUDCompositor:
    """
    Master HUD compositor: orchestrates all Phase 13 UI panels each tick.

    Parameters
    ----------
    viewport : Viewport
        Physics render loop (Phase 12). Source of RenderFrame.
    registry : TelemetryRegistry
        Simulation telemetry (Phase 5). Source of TelemetrySnapshot.
    celestial : CelestialRenderer
        Orbital globe renderer (Phase 12).
    vehicle : VehicleRenderer
        Body-frame vehicle renderer (Phase 12).
    pfd : PrimaryFlightDisplay
        PFD panel builder (Phase 13).
    orbital_deck : OrbitalDeck
        Orbital elements panel builder (Phase 13).
    avionics : AvionicsPanel
        Avionics panel builder (Phase 13).
    config : HUDConfig | None
        HUD configuration. Defaults to HUDConfig().
    """

    def __init__(
        self,
        viewport: Viewport,
        registry: TelemetryRegistry,
        celestial: CelestialRenderer,
        vehicle: VehicleRenderer,
        pfd: PrimaryFlightDisplay,
        orbital_deck: OrbitalDeck,
        avionics: AvionicsPanel,
        config: Optional[HUDConfig] = None,
    ) -> None:
        if not isinstance(viewport, Viewport):
            raise TypeError("viewport must be a Viewport")
        if not isinstance(registry, TelemetryRegistry):
            raise TypeError("registry must be a TelemetryRegistry")
        if not isinstance(celestial, CelestialRenderer):
            raise TypeError("celestial must be a CelestialRenderer")
        if not isinstance(vehicle, VehicleRenderer):
            raise TypeError("vehicle must be a VehicleRenderer")
        if not isinstance(pfd, PrimaryFlightDisplay):
            raise TypeError("pfd must be a PrimaryFlightDisplay")
        if not isinstance(orbital_deck, OrbitalDeck):
            raise TypeError("orbital_deck must be an OrbitalDeck")
        if not isinstance(avionics, AvionicsPanel):
            raise TypeError("avionics must be an AvionicsPanel")

        self._viewport = viewport
        self._registry = registry
        self._celestial = celestial
        self._vehicle = vehicle
        self._pfd = pfd
        self._orbital_deck = orbital_deck
        self._avionics = avionics
        self._config = config if config is not None else HUDConfig()
        if not isinstance(self._config, HUDConfig):
            raise TypeError("config must be a HUDConfig")
        self._tick: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> HUDConfig:
        return self._config

    @property
    def tick_number(self) -> int:
        return self._tick

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, display_time: Optional[float] = None) -> HUDFrame:
        """
        Build one complete HUDFrame for the current display tick.

        Parameters
        ----------
        display_time : float | None
            Mission time [s] to render. None = latest available.

        Returns
        -------
        HUDFrame
        """
        self._tick += 1
        cfg = self._config

        # Step 1: get render frame
        render_frame: Optional[RenderFrame] = self._viewport.get_render_frame(
            display_time=display_time
        )

        if render_frame is None:
            return HUDFrame(
                render_frame=None,
                pfd=None,
                orbital_deck=None,
                avionics=None,
                celestial_scene=None,
                vehicle_scene=None,
                alerts=[],
                tick_number=self._tick,
                mission_time=0.0,
                has_data=False,
            )

        mission_time = render_frame.mission_time

        # Step 2: PFD
        pfd_state: Optional[PFDState] = None
        if cfg.show_pfd:
            pfd_state = self._pfd.build(render_frame)

        # Step 3: Orbital deck
        orbital_state: Optional[OrbitalDeckState] = None
        if cfg.show_orbital_deck:
            orbital_state = self._orbital_deck.build(render_frame)

        # Step 4: Latest snapshot for avionics
        latest_snap: Optional[TelemetrySnapshot] = self._registry.latest

        # Step 5: AI Monitor alerts
        alerts: List[AlertMessage] = []
        if cfg.show_alerts and latest_snap is not None:
            monitor_cfg = cfg.monitor_config if cfg.monitor_config is not None else MonitorConfig()
            try:
                all_alerts = assess(self._registry, monitor_cfg)
                alerts = all_alerts[: cfg.alert_max_display]
            except Exception:
                alerts = []

        # Step 6: Avionics
        avionics_state: Optional[AvionicsState] = None
        if cfg.show_avionics and latest_snap is not None:
            avionics_state = self._avionics.build(latest_snap, alerts)

        # Step 7: Celestial scene
        celestial_scene: Optional[CelestialScene] = None
        if cfg.show_celestial:
            celestial_scene = self._celestial.build(render_frame)

        # Step 8: Vehicle scene
        vehicle_scene: Optional[VehicleScene] = None
        if cfg.show_vehicle:
            vehicle_scene = self._vehicle.build(render_frame)

        return HUDFrame(
            render_frame=render_frame,
            pfd=pfd_state,
            orbital_deck=orbital_state,
            avionics=avionics_state,
            celestial_scene=celestial_scene,
            vehicle_scene=vehicle_scene,
            alerts=alerts,
            tick_number=self._tick,
            mission_time=mission_time,
            has_data=True,
        )

    def reset(self) -> None:
        """Reset the HUD tick counter and clear celestial ground track."""
        self._tick = 0
        self._celestial.clear_ground_track()

    def __repr__(self) -> str:
        return (
            f"HUDCompositor(tick={self._tick}, "
            f"viewport={self._viewport!r})"
        )
