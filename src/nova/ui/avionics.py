"""
nova.ui.avionics
=================
Avionics instrumentation panel for Project NOVA glass cockpit.

Architectural role
------------------
Phase 13 — UI Glass Cockpit.
Pipeline stage: Stage 13 (UI Engine). Consumes a TelemetrySnapshot and
a list of AlertMessages from the AI Monitor (Phase 6) and produces an
AvionicsState frozen dataclass containing all data needed to render the
avionics panel.

Design
------
The avionics panel displays:
  Engine health:
    - Throttle [%]
    - Thrust [kN]
    - Mass flow rate [kg s⁻¹]
    - Isp [s]
    - Engine status (ACTIVE / IDLE / FAILED)

  SAS status:
    - SAS enabled / disabled flag
    - Angular rates (p, q, r) [deg s⁻¹]
    - Rate magnitudes

  Power / resource:
    - Vehicle mass [kg] / propellant remaining (from mass flow accumulation)
    - Structural health: worst margin, critical joint ID, failure flag

  Alerts:
    - Active alerts grouped by severity (CRITICAL, WARNING, CAUTION, INFO)
    - Alert count per severity

I/O contract
------------
Input  : TelemetrySnapshot (latest from registry), List[AlertMessage]
Output : AvionicsState (frozen dataclass)

No Pygame calls. No physics.

References
----------
- NOVA Engineering Handoff §12 Phase 13
- FAA AC 25.1322-1 (Flight Crew Alerting)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nova.ai.monitor import AlertMessage, AlertSeverity
from nova.core.telemetry_registry import TelemetrySnapshot

# ---------------------------------------------------------------------------
# Engine status enum
# ---------------------------------------------------------------------------

class EngineStatus:
    """Engine operating status strings."""
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# EngineDisplayData
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineDisplayData:
    """
    Engine health and performance data for the avionics panel.

    Attributes
    ----------
    status : str
        One of EngineStatus constants.
    throttle_pct : float
        Engine throttle [%], range [0, 100].
    thrust_kn : float
        Engine thrust [kN].
    mass_flow_kg_s : float
        Propellant mass flow rate [kg s⁻¹].
    isp_s : float
        Effective specific impulse [s].
    twr : float
        Thrust-to-weight ratio (dimensionless). 0 if engine off.
    gimbal_pitch_deg : float
        Engine gimbal pitch angle [degrees]. Derived from torque vector.
    gimbal_yaw_deg : float
        Engine gimbal yaw angle [degrees].
    """

    status: str
    throttle_pct: float
    thrust_kn: float
    mass_flow_kg_s: float
    isp_s: float
    twr: float
    gimbal_pitch_deg: float
    gimbal_yaw_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", str(self.status))
        for attr in ("throttle_pct", "thrust_kn", "mass_flow_kg_s",
                     "isp_s", "twr", "gimbal_pitch_deg", "gimbal_yaw_deg"):
            object.__setattr__(self, attr, float(getattr(self, attr)))

    def __repr__(self) -> str:
        return (
            f"EngineDisplayData(status={self.status}, "
            f"throttle={self.throttle_pct:.1f}%, "
            f"thrust={self.thrust_kn:.1f}kN, "
            f"Isp={self.isp_s:.0f}s)"
        )


# ---------------------------------------------------------------------------
# AngularRateData
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AngularRateData:
    """
    Vehicle angular rate display data.

    Attributes
    ----------
    roll_rate_deg_s : float
        Roll rate p [deg s⁻¹].
    pitch_rate_deg_s : float
        Pitch rate q [deg s⁻¹].
    yaw_rate_deg_s : float
        Yaw rate r [deg s⁻¹].
    total_rate_deg_s : float
        Magnitude ‖ω‖ [deg s⁻¹].
    """

    roll_rate_deg_s: float
    pitch_rate_deg_s: float
    yaw_rate_deg_s: float
    total_rate_deg_s: float

    def __post_init__(self) -> None:
        for attr in ("roll_rate_deg_s", "pitch_rate_deg_s",
                     "yaw_rate_deg_s", "total_rate_deg_s"):
            object.__setattr__(self, attr, float(getattr(self, attr)))


# ---------------------------------------------------------------------------
# StructuralHealthData
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructuralHealthData:
    """
    Structural health monitor display data.

    Attributes
    ----------
    worst_margin : float
        Lowest structural safety margin across all joints (1.0 = 100% margin).
    critical_joint_id : str
        Identifier of the joint with the worst margin. Empty string if none.
    any_failure : bool
        True if any structural failure has been detected.
    health_pct : float
        Structural health percentage = worst_margin × 100. Clamped [0, 100].
    status_label : str
        Human-readable status: "NOMINAL", "WARNING", "CRITICAL", "FAILED".
    """

    worst_margin: float
    critical_joint_id: str
    any_failure: bool
    health_pct: float
    status_label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "worst_margin", float(self.worst_margin))
        object.__setattr__(self, "critical_joint_id", str(self.critical_joint_id))
        object.__setattr__(self, "any_failure", bool(self.any_failure))
        object.__setattr__(self, "health_pct", float(self.health_pct))
        object.__setattr__(self, "status_label", str(self.status_label))


# ---------------------------------------------------------------------------
# AlertSummary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlertSummary:
    """
    Grouped summary of active AI monitor alerts.

    Attributes
    ----------
    critical : list[AlertMessage]
        All CRITICAL severity alerts.
    warning : list[AlertMessage]
        All WARNING severity alerts.
    caution : list[AlertMessage]
        All CAUTION severity alerts.
    info : list[AlertMessage]
        All INFO severity alerts.
    n_critical : int
        Count of critical alerts.
    n_warning : int
        Count of warnings.
    n_caution : int
        Count of cautions.
    master_warning : bool
        True if any CRITICAL alert is active.
    master_caution : bool
        True if any WARNING or CAUTION alert is active.
    """

    critical: List[AlertMessage]
    warning: List[AlertMessage]
    caution: List[AlertMessage]
    info: List[AlertMessage]
    n_critical: int
    n_warning: int
    n_caution: int
    master_warning: bool
    master_caution: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "critical", list(self.critical))
        object.__setattr__(self, "warning", list(self.warning))
        object.__setattr__(self, "caution", list(self.caution))
        object.__setattr__(self, "info", list(self.info))
        for attr in ("n_critical", "n_warning", "n_caution"):
            object.__setattr__(self, attr, int(getattr(self, attr)))
        object.__setattr__(self, "master_warning", bool(self.master_warning))
        object.__setattr__(self, "master_caution", bool(self.master_caution))

    @property
    def all_active(self) -> List[AlertMessage]:
        """All alerts across all severity levels, CRITICAL first."""
        return self.critical + self.warning + self.caution + self.info

    @property
    def any_active(self) -> bool:
        """True if there are any active alerts."""
        return bool(self.critical or self.warning or self.caution or self.info)

    def __repr__(self) -> str:
        return (
            f"AlertSummary(CRIT={self.n_critical}, "
            f"WARN={self.n_warning}, "
            f"CAUT={self.n_caution}, "
            f"MW={self.master_warning})"
        )


# ---------------------------------------------------------------------------
# AvionicsState — complete avionics panel data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AvionicsState:
    """
    Complete frozen data bundle for the avionics panel.

    Attributes
    ----------
    mission_time : float
        Mission elapsed time [s].
    engine : EngineDisplayData
        Engine health and performance.
    angular_rates : AngularRateData
        Vehicle angular rates.
    structural : StructuralHealthData
        Structural health summary.
    alerts : AlertSummary
        Active AI monitor alerts.
    vehicle_mass_kg : float
        Current vehicle mass [kg].
    downrange_km : float
        Downrange distance [km].
    """

    mission_time: float
    engine: EngineDisplayData
    angular_rates: AngularRateData
    structural: StructuralHealthData
    alerts: AlertSummary
    vehicle_mass_kg: float
    downrange_km: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mission_time", float(self.mission_time))
        object.__setattr__(self, "vehicle_mass_kg", float(self.vehicle_mass_kg))
        object.__setattr__(self, "downrange_km", float(self.downrange_km))

    def __repr__(self) -> str:
        return (
            f"AvionicsState(t={self.mission_time:.1f}s, "
            f"engine={self.engine.status}, "
            f"MW={self.alerts.master_warning})"
        )


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _classify_engine(throttle: float, thrust_n: float) -> str:
    """Determine engine status string from throttle and thrust."""
    if thrust_n > 1.0:
        return EngineStatus.ACTIVE
    if throttle > 0.0:
        return EngineStatus.IDLE
    return EngineStatus.IDLE


def _structural_label(margin: float, failed: bool) -> str:
    """Convert structural margin to a display label."""
    if failed:
        return "FAILED"
    if margin < 0.10:
        return "CRITICAL"
    if margin < 0.25:
        return "WARNING"
    return "NOMINAL"


def _group_alerts(alerts: List[AlertMessage]) -> AlertSummary:
    """Group alert messages by severity."""
    critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
    warning = [a for a in alerts if a.severity == AlertSeverity.WARNING]
    caution = [a for a in alerts if a.severity == AlertSeverity.CAUTION]
    info = [a for a in alerts if a.severity == AlertSeverity.INFO]
    return AlertSummary(
        critical=critical,
        warning=warning,
        caution=caution,
        info=info,
        n_critical=len(critical),
        n_warning=len(warning),
        n_caution=len(caution),
        master_warning=len(critical) > 0,
        master_caution=len(warning) > 0 or len(caution) > 0,
    )


# ---------------------------------------------------------------------------
# AvionicsPanel builder
# ---------------------------------------------------------------------------

class AvionicsPanel:
    """
    Produces AvionicsState from a TelemetrySnapshot each display tick.

    Parameters
    ----------
    sas_enabled : bool
        Current SAS enable state. Updated externally by guidance layer.
    """

    def __init__(self, sas_enabled: bool = True) -> None:
        self._sas_enabled = bool(sas_enabled)

    @property
    def sas_enabled(self) -> bool:
        return self._sas_enabled

    @sas_enabled.setter
    def sas_enabled(self, value: bool) -> None:
        self._sas_enabled = bool(value)

    def build(
        self,
        snapshot: TelemetrySnapshot,
        alerts: Optional[List[AlertMessage]] = None,
    ) -> AvionicsState:
        """
        Build an AvionicsState from the latest TelemetrySnapshot.

        Parameters
        ----------
        snapshot : TelemetrySnapshot
            Latest frozen telemetry snapshot from the registry.
        alerts : list[AlertMessage] | None
            Active alerts from the AI Monitor. None treated as empty list.

        Returns
        -------
        AvionicsState
        """
        if not isinstance(snapshot, TelemetrySnapshot):
            raise TypeError(
                f"snapshot must be a TelemetrySnapshot; got {type(snapshot).__name__}"
            )
        alerts = alerts if alerts is not None else []

        vs = snapshot.vehicle_state
        omega = vs.omega_body    # [p, q, r] rad/s

        # Engine
        throttle_pct = snapshot.throttle * 100.0
        thrust_kn = snapshot.thrust_magnitude / 1_000.0
        engine_status = _classify_engine(snapshot.throttle, snapshot.thrust_magnitude)

        # Gimbal angles from torque_gimbal vector (proxy: atan2 of torque components)
        tq = snapshot.torque_gimbal
        gimbal_p = math.degrees(math.atan2(float(tq[1]), max(1.0, snapshot.thrust_magnitude)))
        gimbal_y = math.degrees(math.atan2(float(tq[2]), max(1.0, snapshot.thrust_magnitude)))

        engine = EngineDisplayData(
            status=engine_status,
            throttle_pct=throttle_pct,
            thrust_kn=thrust_kn,
            mass_flow_kg_s=snapshot.mass_flow_rate,
            isp_s=snapshot.isp_effective,
            twr=snapshot.twr,
            gimbal_pitch_deg=gimbal_p,
            gimbal_yaw_deg=gimbal_y,
        )

        # Angular rates
        p_deg = math.degrees(float(omega[0]))
        q_deg = math.degrees(float(omega[1]))
        r_deg = math.degrees(float(omega[2]))
        omega_total = math.degrees(
            float((float(omega[0])**2 + float(omega[1])**2 + float(omega[2])**2) ** 0.5)
        )
        angular_rates = AngularRateData(
            roll_rate_deg_s=p_deg,
            pitch_rate_deg_s=q_deg,
            yaw_rate_deg_s=r_deg,
            total_rate_deg_s=omega_total,
        )

        # Structural health
        margin = snapshot.worst_structural_margin
        health_pct = max(0.0, min(100.0, margin * 100.0))
        structural = StructuralHealthData(
            worst_margin=margin,
            critical_joint_id=snapshot.critical_joint_id,
            any_failure=snapshot.any_structural_failure,
            health_pct=health_pct,
            status_label=_structural_label(margin, snapshot.any_structural_failure),
        )

        # Alerts
        alert_summary = _group_alerts(alerts)

        return AvionicsState(
            mission_time=snapshot.mission_time,
            engine=engine,
            angular_rates=angular_rates,
            structural=structural,
            alerts=alert_summary,
            vehicle_mass_kg=vs.mass,
            downrange_km=snapshot.downrange_distance / 1_000.0,
        )

    def __repr__(self) -> str:
        return f"AvionicsPanel(sas_enabled={self._sas_enabled})"
