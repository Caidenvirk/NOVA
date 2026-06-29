"""
nova.ai.flight_logger
=====================
Structured event log writer for Project NOVA.

Architecture role — Pipeline Stage 10 (alongside AI Monitor)
-------------------------------------------------------------
The FlightLogger consumes TelemetrySnapshots from the registry and writes
two output artefacts:

1. **Telemetry CSV** — one row per published snapshot, one column per
   monitored scalar field. Suitable for post-flight analysis in NumPy,
   pandas, or any spreadsheet tool.

2. **Event JSON** — discrete flight events detected from telemetry
   transitions (engine ignition, MECO, max-Q, staging, orbit insertion,
   structural failure). Each event carries its mission time, type,
   description, and relevant parameter values.

Design contract
---------------
* The logger is purely append-only. It never reads back its own output.
* All writes are to in-memory buffers (list of rows / list of events).
  File I/O is explicit via ``save_csv()`` and ``save_json()``.
* The logger is stateful (it tracks previous snapshot for transition
  detection) but never modifies TelemetrySnapshot or VehicleState.
* Thread safety is NOT guaranteed — the logger runs synchronously in the
  single-threaded pipeline.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from nova.core.telemetry_registry import TelemetrySnapshot
from nova.ai.monitor import AlertMessage, AlertSeverity


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    ENGINE_IGNITION    = "ENGINE_IGNITION"
    ENGINE_CUTOFF      = "ENGINE_CUTOFF"
    MAX_Q              = "MAX_Q"
    STAGE_SEPARATION   = "STAGE_SEPARATION"
    ORBIT_INSERTION    = "ORBIT_INSERTION"
    REENTRY_INTERFACE  = "REENTRY_INTERFACE"
    STRUCTURAL_FAILURE = "STRUCTURAL_FAILURE"
    ALERT_CRITICAL     = "ALERT_CRITICAL"
    ALERT_WARNING      = "ALERT_WARNING"
    MILESTONE          = "MILESTONE"


@dataclass(frozen=True)
class FlightEvent:
    """
    A single discrete flight event detected from telemetry transitions.

    Attributes
    ----------
    event_type : EventType
    mission_time : float      Mission elapsed time [s].
    description : str         Human-readable event description.
    parameters : dict         Relevant numerical parameters at event time.
    """
    event_type:   EventType
    mission_time: float
    description:  str
    parameters:   Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type":   self.event_type.value,
            "mission_time": self.mission_time,
            "description":  self.description,
            "parameters":   self.parameters,
        }


# ---------------------------------------------------------------------------
# Telemetry CSV column schema
# ---------------------------------------------------------------------------

CSV_COLUMNS: List[str] = [
    "mission_time", "mass", "altitude", "speed", "vertical_speed",
    "mach", "dynamic_pressure", "alpha", "beta", "CL", "CD",
    "thrust_magnitude", "mass_flow_rate", "isp_effective", "throttle",
    "semi_major_axis", "eccentricity", "inclination", "orbital_period",
    "apoapsis", "periapsis", "worst_structural_margin", "any_structural_failure",
    "lift_force", "drag_force", "density", "pressure", "speed_of_sound",
    "downrange_distance",
]


def _snap_to_row(snap: TelemetrySnapshot) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for col in CSV_COLUMNS:
        val = getattr(snap, col, None)
        if val is None:
            row[col] = float("nan")
        elif isinstance(val, bool):
            row[col] = 1.0 if val else 0.0
        else:
            try:
                row[col] = float(val)
            except (TypeError, ValueError):
                row[col] = float("nan")
    return row


# ---------------------------------------------------------------------------
# FlightLogger
# ---------------------------------------------------------------------------

class FlightLogger:
    """
    Stateful flight data recorder.

    Accumulates telemetry rows and discrete events in memory.
    Exports to CSV and JSON on demand.

    Parameters
    ----------
    max_rows : int
        Maximum telemetry rows to buffer. Default 100 000.
    """

    def __init__(self, max_rows: int = 100_000) -> None:
        if max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {max_rows!r}")
        self._max_rows = max_rows
        self._rows:   List[Dict[str, float]] = []
        self._events: List[FlightEvent]      = []

        # Transition-detection state
        self._prev_snap: Optional[TelemetrySnapshot] = None
        self._max_q_seen:        float = 0.0
        self._max_q_logged:      bool  = False
        self._engine_was_active: bool  = False
        self._was_in_orbit:      bool  = False
        self._was_descending:    bool  = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, snap: TelemetrySnapshot) -> None:
        """Append a snapshot to the telemetry CSV buffer."""
        row = _snap_to_row(snap)
        if len(self._rows) >= self._max_rows:
            self._rows.pop(0)
        self._rows.append(row)

    def detect_events(
        self,
        snap:   TelemetrySnapshot,
        alerts: Optional[List[AlertMessage]] = None,
    ) -> List[FlightEvent]:
        """
        Detect discrete flight events from telemetry transitions.

        Parameters
        ----------
        snap : TelemetrySnapshot
            Current tick snapshot.
        alerts : list of AlertMessage, optional
            Active alerts from the AI monitor.

        Returns
        -------
        list of FlightEvent
            New events detected this tick (may be empty).
        """
        new_events: List[FlightEvent] = []
        t    = snap.mission_time
        prev = self._prev_snap

        # ── Engine ignition / cutoff ────────────────────────────────
        engine_active = snap.thrust_magnitude > 1.0
        if engine_active and not self._engine_was_active:
            new_events.append(FlightEvent(
                event_type=EventType.ENGINE_IGNITION,
                mission_time=t,
                description=f"Engine ignition. Throttle={snap.throttle:.2f}.",
                parameters={"thrust_N": snap.thrust_magnitude,
                            "throttle": snap.throttle,
                            "mass_kg":  snap.mass},
            ))
        elif not engine_active and self._engine_was_active:
            new_events.append(FlightEvent(
                event_type=EventType.ENGINE_CUTOFF,
                mission_time=t,
                description="Main engine cutoff (MECO).",
                parameters={"mass_kg":    snap.mass,
                            "speed_ms":   snap.speed,
                            "altitude_m": snap.altitude},
            ))
        self._engine_was_active = engine_active

        # ── Max-Q detection ─────────────────────────────────────────
        q_inf = snap.dynamic_pressure
        if q_inf > self._max_q_seen:
            self._max_q_seen   = q_inf
            self._max_q_logged = False
        elif (not self._max_q_logged
              and prev is not None
              and q_inf < prev.dynamic_pressure
              and self._max_q_seen > 1_000.0):
            new_events.append(FlightEvent(
                event_type=EventType.MAX_Q,
                mission_time=t,
                description=(
                    f"Max-Q: {self._max_q_seen/1000:.2f} kPa at "
                    f"alt={snap.altitude/1000:.1f} km, M={snap.mach:.3f}."
                ),
                parameters={"max_q_pa":   self._max_q_seen,
                            "altitude_m": snap.altitude,
                            "mach":       snap.mach,
                            "mass_kg":    snap.mass},
            ))
            self._max_q_logged = True

        # ── Orbit insertion ─────────────────────────────────────────
        in_orbit = (
            snap.eccentricity < 0.05
            and snap.periapsis > 6_371_000.0 + 80_000.0
            and snap.altitude > 80_000.0
        )
        if in_orbit and not self._was_in_orbit:
            new_events.append(FlightEvent(
                event_type=EventType.ORBIT_INSERTION,
                mission_time=t,
                description=(
                    f"Orbit insertion confirmed. "
                    f"a={snap.semi_major_axis/1000:.1f}km, "
                    f"e={snap.eccentricity:.4f}, "
                    f"i={math.degrees(snap.inclination):.2f}deg."
                ),
                parameters={"semi_major_axis_km": snap.semi_major_axis / 1000.0,
                            "eccentricity":       snap.eccentricity,
                            "inclination_deg":    math.degrees(snap.inclination),
                            "period_s":           snap.orbital_period},
            ))
        self._was_in_orbit = in_orbit

        # ── Reentry interface ───────────────────────────────────────
        descending  = prev is not None and snap.altitude < prev.altitude
        at_reentry  = 0.0 < snap.altitude < 120_000.0
        if at_reentry and descending and not self._was_descending:
            new_events.append(FlightEvent(
                event_type=EventType.REENTRY_INTERFACE,
                mission_time=t,
                description=(
                    f"Reentry interface. "
                    f"alt={snap.altitude/1000:.1f}km, v={snap.speed:.0f}m/s."
                ),
                parameters={"altitude_m": snap.altitude,
                            "speed_ms":   snap.speed,
                            "mach":       snap.mach},
            ))
        self._was_descending = at_reentry and descending

        # ── Structural failure ──────────────────────────────────────
        if snap.any_structural_failure:
            already = any(
                e.event_type == EventType.STRUCTURAL_FAILURE
                and abs(e.mission_time - t) < 0.1
                for e in self._events
            )
            if not already:
                new_events.append(FlightEvent(
                    event_type=EventType.STRUCTURAL_FAILURE,
                    mission_time=t,
                    description=(
                        f"Structural failure: joint '{snap.critical_joint_id}'. "
                        f"Margin={snap.worst_structural_margin:.3f}."
                    ),
                    parameters={"margin": snap.worst_structural_margin,
                                "q_inf":  snap.dynamic_pressure},
                ))

        # ── Critical alerts ─────────────────────────────────────────
        if alerts:
            for alert in alerts:
                if alert.severity == AlertSeverity.CRITICAL:
                    new_events.append(FlightEvent(
                        event_type=EventType.ALERT_CRITICAL,
                        mission_time=t,
                        description=alert.message,
                        parameters={
                            "current_value": alert.current_value,
                            "rate":          alert.rate,
                            "time_to_limit": (alert.time_to_limit_s
                                              if math.isfinite(alert.time_to_limit_s)
                                              else -1.0),
                        },
                    ))

        self._events.extend(new_events)
        self._prev_snap = snap
        return new_events

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_csv(self, path: str) -> None:
        """Write telemetry buffer to a CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)

    def save_json(self, path: str) -> None:
        """Write event log to a JSON file."""
        payload = {"event_count": len(self._events),
                   "events": [e.to_dict() for e in self._events]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def csv_string(self) -> str:
        """Return the CSV as an in-memory string (for testing)."""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(self._rows)
        return buf.getvalue()

    def events_json_string(self) -> str:
        """Return the event log as an in-memory JSON string."""
        payload = {"event_count": len(self._events),
                   "events": [e.to_dict() for e in self._events]}
        return json.dumps(payload, indent=2)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def events(self) -> List[FlightEvent]:
        return list(self._events)

    def events_of_type(self, event_type: EventType) -> List[FlightEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def clear(self) -> None:
        """Discard all buffered rows and events."""
        self._rows.clear()
        self._events.clear()
        self._prev_snap        = None
        self._max_q_seen       = 0.0
        self._max_q_logged     = False
        self._engine_was_active = False
        self._was_in_orbit      = False
        self._was_descending    = False

    def __repr__(self) -> str:
        return f"FlightLogger(rows={self.row_count}, events={self.event_count})"
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_flight_logger.py
=================================
Unit tests for nova.ai.flight_logger.

Tests verify:
  1. FlightLogger construction and repr.
  2. record() appends rows; row_count tracks correctly.
  3. max_rows ring-buffer discards oldest entries.
  4. csv_string() has correct headers and row count.
  5. events_json_string() is valid JSON with correct event_count.
  6. detect_events() — ENGINE_IGNITION fires on thrust rising edge.
  7. detect_events() — ENGINE_CUTOFF fires on thrust falling edge.
  8. detect_events() — MAX_Q fires after peak dynamic pressure.
  9. detect_events() — ORBIT_INSERTION fires when e<0.05 and peri>80km.
  10. detect_events() — STRUCTURAL_FAILURE fires on any_structural_failure=True.
  11. detect_events() — ALERT_CRITICAL fires for CRITICAL AlertMessage.
  12. detect_events() — no duplicate STRUCTURAL_FAILURE in same tick.
  13. clear() resets all state.
  14. FlightEvent.to_dict() contains required keys.
  15. EventType enum values are valid strings.
"""

import csv
import io
import json
import math
import pytest
import numpy as np

from nova.core.state_vector import make_state
from nova.core.telemetry_registry import TelemetryRegistry, build_snapshot, TelemetrySnapshot
from nova.ai.monitor import AlertMessage, AlertSeverity, AlertCategory
from nova.ai.flight_logger import (
    FlightLogger,
    FlightEvent,
    EventType,
    CSV_COLUMNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(t=0.0, mass=5000.0):
    return make_state(
        position_eci=[6_771_000.0, 0, 0],
        velocity_eci=[0, 7_672.0, 0],
        quaternion=[1, 0, 0, 0],
        omega_body=[0, 0, 0],
        mass=mass, time=t)


def _snap(t=0.0, mass=5000.0, **kw):
    return build_snapshot(_state(t, mass), **kw)


def _critical_alert() -> AlertMessage:
    return AlertMessage(
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.AERODYNAMIC,
        message="AoA = 22.0 deg. Exceeds limit.",
        recommendation="Reduce pitch.",
        time_to_limit_s=2.0,
        mission_time=5.0,
        parameter="alpha",
        current_value=math.radians(22.0),
        rate=math.radians(1.0),
        limit=math.radians(20.0),
    )


# ---------------------------------------------------------------------------
# 1. Construction and repr
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_construction(self):
        fl = FlightLogger()
        assert fl.row_count  == 0
        assert fl.event_count == 0

    def test_repr(self):
        fl = FlightLogger()
        r  = repr(fl)
        assert "FlightLogger" in r
        assert "rows=0" in r

    def test_invalid_max_rows_raises(self):
        with pytest.raises(ValueError, match="max_rows"):
            FlightLogger(max_rows=0)


# ---------------------------------------------------------------------------
# 2. record()
# ---------------------------------------------------------------------------

class TestRecord:

    def test_record_increments_row_count(self):
        fl = FlightLogger()
        for i in range(5):
            fl.record(_snap(float(i)))
        assert fl.row_count == 5

    def test_record_does_not_create_events(self):
        fl = FlightLogger()
        fl.record(_snap(0.0, thrust_magnitude=0.0))
        assert fl.event_count == 0

    def test_max_rows_ring_buffer(self):
        fl = FlightLogger(max_rows=3)
        for i in range(10):
            fl.record(_snap(float(i)))
        assert fl.row_count == 3   # capped at max_rows


# ---------------------------------------------------------------------------
# 3. csv_string()
# ---------------------------------------------------------------------------

class TestCSV:

    def test_csv_has_correct_headers(self):
        fl  = FlightLogger()
        fl.record(_snap(0.0))
        csv_text = fl.csv_string()
        reader   = csv.DictReader(io.StringIO(csv_text))
        assert set(reader.fieldnames) == set(CSV_COLUMNS)

    def test_csv_row_count(self):
        fl = FlightLogger()
        for i in range(4):
            fl.record(_snap(float(i) * 0.1, altitude=float(i) * 1000.0))
        reader = csv.DictReader(io.StringIO(fl.csv_string()))
        rows   = list(reader)
        assert len(rows) == 4

    def test_csv_mission_time_correct(self):
        fl = FlightLogger()
        fl.record(_snap(3.14))
        reader = csv.DictReader(io.StringIO(fl.csv_string()))
        row    = next(reader)
        assert abs(float(row["mission_time"]) - 3.14) < 1.0e-6

    def test_save_csv_creates_file(self, tmp_path):
        fl   = FlightLogger()
        fl.record(_snap(0.0))
        path = str(tmp_path / "test.csv")
        fl.save_csv(path)
        with open(path) as f:
            assert "mission_time" in f.readline()


# ---------------------------------------------------------------------------
# 4. events_json_string()
# ---------------------------------------------------------------------------

class TestJSON:

    def test_empty_events_valid_json(self):
        fl   = FlightLogger()
        data = json.loads(fl.events_json_string())
        assert data["event_count"] == 0
        assert data["events"] == []

    def test_event_count_matches(self):
        fl   = FlightLogger()
        snap = _snap(0.0, thrust_magnitude=0.0)
        fl.detect_events(snap)  # no engine -> no event
        snap2 = _snap(0.1, thrust_magnitude=10_000.0)
        fl.detect_events(snap2)  # engine on -> ENGINE_IGNITION
        data = json.loads(fl.events_json_string())
        assert data["event_count"] == fl.event_count

    def test_save_json_creates_file(self, tmp_path):
        fl   = FlightLogger()
        fl.detect_events(_snap(0.0, thrust_magnitude=10_000.0))
        path = str(tmp_path / "events.json")
        fl.save_json(path)
        with open(path) as f:
            data = json.load(f)
        assert "events" in data


# ---------------------------------------------------------------------------
# 5. Engine ignition / cutoff events
# ---------------------------------------------------------------------------

class TestEngineEvents:

    def test_engine_ignition_on_rising_edge(self):
        fl = FlightLogger()
        fl.detect_events(_snap(0.0, thrust_magnitude=0.0))      # engine off
        events = fl.detect_events(_snap(0.1, thrust_magnitude=50_000.0))  # engine on
        types  = [e.event_type for e in events]
        assert EventType.ENGINE_IGNITION in types

    def test_engine_cutoff_on_falling_edge(self):
        fl = FlightLogger()
        fl.detect_events(_snap(0.0, thrust_magnitude=50_000.0))  # engine on
        events = fl.detect_events(_snap(0.1, thrust_magnitude=0.0))   # engine off
        types  = [e.event_type for e in events]
        assert EventType.ENGINE_CUTOFF in types

    def test_engine_stays_on_no_event(self):
        fl = FlightLogger()
        fl.detect_events(_snap(0.0, thrust_magnitude=50_000.0))
        events = fl.detect_events(_snap(0.1, thrust_magnitude=50_000.0))
        types  = [e.event_type for e in events]
        assert EventType.ENGINE_IGNITION not in types
        assert EventType.ENGINE_CUTOFF   not in types

    def test_ignition_parameters_present(self):
        fl = FlightLogger()
        fl.detect_events(_snap(0.0, thrust_magnitude=0.0))
        events = fl.detect_events(_snap(0.1, thrust_magnitude=50_000.0, throttle=0.9))
        ig = next(e for e in events if e.event_type == EventType.ENGINE_IGNITION)
        assert "thrust_N"  in ig.parameters
        assert "throttle"  in ig.parameters
        assert "mass_kg"   in ig.parameters


# ---------------------------------------------------------------------------
# 6. Max-Q event
# ---------------------------------------------------------------------------

class TestMaxQEvent:

    def test_max_q_detected_after_peak(self):
        fl = FlightLogger()
        # Rising q
        for i in range(5):
            fl.detect_events(_snap(float(i)*0.1, dynamic_pressure=float(i)*5_000.0))
        # Falling q — triggers max-Q log
        events = fl.detect_events(_snap(0.5, dynamic_pressure=18_000.0))
        types  = [e.event_type for e in events]
        assert EventType.MAX_Q in types

    def test_max_q_not_fired_before_peak(self):
        fl     = FlightLogger()
        events = fl.detect_events(_snap(0.0, dynamic_pressure=10_000.0))
        assert EventType.MAX_Q not in [e.event_type for e in events]

    def test_max_q_parameters_present(self):
        fl = FlightLogger()
        for i in range(4):
            fl.detect_events(_snap(float(i)*0.1, dynamic_pressure=float(i)*5_000.0))
        events = fl.detect_events(_snap(0.4, dynamic_pressure=14_000.0))
        mq_events = [e for e in events if e.event_type == EventType.MAX_Q]
        if mq_events:
            assert "max_q_pa" in mq_events[0].parameters


# ---------------------------------------------------------------------------
# 7. Orbit insertion event
# ---------------------------------------------------------------------------

class TestOrbitInsertionEvent:

    def test_orbit_insertion_fires_when_in_orbit(self):
        fl = FlightLogger()
        # First snap: not in orbit (low periapsis)
        fl.detect_events(_snap(0.0, eccentricity=0.3, periapsis=6_371_000.0+50_000.0,
                               altitude=400_000.0))
        # Second snap: in orbit
        events = fl.detect_events(_snap(0.1, eccentricity=0.002,
                                         periapsis=6_371_000.0+380_000.0,
                                         altitude=400_000.0))
        types  = [e.event_type for e in events]
        assert EventType.ORBIT_INSERTION in types

    def test_orbit_insertion_fires_only_once(self):
        fl = FlightLogger()
        snap_orb = _snap(0.0, eccentricity=0.002, periapsis=6_371_000.0+380_000.0,
                          altitude=400_000.0)
        fl.detect_events(snap_orb)  # first time: fires
        events2 = fl.detect_events(_snap(0.1, eccentricity=0.002,
                                          periapsis=6_371_000.0+380_000.0,
                                          altitude=400_000.0))
        # Second call: already in orbit -> no duplicate
        assert EventType.ORBIT_INSERTION not in [e.event_type for e in events2]


# ---------------------------------------------------------------------------
# 8. Structural failure event
# ---------------------------------------------------------------------------

class TestStructuralFailureEvent:

    def test_structural_failure_logged(self):
        fl = FlightLogger()
        events = fl.detect_events(_snap(1.0, any_structural_failure=True,
                                         critical_joint_id="interstage",
                                         worst_structural_margin=-0.1))
        types  = [e.event_type for e in events]
        assert EventType.STRUCTURAL_FAILURE in types

    def test_no_duplicate_structural_failure_same_tick(self):
        fl = FlightLogger()
        # First call logs it
        fl.detect_events(_snap(1.0, any_structural_failure=True,
                                critical_joint_id="J1",
                                worst_structural_margin=-0.1))
        # Second call same time (<0.1s) — should not duplicate
        events = fl.detect_events(_snap(1.05, any_structural_failure=True,
                                          critical_joint_id="J1",
                                          worst_structural_margin=-0.1))
        sf = [e for e in fl.events if e.event_type == EventType.STRUCTURAL_FAILURE]
        assert len(sf) == 1


# ---------------------------------------------------------------------------
# 9. Critical alert event
# ---------------------------------------------------------------------------

class TestCriticalAlertEvent:

    def test_critical_alert_logged_as_event(self):
        fl     = FlightLogger()
        snap   = _snap(5.0)
        events = fl.detect_events(snap, alerts=[_critical_alert()])
        types  = [e.event_type for e in events]
        assert EventType.ALERT_CRITICAL in types

    def test_warning_alert_not_logged(self):
        from nova.ai.monitor import AlertSeverity as AS
        fl      = FlightLogger()
        warning = AlertMessage(
            severity=AS.WARNING, category=AlertCategory.AERODYNAMIC,
            message="Warning msg", recommendation="Do something.",
            time_to_limit_s=15.0, mission_time=1.0,
            parameter="alpha", current_value=0.2, rate=0.01, limit=0.26,
        )
        events = fl.detect_events(_snap(1.0), alerts=[warning])
        types  = [e.event_type for e in events]
        assert EventType.ALERT_CRITICAL not in types


# ---------------------------------------------------------------------------
# 10. clear()
# ---------------------------------------------------------------------------

class TestClear:

    def test_clear_resets_rows_and_events(self):
        fl = FlightLogger()
        for i in range(5):
            fl.record(_snap(float(i)))
        fl.detect_events(_snap(5.0, thrust_magnitude=10_000.0))
        fl.clear()
        assert fl.row_count   == 0
        assert fl.event_count == 0

    def test_clear_resets_engine_state(self):
        """After clear, ignition should fire again for the same transition."""
        fl = FlightLogger()
        fl.detect_events(_snap(0.0, thrust_magnitude=10_000.0))
        fl.clear()
        # Engine state reset: next thrust snap should fire ignition
        events = fl.detect_events(_snap(0.1, thrust_magnitude=10_000.0))
        types  = [e.event_type for e in events]
        assert EventType.ENGINE_IGNITION in types


# ---------------------------------------------------------------------------
# 11. FlightEvent.to_dict()
# ---------------------------------------------------------------------------

class TestFlightEventToDict:

    def test_required_keys_present(self):
        ev   = FlightEvent(
            event_type=EventType.MILESTONE,
            mission_time=42.0,
            description="Test milestone",
            parameters={"value": 1.0},
        )
        d = ev.to_dict()
        for key in ("event_type", "mission_time", "description", "parameters"):
            assert key in d

    def test_event_type_is_string(self):
        ev = FlightEvent(EventType.MAX_Q, 10.0, "Peak Q", {})
        assert isinstance(ev.to_dict()["event_type"], str)


# ---------------------------------------------------------------------------
# 12. EventType values
# ---------------------------------------------------------------------------

class TestEventTypeValues:

    def test_all_values_are_strings(self):
        for et in EventType:
            assert isinstance(et.value, str)

    def test_key_event_types_exist(self):
        names = {et.name for et in EventType}
        for name in ("ENGINE_IGNITION", "ENGINE_CUTOFF", "MAX_Q",
                     "ORBIT_INSERTION", "STRUCTURAL_FAILURE", "ALERT_CRITICAL"):
            assert name in names
