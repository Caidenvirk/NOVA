"""
nova.ai.derivatives
===================
Time-derivative estimators for scalar and vector telemetry attributes.

Architecture role — Pipeline Stage 10 (AI Monitor input)
---------------------------------------------------------
Feeds nova.ai.monitor with rate-of-change quantities computed directly
from the immutable TelemetryRegistry snapshot buffer. No VehicleState
is ever read directly; all data arrives through the registry.

Methods provided
----------------

Scalar derivatives
~~~~~~~~~~~~~~~~~~
``scalar_derivative(registry, attr, n_points)``
    Finite-difference estimate of d(attr)/dt for any float field on
    TelemetrySnapshot. Supports 2-point backward and 3-point central
    difference. Returns None when the buffer is too short.

``scalar_derivative_at(snapshots, attr, n_points)``
    Same but accepts a pre-sliced list of snapshots directly, avoiding
    a second registry query. Used inside the monitor for batch evaluation.

Vector derivatives
~~~~~~~~~~~~~~~~~~
``vector_derivative(registry, attr, n_points)``
    For ndarray fields on TelemetrySnapshot (force_net, position_eci, …).
    Returns a (3,) float64 array [units/s] or None.

Extrapolation utilities
~~~~~~~~~~~~~~~~~~~~~~~
``time_to_limit(value, rate, limit)``
    Given a current value, its time derivative, and a threshold, return
    the number of seconds until the value crosses the limit.
    Returns math.inf if rate is driving away from the limit.

``predict_value(value, rate, dt)``
    Linear extrapolation: value + rate * dt.

Numerical basis
---------------
2-point backward difference (O(dt) accurate):
    dX/dt ≈ (X_n − X_{n-1}) / (t_n − t_{n-1})

3-point central difference (O(dt²) accurate):
    dX/dt ≈ (X_{n+1} − X_{n-1}) / (2·dt)
    implemented as: 0.5·[(X_n−X_{n-2})/dt₁ + (X_n−X_{n-1})/dt₂]

All derivatives are computed on whatever timestamps are actually in the
buffer; unequal spacing is handled correctly.

References
----------
- Fornberg (1988), "Generation of Finite Difference Formulas on
  Arbitrarily Spaced Grids", Mathematics of Computation 51(184).
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from nova.core.telemetry_registry import TelemetryRegistry, TelemetrySnapshot


# ---------------------------------------------------------------------------
# Scalar derivative
# ---------------------------------------------------------------------------

def scalar_derivative(
    registry:  TelemetryRegistry,
    attr:      str,
    n_points:  int = 2,
) -> Optional[float]:
    """
    Estimate d(attr)/dt [units/s] using the last ``n_points`` registry snapshots.

    Parameters
    ----------
    registry : TelemetryRegistry
        Source of telemetry snapshots. Must have ≥ n_points entries.
    attr : str
        Name of a scalar float attribute on TelemetrySnapshot
        (e.g. ``"dynamic_pressure"``, ``"altitude"``, ``"mach"``).
    n_points : int
        Number of snapshots to use.  2 = backward difference (default),
        3 = central difference (higher accuracy).

    Returns
    -------
    float or None
        Estimated derivative, or None if insufficient data or zero Δt.

    Raises
    ------
    ValueError
        If n_points < 2.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be ≥ 2, got {n_points!r}")

    snapshots = registry.last_n(n_points)
    if len(snapshots) < n_points:
        return None

    return scalar_derivative_at(snapshots, attr)


def scalar_derivative_at(
    snapshots: List[TelemetrySnapshot],
    attr:      str,
) -> Optional[float]:
    """
    Estimate d(attr)/dt from a pre-sliced list of snapshots (oldest first).

    Parameters
    ----------
    snapshots : list of TelemetrySnapshot
        Must contain ≥ 2 entries, in chronological order.
    attr : str
        Scalar float attribute name on TelemetrySnapshot.

    Returns
    -------
    float or None
    """
    if len(snapshots) < 2:
        return None

    try:
        values = [float(getattr(s, attr)) for s in snapshots]
        times  = [s.mission_time for s in snapshots]
    except AttributeError:
        return None

    n  = len(snapshots)
    dt = times[-1] - times[0]
    if abs(dt) < 1.0e-12:
        return None

    if n == 2:
        # First-order backward difference
        return (values[-1] - values[-2]) / (times[-1] - times[-2])
    elif n == 3:
        # Second-order: average of two backward differences
        dt1 = times[1] - times[0]
        dt2 = times[2] - times[1]
        if abs(dt1) < 1.0e-12 or abs(dt2) < 1.0e-12:
            return None
        dv1 = (values[1] - values[0]) / dt1
        dv2 = (values[2] - values[1]) / dt2
        return 0.5 * (dv1 + dv2)
    else:
        # Fall back to endpoint difference for n > 3
        return (values[-1] - values[0]) / dt


# ---------------------------------------------------------------------------
# Vector derivative
# ---------------------------------------------------------------------------

def vector_derivative(
    registry:  TelemetryRegistry,
    attr:      str,
    n_points:  int = 2,
) -> Optional[np.ndarray]:
    """
    Estimate d(attr)/dt for a 3-element ndarray telemetry attribute.

    Parameters
    ----------
    registry : TelemetryRegistry
    attr : str
        Name of an ndarray(3,) attribute on TelemetrySnapshot
        (e.g. ``"force_net"``, ``"position_eci"``).
    n_points : int
        Number of snapshots to use (2 or 3).

    Returns
    -------
    ndarray, shape (3,), float64 or None
        Component-wise derivative [units/s].
    """
    if n_points < 2:
        raise ValueError(f"n_points must be ≥ 2, got {n_points!r}")

    snapshots = registry.last_n(n_points)
    if len(snapshots) < n_points:
        return None

    try:
        arrays = [np.asarray(getattr(s, attr), dtype=np.float64) for s in snapshots]
        times  = [s.mission_time for s in snapshots]
    except AttributeError:
        return None

    # Validate shapes
    if any(a.shape != (3,) for a in arrays):
        return None

    dt = times[-1] - times[0]
    if abs(dt) < 1.0e-12:
        return None

    if len(snapshots) == 2:
        return (arrays[-1] - arrays[-2]) / (times[-1] - times[-2])
    elif len(snapshots) == 3:
        dt1 = times[1] - times[0]
        dt2 = times[2] - times[1]
        if abs(dt1) < 1.0e-12 or abs(dt2) < 1.0e-12:
            return None
        dv1 = (arrays[1] - arrays[0]) / dt1
        dv2 = (arrays[2] - arrays[1]) / dt2
        return 0.5 * (dv1 + dv2)
    else:
        return (arrays[-1] - arrays[0]) / dt


# ---------------------------------------------------------------------------
# Extrapolation utilities
# ---------------------------------------------------------------------------

def time_to_limit(
    value: float,
    rate:  float,
    limit: float,
) -> float:
    """
    Estimate time [s] until ``value`` crosses ``limit`` at constant ``rate``.

    If the rate is zero or moving away from the limit, returns math.inf.
    If value has already exceeded the limit, returns 0.0.

    Parameters
    ----------
    value : float
        Current value of the quantity.
    rate : float
        Time derivative [units/s]. Positive = increasing.
    limit : float
        Threshold value that must not be exceeded.

    Returns
    -------
    float
        Estimated seconds to limit, or math.inf if not converging.

    Examples
    --------
    >>> time_to_limit(50_000, 2_000, 100_000)   # q_inf crossing max-Q
    25.0
    >>> time_to_limit(50_000, -500, 100_000)    # q_inf decreasing — no crossing
    inf
    """
    gap = limit - value

    # Already exceeded
    if gap <= 0.0:
        if rate >= 0.0:
            return 0.0         # at or past limit and still increasing
        else:
            return math.inf    # past limit but recovering

    # Rate is zero or moving away
    if rate <= 0.0:
        return math.inf

    return gap / rate


def predict_value(value: float, rate: float, dt: float) -> float:
    """
    Linear extrapolation of a value forward by ``dt`` seconds.

    predicted = value + rate · dt

    Parameters
    ----------
    value : float
        Current value.
    rate : float
        Time derivative [units/s].
    dt : float
        Prediction horizon [s].

    Returns
    -------
    float
    """
    return value + rate * dt


# ---------------------------------------------------------------------------
# Batch derivative pack (used by monitor to evaluate all channels at once)
# ---------------------------------------------------------------------------

def compute_derivative_pack(
    registry:  TelemetryRegistry,
    attrs:     List[str],
    n_points:  int = 2,
) -> dict:
    """
    Compute derivatives for a list of scalar attributes in one pass.

    Parameters
    ----------
    registry : TelemetryRegistry
    attrs : list of str
        Scalar attribute names to differentiate.
    n_points : int

    Returns
    -------
    dict mapping attr → float or None
    """
    snapshots = registry.last_n(n_points)
    result    = {}
    for attr in attrs:
        if len(snapshots) < n_points:
            result[attr] = None
        else:
            result[attr] = scalar_derivative_at(snapshots, attr)
    return result
# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
"""
tests/unit/test_derivatives.py
================================
Unit tests for nova.ai.derivatives.
"""

import math
import pytest
import numpy as np

from nova.core.state_vector import make_state
from nova.core.telemetry_registry import TelemetryRegistry, build_snapshot
from nova.ai.derivatives import (
    scalar_derivative,
    scalar_derivative_at,
    vector_derivative,
    time_to_limit,
    predict_value,
    compute_derivative_pack,
)


def _state(t: float = 0.0):
    return make_state(
        position_eci=[6_771_000.0, 0.0, 0.0],
        velocity_eci=[0.0, 7_672.0, 0.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        omega_body=[0.0, 0.0, 0.0],
        mass=1000.0,
        time=t,
    )


def _filled(n, dt=0.1, kw_fn=None):
    reg = TelemetryRegistry()
    for i in range(n):
        t  = float(i) * dt
        kw = kw_fn(i) if kw_fn else {}
        reg.publish(build_snapshot(_state(t), **kw))
    return reg


class TestScalarDerivative:

    def test_linear_2pt(self):
        reg = _filled(5, kw_fn=lambda i: {"altitude": 1000.0 * i * 0.1})
        d   = scalar_derivative(reg, "altitude", n_points=2)
        assert d is not None and abs(d - 1000.0) < 1.0

    def test_linear_3pt(self):
        reg = _filled(5, kw_fn=lambda i: {"altitude": 1000.0 * i * 0.1})
        d   = scalar_derivative(reg, "altitude", n_points=3)
        assert d is not None and abs(d - 1000.0) < 1.0

    def test_constant_zero_derivative(self):
        reg = _filled(5, kw_fn=lambda i: {"mach": 7.5})
        d   = scalar_derivative(reg, "mach", n_points=2)
        assert d is not None and abs(d) < 1.0e-8

    def test_insufficient_data_none(self):
        reg = _filled(1, kw_fn=lambda i: {"altitude": 500.0})
        assert scalar_derivative(reg, "altitude", n_points=2) is None

    def test_empty_registry_none(self):
        assert scalar_derivative(TelemetryRegistry(), "altitude", n_points=2) is None

    def test_unknown_attr_none(self):
        reg = _filled(3)
        assert scalar_derivative(reg, "no_such_field", n_points=2) is None

    def test_n_points_lt2_raises(self):
        reg = _filled(3)
        with pytest.raises(ValueError, match="n_points"):
            scalar_derivative(reg, "altitude", n_points=1)

    def test_zero_time_delta_none(self):
        reg = TelemetryRegistry()
        reg.publish(build_snapshot(_state(5.0), altitude=1000.0))
        reg.publish(build_snapshot(_state(5.0), altitude=2000.0))
        assert scalar_derivative_at(reg.last_n(2), "altitude") is None

    def test_negative_rate(self):
        reg = _filled(5, kw_fn=lambda i: {"altitude": 10_000.0 - 500.0 * i * 0.1})
        d   = scalar_derivative(reg, "altitude", n_points=2)
        assert d is not None and d < 0.0


class TestVectorDerivative:

    def test_linear_force_derivative(self):
        reg = TelemetryRegistry()
        for i in range(4):
            t  = float(i) * 0.1
            F  = np.array([100.0 * t, 0.0, 0.0], dtype=np.float64)
            reg.publish(build_snapshot(_state(t), force_gravity=F))
        dv = vector_derivative(reg, "force_gravity", n_points=2)
        assert dv is not None and dv.shape == (3,)
        assert abs(dv[0] - 100.0) < 1.0
        assert abs(dv[1]) < 1.0e-8

    def test_constant_vector_zero(self):
        reg = TelemetryRegistry()
        F   = np.array([1000.0, -500.0, 200.0], dtype=np.float64)
        for i in range(3):
            reg.publish(build_snapshot(_state(float(i) * 0.1), force_net=F.copy()))
        dv = vector_derivative(reg, "force_net", n_points=2)
        assert dv is not None and np.allclose(dv, 0.0, atol=1.0e-8)

    def test_insufficient_data_none(self):
        reg = TelemetryRegistry()
        reg.publish(build_snapshot(_state(0.0)))
        assert vector_derivative(reg, "force_net", n_points=2) is None

    def test_unknown_attr_none(self):
        reg = _filled(3)
        assert vector_derivative(reg, "bad_field", n_points=2) is None

    def test_n_points_lt2_raises(self):
        reg = _filled(3)
        with pytest.raises(ValueError, match="n_points"):
            vector_derivative(reg, "force_net", n_points=1)


class TestTimeToLimit:

    def test_basic_crossing(self):
        assert abs(time_to_limit(50_000.0, 2_000.0, 100_000.0) - 25.0) < 1.0e-8

    def test_at_limit_positive_rate(self):
        assert time_to_limit(100_000.0, 500.0, 100_000.0) == 0.0

    def test_exceeded_positive_rate(self):
        assert time_to_limit(110_000.0, 500.0, 100_000.0) == 0.0

    def test_exceeded_negative_rate(self):
        assert time_to_limit(110_000.0, -500.0, 100_000.0) == math.inf

    def test_zero_rate_inf(self):
        assert time_to_limit(50_000.0, 0.0, 100_000.0) == math.inf

    def test_negative_rate_moving_away(self):
        assert time_to_limit(50_000.0, -100.0, 100_000.0) == math.inf

    def test_small_gap_fast_rate(self):
        assert abs(time_to_limit(99_000.0, 10_000.0, 100_000.0) - 0.1) < 1.0e-8


class TestPredictValue:

    def test_linear(self):
        assert predict_value(100.0, 50.0, 2.0) == 200.0

    def test_zero_rate(self):
        assert predict_value(500.0, 0.0, 10.0) == 500.0

    def test_negative_rate(self):
        assert predict_value(100.0, -10.0, 5.0) == 50.0

    def test_zero_dt(self):
        assert predict_value(300.0, 99.0, 0.0) == 300.0


class TestComputeDerivativePack:

    def test_all_channels_returned(self):
        reg   = _filled(5, kw_fn=lambda i: {
            "altitude":         1000.0 * i * 0.1,
            "dynamic_pressure": 500.0  * i * 0.1,
        })
        attrs = ["altitude", "dynamic_pressure", "mach"]
        pack  = compute_derivative_pack(reg, attrs, n_points=2)
        assert set(pack.keys()) == set(attrs)
        assert pack["altitude"] is not None
        assert abs(pack["altitude"] - 1000.0) < 1.0
        assert pack["dynamic_pressure"] is not None

    def test_insufficient_data_all_none(self):
        reg  = _filled(1)
        pack = compute_derivative_pack(reg, ["altitude", "mach"], n_points=2)
        assert pack["altitude"] is None and pack["mach"] is None

    def test_empty_attrs_empty_dict(self):
        reg  = _filled(3)
        pack = compute_derivative_pack(reg, [], n_points=2)
        assert pack == {}
