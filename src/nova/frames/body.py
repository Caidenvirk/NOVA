"""
nova.frames.body
================
Vehicle body frame container for Project NOVA.

Architectural role
------------------
Phase 7 — Frame Definition Classes.
Pipeline stage: Stage 3 (force/torque accumulation), Stage 4 (RK4 integration
attitude propagation), Stage 11 (telemetry). The body frame is where all
aerodynamic forces and angular dynamics are computed.

I/O contract
------------
Input  : VehicleState (quaternion, omega_body, position_eci), MassModel
Output : BodyFrame instance (frozen dataclass) providing CoM-relative geometry,
         body↔ENU DCM, angular velocity properties, and CoM-shifted vectors.

Physical basis
--------------
The Body frame is a right-handed Cartesian system fixed to the vehicle:
  +X → longitudinal axis (forward)
  +Y → lateral axis (right wing)
  +Z → normal axis (down, in standard NED-aligned body convention)

The origin is nominally at the vehicle reference point, but the physically
relevant origin for rigid-body dynamics is the Centre of Mass (CoM). The
MassModel provides the CoM offset (com_body [m]) from the structural
reference point. All moment calculations must apply this offset.

The attitude quaternion q (scalar-first, Hamilton convention) encodes the
rotation from ENU to Body:
  v_body = T_ENU_to_body(q) @ v_enu

Angular velocity omega_body = [p, q, r] (roll, pitch, yaw rates) [rad s⁻¹].

References
----------
- Stevens & Lewis, "Aircraft Control and Simulation", §1.3, §2.1
- Titterton & Weston, "Strapdown Inertial Navigation Technology", §3.3
- Diebel, "Representing Attitude", Stanford Tech Report (2006), §5
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nova.core.constants import (
    EARTH_OMEGA,
    QUATERNION_NORM_TOL,
    TRANSFORM_IDENTITY_TOL,
)
from nova.core.state_vector import VehicleState
from nova.frames.transforms import (
    T_ENU_to_body,
    T_body_to_ENU,
    xi_matrix,
    quaternion_to_euler,
    assert_dcm_orthogonal,
)
from nova.vehicle.mass_model import MassModel

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

DCM = np.ndarray  # shape (3, 3), dtype float64


# ---------------------------------------------------------------------------
# BodyFrame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BodyFrame:
    """
    Immutable snapshot of the vehicle body frame at a single simulation tick.

    Combines the vehicle's attitude quaternion (from VehicleState) with the
    instantaneous Centre of Mass location (from MassModel) to provide a
    complete geometric and kinematic description of the body frame.

    Attributes
    ----------
    quaternion : ndarray, shape (4,), dtype float64
        Unit attitude quaternion (q0, q1, q2, q3) scalar-first.
        Encodes T_ENU→Body rotation.
    omega_body : ndarray, shape (3,), dtype float64
        Angular velocity in body frame [rad s⁻¹]: [p (roll), q (pitch), r (yaw)].
    com_body : ndarray, shape (3,), dtype float64
        Centre of mass position in the body frame [m], relative to the
        structural reference point. Provided by MassModel.
    total_mass : float
        Total vehicle mass [kg] at this tick.
    epoch_time : float
        Mission-elapsed time [s]. Non-negative.
    inertia_body : ndarray, shape (3, 3), dtype float64
        Inertia tensor in the body frame [kg m²] about the CoM.
    inertia_inv : ndarray, shape (3, 3), dtype float64
        Inverse of inertia_body [kg⁻¹ m⁻²].
    """

    quaternion: np.ndarray
    omega_body: np.ndarray
    com_body: np.ndarray
    total_mass: float
    epoch_time: float
    inertia_body: np.ndarray
    inertia_inv: np.ndarray

    def __post_init__(self) -> None:
        # --- quaternion ---
        q = np.asarray(self.quaternion, dtype=np.float64)
        if q.shape != (4,):
            raise ValueError(f"quaternion must have shape (4,); got {q.shape}")
        norm = float(np.linalg.norm(q))
        if abs(norm - 1.0) > QUATERNION_NORM_TOL * 1_000:
            # Use a generous tolerance here; VehicleState enforces tighter
            raise ValueError(
                f"quaternion is not unit: ‖q‖ = {norm:.10f} "
                f"(tolerance {QUATERNION_NORM_TOL:.1e})"
            )
        object.__setattr__(self, "quaternion", q)

        # --- omega_body ---
        omega = np.asarray(self.omega_body, dtype=np.float64)
        if omega.shape != (3,):
            raise ValueError(f"omega_body must have shape (3,); got {omega.shape}")
        object.__setattr__(self, "omega_body", omega)

        # --- com_body ---
        com = np.asarray(self.com_body, dtype=np.float64)
        if com.shape != (3,):
            raise ValueError(f"com_body must have shape (3,); got {com.shape}")
        object.__setattr__(self, "com_body", com)

        # --- total_mass ---
        m = float(self.total_mass)
        if m <= 0.0:
            raise ValueError(f"total_mass must be positive; got {m:.6g}")
        object.__setattr__(self, "total_mass", m)

        # --- epoch_time ---
        t = float(self.epoch_time)
        if t < 0.0:
            raise ValueError(f"epoch_time must be non-negative; got {t:.6g}")
        object.__setattr__(self, "epoch_time", t)

        # --- inertia_body ---
        I = np.asarray(self.inertia_body, dtype=np.float64)
        if I.shape != (3, 3):
            raise ValueError(f"inertia_body must have shape (3, 3); got {I.shape}")
        # Check symmetry
        asym = float(np.max(np.abs(I - I.T)))
        if asym > 1.0e-6 * float(np.max(np.abs(I))):
            raise ValueError(
                f"inertia_body is not symmetric: max asymmetry = {asym:.3e}"
            )
        object.__setattr__(self, "inertia_body", I)

        # --- inertia_inv ---
        Iinv = np.asarray(self.inertia_inv, dtype=np.float64)
        if Iinv.shape != (3, 3):
            raise ValueError(f"inertia_inv must have shape (3, 3); got {Iinv.shape}")
        object.__setattr__(self, "inertia_inv", Iinv)

    # ------------------------------------------------------------------
    # Attitude: quaternion-derived properties
    # ------------------------------------------------------------------

    @property
    def dcm_enu_to_body(self) -> DCM:
        """
        Direction cosine matrix T_ENU→Body from the attitude quaternion.

        Returns
        -------
        ndarray, shape (3, 3), dtype float64
        """
        T = T_ENU_to_body(self.quaternion)
        assert_dcm_orthogonal(T, "T_ENU_to_body")
        return T

    @property
    def dcm_body_to_enu(self) -> DCM:
        """
        Direction cosine matrix T_Body→ENU (transpose of T_ENU→Body).

        Returns
        -------
        ndarray, shape (3, 3), dtype float64
        """
        T = T_body_to_ENU(self.quaternion)
        assert_dcm_orthogonal(T, "T_body_to_ENU")
        return T

    @property
    def euler_angles(self) -> tuple[float, float, float]:
        """
        ZYX Tait-Bryan Euler angles (roll, pitch, yaw) [rad].

        These are derived from the quaternion for **display purposes only**.
        All internal physics use the quaternion representation.

        Returns
        -------
        (roll, pitch, yaw) : tuple[float, float, float]
            Angles in radians.
        """
        return quaternion_to_euler(self.quaternion)

    @property
    def roll_rad(self) -> float:
        """Roll angle φ [rad]."""
        r, _, _ = self.euler_angles
        return r

    @property
    def pitch_rad(self) -> float:
        """Pitch angle θ [rad]."""
        _, p, _ = self.euler_angles
        return p

    @property
    def yaw_rad(self) -> float:
        """Yaw angle ψ [rad]."""
        _, _, y = self.euler_angles
        return y

    @property
    def xi(self) -> np.ndarray:
        """
        Quaternion kinematic matrix Ξ(q), shape (4, 3).

        Used in the attitude ODE: dq/dt = ½ · Ξ(q) · ω_body
        """
        return xi_matrix(self.quaternion)

    @property
    def qdot(self) -> np.ndarray:
        """
        Quaternion time derivative dq/dt, shape (4,).

        dq/dt = ½ · Ξ(q) · ω_body

        Returns
        -------
        ndarray, shape (4,), dtype float64
        """
        return 0.5 * (self.xi @ self.omega_body)

    # ------------------------------------------------------------------
    # Angular kinematic properties
    # ------------------------------------------------------------------

    @property
    def roll_rate(self) -> float:
        """Roll rate p [rad s⁻¹]."""
        return float(self.omega_body[0])

    @property
    def pitch_rate(self) -> float:
        """Pitch rate q [rad s⁻¹]."""
        return float(self.omega_body[1])

    @property
    def yaw_rate(self) -> float:
        """Yaw rate r [rad s⁻¹]."""
        return float(self.omega_body[2])

    @property
    def angular_speed(self) -> float:
        """Total angular speed ‖ω‖ [rad s⁻¹]."""
        return float(np.linalg.norm(self.omega_body))

    # ------------------------------------------------------------------
    # Mass / inertia properties
    # ------------------------------------------------------------------

    @property
    def Ixx(self) -> float:
        """Principal roll moment of inertia [kg m²]."""
        return float(self.inertia_body[0, 0])

    @property
    def Iyy(self) -> float:
        """Principal pitch moment of inertia [kg m²]."""
        return float(self.inertia_body[1, 1])

    @property
    def Izz(self) -> float:
        """Principal yaw moment of inertia [kg m²]."""
        return float(self.inertia_body[2, 2])

    @property
    def angular_momentum_body(self) -> np.ndarray:
        """
        Angular momentum vector in the body frame h = I · ω [kg m² s⁻¹].

        Returns
        -------
        ndarray, shape (3,), dtype float64
        """
        return (self.inertia_body @ self.omega_body).astype(np.float64)

    # ------------------------------------------------------------------
    # CoM-relative geometry
    # ------------------------------------------------------------------

    def shift_to_com(self, point_body: np.ndarray) -> np.ndarray:
        """
        Express a body-frame point relative to the CoM instead of the
        structural reference origin.

        r_from_com = r_body − com_body

        Parameters
        ----------
        point_body : array_like, shape (3,)
            Point in body frame relative to structural reference [m].

        Returns
        -------
        ndarray, shape (3,), dtype float64
            Offset vector from CoM to point [m].
        """
        return (np.asarray(point_body, dtype=np.float64) - self.com_body).astype(np.float64)

    def moment_arm(self, force_application_body: np.ndarray) -> np.ndarray:
        """
        Compute the moment arm (lever arm) from the CoM to a force
        application point.

        r_arm = point_body − com_body

        Parameters
        ----------
        force_application_body : array_like, shape (3,)
            Point where the force is applied, in body frame [m].

        Returns
        -------
        ndarray, shape (3,), dtype float64
            Lever arm vector [m].
        """
        return self.shift_to_com(force_application_body)

    def rotate_to_enu(self, vector_body: np.ndarray) -> np.ndarray:
        """
        Rotate a body-frame vector to the ENU frame.

        Parameters
        ----------
        vector_body : array_like, shape (3,)

        Returns
        -------
        ndarray, shape (3,), dtype float64
        """
        T = T_body_to_ENU(self.quaternion)
        return (T @ np.asarray(vector_body, dtype=np.float64)).astype(np.float64)

    def rotate_from_enu(self, vector_enu: np.ndarray) -> np.ndarray:
        """
        Rotate an ENU vector into the body frame.

        Parameters
        ----------
        vector_enu : array_like, shape (3,)

        Returns
        -------
        ndarray, shape (3,), dtype float64
        """
        T = T_ENU_to_body(self.quaternion)
        return (T @ np.asarray(vector_enu, dtype=np.float64)).astype(np.float64)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        r, p, y = (math.degrees(a) for a in self.euler_angles)
        return (
            f"BodyFrame(t={self.epoch_time:.3f}s, m={self.total_mass:.2f}kg, "
            f"roll={r:.2f}°, pitch={p:.2f}°, yaw={y:.2f}°, "
            f"ω=[{self.roll_rate:.4f},{self.pitch_rate:.4f},{self.yaw_rate:.4f}] rad/s)"
        )


# ---------------------------------------------------------------------------
# Convenience constructor from VehicleState + MassModel
# ---------------------------------------------------------------------------

def body_from_state(
    state: VehicleState,
    mass_model: MassModel,
) -> BodyFrame:
    """
    Construct a BodyFrame from a VehicleState and a MassModel snapshot.

    This is the canonical factory used inside the pipeline after the mass
    model has been evaluated for the current tick.

    Parameters
    ----------
    state : VehicleState
        Current simulation state (frozen). Uses state.quaternion,
        state.omega_body, state.mass, state.time.
    mass_model : MassModel
        Evaluated mass model for the current tick. Provides com_body,
        inertia_body, inertia_inv.

    Returns
    -------
    BodyFrame
    """
    return BodyFrame(
        quaternion=state.quaternion.copy(),
        omega_body=state.omega_body.copy(),
        com_body=mass_model.com_body.copy(),
        total_mass=mass_model.total_mass,
        epoch_time=state.time,
        inertia_body=mass_model.inertia_body.copy(),
        inertia_inv=mass_model.inertia_inv.copy(),
    )
