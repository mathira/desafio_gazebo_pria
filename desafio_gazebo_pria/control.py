"""Pure PID goal-seeking and LiDAR safety decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class PIDConfig:
    kp: float = 1.5
    ki: float = 0.0
    kd: float = 0.1
    integral_limit: float = 0.5
    max_angular_speed: float = 1.5
    derivative_alpha: float = 0.2


@dataclass(frozen=True)
class PIDState:
    integral: float = 0.0
    previous_error: float | None = None
    derivative: float = 0.0


@dataclass(frozen=True)
class ControlConfig:
    pid: PIDConfig = field(default_factory=PIDConfig)
    max_linear_speed: float = 0.22
    arrival_tolerance: float = 0.10
    alignment_tolerance: float = 0.35
    distance_gain: float = 0.5
    stop_distance: float = 0.20
    recovery_linear_speed: float = 0.10
    min_linear_speed: float = 0.04
    contact_distance: float = 0.12
    rear_clearance: float = 0.30
    lateral_slow_distance: float = 0.30
    front_slow_distance: float = 0.45


@dataclass(frozen=True)
class PIDOutput:
    linear_x: float
    angular_z: float
    reached: bool


@dataclass(frozen=True)
class TwistDecision:
    linear_x: float
    angular_z: float
    intervention: bool
    recovery: bool = False


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the inclusive interval [-pi, pi]."""
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi and angle > 0.0 else wrapped


def control_sector_ranges(
    ranges: Sequence[float] | np.ndarray,
    angle_min: float,
    angle_increment: float,
) -> np.ndarray:
    """Return minimum [front, left, right, rear] clearance for a scan.

    Missing or invalid sectors remain ``nan`` so the supervisor can treat
    them as unsafe instead of mistaking absent measurements for open space.
    """
    sector_ranges = [math.nan] * 4
    if not math.isfinite(angle_min) or not math.isfinite(angle_increment):
        return np.asarray(sector_ranges)
    measurements = np.asarray(ranges, dtype=float).reshape(-1)
    for index, measured in enumerate(measurements):
        if not math.isfinite(measured) or measured <= 0.0:
            continue
        angle = wrap_angle(float(angle_min) + index * float(angle_increment))
        if -math.pi / 4.0 <= angle <= math.pi / 4.0:
            sector = 0
        elif math.pi / 4.0 < angle < 3.0 * math.pi / 4.0:
            sector = 1
        elif -3.0 * math.pi / 4.0 < angle < -math.pi / 4.0:
            sector = 2
        else:
            sector = 3
        previous = sector_ranges[sector]
        sector_ranges[sector] = measured if math.isnan(previous) else min(previous, measured)
    return np.asarray(sector_ranges, dtype=float)


def pid_angular_command(
    error: float,
    state: PIDState,
    dt: float,
    config: PIDConfig,
) -> tuple[float, PIDState]:
    """Apply bounded PID control and return an immutable next state."""
    if not all(math.isfinite(v) for v in (error, dt, state.integral, state.derivative)):
        return 0.0, PIDState()
    safe_dt = max(float(dt), 1e-6)
    integral = max(
        -config.integral_limit,
        min(config.integral_limit, state.integral + error * safe_dt),
    )
    derivative = (
        0.0
        if state.previous_error is None
        else wrap_angle(error - state.previous_error) / safe_dt
    )
    derivative = config.derivative_alpha * derivative + (1.0 - config.derivative_alpha) * state.derivative
    command = config.kp * error + config.ki * integral + config.kd * derivative
    return max(-config.max_angular_speed, min(config.max_angular_speed, command)), PIDState(integral, error, derivative)


def target_twist(
    pose: tuple[float, float, float],
    target: tuple[float, float],
    state: PIDState,
    dt: float,
    config: ControlConfig,
) -> tuple[PIDOutput, PIDState]:
    """Return a target-seeking twist, turning in place before driving."""
    if not all(math.isfinite(v) for v in (*pose, *target, dt)):
        return PIDOutput(0.0, 0.0, False), PIDState()
    dx, dy = target[0] - pose[0], target[1] - pose[1]
    distance = math.hypot(dx, dy)
    if distance <= config.arrival_tolerance:
        return PIDOutput(0.0, 0.0, True), PIDState()
    error = wrap_angle(math.atan2(dy, dx) - pose[2])
    angular, next_state = pid_angular_command(error, state, dt, config.pid)
    if abs(error) > config.alignment_tolerance:
        linear = 0.0
    else:
        base_linear = min(config.max_linear_speed, max(config.min_linear_speed, config.distance_gain * distance))
        tapered_linear = base_linear * min(
            1.0,
            (distance - config.arrival_tolerance) / max(config.arrival_tolerance, 1e-6),
        )
        linear = min(config.max_linear_speed, max(config.min_linear_speed, tapered_linear))
    return PIDOutput(linear, angular, False), next_state


def safe_twist(
    nominal_linear: float,
    nominal_angular: float,
    sectors: np.ndarray,
    stalled: bool,
    config: ControlConfig,
) -> TwistDecision:
    """Apply conservative obstacle and recovery constraints to a twist."""
    values = np.asarray(sectors, dtype=float).reshape(-1).tolist()
    if len(values) < 4:
        return TwistDecision(0.0, 0.0, intervention=True)
    front, left, right, rear = values[:4]
    if (not all(math.isfinite(value) for value in (front, left, right, rear, nominal_linear, nominal_angular))
            or min(front, left, right, rear) <= 0.0):
        return TwistDecision(0.0, 0.0, intervention=True)

    turn = config.pid.max_angular_speed if left >= right else -config.pid.max_angular_speed
    if stalled or front <= config.contact_distance:
        if rear > config.rear_clearance:
            return TwistDecision(
                -config.recovery_linear_speed,
                turn,
                intervention=True,
                recovery=True,
            )
        return TwistDecision(0.0, turn, intervention=True, recovery=True)
    if front < config.stop_distance:
        return TwistDecision(0.0, turn, intervention=True)
    if nominal_linear < 0.0:
        return TwistDecision(0.0, float(nominal_angular), intervention=True)
    lateral_scale = max(0.0, min(1.0, (min(left, right) - config.contact_distance)
                                / (config.lateral_slow_distance - config.contact_distance)))
    front_scale = max(0.0, min(1.0, (front - config.stop_distance)
                              / (config.front_slow_distance - config.stop_distance)))
    linear = min(nominal_linear, config.max_linear_speed) * min(lateral_scale, front_scale)
    angular = max(-config.pid.max_angular_speed, min(config.pid.max_angular_speed, nominal_angular))
    return TwistDecision(float(linear), float(angular), intervention=linear != nominal_linear or angular != nominal_angular)
