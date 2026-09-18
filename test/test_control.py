import math

import numpy as np
import pytest

from desafio_gazebo_pria.control import (
    ControlConfig,
    PIDConfig,
    PIDState,
    control_sector_ranges,
    pid_angular_command,
    safe_twist,
    target_twist,
)
from desafio_gazebo_pria.safety_supervisor import SafetyState


def test_pid_clamps_integral_and_angular_output():
    config = PIDConfig(
        kp=3.0,
        ki=2.0,
        kd=0.0,
        integral_limit=0.2,
        max_angular_speed=0.7,
    )
    command, state = pid_angular_command(1.0, PIDState(), 1.0, config)
    assert command == 0.7
    assert state.integral == 0.2


def test_target_twist_turns_in_place_until_aligned():
    output, _ = target_twist(
        (0.0, 0.0, 0.0),
        (1.0, 1.0),
        PIDState(),
        0.1,
        ControlConfig(),
    )
    assert output.linear_x == 0.0 and output.angular_z > 0.0 and not output.reached


def test_safe_twist_stops_forward_motion_and_turns_to_clearer_side():
    decision = safe_twist(
        0.15,
        0.0,
        [0.18, 0.30, 1.20, 1.0],
        False,
        ControlConfig(),
    )
    assert decision.linear_x == 0.0 and decision.angular_z < 0.0 and decision.intervention


def test_safe_twist_turns_away_from_a_blocked_front_without_forward_command():
    decision = safe_twist(0.0, 0.0, [0.18, 0.30, 1.20, 1.0], False, ControlConfig())
    assert decision.angular_z < 0.0 and decision.intervention


def test_control_sector_ranges_returns_numpy_array():
    sectors = control_sector_ranges([4.0, 3.0, 2.0, 1.0], -math.pi, math.pi / 2.0)
    assert isinstance(sectors, np.ndarray)
    assert np.allclose(sectors, [2.0, 1.0, 3.0, 4.0])


def test_safe_twist_blocks_missing_sector_data():
    decision = safe_twist(0.15, 0.0, [1.0, 1.0, 1.0], False, ControlConfig())
    assert decision.linear_x == 0.0 and decision.angular_z == 0.0 and decision.intervention


def test_safe_twist_blocks_nonfinite_sector_data():
    decision = safe_twist(0.15, 0.0, [1.0, math.nan, 1.0, 1.0], False, ControlConfig())
    assert decision.linear_x == 0.0 and decision.angular_z == 0.0 and decision.intervention


def test_safe_twist_does_not_reverse_when_rear_is_blocked():
    config = ControlConfig()
    decision = safe_twist(0.15, 0.0, [1.0, 0.30, 1.20, 0.10], True, config)
    assert decision.linear_x == 0.0 and decision.recovery and decision.intervention


def test_safe_twist_reverses_only_during_recovery_with_safe_rear_clearance():
    config = ControlConfig()
    decision = safe_twist(0.15, 0.0, [1.0, 0.30, 1.20, 1.0], True, config)
    assert decision.linear_x == -config.recovery_linear_speed
    assert decision.recovery and decision.intervention


def test_safe_twist_blocks_nonrecovery_reverse_command():
    decision = safe_twist(-0.15, 0.0, [1.0, 1.0, 1.0, 1.0], False, ControlConfig())
    assert decision.linear_x == 0.0 and decision.angular_z == 0.0 and decision.intervention


def test_contact_recovers_immediately_and_uses_separate_rear_margin():
    config = ControlConfig(contact_distance=0.12, rear_clearance=0.35)
    rear_blocked = safe_twist(0.1, 0.0, [0.1, 1.0, 0.5, 0.30], False, config)
    rear_clear = safe_twist(0.1, 0.0, [0.1, 1.0, 0.5, 0.50], False, config)
    assert rear_blocked.recovery and rear_blocked.linear_x == 0.0
    assert rear_clear.recovery and rear_clear.linear_x < 0.0


def test_lateral_clearance_slows_forward_speed():
    result = safe_twist(0.2, 0.0, [1.0, 0.18, 1.0, 1.0], False, ControlConfig())
    assert 0.0 < result.linear_x < 0.2 and result.intervention


def test_minimum_forward_speed_and_arrival_slowdown():
    config = ControlConfig(max_linear_speed=0.20, min_linear_speed=0.05, distance_gain=1.0)
    far, _ = target_twist((0.0, 0.0, 0.0), (1.0, 0.0), PIDState(), 0.1, config)
    near, _ = target_twist((0.0, 0.0, 0.0), (0.11, 0.0), PIDState(), 0.1, config)
    assert far.linear_x == config.max_linear_speed
    assert near.linear_x == config.min_linear_speed
    assert 0.0 < near.linear_x < far.linear_x


def test_target_twist_never_exceeds_max_linear_speed_when_min_is_misconfigured():
    config = ControlConfig(max_linear_speed=0.10, min_linear_speed=0.20, distance_gain=1.0)

    output, _ = target_twist(
        (0.0, 0.0, 0.0),
        (1.0, 0.0),
        PIDState(),
        0.1,
        config,
    )

    assert output.linear_x == config.max_linear_speed


def test_unobstructed_target_reaches_before_safety_stall_recovery():
    """Tapering the minimum speed below useful motion leaves arrival unreachable."""
    config = ControlConfig()
    safety = SafetyState(
        config=config,
        sensor_timeout=1.0,
        stall_timeout=2.0,
        progress_distance=0.05,
    )
    pid_state = PIDState()
    target = (0.4, 0.0)
    pose = (0.0, 0.0, 0.0)
    dt = 0.05
    now = 0.0

    for _ in range(400):
        safety.set_scan([1.0, 1.0, 1.0, 1.0], stamp=now)
        safety.set_pose(pose, stamp=now)
        output, pid_state = target_twist(pose, target, pid_state, dt, config)
        safety.set_nominal(output.linear_x, output.angular_z, stamp=now)

        if output.reached:
            break

        decision = safety.compute(now)
        if decision.recovery:
            pytest.fail(
                "safety recovery started before target_reached with "
                f"{target[0] - pose[0]:.6f} m remaining"
            )

        pose = (
            pose[0] + decision.linear_x * math.cos(pose[2]) * dt,
            pose[1] + decision.linear_x * math.sin(pose[2]) * dt,
            pose[2] + decision.angular_z * dt,
        )
        now += dt
    else:
        pytest.fail("target was not reached within the simulated control window")

    assert output.reached


def test_pid_filters_derivative_step():
    config = PIDConfig(kp=0.0, ki=0.0, kd=1.0, derivative_alpha=0.25)
    output, _ = pid_angular_command(1.0, PIDState(previous_error=0.0), 1.0, config)
    assert output == 0.25


@pytest.mark.parametrize("field", ("pose", "target"))
def test_pure_target_control_rejects_nonfinite_values(field):
    pose = (math.nan, 0.0, 0.0) if field == "pose" else (0.0, 0.0, 0.0)
    target = (math.inf, 0.0) if field == "target" else (1.0, 0.0)
    output, state = target_twist(pose, target, PIDState(), 0.1, ControlConfig())
    assert (output.linear_x, output.angular_z) == (0.0, 0.0)


def test_invalid_scan_geometry_does_not_invent_rear_measurements():
    assert np.all(np.isnan(control_sector_ranges([1.0] * 4, math.nan, 0.1)))
