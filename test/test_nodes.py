import math

import pytest
import numpy as np

from desafio_gazebo_pria.coverage_mapper import MappingState
from desafio_gazebo_pria.frontier_planner import PlannerState
from desafio_gazebo_pria.pid_controller import PIDControllerState
from desafio_gazebo_pria.safety_supervisor import SafetyState
from desafio_gazebo_pria.grid_mapping import GridSpec


def test_planner_consumes_reached_and_recovery_without_repeating_targets():
    state = PlannerState(max_replans=3)
    grid = np.zeros((9, 9), dtype=np.int8)
    monitoring = np.zeros_like(grid)
    monitoring[0, :] = -1
    monitoring[-1, :] = -1
    spec = GridSpec(9, 9, 1.0, 0.0, 0.0)
    first = state.plan(grid, monitoring, spec, (4.5, 4.5, 0.0), 0, 2)
    assert first is not None
    assert state.plan(grid, monitoring, spec, (4.5, 4.5, 0.0), 0, 2) == first
    state.invalidate_target()
    second = state.plan(grid, monitoring, spec, (4.5, 4.5, 0.0), 0, 2)
    assert second is not None and second != first
    state.invalidate_target()
    third = state.plan(grid, monitoring, spec, (4.5, 4.5, 0.0), 0, 2)
    assert third not in (None, first, second)
    state.invalidate_target()
    assert state.plan(grid, monitoring, spec, (4.5, 4.5, 0.0), 0, 2) is None
    assert state.finished


def test_pid_rechecks_straight_segment_when_map_obstacle_appears():
    state = PIDControllerState()
    state.set_pose((1.5, 1.5, 0.0), 0.0)
    state.set_target((3.5, 1.5), 0.0)
    state.set_scan([2.0] * 4, 0.0)
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1, 2] = 100
    state.set_map(grid, GridSpec(5, 5, 1.0, 0.0, 0.0))
    result = state.compute(0.0)
    assert (result.linear_x, result.angular_z) == (0.0, 0.0)


def test_pid_fuses_imu_heading_and_stops_when_required_imu_is_stale():
    state = PIDControllerState(require_imu=True)
    state.set_pose((0.0, 0.0, 0.0), 0.0)
    state.set_target((1.0, 0.0), 0.0)
    state.set_scan([2.0] * 4, 0.0)
    assert state.compute(0.0).linear_x == 0.0
    state.set_imu(0.5, 0.0)
    assert state.compute(0.0).angular_z < 0.0
    state.set_pose((0.0, 0.0, 0.0), 1.0)
    state.set_target((1.0, 0.0), 1.0)
    state.set_scan([2.0] * 4, 1.0)
    output = state.compute(1.0)
    assert (output.linear_x, output.angular_z) == (0.0, 0.0)


def test_mapping_state_rejects_scan_until_pose_exists():
    assert MappingState().integrate_scan([1.0], 0.0, 0.1, 5.0) == 0


def test_planner_state_finishes_when_coverage_reaches_target():
    state = PlannerState(target_coverage=0.9)

    assert state.update_coverage(0.9) and state.finished


def test_planner_state_remains_finished_after_coverage_regresses():
    state = PlannerState(target_coverage=0.9)

    assert state.update_coverage(0.9)
    assert state.update_coverage(0.1)
    assert state.finished


def test_planner_state_finishes_when_mission_duration_expires():
    """An unattended mission must publish its terminal state at its time budget."""
    state = PlannerState(target_coverage=0.9, max_mission_seconds=5.0)

    assert not state.update_time(10.0)
    assert not state.update_time(14.99)
    assert state.update_time(15.0)
    assert state.finished


@pytest.mark.parametrize("duration", (0.0, -1.0, math.inf, math.nan))
def test_planner_state_rejects_non_positive_or_non_finite_mission_duration(duration):
    """An invalid budget would make terminal safety behavior undefined."""
    with pytest.raises(ValueError):
        PlannerState(max_mission_seconds=duration)


def test_steady_deadline_expires_without_simulated_time_updates():
    """A stopped simulation clock must not let the external safety cap run forever."""
    from desafio_gazebo_pria.frontier_planner import MissionDeadline

    deadline = MissionDeadline(max_mission_seconds=5.0)

    assert not deadline.update_time(100.0)
    assert deadline.update_time(105.0)


def test_terminal_endpoints_construct_with_durable_completion_qos(monkeypatch):
    """Late PID or safety startup must receive an already-published terminal event."""
    rclpy = pytest.importorskip("rclpy")
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
    from desafio_gazebo_pria.frontier_planner import FrontierPlanner
    from desafio_gazebo_pria.pid_controller import PIDController
    from desafio_gazebo_pria.safety_supervisor import SafetySupervisor

    records = []
    create_publisher = Node.create_publisher
    create_subscription = Node.create_subscription

    def record_publisher(node, message_type, topic, qos_profile, *args, **kwargs):
        if topic == "/mission_finished":
            records.append(("publisher", qos_profile))
        return create_publisher(node, message_type, topic, qos_profile, *args, **kwargs)

    def record_subscription(node, message_type, topic, callback, qos_profile, *args, **kwargs):
        if topic == "/mission_finished":
            records.append(("subscription", qos_profile))
        return create_subscription(
            node, message_type, topic, callback, qos_profile, *args, **kwargs
        )

    monkeypatch.setattr(Node, "create_publisher", record_publisher)
    monkeypatch.setattr(Node, "create_subscription", record_subscription)
    rclpy.init()
    nodes = [FrontierPlanner(), PIDController(), SafetySupervisor()]
    try:
        assert [kind for kind, _ in records] == [
            "publisher",
            "subscription",
            "subscription",
        ]
        for _, qos_profile in records:
            assert qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
            assert qos_profile.reliability == ReliabilityPolicy.RELIABLE
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()


def test_terminal_callbacks_publish_zero_without_waiting_for_sim_time():
    """A frozen `/clock` must not leave either node's last command active."""
    rclpy = pytest.importorskip("rclpy")
    from std_msgs.msg import Bool
    from desafio_gazebo_pria.pid_controller import PIDController
    from desafio_gazebo_pria.safety_supervisor import SafetySupervisor

    class Collector:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    rclpy.init()
    pid = PIDController()
    safety = SafetySupervisor()
    pid_output = Collector()
    safety_output = Collector()
    pid._publisher = pid_output
    safety._publisher = safety_output
    try:
        terminal = Bool(data=True)
        pid._on_finished(terminal)
        safety._on_finished(terminal)

        assert pid._state.finished and safety._state.finished
        assert [(message.linear.x, message.angular.z) for message in pid_output.messages] == [
            (0.0, 0.0)
        ]
        assert [
            (message.linear.x, message.angular.z) for message in safety_output.messages
        ] == [(0.0, 0.0)]
    finally:
        pid.destroy_node()
        safety.destroy_node()
        rclpy.shutdown()


def test_supervisor_returns_zero_when_scan_is_stale():
    """Removing the scan freshness guard would allow blind motion."""
    state = SafetyState(sensor_timeout=0.5)
    state.set_nominal(0.1, 0.0, stamp=10.0)
    state.set_scan([2.0, 2.0, 2.0, 2.0], stamp=10.0)
    state.set_pose((0.0, 0.0, 0.0), stamp=10.0)

    decision = state.compute(now=10.6)

    assert decision.linear_x == 0.0 and decision.angular_z == 0.0


def test_supervisor_returns_zero_for_stale_nominal_without_a_stall_window():
    """A stopped PID stream must not leave its last forward command active."""
    state = SafetyState(sensor_timeout=1.0, stall_timeout=2.0)
    state.set_nominal(0.1, 0.0, stamp=0.0)
    state.set_pose((0.0, 0.0, 0.0), stamp=3.0)
    state.set_scan([1.0, 1.0, 1.0, 1.0], stamp=3.0)

    decision = state.compute(now=3.0)

    assert decision.linear_x == 0.0 and decision.angular_z == 0.0


def test_supervisor_requests_recovery_after_stalled_forward_command():
    """Removing the insufficient-progress check would leave a blocked robot stuck."""
    state = SafetyState(sensor_timeout=1.0, stall_timeout=2.0)
    state.set_nominal(0.1, 0.0, stamp=0.0)
    state.set_pose((0.0, 0.0, 0.0), stamp=0.0)
    for stamp in (0.5, 1.0, 1.5, 2.0, 2.1):
        state.set_nominal(0.1, 0.0, stamp=stamp)
        state.set_scan([1.0, 0.5, 1.0, 1.0], stamp=stamp)
        state.set_pose((0.01, 0.0, 0.0), stamp=stamp)
        decision = state.compute(now=stamp)

    assert decision.recovery


def test_supervisor_cannot_restart_recovery_from_expired_command():
    state = SafetyState(sensor_timeout=0.5, stall_timeout=2.0)
    state.set_pose((0.0, 0.0, 0.0), stamp=0.0)
    state.set_nominal(0.1, 0.0, stamp=0.0)
    state.set_scan([1.0] * 4, stamp=2.1)
    state.set_pose((0.0, 0.0, 0.0), stamp=2.1)
    decision = state.compute(2.1)
    assert (decision.linear_x, decision.angular_z, decision.recovery) == (0.0, 0.0, False)


@pytest.mark.parametrize("kind", ("pose", "scan", "target", "time"))
def test_pid_rejects_nonfinite_control_state(kind):
    state = PIDControllerState()
    state.set_target((1.0, 0.0), 0.0)
    state.set_pose((0.0, 0.0, 0.0), 0.0)
    state.set_scan([1.0] * 4, 0.0)
    if kind == "pose":
        state.set_pose((math.nan, 0.0, 0.0), 0.0)
    if kind == "target":
        state.set_target((1.0, math.inf), 0.0)
    if kind == "scan":
        state.set_scan([1.0, math.nan, 1.0, 1.0], 0.0)
    output = state.compute(math.nan if kind == "time" else 0.0)
    assert (output.linear_x, output.angular_z, output.reached) == (0.0, 0.0, False)


def test_safety_rejects_nonfinite_pose():
    state = SafetyState()
    state.set_pose((math.nan, 0.0, 0.0), 0.0)
    state.set_scan([1.0] * 4, 0.0)
    state.set_nominal(0.1, 0.0, 0.0)
    decision = state.compute(0.0)
    assert (decision.linear_x, decision.angular_z) == (0.0, 0.0)


def test_supervisor_returns_zero_after_mission_finished():
    """Removing the terminal-state guard would permit motion after completion."""
    state = SafetyState(sensor_timeout=1.0)
    state.set_nominal(0.1, 0.0, stamp=3.0)
    state.set_scan([1.0, 1.0, 1.0, 1.0], stamp=3.0)
    state.set_pose((0.0, 0.0, 0.0), stamp=3.0)
    state.set_finished(True)

    decision = state.compute(now=3.0)

    assert decision.linear_x == 0.0 and decision.angular_z == 0.0


def test_supervisor_returns_zero_without_a_nominal_command():
    """Passing an absent nominal command to safety arbitration must not crash."""
    state = SafetyState(sensor_timeout=1.0)
    state.set_scan([1.0, 1.0, 1.0, 1.0], stamp=3.0)
    state.set_pose((0.0, 0.0, 0.0), stamp=3.0)

    decision = state.compute(now=3.0)

    assert decision.linear_x == 0.0 and decision.angular_z == 0.0


def test_pid_state_resets_on_replaced_target_and_arrival():
    """Keeping PID memory across targets can command a turn in the wrong direction."""
    state = PIDControllerState(sensor_timeout=1.0)
    state.set_pose((0.0, 0.0, 0.0), stamp=0.0)
    state.set_scan([1.0, 1.0, 1.0, 1.0], stamp=0.0)
    state.set_target((0.0, 1.0), stamp=0.0)
    state.compute(now=0.1)
    assert state.pid_state.previous_error is not None

    state.set_target((1.0, 0.0), stamp=0.1)
    assert state.pid_state.previous_error is None


def test_pid_state_keeps_history_for_an_unchanged_target():
    """Repeated planner publications must not erase PID integral/derivative state."""
    state = PIDControllerState(sensor_timeout=1.0)
    state.set_pose((0.0, 0.0, 0.0), stamp=0.0)
    state.set_scan([1.0, 1.0, 1.0, 1.0], stamp=0.0)
    state.set_target((0.0, 1.0), stamp=0.0)
    state.compute(now=0.1)
    previous_pid_state = state.pid_state
    previous_compute = state._last_compute

    state.set_target((0.0, 1.0), stamp=0.2)

    assert state.pid_state == previous_pid_state
    assert state._last_compute == previous_compute


def test_pid_state_resets_history_for_a_changed_target():
    """Changing the target must discard PID state calculated for the old heading."""
    state = PIDControllerState(sensor_timeout=1.0)
    state.set_pose((0.0, 0.0, 0.0), stamp=0.0)
    state.set_scan([1.0, 1.0, 1.0, 1.0], stamp=0.0)
    state.set_target((0.0, 1.0), stamp=0.0)
    state.compute(now=0.1)

    state.set_target((1.0, 0.0), stamp=0.2)

    assert state.pid_state.previous_error is None
    assert state._last_compute is None
    state.set_pose((1.0, 0.0, 0.0), stamp=0.1)

    output = state.compute(now=0.2)

    assert output.reached
    assert state.target is None
    assert state.pid_state.previous_error is None


def test_pid_state_returns_zero_without_fresh_target_and_pose():
    """Dropping input freshness checks would reuse obsolete navigation commands."""
    state = PIDControllerState(sensor_timeout=0.5)

    output = state.compute(now=1.0)

    assert output.linear_x == 0.0 and output.angular_z == 0.0 and not output.reached
