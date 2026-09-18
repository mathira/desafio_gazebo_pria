"""Safety arbitration and the sole ROS publisher for `/cmd_vel`."""

from __future__ import annotations

import math

import numpy as np

from .control import ControlConfig, TwistDecision, safe_twist


class SafetyState:
    """Track timestamped command, scan, and odometry for conservative safety."""

    def __init__(
        self,
        config: ControlConfig | None = None,
        sensor_timeout: float = 0.5,
        stall_timeout: float = 2.0,
        progress_distance: float = 0.05,
    ) -> None:
        bounds = (sensor_timeout, stall_timeout, progress_distance)
        if any(not math.isfinite(value) or value <= 0.0 for value in bounds):
            raise ValueError("safety timing and progress bounds must be finite and positive")
        self.config = config or ControlConfig()
        self.sensor_timeout = float(sensor_timeout)
        self.stall_timeout = float(stall_timeout)
        self.progress_distance = float(progress_distance)
        self.nominal_linear: float | None = None
        self.nominal_angular: float | None = None
        self._nominal_stamp: float | None = None
        self.sectors: np.ndarray | None = None
        self._scan_stamp: float | None = None
        self.pose: tuple[float, float, float] | None = None
        self._pose_stamp: float | None = None
        self._progress_pose: tuple[float, float] | None = None
        self._progress_stamp: float | None = None
        self.finished = False

    def set_nominal(self, linear_x: float, angular_z: float, stamp: float) -> None:
        if self._nominal_stale(float(stamp)):
            self._progress_pose = None
            self._progress_stamp = None
        self.nominal_linear = float(linear_x)
        self.nominal_angular = float(angular_z)
        self._nominal_stamp = float(stamp)
        if not self._forward_commanded():
            self._progress_pose = None
            self._progress_stamp = None
        elif self.pose is not None and self._progress_pose is None:
            self._progress_pose = self.pose[:2]
            self._progress_stamp = float(stamp)

    def set_scan(self, sectors, stamp: float) -> None:
        self.sectors = np.asarray(sectors, dtype=float).reshape(-1)
        self._scan_stamp = float(stamp)

    def set_pose(self, pose: tuple[float, float, float], stamp: float) -> None:
        self.pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        self._pose_stamp = float(stamp)
        if not self._forward_commanded():
            return
        if self._progress_pose is None:
            self._progress_pose = self.pose[:2]
            self._progress_stamp = float(stamp)
            return
        progress = math.hypot(
            self.pose[0] - self._progress_pose[0], self.pose[1] - self._progress_pose[1]
        )
        if progress >= self.progress_distance:
            self._progress_pose = self.pose[:2]
            self._progress_stamp = float(stamp)

    def set_finished(self, finished: bool) -> None:
        if finished:
            self.finished = True

    def _forward_commanded(self) -> bool:
        return self.nominal_linear is not None and self.nominal_linear > 0.0

    def _nominal_stale(self, now: float) -> bool:
        if (
            self.nominal_linear is None
            or self.nominal_angular is None
            or self._nominal_stamp is None
            or not math.isfinite(self.nominal_linear)
            or not math.isfinite(self.nominal_angular)
            or not math.isfinite(self._nominal_stamp)
        ):
            return True
        return not 0.0 <= now - self._nominal_stamp <= self.sensor_timeout

    def _sensors_stale(self, now: float) -> bool:
        stamps = (self._scan_stamp, self._pose_stamp)
        return any(
            stamp is None
            or not math.isfinite(stamp)
            or not 0.0 <= now - stamp <= self.sensor_timeout
            for stamp in stamps
        )

    def _progress_age(self, now: float) -> float:
        return math.inf if self._progress_stamp is None else now - self._progress_stamp

    def compute(self, now: float) -> TwistDecision:
        """Return zero for stale/final state, otherwise delegate safety arbitration."""
        now = float(now)
        if (self.finished or not math.isfinite(now) or self._sensors_stale(now)
                or self._nominal_stale(now) or self.pose is None
                or not all(math.isfinite(value) for value in self.pose)):
            self._progress_pose = None
            self._progress_stamp = None
            return TwistDecision(0.0, 0.0, intervention=True)
        stalled = self._forward_commanded() and self._progress_age(now) >= self.stall_timeout
        return safe_twist(
            self.nominal_linear,
            self.nominal_angular,
            self.sectors,
            stalled,
            self.config,
        )


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


try:  # Keep SafetyState importable in pure unit tests without ROS.
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool

    from .control import PIDConfig, control_sector_ranges
    from .frontier_planner import mission_finished_qos
except ImportError:  # pragma: no cover - exercised in the ROS Jazzy environment.
    rclpy = None
    SafetySupervisor = None
else:

    class SafetySupervisor(Node):
        """Publish final safe motion and recovery transitions from sensor state."""

        def __init__(self) -> None:
            super().__init__("safety_supervisor")
            defaults = {
                "max_angular_speed": 1.5,
                "stop_distance": 0.20,
                "recovery_linear_speed": 0.10,
                "sensor_timeout": 0.5,
                "stall_timeout": 2.0,
                "progress_distance": 0.05,
                "control_period": 0.05,
                "max_linear_speed": 0.22,
                "contact_distance": 0.12,
                "rear_clearance": 0.30,
                "lateral_slow_distance": 0.30,
                "front_slow_distance": 0.45,
            }
            for name, value in defaults.items():
                self.declare_parameter(name, value)
            values = {name: float(self.get_parameter(name).value) for name in defaults}
            if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
                raise ValueError("safety supervisor bounds must be finite and positive")
            config = ControlConfig(
                pid=PIDConfig(max_angular_speed=values["max_angular_speed"]),
                stop_distance=values["stop_distance"],
                recovery_linear_speed=values["recovery_linear_speed"],
                max_linear_speed=values["max_linear_speed"],
                contact_distance=values["contact_distance"],
                rear_clearance=values["rear_clearance"],
                lateral_slow_distance=values["lateral_slow_distance"],
                front_slow_distance=values["front_slow_distance"],
            )
            self._state = SafetyState(
                config,
                sensor_timeout=values["sensor_timeout"],
                stall_timeout=values["stall_timeout"],
                progress_distance=values["progress_distance"],
            )
            self._recovering = False
            self._publisher = self.create_publisher(Twist, "/cmd_vel", 10)
            self._recovery_publisher = self.create_publisher(Bool, "/recovery_requested", 10)
            self.create_subscription(Twist, "/pid_cmd_vel", self._on_nominal, 10)
            self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
            self.create_subscription(Odometry, "/odom", self._on_odometry, 10)
            self.create_subscription(
                Bool, "/mission_finished", self._on_finished, mission_finished_qos()
            )
            self._timer = self.create_timer(values["control_period"], self._on_timer)

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds / 1_000_000_000.0

        def _on_nominal(self, message: Twist) -> None:
            self._state.set_nominal(message.linear.x, message.angular.z, self._now())

        def _on_scan(self, message: LaserScan) -> None:
            self._state.set_scan(
                control_sector_ranges(
                    message.ranges, message.angle_min, message.angle_increment
                ),
                self._now(),
            )

        def _on_odometry(self, message: Odometry) -> None:
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            self._state.set_pose(
                (
                    position.x,
                    position.y,
                    _yaw_from_quaternion(
                        orientation.x, orientation.y, orientation.z, orientation.w
                    ),
                ),
                self._now(),
            )

        def _on_finished(self, message: Bool) -> None:
            self._state.set_finished(message.data)
            if message.data:
                self._publish(0.0, 0.0)

        def _publish(self, linear_x: float, angular_z: float) -> None:
            message = Twist()
            message.linear.x = linear_x
            message.angular.z = angular_z
            self._publisher.publish(message)

        def _on_timer(self) -> None:
            decision = self._state.compute(self._now())
            self._publish(decision.linear_x, decision.angular_z)
            if decision.recovery and not self._recovering:
                self._recovery_publisher.publish(Bool(data=True))
            self._recovering = decision.recovery

        def stop(self) -> None:
            self._publish(0.0, 0.0)


def main(args: list[str] | None = None) -> None:
    """Run the safety supervisor node."""
    if rclpy is None:
        raise RuntimeError("safety_supervisor requires a ROS 2 Python environment")
    from .lifecycle import run_motion_node
    run_motion_node(SafetySupervisor, args)
