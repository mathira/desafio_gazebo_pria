"""PID target following with a ROS adapter that publishes nominal velocity only."""

from __future__ import annotations

import math

import numpy as np

from .control import ControlConfig, PIDOutput, PIDState, target_twist, wrap_angle
from .frontier import segment_safe, clearance_cells_for
from .grid_mapping import GridSpec


class PIDControllerState:
    """Timestamped target-following state independent of ROS messages."""

    def __init__(
        self,
        config: ControlConfig | None = None,
        sensor_timeout: float = 0.5,
        require_imu: bool = False,
    ) -> None:
        if not math.isfinite(sensor_timeout) or sensor_timeout <= 0.0:
            raise ValueError("sensor_timeout must be finite and positive")
        self.config = config or ControlConfig()
        self.sensor_timeout = float(sensor_timeout)
        self.target: tuple[float, float] | None = None
        self._target_stamp: float | None = None
        self.pose: tuple[float, float, float] | None = None
        self._pose_stamp: float | None = None
        self.sectors: np.ndarray | None = None
        self._scan_stamp: float | None = None
        self.pid_state = PIDState()
        self._last_compute: float | None = None
        self.finished = False
        self.grid = None
        self.grid_spec = None
        self.require_imu = require_imu
        self.imu_yaw = None
        self._imu_stamp = None

    def set_imu(self, yaw, stamp):
        self.imu_yaw, self._imu_stamp = float(yaw), float(stamp)

    def set_map(self, grid, spec):
        self.grid, self.grid_spec = np.asarray(grid), spec

    def set_target(self, target: tuple[float, float], stamp: float) -> None:
        """Refresh an equal target, resetting PID memory only on a replacement."""
        next_target = (float(target[0]), float(target[1]))
        if self.target is not None and all(
            math.isclose(current, incoming, rel_tol=0.0, abs_tol=1e-9)
            for current, incoming in zip(self.target, next_target)
        ):
            self._target_stamp = float(stamp)
            return
        self.target = next_target
        self._target_stamp = float(stamp)
        self.pid_state = PIDState()
        self._last_compute = None

    def set_pose(self, pose: tuple[float, float, float], stamp: float) -> None:
        self.pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        self._pose_stamp = float(stamp)

    def set_scan(self, sectors, stamp: float) -> None:
        self.sectors = np.asarray(sectors, dtype=float).reshape(-1)
        self._scan_stamp = float(stamp)

    def set_finished(self, finished: bool) -> None:
        if finished:
            self.finished = True
            self.target = None
            self._target_stamp = None
            self.pid_state = PIDState()
            self._last_compute = None

    def _stale(self, now: float) -> bool:
        stamps = (self._target_stamp, self._pose_stamp, self._scan_stamp)
        return any(
            stamp is None
            or not math.isfinite(stamp)
            or not 0.0 <= now - stamp <= self.sensor_timeout
            for stamp in stamps
        )

    def _front_is_clear(self) -> bool:
        return (
            self.sectors is not None
            and self.sectors.size >= 1
            and math.isfinite(float(self.sectors[0]))
            and float(self.sectors[0]) >= self.config.stop_distance
        )

    def compute(self, now: float) -> PIDOutput:
        """Return a safe nominal command or zero until all control inputs are fresh."""
        now = float(now)
        imu_valid = (self.imu_yaw is not None and math.isfinite(self.imu_yaw)
                     and self._imu_stamp is not None and math.isfinite(self._imu_stamp)
                     and 0 <= now - self._imu_stamp <= self.sensor_timeout)
        if (self.finished or self.target is None or self.pose is None or self._stale(now)
                or (self.require_imu and not imu_valid)
                or not math.isfinite(now)
                or not all(math.isfinite(value) for value in (*self.target, *self.pose))
                or self.sectors is None or self.sectors.size != 4
                or not np.all(np.isfinite(self.sectors)) or np.any(self.sectors <= 0.0)):
            self.pid_state = PIDState()
            self._last_compute = None
            return PIDOutput(0.0, 0.0, False)
        if self.grid is not None and not segment_safe(
                self.grid, self.grid_spec, self.pose[:2], self.target,
                clearance_cells_for(self.config.stop_distance, self.grid_spec.resolution)):
            return PIDOutput(0.0, 0.0, False)
        dt = 1e-6 if self._last_compute is None else max(now - self._last_compute, 1e-6)
        pose = self.pose
        if imu_valid:
            pose = (*pose[:2], wrap_angle(pose[2] + 0.2 * wrap_angle(self.imu_yaw - pose[2])))
        output, self.pid_state = target_twist(pose, self.target, self.pid_state, dt, self.config)
        self._last_compute = now
        if output.reached:
            self.target = None
            self._target_stamp = None
            self.pid_state = PIDState()
            self._last_compute = None
            return output
        if output.linear_x > 0.0 and not self._front_is_clear():
            return PIDOutput(0.0, output.angular_z, False)
        return output


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


try:  # Keep PIDControllerState usable in pure unit tests without ROS.
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry, OccupancyGrid
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan, Imu
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Bool

    from .control import PIDConfig, control_sector_ranges
    from .frontier_planner import mission_finished_qos
except ImportError:  # pragma: no cover - exercised in the ROS Jazzy environment.
    rclpy = None
    PIDController = None
else:

    class PIDController(Node):
        """Convert navigation targets into `/pid_cmd_vel`, never `/cmd_vel`."""

        def __init__(self) -> None:
            super().__init__("pid_controller")
            defaults = {
                "kp": 1.5,
                "ki": 0.0,
                "kd": 0.1,
                "integral_limit": 0.5,
                "max_angular_speed": 1.5,
                "max_linear_speed": 0.22,
                "arrival_tolerance": 0.10,
                "alignment_tolerance": 0.35,
                "distance_gain": 0.5,
                "stop_distance": 0.20,
                "sensor_timeout": 0.5,
                "control_period": 0.05,
                "min_linear_speed": 0.04,
                "derivative_alpha": 0.2,
            }
            for name, value in defaults.items():
                self.declare_parameter(name, value)
            values = {name: float(self.get_parameter(name).value) for name in defaults}
            if any(not math.isfinite(value) or value <= 0.0 for name, value in values.items() if name != "ki"):
                raise ValueError("PID controller bounds must be finite and positive")
            if not math.isfinite(values["ki"]) or values["ki"] < 0.0:
                raise ValueError("ki must be finite and non-negative")
            config = ControlConfig(
                pid=PIDConfig(
                    kp=values["kp"],
                    ki=values["ki"],
                    kd=values["kd"],
                    integral_limit=values["integral_limit"],
                    max_angular_speed=values["max_angular_speed"],
                    derivative_alpha=values["derivative_alpha"],
                ),
                max_linear_speed=values["max_linear_speed"],
                arrival_tolerance=values["arrival_tolerance"],
                alignment_tolerance=values["alignment_tolerance"],
                distance_gain=values["distance_gain"],
                stop_distance=values["stop_distance"],
                min_linear_speed=values["min_linear_speed"],
            )
            self._state = PIDControllerState(config, values["sensor_timeout"], require_imu=True)
            self._publisher = self.create_publisher(Twist, "/pid_cmd_vel", 10)
            self._target_reached_publisher = self.create_publisher(Bool, "/target_reached", 10)
            self.create_subscription(PoseStamped, "/navigation_target", self._on_target, 10)
            self.create_subscription(Odometry, "/odom", self._on_odometry, 10)
            self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
            self.create_subscription(Imu, "/imu", self._on_imu, qos_profile_sensor_data)
            self.create_subscription(OccupancyGrid, "/coverage_map", self._on_map, 10)
            self.create_subscription(
                Bool, "/mission_finished", self._on_finished, mission_finished_qos()
            )
            self._timer = self.create_timer(values["control_period"], self._on_timer)

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds / 1_000_000_000.0

        def _on_target(self, message: PoseStamped) -> None:
            self._state.set_target(
                (message.pose.position.x, message.pose.position.y), self._now()
            )

        def _on_imu(self, message: Imu) -> None:
            q = message.orientation
            values = (q.x, q.y, q.z, q.w)
            yaw = _yaw_from_quaternion(*values) if (
                all(math.isfinite(v) for v in values)
                and sum(v*v for v in values) > 0.5
                and message.orientation_covariance[0] >= 0.0) else math.nan
            self._state.set_imu(yaw, self._now())

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

        def _on_scan(self, message: LaserScan) -> None:
            self._state.set_scan(
                control_sector_ranges(
                    message.ranges, message.angle_min, message.angle_increment
                ),
                self._now(),
            )

        def _on_map(self, message: OccupancyGrid) -> None:
            info = message.info
            if (info.width > 0 and info.height > 0 and math.isfinite(info.resolution)
                    and info.resolution > 0 and len(message.data) == info.width * info.height):
                self._state.set_map(np.asarray(message.data).reshape(info.height, info.width),
                    GridSpec(info.width, info.height, info.resolution,
                             info.origin.position.x, info.origin.position.y))

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
            output = self._state.compute(self._now())
            self._publish(output.linear_x, output.angular_z)
            if output.reached:
                self._target_reached_publisher.publish(Bool(data=True))

        def stop(self) -> None:
            self._publish(0.0, 0.0)


def main(args: list[str] | None = None) -> None:
    """Run the PID controller node."""
    if rclpy is None:
        raise RuntimeError("pid_controller requires a ROS 2 Python environment")
    from .lifecycle import run_motion_node
    run_motion_node(PIDController, args)
