"""ROS adapter that converts monitoring-map frontiers into local targets."""

from __future__ import annotations

import math

import numpy as np

from .frontier import clearance_cells_for, grid_to_world, nearest_frontier_step, segment_safe
from .grid_mapping import GridSpec


def mission_finished_qos():
    """Return the durable terminal-event contract shared by mission nodes."""
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )


class MissionDeadline:
    """Latch elapsed wall or simulation time independently of the clock source."""

    def __init__(
        self, max_mission_seconds: float, start_time: float | None = None
    ) -> None:
        self.max_mission_seconds = float(max_mission_seconds)
        if (
            not math.isfinite(self.max_mission_seconds)
            or self.max_mission_seconds <= 0.0
        ):
            raise ValueError("max_mission_seconds must be finite and positive")
        self._start_time = None if start_time is None else float(start_time)

    def update_time(self, now: float) -> bool:
        """Return true once this clock has reached the configured duration."""
        now = float(now)
        if not math.isfinite(now):
            return False
        if self._start_time is None:
            self._start_time = now
            return False
        return now - self._start_time >= self.max_mission_seconds


class PlannerState:
    """Track the mission completion latch independently of ROS messages."""

    def __init__(
        self,
        target_coverage: float = 0.98,
        max_mission_seconds: float = 1800.0,
        max_replans: int = 500,
    ) -> None:
        self.target_coverage = float(target_coverage)
        self.max_mission_seconds = float(max_mission_seconds)
        self._deadline = MissionDeadline(self.max_mission_seconds)
        self.finished = False
        if max_replans <= 0:
            raise ValueError("max_replans must be positive")
        self.max_replans = max_replans
        self.replans = 0
        self.target_cell = None
        self.excluded = set()

    def invalidate_target(self):
        if self.target_cell is not None:
            self.excluded.add(self.target_cell)
        self.target_cell = None

    def plan(self, grid, monitoring, spec, pose, clearance, lookahead):
        if self.finished or not all(math.isfinite(v) for v in pose):
            return None
        start = _world_to_grid(spec, *pose[:2])
        if self.target_cell is not None:
            target = grid_to_world(spec, *self.target_cell)
            if segment_safe(grid, spec, pose[:2], target, clearance) and math.dist(pose[:2], target) > 0.1:
                return target
            self.invalidate_target()
        if self.replans >= self.max_replans:
            self.finished = True
            return None
        step = nearest_frontier_step(grid, start, clearance, lookahead,
                                     monitoring, self.excluded, 0.15 / spec.resolution)
        self.replans += 1
        if step is None:
            self.finished = True
            return None
        self.target_cell = step
        return grid_to_world(spec, *step)

    def update_coverage(self, coverage: float) -> bool:
        if self.finished:
            return True
        self.finished = math.isfinite(coverage) and coverage >= self.target_coverage
        return self.finished

    def update_time(self, now: float) -> bool:
        """Latch terminal state once the configured mission duration elapses."""
        if self.finished:
            return True
        self.finished = self._deadline.update_time(now)
        return self.finished


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _world_to_grid(spec: GridSpec, x: float, y: float) -> tuple[int, int]:
    return (
        math.floor((float(y) - spec.origin_y) / spec.resolution),
        math.floor((float(x) - spec.origin_x) / spec.resolution),
    )


try:  # Keep PlannerState importable without ROS for pure unit tests.
    import rclpy
    from rclpy.clock import Clock, ClockType
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32
except ImportError:  # pragma: no cover - exercised when ROS is installed.
    rclpy = None
    FrontierPlanner = None
else:

    class FrontierPlanner(Node):
        """Publish a lookahead frontier target in the `odom` frame."""

        def __init__(self) -> None:
            super().__init__("frontier_planner")
            self.declare_parameter("stop_distance", 0.20)
            self.declare_parameter("lookahead_cells", 8)
            self.declare_parameter("target_coverage", 0.98)
            self.declare_parameter("max_mission_seconds", 1800.0)
            self.declare_parameter("max_replans", 500)
            stop_distance = float(self.get_parameter("stop_distance").value)
            lookahead = int(self.get_parameter("lookahead_cells").value)
            target_coverage = float(self.get_parameter("target_coverage").value)
            max_mission_seconds = float(
                self.get_parameter("max_mission_seconds").value
            )
            if not math.isfinite(stop_distance) or stop_distance < 0.0:
                raise ValueError("stop_distance must be finite and non-negative")
            if lookahead < 0:
                raise ValueError("lookahead_cells must be non-negative")
            if not 0.0 < target_coverage <= 1.0:
                raise ValueError("target_coverage must be in (0, 1]")
            if (
                not math.isfinite(max_mission_seconds)
                or max_mission_seconds <= 0.0
            ):
                raise ValueError("max_mission_seconds must be finite and positive")

            self._stop_distance = stop_distance
            self._lookahead = lookahead
            self._state = PlannerState(target_coverage, max_mission_seconds,
                                       int(self.get_parameter("max_replans").value))
            self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self._steady_deadline = MissionDeadline(
                max_mission_seconds,
                self._steady_clock.now().nanoseconds / 1_000_000_000.0,
            )
            self._finish_published = False
            self._pose: tuple[float, float, float] | None = None
            self._latest_map: OccupancyGrid | None = None
            self._monitoring_map = None
            self._target_publisher = self.create_publisher(
                PoseStamped, "/navigation_target", 10
            )
            self._finished_publisher = self.create_publisher(
                Bool, "/mission_finished", mission_finished_qos()
            )
            self.create_subscription(OccupancyGrid, "/coverage_map", self._on_map, 10)
            self.create_subscription(OccupancyGrid, "/monitoring_map", self._on_monitoring, 10)
            self.create_subscription(Odometry, "/odom", self._on_odometry, 10)
            self.create_subscription(
                Float32, "/coverage_reachable", self._on_coverage, 10
            )
            self.create_subscription(Bool, "/recovery_requested", self._on_recovery, 10)
            self.create_subscription(Bool, "/target_reached", self._on_recovery, 10)
            self._timer = self.create_timer(0.1, self._on_timer)
            self._steady_timer = self.create_timer(
                0.1, self._on_steady_watchdog, clock=self._steady_clock
            )

        def _on_timer(self) -> None:
            if self._state.update_time(
                self.get_clock().now().nanoseconds / 1_000_000_000.0
            ):
                self._finish()
            elif self._latest_map is not None:
                self._plan(self._latest_map)

        def _on_steady_watchdog(self) -> None:
            if self._steady_deadline.update_time(
                self._steady_clock.now().nanoseconds / 1_000_000_000.0
            ):
                self._finish()

        def _on_odometry(self, message: Odometry) -> None:
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            self._pose = (
                position.x,
                position.y,
                _yaw_from_quaternion(
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                ),
            )

        def _on_coverage(self, message: Float32) -> None:
            if self._state.update_coverage(message.data):
                self._finish()

        def _on_recovery(self, message: Bool) -> None:
            if message.data and self._latest_map is not None:
                self._state.invalidate_target()
                self._plan(self._latest_map)

        def _on_monitoring(self, message: OccupancyGrid) -> None:
            self._monitoring_map = message

        def _on_map(self, message: OccupancyGrid) -> None:
            self._latest_map = message

        def _plan(self, message: OccupancyGrid) -> None:
            if self._state.finished or self._pose is None or self._monitoring_map is None:
                return
            width = int(message.info.width)
            height = int(message.info.height)
            resolution = float(message.info.resolution)
            if width <= 0 or height <= 0 or not math.isfinite(resolution) or resolution <= 0.0:
                return
            grid = np.asarray(message.data, dtype=np.int8)
            monitoring = np.asarray(self._monitoring_map.data, dtype=np.int8)
            if grid.size != width * height or monitoring.size != grid.size:
                return
            origin = message.info.origin.position
            spec = GridSpec(width, height, resolution, origin.x, origin.y)
            target_xy = self._state.plan(
                grid.reshape(height, width),
                monitoring.reshape(height, width), spec, self._pose,
                clearance_cells_for(self._stop_distance, resolution),
                self._lookahead,
            )
            if target_xy is None:
                self._finish()
                return
            target_x, target_y = target_xy
            heading = math.atan2(target_y - self._pose[1], target_x - self._pose[0])
            target = PoseStamped()
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = "odom"
            target.pose.position.x = target_x
            target.pose.position.y = target_y
            target.pose.orientation.z = math.sin(heading / 2.0)
            target.pose.orientation.w = math.cos(heading / 2.0)
            self._target_publisher.publish(target)

        def _finish(self) -> None:
            if self._finish_published:
                return
            self._state.finished = True
            self._finish_published = True
            self._finished_publisher.publish(Bool(data=True))


def main(args: list[str] | None = None) -> None:
    """Run the frontier planner as a ROS 2 node."""
    if rclpy is None:
        raise RuntimeError("frontier_planner requires a ROS 2 Python environment")
    from .lifecycle import run_node
    run_node(FrontierPlanner, args)
