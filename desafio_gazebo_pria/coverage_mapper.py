"""ROS adapter that integrates odometry-aligned laser scans into a grid."""

from __future__ import annotations

import math

import numpy as np

from .grid_mapping import GridSpec, _world_to_grid, reachable_coverage, update_scan


class MappingState:
    """Keep map data independent from ROS callback and message mechanics."""

    def __init__(self, spec: GridSpec | None = None, monitoring_radius: float = 0.75) -> None:
        self.spec = spec or GridSpec(
            width=400,
            height=400,
            resolution=0.05,
            origin_x=-10.0,
            origin_y=-10.0,
        )
        self.grid = np.full((self.spec.height, self.spec.width), -1, dtype=np.int8)
        if not math.isfinite(monitoring_radius) or monitoring_radius <= 0:
            raise ValueError("monitoring_radius must be finite and positive")
        self.monitoring_radius = monitoring_radius
        self.monitored = np.zeros(self.grid.shape, dtype=bool)
        self.pose: tuple[float, float, float] | None = None

    @property
    def monitoring_grid(self):
        return np.where(self.grid == 100, 100, np.where(self.monitored & (self.grid == 0), 0, -1)).astype(np.int8)

    def coverage(self):
        if self.pose is None:
            return 0.0
        return reachable_coverage(self.grid, self.monitored, _world_to_grid(self.spec, *self.pose[:2]))

    def update_pose(self, x: float, y: float, yaw: float) -> None:
        self.pose = (float(x), float(y), float(yaw)) if all(math.isfinite(v) for v in (x, y, yaw)) else None

    def integrate_scan(
        self,
        ranges,
        angle_min: float,
        angle_increment: float,
        range_max: float,
    ) -> int:
        if self.pose is None:
            return 0
        before = int(np.count_nonzero(self.monitored))
        update_scan(
            self.grid,
            self.spec,
            self.pose,
            ranges,
            angle_min,
            angle_increment,
            range_max,
            self.monitored,
            self.monitoring_radius,
        )
        return int(np.count_nonzero(self.monitored)) - before


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract a planar heading from an odometry orientation."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


try:  # Keep the pure MappingState usable in the host-side test environment.
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float32, Int32
except ImportError:  # pragma: no cover - exercised when ROS is installed.
    rclpy = None
    CoverageMapper = None
else:

    class CoverageMapper(Node):
        """Publish coverage and monitoring grids from `/scan` and `/odom`."""

        def __init__(self) -> None:
            super().__init__("coverage_mapper")
            self.declare_parameter("map_width", 400)
            self.declare_parameter("map_height", 400)
            self.declare_parameter("map_resolution", 0.05)
            self.declare_parameter("map_origin_x", -10.0)
            self.declare_parameter("map_origin_y", -10.0)
            self.declare_parameter("monitoring_radius", 0.75)
            spec = GridSpec(
                width=int(self.get_parameter("map_width").value),
                height=int(self.get_parameter("map_height").value),
                resolution=float(self.get_parameter("map_resolution").value),
                origin_x=float(self.get_parameter("map_origin_x").value),
                origin_y=float(self.get_parameter("map_origin_y").value),
            )
            if spec.width <= 0 or spec.height <= 0 or spec.resolution <= 0.0:
                raise ValueError("map dimensions and resolution must be positive")

            self._state = MappingState(spec, float(self.get_parameter("monitoring_radius").value))
            self._coverage_map_publisher = self.create_publisher(
                OccupancyGrid, "/coverage_map", 10
            )
            self._monitoring_map_publisher = self.create_publisher(
                OccupancyGrid, "/monitoring_map", 10
            )
            self._coverage_publisher = self.create_publisher(
                Float32, "/coverage_reachable", 10
            )
            self._new_cells_publisher = self.create_publisher(
                Int32, "/newly_monitored_cells", 10
            )
            self.create_subscription(Odometry, "/odom", self._on_odometry, 10)
            self.create_subscription(LaserScan, "/scan", self._on_scan, 10)

        def _on_odometry(self, message: Odometry) -> None:
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            self._state.update_pose(
                position.x,
                position.y,
                _yaw_from_quaternion(
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                ),
            )

        def _map_message(self, stamp, monitoring=False) -> OccupancyGrid:
            message = OccupancyGrid()
            message.header.stamp = stamp
            message.header.frame_id = "odom"
            message.info.resolution = self._state.spec.resolution
            message.info.width = self._state.spec.width
            message.info.height = self._state.spec.height
            message.info.origin.position.x = self._state.spec.origin_x
            message.info.origin.position.y = self._state.spec.origin_y
            message.info.origin.orientation.w = 1.0
            grid = self._state.monitoring_grid if monitoring else self._state.grid
            message.data = grid.reshape(-1).tolist()
            return message

        def _on_scan(self, message: LaserScan) -> None:
            newly_known = self._state.integrate_scan(
                message.ranges,
                message.angle_min,
                message.angle_increment,
                message.range_max,
            )
            if self._state.pose is None:
                return
            self._coverage_map_publisher.publish(self._map_message(message.header.stamp))
            self._monitoring_map_publisher.publish(self._map_message(message.header.stamp, monitoring=True))
            self._coverage_publisher.publish(
                Float32(data=self._state.coverage())
            )
            self._new_cells_publisher.publish(Int32(data=newly_known))


def main(args: list[str] | None = None) -> None:
    """Run the coverage mapper as a ROS 2 node."""
    if rclpy is None:
        raise RuntimeError("coverage_mapper requires a ROS 2 Python environment")
    from .lifecycle import run_node
    run_node(CoverageMapper, args)
