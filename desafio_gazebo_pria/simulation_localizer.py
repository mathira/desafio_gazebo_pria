"""Republish the simulator's true pose as the mission's odometry and TF."""

from __future__ import annotations


def transform_from_odometry(odometry):
    """Build the `odom -> base_footprint` transform carried by an odometry message."""
    from geometry_msgs.msg import TransformStamped

    transform = TransformStamped()
    transform.header.stamp = odometry.header.stamp
    transform.header.frame_id = odometry.header.frame_id
    transform.child_frame_id = odometry.child_frame_id
    transform.transform.translation.x = odometry.pose.pose.position.x
    transform.transform.translation.y = odometry.pose.pose.position.y
    transform.transform.translation.z = odometry.pose.pose.position.z
    transform.transform.rotation = odometry.pose.pose.orientation
    return transform


def main(args: list[str] | None = None) -> None:
    """Publish true odometry and its matching transform for the PID mission."""
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from tf2_ros import TransformBroadcaster

    class SimulationLocalizer(Node):
        """Make Gazebo's model pose the sole mission `/odom` source."""

        def __init__(self) -> None:
            super().__init__("simulation_localizer")
            self.declare_parameter("odometry_topic", "/odom_truth")
            self.declare_parameter("output_topic", "/odom")
            self._publisher = self.create_publisher(
                Odometry, str(self.get_parameter("output_topic").value), 10
            )
            self._broadcaster = TransformBroadcaster(self)
            self.create_subscription(
                Odometry,
                str(self.get_parameter("odometry_topic").value),
                self._on_odometry,
                10,
            )

        def _on_odometry(self, message: Odometry) -> None:
            self._publisher.publish(message)
            self._broadcaster.sendTransform(transform_from_odometry(message))

    from .lifecycle import run_node
    run_node(SimulationLocalizer, args)
