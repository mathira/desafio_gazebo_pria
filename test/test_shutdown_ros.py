"""Exercise SIGINT while the supervisor owns a nonzero command."""

import math
import os
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize(
    "module",
    ("desafio_gazebo_pria.coverage_mapper", "desafio_gazebo_pria.frontier_planner"),
)
def test_non_motion_nodes_exit_sigint_without_rclpy_traceback(module):
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(140 + os.getpid() % 40)
    child = subprocess.Popen(
        [sys.executable, "-c", f"from {module} import main; main()"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        time.sleep(0.5)
        os.kill(child.pid, signal.SIGINT)
        output, _ = child.communicate(timeout=3.0)
        assert child.returncode == 0, output
        assert "Traceback" not in output
        assert "RCLError" not in output
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_sigint_delivers_final_zero_before_invalidating_ros_context():
    original_domain = os.environ.get("ROS_DOMAIN_ID")
    os.environ["ROS_DOMAIN_ID"] = str(180 + os.getpid() % 40)
    rclpy = pytest.importorskip("rclpy")
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan

    rclpy.init()
    observer = Node("shutdown_observer")
    received = []
    cmd_vel_subscription = observer.create_subscription(
        Twist, "/cmd_vel", lambda m: received.append((m.linear.x, m.angular.z)), 10
    )
    command = observer.create_publisher(Twist, "/pid_cmd_vel", 10)
    odometry = observer.create_publisher(Odometry, "/odom", 10)
    scan = observer.create_publisher(LaserScan, "/scan", 10)
    child = subprocess.Popen([sys.executable, "-c",
        "from desafio_gazebo_pria.safety_supervisor import main; main()"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and (
            command.get_subscription_count() == 0
            or odometry.get_subscription_count() == 0
            or scan.get_subscription_count() == 0
            or cmd_vel_subscription.get_publisher_count() == 0
        ):
            if child.poll() is not None:
                output, _ = child.communicate(timeout=3.0)
                raise AssertionError(output or "supervisor exited before DDS discovery")
            rclpy.spin_once(observer, timeout_sec=0.05)
        assert command.get_subscription_count() > 0, "supervisor did not subscribe to /pid_cmd_vel"
        assert odometry.get_subscription_count() > 0, "supervisor did not subscribe to /odom"
        assert scan.get_subscription_count() > 0, "supervisor did not subscribe to /scan"
        assert cmd_vel_subscription.get_publisher_count() > 0, "supervisor did not publish /cmd_vel"

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not any(v > 0 for v, _ in received):
            nominal = Twist()
            nominal.linear.x = 0.1
            pose = Odometry()
            pose.pose.pose.orientation.w = 1.0
            laser = LaserScan()
            laser.angle_min = -math.pi
            laser.angle_increment = math.pi / 2
            laser.ranges = [2.0] * 4
            command.publish(nominal)
            odometry.publish(pose)
            scan.publish(laser)
            rclpy.spin_once(observer, timeout_sec=0.05)
        assert any(v > 0 for v, _ in received), "supervisor never commanded motion"
        received.clear()
        os.kill(child.pid, signal.SIGINT)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and (child.poll() is None or not received):
            rclpy.spin_once(observer, timeout_sec=0.05)
        output, _ = child.communicate(timeout=3.0)
        assert received and received[-1] == (0.0, 0.0), output
        assert child.returncode == 0, output
        assert "Traceback" not in output
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        observer.destroy_node()
        rclpy.shutdown()
        if original_domain is None:
            os.environ.pop("ROS_DOMAIN_ID", None)
        else:
            os.environ["ROS_DOMAIN_ID"] = original_domain
