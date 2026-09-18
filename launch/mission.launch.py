"""Launch a self-contained autonomous PID mission in TurtleBot3 stage 4."""

from __future__ import annotations

import shlex
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.utilities import normalize_to_list_of_substitutions
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare


TRUE_ODOMETRY_TOPIC = "odom_truth"
"""Gazebo odometry generated from the Burger model pose, not wheel integration."""


_ROBOT_INCLUDE = (
    "<include><uri>model://turtlebot3_burger</uri><name>burger</name>"
    "<pose>0 0 0.01 0 0 0</pose>"
    "<plugin filename=\"gz-sim-odometry-publisher-system\""
    " name=\"gz::sim::systems::OdometryPublisher\">"
    "<odom_frame>odom</odom_frame>"
    "<robot_base_frame>base_footprint</robot_base_frame>"
    f"<odom_topic>{TRUE_ODOMETRY_TOPIC}</odom_topic>"
    "<dimensions>3</dimensions>"
    "</plugin></include>"
)


class Stage4LaunchSource(PythonLaunchDescriptionSource):
    """Add the upstream launcher's missing `use_gui` switch without forking it."""

    def _get_launch_description(self, location):
        description = super()._get_launch_description(location)
        actions = []
        gui_count = 0
        for action in description.entities:
            arguments = dict(action.launch_arguments) if isinstance(
                action, IncludeLaunchDescription
            ) else {}
            gz_args = arguments.get("gz_args")
            if isinstance(gz_args, str) and "-g" in shlex.split(gz_args):
                action = GroupAction(
                    actions=[action],
                    condition=IfCondition(LaunchConfiguration("use_gui")),
                )
                gui_count += 1
            actions.append(action)
        if gui_count != 1:
            raise RuntimeError("Official stage4 launcher no longer has one GUI include")
        return LaunchDescription(actions)


class InitialRobotBridgeSource(PythonLaunchDescriptionSource):
    """Keep upstream sensor bridges while removing its late, wheel-pose robot spawn."""

    def _get_launch_description(self, location):
        description = super()._get_launch_description(location)
        spawners = [
            action
            for action in description.entities
            if isinstance(action, Node) and action.node_executable == "create"
        ]
        if len(spawners) != 1:
            raise RuntimeError("Official Burger launcher no longer has one create action")
        actions = [action for action in description.entities if action is not spawners[0]]
        bridges = [
            action
            for action in actions
            if isinstance(action, Node) and action.node_executable == "parameter_bridge"
        ]
        if not bridges:
            raise RuntimeError("Official Burger launcher no longer has a bridge action")
        diverted = [
            (
                normalize_to_list_of_substitutions(source),
                normalize_to_list_of_substitutions(target),
            )
            for source, target in (
                ("/odom", "/odom_wheel"),
                ("/tf", "/tf_wheel"),
                ("/cmd_vel", "/cmd_vel_upstream"),
            )
        ]
        for bridge in bridges:
            bridge._Node__remappings = list(bridge._Node__remappings or []) + diverted
        return LaunchDescription(actions)


class TrainingStage4LaunchSource(Stage4LaunchSource):
    """Put Burger and its true-pose publisher in the Gazebo world at startup."""

    def _get_launch_description(self, location):
        description = super()._get_launch_description(location)
        share = Path(location).parent.parent
        official_world = share / "worlds" / "turtlebot3_dqn_stage4.world"
        world_content = official_world.read_text(encoding="utf-8")
        root = ET.fromstring(world_content)
        world = root.find("world")
        if world is None or world.get("name") != "dqn":
            raise RuntimeError("Official stage4 world layout changed")
        world.append(ET.fromstring(_ROBOT_INCLUDE))

        temporary = tempfile.TemporaryDirectory(prefix="turtlebot3-pid-nav-")
        derived_world = Path(temporary.name) / official_world.name
        ET.ElementTree(root).write(derived_world, encoding="unicode")

        actions = []
        server_count = spawn_count = 0
        for action in description.entities:
            if isinstance(action, IncludeLaunchDescription):
                arguments = dict(action.launch_arguments)
                gz_args = arguments.get("gz_args")
                if isinstance(gz_args, list) and len(gz_args) == 2 and gz_args[-1] == str(
                    official_world
                ):
                    arguments["gz_args"] = [gz_args[0], str(derived_world)]
                    action = IncludeLaunchDescription(
                        action.launch_description_source,
                        launch_arguments=arguments.items(),
                    )
                    server_count += 1
                elif set(arguments) == {"x_pose", "y_pose"}:
                    action.launch_description_source.get_launch_description(LaunchContext())
                    if Path(action.launch_description_source.location).name != "spawn_turtlebot3.launch.py":
                        raise RuntimeError("Official stage4 robot spawn include changed")
                    action = IncludeLaunchDescription(
                        InitialRobotBridgeSource(action.launch_description_source.location),
                        launch_arguments=arguments.items(),
                    )
                    spawn_count += 1
            actions.append(action)
        if server_count != 1 or spawn_count != 1:
            temporary.cleanup()
            raise RuntimeError("Official stage4 launcher layout changed")

        def cleanup(_context):
            temporary.cleanup()

        actions.append(
            RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)]))
        )
        return LaunchDescription(actions)


def generate_launch_description():
    """Compose Gazebo, true-pose localization, autonomous nodes, and optional RViz."""
    package = "desafio_gazebo_pria"
    share = FindPackageShare(package)
    sim = {
        "use_sim_time": ParameterValue(
            LaunchConfiguration("use_sim_time"), value_type=bool
        )
    }
    config = PathJoinSubstitution([share, "config", "pid_nav.yaml"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_gui", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("target_coverage", default_value="0.98"),
            DeclareLaunchArgument("max_mission_seconds", default_value="1800"),
            SetEnvironmentVariable("TURTLEBOT3_MODEL", "burger"),
            AppendEnvironmentVariable(
                "GZ_SIM_SYSTEM_PLUGIN_PATH",
                PathJoinSubstitution(
                    [FindPackagePrefix("turtlebot3_gazebo"), "lib", "turtlebot3_gazebo"]
                ),
            ),
            IncludeLaunchDescription(
                TrainingStage4LaunchSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("turtlebot3_gazebo"),
                            "launch",
                            "turtlebot3_dqn_stage4.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "use_sim_time": LaunchConfiguration("use_sim_time")
                }.items(),
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="mission_velocity_bridge",
                arguments=["/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"],
                parameters=[sim],
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="true_odometry_bridge",
                arguments=[
                    f"/{TRUE_ODOMETRY_TOPIC}@nav_msgs/msg/Odometry[gz.msgs.Odometry"
                ],
                parameters=[sim],
            ),
            Node(
                package=package,
                executable="simulation_localizer",
                name="simulation_localizer",
                parameters=[sim, {"odometry_topic": f"/{TRUE_ODOMETRY_TOPIC}"}],
            ),
            Node(
                package=package,
                executable="coverage_mapper",
                name="coverage_mapper",
                parameters=[config, sim],
            ),
            Node(
                package=package,
                executable="frontier_planner",
                name="frontier_planner",
                parameters=[
                    config,
                    sim,
                    {
                        "target_coverage": ParameterValue(
                            LaunchConfiguration("target_coverage"), value_type=float
                        ),
                        "max_mission_seconds": ParameterValue(
                            LaunchConfiguration("max_mission_seconds"), value_type=float
                        ),
                    },
                ],
            ),
            Node(
                package=package,
                executable="pid_controller",
                name="pid_controller",
                parameters=[config, sim],
            ),
            Node(
                package=package,
                executable="safety_supervisor",
                name="safety_supervisor",
                parameters=[config, sim],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", PathJoinSubstitution([share, "rviz", "pid_navigation.rviz"])],
                parameters=[sim],
                condition=IfCondition(LaunchConfiguration("use_rviz")),
            ),
        ]
    )
