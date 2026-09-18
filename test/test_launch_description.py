from pathlib import Path
import runpy

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_mission_launch_contains_world_and_autonomous_pipeline():
    """Removing a mission component would leave no end-to-end autonomous run."""
    source = (PACKAGE_ROOT / "launch" / "mission.launch.py").read_text()

    for required in (
        "turtlebot3_dqn_stage4.launch.py",
        "safety_supervisor",
        "pid_controller",
        "frontier_planner",
    ):
        assert required in source


def test_mission_launch_excludes_manual_and_map_localization_stacks():
    """The mission must not silently fall back to a manual or prebuilt-map stack."""
    source = (PACKAGE_ROOT / "launch" / "mission.launch.py").read_text().lower()

    for forbidden in ("nav2", "slam_toolbox", "amcl", "teleop", "dqn_explorer"):
        assert forbidden not in source


def test_readme_documents_autonomous_command_and_sensor_interfaces():
    """Removing operator-critical interface guidance makes the mission unsafe to run."""
    readme = (PACKAGE_ROOT / "README.md").read_text()

    assert "ros2 launch desafio_gazebo_pria mission.launch.py" in readme
    assert "/scan" in readme and "/odom" in readme and "/cmd_vel" in readme
    assert "PID" in readme and "frontier" in readme.lower()
    assert "`pid_controller`" in readme and "`/pid_cmd_vel`" in readme
    assert "`safety_supervisor`" in readme and "`/cmd_vel`" in readme
    assert "target_coverage" in readme
    assert "tiempo máximo" in readme
    assert "/mission_finished" in readme
    assert "no quedan objetivos seguros" in readme
    assert "cuando llega al tiempo máximo" in readme


def test_mission_launch_binds_its_duration_to_the_terminal_planner(monkeypatch):
    """A declared duration without a node parameter leaves a stalled mission running."""
    pytest.importorskip("launch_ros")
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch.launch_description_sources import get_launch_description_from_python_launch_file
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters

    monkeypatch.setenv("TURTLEBOT3_MODEL", "burger")
    description = get_launch_description_from_python_launch_file(
        str(PACKAGE_ROOT / "launch" / "mission.launch.py")
    )
    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    context.launch_configurations["max_mission_seconds"] = "17.5"
    planner = next(
        entity
        for entity in description.entities
        if isinstance(entity, Node) and entity.node_executable == "frontier_planner"
    )

    parameters = evaluate_parameters(context, planner._Node__parameters)

    assert parameters[-1]["max_mission_seconds"] == 17.5


def test_training_stage4_resolves_the_nested_spawn_source(monkeypatch):
    """A substitution object is not a filesystem path for the nested bridge source."""
    pytest.importorskip("launch_ros")
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchContext
    from launch.actions import IncludeLaunchDescription
    from launch_ros.actions import Node

    monkeypatch.setenv("TURTLEBOT3_MODEL", "burger")
    module = runpy.run_path(str(PACKAGE_ROOT / "launch" / "mission.launch.py"))
    context = LaunchContext()
    context.launch_configurations["use_gui"] = "false"
    source = module["TrainingStage4LaunchSource"](
        str(
            Path(get_package_share_directory("turtlebot3_gazebo"))
            / "launch"
            / "turtlebot3_dqn_stage4.launch.py"
        )
    )

    description = source.get_launch_description(context)
    spawn = next(
        action
        for action in description.entities
        if isinstance(action, IncludeLaunchDescription)
        and isinstance(action.launch_description_source, module["InitialRobotBridgeSource"])
    )
    nested = spawn.launch_description_source.get_launch_description(context)

    assert any(
        isinstance(action, Node) and action.node_executable == "parameter_bridge"
        for action in nested.entities
    )
