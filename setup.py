from glob import glob
import os

from setuptools import find_packages, setup


package_name = "desafio_gazebo_pria"


def resource_files(directory):
    files = [path for path in glob(os.path.join(directory, "*")) if os.path.isfile(path)]
    if not files:
        return []
    return [(os.path.join("share", package_name, directory), files)]


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "Descripcion Desafio Gazebo ROS 2.pdf"]),
        *resource_files("launch"),
        *resource_files("config"),
        *resource_files("rviz"),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Mathias Rodriguez",
    maintainer_email="mathias.rodriguez@utec.edu.uy",
    description=(
        "Autonomous TurtleBot3 PID frontier navigation with true odometry and "
        "safety-supervised velocity control."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "coverage_mapper = desafio_gazebo_pria.coverage_mapper:main",
            "frontier_planner = desafio_gazebo_pria.frontier_planner:main",
            "pid_controller = desafio_gazebo_pria.pid_controller:main",
            "safety_supervisor = desafio_gazebo_pria.safety_supervisor:main",
            "simulation_localizer = desafio_gazebo_pria.simulation_localizer:main",
        ],
    },
)
