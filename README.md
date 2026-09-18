# Desafío Gazebo ROS 2

**Estudiante:** Mathias Rodriguez  
**Institución:** UTEC - ITR Norte Rivera  
**Programa:** PRIA  
**Materia:** Proyecto de Robots 2

## Descripción

En este paquete implementé un sistema de navegación autónoma para el desafío de Gazebo. El robot observa el escenario con el LiDAR, construye un mapa de ocupación, busca zonas nuevas y se mueve usando un controlador PID.

La consigna está en [este PDF](./Descripcion%20Desafio%20Gazebo%20ROS%202.pdf). También tomé como referencia [turtlebot3_control_ros2](https://github.com/ricardoGrando/turtlebot3_control_ros2).

## Compilar

Desde el contenedor:

```bash
cd /ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select desafio_gazebo_pria
source /ros2_ws/install/setup.bash
```

## Ejecutar

Con Gazebo y RViz:

```bash
ros2 launch desafio_gazebo_pria mission.launch.py use_rviz:=true
```

Sólo con Gazebo:

```bash
ros2 launch desafio_gazebo_pria mission.launch.py use_rviz:=false
```

El robot comienza la misión solo. No se debe iniciar teleoperación ni publicar manualmente en `/cmd_vel`.

## Qué hace el sistema

- `coverage_mapper` construye el mapa y calcula el área observada.
- `frontier_planner` elige nuevos objetivos seguros.
- `pid_controller` controla la posición y orientación del robot.
- `safety_supervisor` limita velocidades, detecta obstáculos y detiene el robot ante fallas.
- `simulation_localizer` entrega la odometría usada por la misión.

Los datos principales son `/scan`, `/odom`, `/coverage_reachable`, `/navigation_target` y `/mission_finished`. El `pid_controller` publica el comando nominal en `/pid_cmd_vel` y el único nodo que publica el comando final en `/cmd_vel` es `safety_supervisor`.

## Verificación

En otra terminal del contenedor:

```bash
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
ros2 topic echo --once /coverage_reachable
ros2 topic info /cmd_vel -v
```

La cobertura es un porcentaje calculado sobre el espacio libre alcanzable que fue descubierto y observado de cerca. La misión termina cuando alcanza `target_coverage`, cuando no quedan objetivos seguros o cuando llega al tiempo máximo configurado.

Para ejecutar las pruebas:

```bash
cd /ros2_ws
pytest -q src/desafio_gazebo_pria/test
```
