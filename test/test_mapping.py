import numpy as np

from desafio_gazebo_pria.grid_mapping import (
    GridSpec,
    reachable_coverage,
    update_scan,
)


def test_scan_marks_free_ray_cells_and_terminal_obstacle():
    spec = GridSpec(width=9, height=9, resolution=1.0, origin_x=-4.5, origin_y=-4.5)
    grid = np.full((9, 9), -1, dtype=np.int8)

    assert update_scan(grid, spec, (0.0, 0.0, 0.0), [2.0], 0.0, 1.0, 5.0) == 3
    assert grid[4, 5] == 0 and grid[4, 6] == 100


def test_reachable_coverage_counts_only_monitored_connected_free_cells():
    grid = np.array([[0, 0, 100, 0], [0, 100, 100, -1]], dtype=np.int8)
    monitored = np.array([[True, False, True, True], [False, True, True, True]])
    assert reachable_coverage(grid, monitored, (0, 0)) == 1.0 / 3.0


def test_distant_free_rays_are_not_counted_as_close_inspection():
    from desafio_gazebo_pria.coverage_mapper import MappingState
    spec = GridSpec(11, 11, 1.0, -5.5, -5.5)
    state = MappingState(spec, monitoring_radius=1.1)
    state.update_pose(0.0, 0.0, 0.0)
    assert state.integrate_scan([4.0], 0.0, 1.0, 5.0) == 2
    assert state.grid[5, 8] == 0
    assert state.monitoring_grid[5, 8] == -1
    assert state.monitoring_grid[5, 6] == 0
    assert state.monitoring_grid[5, 9] == 100
    assert state.coverage() == 0.5
    assert state.integrate_scan([4.0], 0.0, 1.0, 5.0) == 0


def test_max_range_ray_does_not_create_an_obstacle_endpoint():
    spec = GridSpec(width=9, height=9, resolution=1.0, origin_x=-4.5, origin_y=-4.5)
    grid = np.full((9, 9), -1, dtype=np.int8)

    assert update_scan(grid, spec, (0.0, 0.0, 0.0), [4.0], 0.0, 1.0, 4.0) == 4
    assert grid[4, 7] == 0 and grid[4, 8] == -1
