import numpy as np

from desafio_gazebo_pria.frontier import (
    clearance_cells_for,
    grid_to_world,
    nearest_frontier_step,
)
from desafio_gazebo_pria.grid_mapping import GridSpec


def test_nearest_frontier_returns_lookahead_on_free_path():
    grid = np.full((7, 7), 100, dtype=np.int8)
    grid[3, 1:6] = 0
    grid[2, 5] = -1

    assert nearest_frontier_step(grid, (3, 1), clearance=0, lookahead=2) == (3, 3)


def test_frontier_search_returns_none_without_unknown_neighbor():
    assert nearest_frontier_step(np.zeros((3, 3), dtype=np.int8), (1, 1), 0, 1) is None


def test_frontier_with_occupied_clearance_region_is_rejected():
    grid = np.full((7, 7), 100, dtype=np.int8)
    grid[3, 1:4] = 0
    grid[2, 3] = -1

    assert nearest_frontier_step(grid, (3, 1), clearance=1, lookahead=2) is None


def test_nonzero_clearance_frontier_returns_a_free_lookahead_path_cell():
    grid = np.full((9, 9), 100, dtype=np.int8)
    grid[4, 1:8] = 0
    grid[3:6, 6:9] = 0
    grid[3, 3] = -1
    grid[5, 3] = 100
    grid[3, 7] = -1

    step = nearest_frontier_step(grid, (4, 1), clearance=1, lookahead=6)

    assert step is None  # The only corridor violates clearance all along its length.


def test_lookahead_never_cuts_across_an_occupied_corner():
    grid = np.full((7, 7), 100, dtype=np.int8)
    grid[4, 1:5] = 0
    grid[1:5, 4] = 0
    grid[0, 4] = -1
    assert nearest_frontier_step(grid, (4, 1), 0, 20) == (4, 4)


def test_clearance_is_enforced_on_intermediate_cells_from_safe_start():
    grid = np.zeros((9, 11), dtype=np.int8)
    grid[:, 5] = 100
    grid[4, 5] = 0
    grid[4, 10] = -1
    assert nearest_frontier_step(grid, (4, 2), 1, 3) is None


def test_clearance_cells_for_rounds_up_to_whole_cells():
    assert clearance_cells_for(0.20, 0.05) == 4
    assert clearance_cells_for(0.21, 0.05) == 5


def test_grid_to_world_returns_the_cell_centre_from_the_grid_origin():
    spec = GridSpec(width=8, height=8, resolution=0.5, origin_x=-1.0, origin_y=-2.0)

    assert grid_to_world(spec, row=2, column=3) == (0.75, -0.75)
