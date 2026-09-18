"""Clearance-aware frontier selection for occupancy grids."""

from __future__ import annotations

import collections
import math

import numpy as np

from .grid_mapping import FREE, OCCUPIED, UNKNOWN, GridSpec, _world_to_grid


def clearance_cells_for(stop_distance: float, resolution: float) -> int:
    """Return the whole-cell clearance required for a stopping distance."""
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    return max(0, math.ceil(float(stop_distance) / float(resolution)))


def grid_to_world(spec: GridSpec, row: int, column: int) -> tuple[float, float]:
    """Return the centre of a grid cell in world coordinates."""
    return (
        spec.origin_x + (int(column) + 0.5) * spec.resolution,
        spec.origin_y + (int(row) + 0.5) * spec.resolution,
    )


def _in_bounds(grid: np.ndarray, cell: tuple[int, int]) -> bool:
    row, column = cell
    return 0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]


def _free_four_neighbors(grid: np.ndarray, cell: tuple[int, int]):
    row, column = cell
    for neighbor in ((row - 1, column), (row, column - 1), (row, column + 1), (row + 1, column)):
        if _in_bounds(grid, neighbor) and grid[neighbor] == FREE:
            yield neighbor


def _has_unknown_neighbor(grid: np.ndarray, cell: tuple[int, int]) -> bool:
    row, column = cell
    return any(
        _in_bounds(grid, neighbor) and grid[neighbor] == UNKNOWN
        for neighbor in ((row - 1, column), (row, column - 1), (row, column + 1), (row + 1, column))
    )


def _has_clearance(grid: np.ndarray, cell: tuple[int, int], clearance: int) -> bool:
    row, column = cell
    return not np.any(grid[max(0, row-clearance):row+clearance+1,
                           max(0, column-clearance):column+clearance+1] == OCCUPIED)


def segment_safe(grid, spec, start_xy, target_xy, clearance):
    """Conservatively check every cell touched by a straight control segment."""
    start = np.asarray(start_xy, dtype=float)
    delta = np.asarray(target_xy, dtype=float) - start
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(delta)):
        return False
    steps = max(1, math.ceil(float(np.linalg.norm(delta)) / (spec.resolution / 4)))
    previous = None
    for index in range(steps + 1):
        cell = _world_to_grid(spec, *(start + delta * index / steps))
        checked = [cell]
        if previous is not None and cell[0] != previous[0] and cell[1] != previous[1]:
            checked.extend([(previous[0], cell[1]), (cell[0], previous[1])])
        for touched in checked:
            if (not _in_bounds(grid, touched) or grid[touched] != FREE
                    or not _has_clearance(grid, touched, clearance)):
                return False
        previous = cell
    return True


def _is_safe_frontier(grid: np.ndarray, cell: tuple[int, int], clearance: int) -> bool:
    return _has_unknown_neighbor(grid, cell) and _has_clearance(grid, cell, clearance)


def _path_from_parents(
    cell: tuple[int, int], parents: dict[tuple[int, int], tuple[int, int] | None]
) -> list[tuple[int, int]]:
    path = []
    while cell is not None:
        path.append(cell)
        cell = parents[cell]
    return list(reversed(path))


def nearest_frontier_step(
    grid: np.ndarray,
    start: tuple[int, int],
    clearance: int,
    lookahead: int,
    monitoring: np.ndarray | None = None,
    excluded: set | None = None,
    minimum_distance: float = 0.0,
) -> tuple[int, int] | None:
    """Return a free lookahead cell on the shortest safe path to a frontier."""
    cells = np.asarray(grid)
    safe_clearance = max(0, int(clearance))
    safe_lookahead = max(0, int(lookahead))
    frontier_grid = cells if monitoring is None else np.asarray(monitoring)
    excluded = excluded or set()
    if (not _in_bounds(cells, start) or cells[start] != FREE
            or not _has_clearance(cells, start, safe_clearance)):
        return None

    queue = collections.deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if _has_unknown_neighbor(frontier_grid, cell) and cell != start:
            path = _path_from_parents(cell, parents)
            index = min(safe_lookahead, len(path) - 1)
            # PID follows a straight segment. Stop the lookahead at the first
            # corner instead of commanding a diagonal across blocked cells.
            if len(path) > 1:
                direction = (path[1][0] - start[0], path[1][1] - start[1])
                for step in range(2, index + 1):
                    if (path[step][0] - path[step - 1][0],
                            path[step][1] - path[step - 1][1]) != direction:
                        index = step - 1
                        break
            candidate = path[index]
            if candidate not in excluded and math.dist(candidate, start) > minimum_distance:
                return candidate
        for neighbor in _free_four_neighbors(cells, cell):
            if neighbor not in parents and _has_clearance(cells, neighbor, safe_clearance):
                parents[neighbor] = cell
                queue.append(neighbor)
    return None
