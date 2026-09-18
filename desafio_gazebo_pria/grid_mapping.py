"""Pure occupancy-grid scan integration helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections import deque
from dataclasses import dataclass
import math

import numpy as np


UNKNOWN = -1
FREE = 0
OCCUPIED = 100


@dataclass(frozen=True)
class GridSpec:
    """Geometry for a row-major occupancy grid."""

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float


def _world_to_grid(spec: GridSpec, x: float, y: float) -> tuple[int, int]:
    return (
        math.floor((y - spec.origin_y) / spec.resolution),
        math.floor((x - spec.origin_x) / spec.resolution),
    )


def _in_bounds(grid: np.ndarray, row: int, column: int) -> bool:
    return 0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]


def _bresenham_cells(
    start: tuple[int, int], end: tuple[int, int]
) -> Iterable[tuple[int, int]]:
    """Yield every integer grid cell on the line from ``start`` to ``end``."""
    row, column = start
    end_row, end_column = end
    delta_row = abs(end_row - row)
    delta_column = abs(end_column - column)
    step_row = 1 if row < end_row else -1
    step_column = 1 if column < end_column else -1
    error = delta_column - delta_row

    while True:
        yield row, column
        if (row, column) == (end_row, end_column):
            return
        doubled_error = 2 * error
        if doubled_error > -delta_row:
            error -= delta_row
            column += step_column
        if doubled_error < delta_column:
            error += delta_column
            row += step_row


def _mark(grid: np.ndarray, cell: tuple[int, int], value: int) -> int:
    row, column = cell
    if not _in_bounds(grid, row, column):
        return 0
    was_unknown = grid[row, column] == UNKNOWN
    grid[row, column] = value
    return int(was_unknown)


def update_scan(
    grid: np.ndarray,
    spec: GridSpec,
    pose: tuple[float, float, float],
    ranges: Sequence[float] | np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_max: float,
    monitored: np.ndarray | None = None,
    monitoring_radius: float = 0.75,
) -> int:
    """Integrate a laser scan, returning the count of newly known cells."""
    if not all(math.isfinite(value) for value in (*pose, angle_min, angle_increment, range_max)):
        return 0
    robot_x, robot_y, heading = pose
    start = _world_to_grid(spec, robot_x, robot_y)
    changed = 0
    measurements = np.asarray(ranges, dtype=float).reshape(-1)

    for index, measured in enumerate(measurements):
        if not math.isfinite(measured) or measured <= 0.0:
            continue
        distance = min(float(measured), float(range_max))
        if distance <= 0.0:
            continue
        angle = heading + float(angle_min) + index * float(angle_increment)
        end = _world_to_grid(
            spec,
            robot_x + distance * math.cos(angle),
            robot_y + distance * math.sin(angle),
        )
        cells = tuple(_bresenham_cells(start, end))
        has_obstacle = measured < range_max
        for cell in cells[:-1]:
            changed += _mark(grid, cell, FREE)
            if monitored is not None and _in_bounds(grid, *cell):
                row, column = cell
                x = spec.origin_x + (column + 0.5) * spec.resolution
                y = spec.origin_y + (row + 0.5) * spec.resolution
                if math.hypot(x - robot_x, y - robot_y) <= monitoring_radius:
                    monitored[cell] = True
        if has_obstacle:
            changed += _mark(grid, cells[-1], OCCUPIED)

    return changed


def reachable_coverage(grid: np.ndarray, monitored: np.ndarray, start: tuple[int, int]) -> float:
    """Close-inspected fraction of the discovered connected free component.

    Unknown space and disconnected free islands cannot be classified reachable
    without a prior map, so are excluded. This online estimate can decrease as
    new reachable free cells are discovered; it is not total world coverage.
    """
    cells = np.asarray(grid)
    if cells.size == 0 or not _in_bounds(cells, *start) or cells[start] != FREE:
        return 0.0
    queue, seen = deque([start]), {start}
    inspected = 0
    while queue:
        row, column = queue.popleft()
        inspected += bool(monitored[row, column])
        for neighbor in ((row - 1, column), (row + 1, column),
                         (row, column - 1), (row, column + 1)):
            if neighbor not in seen and _in_bounds(cells, *neighbor) and cells[neighbor] == FREE:
                seen.add(neighbor)
                queue.append(neighbor)
    return inspected / len(seen)
