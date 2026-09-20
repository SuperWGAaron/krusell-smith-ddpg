"""Small, dependency-free tensor interpolation routines."""

from __future__ import annotations

import torch
from torch import Tensor


def linear_indices(grid: Tensor, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return bracketing indices and upper-point weight for a sorted 1-D grid."""

    if grid.ndim != 1 or grid.numel() < 2:
        raise ValueError("grid must be one-dimensional with at least two points")
    clipped = query.clamp(min=grid[0], max=grid[-1])
    upper = torch.searchsorted(grid, clipped, right=False).clamp(1, grid.numel() - 1)
    lower = upper - 1
    denominator = grid[upper] - grid[lower]
    weight = (clipped - grid[lower]) / denominator
    return lower, upper, weight


def bilinear(
    x_grid: Tensor,
    y_grid: Tensor,
    values: Tensor,
    x: Tensor,
    y: Tensor,
) -> Tensor:
    """Bilinearly interpolate ``values[x_index, y_index]`` at broadcastable queries."""

    if values.shape[:2] != (x_grid.numel(), y_grid.numel()):
        raise ValueError(
            "leading value dimensions must match x_grid and y_grid: "
            f"got {tuple(values.shape[:2])}"
        )
    x, y = torch.broadcast_tensors(x, y)
    x0, x1, wx = linear_indices(x_grid, x)
    y0, y1, wy = linear_indices(y_grid, y)
    v00 = values[x0, y0]
    v10 = values[x1, y0]
    v01 = values[x0, y1]
    v11 = values[x1, y1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )
