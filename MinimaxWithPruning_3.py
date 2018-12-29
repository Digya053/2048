# Implements the minimax algorithm with alpha-beta pruning.

import numpy as np
import time
import math
from Grid_3 import *


def decision(grid, max=True):
    limit = 4
    start = time.clock()

    return maximize(
        grid=grid,
        alpha=-np.inf,
        beta=np.inf,
        depth=limit,
        start=start) if max else minimize(
        grid=grid,
        alpha=-np.inf,
        beta=np.inf,
        depth=limit,
        start=start)


def maximize(grid, alpha, beta, depth, start):
    if not grid.canMove() or depth == 0 or (time.clock() - start) > 0.04:
        return eval(grid)

    max_utility = -np.inf

    for child in get_children(grid):
        max_utility = max(
            max_utility,
            minimize(
                grid=child,
                alpha=alpha,
                beta=beta,
                depth=depth - 1,
                start=start))

        if max_utility >= beta:
            break

        alpha = max(max_utility, alpha)

    return max_utility


def minimize(grid, alpha, beta, depth, start):
    if not grid.canMove() or depth == 0 or (time.clock() - start) > 0.04:
        return eval(grid)

    min_utility = np.inf

    available_cells = grid.getAvailableCells()

    children = []

    for cell in available_cells:
        new_grid2 = grid.clone()
        new_grid4 = grid.clone()

        new_grid2.insertTile(cell, 2)
        new_grid4.insertTile(cell, 4)

        children.append(new_grid2)
        children.append(new_grid4)

    for child in children:
        min_utility = min(
            min_utility,
            maximize(
                grid=child,
                alpha=alpha,
                beta=beta,
                depth=depth - 1,
                start=start))

        if min_utility <= alpha:
            break

        beta = min(min_utility, beta)

    return min_utility


def get_child(grid, dir):
    temp = grid.clone()
    temp.move(dir)
    return temp


def get_children(grid):
    children = []
    for move in grid.getAvailableMoves():
        children.append(get_child(grid, move))
    return children


def eval(grid):
    gradients = [
        [[3, 2, 1, 0], [2, 1, 0, -1], [1, 0, -1, -2], [0, -1, -2, -3]],
        [[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1], [-3, -2, -1, -0]],
        [[0, -1, -2, -3], [1, 0, -1, -2], [2, 1, 0, -1], [3, 2, 1, 0]],
        [[-3, -2, -1, 0], [-2, -1, 0, 1], [-1, 0, 1, 2], [0, 1, 2, 3]]
    ]

    values = [0, 0, 0, 0]

    for i in range(4):
        for x in range(4):
            for y in range(4):
                values[i] += gradients[i][x][y] * grid.map[x][y]

    return max(values)