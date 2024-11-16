import json
from math import sqrt
from .tsp import Matrix, node, distance
from pathlib import Path
from source.helpers import file_path


def parse(input: str) -> tuple[Matrix, list[node]]:
    """
    :returns: adjacency matrix and node lookup table (by index)
    Indexes start at 1
    """
    positions: list[node] = []
    """index, x coord, y coord"""

    for index, line in enumerate(input.splitlines()[1:]):
        _, x, y = line.split(" ")
        positions.append((index, float(x), float(y)))

    matrix: Matrix = [[] for _ in range(len(positions))]

    for index, x_coord, y_coord in positions[:len(positions)-1]:
        for other_index, other_x_coord, other_y_coord in positions[index:]:
            matrix[index-1].append((other_index, other_x_coord, other_y_coord))
            matrix[other_index-1].append((index, x_coord, y_coord))

    for row_index, row in enumerate(matrix):
        matrix[row_index] = list(
            sorted(row, key=lambda x: distance(positions[row_index], x)))

    return matrix, positions
