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

    for line in input.splitlines()[1:]:
        index, x, y = line.split(" ")
        positions.append((int(index), float(x), float(y)))

    matrix: Matrix = [[] for _ in range(len(positions))]

    for index, x_coord, y_coord in positions[:len(positions)-1]:
        for other_index, other_x_coord, other_y_coord in positions[index:]:
            matrix[index-1].append((other_index, other_x_coord, other_y_coord))
            matrix[other_index-1].append((index, x_coord, y_coord))

    for row_index, row in enumerate(matrix):
        matrix[row_index] = list(sorted(row, key=lambda x: distance(positions[row_index], x)))
    
    # Cache for future use
    if __name__ == "__main__":
        path_to_cache = Path(__file__).parent.joinpath("./matrix.txt")
        path_to_cache.touch(0o666)
        path_to_cache.write_text(json.dumps(matrix))
        path_to_cache = Path(__file__).parent.joinpath("./lookup_list.txt")
        path_to_cache.touch(0o666)
        path_to_cache.write_text(json.dumps(positions))

    return matrix, positions

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to input file", type=file_path)
