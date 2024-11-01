import json
from math import sqrt
from .tsp import Matrix
from pathlib import Path
from source.helpers import file_path

def parse(input: str) -> Matrix:
    
    positions: list[list[int]] = [list(map(int, line.split(" "))) for line in input.splitlines()[1:]]
    """index, x coord, y coord"""

    matrix: Matrix = [[] for _ in range(len(positions))]

    # calculates top-right triangle of adjacency matrix and copies it to the bottom-left triangle
    for index, x_coord, y_coord in positions[:len(positions)-1]:
        for other_index, other_x_coord, other_y_coord in positions[index:]:
            # euclidean distance
            distance: float = sqrt((x_coord-other_x_coord)**2 + (y_coord-other_y_coord)**2)
            matrix[index-1].append((other_index, distance))
            matrix[other_index-1].append((index, distance))

    matrix = [list(sorted(row)) for row in matrix]
    
    # Cache for future use
    if __name__ == "__main__":
        path_to_cache = Path(__file__).parent.joinpath("./cache.txt")
        path_to_cache.touch(0o666)
        path_to_cache.write_text(json.dumps(matrix))

    return matrix

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to input file", type=file_path)
