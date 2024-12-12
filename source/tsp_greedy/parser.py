from source.mapping import Matrix, node, node_distance


def parse(input: str) -> tuple[Matrix, list[node]]:
    """
    :returns: adjacency matrix and node lookup table (by index)
    Indexes start at 1
    """
    positions: list[node] = []
    """index, x coord, y coord"""

    for index, line in enumerate(input.splitlines()[1:]):
        _, x, y = line.strip().split(" ")
        positions.append((index, float(x), float(y)))

    matrix: Matrix = [[] for _ in range(len(positions))]

    for index, x_coord, y_coord in positions[:len(positions)-1]:
        for other_index, other_x_coord, other_y_coord in positions[index+1:]:
            matrix[index].append((other_index, other_x_coord, other_y_coord))
            matrix[other_index].append((index, x_coord, y_coord))

    for row_index, row in enumerate(matrix):
        matrix[row_index] = list(
            sorted(row, key=lambda x: node_distance(positions[row_index], x)))

    return matrix, positions
