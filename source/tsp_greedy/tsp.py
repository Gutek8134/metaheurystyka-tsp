type index = int
type neighbor = tuple[index, float]
type Matrix = list[list[neighbor]]


def tsp(matrix: Matrix, home_index: index) -> tuple[float, list[index]]:
    """
    Greedy algorithm for traveling salesman problem

    :param matrix: adjacency matrix in which neighbors are in format [index, distance] sorted by distance
    :param home_index: id of node from which the salesman starts and returns to at the end

    :return: path length, indexes 
    """
    # Initialization
    visited_nodes: set[index] = {home_index}
    path: list[index] = [home_index]
    path_length: float = 0.

    nodes_to_visit: int = len(matrix)
    current_position: index = home_index

    # Traveling
    while len(visited_nodes) < nodes_to_visit:
        next_destination, distance = find_next_destination(
            matrix[current_position], visited_nodes)

        # region travel
        visited_nodes.add(next_destination)
        path.append(next_destination)
        path_length += distance

        current_position = next_destination

        # endregion travel

    # Going back home
    [_, distance_home] = find_by_index(matrix[current_position], home_index)
    path.append(home_index)
    path_length += distance_home

    return path_length, path


def find_next_destination(neighbors: list[neighbor], visited_nodes: set[index]) -> neighbor:
    """
    Helper getting next unvisited neighbor
    """
    for neighbor_index, neighbor_distance in neighbors:
        if neighbor_index not in visited_nodes:
            return neighbor_index, neighbor_distance

    raise ValueError("No unvisited neighbors")


def find_by_index(neighbors: list[neighbor], node_index: index) -> neighbor:
    """
    Finds neighbor in list by node index, not index in list
    """
    for neighbor in neighbors:
        if neighbor[0] == node_index:
            return neighbor

    raise ValueError(f"Neighbor of index {node_index} not present")
