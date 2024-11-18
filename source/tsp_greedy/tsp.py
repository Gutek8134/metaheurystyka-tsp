from source.mapping import node_distance, Matrix, node, index


def tsp(matrix: Matrix, home_node: node) -> tuple[float, list[index]]:
    """
    Greedy algorithm for traveling salesman problem

    :param matrix: adjacency matrix in which nodes are in format [x, y] sorted by distance
    :param home_index: id of node from which the salesman starts and returns to at the end

    :return: path length, indexes 
    """
    # Initialization
    visited_nodes: set[index] = {home_node[0]}
    path: list[index] = [home_node[0]]
    path_length: float = 0.

    nodes_to_visit: int = len(matrix)
    current_position: node = home_node

    # Traveling
    while len(visited_nodes) < nodes_to_visit:
        next_destination = find_next_destination(
            matrix[current_position[0]], visited_nodes)
        next_index, x, y = next_destination
        # region travel
        visited_nodes.add(next_index)
        path.append(next_index)
        path_length += node_distance(current_position, next_destination)

        current_position = next_destination

        # endregion travel

    # Going back home
    path_length += node_distance(current_position, home_node)

    return path_length, path


def find_next_destination(neighbors: list[node], visited_nodes: set[index]) -> node:
    """
    Helper getting next unvisited neighbor
    """
    for neighbor_index, neighbor_x, neighbor_y in neighbors:
        if neighbor_index not in visited_nodes:
            return neighbor_index, neighbor_x, neighbor_y

    raise ValueError("No unvisited neighbors")
