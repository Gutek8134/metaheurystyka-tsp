from re import L, S
import numpy as np
from numpy.typing import NDArray, ArrayLike

from source.mapping import random_paths, path_length, random_swap_sequence, subtract_paths

node_type = np.dtype(
    [("index", np.uint32), ("x", np.double), ("y", np.double)])


def SHO(nodes: NDArray | ArrayLike, initial_path: NDArray[np.uint32], population_size: int, max_iterations: int, blur_coefficient: float, max_distance_coefficient: float, blur_length: int = 2, max_h=1) -> tuple[NDArray[np.uint32], float]:
    # region initialization
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes, node_type)

    elif nodes.dtype != node_type:
        nodes = nodes.astype(node_type, copy=False)

    # Check for 1-based indexing and convert to 0-based
    if nodes[0][0] == 1:
        for node in nodes:
            node[0] -= 1

    if initial_path[0] == 1:
        initial_path -= 1

    # Length of the path
    number_of_cities: int = nodes.size

    hyenas_population: NDArray[np.uint32] = np.array(list(random_paths(
        population_size-1, number_of_cities)) + [tuple(initial_path)], dtype=np.uint32)

    hyenas_fitness: NDArray[np.double] = np.apply_along_axis(
        path_length, 1, hyenas_population, nodes=nodes)

    hyenas_indexes: NDArray[np.uint32] = np.zeros(
        (population_size, number_of_cities), dtype=np.uint32)

    for i, row in enumerate(hyenas_population):
        hyenas_indexes[i] = np.unique(
            hyenas_population[i], return_index=True)[1]

    # endregion initialization
    prey_index: np.intp = hyenas_fitness.argmin()

    # D_h
    distance_from_prey: NDArray[np.uint32] = np.zeros(
        (population_size), dtype=np.uint32)
    swaps_from_prey: list[list[tuple[int, int]]] = []

    for i, hyena in enumerate(hyenas_population):
        if i == prey_index:
            swaps_from_prey.append([])
            distance_from_prey[i] = 0
            continue

        # Changes the permutation just a little
        blurred_prey: NDArray[np.uint32] = np.copy(
            hyenas_population[prey_index])
        swap: NDArray[np.uint32] = random_swap_sequence(
            number_of_cities, blur_length)[:, np.random.random(blur_length) <= blur_coefficient]

        blurred_prey[[swap[0], swap[1]]] = blurred_prey[[swap[1], swap[0]]]

        # Calculates distance based on the altered position
        swaps_from_prey.append(subtract_paths(
            blurred_prey, hyena, hyenas_indexes[i]))
        distance_from_prey[i] = len(swaps_from_prey[i])

    for iteration_count in range(max_iterations):
        # Update cluster
        cluster_hyenas_indexes, = (distance_from_prey <= np.random.uniform(
            0.5, 1, (population_size))*max_distance_coefficient).nonzero()
        cluster: NDArray[np.uint32]
        if cluster_hyenas_indexes.size > 0:
            cluster = hyenas_population[cluster_hyenas_indexes]
        else:
            cluster = np.array([], dtype=np.uint32)

        # Counts how many times each city appeared at the position
        cluster_counts = [np.array([np.count_nonzero(cluster[:, i] == j) for j in range(number_of_cities)], dtype=np.uint32)
                          for i in range(number_of_cities)]

        # Technically speaking it's closer to median
        average_cluster_position = np.zeros(
            (number_of_cities), dtype=np.uint32)

        # As much as I'd want to vectorize, I'm unable to
        available_positions = np.ones(number_of_cities, dtype=np.uint32)
        for i in range(number_of_cities):
            temp = np.where(available_positions > 0,
                            cluster_counts[i], 0).argmax()
            if available_positions[temp] == 0:
                temp = np.where(available_positions == 1)[0][0]
            average_cluster_position[i] = temp
            available_positions[temp] = 0

        hyenas_population[cluster_hyenas_indexes] = average_cluster_position

        h: float = max_h - iteration_count*max_h/max_iterations
        E: float = 2*h*np.random.random_sample() - h
        swap_op: tuple[int, int]

        for i in range(population_size):
            if i == prey_index:
                continue

            hyenas_population[i] = np.copy(hyenas_population[prey_index])
            for swap_op in reversed(swaps_from_prey[i]):
                if np.random.random_sample() <= E:
                    hyenas_population[i][[swap_op[0], swap_op[1]]] = hyenas_population[i][[
                        swap_op[1], swap_op[0]]]

        hyenas_fitness = np.apply_along_axis(
            path_length, 1, hyenas_population, nodes=nodes)

        for i, row in enumerate(hyenas_population):
            for j, city_index in enumerate(row):
                hyenas_indexes[i][city_index] = j

        prey_index = hyenas_fitness.argmin()
        for i, hyena in enumerate(hyenas_population):
            if i == prey_index:
                swaps_from_prey.append([])
                distance_from_prey[i] = 0
                continue

            # Changes the permutation just a little
            blurred_prey: NDArray[np.uint32] = np.copy(
                hyenas_population[prey_index])
            swap: NDArray[np.uint32] = random_swap_sequence(
                number_of_cities, blur_length)[:, np.random.random(blur_length) <= blur_coefficient]
            blurred_prey[[swap[0], swap[1]]
                         ] = blurred_prey[[swap[1], swap[0]]]

            # Calculates distance based on the altered position
            swaps_from_prey.append(subtract_paths(
                blurred_prey, hyena, hyenas_indexes[i]))
            distance_from_prey[i] = len(swaps_from_prey[i])

        print(f"\riteration {iteration_count+1}/{max_iterations}", end="")
    print()

    return hyenas_population[prey_index], hyenas_fitness[prey_index]
