"""
Created based on
https://github.com/nsn39/particle-swarm/
"""


import random
import numpy as np
from numpy.typing import NDArray, ArrayLike
from ..mapping import random_paths, path_length, subtract_paths

node_type = np.dtype(
    [("index", np.uint32), ("x", np.double), ("y", np.double)])


def PSO(nodes: NDArray | ArrayLike, initial_path: NDArray[np.uint32], population_size: int, max_iterations: int, alpha: float = 1, beta: float = 1) -> tuple[NDArray[np.uint32], float]:
    """
    :param nodes: must start with node labeled with 1 or 0, whichever is lowest
    """

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

    # Saved for showing off how the path was created
    evolutions: NDArray[np.uint32] = np.zeros(
        (max_iterations, number_of_cities), dtype=np.uint32)
    evolutions_costs: NDArray[np.double] = np.full(
        (max_iterations), -1, dtype=np.double)

    # Instead of particle object, each of potential fields is saved as a cumulative array
    particle_solutions = np.array(list(random_paths(
        population_size-1, number_of_cities)) + [tuple(initial_path)], dtype=np.uint32)
    """
    2D array, first index is an index of a particle
    Solution is a permutation of 0..number of cities-1, starting with 0, representing sequence of visiting cities
    """
    personal_bests_indexes = np.zeros(
        (population_size, number_of_cities), dtype=np.uint32)
    """
    2D array, first index is an index of a particle
    Holds indexes of every city in given particle's best solution
    """

    # At first, the best path is the only path you know
    personal_bests: NDArray[np.uint32] = particle_solutions.copy()
    """
    2D array where first index is index of particle, and points to an array of indexes representing a potential solution
    """

    personal_best_lengths: NDArray[np.double] = np.array(
        [path_length(path, nodes) for path in personal_bests], dtype=np.double)

    global_best: NDArray[np.uint32]
    """
    global best solution
    """

    global_best_length: float

    global_best_indexes: NDArray[np.uint32]

    # Indexes initialization
    for index, solution in enumerate(personal_bests):

        temp_personal_indexes = np.zeros((number_of_cities), dtype=np.uint32)
        for j, city_index in enumerate(solution):
            temp_personal_indexes[city_index] = j

        personal_bests_indexes[index] = temp_personal_indexes

    # print(personal_best_lengths)
    # endregion initialization

    print()
    for iteration_count in range(max_iterations):
        # Update global best
        global_best = min(personal_bests,
                          key=lambda x: path_length(x, nodes))
        global_best_length = path_length(global_best, nodes)

        # if iteration_count == 0:
        #     print("Global best", global_best_length)
        #     print("Initial", path_length(nodes, initial_path))

        evolutions[iteration_count] = global_best
        evolutions_costs[iteration_count] = global_best_length

        temp_global_indexes = np.zeros((number_of_cities), dtype=np.uint32)
        for i, city_index in enumerate(global_best):
            temp_global_indexes[city_index] = i

        global_best_indexes = temp_global_indexes

        for j, particle_path in enumerate(particle_solutions):
            # (pbest - x(t-1))
            pbest_difference: list[tuple[int, int]] = subtract_paths(
                particle_path, personal_bests[j], personal_bests_indexes[j])

            # (gbest - x(t-1))
            gbest_difference: list[tuple[int, int]] = subtract_paths(
                particle_path, global_best, global_best_indexes)

            for swap_operator in pbest_difference:
                if random.random() <= alpha:
                    # Swap
                    particle_path[[swap_operator[0], swap_operator[1]]] = particle_path[[
                        swap_operator[1], swap_operator[0]]]

            for swap_operator in gbest_difference:
                if random.random() <= beta:
                    # Swap
                    particle_path[[swap_operator[0], swap_operator[1]]] = particle_path[[
                        swap_operator[1], swap_operator[0]]]

            # Update personal best
            new_path_length = path_length(particle_path, nodes)
            # print(i, personal_best_lengths[i], new_path_length)
            if personal_best_lengths[j] > new_path_length:
                personal_bests[j] = particle_path
                personal_best_lengths[j] = new_path_length

                # Update indexes
                temp_personal_indexes = np.zeros(
                    (number_of_cities), dtype=np.uint32)
                for k, city_index in enumerate(solution):
                    temp_personal_indexes[city_index] = k
                personal_bests_indexes[j] = temp_personal_indexes

        print(f"\riteration {iteration_count+1}/{max_iterations}", end="")
    print()
    global_best = min(personal_bests,
                      key=lambda x: path_length(x, nodes))
    global_best_length = path_length(global_best, nodes)

    return global_best, global_best_length
