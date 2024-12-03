import random
from re import L, S
import numpy as np
from numpy.typing import NDArray, ArrayLike

from source.mapping import mutate, path_difference_length, random_paths, path_length, random_swap_sequence, subtract_paths

node_type = np.dtype(
    [("index", np.uint32), ("x", np.double), ("y", np.double)])


def S_ROA(nodes: NDArray | ArrayLike, initial_path: NDArray[np.uint32], population_size: int, max_iterations: int, blur_coefficient: float, max_distance_coefficient: float, blur_length: int = 2, max_speed: int = 600, max_h=1) -> tuple[NDArray[np.uint32], float]:
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
    number_of_followers: int = population_size//4
    number_of_overtakers: int = population_size//4
    number_of_attackers: int = population_size//4
    number_of_bypassers: int = population_size - \
        number_of_followers-number_of_overtakers-number_of_attackers

    followers_indexes: NDArray[np.uint32] = np.arange(
        0, number_of_followers, dtype=np.uint32)
    overtakers_indexes: NDArray[np.uint32] = np.arange(
        number_of_followers, number_of_followers+number_of_overtakers, dtype=np.uint32)
    attackers_indexes: NDArray[np.uint32] = np.arange(
        number_of_followers+number_of_overtakers, number_of_followers+number_of_overtakers+number_of_attackers, dtype=np.uint32)
    bypassers_indexes: NDArray[np.uint32] = np.arange(
        number_of_followers+number_of_overtakers+number_of_attackers, population_size, dtype=np.uint32)

    rider_hyenas: NDArray[np.uint32] = np.array(list(random_paths(
        population_size-1, number_of_cities)) + [tuple(initial_path)], dtype=np.uint32)

    rider_hyenas_indexes: NDArray[np.uint32] = np.zeros(
        (population_size, number_of_cities), dtype=np.uint32)

    for i, row in enumerate(rider_hyenas):
        rider_hyenas_indexes[i] = np.unique(
            rider_hyenas[i], return_index=True)[1]

    steering_angles: NDArray[np.double] = np.deg2rad(np.arange(
        population_size, dtype=np.double) * 360/population_size)
    gears: NDArray[np.uint8] = np.ones(population_size, dtype=np.uint8)
    accelerators: NDArray[np.double] = np.zeros(
        population_size, dtype=np.double)
    decelerators: NDArray[np.double] = np.ones(
        population_size, dtype=np.double)

    leader_index: np.intp = np.apply_along_axis(
        path_length, 1, rider_hyenas, nodes=nodes).argmin()

    leader_indexes: NDArray[np.uint32] = np.unique(
        rider_hyenas[leader_index], return_index=True)[1]

    rider_hyenas_success_rates: NDArray[np.double] = np.zeros(population_size)
    for i, rider in enumerate(rider_hyenas):
        rider_hyenas_success_rates[i] = 1 / \
            path_difference_length(
                rider, rider_hyenas[leader_index], leader_indexes)

    leader_index = rider_hyenas_success_rates.argmax()
    leader_indexes: NDArray[np.uint32] = np.unique(
        rider_hyenas[leader_index], return_index=True)[1]

    activity: NDArray[np.bool_] = np.zeros(population_size, dtype=np.bool_)
    # endregion initialization

    # D_h
    distance_from_prey: NDArray[np.uint32] = np.zeros(
        (population_size), dtype=np.uint32)
    swaps_from_prey: list[list[tuple[int, int]]] = []

    for i, hyena in enumerate(rider_hyenas):
        if i == leader_index:
            swaps_from_prey.append([])
            distance_from_prey[i] = 0
            continue

        # Changes the permutation just a little
        blurred_prey: NDArray[np.uint32] = np.copy(
            rider_hyenas[leader_index])
        swaps_array: NDArray[np.uint32] = random_swap_sequence(
            number_of_cities, blur_length)[:, np.random.random(blur_length) <= blur_coefficient]

        blurred_prey[[swaps_array[0], swaps_array[1]]
                     ] = blurred_prey[[swaps_array[1], swaps_array[0]]]

        # Calculates distance based on the altered position
        swaps_from_prey.append(subtract_paths(
            blurred_prey, hyena, rider_hyenas_indexes[i]))
        distance_from_prey[i] = len(swaps_from_prey[i])

    for iteration_count in range(max_iterations):
        # Update cluster
        cluster_hyenas_indexes, = (distance_from_prey <= np.random.uniform(
            0.5, 1, (population_size))*max_distance_coefficient).nonzero()
        cluster: NDArray[np.uint32]
        if cluster_hyenas_indexes.size > 0:
            cluster = rider_hyenas[cluster_hyenas_indexes]
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
        rider_hyenas[cluster_hyenas_indexes[activity[cluster_hyenas_indexes]]
                     ] = average_cluster_position

        h: float = max_h - iteration_count*max_h/max_iterations
        E: float = 2*h*np.random.random_sample() - h
        swap_op: tuple[int, int]

        steering_abscos = np.abs(np.cos(steering_angles))
        max_distance_to_travel = steering_abscos*(1/3*max_iterations)*(
            (gears*(max_speed/5))+(max_speed*accelerators)+((decelerators-1)*max_speed))
        for i in range(population_size):
            if activity[i]:
                if i == leader_index:
                    continue

                rider_hyenas[i] = np.copy(rider_hyenas[leader_index])
                for swap_op in reversed(swaps_from_prey[i]):
                    if np.random.random_sample() <= E:
                        rider_hyenas[i][[swap_op[0], swap_op[1]]] = rider_hyenas[i][[
                            swap_op[1], swap_op[0]]]

        # Followers update
        for index in followers_indexes[activity[followers_indexes]]:
            swaps = subtract_paths(
                rider_hyenas[index], rider_hyenas[leader_index], leader_indexes)
            if len(swaps) > max_distance_to_travel[i]:
                random.shuffle(swaps)
                swaps = swaps[:max_distance_to_travel[i]-1]
            for swap in swaps:
                rider_hyenas[index, [swap[0], swap[1]]
                             ] = rider_hyenas[index, [swap[1], swap[0]]]

        # Overtakers update
        direction_indicators = 1 - \
            (2/(1-np.log(rider_hyenas_success_rates /
                         rider_hyenas_success_rates.max())))
        for index in overtakers_indexes[activity[overtakers_indexes]]:
            swaps = subtract_paths(
                rider_hyenas[index], rider_hyenas[leader_index], leader_indexes)
            for swap in swaps:
                if random.random() <= direction_indicators[index]:
                    rider_hyenas[index, [swap[0], swap[1]]
                                 ] = rider_hyenas[index, [swap[1], swap[0]]]

        # Attackers update
        for index in attackers_indexes[activity[attackers_indexes]]:
            swaps = subtract_paths(
                rider_hyenas[index], rider_hyenas[leader_index], leader_indexes)
            for swap in swaps:
                if random.random() <= steering_abscos[index]:
                    rider_hyenas[index, [swap[0], swap[1]]
                                 ] = rider_hyenas[index, [swap[1], swap[0]]]

        # Bypassers update
        for index in bypassers_indexes[activity[bypassers_indexes]]:
            mutated_path: NDArray[np.uint32] = mutate(rider_hyenas[random.randrange(
                population_size)], rider_hyenas[random.randrange(population_size)], random.random(), number_of_cities)
            swaps: list[tuple[int, int]] = subtract_paths(
                rider_hyenas[index], mutated_path, np.unique_inverse(mutated_path)[1])
            swap_chance: float = random.random()
            for swap in swaps:
                if random.random() <= swap_chance:
                    rider_hyenas[index, [swap[0], swap[1]]
                                 ] = rider_hyenas[index, [swap[1], swap[0]]]

        rider_hyenas_success_rates = np.apply_along_axis(
            path_length, 1, rider_hyenas, nodes=nodes)

        for i, row in enumerate(rider_hyenas):
            for j, city_index in enumerate(row):
                rider_hyenas_indexes[i][city_index] = j
         # Activity
        activity = np.roll(
            rider_hyenas_success_rates, -1) > rider_hyenas_success_rates[iteration_count % population_size]

        # Steering angle
        steering_angles = np.where(activity, np.roll(
            steering_angles, 1), np.roll(steering_angles, -1))

        # Gear
        gears = np.clip(np.where(activity, gears-1, gears+1), 1, 4)

        # Accelerator
        accelerators = gears/4
        # Decelerator
        decelerators = 1-accelerators

        leader_index = rider_hyenas_success_rates.argmin()
        for i, hyena in enumerate(rider_hyenas):
            if i == leader_index:
                swaps_from_prey.append([])
                distance_from_prey[i] = 0
                continue

            # Changes the permutation just a little
            blurred_prey: NDArray[np.uint32] = np.copy(
                rider_hyenas[leader_index])
            swaps_array: NDArray[np.uint32] = random_swap_sequence(
                number_of_cities, blur_length)[:, np.random.random(blur_length) <= blur_coefficient]
            blurred_prey[[swaps_array[0], swaps_array[1]]
                         ] = blurred_prey[[swaps_array[1], swaps_array[0]]]

            # Calculates distance based on the altered position
            swaps_from_prey.append(subtract_paths(
                blurred_prey, hyena, rider_hyenas_indexes[i]))
            distance_from_prey[i] = len(swaps_from_prey[i])

        print(f"\riteration {iteration_count+1}/{max_iterations}", end="")
    print()

    return rider_hyenas[leader_index], rider_hyenas_success_rates[leader_index]
