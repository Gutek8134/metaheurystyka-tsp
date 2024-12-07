import random
import numpy as np
from numpy.typing import NDArray, ArrayLike

from source.mapping import mutate, path_difference_length, path_length, random_paths, subtract_paths

node_type = np.dtype(
    [("index", np.uint32), ("x", np.double), ("y", np.double)])


def ROA(nodes: NDArray | ArrayLike, initial_path: NDArray[np.uint32], number_of_riders: int, max_iterations: int, max_speed: int = 600, swap_chance: float = 0.6) -> tuple[NDArray[np.uint32], float]:
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

    number_of_followers: int = number_of_riders//4
    number_of_overtakers: int = number_of_riders//4
    number_of_attackers: int = number_of_riders//4
    number_of_bypassers: int = number_of_riders - \
        number_of_followers-number_of_overtakers-number_of_attackers

    followers_indexes: NDArray[np.uint32] = np.arange(
        0, number_of_followers, dtype=np.uint32)
    overtakers_indexes: NDArray[np.uint32] = np.arange(
        number_of_followers, number_of_followers+number_of_overtakers, dtype=np.uint32)
    attackers_indexes: NDArray[np.uint32] = np.arange(
        number_of_followers+number_of_overtakers, number_of_followers+number_of_overtakers+number_of_attackers, dtype=np.uint32)
    bypassers_indexes: NDArray[np.uint32] = np.arange(
        number_of_followers+number_of_overtakers+number_of_attackers, number_of_riders, dtype=np.uint32)

    riders: NDArray[np.uint32] = np.array(list(random_paths(
        number_of_riders-1, number_of_cities)) + [tuple(initial_path)], dtype=np.uint32)

    # riders_indexes: NDArray[np.uint32] = np.zeros(
    #     (number_of_riders, number_of_cities), dtype=np.uint32)

    # for i, row in enumerate(riders):
    #     for j, city_index in enumerate(row):
    #         riders_indexes[i][city_index] = j

    steering_angles: NDArray[np.double] = np.deg2rad(np.arange(
        number_of_riders, dtype=np.double) * 360/number_of_riders)
    gears: NDArray[np.uint8] = np.ones(number_of_riders, dtype=np.uint8)
    accelerators: NDArray[np.double] = np.zeros(
        number_of_riders, dtype=np.double)
    decelerators: NDArray[np.double] = np.ones(
        number_of_riders, dtype=np.double)

    leader_index: np.intp = np.apply_along_axis(
        path_length, 1, riders, nodes=nodes).argmin()

    leader_indexes: NDArray[np.uint32] = np.unique(
        riders[leader_index], return_index=True)[1]

    # Higher is better
    success_rates: NDArray[np.double] = np.zeros(number_of_riders)
    for i, rider in enumerate(riders):
        success_rates[i] = 1 / \
            path_difference_length(rider, riders[leader_index], leader_indexes)

    leader_index = success_rates.argmax()
    leader_indexes: NDArray[np.uint32] = np.unique(
        riders[leader_index], return_index=True)[1]

    activity: NDArray[np.bool_] = np.zeros(number_of_riders, dtype=np.bool_)
    # endregion initialization

    for iteration_count in range(max_iterations):
        # Followers update
        steering_abscos = np.abs(np.cos(steering_angles))
        max_distance_to_travel = steering_abscos*(1/3*max_iterations)*(
            (gears*(max_speed/5))+(max_speed*accelerators)+((decelerators-1)*max_speed))

        leader_found: bool = False
        for index in followers_indexes:
            if not leader_found and index == leader_index:
                leader_found = True
                continue
            swaps = subtract_paths(
                riders[index], riders[leader_index], leader_indexes)
            if len(swaps) > max_distance_to_travel[i]:
                random.shuffle(swaps)
                swaps = swaps[:max_distance_to_travel[i]-1]
            for swap in swaps:
                riders[index, [swap[0], swap[1]]
                       ] = riders[index, [swap[1], swap[0]]]

        # Overtakers update
        direction_indicators = 1 - \
            (2/(1-np.log(success_rates/success_rates.max())))
        for index in overtakers_indexes:
            if not leader_found and index == leader_index:
                leader_found = True
                continue
            swaps = subtract_paths(
                riders[index], riders[leader_index], leader_indexes)
            for swap in swaps:
                if random.random() <= direction_indicators[index]:
                    riders[index, [swap[0], swap[1]]
                           ] = riders[index, [swap[1], swap[0]]]

        # Attackers update
        for index in attackers_indexes:
            if not leader_found and index == leader_index:
                leader_found = True
                continue
            swaps = subtract_paths(
                riders[index], riders[leader_index], leader_indexes)
            for swap in swaps:
                if random.random() <= steering_abscos[index]:
                    riders[index, [swap[0], swap[1]]
                           ] = riders[index, [swap[1], swap[0]]]

        # Bypassers update
        for index in bypassers_indexes:
            if not leader_found and index == leader_index:
                leader_found = True
                continue
            mutated_path: NDArray[np.uint32] = mutate(riders[random.randrange(
                number_of_riders)], riders[random.randrange(number_of_riders)], random.random(), number_of_cities)
            swaps: list[tuple[int, int]] = subtract_paths(
                riders[index], mutated_path, np.unique_inverse(mutated_path)[1])
            for swap in swaps:
                if random.random() <= swap_chance:
                    riders[index, [swap[0], swap[1]]
                           ] = riders[index, [swap[1], swap[0]]]

        # Parameter update

        # Success rate
        success_rates = 1/np.apply_along_axis(
            path_difference_length, 1, riders, path_b=riders[leader_index], path_b_indexes=leader_indexes)

        # Activity
        activity = np.roll(
            success_rates, -1) > success_rates[iteration_count % number_of_riders]

        # Steering angle
        steering_angles = np.where(activity, np.roll(
            steering_angles, 1), np.roll(steering_angles, -1))

        # Gear
        gears = np.clip(np.where(activity, gears-1, gears+1), 1, 4)

        # Accelerator
        accelerators = gears/4
        # Decelerator
        decelerators = 1-accelerators

        print(f"\riteration {iteration_count+1}/{max_iterations}", end="")
    print()

    return riders[leader_index], path_length(riders[leader_index], nodes)
