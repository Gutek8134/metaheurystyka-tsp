import asyncio
from types import EllipsisType
import numpy as np
import random
from source.tsp_greedy.tsp import node, index, distance
from .mapping import get_coordinates, get_path

# TODO: switch lists and tuples to np.array
# TODO: async search


def SHO(nodes: list[node], number_of_hyenas: int, max_iterations: int, initial_prey_position: tuple[int, int], initial_paths: list[list[index]] | EllipsisType = ..., initial_coordinates: list[tuple[int, int]] | EllipsisType = ...) -> list[node]:

    hyenas_positions: list[tuple[int, int]]
    hyenas_paths: list[list[index]]
    number_of_cities: int = len(nodes)

    if initial_paths != ... and initial_coordinates != ...:
        raise ValueError("Both initial paths and initial coordinates are set")

    if isinstance(initial_paths, list):
        if len(initial_paths) >= number_of_hyenas:
            raise ValueError(
                "There are more initial paths than hyenas (1 hyena is reserved)")

        if not (isinstance(initial_paths[0], list) and isinstance(initial_paths[0][0], int)):
            raise TypeError("Initial paths must be a list of lists of indexes")

        hyenas_positions = list(map(get_coordinates, initial_paths))
        hyenas_paths = initial_paths

    elif initial_paths != ...:
        raise TypeError("Initial paths are set and not a list")

    elif isinstance(initial_coordinates, list):

        if len(initial_coordinates) >= number_of_hyenas:
            raise ValueError(
                "There are more initial coordinates than hyenas (1 hyena is reserved)")

        if not (isinstance(initial_coordinates[0], tuple) and isinstance(initial_coordinates[0][0], int)):
            raise TypeError(
                "Initial coordinates must be a list of tuples with x,y coordinates")

        hyenas_positions = initial_coordinates
        hyenas_paths = list(map(lambda x: get_path(
            x, number_of_cities), hyenas_positions))

    elif initial_coordinates != ...:
        raise TypeError("Initial coordinates are set and not a list")

    else:
        hyenas_positions = []
        hyenas_paths = []

    max_x = 1
    max_y = 1

    for i in range(number_of_cities-1, 1, -2):
        max_x *= i
        max_y *= i-1

    # Make sure there are exactly specified number of hyenas, including initial prey
    if len(hyenas_positions) < number_of_hyenas-1:

        additional_positions = ((random.randrange(0, max_x), random.randrange(
            0, max_y)) for _ in range(number_of_hyenas-1-len(hyenas_positions)))

        hyenas_positions.extend(
            additional_positions
        )
        hyenas_paths.extend(map(lambda x: get_path(
            x, number_of_cities), additional_positions))

    hyenas_positions.append(initial_prey_position)
    hyenas_paths.append(get_path(initial_prey_position, number_of_cities))

    # SHO start
    iteration_count: int = 0

    # Determine best hyena
    best_path_length: float = path_length(hyenas_paths[0], nodes)
    best_path: list[index] = hyenas_paths[0]
    best_hyena_position: tuple[int, int] = hyenas_positions[0]
    best_hyena_index: int = 0
    for i, path in enumerate(hyenas_paths[1:], start=1):
        if (length := path_length(path, nodes)) < best_path_length:
            best_path = path
            best_hyena_position = hyenas_positions[i]
            best_hyena_index = i
            best_path_length = length

    # Determine the cluster and calculate average of positions of hyenas in the cluster
    cluster: set[int] = set()
    cluster_vector: list[int] = [0, 0]

    motion_blur: tuple[float, float] = (
        2 * random.uniform(0, 1),
        2 * random.uniform(0, 1)
    )

    M: tuple[float, float] = random.uniform(0.5, 1), random.uniform(0.5, 1)
    for i, position in enumerate(hyenas_positions):
        if i == best_hyena_index:
            continue

        distance = distance_vector(position, best_hyena_position, motion_blur)
        if distance[0] < M[0] and distance[1] < M[1]:
            cluster.add(i)
            cluster_vector[0] += position[0]
            cluster_vector[1] = position[1]

    if len(cluster) > 0:
        cluster_vector[0] //= len(cluster)
        cluster_vector[1] //= len(cluster)

    while iteration_count < max_iterations:
        print(best_path_length, best_hyena_position)

        # Described as h vector in papers
        hunt_coefficient: float = 5 - ((iteration_count*5)/max_iterations)

        # Described as B vector in papers
        motion_blur: tuple[float, float] = (
            2 * random.uniform(0, 1),
            2 * random.uniform(0, 1)
        )

        # Described as E vector in papers
        effort: tuple[float, float] = (
            2*hunt_coefficient * random.uniform(0, 1) - hunt_coefficient,
            2*hunt_coefficient * random.uniform(0, 1) - hunt_coefficient
        )

        # attack with the cluster hyenas
        for hyena_index in cluster:
            hyenas_positions[hyena_index] = (
                hyenas_positions[hyena_index][0] + cluster_vector[0],
                hyenas_positions[hyena_index][1] + cluster_vector[1]
            )
            hyenas_paths[hyena_index] = get_path(
                hyenas_positions[hyena_index], number_of_cities)

        for i, position in enumerate(hyenas_positions):
            if i == best_hyena_index:
                continue

            distance = distance_vector(
                position, best_hyena_position, motion_blur)

            x = int(best_hyena_position[0] - effort[0]*distance[0])
            if x < 0:
                x = 0
            elif x > max_x:
                x = max_x

            y = int(best_hyena_position[1] - effort[1]*distance[1])
            if y < 0:
                y = 0
            elif y > max_y:
                y = max_y

            hyenas_positions[i] = (x, y)

        # Get new best spotted hyena
        for i, path in enumerate(hyenas_paths[1:], start=1):
            if (length := path_length(path, nodes)) < best_path_length:
                best_path = path
                best_hyena_position = hyenas_positions[i]
                best_hyena_index = i
                best_path_length = length

        # Update cluster
        cluster_vector = [0, 0]
        M: tuple[float, float] = random.uniform(0.5, 1), random.uniform(0.5, 1)
        for i, position in enumerate(hyenas_positions):
            if i == best_hyena_index:
                continue

            distance = distance_vector(
                position, best_hyena_position, motion_blur)
            if distance[0] < M[0] and distance[1] < M[1]:
                cluster.add(i)
                cluster_vector[0] += position[0]
                cluster_vector[1] = position[1]

        if len(cluster) > 0:
            cluster_vector[0] //= len(cluster)
            cluster_vector[1] //= len(cluster)

        iteration_count += 1

    return list(map(lambda x: nodes[x-1], best_path))


def path_length(path: list[int], nodes: list[node]) -> float:
    cost: float = 0.
    for path_index, city_index in enumerate(path[:-1]):
        cost += distance(nodes[city_index], nodes[path[path_index+1]])
    cost += distance(nodes[path[0]], nodes[path[-1]])

    return cost


def distance_vector(a: tuple[int, int], b: tuple[int, int], motion_blur: tuple[float, float]) -> tuple[float, float]:
    return (
        abs(motion_blur[0] * a[0] - b[0]),
        abs(motion_blur[1] * a[1] - b[1])
    )
