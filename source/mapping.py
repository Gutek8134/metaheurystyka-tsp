"""
Representing solutions as swap sequences
https://research.ijcaonline.org/volume47/number15/pxc3880348.pdf
"""

from math import sqrt
import numpy as np
from numpy.typing import NDArray
import random

type index = int
type node = tuple[index, float, float]
type Matrix = list[list[node]]

rng: np.random.Generator = np.random.default_rng()


def random_paths(number_of_paths: int, number_of_cities: int) -> set[tuple[int, ...]]:
    """
    :return paths: set of 0-based tuples being permutations of 0..number of cities-1 that represent a path
    """
    paths: set[tuple[int, ...]] = set()
    cities: list[int] = list(range(1, number_of_cities))

    for _ in range(number_of_paths):
        random.shuffle(cities)
        while (tuple([0]+cities)) in paths:
            random.shuffle(cities)
        paths.add(tuple([0]+cities))

    return paths


def subtract_paths(path_a: NDArray[np.uint32], path_b: NDArray[np.uint32], path_b_indexes: NDArray[np.uint32]) -> list[tuple[int, int]]:
    """
    Gives path_b - path_a

    :params:
    For indexes, if path looked like `[1,2,0]` you'd input `[2,0,1]`

    :returns:
    List of swap operators, here represented by pair of ints
    """
    difference: list[tuple[int, int]] = []
    path_a_copy = path_a.copy()
    for i, element in enumerate(path_b):
        if element != path_a_copy[i]:
            # Swap the elements in copy of path_a
            cache = path_b_indexes[path_a_copy[i]]
            path_a_copy[[i, cache]
                        ] = path_a_copy[[cache, i]]

            # Add the operator to difference
            difference.append((i, int(cache)))

    return difference


def random_swap_sequence(number_of_cities: int, length: int) -> NDArray[np.uint32]:
    return np.asarray(rng.choice(a=number_of_cities, size=(2, length), replace=False), dtype=np.uint32)


def path_length(path: NDArray[np.uint32], nodes) -> float:
    """
    Calculate length of path using euclidean distance

    :param nodes: 1D array composed of (index: int, x: float, y: float)
    :param path: permutation of 0..number of cities-1 - 0-based indexes of cities in TSP
    """

    cost: float = 0
    for path_index, city_index in enumerate(path[:-1]):
        cost += distance(nodes[city_index], nodes[path[path_index+1]])
    cost += distance(nodes[path[0]], nodes[path[-1]])

    return cost


def node_path_length(path: list[int], nodes: list[node]) -> float:
    cost: float = 0.
    for path_index, city_index in enumerate(path[:-1]):
        cost += node_distance(nodes[city_index], nodes[path[path_index+1]])
    cost += node_distance(nodes[path[0]], nodes[path[-1]])

    return cost


def node_distance(a: node, b: node) -> float:
    return sqrt((a[1]-b[1])**2+(a[2]-b[2])**2)


def distance(a: NDArray, b: NDArray) -> float:
    """
    Euclidean distance

    :params: numpy scalar of type `particles.node` (index: int, x: float, y: float)
    """
    return sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2)