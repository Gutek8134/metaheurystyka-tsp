"""
Helpers to map TSP solution space into a 2d space
https://core.ac.uk/download/pdf/81128867.pdf
"""
from itertools import cycle, islice
from operator import le
from ..tsp_greedy.tsp import index


def get_coordinates(path: list[index]) -> tuple[int, int]:
    """
    Changes path to x, y coordinates
    :param path: list of indexes of nodes on path, each must appear once
    """

    number_of_cities = len(path)
    position_of_city_1 = path.index(1)

    # Shift path so it starts from 1
    path_from_1: list[index] = list(
        islice(cycle(path), position_of_city_1, position_of_city_1+number_of_cities))

    bag = list(range(1, number_of_cities+2))

    x = 0
    y = 0

    for i, path_index in enumerate(path_from_1[1:], start=1):
        for j in range(1, len(bag)):
            if path_index == bag[j]:
                block = 1
                if i % 2 == 0:
                    for k in range(3, number_of_cities-i-1, 2):
                        block *= k
                    x += (j-1)*block

                else:
                    for k in range(2, number_of_cities-i-1, 2):
                        block *= k
                    y += (j-1)*block

                for l in range(j, number_of_cities):
                    bag[l] = bag[l+1]
                break

    return y, x
