import asyncio
from types import EllipsisType
import numpy as np
import random
from source.tsp_greedy.tsp import node, index, distance
from .mapping import get_coordinates, get_path

# TODO: switch lists and tuples to np.array
# TODO: async search

def SHO(nodes: list[node], number_of_hyenas: int, max_iterations: int, initial_paths: list[list[index]]|EllipsisType=..., initial_coordinates:list[tuple[int, int]]|EllipsisType=..., _number_of_cities:int|EllipsisType=...) -> list[node]:

    hyenas_positions: list[tuple[int, int]]
    number_of_cities: int

    if initial_paths != ... and initial_coordinates != ...:
        raise ValueError("Both initial paths and initial coordinates are set")

    if isinstance(initial_paths, list):
        if len(initial_paths) > number_of_hyenas:
            raise ValueError("There are more initial paths than hyenas")
        
        if not (isinstance(initial_paths[0], list) and isinstance(initial_paths[0][0], int)):
            raise TypeError("Initial paths must be a list of lists of indexes")
        
        hyenas_positions = list(map(get_coordinates, initial_paths))
        number_of_cities = len(initial_paths[0])

    
    elif initial_paths != ...:
        raise TypeError("Initial paths are set and not a list")
    
    elif isinstance(initial_coordinates, list):
        if not isinstance(_number_of_cities, int):
            raise TypeError("When giving initial coordinates you have to supply number of cities")

        if len(initial_coordinates) > number_of_hyenas:
            raise ValueError("There are more initial coordinates than hyenas")
        
        if not (isinstance(initial_coordinates[0], tuple) and isinstance(initial_coordinates[0][0], int)):
            raise TypeError("Initial coordinates must be a list of tuples with x,y coordinates")
        
        hyenas_positions = initial_coordinates
        number_of_cities = _number_of_cities
    
    elif initial_coordinates != ...:
        raise TypeError("Initial coordinates are set and not a list")

    else:
        hyenas_positions = []
    
    if len(hyenas_positions) < number_of_hyenas:
        max_x = 1
        max_y = 1

        for i in range(len(nodes)-1, 1, -2):
            max_x *= i
            max_y *= i-1

        hyenas_positions.extend(
            random.sample([(x, y) for x in range(max_x) for y in range(max_y)],
                number_of_hyenas-len(hyenas_positions)
            )
        )

    iteration_count: int = 0

    # Described as h vector in papers
    hunt_coefficient: float = 5 - ((iteration_count*5)/max_iterations)

    # Described as B vector in papers
    motion_blur: tuple[float, float] = (
        2 * random.uniform(0,1),
        2 * random.uniform(0,1)
    )

    # Described as E vector in papers
    effort: tuple[float, float] = (
        2*hunt_coefficient * random.uniform(0,1) - hunt_coefficient,
        2*hunt_coefficient * random.uniform(0,1) - hunt_coefficient
    )

    best_path_length: float = path_length(get_path(*hyenas_positions[0], 1), nodes)
    best_path: tuple[int, int] = hyenas_positions[0]
    for path in hyenas_positions[1:]:
        if (length := path_length(get_path(*path, number_of_cities), nodes)) < best_path_length:
            best_path = path
            best_path_length = length

    # Described as C_h group in papers
    best_paths: set[tuple[int, int]] = {best_path}

    while iteration_count < max_iterations:
        # hunting

        pass
        iteration_count += 1

    raise NotImplementedError

def path_length(path: list[int], nodes: list[node]):
    return sum(distance(nodes[index], nodes[index+1]) if index != len(nodes)-1 else distance(nodes[index], nodes[0]) for index in path)