from multiprocessing import Value
import sys
import argparse
from time import monotonic_ns
import random
import numpy as np
from source.S_ROA.rider_hyenas import S_ROA
from source.mapping import node_path_length
from source.instance_generator import random_instance
from source.tsp_greedy.parser import parse
from source.tsp_greedy.tsp import tsp
from numpy.typing import NDArray
import networkx as nx
import networkx.drawing as nxdraw
from matplotlib import pyplot as plt


def draw_path(nodes: list, path: NDArray[np.uint32]) -> None:
    graph = nx.Graph((int(x), int(y)) for x, y in zip(path, np.roll(path, 1)))
    positions: dict[int, tuple[int, int]] = {}
    for node in nodes:
        positions[node[0]] = (node[1], node[2])
    nxdraw.draw(graph, positions, node_size=3)
    plt.show()


def random_random(*, _max_iterations: int | None = None, _population_size: int | None = None, _blur_coff: int | None = None, _swap_chance: int | None = None, _blur_len: int | None = None, _max_distance_coefficient: int | None = None, _max_speed: int | None = None, _no_improvement_iterations: int | None = None, _shuffle: float | None = None) -> None:
    instance_size = 10
    _, nodes = parse(random_instance(
        instance_size, max_x=50, max_y=50))
    path = list(range(1, instance_size))
    random.shuffle(path)
    path.insert(0, 0)
    random_length = node_path_length(path, nodes)
    array_path = np.array(path)
    # More = better, but takes more time
    population_size = _population_size or 10
    max_iterations = _max_iterations or 30
    # Values between 0 and 1
    blur_coff = _blur_coff or 0.7
    swap_chance = _swap_chance or 0.6
    shuffle_chance = _shuffle or 1.
    # Max: instance_size//2
    max_dist_coff = _max_distance_coefficient or 15
    # Not sure what's the max value here
    blur_len = _blur_len or 5
    max_speed = _max_speed or 600
    max_iterations_without_improvement = _no_improvement_iterations or 200

    shuffle = _shuffle is not None
    s_roa_start_time = monotonic_ns()
    s_roa_path, s_roa_length = S_ROA(nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                                     blur_coefficient=blur_coff, max_distance_coefficient=max_dist_coff, blur_length=blur_len, max_speed=max_speed,
                                     swap_chance=swap_chance, max_iterations_without_improvement=max_iterations_without_improvement, shuffle_instead=shuffle, shuffle_chance=shuffle_chance)
    s_roa_end_time = monotonic_ns()

    print(f"Random: {random_length}\n"
          f"S-ROA: {s_roa_length} in {(s_roa_end_time-s_roa_start_time)/1e9}s")

    draw_path(nodes, s_roa_path)


def greedy_random(*, _max_iterations: int | None = None, _population_size: int | None = None, _blur_coff: int | None = None, _swap_chance: int | None = None, _blur_len: int | None = None, _max_distance_coefficient: int | None = None, _max_speed: int | None = None, _no_improvement_iterations: int | None = None, _shuffle: float | None = None) -> None:
    instance_size = 100
    matrix, nodes = parse(random_instance(
        instance_size, max_x=1000, max_y=1000))
    greedy_length, greedy_path = tsp(matrix, nodes[0])
    array_path = np.array(greedy_path)
    # More = better, but takes more time
    population_size = _max_iterations or 200
    max_iterations = _population_size or 500
    # Values between 0 and 1
    blur_coff = _blur_coff or 0.55
    swap_chance = _swap_chance or 0.6
    shuffle_chance = _shuffle or 1.
    # Max: instance_size//2
    max_dist_coff = _blur_len or 22
    # Not sure what's the max value here
    blur_len = _max_distance_coefficient or 22
    max_speed = _max_speed or 600
    max_iterations_without_improvement = _no_improvement_iterations or 200

    shuffle = _shuffle is not None
    s_roa_start_time = monotonic_ns()
    s_roa_path, s_roa_length = S_ROA(nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                                     blur_coefficient=blur_coff, max_distance_coefficient=max_dist_coff, blur_length=blur_len, max_speed=max_speed,
                                     swap_chance=swap_chance, max_iterations_without_improvement=max_iterations_without_improvement, shuffle_instead=shuffle, shuffle_chance=shuffle_chance)
    s_roa_end_time = monotonic_ns()

    print(f"Greedy: {greedy_length}\n"
          f"S-ROA: {s_roa_length} in {(s_roa_end_time-s_roa_start_time)/1e9}s")

    draw_path(nodes, s_roa_path)


def greedy_berlin52(*, _max_iterations: int | None = None, _population_size: int | None = None, _blur_coff: int | None = None, _swap_chance: int | None = None, _blur_len: int | None = None, _max_distance_coefficient: int | None = None, _max_speed: int | None = None, _no_improvement_iterations: int | None = None, _shuffle: float | None = None) -> None:
    with open("berlin52.txt") as f:
        matrix, nodes = parse(f.read())
    greedy_length, greedy_path = tsp(matrix, nodes[0])
    array_path = np.array(greedy_path)
    # More = better, but takes more time
    max_iterations = _max_iterations or 1500
    population_size = _population_size or 400
    # Values between 0 and 1
    blur_coff = _blur_coff or 0.65
    swap_chance = _swap_chance or 0.87
    shuffle_chance = _shuffle or 1.
    # Max: instance_size//2
    blur_len = _blur_len or 26
    # Not sure what's the max value here
    max_dist_coff = _max_distance_coefficient or 28
    max_speed = _max_speed or 1400
    max_iterations_without_improvement = _no_improvement_iterations or 300

    shuffle = _shuffle is not None
    s_roa_start_time = monotonic_ns()
    s_roa_path, s_roa_length = S_ROA(nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                                     blur_coefficient=blur_coff, max_distance_coefficient=max_dist_coff, blur_length=blur_len, max_speed=max_speed,
                                     swap_chance=swap_chance, max_iterations_without_improvement=max_iterations_without_improvement, shuffle_instead=shuffle, shuffle_chance=shuffle_chance)
    s_roa_end_time = monotonic_ns()

    print(f"Settings:\n\t{population_size=}\n\t{blur_coff=}\n\t{swap_chance=}\n\t{max_dist_coff=}\n\t{blur_len=}\n\t{max_speed=}\n"
          f"Greedy: {greedy_length}\n"
          f"S-ROA: {s_roa_length} in {(s_roa_end_time -
                                       s_roa_start_time)/1e9}s\n"
          "Optimum: 7544")

    draw_path(nodes, s_roa_path)


def greedy_bier127(*, _max_iterations: int | None = None, _population_size: int | None = None, _blur_coff: int | None = None, _swap_chance: int | None = None, _blur_len: int | None = None, _max_distance_coefficient: int | None = None, _max_speed: int | None = None, _no_improvement_iterations: int | None = None, _shuffle: float | None = None) -> None:
    with open("bier127.txt") as f:
        matrix, nodes = parse(f.read())
    greedy_length, greedy_path = tsp(matrix, nodes[0])
    array_path = np.array(greedy_path)
    # More = better, but takes more time
    max_iterations = _max_iterations or 800
    population_size = _population_size or 300
    # Values between 0 and 1
    blur_coff = _blur_coff or 0.21
    swap_chance = _swap_chance or 0.86
    shuffle_chance = _shuffle or 1.
    # Max: instance_size//2
    blur_len = _blur_len or 53
    # Not sure what's the max value here
    max_dist_coff = _max_distance_coefficient or 55
    max_speed = _max_speed or 1650
    max_iterations_without_improvement = _no_improvement_iterations or 130

    shuffle = _shuffle is not None
    s_roa_start_time = monotonic_ns()
    s_roa_path, s_roa_length = S_ROA(nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                                     blur_coefficient=blur_coff, max_distance_coefficient=max_dist_coff, blur_length=blur_len, max_speed=max_speed,
                                     swap_chance=swap_chance, max_iterations_without_improvement=max_iterations_without_improvement, shuffle_instead=shuffle, shuffle_chance=shuffle_chance)
    s_roa_end_time = monotonic_ns()

    print(f"Settings:\n\t{population_size=}\n\t{blur_coff=}\n\t{swap_chance=}\n\t{max_dist_coff=}\n\t{blur_len=}\n\t{max_speed=}\n"
          f"Greedy: {greedy_length}\n"
          f"S-ROA: {s_roa_length} in {(s_roa_end_time -
                                       s_roa_start_time)/1e9}s\n"
          "Optimum: 118282")

    draw_path(nodes, s_roa_path)


def greedy_tsp250(*, _max_iterations: int | None = None, _population_size: int | None = None, _blur_coff: int | None = None, _swap_chance: int | None = None, _blur_len: int | None = None, _max_distance_coefficient: int | None = None, _max_speed: int | None = None, _no_improvement_iterations: int | None = None, _shuffle: float | None = None) -> None:
    with open("tsp250.txt") as f:
        matrix, nodes = parse(f.read())
    greedy_length, greedy_path = tsp(matrix, nodes[0])
    array_path = np.array(greedy_path)
    # More = better, but takes more time
    max_iterations = _max_iterations or 500
    population_size = _population_size or 300
    # Values between 0 and 1
    blur_coff = _blur_coff or 0.135
    swap_chance = _swap_chance or 0.8
    shuffle_chance = _shuffle or 1.
    # Max: instance_size//2
    blur_len = _blur_len or 65
    # Not sure what's the max value here
    max_dist_coff = _max_distance_coefficient or 50
    max_speed = _max_speed or 1950
    max_iterations_without_improvement = _no_improvement_iterations or 80

    shuffle = _shuffle is not None
    s_roa_start_time = monotonic_ns()
    s_roa_path, s_roa_length = S_ROA(nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                                     blur_coefficient=blur_coff, max_distance_coefficient=max_dist_coff, blur_length=blur_len, max_speed=max_speed,
                                     swap_chance=swap_chance, max_iterations_without_improvement=max_iterations_without_improvement, shuffle_instead=shuffle, shuffle_chance=shuffle_chance)
    s_roa_end_time = monotonic_ns()

    print(f"Settings:\n\t{population_size=}\n\t{blur_coff=}\n\t{swap_chance=}\n\t{max_dist_coff=}\n\t{blur_len=}\n\t{max_speed=}\n"
          f"Greedy: {greedy_length}\n"
          f"S-ROA: {s_roa_length} in {(s_roa_end_time -
                                       s_roa_start_time)/1e9}s\n"
          "Optimum: 12606")

    draw_path(nodes, s_roa_path)


def greedy_tsp500(*, _max_iterations: int | None = None, _population_size: int | None = None, _blur_coff: int | None = None, _swap_chance: int | None = None, _blur_len: int | None = None, _max_distance_coefficient: int | None = None, _max_speed: int | None = None, _no_improvement_iterations: int | None = None, _shuffle: bool = False) -> None:
    with open("tsp500.txt") as f:
        matrix, nodes = parse(f.read())
    greedy_length, greedy_path = tsp(matrix, nodes[0])
    array_path = np.array(greedy_path)
    # More = better, but takes more time
    max_iterations = _max_iterations or 200
    population_size = _population_size or 400
    # Values between 0 and 1
    blur_coff = _blur_coff or 0.09
    swap_chance = _swap_chance or 0.75
    shuffle_chance = _shuffle or 0.6
    # Max: instance_size//2
    blur_len = _blur_len or 90
    # Not sure what's the max value here
    max_dist_coff = _max_distance_coefficient or 85
    max_speed = _max_speed or 2500

    shuffle = _shuffle != 0 and _shuffle is not None
    max_iterations_without_improvement = _no_improvement_iterations or 60
    s_roa_start_time = monotonic_ns()
    s_roa_path, s_roa_length = S_ROA(nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                                     blur_coefficient=blur_coff, max_distance_coefficient=max_dist_coff, blur_length=blur_len, max_speed=max_speed,
                                     swap_chance=swap_chance, max_iterations_without_improvement=max_iterations_without_improvement, shuffle_instead=shuffle, shuffle_chance=shuffle_chance)
    s_roa_end_time = monotonic_ns()

    print(f"Settings:\n\t{population_size=}\n\t{blur_coff=}\n\t{swap_chance=}\n\t{max_dist_coff=}\n\t{blur_len=}\n\t{max_speed=}\n"
          f"Greedy: {greedy_length}\n"
          f"S-ROA: {s_roa_length} in {(s_roa_end_time -
                                       s_roa_start_time)/1e9}s\n"
          "Optimum: 86789")

    draw_path(nodes, s_roa_path)


def greedy_tsp1000(*, _max_iterations: int | None = None, _population_size: int | None = None, _blur_coff: int | None = None, _swap_chance: int | None = None, _blur_len: int | None = None, _max_distance_coefficient: int | None = None, _max_speed: int | None = None, _no_improvement_iterations: int | None = None, _shuffle: float | None = None) -> None:
    with open("tsp1000.txt") as f:
        matrix, nodes = parse(f.read())
    greedy_length, greedy_path = tsp(matrix, nodes[0])
    array_path = np.array(greedy_path)
    # More = better, but takes more time
    max_iterations = _max_iterations or 150
    population_size = _population_size or 50
    # Values between 0 and 1
    blur_coff = _blur_coff or 0.68
    swap_chance = _swap_chance or 0.75
    shuffle_chance = _shuffle or 1.
    # Max: instance_size//2
    blur_len = _blur_len or 170
    # Not sure what's the max value here
    max_dist_coff = _max_distance_coefficient or 120
    max_speed = _max_speed or 8500
    max_iterations_without_improvement = _no_improvement_iterations or 200

    shuffle = _shuffle is not None
    s_roa_start_time = monotonic_ns()
    s_roa_path, s_roa_length = S_ROA(nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                                     blur_coefficient=blur_coff, max_distance_coefficient=max_dist_coff, blur_length=blur_len, max_speed=max_speed,
                                     swap_chance=swap_chance, max_iterations_without_improvement=max_iterations_without_improvement, shuffle_instead=shuffle, shuffle_chance=shuffle_chance)
    s_roa_end_time = monotonic_ns()

    print(f"Settings:\n\t{population_size=}\n\t{blur_coff=}\n\t{swap_chance=}\n\t{max_dist_coff=}\n\t{blur_len=}\n\t{max_speed=}\n"
          f"Greedy: {greedy_length}\n"
          f"S-ROA: {s_roa_length} in {(s_roa_end_time -
                                       s_roa_start_time)/1e9}s\n"
          "Optimum: 24246")

    draw_path(nodes, s_roa_path)


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    parser = argparse.ArgumentParser()
    parser.add_argument("function", help="Abbreviation of function to run", choices=(
        "rr", "gr", "gb52", "gb127", "gt250", "gt500", "gt1000"))
    parser.add_argument("-i", "--max-iterations", type=int)
    parser.add_argument("-p", "--population-size", type=int)
    parser.add_argument("-b", "--blur-coff", type=float)
    parser.add_argument("-s", "--swap-chance", type=float)
    parser.add_argument("-l", "--blur-len", type=int)
    parser.add_argument("-d", "--max-distance-coefficient", type=int)
    parser.add_argument("-m", "--max-speed", type=int)
    parser.add_argument("-n", "--no-improvement-iterations", type=int)
    parser.add_argument("--shuffle", type=float)
    arguments = parser.parse_args()
    keyword_arguments = dict(("_"+x, y)
                             for x, y in arguments._get_kwargs()[1:])

    if isinstance(arguments.blur_coff, float) and not (0 < arguments.blur_coff < 1):
        raise ValueError(f"Argument blur_coff has to be a number between 0 and 1. It is {
                         arguments.blur_coff}.")
    if isinstance(arguments.swap_chance, float) and not (0 < arguments.swap_chance < 1):
        raise ValueError(f"Argument swap_chance has to be a number between 0 and 1. It is {
                         arguments.swap_chance}.")
    if isinstance(arguments.shuffle, float) and not (0 <= arguments.shuffle <= 1):
        raise ValueError(f"Argument shuffle has to be a number between 0 and 1. It is {
                         arguments.shuffle}.")

    match arguments.function.lower():
        case "rr":
            random_random(**keyword_arguments)

        case "gr":
            greedy_random(**keyword_arguments)

        case "gb52":
            greedy_berlin52(**keyword_arguments)

        case "gb127":
            greedy_bier127(**keyword_arguments)

        case "gt250":
            greedy_tsp250(**keyword_arguments)

        case "gt500":
            greedy_tsp500(**keyword_arguments)

        case "gt1000":
            greedy_tsp1000(**keyword_arguments)

        case _:
            print(f"Unknown abbreviation: {arguments.function.lower()}")
