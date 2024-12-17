import sys
from time import monotonic_ns
import random
import numpy as np
from source.S_ROA.rider_hyenas import S_ROA
from source.mapping import node_path_length
from source.instance_generator import random_instance
from source.tsp_greedy.parser import parse
from source.tsp_greedy.tsp import tsp


def random_random():
    instance_size = 10
    _, nodes = parse(random_instance(
        instance_size, max_x=50, max_y=50))
    path = list(range(1, instance_size))
    random.shuffle(path)
    path.insert(0, 0)
    random_length = node_path_length(path, nodes)
    array_path = np.array(path)
    # More = better, but takes more time
    population_size = 10
    max_iterations = 30
    # Values between 0 and 1
    blur_coff = 0.7
    swap_chance = 0.6
    # Max: instance_size//2
    max_dist_coff = 15
    # Not sure what's the max value here
    blur_len = 5
    max_speed = 600
    s_roa_start_time = monotonic_ns()
    s_roa_path, s_roa_length = S_ROA(nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                                     blur_coefficient=blur_coff, max_distance_coefficient=max_dist_coff, blur_length=blur_len, max_speed=max_speed, swap_chance=swap_chance)
    s_roa_end_time = monotonic_ns()

    print(f"Random: {random_length}\n"
          f"S-ROA: {s_roa_length} in {(s_roa_end_time-s_roa_start_time)/1e9}s")


def greedy_random():
    instance_size = 100
    matrix, nodes = parse(random_instance(
        instance_size, max_x=1000, max_y=1000))
    greedy_length, greedy_path = tsp(matrix, nodes[0])
    array_path = np.array(greedy_path)
    # More = better, but takes more time
    population_size = 200
    max_iterations = 500
    # Values between 0 and 1
    blur_coff = 0.55
    swap_chance = 0.6
    # Max: instance_size//2
    max_dist_coff = 22
    # Not sure what's the max value here
    blur_len = 22
    max_speed = 600
    s_roa_start_time = monotonic_ns()
    s_roa_path, s_roa_length = S_ROA(nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                                     blur_coefficient=blur_coff, max_distance_coefficient=max_dist_coff, blur_length=blur_len, max_speed=max_speed, swap_chance=swap_chance)
    s_roa_end_time = monotonic_ns()

    print(f"Greedy: {greedy_length}\n"
          f"S-ROA: {s_roa_length} in {(s_roa_end_time-s_roa_start_time)/1e9}s")


def greedy_berlin52():
    with open("berlin52.txt") as f:
        matrix, nodes = parse(f.read())
    greedy_length, greedy_path = tsp(matrix, nodes[0])
    array_path = np.array(greedy_path)
    # More = better, but takes more time
    max_iterations = 700
    population_size = 250
    # Values between 0 and 1
    blur_coff = 0.68
    swap_chance = 0.75
    # Max: instance_size//2
    max_dist_coff = 12
    # Not sure what's the max value here
    blur_len = 17
    max_speed = 850
    s_roa_start_time = monotonic_ns()
    s_roa_path, s_roa_length = S_ROA(nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                                     blur_coefficient=blur_coff, max_distance_coefficient=max_dist_coff, blur_length=blur_len, max_speed=max_speed, swap_chance=swap_chance)
    s_roa_end_time = monotonic_ns()

    print(f"Settings:\n\t{population_size=}\n\t{blur_coff=}\n\t{swap_chance=}\n\t{max_dist_coff=}\n\t{blur_len=}\n\t{max_speed=}\n"
          f"Greedy: {greedy_length}\n"
          f"S-ROA: {s_roa_length} in {(s_roa_end_time -
                                       s_roa_start_time)/1e9}s\n"
          "Optimum: 7544")


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    greedy_berlin52()
