import contextlib
import io
import random
import unittest
from multiprocessing import Pool
from typing import Any
from time import monotonic_ns
from functools import partial

from numpy import array

from source.instance_generator import random_instance
from source.mapping import node_path_length
from source.tsp_greedy.parser import parse
from source.tsp_greedy.tsp import tsp
from source.PSO.particles import PSO
from source.SHO.hyenas import SHO
from source.ROA.riders import ROA
from source.S_ROA.rider_hyenas import S_ROA


def time_function(func: partial) -> tuple[str, int, Any]:
    start_time = monotonic_ns()
    with contextlib.redirect_stdout(io.StringIO()):
        func_returns = func()[1]
    end_time = monotonic_ns()
    return func.func.__name__, end_time-start_time, func_returns


class CompareTest(unittest.TestCase):
    def test_compare_greedy(self):
        print()
        instance_size = 100
        matrix, nodes = parse(random_instance(
            instance_size, max_x=1000, max_y=1000))
        greedy_start_time = monotonic_ns()
        greedy_length, greedy_path = tsp(matrix, nodes[0])
        greedy_end_time = monotonic_ns()
        array_path = array(greedy_path)

        population_size = 50
        max_iterations = 300
        pso_alpha = 0.62
        pso_beta = 0.55
        sho_blur_coff = 0.4
        # Max: instance_size//2
        sho_max_dist_coff = 32
        sho_blur_len = 45
        roa_max_speed = 600
        roa_swap_chance = 0.6

        heuristics = (partial(PSO, nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations, alpha=pso_alpha, beta=pso_beta),
                      partial(SHO, nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                              blur_coefficient=sho_blur_coff, max_distance_coefficient=sho_max_dist_coff, blur_length=sho_blur_len),
                      partial(ROA, nodes=nodes, initial_path=array_path,
                              number_of_riders=population_size, max_iterations=max_iterations),
                      partial(S_ROA, nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations, blur_coefficient=sho_blur_coff, max_distance_coefficient=sho_max_dist_coff, blur_length=sho_blur_len))

        with Pool(processes=4) as pool:
            results = pool.map(time_function, heuristics)

        results.append(("greedy", greedy_end_time -
                       greedy_start_time, greedy_length))
        print("\n".join(
            [
                " ".join((
                    f"{result[0]}:{" "*(6-len(result[0]))}",
                    f"time={str(result[1]).zfill(15)}ns",
                    f"path length={float(result[2])}"
                ))
                for result in results
            ]))

    def test_compare_random(self):
        print()
        instance_size = 100
        _, nodes = parse(random_instance(
            instance_size, max_x=1000, max_y=1000))
        random_start_time = monotonic_ns()
        random_path = list(range(1, instance_size))
        random.shuffle(random_path)
        random_path.insert(0, 0)
        random_length = node_path_length(random_path, nodes)
        random_end_time = monotonic_ns()
        array_path = array(random_path)

        population_size = 50
        max_iterations = 300
        pso_alpha = 0.62
        pso_beta = 0.55
        sho_blur_coff = 0.4
        # Max: instance_size//2
        sho_max_dist_coff = 32
        sho_blur_len = 45
        roa_max_speed = 600
        roa_swap_chance = 0.6

        heuristics = (partial(PSO, nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations, alpha=pso_alpha, beta=pso_beta),
                      partial(SHO, nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations,
                              blur_coefficient=sho_blur_coff, max_distance_coefficient=sho_max_dist_coff, blur_length=sho_blur_len),
                      partial(ROA, nodes=nodes, initial_path=array_path,
                              number_of_riders=population_size, max_iterations=max_iterations, max_speed=roa_max_speed, swap_chance=roa_swap_chance),
                      partial(S_ROA, nodes=nodes, initial_path=array_path, population_size=population_size, max_iterations=max_iterations, blur_coefficient=sho_blur_coff, max_distance_coefficient=sho_max_dist_coff, blur_length=sho_blur_len, max_speed=roa_max_speed, swap_chance=roa_swap_chance))

        with Pool(processes=4) as pool:
            results = pool.map(time_function, heuristics)

        results.append(("random", random_end_time -
                       random_start_time, random_length))
        print("\n".join(
            [
                "\t".join((
                    f"{result[0]}:{" "*(6-len(result[0]))}",
                    f"time={str(result[1]).zfill(15)}ns",
                    f"path length={float(result[2])}"
                ))
                for result in results
            ]))


if __name__ == "__main__":
    unittest.main()
