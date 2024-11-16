import random
from time import sleep
import unittest

import numpy as np

from source.PSO.particles import PSO, path_length as PSO_path_length, node_type
from source.SHO.hyenas import path_length
from source.instance_generator import random_instance
from source.tsp_greedy.parser import parse
from source.tsp_greedy.tsp import tsp


class TestPSO(unittest.TestCase):

    def test_burma14_greedy(self):
        print()
        matrix, nodes = parse("""14
1 16.47 96.10
2 16.47 94.44
3 20.09 92.54
4 22.39 93.37
5 25.23 97.24
6 22.00 96.05
7 20.47 97.02
8 17.20 96.29
9 16.30 97.38
10 14.05 98.12
11 16.53 97.38
12 21.52 95.59
13 19.41 97.13
14 20.09 94.55""")

        length, path = tsp(matrix, nodes[0])
        pso_path, pso_length = PSO(nodes, np.array(
            path, dtype=np.uint32), 30, 100, 0.4, 0.7)

        print("Greedy", length, path)
        print("PSO", pso_length, pso_path)

    def test_burma14_random(self):
        print()
        _, nodes = parse("""14
1 16.47 96.10
2 16.47 94.44
3 20.09 92.54
4 22.39 93.37
5 25.23 97.24
6 22.00 96.05
7 20.47 97.02
8 17.20 96.29
9 16.30 97.38
10 14.05 98.12
11 16.53 97.38
12 21.52 95.59
13 19.41 97.13
14 20.09 94.55""")

        path = list(range(1, 14))
        random.shuffle(path)
        path.insert(0, 0)
        length = path_length(path, nodes)
        pso_path, pso_length = PSO(nodes, np.array(
            path, dtype=np.uint32), 50, 1000, 0.4, 0.7)

        print("Random", length, path, flush=True)
        print("PSO", pso_length, pso_path, flush=True)

    def test_random_instance_greedy(self):
        print()
        matrix, nodes = parse(random_instance(100, max_x=1000, max_y=1000))

        length, path = tsp(matrix, nodes[0])
        pso_path, pso_length = PSO(nodes, np.array(
            path, dtype=np.uint32), 50, 1000, 0.7, 0.9)

        print(flush=True)
        print("Greedy", length, path)
        print("PSO", pso_length, pso_path)

    def test_random_instance_random(self):
        print()
        _, nodes = parse(random_instance(100, max_x=1000, max_y=1000))

        path = list(range(1, 100))
        random.shuffle(path)
        path.insert(0, 0)
        length = path_length(path, nodes)
        pso_path, pso_length = PSO(nodes, np.array(
            path, dtype=np.uint32), 50, 100, 0.4, 0.7)

        print("Random", length, path, flush=True)
        print("PSO", pso_length, pso_path, flush=True)


if __name__ == "__main__":
    unittest.main()
