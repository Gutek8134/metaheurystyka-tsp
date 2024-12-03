import random
from time import sleep
import unittest

import numpy as np

from source.SHO.hyenas import SHO
from source.mapping import node_path_length
from source.instance_generator import random_instance
from source.tsp_greedy.parser import parse
from source.tsp_greedy.tsp import tsp


class TestSHO(unittest.TestCase):

    def test_random_instance_greedy(self):
        print()
        instance_size = 100
        matrix, nodes = parse(random_instance(
            instance_size, max_x=1000, max_y=1000))

        length, path = tsp(matrix, nodes[0])
        sho_path, sho_length = SHO(nodes, np.array(
            path, dtype=np.uint32), 50, 1000, 0.7, 5, blur_length=9)

        self.assertEqual(len(set(sho_path)), instance_size)

        print(flush=True)
        print("Greedy", length, path)
        print("SHO", sho_length, sho_path)

    def test_random_instance_random(self):
        print()
        instance_size = 100
        _, nodes = parse(random_instance(
            instance_size, max_x=1000, max_y=1000))

        path = list(range(1, instance_size))
        random.shuffle(path)
        path.insert(0, 0)
        length = node_path_length(path, nodes)
        sho_path, sho_length = SHO(nodes, np.array(
            path, dtype=np.uint32), 50, 100, 0.7, 5)

        self.assertEqual(len(set(sho_path)), instance_size)

        print("Random", length, path, flush=True)
        print("SHO", sho_length, sho_path, flush=True)


if __name__ == "__main__":
    unittest.main()
