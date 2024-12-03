import random
from time import sleep
import unittest

import numpy as np

from source.S_ROA.rider_hyenas import S_ROA
from source.mapping import node_path_length
from source.instance_generator import random_instance
from source.tsp_greedy.parser import parse
from source.tsp_greedy.tsp import tsp


class TestS_ROA(unittest.TestCase):

    def test_random_instance_greedy(self):
        print()
        instance_size = 100
        matrix, nodes = parse(random_instance(
            instance_size, max_x=1000, max_y=1000))

        length, path = tsp(matrix, nodes[0])
        s_roa_path, s_roa_length = S_ROA(nodes, np.array(
            path, dtype=np.uint32), 50, 1000, 0.7, 5, blur_length=9)

        self.assertEqual(len(set(s_roa_path)), instance_size)

        print(flush=True)
        print("Greedy", length, path)
        print("S_ROA", s_roa_length, s_roa_path)

    def test_random_instance_random(self):
        print()
        instance_size = 100
        _, nodes = parse(random_instance(
            instance_size, max_x=1000, max_y=1000))

        path = list(range(1, instance_size))
        random.shuffle(path)
        path.insert(0, 0)
        length = node_path_length(path, nodes)
        s_roa_path, s_roa_length = S_ROA(nodes, np.array(
            path, dtype=np.uint32), 50, 100, 0.7, 5)

        self.assertEqual(len(set(s_roa_path)), instance_size)

        print("Random", length, path, flush=True)
        print("S_ROA", s_roa_length, s_roa_path, flush=True)


if __name__ == "__main__":
    unittest.main()
