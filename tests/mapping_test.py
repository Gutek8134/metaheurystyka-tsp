import unittest

import numpy as np
from source.PSO.particles import node_type
import source.mapping as mapping


class TestMapping(unittest.TestCase):

    def test_distance(self):
        self.assertAlmostEqual(mapping.distance(
            np.array((0, 0, 0), dtype=node_type), np.array((1, 3, 4), dtype=node_type)), 5)

    def test_path_length(self):

        self.assertAlmostEqual(
            mapping.path_length(
                np.array((0, 1, 2), dtype=np.uint32),
                np.array([
                    (0, 0, 0),
                    (1, 0, 3),
                    (2, 4, 0)
                ], dtype=node_type)
            ), mapping.node_path_length([0, 1, 2], [
                (0, 0, 0),
                (1, 0, 3),
                (2, 4, 0)
            ]))

    def test_subtract_paths(self):
        path_a = np.array([0, 1, 2, 3, 4], np.uint32)
        path_a_indexes = np.array([0, 1, 2, 3, 4], np.uint32)
        path_b = np.array([1, 2, 0, 4, 3], np.uint32)
        """
        12043
        21043
        01243
        01234
        """
        self.assertEqual([(0, 1), (2, 0), (3, 4)], mapping.subtract_paths(
            path_b, path_a, path_a_indexes))

    def test_mutate_paths(self):
        number_of_cities: int = 7
        for _ in range(20):
            second = np.arange(number_of_cities, dtype=np.uint32)
            np.random.shuffle(second)
            mutation = mapping.mutate(np.arange(number_of_cities, dtype=np.uint32),
                                      second, 0.6, number_of_cities)
            self.assertListEqual(list(np.unique(mutation)),
                                 list(np.arange(number_of_cities)))
            print(mutation, flush=True)


if __name__ == "__main__":
    unittest.main()
