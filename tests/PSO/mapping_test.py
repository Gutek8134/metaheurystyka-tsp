import unittest

import numpy as np
from source.PSO.particles import node_type
import source.PSO.mapping as mapping
import source.SHO.hyenas as hyenas


class TestMapping(unittest.TestCase):

    def test_distance(self):
        self.assertAlmostEqual(mapping.distance(
            np.array((0, 0, 0), dtype=node_type), np.array((1, 3, 4), dtype=node_type)), 5)

    def test_path_length(self):

        self.assertAlmostEqual(
            mapping.path_length(
                np.array([
                    (0, 0, 0),
                    (1, 0, 3),
                    (2, 4, 0)
                ], dtype=node_type),
                np.array((0, 1, 2), dtype=np.uint32)
            ), hyenas.path_length([0, 1, 2], [
                (0, 0, 0),
                (1, 0, 3),
                (2, 4, 0)
            ]))


if __name__ == "__main__":
    unittest.main()
