import unittest
import source.tsp_greedy.tsp as tsp
import random


def random_neighbors(m: int, ordered_indexes: bool = False) -> list[tsp.neighbor]:
    """
    :param m: Keep it within 2^21
    :ordered_indexes: If set to false, indexes will be randomly chosen within range -2^20, 2^20, otherwise natural numbers from 0 to m
    :return: List of length m containing random tsp.neighbors with unique indexes
    :raises ValueError: On negative m or m greater than 2^21
    """
    if m <= 0 or m >= 1 << 21:
        raise ValueError("m must be in range [0, 2^21]")

    return [(random_index, random.uniform(-(1 << 10), 1 << 10))
            for random_index in (range(m) if ordered_indexes
                                 else random.sample(range(-(1 << 20), 1 << 20), m))]


class TestTSP(unittest.TestCase):

    def test_tsp(self) -> None:
        matrix: tsp.Matrix = \
            [
                [(1, 8), (2, 7), (3, 3)],
                [(0, 8), (2, 2), (3, 3)],
                [(0, 7), (1, 2), (3, 3)],
                [(0, 3), (1, 3), (2, 3)],
            ]

        matrix = [list(sorted(neighbors, key=lambda x: x[1]))
                  for neighbors in matrix]

        distance, path = tsp.tsp(matrix, 0)
        self.assertAlmostEqual(distance, 15)
        self.assertEqual(path, [0, 3, 1, 2, 0])

    def test_find_next_destination(self) -> None:
        for _ in range(100):
            neighbors: list[tsp.neighbor] = random_neighbors(1000, True)
            next_valid_index: int = random.randint(600, 995)
            visited: set[tsp.index] = set(
                range(0, next_valid_index))
            with self.subTest(neighbors=neighbors, visited=visited):
                index, distance = tsp.find_next_destination(neighbors, visited)
                self.assertEqual(index, next_valid_index)
                self.assertAlmostEqual(
                    distance, neighbors[next_valid_index][1])

            with self.assertRaises(ValueError):
                neighbors: list[tsp.neighbor] = random_neighbors(1000, True)
                visited: set[tsp.index] = set(range(0, 1000))
                index, distance = tsp.find_next_destination(neighbors, visited)

    def test_find_by_index(self) -> None:
        for _ in range(100):
            neighbors: list[tsp.neighbor] = random_neighbors(1000)
            to_find: tsp.neighbor = random.choice(neighbors)
            with self.subTest(neighbors=neighbors, to_find=to_find):
                index, distance = tsp.find_by_index(neighbors, to_find[0])
                self.assertEqual(index, to_find[0])
                self.assertAlmostEqual(distance, to_find[1])

        neighbors: list[tsp.neighbor] = random_neighbors(1000)

        with self.assertRaises(ValueError):
            tsp.find_by_index(neighbors, 1 << 22)


if __name__ == "__main__":
    unittest.main(verbosity=1)
