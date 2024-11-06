import unittest
import source.tsp_greedy.tsp as tsp
import random


def random_nodes(m: int, ordered_indexes: bool = False) -> list[tsp.node]:
    """
    :param m: Keep it within 2^21
    :ordered_indexes: If set to false, indexes will be randomly chosen within range -2^20, 2^20, otherwise natural numbers from 0 to m
    :return: List of length m containing random tsp.neighbors with unique indexes
    :raises ValueError: On negative m or m greater than 2^21
    """
    if m <= 0 or m >= 1 << 21:
        raise ValueError("m must be in range [0, 2^21]")

    return [(random_index, random.randrange(-(1 << 10), 1 << 10), random.randrange(-(1 << 10), 1 << 10))
            for random_index in (range(m) if ordered_indexes
                                 else random.sample(range(-(1 << 20), 1 << 20), m))]


class TestTSP(unittest.TestCase):

    def test_tsp(self) -> None:
        matrix: tsp.Matrix = \
            [
                [(1, 2, 3), (2, 2, 7), (3, 3, 3)],
                [(0, 0, 0), (2, 2, 7), (3, 3, 3)],
                [(0, 0, 0), (1, 2, 3), (3, 3, 3)],
                [(0, 0, 0), (1, 2, 3), (2, 2, 7)],
            ]
        nodes: list[tsp.node] = [(0, 0, 0), (1, 2, 3), (2, 2, 7), (3, 3, 3)]

        for index, neighbors in enumerate(matrix):
            matrix[index] = list(
                sorted(neighbors, key=lambda x: tsp.distance(x,nodes[index])))

        distance, path = tsp.tsp(matrix, (0, 0, 0))
        self.assertAlmostEqual(distance, 16.008766790362)
        self.assertEqual(path, [0, 1, 3, 2, 0])

    def test_find_next_destination(self) -> None:
        for _ in range(100):
            nodes: list[tsp.node] = random_nodes(1000, True)
            next_valid_index: int = random.randint(600, 995)
            visited: set[tsp.index] = set(
                range(0, next_valid_index))

            with self.subTest(neighbors=nodes, visited=visited):
                index, x, y = tsp.find_next_destination(nodes, visited)
                self.assertEqual(index, next_valid_index)
                self.assertEqual(x, nodes[next_valid_index][1])
                self.assertEqual(y, nodes[next_valid_index][2])

            with self.assertRaises(ValueError):
                nodes: list[tsp.node] = random_nodes(1000, True)
                visited: set[tsp.index] = set(range(0, 1000))
                index, x, y = tsp.find_next_destination(nodes, visited)

    def test_distance(self) -> None:
        self.assertAlmostEqual(tsp.distance((0,0,0), (1,3,4)), 5)


if __name__ == "__main__":
    unittest.main(verbosity=1)
