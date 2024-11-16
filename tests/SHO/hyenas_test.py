import unittest

from source.SHO.hyenas import SHO, path_length
from source.SHO.mapping import get_coordinates
from source.tsp_greedy.parser import parse
from source.tsp_greedy.tsp import distance, tsp


class TestSHO(unittest.TestCase):

    def test_burma14(self):
        return
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
        sho_path = SHO(nodes, 500, 1000, get_coordinates(path))
        sho_length = path_length([x[0] for x in sho_path], nodes)
        print(get_coordinates(path) == get_coordinates(
            [x[0] for x in sho_path]))
        print(path == [x[0] for x in sho_path])
        print(length, path)
        print(sho_length, sho_path)

    def test_path_length(self):
        self.assertAlmostEqual(path_length([0, 1, 2], [(0, 0, 0),
                                                       (1, 0, 3),
                                                       (2, 4, 0)]), 12)


if __name__ == "__main__":
    unittest.main()
