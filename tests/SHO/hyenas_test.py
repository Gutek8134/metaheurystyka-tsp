import unittest

from source.SHO.hyenas import SHO, path_length
from source.SHO.mapping import get_coordinates
from source.tsp_greedy.parser import parse
from source.tsp_greedy.tsp import distance, tsp

class TestSHO(unittest.TestCase):

    def test_burma14(self):
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
        print(length, path)
        sho_path = SHO(nodes, 50, 100, get_coordinates(path))
        sho_length = sum(distance(node, sho_path[i+1]) for i, node in enumerate(sho_path[:-2])) + distance(sho_path[0], sho_path[-1])
        print(sho_length, sho_path)

if __name__ == "__main__":
    unittest.main()
