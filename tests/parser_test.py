import unittest
import source.tsp_greedy.parser as parser

class TestParser(unittest.TestCase):

    def test_parse(self):
        neighbors = [[(2, 1, 2), (3, 2, 1)], [(1, 1, 1), (3, 2, 1)], [(1, 1, 1), (2, 1, 2)]]
        matrix, _ = parser.parse(
                    "3\n"
                    "1 1 1\n"
                    "2 1 2\n"
                    "3 2 1\n")
        
        for i, line in enumerate(matrix):
            for j, (index, x, y) in enumerate(line):
                self.assertEqual(neighbors[i][j][0], index)
                self.assertEqual(neighbors[i][j][1], x)
                self.assertEqual(neighbors[i][j][2], y)


if __name__ == "__main__":
    unittest.main()
