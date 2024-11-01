import unittest
import source.tsp_greedy.parser as parser

class TestParser(unittest.TestCase):

    def test_parse(self):
        distances = [[(2, 1.0), (3, 1.0)], [(1, 1.0), (3, 1.4142135623730951)], [(1, 1.0), (2, 1.4142135623730951)]]
        for i, line in enumerate(parser.parse(
                    "3\n"
                    "1 1 1\n"
                    "2 1 2\n"
                    "3 2 1\n")):
            for j, (index, distance) in enumerate(line):
                self.assertEqual(distances[i][j][0], index)
                self.assertAlmostEqual(distances[i][j][1], distance)


if __name__ == "__main__":
    unittest.main()
