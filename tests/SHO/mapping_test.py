import unittest
from itertools import permutations
from source.SHO.mapping import get_coordinates


class TestMapping(unittest.TestCase):
    
    def test_get_coordinates(self):
        self.assertEqual(get_coordinates([1,2,3,4,5]), (0,0))
        self.assertEqual(get_coordinates([1,3,4,5,2]), (3,1))
        self.assertEqual(get_coordinates([1,4,5,3,2]), (5,2))

        space: set[int] = set()
        for perm in permutations(list(range(2,6))):
            space.add(get_coordinates(tuple([1]+list(perm))))
        
        self.assertEqual(space, {(i, j) for i in range(8) for j in range(3)})


if __name__ == "__main__":
    unittest.main()
