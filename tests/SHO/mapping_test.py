import unittest
import random
from itertools import permutations
from source.SHO.mapping import get_coordinates, get_path


class TestMapping(unittest.TestCase):
    
    def test_get_coordinates(self):
        self.assertEqual(get_coordinates([1,2,3,4,5]), (0,0))
        self.assertEqual(get_coordinates([1,3,4,5,2]), (3,1))
        self.assertEqual(get_coordinates([1,4,5,3,2]), (5,2))

        space: set[tuple[int, int]] = set()
        for perm in permutations(list(range(2,6))):
            space.add(get_coordinates([1]+list(perm)))
        
        self.assertEqual(space, {(i, j) for i in range(8) for j in range(3)})
    
    def test_get_path(self):
        self.assertEqual(get_path((0,0),5), [1,2,3,4,5])
        self.assertEqual(get_path((3,1),5), [1,3,4,5,2])
        self.assertEqual(get_path((5,2),5), [1,4,5,3,2])
        
    def test_complementarity(self):
        for _ in range(50):
            x,y,n = random.randrange(100, 10000), random.randrange(100, 10000), random.randrange(100,500)
            # print(_,x,y,n)
            self.assertEqual(get_coordinates(get_path((x,y),n)), (x,y))



if __name__ == "__main__":
    unittest.main()
