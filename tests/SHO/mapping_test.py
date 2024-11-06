import unittest
from source.SHO.mapping import get_coordinates

class TestMapping(unittest.TestCase):
    
    def test_get_coordinates(self):
        self.assertEqual(get_coordinates([1,2,3,4,5]), (0,0))


if __name__ == "__main__":
    unittest.main()
