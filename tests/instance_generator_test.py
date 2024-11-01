import random
import unittest
import source.instance_generator as generator

class TestInstanceGenerator(unittest.TestCase):
    def test_generator(self):
        min_x = 1
        max_x = 100
        min_y = 1
        max_y = 100
        for size in random.sample(range(100, 10000), 50):
            instance = generator.random_instance(size, min_x=min_x, max_x = max_x, min_y= min_y, max_y=max_y)
            lines = instance.splitlines()
            self.assertEqual(size, int(lines[0]))
            self.assertEqual(size+1, len(lines))

            unique_coordinates: set[tuple[int, int]] = set()
            for line in lines[1:]:
                _, x, y = line.split(" ")
                x = int(x)
                y = int(y)
                
                self.assertLessEqual(x, max_x)
                self.assertGreaterEqual(x, min_x)
                self.assertLessEqual(y, max_y)
                self.assertGreaterEqual(y, min_y)

                self.assertNotIn((x, y), unique_coordinates)
                unique_coordinates.add((x,y))

            self.assertEqual(len(unique_coordinates), size)

        min_x = 1000
        self.assertRaises(ValueError, lambda: generator.random_instance(10000, min_x=min_x, max_x = max_x, min_y= min_y, max_y=max_y))

        min_x = 1
        self.assertRaises(ValueError, lambda: generator.random_instance(100000, min_x=min_x, max_x = max_x, min_y= min_y, max_y=max_y))


        

if __name__ == "__main__":
    unittest.main()
