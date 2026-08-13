import unittest

from pipeline.parse_pascal import choose_random_split


class TestChooseRandomSplit(unittest.TestCase):
    def test_boundaries(self):
        data = list(range(20))
        train, val = choose_random_split(data, 0)
        self.assertEqual(val, [])
        self.assertEqual(train, data)

        train, val = choose_random_split(data, 100)
        self.assertEqual(train, [])
        self.assertEqual(val, data)

    def test_out_of_range_raises(self):
        data = list(range(20))
        with self.assertRaises(ValueError):
            choose_random_split(data, -1)
        with self.assertRaises(ValueError):
            choose_random_split(data, 101)


