import unittest
from core import Card


class EnvironmentTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_card(self):
        s1 = Card(1, "spade")
        s2 = Card(2, "spade")
        self.assertEqual(Card.sum([s1, s2]), 3)



if __name__ == '__main__':
    unittest.main()
