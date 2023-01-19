import unittest

from src.main.core import Node


class FoundationTest(unittest.TestCase):
    def test_tensor(self):
        a = Node(data=2.0)
        print(a)

        self.assertEqual(2.0, a.data)
        self.assertEqual("Tensor(data=2.0)", str(a))

    def test_add(self):
        a = Node(data=2.0)
        b = Node(data=5.0)

        c = a + b

        self.assertEqual(7.0, c.data)
        self.assertEqual("Tensor(data=7.0)", str(c))

    def test_multiply(self):
        a = Node(data=2.0)
        b = Node(data=-5.0)

        c = a * b

        self.assertEqual(-10.0, c.data)
        self.assertEqual("Tensor(data=-10.0)", str(c))
