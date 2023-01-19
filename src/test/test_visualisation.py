import unittest

from src.main.core import Node
from src.main.visualisation import draw_dot


class VisualisationTests(unittest.TestCase):

    def test_visualisation(self):
        a = Node(data=2.0)
        b = Node(data=-3.0)
        c = Node(data=10.0)

        d = a * b + c

        self.assertEqual(4.0, d.data)

        dot = draw_dot(d)
        self.assertIsNotNone(dot)

        dot.render(directory='doctest-output', view=True)
