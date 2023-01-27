import unittest

from src.main.core import Scalar
from src.main.visualisation import draw_graph


class VisualisationTests(unittest.TestCase):
    def test_visualisation(self):
        a = Scalar(data=2.0)
        b = Scalar(data=-3.0)
        c = Scalar(data=10.0)

        d = a * b + c

        self.assertEqual(4.0, d.data)

        dot = draw_graph(d)
        self.assertIsNotNone(dot)

        dot.render(directory="doctest-output", view=True)
