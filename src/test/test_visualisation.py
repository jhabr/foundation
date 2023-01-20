import unittest

from src.main.core import Value
from src.main.visualisation import draw_dot


class VisualisationTests(unittest.TestCase):
    def test_visualisation(self):
        a = Value(data=2.0)
        b = Value(data=-3.0)
        c = Value(data=10.0)

        d = a * b + c

        self.assertEqual(4.0, d.data)

        dot = draw_dot(d)
        self.assertIsNotNone(dot)

        dot.render(directory="doctest-output", view=True)
