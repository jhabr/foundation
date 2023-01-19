import unittest

from src.main.core import Tensor
from src.main.visualisation import draw_dot


class VisualisationTests(unittest.TestCase):

    def test_visualisation(self):
        a = Tensor(data=2.0)
        b = Tensor(data=-3.0)
        c = Tensor(data=10.0)

        d = a * b + c

        self.assertEqual(4.0, d.data)

        draw_dot(d).render(directory='doctest-output', view=True)
