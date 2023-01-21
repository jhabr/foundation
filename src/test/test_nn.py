import unittest

from src.main.core import Value
from src.main.nn import Neuron, Layer, MLP


class NNTests(unittest.TestCase):
    def test_neuron(self):
        n = Neuron(no_inputs=2)

        self.assertIsNotNone(n.w)
        self.assertEqual(list, type(n.w))
        self.assertEqual(2, len(n.w))

        self.assertIsNotNone(n.b)

        x = [2.0, 3.0]
        out = n(x)
        self.assertIsNotNone(out.data)
        self.assertLessEqual(out.data, 1)
        self.assertGreaterEqual(out.data, -1)

    def test_layer(self):
        layer = Layer(no_inputs=2, no_outputs=2)
        x = [2.0, 3.0]
        outs = layer(x)
        self.assertEqual(2, len(outs))

    def test_mlp(self):
        mlp = MLP(no_inputs=2, no_layer_outputs=[4, 4, 1])
        x = [2.0, 3.0]
        out = mlp(x)
        self.assertIsNotNone(3, len(mlp.layers))
        self.assertIsNotNone(out)
