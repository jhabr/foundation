import unittest

from src.foundation.nn import Neuron, Layer, MLP
from src.foundation.optimizers import SGD


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

        self.assertEqual(2 + 1, len(n.parameters()))  # 2 weights, 1 bias

        self.assertLessEqual("Tanh-Neuron(2)", str(n))

    def test_layer(self):
        layer = Layer(no_inputs=2, no_outputs=2)
        x = [2.0, 3.0]
        outs = layer(x)
        self.assertEqual(2, len(outs))

        self.assertEqual(4 + 2, len(layer.parameters()))  # 4 weights, 2 biases

    def test_mlp(self):
        mlp = MLP(no_inputs=2, no_layer_outputs=[4, 4, 1])
        x = [2.0, 3.0]
        out = mlp(x)
        self.assertIsNotNone(3, len(mlp.layers))
        self.assertIsNotNone(out)

        self.assertEqual(37, len(mlp.parameters()))

        mlp.summary()

    def test_mlp_fit(self):
        xs = [[1.0, 4.0, -1.0], [2.0, -2.0, 0.5], [0.5, 1.0, 3.0], [3.0, 1.0, -1.0]]

        # labels aka desired targets
        ys = [1.0, -1.0, -1.0, 1.0]

        model = MLP(no_inputs=3, no_layer_outputs=[4, 4, 1])
        model.summary()

        optimizer = SGD(learning_rate=0.1)

        history = model.fit(x=xs, y=ys, optimizer=optimizer, epochs=200)
        self.assertIsNotNone(history["loss"])

        predictions = [model(x) for x in xs]
        print(predictions)
        self.assertEqual(4, len(predictions))
