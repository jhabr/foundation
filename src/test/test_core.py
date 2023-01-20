import unittest

from src.main.core import Value


class FoundationTest(unittest.TestCase):
    def test_value(self):
        a = Value(data=2.0)
        print(a)

        self.assertEqual(2.0, a.data)
        self.assertEqual("Value(data=2.0)", str(a))

    def test_add(self):
        a = Value(data=2.0)
        b = Value(data=5.0)

        c = a + b

        self.assertEqual(7.0, c.data)
        self.assertEqual("Value(data=7.0)", str(c))

    def test_multiply(self):
        a = Value(data=2.0)
        b = Value(data=-5.0)

        c = a * b

        self.assertEqual(-10.0, c.data)
        self.assertEqual("Value(data=-10.0)", str(c))

    def test_tanh(self):
        a = Value(data=0.8814)

        self.assertEqual(0.7071199874301226, a.tanh().data)

    def test_backward(self):
        x1 = Value(2.0, label="x1")
        x2 = Value(0.0, label="x2")

        # weights
        w1 = Value(-3.0, label="w1")
        w2 = Value(1.0, label="w2")

        # bias
        bias = Value(6.8813735870195432, label="b")

        # weighted inputs
        x1w1 = x1 * w1
        x1w1.label = "x1w1"
        x2w2 = x2 * w2
        x2w2.label = "x2w2"
        x1w1x2w2 = x1w1 + x2w2
        x1w1x2w2.label = "x1w1x2w2"

        # neuron
        neuron = x1w1x2w2 + bias
        neuron.label = "n"

        # output
        output = neuron.tanh()
        output.label = "o"

        output.backward()

        self.assertEqual(1.0, output.grad)

    def test_rmul(self):
        a = Value(2.0)
        b = 2.0 * a
        self.assertEqual(4.0, b.data)

        b = a * 2.0
        self.assertEqual(4.0, b.data)

    def test_radd(self):
        a = Value(2.0)
        b = 2.0 + a
        self.assertEqual(4.0, b.data)

        b = a + 2.0
        self.assertEqual(4.0, b.data)

    def test_division(self):
        a = Value(2.0)
        b = Value(4.0)

        self.assertEqual(0.5, (a / b).data)

        b = 4.0

        self.assertEqual(0.5, (a/b).data)
