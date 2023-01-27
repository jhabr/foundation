import unittest

from src.main.core import Scalar


class FoundationTest(unittest.TestCase):
    def test_value(self):
        a = Scalar(data=2.0)
        print(a)

        self.assertEqual(2.0, a.data)
        self.assertEqual("Scalar(data=2.0)", str(a))

    def test_add(self):
        a = Scalar(data=2.0)
        b = Scalar(data=5.0)

        c = a + b

        self.assertEqual(7.0, c.data)
        self.assertEqual("Scalar(data=7.0)", str(c))

    def test_multiply(self):
        a = Scalar(data=2.0)
        b = Scalar(data=-5.0)

        c = a * b

        self.assertEqual(-10.0, c.data)
        self.assertEqual("Scalar(data=-10.0)", str(c))

    def test_tanh(self):
        a = Scalar(data=0.8814)

        self.assertEqual(0.7071199874301226, a.tanh().data)

    def test_relu(self):
        a = Scalar(data=12)
        b = Scalar(data=-12)

        self.assertEqual(12, a.relu().data)
        self.assertEqual(0, b.relu().data)

    def test_backward(self):
        x1 = Scalar(2.0, label="x1")
        x2 = Scalar(0.0, label="x2")

        # weights
        w1 = Scalar(-3.0, label="w1")
        w2 = Scalar(1.0, label="w2")

        # bias
        bias = Scalar(6.8813735870195432, label="b")

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
        a = Scalar(2.0)

        b = 2.0 * a
        self.assertEqual(4.0, b.data)

        b = a * 2.0
        self.assertEqual(4.0, b.data)

    def test_radd(self):
        a = Scalar(2.0)

        b = 2.0 + a
        self.assertEqual(4.0, b.data)

        b = a + 2.0
        self.assertEqual(4.0, b.data)

    def test_division(self):
        a = Scalar(2.0)

        b = Scalar(4.0)
        self.assertEqual(0.5, (a / b).data)

        b = 4.0
        self.assertEqual(0.5, (a / b).data)

        b = 4
        self.assertEqual(0.5, (a / b).data)

    def test_power(self):
        a = Scalar(2.0)
        power = Scalar(4.0)

        self.assertEqual(16.0, (a**power).data)

        power = 4.0
        self.assertEqual(16.0, (a**power).data)

        power = 4
        self.assertEqual(16.0, (a**power).data)

    def test_subtract(self):
        a = Scalar(2.0)
        b = Scalar(4.0)

        self.assertEqual(-2.0, (a - b).data)

        b = 4.0
        self.assertEqual(-2, (a - b).data)

        b = 4
        self.assertEqual(-2, (a - b).data)
