import random

from src.main.core import Value


class Neuron:
    def __init__(self, no_inputs: int) -> None:
        """
        The single neuron computation unit

        Parameters:
            no_inputs: int
                number of inputs x into a neuron

        Returns:
            None
        """
        # random initialization of weights for all inputs x_0...x_no_inputs
        self.w = [Value(data=random.uniform(-1, 1)) for _ in range(no_inputs)]
        # bias == over trigger happiness of this neuron
        self.b = Value(data=random.uniform(-1, 1))

    def __call__(self, x: list[float]) -> Value:
        assert len(x) == len(self.w), f"input length of x ({len(x)}) must be equal to number of neuron inputs ({len(self.w)})"
        # activation = w * x + b; sum can start at b
        activation = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), start=self.b)
        # pass through non-linearity => tanh
        out = activation.tanh()
        return out

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, no_inputs: int, no_outputs: int, name: str = "Layer") -> None:
        """
        Layer of neurons.

        Parameters:
            no_inputs: int
                no of inputs x into a neuron
            no_outputs: int
                no of output neurons that this layer produces
            name: str
                the name of the layer
        """
        self.neurons = [Neuron(no_inputs=no_inputs) for _ in range(no_outputs)]
        self.name = name

    def __call__(self, x: list[float]) -> list[Value]:
        outs = [neuron(x) for neuron in self.neurons]
        return outs

    def parameters(self) -> list[Value]:
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP:
    def __init__(self, no_inputs: int, no_layer_outputs: list[int]) -> None:
        """
        Multi-layer perceptron

        Parameters:
            no_inputs: int
                no of inputs x into a neuron, e.g.:
                    x = [1.0, 2.0, 3.0]
                    no_inputs = 3 = len(x)
            no_layer_outputs: list[int]
                list of no of neurons per layer, e.g.
                    no_layer_outputs = [4, 4, 1]
                    => 2 hidden layers with 4 neurons each, 1 output layer with one neuron
        """
        sizes = [no_inputs] + no_layer_outputs  # e.g. [3, 4, 4, 1]
        self.layers = [
            Layer(no_inputs=sizes[i], no_outputs=sizes[i + 1])
            for i in range(len(no_layer_outputs))
        ]

    def __call__(self, x: list[float]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [param for layer in self.layers for param in layer.parameters()]

    def summary(self) -> None:
        print("===== Model Summary =====")

        for index, layer in enumerate(self.layers):
            print(f"{index + 1}. layer: {layer}: {len(layer.parameters())} params")

        print("=========================")
        print(f"Total parameters: {len(self.parameters())}")
