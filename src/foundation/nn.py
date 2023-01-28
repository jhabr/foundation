import random

from src.foundation.core import Scalar
from src.foundation.optimizers import Optimizer

"""
Inspired by https://github.com/karpathy/micrograd/tree/master/micrograd
"""


class Module:

    def parameters(self) -> list[Scalar]:
        return []


class Neuron(Module):
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
        self.w = [Scalar(data=random.uniform(-1, 1)) for _ in range(no_inputs)]
        # bias == over trigger happiness of this neuron
        self.b = Scalar(data=random.uniform(-1, 1))
        self.name = "Tanh-Neuron"

    def __call__(self, x: list[float]) -> Scalar:
        assert len(x) == len(
            self.w
        ), f"input length of x ({len(x)}) must be equal to number of neuron inputs ({len(self.w)})"
        # activation = w * x + b; sum can start at b
        activation = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), start=self.b)
        # pass through non-linearity => tanh
        out = activation.tanh()
        return out

    def __repr__(self) -> str:
        return f"{self.name}({len(self.w)})"

    def parameters(self) -> list[Scalar]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, no_inputs: int, no_outputs: int, name: str = "Dense") -> None:
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

    def __call__(self, x: list[float]) -> list[Scalar]:
        outs = [neuron(x) for neuron in self.neurons]
        return outs

    def __repr__(self) -> str:
        return f"Layer of {len(self.neurons)} {self.neurons[0].name}s"

    def parameters(self) -> list[Scalar]:
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP(Module):
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

    def __call__(self, x: list[float]) -> list[Scalar]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Scalar]:
        return [param for layer in self.layers for param in layer.parameters()]

    def summary(self) -> None:
        print("===== Model Summary =====")

        for index, layer in enumerate(self.layers):
            print(
                f"{index + 1}. {layer.name} {layer}: {len(layer.parameters())} params"
            )

        print("=========================")
        print(f"Total trainable parameters: {len(self.parameters())}")

    def forward(self, x: list[list[float]]) -> list[list[Scalar]]:
        return [self(x_i) for x_i in x]

    def fit(self, x: list[list[float]], y: list[float], optimizer: Optimizer, epochs: int) -> dict:
        """
        Performs training loop - gradient descent

        Parameters
            x: list[list[float]]
                the input values to be fittet
            y: list[float]
                the expected target values (labels)
            optimizer: Optimizer
                the optimizer to be used
            epochs: int
                no of epochs (full x iterations)

        Returns
            history: dict
                the learning history containing loss
        """
        history = {"loss": []}
        optimizer.parameters = self.parameters()

        for i in range(epochs):
            # forward pass
            y_preds = self.forward(x)

            # zero grad
            optimizer.zero_grad()

            # mse loss
            loss = sum([(y_pred[0] - y_i) ** 2 for y_pred, y_i in zip(y_preds, y)])
            history["loss"].append(loss.data)

            print(f"epoch {i} loss: {loss.data}")

            # backward pass
            loss.backward()

            # update of weights and biases
            optimizer.step()

        return history
