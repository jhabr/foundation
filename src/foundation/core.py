from __future__ import annotations

import math
from typing import Union, Callable

"""
Inspired by https://github.com/karpathy/micrograd/tree/master/micrograd
"""


class Scalar:
    def __init__(
        self,
        data: float,
        children: tuple[Scalar, ...] = (),
        operation: str = "",
        label: str = "",
    ) -> None:
        """
        A differentiable scalar.

        Parameters:
            data: float
                the value of the scalar
            children: tuple[Scalar, ...]
                the children of the scalar
            operation: str
                the operation that was performed on the scalar
            label: str
                the label of the scalar

        Returns:
            None
        """
        self.data = data
        self.grad = 0.0  # derivative of itself with respect to the loss function
        self._backward: Callable = lambda: None
        self.children = set(children)
        self.operation = operation
        self.label = label

    def __repr__(self) -> str:
        """
        Representation of the scalar.

        Returns:
             representation: str
                the string representation of the scalar
        """
        return f"Scalar(data={self.data})"

    def __radd__(self, other: Union[int, float]) -> Scalar:
        """
        Add operation to add a scalar to a number.

        Fallback for addition, otherwise this will not work:

        a = Value(2.0)
        1.0 + a
        => TypeError: unsupported operand type(s) for +: 'float' and 'Value'

        Parameters:
            other: Union[int, float]
                the number to add to the scalar

        Returns:
            out: Scalar
                the result of the addition
        """
        return self + other

    def __add__(self, other: Union[Scalar, int, float]) -> Scalar:
        """
        Add operation to add a scalar to another scalar.

        Parameters:
            other: Union[Scalar, int, float]
                the scalar to add to the scalar

        Returns:
            out: Scalar
                the result of the addition
        """
        other = other if isinstance(other, Scalar) else Scalar(data=other)

        out = Scalar(data=self.data + other.data, children=(self, other), operation="+")

        def _backward():
            """
            local derivative
            """
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other: Union[int, float]) -> Scalar:
        """
        Multiply operation to multiply a scalar with a number.
        Fallback for multiplication, otherwise this will not work:

        a = Value(2.0)
        1.0 + a
        => TypeError: unsupported operand type(s) for *: 'float' and 'Value'

        Parameters:
            other: Union[int, float]
                the number to multiply with the scalar

        Returns:
            out: Scalar
                the result of the multiplication
        """
        return self * other

    def __mul__(self, other: Union[Scalar, int, float]) -> Scalar:
        """
        Multiply operation to multiply a scalar with another scalar.

        Parameters:
            other: Union[Scalar, int, float]
                the scalar to multiply with the scalar

        Returns:
            out: Scalar
                the result of the multiplication
        """
        other = other if isinstance(other, Scalar) else Scalar(data=other)

        out = Scalar(data=self.data * other.data, children=(self, other), operation="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def tanh(self) -> Scalar:
        """
        Hyperbolic tangent operation.

        Returns:
            out: Scalar
                the result of the hyperbolic tangent operation
        """
        tanh = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Scalar(data=tanh, children=(self,), operation="tanh")

        def _backward():
            self.grad += (1 - tanh**2) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> Scalar:
        """
        Rectified linear unit operation.

        Returns:
            out: Scalar
                the result of the rectified linear unit operation
        """
        out = Scalar(
            data=0 if self.data < 0 else self.data, children=(self,), operation="relu"
        )

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out

    def exp(self) -> Scalar:
        """
        Exponential operation.

        Returns:
            out: Scalar
                the result of the exponential operation
        """
        out = Scalar(data=math.exp(self.data), children=(self,), operation="exp")

        def _backward():
            self.grad += out.data * out.grad  # derivative of e^x == e^x

        out._backward = _backward

        return out

    def __truediv__(self, other: Union[Scalar, int, float]) -> Scalar:
        """
        Divide operation to divide a scalar by another scalar.

        Parameters:
            other: Union[Scalar, int, float]
                the scalar to divide the scalar by

        Returns:
            out: Scalar
                the result of the division
        """
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        return self * other**-1

    def __pow__(self, other: Union[Scalar, int, float]) -> Scalar:
        """
        Power operation to raise a scalar to the power of another scalar.

        Parameters:
            other: Union[Scalar, int, float]
                the scalar to raise the scalar to the power of

        Returns:
            out: Scalar
                the result of the power operation
        """
        other = other.data if isinstance(other, Scalar) else other
        out = Scalar(data=self.data**other, children=(self,), operation=f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __neg__(self) -> Scalar:
        """
        Negation operation to negate a scalar.

        Returns:
            out: Scalar
                the result of the negation operation
        """
        return -1 * self

    def __sub__(self, other: Union[Scalar, int, float]) -> Scalar:
        """
        Subtract operation to subtract a scalar from another scalar.

        Parameters:
            other: Union[Scalar, int, float]
                the scalar to subtract from the scalar

        Returns:
            out: Scalar
                the result of the subtraction
        """
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        return self + -other

    def backward(self):
        graph = Graph()
        graph.build_topo(value=self)

        self.grad = 1.0

        for value in reversed(graph.topo):
            value._backward()


Vector = list[Scalar]


class Graph:
    def __init__(self) -> None:
        """
        A topological graph of all differentiable scalars.

        Returns:
            None
        """
        self.topo = []
        self.visited = set()

    def build_topo(self, value: Scalar) -> None:
        """
        Build the topological graph.

        Parameters:
            value: Scalar
                the value to build the graph from

        Returns:
            None
        """
        if value not in self.visited:
            self.visited.add(value)

            for child in value.children:
                self.build_topo(child)
            self.topo.append(value)
