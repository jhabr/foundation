from __future__ import annotations

import math
from typing import Union


class Value:
    def __init__(
        self, data: float, children: tuple = (), operation: str = "", label: str = ""
    ) -> None:
        self.data = data
        self.grad = 0.0  # derivative of itself with respect to the loss function
        self._backward = lambda: None
        self.children = set(children)
        self.operation = operation
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __radd__(self, other: Union[int, float]) -> Value:
        """
        Fallback for addition, otherwise this will not work:

        a = Value(2.0)
        1.0 + a
        => TypeError: unsupported operand type(s) for +: 'float' and 'Value'
        """
        return self * other

    def __add__(self, other: Union[Value, int, float]) -> Value:
        other = other if isinstance(other, Value) else Value(data=other)

        out = Value(data=self.data + other.data, children=(self, other), operation="+")

        def _backward():
            """
            local derivative
            """
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other: Union[int, float]) -> Value:
        """
        Fallback for multiplication, otherwise this will not work:

        a = Value(2.0)
        1.0 + a
        => TypeError: unsupported operand type(s) for *: 'float' and 'Value'
        """
        return self * other

    def __mul__(self, other: Union[Value, int, float]) -> Value:
        other = other if isinstance(other, Value) else Value(data=other)

        out = Value(data=self.data * other.data, children=(self, other), operation="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def tanh(self) -> Value:
        tanh = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(data=tanh, children=(self,), operation="tanh")

        def _backward():
            self.grad += (1 - tanh**2) * out.grad
        out._backward = _backward

        return out

    def exp(self) -> Value:
        out = Value(data=math.exp(self.data), children=(self,), operation="exp")

        def _backward():
            self.grad += out.data * out.grad  # derivative of e^x == e^x
        out._backward = _backward

        return out

    def __truediv__(self, other: Union[Value, int, float]) -> Value:
        other = other if isinstance(other, Value) else Value(data=other)
        return self * other ** -1

    def __pow__(self, other: Union[Value, int, float]) -> Value:
        other = other.data if isinstance(other, Value) else other
        out = Value(data=self.data ** other, children=(self,), operation=f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self) -> Value:
        return -1 * self

    def __sub__(self, other: Union[Value, int, float]) -> Value:
        other = other if isinstance(other, Value) else Value(data=other)
        return self + -other

    def backward(self):
        graph = Graph()
        graph.build_topo(value=self)

        self.grad = 1.0

        for value in reversed(graph.topo):
            value._backward()


class Graph:
    def __init__(self) -> None:
        self.topo = []
        self.visited = set()

    def build_topo(self, value: Value) -> None:
        if value not in self.visited:
            self.visited.add(value)

            for child in value.children:
                self.build_topo(child)
            self.topo.append(value)
