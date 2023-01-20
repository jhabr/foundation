from __future__ import annotations

import math


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

    def __add__(self, other: Value) -> Value:
        out = Value(
            data=self.data + other.data, children=(self, other), operation="+"
        )

        def _backward():
            """
            local derivative
            """
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Value) -> Value:
        out = Value(
            data=self.data * other.data, children=(self, other), operation="*"
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def tanh(self) -> Value:
        tanh = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(data=tanh, children=(self,), label="tanh")

        def _backward():
            self.grad += (1 - tanh**2) * out.grad

        out._backward = _backward

        return out

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
