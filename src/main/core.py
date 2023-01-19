from __future__ import annotations


class Node:
    def __init__(
        self, data: float, _children: tuple = (), _operation: str = "", label: str = ""
    ) -> None:
        self.data = data
        self.grad = 0.0  # derivative of itself with respect to the loss function
        self._children = set(_children)
        self._operation = _operation
        self.label = label

    def __repr__(self) -> str:
        return f"Tensor(data={self.data})"

    def __add__(self, other: Node) -> Node:
        return Node(
            data=self.data + other.data, _children=(self, other), _operation="+"
        )

    def __mul__(self, other: Node) -> Node:
        return Node(
            data=self.data * other.data, _children=(self, other), _operation="*"
        )

    @property
    def children(self) -> set:
        return self._children

    @property
    def operation(self) -> str:
        return self._operation
