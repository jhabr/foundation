from __future__ import annotations


class Tensor:
    def __init__(
        self, data: float, _children: tuple = (), _operation: str = ""
    ) -> None:
        self.data = data
        self._children = set(_children)
        self._operation = _operation

    def __repr__(self) -> str:
        return f"Tensor(data={self.data})"

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor(
            data=self.data + other.data, _children=(self, other), _operation="+"
        )

    def __mul__(self, other: Tensor) -> Tensor:
        return Tensor(
            data=self.data * other.data, _children=(self, other), _operation="*"
        )

    @property
    def children(self) -> set:
        return self._children

    @property
    def operation(self) -> str:
        return self._operation
