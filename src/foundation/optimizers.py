from src.foundation.core import Scalar


class Optimizer:

    parameters: list[Scalar]

    def __init__(self, learning_rate: float = 0.001) -> None:
        """
        Base class for all optimizers.

        Parameters:
            learning_rate: float
                the learning rate aka step size
        """
        self.lr = learning_rate

    def zero_grad(self) -> None:
        """
        Sets the gradients of all parameters to zero.

        Returns:
            None
        """
        for param in self.parameters:
            param.grad = 0.0

    def step(self) -> None:
        """
        Performs a single optimization step.

        Returns:
            None
        """
        raise NotImplementedError


class SGD(Optimizer):
    def step(self) -> None:
        """
        Performs a single optimization step.

        Returns:
            None
        """
        for param in self.parameters:
            # modify the gradient by a small step size in the direction of the gradient
            param.data += -1 * self.lr * param.grad
