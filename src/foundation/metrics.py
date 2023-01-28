from src.foundation.core import Scalar, Vector


def mean_squared_error(y_true: list[float], y_preds: list[Vector]) -> Scalar:
    return sum([(y_pred[0] - y_i) ** 2 for y_pred, y_i in zip(y_preds, y_true)])
