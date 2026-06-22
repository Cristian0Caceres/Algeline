import numpy as np


def error_l1(actual: np.ndarray, anterior: np.ndarray) -> float:
    return np.linalg.norm(actual - anterior, 1)


def convergio(
    actual: np.ndarray,
    anterior: np.ndarray,
    tolerancia: float = 1e-6
) -> bool:
    return error_l1(actual, anterior) < tolerancia