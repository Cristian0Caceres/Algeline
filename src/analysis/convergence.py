import numpy as np


def error_l1(actual: np.ndarray, anterior: np.ndarray) -> float:
    """
    Calcula la distancia L1 entre dos vectores PageRank.
    """
    return np.linalg.norm(actual - anterior, 1)


def convergio(
    actual: np.ndarray,
    anterior: np.ndarray,
    tolerancia: float = 1e-6
) -> bool:
    """
    Determina si el algoritmo alcanzó convergencia.
    """
    return error_l1(actual, anterior) < tolerancia
