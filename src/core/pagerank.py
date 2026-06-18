import numpy as np

def pagerank(
    matriz: np.ndarray,
    iteraciones: int = 100,
    damping: float = 0.85,
    tolerancia: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, int]:
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("la matriz ha de ser cuadrada.")

    n = matriz.shape[0]
    PR = np.ones(n) / n

    for i in range(1, iteraciones + 1):
        PR_nuevo = damping * matriz @ PR + (1 - damping) / n

        if verbose:
            print(f"Iteración {i}: {PR_nuevo}")

        if np.linalg.norm(PR_nuevo - PR, 1) < tolerancia:
            return PR_nuevo, i

        PR = PR_nuevo

    return PR, iteraciones