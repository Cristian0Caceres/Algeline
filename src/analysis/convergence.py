import numpy as np

def analizar_convergencia(
    matriz: np.ndarray,
    damping: float = 0.85,
    max_iters: int = 100,
    tolerancia: float = 1e-6
) -> tuple[np.ndarray, list[float], int]:
    """
    Ejecuta el algoritmo PageRank y registra la diferencia L1 en cada iteración.
    
    Retorna:
        - PR: vector de PageRank final (np.ndarray)
        - historial_diferencias: lista de diferencias L1 de cada iteración (list[float])
        - iters: iteraciones hasta convergencia (int)
    """
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("la matriz ha de ser cuadrada.")

    n = matriz.shape[0]
    PR = np.ones(n) / n
    historial_diferencias = []

    for i in range(1, max_iters + 1):
        PR_nuevo = damping * matriz @ PR + (1 - damping) / n
        
        diff = np.linalg.norm(PR_nuevo - PR, 1)
        historial_diferencias.append(float(diff))

        if diff < tolerancia:
            return PR_nuevo, historial_diferencias, i

        PR = PR_nuevo

    return PR, historial_diferencias, max_iters
