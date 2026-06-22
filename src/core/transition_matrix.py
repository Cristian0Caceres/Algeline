import networkx as nx
import numpy as np
try:
    from graph import obtener_matriz_adyacencia
except (ImportError, ModuleNotFoundError):
    from src.core.graph import obtener_matriz_adyacencia


def crear_matriz(grafo: nx.DiGraph) -> tuple[np.ndarray, list]:
    W_adj, nodos = obtener_matriz_adyacencia(grafo)
    n = len(nodos)
    
    # En NetworkX, W_adj[i, j] = 1 si hay arista de i a j.
    # Para PageRank, necesitamos W[i, j] = 1 si hay arista de j a i.
    # Por tanto, transponemos para obtener la matriz con las orientaciones correctas.
    W = W_adj.T
    
    grados_salida = W.sum(axis=0)
    
    # Inicializamos M
    M = np.zeros((n, n))
    
    # Máscara para nodos sin salida (dangling nodes)
    dangling_mask = (grados_salida == 0)
    
    # Asignamos 1/n a las columnas correspondientes a nodos dangling
    M[:, dangling_mask] = 1 / n
    
    # Normalizamos el resto de columnas por su grado de salida
    non_dangling_mask = ~dangling_mask
    if np.any(non_dangling_mask):
        M[:, non_dangling_mask] = W[:, non_dangling_mask] / grados_salida[non_dangling_mask]
        
    return M, nodos