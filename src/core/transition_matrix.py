import networkx as nx
import numpy as np

def crear_matriz(grafo: nx.DiGraph) -> tuple[np.ndarray, list]:
    nodos = list(grafo.nodes())
    n = len(nodos)
    indice = {nodo:i for i, nodo in enumerate(nodos)}
    
    
    M = np.zeroes((n, n))
    
    
    for j, origen in enumerate(nodos):
        salidas = grafo.out_degree(origen)
        
        if salidas == 0:
            M[:, j] = 1/n
            
        else:
            for destiono in grafo.neighbors(origen):
                i = indice[destino]
                M[i][j] = 1 / salidas
                
    return M , nodos
