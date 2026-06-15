import networkx as nx
import numpy as np
import pagerank as pr


G = nx.DiGraph()

G.add_edge("A", "B")
G.add_edge("A", "C")
G.add_edge("B", "C")
G.add_edge("C", "A")

def crearMatrix(Grafo):
    nodos = list(Grafo.nodes())
    n = len(nodos)

    M = np.zeros((n, n))

    for j, origen in enumerate(nodos):

        salidas = Grafo.out_degree(origen)

        for destino in Grafo.neighbors(origen):

            i = nodos.index(destino)

            M[i][j] = 1 / salidas
    print(M)
    return M

#testeo todo feito todo horrible
matriz = crearMatrix(G)
pr.Pagerank(matriz)
