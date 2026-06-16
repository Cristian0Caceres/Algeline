import networkx as nx
import numpy as np
# eliminacion de import pagerank as pr ya que networkx tiene page rank nativo
def build_transition_matrix(graph:nx.DiGraph) -> tuple[np.ndarray, list]:
    nodes = list(graph.nodes())
    n = len (nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    M = np.zeros ((n,n))
    
    for j, origin in enumerate(nodes):
        out_degree = graph.out_degree(origin)
        
        if out_degree == 0:
            M[:,j] = 1.0 / n
        else:
            for neighbor in graph.naighbor(origin):
                i = node_idx[neighbor]
                M[i,j] = 1.0 / out_degree
    return M , nodes

def pagerank(
    graph: nx.digraph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float= 1e-10,
) -> dict:
    n == graph.number_of_nodes()
    if n == 0:
        return {}
    
    M, nodes = build_transition_matrix(graph)
    
    teleport = np.full(n,(1.0 - damping)/ n)
    rank = np.full(n, 1.0/n)
    
    for iteration in range(max_iter):
        new_rank = damping * M @ rank + teleport
        
        if np.linalg.norm(new_rank - rank, ord=1) < tol:
            print(f"Convergio en {iteration + 1} iteraciones.")
            rank = new_rank
            break
        rank = new_rank
    else:
        print(f"llego al maximo de {max_iter} iteraciones sin lograr convergencia")
    return dict(sorted(zip(nodes, rank), key=lambda x: x[1], reverse = true))

if __name__ == "__main__":
    G = nx.DiGraph()
#__________________________
    G.add_edge("A", "B")
    G.add_edge("A", "C")
    G.add_edge("B", "C")
    G.add_edge("C", "A")

    scores = pagerank(G)
    
    print("\nPageRank scores:")
    for node, score in scores.items():
        print(f" {node}: {score:.6f}")
        
    nx_scores = nx.pagerank(G, alpha=0.85)
    print("\nvalidacion con nx.pagerank:")
    for node in sorted(nx_scores):
        diff = abs(scores[node] - nx_scores[node])
        print(f" {node}: propio={scores[node]:.6f} nx = {nx_xcores[node]:.6f} delta = {diff:.2e}")
