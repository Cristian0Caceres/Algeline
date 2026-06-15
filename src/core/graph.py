import networkx as nx

# Crear grafo dirigido
G = nx.DiGraph()

# Agregar aristas
G.add_edge("A", "B")
G.add_edge("A", "C")
G.add_edge("B", "C")
G.add_edge("C", "A")
G.add_edge("D", "C")

print("Nodos:", G.nodes())
print("Aristas:", G.edges())