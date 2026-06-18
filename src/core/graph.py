import networkx as nx
import json # si

# Crear grafo dirigido
Grafo = nx.DiGraph()


#Cargar json
with open("datos.json") as a:
    datos = json.load(a)

for nodo in datos["nodos"]:
    Grafo.add_node(nodo["id"])

for arista in datos["aristas"]:
    Grafo.add_edge(arista["source"],arista["target"])

print(Grafo.nodes())
print(Grafo.edges())

