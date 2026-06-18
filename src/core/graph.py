import networkx as nx
import json # si


def cargar_grafo(ruta: str) -> nx.DiGraph:
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
        
        Grafo=nx.DiGraph()
        
        for nodo in datos["nodos"]:
            Grafo.add_node(nodo["id"])

        for arista in datos["aristas"]:
            Grafo.add_edge(arista["source"], arista["target"])
        
        return Grafo