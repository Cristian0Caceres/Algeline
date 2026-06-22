import networkx as nx
import json
import numpy as np

def cargar_grafo(ruta: str) -> nx.DiGraph:
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)

    grafo = nx.DiGraph()

    for nodo in datos["nodos"]:
        grafo.add_node(nodo["id"])

    for arista in datos["aristas"]:
        grafo.add_edge(arista["source"], arista["target"])

    return grafo

def obtener_matriz_adyacencia(grafo: nx.DiGraph) -> tuple[np.ndarray, list]:
    nodos = list(grafo.nodes())
    adyacencia = nx.to_numpy_array(grafo, nodelist=nodos)
    return adyacencia, nodos