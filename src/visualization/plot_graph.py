import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def dibujar_red_en_eje(grafo: nx.DiGraph, ax: plt.Axes, ranking_dict: dict = None):
    """
    Dibuja el grafo en un eje de Matplotlib proporcionado.
    Los nodos se dimensionan y colorean de acuerdo al ranking de PageRank si se proporciona.
    """
    ax.clear()
    ax.axis("off")
    
    # Layout consistente
    pos = nx.spring_layout(grafo, seed=42)
    
    if ranking_dict:
        nodos_ordenados = list(grafo.nodes())
        sizes = [300 + 4000 * ranking_dict[node] for node in nodos_ordenados]
        colors = [ranking_dict[node] for node in nodos_ordenados]
        cmap = plt.cm.plasma
    else:
        sizes = 500
        colors = "skyblue"
        cmap = None

    nx.draw_networkx_nodes(
        grafo, pos, 
        ax=ax, 
        node_size=sizes, 
        node_color=colors, 
        cmap=cmap,
        edgecolors="black", 
        linewidths=1
    )

    nx.draw_networkx_edges(
        grafo, pos, 
        ax=ax, 
        edge_color="gray", 
        width=1.2, 
        arrows=True, 
        arrowstyle="-|>", 
        arrowsize=15,
        node_size=sizes
    )

    nx.draw_networkx_labels(
        grafo, pos, 
        ax=ax, 
        font_size=9, 
        font_color="black", 
        font_family="sans-serif", 
        font_weight="bold"
    )

    if ranking_dict:
        ax.set_title("Red ponderada por PageRank", fontsize=11, fontweight="bold", pad=8)
    else:
        ax.set_title("Estructura de la Red", fontsize=11, fontweight="bold", pad=8)

def exportar_grafico_red(grafo: nx.DiGraph, ruta_salida: str, ranking_dict: dict = None):
    """
    Genera y guarda el gráfico del grafo en disco.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    dibujar_red_en_eje(grafo, ax, ranking_dict)
    fig.tight_layout()
    fig.savefig(ruta_salida, format=ruta_salida.split('.')[-1], dpi=300)
    plt.close(fig)
