import matplotlib.pyplot as plt

def dibujar_rankings_en_eje(rankings_ordenados: list[tuple], ax: plt.Axes):
    """
    Dibuja un gráfico de barras horizontales con el ranking de nodos.
    """
    ax.clear()
    if not rankings_ordenados:
        ax.text(0.5, 0.5, "Sin datos de ranking", ha='center', va='center', color='gray')
        return
        
    nodos, scores = zip(*rankings_ordenados)
    
    # Invertir para que el nodo más importante aparezca en la parte superior
    nodos = list(nodos)[::-1]
    scores = list(scores)[::-1]
    
    y_pos = range(len(nodos))
    
    # Dibujar barras horizontales
    bars = ax.barh(y_pos, scores, color="skyblue", edgecolor="black", height=0.6)
    
    # Configurar etiquetas y límites
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nodos, fontweight="bold")
    ax.set_xlabel("Puntuación de PageRank")
    ax.set_title("Ranking de Importancia por Nodo", fontsize=11, fontweight="bold", pad=8)
    
    # Añadir valores al final de las barras
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, f"{width:.5f}", 
                ha='left', va='center', fontsize=9, fontweight='bold')
                
    # Dar margen en x
    ax.set_xlim(0, max(scores) * 1.15)
    
    # Omitir bordes superior y derecho
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def exportar_grafico_rankings(rankings_ordenados: list[tuple], ruta_salida: str):
    """
    Genera y guarda el gráfico de rankings en disco.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    dibujar_rankings_en_eje(rankings_ordenados, ax)
    fig.tight_layout()
    fig.savefig(ruta_salida, format=ruta_salida.split('.')[-1], dpi=300)
    plt.close(fig)
