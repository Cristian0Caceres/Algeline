import matplotlib.pyplot as plt

def dibujar_convergencia_en_eje(historial_diferencias: list[float], ax: plt.Axes, label: str = "Diferencia"):
    """
    Dibuja la evolución de la norma de diferencia entre iteraciones sucesivas.
    """
    ax.clear()
    if not historial_diferencias:
        ax.text(0.5, 0.5, "Sin datos de convergencia", ha='center', va='center', color='gray')
        return
        
    ax.plot(range(1, len(historial_diferencias) + 1), historial_diferencias, marker='o', linestyle='-', color='teal', label=label)
    ax.set_yscale('log')
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Norma L1 de Diferencia (Escala Log)")
    ax.set_title("Evolución de Convergencia", fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

def dibujar_comparacion_damping_en_eje(comparaciones: dict[float, list[float]], ax: plt.Axes):
    """
    Compara la evolución de la convergencia para distintos damping factors.
    comparaciones: dicc con clave = damping factor y valor = lista de diferencias
    """
    ax.clear()
    if not comparaciones:
        ax.text(0.5, 0.5, "Sin datos de comparación", ha='center', va='center', color='gray')
        return
        
    for d, diffs in comparaciones.items():
        ax.plot(range(1, len(diffs) + 1), diffs, marker='x', linestyle='--', label=f"d = {d}")
        
    ax.set_yscale('log')
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Norma L1 de Diferencia (Escala Log)")
    ax.set_title("Comparación de Convergencia por Damping Factor", fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

def exportar_convergencia(historial_diferencias: list[float], ruta_salida: str, label: str = "Diferencia"):
    fig, ax = plt.subplots(figsize=(8, 6))
    dibujar_convergencia_en_eje(historial_diferencias, ax, label)
    fig.tight_layout()
    fig.savefig(ruta_salida, format=ruta_salida.split('.')[-1], dpi=300)
    plt.close(fig)
