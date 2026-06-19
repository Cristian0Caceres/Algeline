# -*- coding: utf-8 -*-
"""Módulo de Análisis de Convergencia de PageRank.

JUSTIFICACIÓN TEÓRICA Y CONTEXTO ACADÉMICO:
------------------------------------------
Este módulo ha sido desarrollado para cumplir con las especificaciones de las
secciones III.A y III.B del proyecto de investigación sobre PageRank. La medición de
convergencia en el algoritmo PageRank es una aplicación directa de la teoría de
Cadenas de Markov y la teoría espectral de matrices.

1. Modelado como Cadena de Markov:
   El algoritmo PageRank modela el comportamiento de un usuario que navega por la web
   como una cadena de Markov discreta y homogénea en el tiempo. La red se representa
   como un grafo dirigido, cuya estructura se traduce en una matriz de transición $P$.
   Para garantizar la existencia de un único estado estacionario (vector de
   probabilidad invariante) y la convergencia del algoritmo, la cadena de Markov debe
   ser ergódica (irreducible y aperiódica).

2. Factor de Amortiguamiento (Damping Factor) y Teorema de Perron-Frobenius:
   La introduccion del damping factor $d \\in (0, 1)$ resuelve el problema de los sumideros
   (nodos sin enlaces salientes o 'dangling nodes') y los ciclos cerrados. Modifica
   la matriz de transición para dar origen a la matriz de Google $M$:
       $M = d * P + (1-d)/n * ones_matrix$
   Esta matriz es completamente positiva ($M_{ij} > 0$), lo que por el Teorema de
   Perron-Frobenius garantiza que:
   - El autovalor maximo es unico y vale lambda_1 = 1.
   - El autovector correspondiente v* (el vector PageRank), tiene componentes
     estrictamente positivas y su norma L1 es 1.
   - Para cualquier vector de probabilidad inicial v^{(0)}, el metodo de las potencias:
         v^{(k+1)} = M * v^{(k)}
     converge linealmente hacia el estado estacionario v*.
   - La tasa de convergencia esta dominada por el segundo autovalor en magnitud,
     donde |lambda_2| <= d. Por tanto, el error en la iteracion k se comporta
     como O(d^k).

3. Justificación de las Métricas de Convergencia Elegidas:
   Para medir cuantitativamente la discrepancia entre iteraciones consecutivas, se
   utilizan tres métricas basadas en normas vectoriales:
   - Norma L1 (Distancia de Variacion Total): Representa la suma absoluta de las
     diferencias. En teoria de probabilidad, la mitad de la norma L1 representa la
     distancia de variacion total, que es la medida natural para cuantificar la
     diferencia de probabilidad agregada en el sistema.
   - Norma L2 (Distancia Euclidiana): Representa la distancia geometrica en el espacio
     euclidiano R^n. Penaliza de forma cuadratica las variaciones grandes,
     siendo util para identificar fluctuaciones severas en un subconjunto de nodos.
   - Diferencia Maxima Absoluta (Norma L-infinito): Mide el cambio mas grande
     experimentado por cualquier nodo individual de la red. Es la metrica mas robusta
     para criterios de parada practicos, asegurando que ningun nodo cambie su ranking
     de manera significativa antes de detener el algoritmo.

REFERENCIAS BIBLIOGRÁFICAS REALES:
----------------------------------
1. Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual Web search
   engine. Computer Networks and ISDN Systems, 30(1-7), 107-117.
2. Langville, A. N., & Meyer, C. D. (2006). Google's PageRank and Beyond: The Science
   of Search Engine Rankings. Princeton University Press.
3. Haveliwala, T. H. (2002). Topic-sensitive PageRank. Proceedings of the 11th
   international conference on World Wide Web, 517-526.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes 
from typing import List, Tuple, Optional, Dict

def calculate_l1_norm(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula la norma L1 de la diferencia entre dos vectores.

    La norma L1 mide la suma de las diferencias absolutas de cada componente.
    Representa la distancia de variación total en distribuciones de probabilidad.
    Complejidad temporal: O(n) donde n es la dimensión del vector.

    Args:
        v1: Primer vector numpy de forma (n,).
        v2: Segundo vector numpy de forma (n,).

    Returns:
        float: Valor de la norma L1 de la diferencia.
    """
    return float(np.sum(np.abs(v1 - v2)))


def calculate_l2_norm(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula la norma L2 (euclidiana) de la diferencia entre dos vectores.

    La norma L2 mide la distancia geométrica en el espacio euclídeo. Penaliza
    con mayor fuerza las fluctuaciones individuales grandes.
    Complejidad temporal: O(n) donde n es la dimensión del vector.

    Args:
        v1: Primer vector numpy de forma (n,).
        v2: Segundo vector numpy de forma (n,).

    Returns:
        float: Valor de la norma L2 de la diferencia.
    """
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def calculate_max_absolute_difference(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula la diferencia máxima absoluta (norma L-infinito) entre dos vectores.

    Esta norma captura el cambio máximo sufrido por un solo nodo en la red.
    Es la métrica más restrictiva y común para criterios de parada.
    Complejidad temporal: O(n) donde n es la dimensión del vector.

    Args:
        v1: Primer vector numpy de forma (n,).
        v2: Segundo vector numpy de forma (n,).

    Returns:
        float: Valor de la diferencia máxima absoluta.
    """
    return float(np.max(np.abs(v1 - v2)))


def record_pagerank_history(
    M: np.ndarray,
    v0: np.ndarray,
    max_iter: int,
    tol: Optional[float] = None,
    norm_type: str = "max_diff"
) -> List[np.ndarray]:
    """Ejecuta el método iterativo de PageRank y registra el vector de cada iteración.

    Esta función permite almacenar el historial completo del vector de rankings
    a lo largo de las iteraciones. Si se proporciona una tolerancia (`tol`), el método
    puede detenerse antes de alcanzar `max_iter` si se detecta convergencia.
    Complejidad temporal: O(k * n^2) donde k es el número de iteraciones y n el número de nodos.

    Args:
        M: Matriz de transición de Google de forma (n, n), estocástica por columnas.
        v0: Vector de probabilidad inicial de forma (n,).
        max_iter: Cantidad máxima de iteraciones a ejecutar.
        tol: Tolerancia opcional para parada temprana.
        norm_type: Norma para evaluar la parada temprana ('l1', 'l2' o 'max_diff').

    Returns:
        List[np.ndarray]: Lista que contiene el vector inicial y los vectores resultantes
        de cada iteración.
    """
    history = [v0.copy()]
    v_current = v0.copy()

    for _ in range(max_iter):
        v_next = M @ v_current
        history.append(v_next.copy())

        if tol is not None:
            if norm_type == "l1":
                err = calculate_l1_norm(v_next, v_current)
            elif norm_type == "l2":
                err = calculate_l2_norm(v_next, v_current)
            else:
                err = calculate_max_absolute_difference(v_next, v_current)
            
            if err < tol:
                break
                
        v_current = v_next

    return history


def check_convergence(
    history: List[np.ndarray],
    tolerance: float,
    max_iter: int,
    norm_type: str = "max_diff"
) -> Tuple[bool, int, float]:
    """Determina si un historial de vectores PageRank converge según parámetros dados.

    Analiza el historial de forma retrospectiva, comparando cada vector v^{(k)}
    con su predecesor v^{(k-1)} utilizando la métrica especificada. Identifica la primera
    iteración en la que el error cae por debajo de la tolerancia indicada.
    Complejidad temporal: O(k * n) donde k es la longitud del historial y n es la dimensión.

    Args:
        history: Lista de vectores numpy que representan el historial por iteración.
        tolerance: Umbral de tolerancia de error para declarar convergencia.
        max_iter: Límite máximo de iteraciones permitidas para el análisis.
        norm_type: Tipo de norma a utilizar ('l1', 'l2', 'max_diff').

    Returns:
        Tuple[bool, int, float]: Una tupla conteniendo:
            - converged (bool): True si se logró la convergencia dentro de los límites.
            - iteration (int): Índice de la iteración de convergencia (1-indexada).
              Si no converge, retorna la última iteración evaluada.
            - final_error (float): El error calculado en dicha iteración.
    """
    if len(history) < 2:
        return False, 0, 0.0

    # Limitar el historial a evaluar según el número máximo de iteraciones
    limit = min(len(history) - 1, max_iter)

    for k in range(1, limit + 1):
        v_curr = history[k]
        v_prev = history[k - 1]

        if norm_type == "l1":
            error = calculate_l1_norm(v_curr, v_prev)
        elif norm_type == "l2":
            error = calculate_l2_norm(v_curr, v_prev)
        elif norm_type == "max_diff":
            error = calculate_max_absolute_difference(v_curr, v_prev)
        else:
            raise ValueError(f"Tipo de norma no reconocido: {norm_type}. Usar 'l1', 'l2' o 'max_diff'.")

        if error < tolerance:
            return True, k, error

    # Si no converge dentro de los límites
    # Calculamos el error final del último paso analizado
    v_last = history[limit]
    v_prev_last = history[limit - 1]
    if norm_type == "l1":
        final_error = calculate_l1_norm(v_last, v_prev_last)
    elif norm_type == "l2":
        final_error = calculate_l2_norm(v_last, v_prev_last)
    else:
        final_error = calculate_max_absolute_difference(v_last, v_prev_last)

    return False, limit, final_error


def plot_convergence_history(
    history: List[np.ndarray],
    norm_types: List[str] = ["l1", "l2", "max_diff"],
    title: str = "Convergencia de PageRank por Iteración"
) -> Figure:
    """Genera un gráfico Matplotlib que muestra el error de convergencia vs iteraciones.

    Muestra de forma simultánea el decaimiento exponencial del error utilizando
    una escala semilogarítmica en el eje Y. Aplica una estética premium para
    su integración en informes académicos.
    Complejidad temporal: O(m * k * n) donde m es el número de normas y k es la longitud del historial.

    Args:
        history: Lista de vectores numpy representando el historial de iteraciones.
        norm_types: Normas a graficar (lista conteniendo 'l1', 'l2', 'max_diff').
        title: Título del gráfico.

    Returns:
        plt.Figure: La figura de Matplotlib que contiene el gráfico.
    """
    if len(history) < 2:
        raise ValueError("El historial debe contener al menos 2 vectores para graficar diferencias.")

    num_iters = len(history) - 1
    iterations = np.arange(1, num_iters + 1)

    errors: Dict[str, List[float]] = {norm: [] for norm in norm_types}

    for k in range(1, len(history)):
        v_curr = history[k]
        v_prev = history[k - 1]
        
        if "l1" in norm_types:
            errors["l1"].append(calculate_l1_norm(v_curr, v_prev))
        if "l2" in norm_types:
            errors["l2"].append(calculate_l2_norm(v_curr, v_prev))
        if "max_diff" in norm_types:
            errors["max_diff"].append(calculate_max_absolute_difference(v_curr, v_prev))

    # Configuración estética
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    
    # Paleta de colores Premium
    colors = {
        "l1": "#2A9D8F",        # Verde menta / Azul verdoso
        "l2": "#E76F51",        # Coral / Terracota
        "max_diff": "#264653"   # Slate Blue oscuro
    }
    
    labels = {
        "l1": "Norma L1 (Variación Total)",
        "l2": "Norma L2 (Distancia Euclidiana)",
        "max_diff": "Diferencia Máxima Absoluta (L-inf)"
    }

    markers = {
        "l1": "o",
        "l2": "s",
        "max_diff": "^"
    }

    for norm in norm_types:
        if norm in errors:
            ax.plot(
                iterations,
                errors[norm],
                label=labels[norm],
                color=colors[norm],
                marker=markers[norm],
                markersize=4,
                linewidth=1.75,
                alpha=0.95
            )

    # Configuración de los ejes
    ax.set_yscale("log")
    ax.set_xlabel("Iteración (k)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Error de Convergencia ||v^(k) - v^(k-1)|| (Escala Log)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15, color="#1D1D1D")
    
    # Líneas de cuadrícula sutiles
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7, color="#CCCCCC")
    
    # Ajuste de bordes (remover los de arriba y derecha para estética limpia)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    # Leyenda premium
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="#FBFBFB",
        edgecolor="#E5E5E5",
        fontsize=10,
        shadow=False
    )
    
    plt.tight_layout()
    return fig
