# -*- coding: utf-8 -*-
"""Módulo de Experimentos de PageRank.

JUSTIFICACIÓN TEÓRICA Y CONTEXTO ACADÉMICO:
------------------------------------------
Este módulo satisface los requerimientos de la sección III.A (puntos 2 y 3) y
III.B del proyecto, que exige implementar pruebas variando parámetros clave del
algoritmo PageRank y generar gráficos que representen los resultados obtenidos.

1. Método Iterativo y Construcción de la Matriz de Transición:
   El solver de PageRank implementado en este módulo sigue el modelo original de
   Brin & Page (1998). La matriz de transición $P$ se construye normalizando las
   columnas de la matriz de adyacencia $A$, de modo que cada columna $j$ representa
   la probabilidad de que un usuario ubicado en el nodo $j$ siga un enlace hacia el
   nodo $i$.

   Manejo de Dangling Nodes:
   Los nodos sin enlaces salientes ('dangling nodes') generarían columnas de ceros,
   lo que rompe la propiedad estocástica de la matriz. Para corregirlo, su
   probabilidad de transición se redistribuye uniformemente entre todos los nodos,
   reemplazando la columna de ceros por $e/n$ (donde $e$ es el vector de unos y
   $n$ el número de nodos). Esta corrección es estándar en la literatura
   (Langville & Meyer, 2006).

   Matriz de Google:
   Una vez corregida la estocasticidad, la matriz de Google se define como:
       $M = d \\cdot P + \\frac{1-d}{n} \\mathbf{e} \\mathbf{e}^T$
   donde $d$ es el damping factor. Esta formulación garantiza que $M$ sea
   primitiva (positiva), satisfaciendo las condiciones del Teorema de
   Perron-Frobenius para la convergencia única.

2. Efecto del Damping Factor sobre la Convergencia:
   El factor de amortiguamiento $d$ controla la velocidad de convergencia del
   método de las potencias. Como demostró Brin & Page, la tasa de convergencia
   geométrica es $O(d^k)$, donde $k$ es el número de iteraciones. Langville &
   Meyer (2006) establecen que el valor $d = 0.85$ es un compromiso empírico
   entre representación estructural de la red y velocidad de convergencia:
   - Un $d$ bajo (ej. $d = 0.5$) converge rápido pero suaviza demasiado las
     diferencias estructurales entre nodos.
   - Un $d$ alto (ej. $d = 0.95$) preserva mejor la estructura pero requiere
     muchas más iteraciones.

3. Experimentos Variando Estructura del Grafo:
   El PDF solicita analizar cómo la estructura de la red afecta el ranking final.
   Se implementan cuatro topologías dirigidas clásicas:
   - Aleatoria (Erdős–Rényi): Modela redes de uso general.
   - Cíclica: Produce rankings perfectamente uniformes, sirviendo como caso base
     matemático para verificar corrección.
   - Estrella: Produce rankings extremadamente concentrados en el hub central.
   - Cadena lineal (Path): Produce una distribución gradiente de importancia.

4. Separación Clara de Responsabilidades:
   El módulo mantiene separación explícita entre:
   - Generación de datos: funciones `generate_directed_graph` y `run_pagerank_solver`.
   - Ejecución de experimentos: funciones `experiment_*`.
   - Visualización: función `plot_experiment_results`.

REFERENCIAS BIBLIOGRÁFICAS REALES:
----------------------------------
1. Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual Web search
   engine. Computer Networks and ISDN Systems, 30(1-7), 107-117.
2. Langville, A. N., & Meyer, C. D. (2006). Google's PageRank and Beyond: The Science
   of Search Engine Rankings. Princeton University Press.
3. Haveliwala, T. H., & Kamvar, S. D. (2003). The second eigenvalue of the Google
   matrix. Stanford University Technical Report, 2003-20.
4. Erdős, P., & Rényi, A. (1960). On the evolution of random graphs. Magyar Tud.
   Akad. Mat. Kutató Int. Közl., 5, 17-61.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Any
import os

from analysis.convergence import (
    check_convergence,
    plot_convergence_history,
    record_pagerank_history,
)


# ---------------------------------------------------------------------------
# SECCIÓN 1: GENERACIÓN DE DATOS
# ---------------------------------------------------------------------------

def generate_directed_graph(
    n_nodes: int,
    structure_type: str,
    edge_probability: float = 0.3,
    seed: int = 42
) -> nx.DiGraph:
    """Genera un grafo dirigido con la topología especificada usando NetworkX.

    Crea distintas estructuras de red dirigida para estudiar cómo la topología
    afecta los resultados del algoritmo PageRank.

    Complejidad temporal:
        - 'random': O(n^2) por generación de aristas aleatoria.
        - 'cycle', 'path', 'star': O(n).

    Args:
        n_nodes: Número de nodos del grafo. Debe ser >= 2.
        structure_type: Tipo de estructura del grafo. Opciones válidas:
            - 'random': Grafo aleatorio estilo Erdős–Rényi con probabilidad
              de arista ``edge_probability``.
            - 'cycle': Grafo cíclico dirigido (0→1→2→...→n-1→0).
            - 'star': Grafo estrella donde el nodo 0 es el hub receptor de
              todos los nodos y emite hacia todos.
            - 'path': Cadena lineal dirigida (0→1→2→...→n-1).
        edge_probability: Probabilidad de arista para grafos aleatorios.
            Ignorado para otras topologías.
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        nx.DiGraph: Grafo dirigido de NetworkX con ``n_nodes`` nodos.

    Raises:
        ValueError: Si ``structure_type`` no es una opción válida o ``n_nodes`` < 2.
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes debe ser >= 2. Se recibió: {n_nodes}.")

    valid_types = {"random", "cycle", "star", "path"}
    if structure_type not in valid_types:
        raise ValueError(
            f"structure_type '{structure_type}' no es válido. "
            f"Use uno de: {valid_types}."
        )

    rng = np.random.default_rng(seed)

    if structure_type == "random":
        # Erdős–Rényi: cada arista dirigida (i, j) con i ≠ j existe con
        # probabilidad edge_probability. Se fuerza conectividad añadiendo
        # un ciclo base para evitar grafos completamente desconectados.
        G = nx.erdos_renyi_graph(n=n_nodes, p=edge_probability, directed=True, seed=seed)
        # Forzar al menos un ciclo para garantizar que no haya dangling nodes totales
        for i in range(n_nodes):
            G.add_edge(i, (i + 1) % n_nodes)

    elif structure_type == "cycle":
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            G.add_edge(i, (i + 1) % n_nodes)

    elif structure_type == "star":
        # El nodo 0 es el hub. Todos los nodos periféricos (1..n-1) apuntan al hub.
        # El hub apunta a todos los periféricos.
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        for i in range(1, n_nodes):
            G.add_edge(i, 0)        # Periférico → Hub (alta centralidad del hub)
            G.add_edge(0, i)        # Hub → Periférico (distribución desde el hub)

    elif structure_type == "path":
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes - 1):
            G.add_edge(i, i + 1)
        # Añadir retorno desde el último nodo al primero para evitar sumidero total
        G.add_edge(n_nodes - 1, 0)

    return G


def run_pagerank_solver(
    G: nx.DiGraph,
    d: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[np.ndarray, List[np.ndarray], Dict[Any, float]]:
    """Ejecuta el algoritmo PageRank iterativo sobre un grafo dirigido.

    Construye la matriz de transición estocástica a partir del grafo,
    aplica el damping factor para obtener la Matriz de Google y ejecuta
    el método de las potencias $v^{(k+1)} = M v^{(k)}$ registrando el
    historial completo de vectores por iteración.

    Complejidad temporal: O(k * n^2) donde k es el número de iteraciones y
    n el número de nodos. Para grafos dispersos puede aproximarse a O(k * m)
    con representación de matrices dispersas, pero se usa densas por claridad.

    Args:
        G: Grafo dirigido de NetworkX con al menos 2 nodos.
        d: Damping factor (factor de amortiguamiento). Clásicamente 0.85.
            Debe estar en el intervalo abierto (0, 1).
        max_iter: Número máximo de iteraciones del método de las potencias.
        tol: Tolerancia de parada temprana basada en la norma L1. Si es
            alcanzada, el método se detiene antes de ``max_iter``.

    Returns:
        Tuple conteniendo:
            - final_vector (np.ndarray): Vector PageRank final de forma (n,),
              con norma L1 = 1.
            - history (List[np.ndarray]): Lista con el vector inicial más el
              vector resultante de cada iteración.
            - node_scores (Dict[Any, float]): Diccionario que mapea cada nodo
              del grafo a su score de PageRank final.

    Raises:
        ValueError: Si ``d`` no está en (0, 1).
    """
    if not (0.0 < d < 1.0):
        raise ValueError(f"El damping factor d debe estar en (0, 1). Se recibió: {d}.")

    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    # --- Construcción de la Matriz de Adyacencia ---
    # A[i, j] = 1 si existe arista de j hacia i (columna j → fila i)
    # Complejidad: O(m) donde m es el número de aristas
    A = np.zeros((n, n), dtype=float)
    for u, v in G.edges():
        col = node_index[u]  # nodo origen (columna)
        row = node_index[v]  # nodo destino (fila)
        A[row, col] = 1.0

    # --- Construcción de la Matriz de Transición Estocástica P ---
    # Normalizar columnas: P[:, j] = A[:, j] / sum(A[:, j]) si sum > 0
    col_sums = A.sum(axis=0)
    P = np.zeros((n, n), dtype=float)
    for j in range(n):
        if col_sums[j] > 0:
            P[:, j] = A[:, j] / col_sums[j]
        else:
            # Dangling node: redistribuir uniformemente
            P[:, j] = 1.0 / n

    # --- Construcción de la Matriz de Google ---
    # M = d * P + (1 - d) / n * ones_matrix
    ones_matrix = np.ones((n, n), dtype=float)
    M = d * P + (1.0 - d) / n * ones_matrix

    # --- Inicialización del vector de probabilidad uniforme ---
    v = np.ones(n, dtype=float) / n

    # --- Ejecutar iteraciones y registrar historial ---
    history = record_pagerank_history(M=M, v0=v, max_iter=max_iter, tol=tol, norm_type="l1")

    final_vector = history[-1]

    # Mapear cada nodo a su score final
    node_scores = {nodes[i]: float(final_vector[i]) for i in range(n)}

    return final_vector, history, node_scores


# ---------------------------------------------------------------------------
# SECCIÓN 2: EXPERIMENTOS
# ---------------------------------------------------------------------------

def experiment_node_scaling(
    nodes_list: List[int],
    structure_type: str = "random",
    d: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """Experimento 1: Analiza PageRank variando el número de nodos.

    Ejecuta PageRank para grafos de distintos tamaños y mide la convergencia
    en cada caso. Permite comparar el impacto del tamaño de la red sobre el
    comportamiento del algoritmo.

    Args:
        nodes_list: Lista de enteros indicando los tamaños de red a evaluar.
        structure_type: Topología del grafo ('random', 'cycle', 'star', 'path').
        d: Damping factor.
        max_iter: Número máximo de iteraciones.
        tol: Tolerancia de convergencia.

    Returns:
        Dict[str, Any]: Diccionario con los resultados, incluyendo:
            - 'nodes_list': Lista de tamaños evaluados.
            - 'histories': Diccionario {n_nodes: history}.
            - 'final_scores': Diccionario {n_nodes: node_scores}.
            - 'convergence': Diccionario {n_nodes: (converged, iter, error)}.
    """
    results: Dict[str, Any] = {
        "experiment": "node_scaling",
        "structure_type": structure_type,
        "nodes_list": nodes_list,
        "d": d,
        "histories": {},
        "final_scores": {},
        "convergence": {},
    }

    for n in nodes_list:
        G = generate_directed_graph(n_nodes=n, structure_type=structure_type)
        final_vec, history, node_scores = run_pagerank_solver(G, d=d, max_iter=max_iter, tol=tol)
        conv = check_convergence(history, tolerance=tol, max_iter=max_iter, norm_type="l1")

        results["histories"][n] = history
        results["final_scores"][n] = node_scores
        results["convergence"][n] = conv

    return results


def experiment_edge_density(
    node_count: int,
    densities: List[float],
    d: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """Experimento 2: Analiza PageRank variando la densidad de enlaces.

    Genera grafos aleatorios con distinta probabilidad de arista y evalúa cómo
    la densidad de la red impacta la distribución final de los rankings.

    Args:
        node_count: Número fijo de nodos en todos los grafos.
        densities: Lista de probabilidades de arista (entre 0 y 1).
        d: Damping factor.
        max_iter: Número máximo de iteraciones.
        tol: Tolerancia de convergencia.

    Returns:
        Dict[str, Any]: Diccionario con los resultados indexados por densidad.
    """
    results: Dict[str, Any] = {
        "experiment": "edge_density",
        "node_count": node_count,
        "densities": densities,
        "d": d,
        "histories": {},
        "final_scores": {},
        "convergence": {},
    }

    for p in densities:
        G = generate_directed_graph(n_nodes=node_count, structure_type="random", edge_probability=p)
        final_vec, history, node_scores = run_pagerank_solver(G, d=d, max_iter=max_iter, tol=tol)
        conv = check_convergence(history, tolerance=tol, max_iter=max_iter, norm_type="l1")

        results["histories"][p] = history
        results["final_scores"][p] = node_scores
        results["convergence"][p] = conv

    return results


def experiment_damping_factor(
    G: nx.DiGraph,
    damping_factors: List[float],
    max_iter: int = 150,
    tol: float = 1e-8
) -> Dict[str, Any]:
    """Experimento 3: Analiza el efecto del damping factor sobre la convergencia.

    Mantiene el grafo fijo y varía el factor de amortiguamiento. Permite
    comparar directamente la velocidad de convergencia y los rankings finales
    obtenidos para distintos valores de d.

    La tasa de convergencia teórica es O(d^k). Para d=0.5 se esperan aprox.
    la mitad de iteraciones que para d=0.85 (Haveliwala & Kamvar, 2003).

    Args:
        G: Grafo dirigido de NetworkX de referencia.
        damping_factors: Lista de valores de d a evaluar (cada uno en (0,1)).
        max_iter: Número máximo de iteraciones.
        tol: Tolerancia de convergencia.

    Returns:
        Dict[str, Any]: Diccionario con los resultados indexados por valor de d.
    """
    results: Dict[str, Any] = {
        "experiment": "damping_factor",
        "damping_factors": damping_factors,
        "max_iter": max_iter,
        "histories": {},
        "final_scores": {},
        "convergence": {},
    }

    for d in damping_factors:
        final_vec, history, node_scores = run_pagerank_solver(G, d=d, max_iter=max_iter, tol=tol)
        conv = check_convergence(history, tolerance=tol, max_iter=max_iter, norm_type="l1")

        results["histories"][d] = history
        results["final_scores"][d] = node_scores
        results["convergence"][d] = conv

    return results


def experiment_max_iterations(
    G: nx.DiGraph,
    max_iters: List[int],
    d: float = 0.85,
    tol: float = 1e-12
) -> Dict[str, Any]:
    """Experimento 4: Analiza el efecto del número máximo de iteraciones.

    Ejecuta PageRank con distintos límites de iteración sobre el mismo grafo.
    Con tolerancia casi nula, el algoritmo se detendrá únicamente al agotar
    las iteraciones, permitiendo comparar cómo el ranking evoluciona con k.

    Args:
        G: Grafo dirigido de NetworkX de referencia.
        max_iters: Lista de límites de iteración a evaluar.
        d: Damping factor.
        tol: Tolerancia de convergencia (muy pequeña para forzar todas las
            iteraciones hasta el límite).

    Returns:
        Dict[str, Any]: Diccionario con los resultados indexados por max_iter.
    """
    results: Dict[str, Any] = {
        "experiment": "max_iterations",
        "max_iters": max_iters,
        "d": d,
        "histories": {},
        "final_scores": {},
        "convergence": {},
    }

    for k in max_iters:
        final_vec, history, node_scores = run_pagerank_solver(G, d=d, max_iter=k, tol=tol)
        conv = check_convergence(history, tolerance=1e-6, max_iter=k, norm_type="l1")

        results["histories"][k] = history
        results["final_scores"][k] = node_scores
        results["convergence"][k] = conv

    return results


# ---------------------------------------------------------------------------
# SECCIÓN 3: VISUALIZACIÓN
# ---------------------------------------------------------------------------

def plot_experiment_results(
    results: Dict[str, Any],
    experiment_type: str,
    output_dir: str = "outputs/figures"
) -> List[plt.Figure]:
    """Genera y guarda gráficos para los resultados de un experimento de PageRank.

    Selecciona el tipo de visualización según el experimento ejecutado.
    Mantiene separación clara entre datos y visualización.

    Gráficos generados por tipo de experimento:
        - 'node_scaling':    Iteraciones de convergencia vs. número de nodos.
        - 'edge_density':    Distribución de rankings vs. densidad de aristas.
        - 'damping_factor':  Convergencia por iteración para cada valor de d
                             (comparación multi-curva) + rankings finales comparados.
        - 'max_iterations':  Evolución de rankings y curvas de convergencia.

    Args:
        results: Diccionario retornado por cualquiera de las funciones
            ``experiment_*``.
        experiment_type: Tipo de experimento ('node_scaling', 'edge_density',
            'damping_factor', 'max_iterations').
        output_dir: Directorio de salida para guardar los gráficos PNG.

    Returns:
        List[plt.Figure]: Lista de figuras Matplotlib generadas.

    Raises:
        ValueError: Si ``experiment_type`` no es reconocido.
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_types = {"node_scaling", "edge_density", "damping_factor", "max_iterations"}
    if experiment_type not in valid_types:
        raise ValueError(
            f"experiment_type '{experiment_type}' no reconocido. "
            f"Use uno de: {valid_types}."
        )

    figures: List[plt.Figure] = []

    if experiment_type == "node_scaling":
        figures.extend(_plot_node_scaling(results, output_dir))

    elif experiment_type == "edge_density":
        figures.extend(_plot_edge_density(results, output_dir))

    elif experiment_type == "damping_factor":
        figures.extend(_plot_damping_factor(results, output_dir))

    elif experiment_type == "max_iterations":
        figures.extend(_plot_max_iterations(results, output_dir))

    return figures


def _plot_node_scaling(results: Dict[str, Any], output_dir: str) -> List[plt.Figure]:
    """Visualiza el experimento de escalado por número de nodos.

    Args:
        results: Resultados del experimento 'node_scaling'.
        output_dir: Directorio de salida.

    Returns:
        List[plt.Figure]: Figuras generadas.
    """
    figs = []
    nodes_list = results["nodes_list"]

    # --- Gráfico 1: Iteraciones hasta convergencia vs. número de nodos ---
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

    iters_to_conv = []
    for n in nodes_list:
        converged, it, err = results["convergence"][n]
        iters_to_conv.append(it)

    bar_colors = cm.viridis(np.linspace(0.25, 0.85, len(nodes_list)))
    bars = ax1.bar(
        [str(n) for n in nodes_list],
        iters_to_conv,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8
    )

    for bar, val in zip(bars, iters_to_conv):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#333333"
        )

    ax1.set_xlabel("Número de Nodos (n)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Iteraciones hasta Convergencia", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_title(
        "Escalado de PageRank: Iteraciones vs. Número de Nodos",
        fontsize=13, fontweight="bold", pad=14
    )
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, color="#CCCCCC")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()

    path1 = os.path.join(output_dir, "exp1_node_scaling_convergence.png")
    fig1.savefig(path1, dpi=150)
    figs.append(fig1)

    # --- Gráfico 2: Importancia relativa de nodos (top-N) para cada tamaño ---
    fig2, axes = plt.subplots(1, len(nodes_list), figsize=(4.5 * len(nodes_list), 5), dpi=150)
    if len(nodes_list) == 1:
        axes = [axes]

    for ax, n in zip(axes, nodes_list):
        scores = results["final_scores"][n]
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k = min(10, len(sorted_scores))
        labels = [str(node) for node, _ in sorted_scores[:top_k]]
        values = [val for _, val in sorted_scores[:top_k]]

        bar_c = cm.plasma(np.linspace(0.2, 0.85, top_k))
        ax.barh(labels[::-1], values[::-1], color=bar_c[::-1], edgecolor="white", linewidth=0.6)
        ax.set_xlabel("Score PageRank", fontsize=9, labelpad=6)
        ax.set_title(f"n={n} nodos", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6, color="#CCCCCC")

    fig2.suptitle("Importancia Relativa de Nodos por Tamaño de Red", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "exp1_node_scaling_rankings.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    figs.append(fig2)

    return figs


def _plot_edge_density(results: Dict[str, Any], output_dir: str) -> List[plt.Figure]:
    """Visualiza el experimento de densidad de enlaces.

    Args:
        results: Resultados del experimento 'edge_density'.
        output_dir: Directorio de salida.

    Returns:
        List[plt.Figure]: Figuras generadas.
    """
    figs = []
    densities = results["densities"]

    # --- Gráfico 1: Comparación de Rankings según densidad ---
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    colors = cm.cool(np.linspace(0.1, 0.9, len(densities)))

    for color, p in zip(colors, densities):
        scores = results["final_scores"][p]
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k = min(10, len(sorted_scores))
        labels = [str(s[0]) for s in sorted_scores[:top_k]]
        values = [s[1] for s in sorted_scores[:top_k]]
        ax1.plot(range(top_k), values, marker="o", markersize=5,
                 linewidth=1.6, label=f"p={p:.2f}", color=color, alpha=0.9)

    ax1.set_xlabel("Rank del Nodo (1 = mayor importancia)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Score PageRank", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_title(
        "Distribución de Rankings según Densidad de Aristas",
        fontsize=13, fontweight="bold", pad=14
    )
    ax1.legend(loc="upper right", fontsize=9, frameon=True, facecolor="#FBFBFB", edgecolor="#E5E5E5")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color="#CCCCCC")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()

    path1 = os.path.join(output_dir, "exp2_edge_density_rankings.png")
    fig1.savefig(path1, dpi=150)
    figs.append(fig1)

    # --- Gráfico 2: Convergencia por densidad ---
    fig2, ax2 = plt.subplots(figsize=(9, 5.5), dpi=150)
    for color, p in zip(colors, densities):
        history = results["histories"][p]
        n_iters = len(history) - 1
        errors = [
            float(np.sum(np.abs(history[k] - history[k - 1])))
            for k in range(1, len(history))
        ]
        ax2.plot(range(1, n_iters + 1), errors, marker="s", markersize=3,
                 linewidth=1.5, label=f"p={p:.2f}", color=color, alpha=0.9)

    ax2.set_yscale("log")
    ax2.set_xlabel("Iteración (k)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Error L1 (Escala Log)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_title("Convergencia de PageRank por Densidad de Aristas", fontsize=13, fontweight="bold", pad=14)
    ax2.legend(loc="upper right", fontsize=9, frameon=True, facecolor="#FBFBFB", edgecolor="#E5E5E5")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7, color="#CCCCCC")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()

    path2 = os.path.join(output_dir, "exp2_edge_density_convergence.png")
    fig2.savefig(path2, dpi=150)
    figs.append(fig2)

    return figs


def _plot_damping_factor(results: Dict[str, Any], output_dir: str) -> List[plt.Figure]:
    """Visualiza el experimento de variación del damping factor.

    Args:
        results: Resultados del experimento 'damping_factor'.
        output_dir: Directorio de salida.

    Returns:
        List[plt.Figure]: Figuras generadas.
    """
    figs = []
    damping_factors = results["damping_factors"]
    colors = cm.magma(np.linspace(0.15, 0.85, len(damping_factors)))

    # --- Gráfico 1: Convergencia por iteración para cada valor de d ---
    fig1, ax1 = plt.subplots(figsize=(9, 5.5), dpi=150)

    for color, d in zip(colors, damping_factors):
        history = results["histories"][d]
        errors = [
            float(np.sum(np.abs(history[k] - history[k - 1])))
            for k in range(1, len(history))
        ]
        n_iters = len(errors)
        ax1.plot(range(1, n_iters + 1), errors, marker="o", markersize=4,
                 linewidth=1.8, label=f"d={d:.2f}", color=color, alpha=0.9)

    ax1.set_yscale("log")
    ax1.set_xlabel("Iteración (k)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Error L1 ||v^(k) - v^(k-1)||₁ (Escala Log)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_title(
        "Efecto del Damping Factor sobre la Convergencia de PageRank",
        fontsize=13, fontweight="bold", pad=14
    )
    ax1.legend(loc="upper right", fontsize=10, frameon=True, facecolor="#FBFBFB", edgecolor="#E5E5E5")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7, color="#CCCCCC")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()

    path1 = os.path.join(output_dir, "exp3_damping_convergence.png")
    fig1.savefig(path1, dpi=150)
    figs.append(fig1)

    # --- Gráfico 2: Comparación de rankings finales para cada valor de d ---
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=150)
    n_nodes = len(next(iter(results["final_scores"].values())))
    node_labels = list(next(iter(results["final_scores"].values())).keys())
    x = np.arange(len(node_labels))
    width = 0.8 / len(damping_factors)

    for i, (color, d) in enumerate(zip(colors, damping_factors)):
        scores = results["final_scores"][d]
        vals = [scores[node] for node in node_labels]
        offset = (i - len(damping_factors) / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width=width * 0.9, color=color,
                label=f"d={d:.2f}", edgecolor="white", linewidth=0.4, alpha=0.9)

    ax2.set_xlabel("Nodo", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Score PageRank", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_title(
        "Comparación de Rankings Finales por Damping Factor",
        fontsize=13, fontweight="bold", pad=14
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in node_labels], rotation=45, ha="right", fontsize=8)
    ax2.legend(loc="upper right", fontsize=9, frameon=True, facecolor="#FBFBFB", edgecolor="#E5E5E5")
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, color="#CCCCCC")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()

    path2 = os.path.join(output_dir, "exp3_damping_rankings_comparison.png")
    fig2.savefig(path2, dpi=150)
    figs.append(fig2)

    return figs


def _plot_max_iterations(results: Dict[str, Any], output_dir: str) -> List[plt.Figure]:
    """Visualiza el experimento de variación del número máximo de iteraciones.

    Args:
        results: Resultados del experimento 'max_iterations'.
        output_dir: Directorio de salida.

    Returns:
        List[plt.Figure]: Figuras generadas.
    """
    figs = []
    max_iters = results["max_iters"]
    colors = cm.cividis(np.linspace(0.15, 0.85, len(max_iters)))

    # --- Gráfico 1: Evolución del top-3 nodos a lo largo de las iteraciones ---
    # Usamos el historial del experimento con más iteraciones como referencia
    ref_key = max(max_iters)
    ref_history = results["histories"][ref_key]
    n_nodes = len(ref_history[0])

    # Identificar los top-3 nodos por score final
    final_scores_ref = results["final_scores"][ref_key]
    top3_nodes = sorted(final_scores_ref.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_indices = [list(final_scores_ref.keys()).index(node) for node, _ in top3_nodes]

    fig1, ax1 = plt.subplots(figsize=(10, 5.5), dpi=150)
    node_colors = ["#E76F51", "#2A9D8F", "#264653"]

    for color, (node, _), idx in zip(node_colors, top3_nodes, top3_indices):
        scores_over_time = [float(ref_history[k][idx]) for k in range(len(ref_history))]
        ax1.plot(range(len(scores_over_time)), scores_over_time, color=color,
                 linewidth=2.0, label=f"Nodo {node}", marker="o", markersize=3, alpha=0.9)

    ax1.set_xlabel("Iteración (k)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Score PageRank", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_title(
        f"Evolución del Score PageRank de los Top-3 Nodos ({ref_key} iteraciones máx.)",
        fontsize=13, fontweight="bold", pad=14
    )
    ax1.legend(loc="upper right", fontsize=10, frameon=True, facecolor="#FBFBFB", edgecolor="#E5E5E5")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color="#CCCCCC")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()

    path1 = os.path.join(output_dir, "exp4_maxiters_ranking_evolution.png")
    fig1.savefig(path1, dpi=150)
    figs.append(fig1)

    # --- Gráfico 2: Convergencia comparada para distintos max_iter ---
    fig2, ax2 = plt.subplots(figsize=(9, 5.5), dpi=150)

    for color, k_max in zip(colors, max_iters):
        history = results["histories"][k_max]
        errors = [
            float(np.sum(np.abs(history[k] - history[k - 1])))
            for k in range(1, len(history))
        ]
        ax2.plot(range(1, len(errors) + 1), errors, marker="^", markersize=4,
                 linewidth=1.6, label=f"max_iter={k_max}", color=color, alpha=0.9)

    ax2.set_yscale("log")
    ax2.set_xlabel("Iteración (k)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Error L1 (Escala Log)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_title(
        "Convergencia de PageRank para Distintos Límites de Iteración",
        fontsize=13, fontweight="bold", pad=14
    )
    ax2.legend(loc="upper right", fontsize=9, frameon=True, facecolor="#FBFBFB", edgecolor="#E5E5E5")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7, color="#CCCCCC")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()

    path2 = os.path.join(output_dir, "exp4_maxiters_convergence.png")
    fig2.savefig(path2, dpi=150)
    figs.append(fig2)

    return figs


# ---------------------------------------------------------------------------
# SECCIÓN 4: EJECUCIÓN AUTÓNOMA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Punto de entrada autónomo del módulo.

    Ejecuta los cuatro experimentos de forma secuencial, genera todos los
    gráficos y los guarda en 'outputs/figures/'. Permite ejecutar el módulo
    directamente con:
        python -m analysis.experiments
    o bien (desde la raíz del proyecto):
        python analysis/experiments.py
    """
    import sys
    import pathlib

    # Asegurar que la raíz del proyecto esté en el path de Python
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    OUTPUT_DIR = os.path.join(project_root, "outputs", "figures")

    print("=" * 65)
    print("  EXPERIMENTOS DE PAGERANK - Algebra Lineal para la Computacion")
    print("=" * 65)

    # ----------------------------------------------------------------
    # EXPERIMENTO 1: Escalado por número de nodos
    # ----------------------------------------------------------------
    print("\n[1/4] Experimento: Variación de número de nodos...")
    res1 = experiment_node_scaling(
        nodes_list=[5, 10, 20, 50],
        structure_type="random",
        d=0.85,
        max_iter=100,
        tol=1e-6
    )
    for n in res1["nodes_list"]:
        converged, it, err = res1["convergence"][n]
        status = "Converge" if converged else "No converge"
        print(f"   n={n:3d} -> {status} en iteracion {it:3d} | Error L1 final: {err:.2e}")
    figs1 = plot_experiment_results(res1, "node_scaling", output_dir=OUTPUT_DIR)
    print(f"   -> {len(figs1)} gráficos guardados en '{OUTPUT_DIR}'")

    # ----------------------------------------------------------------
    # EXPERIMENTO 2: Variación de densidad de enlaces
    # ----------------------------------------------------------------
    print("\n[2/4] Experimento: Variación de densidad de enlaces...")
    res2 = experiment_edge_density(
        node_count=20,
        densities=[0.1, 0.25, 0.5, 0.75],
        d=0.85,
        max_iter=100,
        tol=1e-6
    )
    for p in res2["densities"]:
        converged, it, err = res2["convergence"][p]
        status = "Converge" if converged else "No converge"
        print(f"   p={p:.2f} -> {status} en iteracion {it:3d} | Error L1 final: {err:.2e}")
    figs2 = plot_experiment_results(res2, "edge_density", output_dir=OUTPUT_DIR)
    print(f"   -> {len(figs2)} gráficos guardados en '{OUTPUT_DIR}'")

    # ----------------------------------------------------------------
    # EXPERIMENTO 3: Efecto del damping factor
    # ----------------------------------------------------------------
    print("\n[3/4] Experimento: Efecto del damping factor...")
    G_ref = generate_directed_graph(n_nodes=15, structure_type="random", seed=7)
    res3 = experiment_damping_factor(
        G=G_ref,
        damping_factors=[0.50, 0.75, 0.85, 0.95],
        max_iter=150,
        tol=1e-8
    )
    for d in res3["damping_factors"]:
        converged, it, err = res3["convergence"][d]
        status = "Converge" if converged else "No converge"
        print(f"   d={d:.2f} -> {status} en iteracion {it:3d} | Error L1 final: {err:.2e}")
    figs3 = plot_experiment_results(res3, "damping_factor", output_dir=OUTPUT_DIR)
    print(f"   -> {len(figs3)} gráficos guardados en '{OUTPUT_DIR}'")

    # ----------------------------------------------------------------
    # EXPERIMENTO 4: Variación de número máximo de iteraciones
    # ----------------------------------------------------------------
    print("\n[4/4] Experimento: Variación de número máximo de iteraciones...")
    G_iter = generate_directed_graph(n_nodes=12, structure_type="random", seed=21)
    res4 = experiment_max_iterations(
        G=G_iter,
        max_iters=[5, 20, 50, 100],
        d=0.85,
        tol=1e-12
    )
    for k in res4["max_iters"]:
        converged, it, err = res4["convergence"][k]
        status = "Converge" if converged else "No converge"
        print(f"   max_iter={k:4d} -> {status} en iteracion {it:3d} | Error L1 final: {err:.2e}")
    figs4 = plot_experiment_results(res4, "max_iterations", output_dir=OUTPUT_DIR)
    print(f"   -> {len(figs4)} gráficos guardados en '{OUTPUT_DIR}'")

    # ----------------------------------------------------------------
    # Resumen final
    # ----------------------------------------------------------------
    total_figs = len(figs1) + len(figs2) + len(figs3) + len(figs4)
    print("\n" + "=" * 65)
    print(f"  [OK] Todos los experimentos completados. {total_figs} graficos generados.")
    print(f"  [OK] Figuras guardadas en: {OUTPUT_DIR}")
    print("=" * 65)
    plt.show()
