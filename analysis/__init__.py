"""Módulo de Análisis de PageRank.

Este paquete proporciona herramientas para medir y analizar la convergencia del
algoritmo PageRank, así como para ejecutar experimentos variando la estructura del
grafo y los hiperparámetros del algoritmo.
"""

from analysis.convergence import (
    calculate_l1_norm,
    calculate_l2_norm,
    calculate_max_absolute_difference,
    check_convergence,
    plot_convergence_history,
)

from analysis.experiments import (
    generate_directed_graph,
    run_pagerank_solver,
    experiment_node_scaling,
    experiment_edge_density,
    experiment_damping_factor,
    experiment_max_iterations,
    plot_experiment_results,
)

__all__ = [
    "calculate_l1_norm",
    "calculate_l2_norm",
    "calculate_max_absolute_difference",
    "check_convergence",
    "plot_convergence_history",
    "generate_directed_graph",
    "run_pagerank_solver",
    "experiment_node_scaling",
    "experiment_edge_density",
    "experiment_damping_factor",
    "experiment_max_iterations",
    "plot_experiment_results",
]
