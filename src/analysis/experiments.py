import sys
import os
import networkx as nx
import numpy as np

# Asegurar que el directorio raíz está en el PATH
sys.path.append(".")
from src.core.graph import cargar_grafo
from src.core.transition_matrix import crear_matriz
from src.analysis.convergence import analizar_convergencia

# =====================================================================
# Funciones Generadoras de Topologías Sintéticas (Experimento 2)
# =====================================================================
def generar_red_lineal(n: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G

def generar_red_estrella(n: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # Nodos periféricos apuntan al centro (nodo 0)
    for i in range(1, n):
        G.add_edge(i, 0)
    return G

def generar_red_ciclica(n: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
    return G

def generar_red_aleatoria(n: int, m: int, seed: int = 42) -> nx.DiGraph:
    return nx.gnm_random_graph(n, m, directed=True, seed=seed)

# =====================================================================
# Experimento 1: Variar número de nodos (pequeña / mediana / grande)
# =====================================================================
def correr_experimento_nodos(rutas_redes: dict[str, str]) -> dict[str, dict]:
    resultados = {}
    for nombre, ruta in rutas_redes.items():
        if os.path.exists(ruta):
            grafo = cargar_grafo(ruta)
            M, nodos = crear_matriz(grafo)
            PR, diffs, iters = analizar_convergencia(M)
            resultados[nombre] = {
                "nodos": len(nodos),
                "enlaces": grafo.number_of_edges(),
                "iteraciones": iters,
                "diferencias": diffs
            }
        else:
            print(f"Advertencia: No se encontró la ruta {ruta}")
    return resultados

# =====================================================================
# Experimento 2: Variar estructura de enlaces (lineal, estrella, cíclica, aleatoria)
# =====================================================================
def correr_experimento_estructuras(n: int = 10) -> dict[str, dict]:
    estructuras = {
        "lineal": generar_red_lineal(n),
        "estrella": generar_red_estrella(n),
        "ciclica": generar_red_ciclica(n),
        "aleatoria": generar_red_aleatoria(n, 2 * n, seed=42)
    }
    
    resultados = {}
    for nombre, grafo in estructuras.items():
        M, nodos = crear_matriz(grafo)
        PR, diffs, iters = analizar_convergencia(M)
        resultados[nombre] = {
            "nodos": len(nodos),
            "enlaces": grafo.number_of_edges(),
            "iteraciones": iters,
            "diferencias": diffs
        }
    return resultados

# =====================================================================
# Experimento 3: Variar cantidad de iteraciones (error L1 contra convergencia alta)
# =====================================================================
def correr_experimento_iteraciones(matriz: np.ndarray, iteraciones_lista: list[int] = [5, 20, 50, 100]) -> dict[int, float]:
    # Vector PageRank de referencia (tolerancia muy estricta)
    PR_ref, _, _ = analizar_convergencia(matriz, tolerancia=1e-12, max_iters=1000)
    
    resultados = {}
    for limite in iteraciones_lista:
        # tolerancia=0 fuerza a iterar exactamente 'limite' veces
        PR_lim, _, _ = analizar_convergencia(matriz, max_iters=limite, tolerancia=0)
        error_l1 = float(np.linalg.norm(PR_lim - PR_ref, 1))
        resultados[limite] = error_l1
    return resultados

# =====================================================================
# Experimento 4: Variar Damping Factor (0.5, 0.75, 0.85, 0.95)
# =====================================================================
def correr_experimento_damping(matriz: np.ndarray, damping_lista: list[float] = [0.5, 0.75, 0.85, 0.95]) -> dict[float, dict]:
    resultados = {}
    for d in damping_lista:
        PR, diffs, iters = analizar_convergencia(matriz, damping=d)
        resultados[d] = {
            "iteraciones": iters,
            "diferencias": diffs
        }
    return resultados

# =====================================================================
# Ejecución general de prueba de experimentos
# =====================================================================
def ejecutar_todos_los_experimentos():
    print("=" * 60)
    print("  EJECUTANDO EXPERIMENTOS DE PAGERANK")
    print("=" * 60)
    
    rutas = {
        "pequeña": "data/red_pequena.json",
        "mediana": "data/red_mediana.json",
        "grande": "data/red_grande.json"
    }

    # Experimento 1
    print("\n--- Experimento 1: Variación del Número de Nodos ---")
    res_nodos = correr_experimento_nodos(rutas)
    for k, v in res_nodos.items():
        print(f"Red {k:8s} | Nodos: {v['nodos']:3d} | Enlaces: {v['enlaces']:3d} | Iteraciones: {v['iteraciones']:3d}")

    # Experimento 2
    print("\n--- Experimento 2: Estructuras de Enlaces (N = 10) ---")
    res_est = correr_experimento_estructuras(10)
    for k, v in res_est.items():
        print(f"Topología {k:9s} | Iteraciones para converger: {v['iteraciones']:3d}")

    # Experimento 3 (usando la red mediana)
    print("\n--- Experimento 3: Cantidad de Iteraciones vs Error L1 (Red Mediana) ---")
    if os.path.exists(rutas["mediana"]):
        G_med = cargar_grafo(rutas["mediana"])
        M_med, _ = crear_matriz(G_med)
        res_iters = correr_experimento_iteraciones(M_med, [5, 20, 50, 100])
        for limite, err in res_iters.items():
            print(f"Límite Iteraciones: {limite:3d} | Error L1 respecto al valor real: {err:.8e}")

    # Experimento 4 (usando la red mediana)
    print("\n--- Experimento 4: Variación de Damping Factor (Red Mediana) ---")
    if os.path.exists(rutas["mediana"]):
        res_damping = correr_experimento_damping(M_med, [0.5, 0.75, 0.85, 0.95])
        for d, v in res_damping.items():
            print(f"Damping d: {d:.2f} | Iteraciones para converger: {v['iteraciones']:3d}")
    print("=" * 60)

if __name__ == "__main__":
    ejecutar_todos_los_experimentos()