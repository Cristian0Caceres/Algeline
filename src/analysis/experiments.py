from ..core.graph import cargar_grafo
from ..core.transition_matrix import crear_matriz
from ..core.pagerank import pagerank


def experimento_damping(
    ruta_json: str,
    valores=(0.50, 0.70, 0.85, 0.95)
):
    resultados = []

    grafo = cargar_grafo(ruta_json)
    matriz, nodos = crear_matriz(grafo)

    for damping in valores:

        ranking, iteraciones = pagerank(
            matriz,
            damping=damping
        )

        resultados.append({
            "damping": damping,
            "iteraciones": iteraciones,
            "ranking": ranking.tolist()
        })

    return resultados


def experimento_iteraciones(
    ruta_json: str,
    iteraciones=(10, 25, 50, 100)
):
    resultados = []

    grafo = cargar_grafo(ruta_json)
    matriz, nodos = crear_matriz(grafo)

    for n_iter in iteraciones:

        ranking, _ = pagerank(
            matriz,
            iteraciones=n_iter
        )

        resultados.append({
            "iteraciones": n_iter,
            "ranking": ranking.tolist()
        })

    return resultados


def experimento_redes(rutas_json: list[str]):
    resultados = []

    for ruta in rutas_json:

        grafo = cargar_grafo(ruta)

        matriz, nodos = crear_matriz(grafo)

        ranking, iteraciones = pagerank(matriz)

        resultados.append({
            "archivo": ruta,
            "nodos": len(nodos),
            "iteraciones": iteraciones,
            "ranking": ranking.tolist()
        })

    return resultados