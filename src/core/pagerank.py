def pagerank(grafo, d=0.85, iteraciones=20):
    paginas = list(grafo.keys())
    N = len(paginas)

    # Inicialización
    pr = {p: 1 / N for p in paginas}

    for _ in range(iteraciones):
        nuevo_pr = {}

        for pagina in paginas:
            suma = 0

            # Buscar quién apunta a esta página
            for otra in paginas:
                if pagina in grafo[otra]:
                    suma += pr[otra] / len(grafo[otra])

            nuevo_pr[pagina] = (1 - d) / N + d * suma

        pr = nuevo_pr

    return pr


grafo = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C"]
}

resultado = pagerank(grafo)

for pagina, valor in resultado.items():
    print(f"{pagina}: {valor:.4f}")