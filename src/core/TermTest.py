import numpy as np
from pathlib import Path
from datetime import datetime
from graph import cargar_grafo
from transition_matrix import crear_matriz
from pagerank import pagerank

RUTA_JSON = Path(__file__).resolve().parent / "datos.json"
RUTA_RESULTADO = Path(__file__).resolve().parent / "test_results.txt"

PASS = "PASS"
FAIL = "FAIL"

lineas = []

def test(nombre, condicion):
    estado = PASS if condicion else FAIL
    linea = f"[{estado}] {nombre}"
    print(linea)
    lineas.append(linea)
    return condicion

def separador(titulo):
    bloque = f"\n{'='*40}\n  {titulo}\n{'='*40}"
    print(bloque)
    lineas.append(bloque)

resultados = []

separador("graph.py")

grafo = cargar_grafo(RUTA_JSON)

resultados.append(test("Grafo no es None",                   grafo is not None))
resultados.append(test("Grafo tiene 3 nodos",                grafo.number_of_nodes() == 3))
resultados.append(test("Grafo tiene 4 aristas",              grafo.number_of_edges() == 4))
resultados.append(test("Nodo A existe",                      "A" in grafo.nodes()))
resultados.append(test("Arista A->B existe",                 grafo.has_edge("A", "B")))
resultados.append(test("Arista inexistente B->A no existe",  not grafo.has_edge("B", "A")))

separador("transition_matrix.py")

M, nodos = crear_matriz(grafo)

resultados.append(test("Retorna matriz y lista de nodos",    isinstance(M, np.ndarray) and isinstance(nodos, list)))
resultados.append(test("Matriz es cuadrada",                 M.shape[0] == M.shape[1]))
resultados.append(test("Dimensión coincide con nodos",       M.shape[0] == len(nodos)))
resultados.append(test("Columnas suman 1 (estocástica)",     np.allclose(M.sum(axis=0), 1)))
resultados.append(test("No hay valores negativos",           np.all(M >= 0)))

grafo_dangling = cargar_grafo(RUTA_JSON)
grafo_dangling.add_node("D")
M_d, nodos_d = crear_matriz(grafo_dangling)
resultados.append(test("Nodo dangling: columna suma 1",      np.allclose(M_d.sum(axis=0), 1)))

separador("pagerank.py")

PR, iters = pagerank(M, iteraciones=100, damping=0.85, tolerancia=1e-6)

resultados.append(test("Retorna vector y número de iteraciones", isinstance(PR, np.ndarray) and isinstance(iters, int)))
resultados.append(test("Vector PR suma 1",                   np.isclose(PR.sum(), 1)))
resultados.append(test("Todos los valores son positivos",    np.all(PR > 0)))
resultados.append(test("Convergió antes del máximo",         iters < 100))
resultados.append(test("Dimensión de PR coincide con nodos", len(PR) == len(nodos)))

PR_verbose, _ = pagerank(M, verbose=False)
resultados.append(test("verbose=False no rompe nada",        PR_verbose is not None))

try:
    pagerank(np.ones((2, 3)))
    resultados.append(test("Matriz no cuadrada lanza ValueError", False))
except ValueError:
    resultados.append(test("Matriz no cuadrada lanza ValueError", True))

PR_d100, _ = pagerank(M, damping=1.0)
resultados.append(test("damping=1.0 no rompe nada",          PR_d100 is not None))

PR_d0, _ = pagerank(M, damping=0.0)
resultados.append(test("damping=0.0 → distribución uniforme", np.allclose(PR_d0, 1 / M.shape[0])))

separador("RESUMEN")
total  = len(resultados)
passed = sum(resultados)
failed = total - passed
resumen = f"  Total : {total}\n  PASS  : {passed}\n  FAIL  : {failed}\n"
print(resumen)
lineas.append(resumen)
veredicto = "  Todo correcto." if failed == 0 else "  Hay errores que revisar."
print(veredicto)
lineas.append(veredicto)

with open(RUTA_RESULTADO, "w", encoding="utf-8") as f:
    f.write(f"Test ejecutado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("\n".join(lineas))

print(f"\nResultados guardados en: {RUTA_RESULTADO}")