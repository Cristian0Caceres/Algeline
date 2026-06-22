import sys
import os
import networkx as nx
import numpy as np

# Agregar el directorio raíz al path para resolver las importaciones del core y visualization
sys.path.append(".")

from src.core import graph as core_graph
from src.core import transition_matrix as core_tm
from src.core import pagerank as core_pr

# Importar widgets de la interfaz
try:
    from barra_de_opciones import BarraDeOpcionesWidget
    from grafico import GraficoWidget
    from matrices import MatricesWidget
except (ImportError, ModuleNotFoundError):
    from src.visualization.Interfaz.barra_de_opciones import BarraDeOpcionesWidget
    from src.visualization.Interfaz.grafico import GraficoWidget
    from src.visualization.Interfaz.matrices import MatricesWidget

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QMessageBox
)

class PageRankColumnApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Algeline PageRank - Layout de Columna")
        self.resize(1000, 650)

        # Variables internas
        self.ruta_archivo = None
        self.grafo = None
        self.matriz_transicion = None
        self.nodos_nombres = None

        self.setup_ui()

    def setup_ui(self):
        # Widget Central y Layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)

        # 1. BARRA DE OPCIONES (Lateral izquierdo)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(15)
        
        title_label = QLabel("PageRank Visualizer")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        sidebar_layout.addWidget(title_label)

        # Barra de opciones (vertical)
        self.barra_opciones = BarraDeOpcionesWidget(horizontal=False)
        self.barra_opciones.archivo_cargado.connect(self.cargar_archivo)
        self.barra_opciones.calcular_solicitado.connect(self.calcular_pagerank)
        sidebar_layout.addWidget(self.barra_opciones)
        
        sidebar_layout.addStretch()
        
        sidebar_container = QWidget()
        sidebar_container.setLayout(sidebar_layout)
        sidebar_container.setFixedWidth(240)
        main_layout.addWidget(sidebar_container)

        # 2. CONTENEDOR DE PESTAÑAS (Central)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Pestaña 1: Gráfico
        self.grafico_widget = GraficoWidget()
        self.tabs.addTab(self.grafico_widget, "Gráfico")

        # Pestaña 2: Matrices
        self.matrices_widget = MatricesWidget()
        self.tabs.addTab(self.matrices_widget, "Matrices")

    def cargar_archivo(self, ruta):
        try:
            self.ruta_archivo = ruta
            self.grafo = core_graph.cargar_grafo(ruta)
            self.matriz_transicion, self.nodos_nombres = core_tm.crear_matriz(self.grafo)
            
            # Adyacencia
            W_adj, nodos = core_graph.obtener_matriz_adyacencia(self.grafo)
            self.matrices_widget.llenar_tabla_adyacencia(W_adj, nodos)
            
            # Actualizar barra opciones
            nombre_archivo = os.path.basename(ruta)
            self.barra_opciones.set_archivo_label(nombre_archivo)
            self.barra_opciones.set_calcular_enabled(True)
            self.barra_opciones.set_stats_text("Iteraciones: -")
            self.matrices_widget.limpiar_rankings()
            
            # Dibujar grafo estático
            self.grafico_widget.dibujar_grafo(self.grafo, None)
        except Exception as e:
            QMessageBox.critical(self, "Error de Lectura", f"No se pudo cargar la red:\n{str(e)}")

    def calcular_pagerank(self, damping, max_iters, tol):
        if self.grafo is None or self.matriz_transicion is None:
            return

        try:
            vector_pr, iters = core_pr.pagerank(
                self.matriz_transicion, 
                iteraciones=max_iters, 
                damping=damping, 
                tolerancia=tol
            )
            rankings_calculados = core_pr.obtener_ranking_ordenado(vector_pr, self.nodos_nombres)
            
            if iters < max_iters:
                self.barra_opciones.set_stats_text(f"Iteraciones: {iters} (Convergió)")
            else:
                self.barra_opciones.set_stats_text(f"Iteraciones: {iters} (Límite Máx)")
                
            self.matrices_widget.llenar_tabla_rankings(rankings_calculados)
            self.grafico_widget.dibujar_grafo(self.grafo, dict(rankings_calculados))
        except Exception as e:
            QMessageBox.critical(self, "Error de Cálculo", f"Error al procesar PageRank:\n{str(e)}")

def main():
    app = QApplication(sys.argv)
    window = PageRankColumnApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
