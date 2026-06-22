import sys
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Agregar path para imports del visualizador modular
sys.path.append(".")
from src.visualization import plot_graph

class GraficoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figura = Figure()
        self.canvas = FigureCanvas(self.figura)
        layout.addWidget(self.canvas)

        self.dibujar_bienvenida()

    def dibujar_bienvenida(self):
        self.figura.clear()
        ax = self.figura.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5, 0.5, 
            "Carga un archivo de red JSON para comenzar la visualización", 
            ha="center", va="center", 
            color="gray", fontsize=12, 
            fontweight="bold"
        )
        self.canvas.draw()

    def dibujar_grafo(self, grafo, ranking_dict=None):
        self.figura.clear()
        ax = self.figura.add_subplot(111)
        plot_graph.dibujar_red_en_eje(grafo, ax, ranking_dict)
        self.figura.tight_layout()
        self.canvas.draw()
