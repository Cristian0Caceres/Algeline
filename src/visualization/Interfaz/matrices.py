from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView
)

class MatricesWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Tabla 1: Matriz de Adyacencia
        layout.addWidget(QLabel("Matriz de Adyacencia (W):"))
        self.tabla_adyacencia = QTableWidget()
        self.tabla_adyacencia.setAlternatingRowColors(True)
        layout.addWidget(self.tabla_adyacencia)

        # Tabla 2: Rankings de PageRank
        layout.addWidget(QLabel("Rankings de Importancia (PageRank):"))
        self.tabla_rankings = QTableWidget()
        self.tabla_rankings.setColumnCount(3)
        self.tabla_rankings.setHorizontalHeaderLabels(["Pos.", "Nodo ID", "PageRank"])
        self.tabla_rankings.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_rankings.setAlternatingRowColors(True)
        layout.addWidget(self.tabla_rankings)

    def llenar_tabla_adyacencia(self, W_adj, nodos):
        n = len(nodos)
        self.tabla_adyacencia.setRowCount(n)
        self.tabla_adyacencia.setColumnCount(n)
        self.tabla_adyacencia.setHorizontalHeaderLabels(nodos)
        self.tabla_adyacencia.setVerticalHeaderLabels(nodos)
        self.tabla_adyacencia.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        for i in range(n):
            for j in range(n):
                valor = int(W_adj[i, j])
                item = QTableWidgetItem(str(valor))
                item.setTextAlignment(Qt.AlignCenter)
                self.tabla_adyacencia.setItem(i, j, item)

    def llenar_tabla_rankings(self, rankings_calculados):
        self.tabla_rankings.setRowCount(0)
        for idx, (nodo, score) in enumerate(rankings_calculados):
            row_pos = self.tabla_rankings.rowCount()
            self.tabla_rankings.insertRow(row_pos)
            
            item_pos = QTableWidgetItem(str(idx + 1))
            item_nodo = QTableWidgetItem(str(nodo))
            item_score = QTableWidgetItem(f"{score:.5f}")
            
            item_pos.setTextAlignment(Qt.AlignCenter)
            item_nodo.setTextAlignment(Qt.AlignCenter)
            item_score.setTextAlignment(Qt.AlignCenter)
            
            self.tabla_rankings.setItem(row_pos, 0, item_pos)
            self.tabla_rankings.setItem(row_pos, 1, item_nodo)
            self.tabla_rankings.setItem(row_pos, 2, item_score)

    def limpiar_rankings(self):
        self.tabla_rankings.setRowCount(0)
