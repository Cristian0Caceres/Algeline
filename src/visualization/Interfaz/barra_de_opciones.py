import os
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QFrame
)

class BarraDeOpcionesWidget(QWidget):
    # Señales para comunicar eventos a la ventana principal
    archivo_cargado = Signal(str)  # Emite la ruta del archivo cargado
    calcular_solicitado = Signal(float, int, float)  # Emite (damping, max_iters, tol)

    def __init__(self, horizontal=True, parent=None):
        super().__init__(parent)
        self.horizontal = horizontal
        self.setup_ui()

    def setup_ui(self):
        if self.horizontal:
            # Layout horizontal para el diseño de fila
            self.layout_principal = QHBoxLayout(self)
            self.layout_principal.setContentsMargins(10, 5, 10, 5)
            self.layout_principal.setSpacing(15)
        else:
            # Layout vertical para el diseño de columna (sidebar)
            self.layout_principal = QVBoxLayout(self)
            self.layout_principal.setContentsMargins(0, 0, 0, 0)
            self.layout_principal.setSpacing(15)

        # Botón y etiqueta de carga de archivos
        self.btn_cargar = QPushButton("Cargar JSON")
        self.btn_cargar.clicked.connect(self.cargar_archivo_dialog)
        self.layout_principal.addWidget(self.btn_cargar)

        self.label_archivo = QLabel("Sin archivo cargado")
        if self.horizontal:
            self.label_archivo.setFixedWidth(150)
        else:
            self.label_archivo.setWordWrap(True)
        self.layout_principal.addWidget(self.label_archivo)

        # Separador (solo para layout horizontal)
        if self.horizontal:
            self.layout_principal.addWidget(self.crear_separador_vertical())

        # Controles de parámetros
        # Damping (d)
        self.lbl_damping = QLabel("d (Amortiguamiento):")
        self.spin_damping = QDoubleSpinBox()
        self.spin_damping.setRange(0.0, 1.0)
        self.spin_damping.setSingleStep(0.05)
        self.spin_damping.setValue(0.85)
        self.spin_damping.setFixedWidth(65)
        
        if self.horizontal:
            self.layout_principal.addWidget(self.lbl_damping)
            self.layout_principal.addWidget(self.spin_damping)
        else:
            # En vertical usamos un layout de formulario simple
            self.add_parametro_vertical(self.lbl_damping, self.spin_damping)

        # Iteraciones máximas
        self.lbl_iters = QLabel("Iter. Máx:")
        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(1, 1000)
        self.spin_iters.setValue(100)
        self.spin_iters.setFixedWidth(60)
        
        if self.horizontal:
            self.layout_principal.addWidget(self.lbl_iters)
            self.layout_principal.addWidget(self.spin_iters)
        else:
            self.add_parametro_vertical(self.lbl_iters, self.spin_iters)

        # Tolerancia
        self.lbl_tol = QLabel("Tolerancia (ε):")
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(1e-8, 1e-1)
        self.spin_tol.setDecimals(6)
        self.spin_tol.setValue(1e-6)
        self.spin_tol.setFixedWidth(90)
        
        if self.horizontal:
            self.layout_principal.addWidget(self.lbl_tol)
            self.layout_principal.addWidget(self.spin_tol)
        else:
            self.add_parametro_vertical(self.lbl_tol, self.spin_tol)

        # Separador (solo para layout horizontal)
        if self.horizontal:
            self.layout_principal.addWidget(self.crear_separador_vertical())

        # Botón Calcular
        self.btn_calcular = QPushButton("Calcular")
        self.btn_calcular.setEnabled(False)
        self.btn_calcular.clicked.connect(self.emitir_calcular)
        self.layout_principal.addWidget(self.btn_calcular)

        # Estadísticas
        self.label_stats = QLabel("Iteraciones: -")
        self.layout_principal.addWidget(self.label_stats)

        if self.horizontal:
            self.layout_principal.addStretch()

    def crear_separador_vertical(self):
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def add_parametro_vertical(self, label, widget):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(widget)
        self.layout_principal.addWidget(row)

    def cargar_archivo_dialog(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, 
            "Seleccionar Red JSON", 
            "data", 
            "Archivos JSON (*.json)"
        )
        if ruta:
            # Emitir la ruta para que la ventana principal cargue el grafo
            self.archivo_cargado.emit(ruta)

    def emitir_calcular(self):
        damping = self.spin_damping.value()
        max_iters = self.spin_iters.value()
        tol = self.spin_tol.value()
        self.calcular_solicitado.emit(damping, max_iters, tol)

    # Métodos expuestos para actualizar el estado del widget
    def set_archivo_label(self, texto):
        self.label_archivo.setText(texto)

    def set_calcular_enabled(self, enabled):
        self.btn_calcular.setEnabled(enabled)

    def set_stats_text(self, texto, color=None):
        self.label_stats.setText(texto)
        if color:
            self.label_stats.setStyleSheet(f"color: {color};")
        else:
            self.label_stats.setStyleSheet("")
