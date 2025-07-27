import sys
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QFileDialog,
    QWidget, QHBoxLayout, QVBoxLayout, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
from data_ingestion.csv_loader import load_csv
from visualization.plot_widget import PlotWidget
from visualization.plot_manager import PlotManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TimeSeries Analysis Window")
        self._apply_dark_theme()
        self.resize(1200, 800)

        file_menu = self.menuBar().addMenu("File")
        open_action = file_menu.addAction("Open CSV")
        open_action.triggered.connect(self.open_csv)

        container = QWidget()
        main_layout = QHBoxLayout()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.sidebar = QTreeWidget()
        self.sidebar.setHeaderHidden(True)
        self.sidebar.setFixedWidth(250)
        for section, children in {
            "Analysis": ["Summary", "Statistics"],
            "Indicators": ["EMA", "RSI", "MACD"],
            "Forecast": ["Prophet", "ARIMA"],
            "Anomalies": ["IsolationForest", "Autoencoder"]
        }.items():
            parent = QTreeWidgetItem(self.sidebar, [section])
            for child in children:
                QTreeWidgetItem(parent, [child])
            parent.setExpanded(False)
        self.sidebar.setStyleSheet(
            "QTreeWidget { background-color: #111111; color: #FFFFFF; border: none; }"
            "QTreeWidget::item { padding: 8px; }"
            "QTreeWidget::item:selected { background-color: #333333; }"
        )
        main_layout.addWidget(self.sidebar)

        self.plot_widget = PlotWidget()
        main_layout.addWidget(self.plot_widget, stretch=1)

        self.plot_manager = PlotManager(self.plot_widget)

    def _apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#000000"))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor("#222222"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#333333"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor("#111111"))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.setPalette(palette)

    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV file", filter="CSV Files (*.csv)")
        if path:
            df = load_csv(path, date_col=None, index_col=None)
            self.plot_manager.render(df)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
