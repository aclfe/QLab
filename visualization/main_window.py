from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QWidget,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QToolBar,
)
from PyQt6.QtGui import QAction
from data_ingestion.csv_loader import load_csv
from visualization.chart_canvas import ChartCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Series Analysis Framework")
        self.setGeometry(50, 50, 1200, 600)

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        open_action = QAction("Open CSV", self)
        open_action.triggered.connect(self.open_csv)
        toolbar.addAction(open_action)

        main_layout = QHBoxLayout()

        self.sidebar = QTreeWidget()
        self.sidebar.setHeaderHidden(True)
        self.sidebar.setFixedWidth(250)
        sections = {
            "Data View": ["Show Raw Data", "Show Summary"],
            "Time Series Analysis": ["Rolling Mean", "Differencing", "Seasonal Decompose", "Autocorrelation"],
            "Forecasting": ["ARIMA", "Prophet"],
            "Anomaly Detection": ["Isolation Forest", "Z-Score"]
        }
        for section, methods in sections.items():
            parent = QTreeWidgetItem(self.sidebar, [section])
            for m in methods:
                QTreeWidgetItem(parent, [m])
            parent.setExpanded(False)
        self.sidebar.setStyleSheet(
            "QTreeWidget { background-color: #2e2e2e; color: white; border: none; }"
            "QTreeWidget::item { padding: 5px 10px; }"
            "QTreeWidget::item:selected { background-color: #444444; }"
        )
        self.sidebar.itemClicked.connect(self.on_sidebar_click)

        self.canvas = ChartCanvas(self)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.canvas, stretch=1)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.current_df = None

    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            df = load_csv(path)
            self.current_df = df
            self.canvas.plot_data(df)

    def on_sidebar_click(self, item, column):
        if not item.childCount():
            method = item.text(0)
            if self.current_df is None:
                return
            if method == "Rolling Mean":
                df_roll = self.current_df.rolling(window=5).mean()
                self.canvas.plot_data(df_roll)
            elif method == "Differencing":
                df_diff = self.current_df.diff().dropna()
                self.canvas.plot_data(df_diff)
            elif method == "Show Raw Data":
                self.canvas.plot_table(self.current_df)
            elif method == "Show Summary":
                summary = self.current_df.describe()
                self.canvas.plot_table(summary)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()