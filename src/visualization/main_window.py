from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QLabel,
    QTextEdit,
    QSplitter,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QPushButton
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
from pathlib import Path
from src.data_ingestion.csv_loader import load_csv
from src.visualization.chart_canvas import ChartCanvas
from src.visualization.chart_canvas import ChartCanvas
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Series Analysis Framework")
        self.setGeometry(50, 50, 1200, 800)

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        add_series_action = QAction("Add Series", self)
        add_series_action.triggered.connect(self.add_series)
        toolbar.addAction(add_series_action)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        container = QWidget()
        main_layout = QHBoxLayout(container)
        main_layout.addWidget(main_splitter)
        self.setCentralWidget(container)

        self.sidebar = QTreeWidget()
        self.sidebar.setHeaderHidden(True)
        self.sidebar.setFixedWidth(300)
        methods = {
            "Data View": ["Show Raw Data", "Show Summary"],
            "Time Series Analysis": ["Rolling Mean", "Differencing", "Seasonal Decompose", "Autocorrelation"],
            "Forecasting": ["ARIMA", "Prophet"],
            "Anomaly Detection": ["Isolation Forest", "Z-Score"],
            "AI Analysis": ["Generate Insights"]
        }
        for section, items in methods.items():
            parent = QTreeWidgetItem(self.sidebar, [section])
            for item_text in items:
                QTreeWidgetItem(parent, [item_text])
            parent.setExpanded(False)
        self.sidebar.itemClicked.connect(self.on_sidebar_click)
        main_splitter.addWidget(self.sidebar)

        self.tab_widget = QTabWidget()
        main_splitter.addWidget(self.tab_widget)

        self.series_frames = {}
        self.series_selected = {}
        self.result_widgets = {}

    def add_series(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add CSV Series", "", "CSV Files (*.csv)")
        for path in paths:
            df = load_csv(path)
            cols = self.select_column(df)
            if cols:
                tab_name = Path(path).stem
                self._create_tab(tab_name, df, cols)

    def select_column(self, df):
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Column to Plot")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Select one column to plot against time index:"))
        listw = QListWidget(dlg)
        listw.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        for col in df.columns:
            item = QListWidgetItem(col)
            listw.addItem(item)
        layout.addWidget(listw)
        ok_btn = QPushButton("OK", dlg)
        ok_btn.clicked.connect(dlg.accept)
        layout.addWidget(ok_btn)
        dlg.exec()
        current = listw.currentItem()
        return [current.text()] if current else []

    def _create_tab(self, name, df, cols):
        tab = QWidget()
        splitter = QSplitter(Qt.Orientation.Vertical, tab)

        chart = ChartCanvas(splitter)
        chart.plot_data(df, cols)

        result = QTextEdit(splitter)
        result.setReadOnly(True)
        result.hide()

        layout = QVBoxLayout(tab)
        layout.addWidget(splitter)

        self.series_frames[name] = df
        self.series_selected[name] = cols
        self.result_widgets[name] = result

        self.tab_widget.addTab(tab, name)

    def on_sidebar_click(self, item, _):
        if item.childCount() > 0:
            return
        method = item.text(0)
        idx = self.tab_widget.currentIndex()
        if idx < 0:
            return
        tab_name = self.tab_widget.tabText(idx)
        df = self.series_frames.get(tab_name)
        cols = self.series_selected.get(tab_name)
        tab = self.tab_widget.widget(idx)
        chart = tab.findChild(ChartCanvas)
        result = self.result_widgets.get(tab_name)
        if df is None or not cols or not chart:
            return

        data = df[cols]
        if method == "Rolling Mean":
            chart.plot_data(data.rolling(5).mean(), cols)
            result.hide()
        elif method == "Differencing":
            chart.plot_data(data.diff().dropna(), cols)
            result.hide()
        elif method == "Show Raw Data":
            result.setText(data.to_string())
            result.show()
        elif method == "Show Summary":
            result.setText(data.describe().to_string())
            result.show()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
