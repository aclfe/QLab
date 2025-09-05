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
from src.timeseries_methods.methods import rolling_mean, differencing, seasonal_decompose
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Series Analysis Framework")
        self.setGeometry(50, 50, 1200, 600)

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        add_series_action = QAction("Add Series", self)
        add_series_action.triggered.connect(self.add_series)
        toolbar.addAction(add_series_action)

        self.show_grids_action = QAction("Show Grids", self)
        self.show_grids_action.setCheckable(True)
        self.show_grids_action.setChecked(False)
        self.show_grids_action.toggled.connect(self.toggle_grids)
        toolbar.addAction(self.show_grids_action)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        container = QWidget()
        main_layout = QHBoxLayout(container)
        main_layout.addWidget(main_splitter)
        self.setCentralWidget(container)

        self.sidebar = QTreeWidget()
        self.sidebar.setHeaderHidden(True)
        self.sidebar.setFixedWidth(300)
        methods = {
            "Data View": ["Show Raw Data", "Show Summary", "Show Original Graph"],
            "Time Series Analysis": ["Rolling Mean", "Differencing", "Seasonal Decompose"],
            "Forecasting": ["ARIMA", "Prophet"],
            "Anomaly Detection": ["Isolation Forest", "Z-Score"]
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

        view3d_btn = QPushButton("View in 3D", tab)
        view3d_btn.clicked.connect(lambda: self._show_3d_view(name))
        view3d_btn.setFixedWidth(100)
        view3d_btn.setStyleSheet("background-color: #2d2d2d; color: white;")

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(view3d_btn)
        layout.addLayout(btn_layout)

        self.series_frames[name] = df
        self.series_selected[name] = cols
        self.result_widgets[name] = result

        if not hasattr(self, "original_data"): 
            self.original_data = {}
        self.original_data[name] = (df.copy(), cols.copy())

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
            chart.plot_data(rolling_mean(data), cols)
            result.hide()
        elif method == "Differencing":
            chart.plot_data(differencing(data), cols)
            result.hide()
        elif method == "Seasonal Decompose":
            chart.plot_data(seasonal_decompose(data), ['trend', 'seasonal', 'resid'])
            result.hide()
        elif method == "Show Raw Data":
            result.setText(data.to_string())
            result.show()
        elif method == "Show Summary":
            result.setText(data.describe().to_string())
            result.show()
        elif method == "Show Original Graph":
            if hasattr(self, "original_data") and tab_name in self.original_data:
                orig_df, orig_cols = self.original_data[tab_name]
                chart.plot_data(orig_df, orig_cols)
                result.hide()

    def toggle_grids(self, checked):
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            chart = tab.findChild(ChartCanvas)
            if chart:
                chart.set_grid(checked)

    def _show_3d_view(self, tab_name):
        df = self.series_frames.get(tab_name)
        if df is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Second Column for 3D View")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Select another column to plot in 3D:"))
        listw = QListWidget(dlg)
        listw.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
    
        current_col = self.series_selected[tab_name][0]
        for col in df.columns:
            if col != current_col:
                item = QListWidgetItem(col)
                listw.addItem(item)
        
        layout.addWidget(listw)
        ok_btn = QPushButton("OK", dlg)
        ok_btn.clicked.connect(dlg.accept)
        layout.addWidget(ok_btn)
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            current = listw.currentItem()
            if current:
                second_col = current.text()
                tab = self.tab_widget.widget(self.tab_widget.currentIndex())
                chart = tab.findChild(ChartCanvas)
                if chart:
                    chart.plot_3d_data(df, self.series_selected[tab_name][0], second_col)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
