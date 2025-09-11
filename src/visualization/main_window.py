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
    QPushButton,
    QSpinBox
)

from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QRunnable, QThreadPool, QObject, pyqtSignal
from pathlib import Path
from src.data_ingestion.csv_loader import load_csv
from src.visualization.chart_canvas import ChartCanvas
from src.preprocessing.methods import (
    rolling_mean,
    differencing,
    ewma,
    log_transform,
    boxcox_transform,
    boxcox_inverse,
    standard_scale,
    minmax_scale,
)
from src.analysis.methods import (
    plot_distribution,
    plot_before_after,
    seasonal_decompose,
    build_seasonal_decomposition_figure,
    prepare_acf_pacf_data,
    build_acf_pacf_figure,
    stationarity_tests,
)
from src.filters.methods import (
    hp_filter_decompose,
    build_hp_filter_figure,
)
from src.forecasting.methods import (
    arima_forecast_with_ci,
    sarimax_forecast_with_ci,
    holt_winters_forecast,
    prophet_forecast,
)
from src.anomaly_detection.methods import (
    zscore_anomalies,
    isolation_forest_anomalies,
)
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Series Analysis Framework")
        self.setGeometry(50, 50, 1200, 600)
        self._thread_pool = QThreadPool.globalInstance()

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
            "Preprocessing": [
                "Rolling Mean",
                "Differencing",
                "EWMA",
                "Log Transform",
                "Box-Cox Transform",
                "Standard Scale (Z)",
                "MinMax Scale"
            ],
            "Analysis": [
                "Seasonal Decompose",
                "Show ACF/PACF",
                "Stationarity Tests"
            ],
            "Forecasting": ["ARIMA", "SARIMAX", "Holt-Winters", "Prophet"],
            "Filters": ["HP Filter"],
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
            try:
                dlg = QDialog(self)
                dlg.setWindowTitle("Rolling Mean Parameters")
                layout = QVBoxLayout(dlg)
                layout.addWidget(QLabel("Window size:"))
                win_spin = QSpinBox(dlg); win_spin.setRange(1, 100); win_spin.setValue(5); layout.addWidget(win_spin)
                btn_layout = QHBoxLayout(); ok_btn = QPushButton("OK", dlg); cancel_btn = QPushButton("Cancel", dlg)
                ok_btn.clicked.connect(dlg.accept); cancel_btn.clicked.connect(dlg.reject)
                btn_layout.addWidget(ok_btn); btn_layout.addWidget(cancel_btn); layout.addLayout(btn_layout)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                window = int(win_spin.value())
                self._run_async(lambda: rolling_mean(data, window=window), lambda r: (chart.plot_data(r, cols), result.hide()), result)
            except Exception as e:
                result.setText(f"Error in Rolling Mean: {str(e)}"); result.show()
        elif method == "Differencing":
            self._run_async(lambda: differencing(data), lambda r: (chart.plot_data(r, cols), result.hide()), result)
        elif method == "EWMA":
            try:
                dlg = QDialog(self)
                dlg.setWindowTitle("EWMA Parameters")
                layout = QVBoxLayout(dlg)
                layout.addWidget(QLabel("Span:"))
                span_spin = QSpinBox(dlg); span_spin.setRange(1, 100); span_spin.setValue(12); layout.addWidget(span_spin)
                btn_layout = QHBoxLayout(); ok_btn = QPushButton("OK", dlg); cancel_btn = QPushButton("Cancel", dlg)
                ok_btn.clicked.connect(dlg.accept); cancel_btn.clicked.connect(dlg.reject)
                btn_layout.addWidget(ok_btn); btn_layout.addWidget(cancel_btn); layout.addLayout(btn_layout)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                span = int(span_spin.value())
                self._run_async(lambda: ewma(data, span=span), lambda r: (chart.plot_data(r, cols), result.hide()), result)
            except Exception as e:
                result.setText(f"Error in EWMA: {str(e)}"); result.show()
        elif method == "Log Transform":
            try:
                transformed, offset = log_transform(data)
                chart.plot_data(transformed, transformed.columns.tolist())
                fig1 = plot_distribution(transformed, title_prefix="Log Transform Distribution")
                fig2 = plot_before_after(data, transformed, title_prefix="Log Transform")
                import matplotlib.pyplot as _plt
                _plt.show()
                result.setText(f"Applied log transform. Offset used: {offset}")
                result.show()
            except Exception as e:
                result.setText(f"Error in log transform: {str(e)}")
                result.show()
        elif method == "Box-Cox Transform":
            try:
                transformed, lmbda, shift = boxcox_transform(data)
                chart.plot_data(transformed, transformed.columns.tolist())
                fig1 = plot_distribution(transformed, title_prefix="Box-Cox Distribution")
                fig2 = plot_before_after(data, transformed, title_prefix="Box-Cox")
                import matplotlib.pyplot as _plt
                _plt.show()
                result.setText(f"Applied Box-Cox. lambda={lmbda:.4f}, shift={shift:.4f}\nUse inverse with same parameters to restore.")
                result.show()
            except Exception as e:
                result.setText(f"Error in Box-Cox transform: {str(e)}")
                result.show()
        elif method == "Standard Scale (Z)":
            self._run_async(lambda: standard_scale(data), lambda scaled: (chart.plot_data(scaled, scaled.columns.tolist()), self._show_dist("Z-Scale Distribution", scaled), result.hide()), result)
        elif method == "MinMax Scale":
            self._run_async(lambda: minmax_scale(data), lambda scaled: (chart.plot_data(scaled, scaled.columns.tolist()), self._show_dist("MinMax Distribution", scaled), result.hide()), result)
        elif method == "Seasonal Decompose":
            self._run_async(
                lambda: seasonal_decompose(data),
                lambda decomp: (build_seasonal_decomposition_figure(decomp), plt.show(), result.hide()),
                result,
            )
        elif method == "ARIMA":
            try:
                params = self._ask_arima_params(default_steps=30)
                if params is None:
                    return
                p, d, q, steps = params
                self._run_async(lambda: arima_forecast_with_ci(data, order=(p, d, q), steps=steps), lambda r: (chart.plot_data(r[0], [r[0].columns[0], 'Forecast'], ci_lower=r[1], ci_upper=r[2]), result.hide()), result)
            except Exception as e:
                result.setText(f"Error in ARIMA forecasting: {str(e)}")
                result.show()
        elif method == "Show ACF/PACF":
            self._run_async(
                lambda: prepare_acf_pacf_data(data, lags=40),
                lambda out: (build_acf_pacf_figure(out[0], out[1]), plt.show(), result.hide()),
                result,
            )
        elif method == "SARIMAX":
            try:
                dlg = QDialog(self)
                dlg.setWindowTitle("SARIMAX Parameters")
                layout = QVBoxLayout(dlg)
                layout.addWidget(QLabel("p, d, q:"))
                p_spin = QSpinBox(dlg); p_spin.setRange(0, 10); p_spin.setValue(1); layout.addWidget(p_spin)
                d_spin = QSpinBox(dlg); d_spin.setRange(0, 2); d_spin.setValue(0); layout.addWidget(d_spin)
                q_spin = QSpinBox(dlg); q_spin.setRange(0, 10); q_spin.setValue(1); layout.addWidget(q_spin)
                layout.addWidget(QLabel("P, D, Q, m (seasonal period):"))
                P_spin = QSpinBox(dlg); P_spin.setRange(0, 10); P_spin.setValue(0); layout.addWidget(P_spin)
                D_spin = QSpinBox(dlg); D_spin.setRange(0, 2); D_spin.setValue(0); layout.addWidget(D_spin)
                Q_spin = QSpinBox(dlg); Q_spin.setRange(0, 10); Q_spin.setValue(0); layout.addWidget(Q_spin)
                m_spin = QSpinBox(dlg); m_spin.setRange(0, 365); m_spin.setValue(0); layout.addWidget(m_spin)
                layout.addWidget(QLabel("Steps ahead:"))
                steps_spin = QSpinBox(dlg); steps_spin.setRange(1, 5000); steps_spin.setValue(30); layout.addWidget(steps_spin)
                btn_layout = QHBoxLayout()
                ok_btn = QPushButton("OK", dlg); cancel_btn = QPushButton("Cancel", dlg)
                ok_btn.clicked.connect(dlg.accept); cancel_btn.clicked.connect(dlg.reject)
                btn_layout.addWidget(ok_btn); btn_layout.addWidget(cancel_btn); layout.addLayout(btn_layout)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                order = (int(p_spin.value()), int(d_spin.value()), int(q_spin.value()))
                seasonal_order = (int(P_spin.value()), int(D_spin.value()), int(Q_spin.value()), int(m_spin.value()))
                steps = int(steps_spin.value())
                self._run_async(lambda: sarimax_forecast_with_ci(data, order=order, seasonal_order=seasonal_order, steps=steps), lambda r: (chart.plot_data(r[0], [r[0].columns[0], 'Forecast'], ci_lower=r[1], ci_upper=r[2]), result.hide()), result)
            except Exception as e:
                result.setText(f"Error in SARIMAX forecasting: {str(e)}")
                result.show()
        elif method == "Holt-Winters":
            try:
                dlg = QDialog(self)
                dlg.setWindowTitle("Holt-Winters Parameters")
                layout = QVBoxLayout(dlg)
                layout.addWidget(QLabel("Seasonal periods (m):"))
                m_spin = QSpinBox(dlg); m_spin.setRange(0, 365); m_spin.setValue(12); layout.addWidget(m_spin)
                layout.addWidget(QLabel("Steps ahead:"))
                steps_spin = QSpinBox(dlg); steps_spin.setRange(1, 5000); steps_spin.setValue(30); layout.addWidget(steps_spin)
                btn_layout = QHBoxLayout(); ok_btn = QPushButton("OK", dlg); cancel_btn = QPushButton("Cancel", dlg)
                ok_btn.clicked.connect(dlg.accept); cancel_btn.clicked.connect(dlg.reject)
                btn_layout.addWidget(ok_btn); btn_layout.addWidget(cancel_btn); layout.addLayout(btn_layout)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                m = int(m_spin.value()); steps = int(steps_spin.value())
                self._run_async(lambda: holt_winters_forecast(data, steps=steps, seasonal_periods=(m if m > 0 else None), trend='add', seasonal=('add' if m > 0 else None)), lambda forecast_df: (chart.plot_data(forecast_df, [forecast_df.columns[0], 'Forecast']), result.hide()), result)
            except Exception as e:
                result.setText(f"Error in Holt-Winters forecasting: {str(e)}")
                result.show()
        elif method == "Prophet":
            try:
                dlg = QDialog(self)
                dlg.setWindowTitle("Prophet Parameters")
                layout = QVBoxLayout(dlg)
                layout.addWidget(QLabel("Steps ahead (periods):"))
                steps_spin = QSpinBox(dlg); steps_spin.setRange(1, 5000); steps_spin.setValue(30); layout.addWidget(steps_spin)
                btn_layout = QHBoxLayout(); ok_btn = QPushButton("OK", dlg); cancel_btn = QPushButton("Cancel", dlg)
                ok_btn.clicked.connect(dlg.accept); cancel_btn.clicked.connect(dlg.reject)
                btn_layout.addWidget(ok_btn); btn_layout.addWidget(cancel_btn); layout.addLayout(btn_layout)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                periods = int(steps_spin.value())
                self._run_async(lambda: prophet_forecast(data, periods=periods), lambda forecast_df: (chart.plot_data(forecast_df, [forecast_df.columns[0], 'Forecast']), result.hide()), result)
            except Exception as e:
                result.setText(f"Error in Prophet forecasting: {str(e)}")
                result.show()
        elif method == "HP Filter":
            try:
                dlg = QDialog(self)
                dlg.setWindowTitle("HP Filter Parameter")
                layout = QVBoxLayout(dlg)
                layout.addWidget(QLabel("lambda (smoothing):"))
                lam_spin = QSpinBox(dlg); lam_spin.setRange(1, 1000000); lam_spin.setValue(1600); layout.addWidget(lam_spin)
                btn_layout = QHBoxLayout(); ok_btn = QPushButton("OK", dlg); cancel_btn = QPushButton("Cancel", dlg)
                ok_btn.clicked.connect(dlg.accept); cancel_btn.clicked.connect(dlg.reject)
                btn_layout.addWidget(ok_btn); btn_layout.addWidget(cancel_btn); layout.addLayout(btn_layout)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                lam = int(lam_spin.value())
                self._run_async(
                    lambda: hp_filter_decompose(data, lamb=lam),
                    lambda parts: (build_hp_filter_figure(parts[0], parts[1], parts[2], lamb=lam), plt.show(), result.hide()),
                    result,
                )
            except Exception as e:
                result.setText(f"Error in HP filter: {str(e)}")
                result.show()
        elif method == "Z-Score":
            try:
                dlg = QDialog(self)
                dlg.setWindowTitle("Z-Score Threshold")
                layout = QVBoxLayout(dlg)
                layout.addWidget(QLabel("Absolute Z-score threshold:"))
                thr = QSpinBox(dlg)
                thr.setRange(1, 10)
                thr.setValue(3)
                layout.addWidget(thr)
                btn_layout = QHBoxLayout(); ok_btn = QPushButton("OK", dlg); cancel_btn = QPushButton("Cancel", dlg)
                ok_btn.clicked.connect(dlg.accept); cancel_btn.clicked.connect(dlg.reject)
                btn_layout.addWidget(ok_btn); btn_layout.addWidget(cancel_btn); layout.addLayout(btn_layout)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                thr_val = float(thr.value())
                self._run_async(
                    lambda: zscore_anomalies(data, threshold=thr_val),
                    lambda r: (
                        self._plot_anomalies(chart, data, cols[0], r[2]),
                        result.setText(
                            f"Z-Score anomalies: {int(r[1].sum())}. Threshold: {thr_val}\n"
                            f"Mean Z: {r[0]['ZScore'].mean():.3f}  Max |Z|: {r[0]['ZScore'].abs().max():.3f}"
                        ),
                        result.show()
                    ),
                    result
                )
            except Exception as e:
                result.setText(f"Error in Z-Score anomalies: {str(e)}")
                result.show()
        elif method == "Isolation Forest":
            try:
                dlg = QDialog(self)
                dlg.setWindowTitle("Isolation Forest Parameters")
                layout = QVBoxLayout(dlg)
                layout.addWidget(QLabel("Contamination (percentage of anomalies, 0.1% - 20%):"))
                contam = QSpinBox(dlg)
                contam.setRange(1, 20)
                contam.setValue(1)
                layout.addWidget(contam)
                btn_layout = QHBoxLayout(); ok_btn = QPushButton("OK", dlg); cancel_btn = QPushButton("Cancel", dlg)
                ok_btn.clicked.connect(dlg.accept); cancel_btn.clicked.connect(dlg.reject)
                btn_layout.addWidget(ok_btn); btn_layout.addWidget(cancel_btn); layout.addLayout(btn_layout)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                contamination = float(contam.value()) / 100.0
                self._run_async(lambda: isolation_forest_anomalies(data, contamination=contamination), lambda r: (self._plot_anomalies(chart, data, cols[0], r[2]), result.setText(f"Isolation Forest anomalies: {int(r[1].sum())}. Contamination: {contamination:.3%}\nScore stats -> mean: {r[0]['IFScore'].mean():.3f}, max: {r[0]['IFScore'].max():.3f}"), result.show()), result)
            except Exception as e:
                result.setText(f"Error in Isolation Forest anomalies: {str(e)}")
                result.show()
        elif method == "Stationarity Tests":
            self._run_async(lambda: stationarity_tests(data), lambda report: (result.setText(report), result.show()), result)
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

    def _run_async(self, fn, on_success, result_widget):
        class _Bridge(QObject):
            finished = pyqtSignal(object)
            error = pyqtSignal(object)
        bridge = _Bridge()
        bridge.finished.connect(on_success)
        bridge.error.connect(lambda ex: (result_widget.setText(str(ex)), result_widget.show()))

        class _Task(QRunnable):
            def __init__(self, fn, bridge):
                super().__init__()
                self.fn = fn
                self.bridge = bridge
            def run(self_inner):
                try:
                    out = self_inner.fn()
                    self_inner.bridge.finished.emit(out)
                except Exception as ex:
                    self_inner.bridge.error.emit(ex)
        self._thread_pool.start(_Task(fn, bridge))

    def _show_dist(self, title, data):
        try:
            fig = plot_distribution(data, title_prefix=title)
            import matplotlib.pyplot as _plt
            _plt.show()
        except Exception:
            pass

    def _plot_anomalies(self, chart, data, col_name, anomalies_df):
        overlay_df = data.copy()
        overlay_df['Anomaly'] = anomalies_df.iloc[:, 0]
        chart.plot_data(overlay_df, [col_name, 'Anomaly'])

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

    def _ask_arima_params(self, default_steps=30, max_steps=5000):
        dlg = QDialog(self)
        dlg.setWindowTitle("ARIMA Parameters")
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("p (AR terms):"))
        p_spin = QSpinBox(dlg)
        p_spin.setRange(0, 10)
        p_spin.setValue(1)
        layout.addWidget(p_spin)

        layout.addWidget(QLabel("d (differences):"))
        d_spin = QSpinBox(dlg)
        d_spin.setRange(0, 2)
        d_spin.setValue(0)
        layout.addWidget(d_spin)

        layout.addWidget(QLabel("q (MA terms):"))
        q_spin = QSpinBox(dlg)
        q_spin.setRange(0, 10)
        q_spin.setValue(1)
        layout.addWidget(q_spin)

        layout.addWidget(QLabel("Steps ahead to forecast:"))
        steps_spin = QSpinBox(dlg)
        steps_spin.setRange(1, max_steps)
        steps_spin.setValue(default_steps)
        layout.addWidget(steps_spin)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK", dlg)
        cancel_btn = QPushButton("Cancel", dlg)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            return int(p_spin.value()), int(d_spin.value()), int(q_spin.value()), int(steps_spin.value())
        return None

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
