from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd

class ChartCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

        self._dragging = False
        self._last_mouse_pos = None
        self.current_df = None

    def plot_data(self, df: pd.DataFrame):
        self.current_df = df
        self.ax.clear()
        self.ax.plot(df.index, df.iloc[:, 0], color='cyan', linewidth=1.0)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_title("Line View", color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.draw()

    def plot_table(self, df: pd.DataFrame):
        self.fig.clf()
        layout = QVBoxLayout()
        table = QTableWidget()
        table.setRowCount(len(df.index))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i, row in enumerate(df.values):
            for j, val in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(val)))
        parent = self.parent()
        parent.layout().removeWidget(self)
        table_container = QWidget()
        vlay = QVBoxLayout(table_container)
        vlay.addWidget(table)
        parent.layout().addWidget(table_container)

    def _on_scroll(self, event):
        base_scale = 1.2
        cur_xlim = self.ax.get_xlim()
        xdata = event.xdata
        if xdata is None:
            return
        scale_factor = base_scale if event.button == 'up' else 1 / base_scale
        left = xdata - (xdata - cur_xlim[0]) * scale_factor
        right = xdata + (cur_xlim[1] - xdata) * scale_factor
        self.ax.set_xlim([left, right])
        self.draw()

    def _on_press(self, event):
        if event.button == 1:
            self._dragging = True
            self._last_mouse_pos = event.x

    def _on_motion(self, event):
        if self._dragging and event.xdata is not None:
            dx = self._last_mouse_pos - event.x
            xlim = self.ax.get_xlim()
            scale = (xlim[1] - xlim[0]) / self.width()
            self.ax.set_xlim([x + dx*scale for x in xlim])
            self.draw()
            self._last_mouse_pos = event.x

    def _on_release(self, event):
        self._dragging = False
        self._last_mouse_pos = None
