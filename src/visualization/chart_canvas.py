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

        self._dragging = False
        self._last_mouse_pos = None

        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)

    def plot_data(self, df: pd.DataFrame, cols=None):

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)

        if cols is None:
            cols = df.columns.tolist()

        for col in cols:
            if col in df.columns:
                self.ax.plot(df.index, df[col], label=col)

        self._format_axes()
        self.draw()

    def plot_table(self, df: pd.DataFrame):

        self.fig.clf()
        self.ax = None

        table = QTableWidget()
        rows, cols = df.shape
        table.setRowCount(rows)
        table.setColumnCount(cols)
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        table.setVerticalHeaderLabels([str(i) for i in df.index])

        for i in range(rows):
            for j in range(cols):
                table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

        parent = self.parent()
        layout = QVBoxLayout(parent)
        layout.addWidget(table)
        parent.setLayout(layout)

    def _format_axes(self):
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_title('Line View', color='white')
        self.ax.tick_params(colors='white')
        legend = self.ax.legend(facecolor='gray', edgecolor='white', labelcolor='white')
        legend.get_frame().set_alpha(0.5)

    def _on_scroll(self, event):
        base_scale = 1.2
        cur_xlim = self.ax.get_xlim()
        xdata = event.xdata or 0
        scale_factor = base_scale if event.button == 'up' else 1/base_scale
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        center = xdata
        left = center - new_width * ((center - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]))
        right = left + new_width
        self.ax.set_xlim(left, right)
        self.draw()

    def _on_press(self, event):
        if event.button == 1:
            self._dragging = True
            self._last_mouse_pos = event.x

    def _on_motion(self, event):
        if self._dragging and event.xdata is not None:
            dx = self._last_mouse_pos - event.x
            left, right = self.ax.get_xlim()
            scale = (right - left) / self.width()
            self.ax.set_xlim(left + dx*scale, right + dx*scale)
            self._last_mouse_pos = event.x
            self.draw()

    def _on_release(self, event):
        self._dragging = False
        self._last_mouse_pos = None
