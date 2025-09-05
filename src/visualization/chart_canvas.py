from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

class ChartCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.ax = None
        self._current_projection = None
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
        self._current_projection = None

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

    def plot_3d_data(self, df, col1, col2):
        try:
            self.fig.clf()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._current_projection = '3d'

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            data1 = pd.to_numeric(df[col1], errors='coerce')
            data2 = pd.to_numeric(df[col2], errors='coerce')
            times = df.index.astype(np.int64) // 10**9

            mask = ~(data1.isna() | data2.isna())
            times = times[mask]
            data1 = data1[mask].to_numpy()
            data2 = data2[mask].to_numpy()

            if len(times) == 0:
                print("No valid data points for 3D plot")
                return

            max_points = 2000  
            if len(times) > max_points:
                step = len(times) // max_points
                times = times[::step]
                data1 = data1[::step]
                data2 = data2[::step]

            time_range = times.max() - times.min()
            if time_range == 0:
                norm_times = np.zeros_like(times, dtype=float)
            else:
                norm_times = (times - times.min()) / time_range
            
            scatter = self.ax.scatter(times, data1, data2, 
                                    c=norm_times,
                                    cmap='plasma',
                                    alpha=0.8,
                                    s=50)

            self.ax.set_xlabel('Time', fontsize=10, labelpad=10)
            self.ax.set_ylabel(col1, fontsize=10, labelpad=10)
            self.ax.set_zlabel(col2, fontsize=10, labelpad=10)

            num_ticks = 5
            tick_indices = np.linspace(0, len(times)-1, num_ticks, dtype=int)
            self.ax.set_xticks(times[tick_indices])
            self.ax.set_xticklabels([pd.Timestamp(t * 10**9).strftime('%Y-%m-%d') 
                                    for t in times[tick_indices]], rotation=45)

            cbar = self.fig.colorbar(scatter, label='Time Progress')
            cbar.ax.yaxis.label.set_color('white')
            cbar.ax.tick_params(colors='white')
            
            self.ax.view_init(elev=20, azim=45)

            self._format_axes()
            self.draw()
        except Exception as e:
            print(f"Error creating 3D plot: {str(e)}")
            self._current_projection = None
            self.plot_data(df, [col1])
        
        self._format_axes()
        self.draw()

    def _format_axes(self):
        if self.ax is None:
            return
            
        self.ax.set_facecolor('#1e1e1e')
        if self._current_projection == '3d':
            self.ax.set_title('3D View', color='white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.zaxis.label.set_color('white')
            self.ax.tick_params(colors='white', axis='x')
            self.ax.tick_params(colors='white', axis='y')
            self.ax.tick_params(colors='white', axis='z')
        else:
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

    def set_grid(self, show):
        self.ax.grid(show)
        self.draw()
