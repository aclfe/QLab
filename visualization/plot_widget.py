from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(facecolor="#222222")
        self.canvas = Canvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def plot(self, plot_fn):
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor="#222222")
        plot_fn(ax)
        ax.tick_params(colors="#FFFFFF")
        for spine in ax.spines.values():
            spine.set_color("#FFFFFF")
        self.canvas.draw()
