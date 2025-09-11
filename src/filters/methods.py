import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter


def hp_filter_decompose(data, lamb=1600):
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("HP filter expects a single series")
        series_name = data.columns[0]
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
    else:
        series_name = 'Value' if getattr(data, 'name', None) is None else data.name
        series = pd.to_numeric(pd.Series(data), errors='coerce').dropna()

    trend, cycle = hpfilter(series, lamb=lamb)
    trend.name = 'Trend'
    cycle.name = 'Cycle'
    return series.to_frame(name=series_name), trend.to_frame(), cycle.to_frame()


def plot_hp_filter(data, lamb=1600, figsize=(10, 6)):
    orig, trend, cycle = hp_filter_decompose(data, lamb=lamb)
    return build_hp_filter_figure(orig, trend, cycle, lamb=lamb, figsize=figsize)


def build_hp_filter_figure(orig, trend, cycle, lamb=1600, figsize=(10, 6)):
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].plot(orig.index, orig.iloc[:, 0].values, label='Original', color='steelblue')
    axes[0].plot(trend.index, trend.iloc[:, 0].values, label='Trend', color='orange')
    axes[0].set_title(f'HP Filter (lambda={lamb})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(cycle.index, cycle.iloc[:, 0].values, label='Cycle', color='green')
    axes[1].set_title('Cycle')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

