import pandas as pd
import mplfinance as mpf


def infer_data_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {'open', 'high', 'low', 'close'}.issubset(cols):
        return 'financial'
    elif any(term in c for term in ['temp', 'pressure'] for c in cols):
        return 'sensor'
    elif any('anomaly' in c or 'label' in c for c in cols):
        return 'anomaly'
    elif any('yhat' in c for c in cols):
        return 'forecast'
    else:
        return 'generic'

class PlotManager:
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget

    def render(self, df: pd.DataFrame):
        dtype = infer_data_type(df)
        if dtype == 'financial':
            self._plot_financial(df)
        elif dtype == 'sensor':
            self._plot_sensor(df)
        elif dtype == 'anomaly':
            self._plot_anomalies(df)
        elif dtype == 'forecast':
            self._plot_forecast(df)
        else:
            self._plot_generic(df)

    def _plot_financial(self, df: pd.DataFrame):
        def plot_candle(ax):
            rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
            ohlc = df.rename(columns=rename_map)[['Open', 'High', 'Low', 'Close']]
            mpf.plot(
                ohlc,
                type='candle',
                style='charles',
                ax=ax,
                volume=False,
                datetime_format='%Y-%m-%d',
                warn_too_much_data=len(ohlc)+1
            )
            ax.set_facecolor('#222222')
        self.plot_widget.plot(plot_candle)

    def _plot_sensor(self, df: pd.DataFrame):
        self.plot_widget.plot(lambda ax: df.plot(ax=ax, title="Sensor Data"))

    def _plot_anomalies(self, df: pd.DataFrame):
        def plot_with_anomalies(ax):
            df.plot(ax=ax, title="Anomaly Detection")
            if 'anomaly' in df.columns:
                anomalies = df[df['anomaly'] == 1]
                ax.scatter(anomalies.index, anomalies.iloc[:, 0], color='red', label='Anomalies')
                ax.legend()
        self.plot_widget.plot(plot_with_anomalies)

    def _plot_forecast(self, df: pd.DataFrame):
        def plot_forecast(ax):
            df['yhat'].plot(ax=ax, label='Forecast')
            if 'yhat_lower' in df.columns and 'yhat_upper' in df.columns:
                ax.fill_between(df.index, df['yhat_lower'], df['yhat_upper'], color='gray', alpha=0.3)
            ax.set_title("Forecast")
            ax.legend()
        self.plot_widget.plot(plot_forecast)

    def _plot_generic(self, df: pd.DataFrame):
        self.plot_widget.plot(lambda ax: df.plot(ax=ax, title="Generic Data"))
