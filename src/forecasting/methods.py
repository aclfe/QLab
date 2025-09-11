import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet


def arima_forecast(data, order=(1, 0, 1), steps=30):

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("ARIMA expects a single series (one column DataFrame or Series)")
        series_name = data.columns[0]
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
    else:
        series_name = 'Value' if data.name is None else data.name
        series = pd.to_numeric(pd.Series(data), errors='coerce').dropna()

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("The data index must be a DatetimeIndex for forecasting.")

    if len(series) < sum(order) + 2:
        raise ValueError("Not enough data points to fit ARIMA model")

    inferred_freq = pd.infer_freq(series.index)
    if inferred_freq:
        series = series.asfreq(inferred_freq)

    model = ARIMA(series, order=order)
    fitted = model.fit()

    forecast = fitted.get_forecast(steps=steps)
    forecast_series = forecast.predicted_mean
    forecast_series.name = 'Forecast'

    combined_df = pd.concat([series.to_frame(name=series_name), forecast_series.to_frame()], axis=1)

    if not series.empty:
        combined_df.loc[series.index[-1], 'Forecast'] = series.iloc[-1]

    return combined_df


def arima_forecast_with_ci(data, order=(1, 0, 1), steps=30, alpha=0.05):

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("ARIMA expects a single series (one column DataFrame or Series)")
        series_name = data.columns[0]
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
    else:
        series_name = 'Value' if getattr(data, 'name', None) is None else data.name
        series = pd.to_numeric(pd.Series(data), errors='coerce').dropna()

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("The data index must be a DatetimeIndex for forecasting.")

    if len(series) < sum(order) + 2:
        raise ValueError("Not enough data points to fit ARIMA model")

    inferred_freq = pd.infer_freq(series.index)
    if inferred_freq:
        series = series.asfreq(inferred_freq)

    model = ARIMA(series, order=order)
    fitted = model.fit()

    forecast_obj = fitted.get_forecast(steps=steps)
    forecast_series = forecast_obj.predicted_mean
    ci = forecast_obj.conf_int(alpha=alpha)
    if ci.shape[1] >= 2:
        ci_lower = ci.iloc[:, 0]
        ci_upper = ci.iloc[:, 1]
    else:
        ci_lower = forecast_series.copy()
        ci_upper = forecast_series.copy()

    combined_df = pd.concat([series.to_frame(name=series_name), forecast_series.to_frame(name='Forecast')], axis=1)
    if not series.empty:
        combined_df.loc[series.index[-1], 'Forecast'] = series.iloc[-1]

    return combined_df, ci_lower, ci_upper


def sarimax_forecast_with_ci(data, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), steps=30, exog=None, alpha=0.05):
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("SARIMAX expects a single series")
        series_name = data.columns[0]
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
    else:
        series_name = 'Value' if getattr(data, 'name', None) is None else data.name
        series = pd.to_numeric(pd.Series(data), errors='coerce').dropna()

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("The data index must be a DatetimeIndex for forecasting.")

    inferred_freq = pd.infer_freq(series.index)
    if inferred_freq:
        series = series.asfreq(inferred_freq)

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    forecast_obj = fitted.get_forecast(steps=steps, exog=None)
    forecast_series = forecast_obj.predicted_mean
    ci = forecast_obj.conf_int(alpha=alpha)
    if ci.shape[1] >= 2:
        ci_lower = ci.iloc[:, 0]
        ci_upper = ci.iloc[:, 1]
    else:
        ci_lower = forecast_series.copy()
        ci_upper = forecast_series.copy()

    combined_df = pd.concat([series.to_frame(name=series_name), forecast_series.to_frame(name='Forecast')], axis=1)
    if not series.empty:
        combined_df.loc[series.index[-1], 'Forecast'] = series.iloc[-1]
    return combined_df, ci_lower, ci_upper


def holt_winters_forecast(data, steps=30, seasonal_periods=None, trend='add', seasonal='add'):
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Holt-Winters expects a single series")
        series_name = data.columns[0]
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
    else:
        series_name = 'Value' if getattr(data, 'name', None) is None else data.name
        series = pd.to_numeric(pd.Series(data), errors='coerce').dropna()

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("The data index must be a DatetimeIndex for forecasting.")

    inferred_freq = pd.infer_freq(series.index)
    if inferred_freq:
        series = series.asfreq(inferred_freq)

    model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fitted = model.fit()
    forecast_series = fitted.forecast(steps)
    forecast_series.name = 'Forecast'
    combined_df = pd.concat([series.to_frame(name=series_name), forecast_series.to_frame()], axis=1)
    if not series.empty:
        combined_df.loc[series.index[-1], 'Forecast'] = series.iloc[-1]
    return combined_df


def prophet_forecast(data, periods=30, freq=None):
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Prophet expects a single series")
        series_name = data.columns[0]
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
    else:
        series_name = 'Value' if getattr(data, 'name', None) is None else data.name
        series = pd.to_numeric(pd.Series(data), errors='coerce').dropna()

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("The data index must be a DatetimeIndex for forecasting.")

    df = pd.DataFrame({'ds': series.index, 'y': series.values})
    m = Prophet()
    m.fit(df)
    if freq is None:
        freq = pd.infer_freq(series.index) or 'D'
    future = m.make_future_dataframe(periods=periods, freq=freq, include_history=True)
    forecast = m.predict(future)
    forecast_series = forecast.set_index('ds')['yhat']
    forecast_series.name = 'Forecast'
    combined_df = pd.concat([series.to_frame(name=series_name), forecast_series.to_frame()], axis=1)
    if not series.empty:
        combined_df.loc[series.index[-1], 'Forecast'] = series.iloc[-1]
    return combined_df