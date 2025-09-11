import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf as _plot_acf, plot_pacf as _plot_pacf
from scipy import stats as scipy_stats


def plot_distribution(data, title_prefix="Distribution"):

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Distribution plotting expects a single series")
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
        name = data.columns[0]
    else:
        series = pd.to_numeric(pd.Series(data), errors='coerce').dropna()
        name = getattr(data, 'name', 'value')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(series, bins='auto', alpha=0.7, color='steelblue', edgecolor='black')
    try:
        series.plot(kind='kde', ax=axes[0], color='orange')
    except Exception:
        pass
    axes[0].set_title(f"{title_prefix}: {name}")
    axes[0].grid(True, alpha=0.3)

    scipy_stats.probplot(series, dist="norm", plot=axes[1])
    axes[1].set_title("QQ Plot")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_before_after(original, transformed, title_prefix="Transform"):

    if isinstance(original, pd.DataFrame):
        orig_series = pd.to_numeric(original.iloc[:, 0], errors='coerce')
        orig_name = original.columns[0]
    else:
        orig_series = pd.to_numeric(pd.Series(original), errors='coerce')
        orig_name = getattr(original, 'name', 'original')

    if isinstance(transformed, pd.DataFrame):
        trans_series = pd.to_numeric(transformed.iloc[:, 0], errors='coerce')
        trans_name = transformed.columns[0]
    else:
        trans_series = pd.to_numeric(pd.Series(transformed), errors='coerce')
        trans_name = getattr(transformed, 'name', 'transformed')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
    axes[0].plot(orig_series.index, orig_series.values, color='steelblue')
    axes[0].set_title(f"{title_prefix}: Original ({orig_name})")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(trans_series.index, trans_series.values, color='orange')
    axes[1].set_title(f"{title_prefix}: Transformed ({trans_name})")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def seasonal_decompose(data, model='additive', period=None):
    from statsmodels.tsa.seasonal import seasonal_decompose as sm_seasonal_decompose

    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            series = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have exactly one column for seasonal decomposition")
    else:
        series = data

    series_clean = series.dropna()

    if len(series_clean) < 2:
        raise ValueError("Not enough data points for seasonal decomposition")

    if period is None:
        if len(series_clean) >= 24:
            period = 12
        elif len(series_clean) >= 14:
            period = 7
        else:
            period = max(2, int(len(series_clean) / 2))

    period = max(2, min(period, len(series_clean) // 2))

    if len(series_clean) < 2 * period:
        period = max(2, len(series_clean) // 2)

    try:
        result = sm_seasonal_decompose(series_clean, model=model, period=period)
        return result
    except Exception as e:
        if period > 2:
            try:
                period = max(2, period // 2)
                result = sm_seasonal_decompose(series_clean, model=model, period=period)
                return result
            except Exception as e2:
                raise ValueError(f"Seasonal decomposition failed with period {period}: {str(e2)}")
        else:
            raise ValueError(f"Seasonal decomposition failed: {str(e)}")


def plot_seasonal_decomposition(data, model='additive', period=None, figsize=(10, 8)):
    decomposition = seasonal_decompose(data, model=model, period=period)
    return build_seasonal_decomposition_figure(decomposition, figsize=figsize)


def build_seasonal_decomposition_figure(decomposition, figsize=(10, 8)):
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle('Seasonal Decomposition', fontsize=14)

    decomp_index = decomposition.observed.index

    axes[0].plot(decomp_index, decomposition.observed, label='Observed', color='blue', linewidth=1)
    axes[0].set_title('Observed')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(decomp_index, decomposition.trend, label='Trend', color='skyblue', linewidth=1)
    axes[1].set_title('Trend')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(decomp_index, decomposition.seasonal, label='Seasonal', color='orange', linewidth=1)
    axes[2].set_title('Seasonal')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(decomp_index, decomposition.resid, label='Residuals', color='lime', linewidth=1)
    axes[3].set_title('Residuals')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_acf_pacf(data, lags=40):
    series, safe_lags = prepare_acf_pacf_data(data, lags)
    return build_acf_pacf_figure(series, safe_lags)


def prepare_acf_pacf_data(data, lags=40):
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("ACF/PACF expects a single series")
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
    else:
        series = pd.to_numeric(pd.Series(data), errors='coerce').dropna()

    n = len(series)
    if n < 5:
        raise ValueError("Not enough data for ACF/PACF (need at least 5 observations).")

    safe_lags = int(min(max(1, lags), max(1, n - 2)))
    return series, safe_lags


def build_acf_pacf_figure(series, lags):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title('ACF')
    _plot_pacf(series, lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title('PACF')
    plt.tight_layout()
    return fig


from statsmodels.tsa.stattools import adfuller, kpss
import warnings

def stationarity_tests(data):
    """
    Run ADF and KPSS tests and return a human-readable report string.
    """
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Stationarity tests expect a single series")
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
        name = data.columns[0]
    else:
        series = pd.to_numeric(pd.Series(data), errors='coerce').dropna()
        name = getattr(data, 'name', 'value')

    if len(series) < 10:
        return "Not enough data for stationarity tests (need at least 10 observations)."

    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(series, autolag='AIC')

    kpss_warn = None
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series, regression='c', nlags='auto')
            for w in wlist:
                if issubclass(w.category, Warning):
                    kpss_warn = str(w.message)
    except Exception:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = np.nan, np.nan, None, {}

    lines = []
    lines.append(f"Stationarity Tests for '{name}'")
    lines.append("")
    lines.append("ADF Test (H0: unit root, non-stationary):")
    lines.append(f"  Statistic: {adf_stat:.4f}")
    lines.append(f"  p-value:   {adf_p:.4g}")
    lines.append(f"  Used lags: {adf_lags}")
    lines.append(f"  N obs:     {adf_nobs}")
    lines.append("  Critical values:")
    for key, val in adf_crit.items():
        lines.append(f"    {key}: {val:.4f}")
    lines.append("")
    lines.append("KPSS Test (H0: level stationary):")
    lines.append(f"  Statistic: {kpss_stat if not np.isnan(kpss_stat) else 'n/a'}")
    lines.append(f"  p-value:   {kpss_p if not np.isnan(kpss_p) else 'n/a'}")
    lines.append(f"  Used lags: {kpss_lags}")
    if kpss_crit:
        lines.append("  Critical values:")
        for key, val in kpss_crit.items():
            lines.append(f"    {key}: {val:.4f}")
    if kpss_warn:
        lines.append("")
        lines.append(f"  Note: {kpss_warn}")

    return "\n".join(lines)

