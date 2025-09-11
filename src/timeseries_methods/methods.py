import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

def stationarity_tests(data):

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