import pandas as pd
import numpy as np

def rolling_mean(data, window=5):
    return data.rolling(window).mean()


def differencing(data, periods=1):
    return data.diff(periods=periods).dropna()


def ewma(data, span=12):

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("EWMA expects a single series")
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce')
        result = series.ewm(span=span, adjust=False).mean()
        return result.to_frame(name=data.columns[0])
    else:
        series = pd.to_numeric(pd.Series(data), errors='coerce')
        return series.ewm(span=span, adjust=False).mean()


def log_transform(data):

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Log transform expects a single series")
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce')
        min_val = series.min()
        offset = 1 - min_val if pd.notna(min_val) and min_val <= 0 else 0.0
        transformed = np.log(series + offset)
        name = f"log({data.columns[0]}{'' if offset == 0 else f'+{offset:.4g}'})"
        return transformed.to_frame(name=name), float(offset)
    else:
        series = pd.to_numeric(pd.Series(data), errors='coerce')
        min_val = series.min()
        offset = 1 - min_val if pd.notna(min_val) and min_val <= 0 else 0.0
        transformed = np.log(series + offset)
        return transformed, float(offset)


def boxcox_transform(data):

    from scipy.stats import boxcox as scipy_boxcox

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Box-Cox expects a single series")
        series_name = data.columns[0]
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce')
    else:
        series_name = getattr(data, 'name', 'value')
        series = pd.to_numeric(pd.Series(data), errors='coerce')

    series_clean = series.dropna()
    if series_clean.empty:
        raise ValueError("No valid numeric data for Box-Cox")

    min_val = series_clean.min()
    shift = 1 - min_val if pd.notna(min_val) and min_val <= 0 else 0.0
    shifted = series_clean + shift

    transformed_values, lmbda = scipy_boxcox(shifted)
    transformed = pd.Series(
        transformed_values,
        index=series_clean.index,
        name=f"BoxCox[{lmbda:.3f}]({series_name}{'' if shift == 0 else f'+{shift:.4g}'})",
    )

    return transformed.to_frame(name=transformed.name), float(lmbda), float(shift)


def boxcox_inverse(transformed, lmbda, shift=0.0):

    from scipy.special import inv_boxcox as scipy_inv_boxcox

    if isinstance(transformed, pd.DataFrame):
        if transformed.shape[1] != 1:
            raise ValueError("Inverse Box-Cox expects a single series")
        series = pd.to_numeric(transformed.iloc[:, 0], errors='coerce')
        inv = pd.Series(scipy_inv_boxcox(series, lmbda) - shift, index=series.index, name=transformed.columns[0])
        return inv.to_frame(name=inv.name)
    else:
        series = pd.to_numeric(pd.Series(transformed), errors='coerce')
        return pd.Series(scipy_inv_boxcox(series, lmbda) - shift, index=series.index, name=getattr(series, 'name', None))


def standard_scale(data):

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Standard scale expects a single series")
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce')
        mean = series.mean()
        std = series.std(ddof=0)
        scaled = (series - mean) / (std if std != 0 else 1)
        return scaled.to_frame(name=f"Z({data.columns[0]})")
    else:
        series = pd.to_numeric(pd.Series(data), errors='coerce')
        mean = series.mean()
        std = series.std(ddof=0)
        return (series - mean) / (std if std != 0 else 1)


def minmax_scale(data, feature_range=(0.0, 1.0)):

    a, b = feature_range
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("MinMax scale expects a single series")
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce')
        min_v, max_v = series.min(), series.max()
        denom = (max_v - min_v) if max_v != min_v else 1
        scaled = a + (series - min_v) * (b - a) / denom
        return scaled.to_frame(name=f"MinMax({data.columns[0]})")
    else:
        series = pd.to_numeric(pd.Series(data), errors='coerce')
        min_v, max_v = series.min(), series.max()
        denom = (max_v - min_v) if max_v != min_v else 1
        return a + (series - min_v) * (b - a) / denom

