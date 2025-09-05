import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose as sm_seasonal_decompose

def rolling_mean(data, window=5):
    return data.rolling(window).mean()

def differencing(data, periods=1):
    return data.diff(periods=periods).dropna()

def seasonal_decompose(data, model='additive', period=None):
    if period is None:
        period = max(2, int(len(data) / 2))
    result = sm_seasonal_decompose(data.iloc[:,0], model=model, period=period)
    return pd.DataFrame({
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid
    })

