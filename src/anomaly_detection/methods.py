import pandas as pd
from sklearn.ensemble import IsolationForest


def zscore_anomalies(data, threshold=3.0):

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Z-Score expects a single series")
        series_name = data.columns[0]
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce')
    else:
        series_name = getattr(data, 'name', 'value')
        series = pd.to_numeric(pd.Series(data), errors='coerce')

    mean = series.mean()
    std = series.std(ddof=0)
    denom = std if std != 0 else 1.0
    z = (series - mean) / denom
    mask = z.abs() >= threshold
    anomalies = series.where(mask)
    z_df = z.to_frame(name='ZScore')
    anomalies_df = anomalies.to_frame(name='Anomaly')
    return z_df, mask, anomalies_df


def isolation_forest_anomalies(data, contamination=0.01, random_state=42):

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Isolation Forest expects a single series")
        series_name = data.columns[0]
        series = pd.to_numeric(data.iloc[:, 0], errors='coerce')
    else:
        series_name = getattr(data, 'name', 'value')
        series = pd.to_numeric(pd.Series(data), errors='coerce')

    values = series.values.reshape(-1, 1)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(values)
    labels = clf.predict(values)
    mask = pd.Series(labels == -1, index=series.index)

    scores = -pd.Series(clf.score_samples(values), index=series.index)
    scores_df = scores.to_frame(name='IFScore')
    anomalies_df = series.where(mask).to_frame(name='Anomaly')
    return scores_df, mask, anomalies_df

