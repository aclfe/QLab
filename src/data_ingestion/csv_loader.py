import pandas as pd
from pathlib import Path
from dateutil.parser import parse as date_parse
import csv

def load_csv(
    path: str,
    date_col: str = None,
    index_col: str = None,
    datetime_cols: list[str] = None
) -> pd.DataFrame:

    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(2048)
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ','

    df = pd.read_csv(file, delimiter=delimiter)
    df.columns = df.columns.str.strip().str.lower()

    if datetime_cols:
        candidates = [c.lower() for c in datetime_cols if c.lower() in df.columns]
    else:
        candidates = [c for c in df.columns if any(k in c for k in ['date','time','timestamp','datetime'])]

    if date_col and date_col.lower() in df.columns:
        candidates = [date_col.lower()]
    if index_col and index_col.lower() in df.columns:
        candidates = [index_col.lower()]

    if candidates:
        if len(candidates) == 1:
            col = candidates[0]
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df.set_index(col, inplace=True)
        else:
            df['__datetime'] = df[candidates].astype(str).agg(' '.join, axis=1)
            df['__datetime'] = df['__datetime'].apply(lambda x: date_parse(x, fuzzy=True))
            df.set_index('__datetime', inplace=True)
            # drop safely
            to_drop = [c for c in candidates if c in df.columns] + ['__datetime']
            df.drop(columns=to_drop, inplace=True, errors='ignore')
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors='raise')
            except Exception:
                pass

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    return df