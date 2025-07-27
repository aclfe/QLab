import pandas as pd
from pathlib import Path

def load_csv(path: str, date_col: str = None, index_col: str = None) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    if 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
        df.set_index('datetime', inplace=True)
        df.drop(['date', 'time'], axis=1, inplace=True)
    elif date_col and date_col.lower() in df.columns:
        col = date_col.lower()
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df.set_index(col, inplace=True)
    elif index_col and index_col.lower() in df.columns:
        df.set_index(index_col.lower(), inplace=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors='raise')
        except Exception:
            pass

    return df