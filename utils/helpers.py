import pandas as pd

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts column names to lowercase and replaces spaces with underscores
    """
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def fill_missing(df: pd.DataFrame, method='ffill') -> pd.DataFrame:
    """
    Fill missing values with forward fill (default) or zero
    """
    if method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'zero':
        df = df.fillna(0)
    return df