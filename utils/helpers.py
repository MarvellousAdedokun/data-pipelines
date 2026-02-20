import pandas as pd
import numpy as np

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts column names to lowercase and replaces spaces with underscores
    """
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def fill_missing(df: pd.DataFrame, method='ffill') -> pd.DataFrame:
    """
    Fill missing values with forward fill (default), zero, or mean
    """
    if method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'zero':
        df = df.fillna(0)
    elif method == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    return df

def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """
    Remove duplicate rows. Can specify subset of columns.
    """
    df = df.drop_duplicates(subset=subset)
    return df

def convert_types(df: pd.DataFrame, type_dict: dict) -> pd.DataFrame:
    """
    Convert column types based on type_dict. Example: {'age': int, 'date': 'datetime64'}
    """
    for col, dtype in type_dict.items():
        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            print(f"[WARNING] Could not convert {col} to {dtype}: {e}")
    return df

def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading/trailing spaces in string columns
    """
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    return df

def remove_special_characters(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Remove non-alphanumeric characters in specified columns
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns
    for col in columns:
        df[col] = df[col].str.replace(r'[^A-Za-z0-9 ]+', '', regex=True)
    return df

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns where all values are the same
    """
    return df.loc[:, df.nunique() > 1]

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Detect outliers using the IQR method
    Returns boolean series: True if outlier
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)

def remove_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove rows with outliers in specified columns
    """
    for col in columns:
        outliers = detect_outliers_iqr(df, col)
        df = df.loc[~outliers]
    return df