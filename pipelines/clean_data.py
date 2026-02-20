# clean_data.py
import pandas as pd
from helpers import (
    standardize_columns,
    fill_missing,
    remove_duplicates,
    convert_types,
    strip_strings,
    remove_special_characters,
    drop_constant_columns,
    remove_outliers
)

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def run_pipeline(
        self, 
        fill_method='ffill', 
        type_dict=None, 
        remove_special_cols=None,
        outlier_cols=None
    ) -> pd.DataFrame:
        """
        Runs the full universal data cleaning pipeline
        """
        # 1. Standardize column names
        self.df = standardize_columns(self.df)
        
        # 2. Remove duplicates
        self.df = remove_duplicates(self.df)
        
        # 3. Fill missing values
        self.df = fill_missing(self.df, method=fill_method)
        
        # 4. Convert column types
        if type_dict:
            self.df = convert_types(self.df, type_dict)
        
        # 5. Trim string columns
        self.df = strip_strings(self.df)
        
        # 6. Remove special characters
        self.df = remove_special_characters(self.df, columns=remove_special_cols)
        
        # 7. Drop constant columns
        self.df = drop_constant_columns(self.df)
        
        # 8. Remove outliers
        if outlier_cols:
            self.df = remove_outliers(self.df, columns=outlier_cols)
        
        return self.df

# Usage Example:
if __name__ == "__main__":
    # Load any dataset
    df = pd.read_csv("dataset.csv")
    
    # Initialize pipeline
    cleaner = DataCleaner(df)
    
    # Optional: define types and columns to clean
    type_dict = {'age': int, 'date': 'datetime64'}
    remove_special_cols = ['name', 'address']
    outlier_cols = ['salary', 'score']
    
    # Run full pipeline
    clean_df = cleaner.run_pipeline(
        fill_method='mean',
        type_dict=type_dict,
        remove_special_cols=remove_special_cols,
        outlier_cols=outlier_cols
    )
    
    print(clean_df.head())