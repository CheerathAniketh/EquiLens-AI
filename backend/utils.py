import pandas as pd

def parse_and_clean(file):
    df = pd.read_csv(file)
    
    # drop rows with missing target
    # fill missing numerics with median
    # fill missing categoricals with mode
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    return df

def bin_continuous(df, col, bins=4):
    # converts age/income into quartile groups
    # so SPD/DI can be calculated on them
    df[col] = pd.qcut(df[col], q=bins, 
                      labels=['Q1','Q2','Q3','Q4'],
                      duplicates='drop')
    return df

def validate_columns(df, target_col, sensitive_col):
    errors = []
    if target_col not in df.columns:
        errors.append(f"{target_col} not found")
    if sensitive_col not in df.columns:
        errors.append(f"{sensitive_col} not found")
    if df[target_col].nunique() > 10:
        errors.append("Target column must be binary or categorical")
    return errors