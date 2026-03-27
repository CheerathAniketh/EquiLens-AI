from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def prepare_features(df, target_col, sensitive_col):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(
            df_encoded[col].astype(str))
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    return X, y

def train_and_evaluate(df, target_col, sensitive_col):
    X, y = prepare_features(df, target_col, sensitive_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # scale features — fixes convergence on large datasets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # convert back to DataFrame to keep column names for SHAP
    import numpy as np
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    model = LogisticRegression(max_iter=3000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, X_train_scaled, X_test_scaled, y_test