import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop customerID — not a feature
    df = df.drop(columns=['customerID'])

    # Fix TotalCharges — it's a string with spaces
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Convert target to binary
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    print(f"Churn rate: {df['Churn'].mean():.1%}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Tenure groups
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5-6yr']
    )

    # Charges per month ratio
    df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Number of services subscribed
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = (df[service_cols] == 'Yes').sum(axis=1)

    print(f"Features after engineering: {df.shape[1]}")
    return df

def build_preprocessor(df: pd.DataFrame):
    target = 'Churn'
    X = df.drop(columns=[target])

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    return preprocessor, cat_cols, num_cols

def process_and_save(raw_path: str, output_dir: str):
    df = load_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/processed.csv", index=False)
    print(f"Processed data saved to {output_dir}/processed.csv")
    return df

if __name__ == "__main__":
    process_and_save(
        raw_path="../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        output_dir="../data/processed"
    )