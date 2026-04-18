import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import joblib
import os
import json
from datetime import datetime

PROCESSED_PATH = "../data/processed/processed.csv"
MODELS_DIR = "../models"
MLFLOW_DIR = "../mlflow_runs"

def load_processed(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y

def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    return preprocessor

def train_and_log(model_name: str, model, X_train, X_test, y_train, y_test, preprocessor):
    mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLFLOW_DIR)}")
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Cross validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"\n{model_name}")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")

        # Log to MLflow
        mlflow.log_param("model", model_name)
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(pipeline, "model")

        return pipeline, roc_auc

def save_best_model(pipeline, model_name: str, metrics: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{MODELS_DIR}/best_model.pkl"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "metrics": metrics
    }
    with open(f"{MODELS_DIR}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest model saved: {model_name}")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    X, y = load_processed(PROCESSED_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = {}
    pipelines = {}

    for name, model in models.items():
        pipeline, roc_auc = train_and_log(
            name, model, X_train, X_test, y_train, y_test, preprocessor
        )
        results[name] = roc_auc
        pipelines[name] = pipeline

    best_name = max(results, key=results.get)
    best_pipeline = pipelines[best_name]

    save_best_model(best_pipeline, best_name, {
        "roc_auc": round(results[best_name], 4),
        "accuracy": round(accuracy_score(y_test, best_pipeline.predict(X_test)), 4),
        "f1_score": round(f1_score(y_test, best_pipeline.predict(X_test)), 4)
    })