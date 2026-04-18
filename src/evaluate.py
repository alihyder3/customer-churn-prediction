import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve
)
import joblib
import os

MODELS_DIR = "../models"
DOCS_DIR = "../docs"

def evaluate_model(model_path: str, data_path: str):
    os.makedirs(DOCS_DIR, exist_ok=True)

    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'], ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC AUC = {auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[2].plot(recall, precision, color='coral', lw=2)
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')

    plt.tight_layout()
    plt.savefig(f"{DOCS_DIR}/evaluation.png", dpi=150)
    print(f"\nEvaluation charts saved to {DOCS_DIR}/evaluation.png")

if __name__ == "__main__":
    evaluate_model(
        model_path=f"{MODELS_DIR}/best_model.pkl",
        data_path="../data/processed/processed.csv"
    )