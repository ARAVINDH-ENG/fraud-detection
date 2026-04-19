import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import joblib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_service.version_manager import register_model

def compute_training_stats(X_train: pd.DataFrame):
    stats_dict = {}
    for col in X_train.columns:
        stats_dict[col] = {
            "mean": float(X_train[col].mean()),
            "std":  float(X_train[col].std()),
            "p25":  float(X_train[col].quantile(0.25)),
            "p75":  float(X_train[col].quantile(0.75))
        }
    os.makedirs("models", exist_ok=True)
    with open("models/training_stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)
    print("Training stats saved.")

def train():
    df = pd.read_csv("data/creditcard.csv")
    X  = df.drop(columns=["Class"])
    y  = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg   = (y_train == 0).sum()
    pos   = (y_train == 1).sum()
    ratio = neg / pos
    print(f"Class ratio (neg/pos): {ratio:.1f}")

    model = XGBClassifier(
        scale_pos_weight=ratio,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4)
    }

    print("\n=== MODEL EVALUATION ===")
    for k, v in metrics.items():
        print(f"{k:12}: {v}")
    print("\n", classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_model.pkl")
    print("Model saved → models/fraud_model.pkl")

    compute_training_stats(X_train)
    register_model("models/fraud_model.pkl", metrics)

if __name__ == "__main__":
    train()
