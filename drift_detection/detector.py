import numpy as np
import pandas as pd
from scipy import stats
import json
import os
from database.db import SessionLocal
from database.models import Transaction

TRAINING_STATS_PATH = "models/training_stats.json"

def detect_drift(recent_features: list) -> dict:
    if not os.path.exists(TRAINING_STATS_PATH):
        return {"message": "Training stats not found. Run training/train.py first."}

    with open(TRAINING_STATS_PATH, "r") as f:
        training_stats = json.load(f)

    df               = pd.DataFrame(recent_features)
    drift_report     = {}
    drifted_features = []

    for col in df.columns:
        if col not in training_stats:
            continue
        train_mean = training_stats[col]["mean"]
        train_std  = training_stats[col]["std"]
        if train_std == 0:
            continue
        reference = np.random.normal(train_mean, train_std, 1000)
        current   = df[col].dropna().values
        if len(current) < 30:
            continue
        ks_stat, p_value = stats.ks_2samp(reference, current)
        drifted = bool(p_value < 0.05)
        drift_report[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value":      round(float(p_value), 4),
            "drifted":      drifted
        }
        if drifted:
            drifted_features.append(col)

    drift_severity = len(drifted_features) / len(drift_report) if drift_report else 0

    return {
        "total_features_checked":  int(len(drift_report)),
        "drifted_features":        drifted_features,
        "drift_detected":          bool(len(drifted_features) > 0),
        "drift_severity_percent":  round(float(drift_severity * 100), 1),
        "recommendation": (
            "RETRAIN" if drift_severity > 0.2 else
            "MONITOR" if drift_severity > 0.1 else
            "STABLE"
        ),
        "details": drift_report
    }

def get_recent_features(limit: int = 1000) -> list:
    db = SessionLocal()
    try:
        transactions = db.query(Transaction).order_by(
            Transaction.timestamp.desc()
        ).limit(limit).all()
        features = []
        for t in transactions:
            try:
                features.append(json.loads(t.features))
            except Exception:
                continue
        return features
    finally:
        db.close()
