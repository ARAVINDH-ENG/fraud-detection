import joblib
import pandas as pd
import shap
from model_service.version_manager import get_active_version

version_info  = get_active_version()
model         = joblib.load(version_info["path"])
MODEL_VERSION = version_info["version"]

shap_explainer = shap.TreeExplainer(model)

EXPECTED_COLUMNS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
    'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
    'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23',
    'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

def get_fraud_probability(features: dict) -> dict:
    df = pd.DataFrame([features])
    df = df[EXPECTED_COLUMNS]

    probability   = float(model.predict_proba(df)[:, 1][0])
    shap_values   = shap_explainer.shap_values(df)

    feature_impact = dict(
        sorted(
            zip(df.columns, shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
    )

    return {
        "probability":       round(probability, 4),
        "model_version":     MODEL_VERSION,
        "shap_top_features": {
            k: round(float(v), 4)
            for k, v in feature_impact.items()
        }
    }
