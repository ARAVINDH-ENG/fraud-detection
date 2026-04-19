import os

MODEL_PATH = "models/fraud_model.pkl"
TRAINING_STATS_PATH = "models/training_stats.json"
VERSION_REGISTRY = "models/version_registry.json"

DB_URL = os.getenv("DATABASE_URL", "sqlite:///fraud.db")

LLM_MODEL = "models/gemini-1.5-flash"

THRESHOLDS = {
    "approve": 0.3,
    "review":  0.7
}

COST_MATRIX = {
    "false_positive": 10,
    "false_negative": 500
}
