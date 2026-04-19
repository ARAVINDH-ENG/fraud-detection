import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def sample_transaction():
    data = {f"V{i}": 0.0 for i in range(1, 29)}
    data["Time"]   = 1000.0
    data["Amount"] = 50.0
    return data

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_returns_200():
    r = client.post("/predict", json=sample_transaction())
    assert r.status_code == 200

def test_predict_response_structure():
    r    = client.post("/predict", json=sample_transaction())
    body = r.json()
    for field in ["transaction_id", "decision", "risk_level", "probability", "cost_analysis", "model_version", "shap_features"]:
        assert field in body

def test_predict_decision_is_valid():
    r = client.post("/predict", json=sample_transaction())
    assert r.json()["decision"] in ["APPROVE", "REVIEW", "REJECT"]

def test_predict_probability_in_range():
    r    = client.post("/predict", json=sample_transaction())
    prob = r.json()["probability"]
    assert 0.0 <= prob <= 1.0

def test_cost_analysis_has_amount():
    r  = client.post("/predict", json=sample_transaction())
    ca = r.json()["cost_analysis"]
    assert "transaction_amount" in ca
    assert ca["transaction_amount"] == 50.0

def test_metrics_endpoint():
    r    = client.get("/metrics")
    body = r.json()
    assert r.status_code == 200
    assert "total_transactions" in body
    assert "fraud_rate_percent" in body

def test_review_queue():
    r = client.get("/review")
    assert r.status_code == 200
    assert "count" in r.json()

def test_invalid_amount_rejected():
    bad          = sample_transaction()
    bad["Amount"] = -100.0
    r            = client.post("/predict", json=bad)
    assert r.status_code == 422
