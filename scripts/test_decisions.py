import requests

BASE = "http://localhost:8000"

def base_transaction():
    t = {f"V{i}": 0.0 for i in range(1, 29)}
    t["Time"]   = 1000.0
    t["Amount"] = 50.0
    return t

def test_approve():
    t = base_transaction()
    r = requests.post(f"{BASE}/predict", json=t)
    d = r.json()
    print(f"APPROVE TEST: {d['decision']} | prob={d['probability']} | fraud_loss=₹{d['cost_analysis']['expected_fraud_loss']}")

def test_reject():
    t = base_transaction()
    t["V14"] = -15.0; t["V10"] = -12.0; t["V12"] = -10.0; t["Amount"] = 5000.0
    r = requests.post(f"{BASE}/predict", json=t)
    d = r.json()
    print(f"REJECT TEST:  {d['decision']} | prob={d['probability']} | fraud_loss=₹{d['cost_analysis']['expected_fraud_loss']}")

def test_review():
    t = base_transaction()
    t["V14"] = -5.0; t["V10"] = -4.0; t["Amount"] = 800.0
    r = requests.post(f"{BASE}/predict", json=t)
    d = r.json()
    print(f"REVIEW TEST:  {d['decision']} | prob={d['probability']} | fraud_loss=₹{d['cost_analysis']['expected_fraud_loss']}")

if __name__ == "__main__":
    test_approve()
    test_reject()
    test_review()
