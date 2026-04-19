import requests
import random
import time
import numpy as np

API_URL = "http://localhost:8000/predict"

def generate_transaction(is_fraud: bool = False) -> dict:
    if is_fraud:
        features = {
            "Time":   random.uniform(0, 172792),
            "Amount": random.uniform(1000, 8000),
            "V1":  round(float(np.random.normal(-3.0, 0.5)), 6),
            "V2":  round(float(np.random.normal(3.0,  0.5)), 6),
            "V3":  round(float(np.random.normal(-5.0, 0.5)), 6),
            "V4":  round(float(np.random.normal(4.0,  0.5)), 6),
            "V5":  round(float(np.random.normal(-2.0, 0.5)), 6),
            "V6":  round(float(np.random.normal(-1.0, 0.3)), 6),
            "V7":  round(float(np.random.normal(-5.0, 0.5)), 6),
            "V8":  round(float(np.random.normal(0.5,  0.3)), 6),
            "V9":  round(float(np.random.normal(-3.0, 0.5)), 6),
            "V10": round(float(np.random.normal(-5.0, 0.5)), 6),
            "V11": round(float(np.random.normal(4.0,  0.5)), 6),
            "V12": round(float(np.random.normal(-8.0, 0.5)), 6),
            "V13": round(float(np.random.normal(0.5,  0.3)), 6),
            "V14": round(float(np.random.normal(-10.0, 0.5)), 6),
            "V15": round(float(np.random.normal(0.2,  0.3)), 6),
            "V16": round(float(np.random.normal(-4.0, 0.5)), 6),
            "V17": round(float(np.random.normal(-8.0, 0.5)), 6),
            "V18": round(float(np.random.normal(-3.0, 0.5)), 6),
            "V19": round(float(np.random.normal(0.5,  0.3)), 6),
            "V20": round(float(np.random.normal(0.5,  0.3)), 6),
            "V21": round(float(np.random.normal(0.8,  0.3)), 6),
            "V22": round(float(np.random.normal(0.3,  0.3)), 6),
            "V23": round(float(np.random.normal(-0.2, 0.3)), 6),
            "V24": round(float(np.random.normal(0.2,  0.3)), 6),
            "V25": round(float(np.random.normal(0.5,  0.3)), 6),
            "V26": round(float(np.random.normal(0.3,  0.3)), 6),
            "V27": round(float(np.random.normal(0.5,  0.3)), 6),
            "V28": round(float(np.random.normal(0.2,  0.3)), 6),
        }
    else:
        features = {
            "Time":   random.uniform(0, 172792),
            "Amount": random.uniform(1, 500),
        }
        for i in range(1, 29):
            features[f"V{i}"] = round(float(np.random.normal(0, 1)), 6)

    return features

def simulate(fraud_rate: float = 0.002, interval: float = 0.5):
    print(f"Simulation started | fraud_rate={fraud_rate} | interval={interval}s")
    count = 0
    while True:
        is_fraud    = random.random() < fraud_rate
        transaction = generate_transaction(is_fraud=is_fraud)
        try:
            response = requests.post(API_URL, json=transaction, timeout=5)
            result   = response.json()
            print(
                f"[{count:04d}] "
                f"fraud={str(is_fraud):<5} | "
                f"prob={result.get('probability', 0):.4f} | "
                f"decision={result.get('decision', '?'):<7} | "
                f"risk={result.get('risk_level', '?')}"
            )
        except Exception as e:
            print(f"[{count:04d}] API error: {e}")
        count += 1
        time.sleep(interval)

if __name__ == "__main__":
    simulate(fraud_rate=0.05, interval=0.3)
