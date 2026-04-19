import requests, time, os

while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    try:
        r = requests.get("http://localhost:8000/metrics", timeout=3)
        m = r.json()
        print("=" * 45)
        print("   FRAUDSHIELD — LIVE METRICS")
        print("=" * 45)
        print(f"  Total Transactions : {m['total_transactions']}")
        print(f"  Approved           : {m['approved']}")
        print(f"  Rejected           : {m['rejected']}")
        print(f"  Under Review       : {m['under_review']}")
        print(f"  Fraud Rate         : {m['fraud_rate_percent']}%")
        print(f"  Approval Rate      : {m['approval_rate_percent']}%")
        print(f"  Avg Probability    : {m['avg_fraud_probability']}")
        print("=" * 45)
        print(f"  Refreshing every 3s... Ctrl+C to stop")
    except Exception as e:
        print(f"API error: {e}")
    time.sleep(3)
