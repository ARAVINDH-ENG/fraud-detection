import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from business_layer.cost_evaluator import evaluate_cost

def test_high_probability_dominant_fraud():
    r = evaluate_cost("REJECT", 0.9, 500.0)
    assert r["dominant_risk"] == "FRAUD"

def test_low_probability_dominant_friction():
    r = evaluate_cost("APPROVE", 0.001, 50.0)
    assert r["dominant_risk"] == "FRICTION"

def test_costs_are_positive():
    r = evaluate_cost("REVIEW", 0.5, 200.0)
    assert r["expected_fraud_loss"] > 0
    assert r["expected_friction_cost"] > 0

def test_amount_reflected_in_fraud_loss():
    # High amount transaction should have higher fraud loss
    r_high = evaluate_cost("REVIEW", 0.8, 5000.0)
    r_low  = evaluate_cost("REVIEW", 0.8, 10.0)
    assert r_high["expected_fraud_loss"] > r_low["expected_fraud_loss"]

def test_transaction_amount_in_output():
    r = evaluate_cost("APPROVE", 0.1, 149.62)
    assert r["transaction_amount"] == 149.62
