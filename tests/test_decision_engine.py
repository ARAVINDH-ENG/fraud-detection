import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_engine.engine import make_decision

def test_low_probability_approve():
    r = make_decision(0.1)
    assert r["decision"] == "APPROVE"
    assert r["risk_level"] == "LOW"

def test_medium_probability_review():
    r = make_decision(0.5)
    assert r["decision"] == "REVIEW"
    assert r["risk_level"] == "MEDIUM"

def test_high_probability_reject():
    r = make_decision(0.85)
    assert r["decision"] == "REJECT"
    assert r["risk_level"] == "HIGH"

def test_boundary_just_below_approve():
    r = make_decision(0.299)
    assert r["decision"] == "APPROVE"

def test_boundary_exactly_at_review():
    r = make_decision(0.3)
    assert r["decision"] == "REVIEW"

def test_boundary_exactly_at_reject():
    r = make_decision(0.7)
    assert r["decision"] == "REJECT"

def test_zero_probability():
    r = make_decision(0.0)
    assert r["decision"] == "APPROVE"

def test_full_probability():
    r = make_decision(1.0)
    assert r["decision"] == "REJECT"

def test_output_contains_costs():
    r = make_decision(0.5)
    assert "fp_cost" in r
    assert "fn_cost" in r
    assert r["fn_cost"] > r["fp_cost"]
