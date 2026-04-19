from config import THRESHOLDS, COST_MATRIX

def make_decision(probability: float) -> dict:
    fp_cost = COST_MATRIX["false_positive"]
    fn_cost = COST_MATRIX["false_negative"]

    if probability < THRESHOLDS["approve"]:
        decision   = "APPROVE"
        risk_level = "LOW"
    elif probability < THRESHOLDS["review"]:
        decision   = "REVIEW"
        risk_level = "MEDIUM"
    else:
        decision   = "REJECT"
        risk_level = "HIGH"

    return {
        "decision":    decision,
        "risk_level":  risk_level,
        "probability": probability,
        "fp_cost":     fp_cost,
        "fn_cost":     fn_cost
    }
