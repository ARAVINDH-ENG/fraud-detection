from config import COST_MATRIX

def evaluate_cost(decision: str, probability: float, amount: float = 0.0) -> dict:
    """
    Evaluates business cost of each decision.
    Amount is factored into fraud loss calculation for realism.
    Expected fraud loss = P(fraud) x max(amount, fn_cost_base)
    Expected friction cost = P(legit) x fp_cost
    """
    fp_cost = COST_MATRIX["false_positive"]
    fn_cost = COST_MATRIX["false_negative"]

    # If amount provided, use it as the fraud loss basis
    # A ₹10,000 transaction fraud costs more than a ₹50 one
    fraud_loss_basis = max(amount, fn_cost) if amount > 0 else fn_cost

    expected_fraud_loss    = round(probability * fraud_loss_basis, 2)
    expected_friction_cost = round((1 - probability) * fp_cost, 2)

    return {
        "decision":               decision,
        "expected_fraud_loss":    expected_fraud_loss,
        "expected_friction_cost": expected_friction_cost,
        "transaction_amount":     round(amount, 2),
        "dominant_risk": (
            "FRAUD"
            if expected_fraud_loss > expected_friction_cost
            else "FRICTION"
        )
    }
