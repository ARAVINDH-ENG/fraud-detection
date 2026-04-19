from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from api.schemas.transaction import TransactionRequest, TransactionResponse
from model_service.predict import get_fraud_probability
from decision_engine.engine import make_decision
from business_layer.cost_evaluator import evaluate_cost
from llm_service.explainer import explain_decision
from database.db import get_db, SessionLocal
from database.models import Transaction, ReviewQueue
from datetime import datetime
import uuid
import json

router = APIRouter()

def update_explanation_background(
    transaction_id: str,
    probability: float,
    shap_features: dict
):
    db = SessionLocal()
    try:
        explanation = explain_decision(probability, shap_features)
        transaction = db.query(Transaction).filter(
            Transaction.id == transaction_id
        ).first()
        if transaction:
            transaction.explanation = explanation
            db.commit()
        review = db.query(ReviewQueue).filter(
            ReviewQueue.transaction_id == transaction_id
        ).first()
        if review:
            review.reason = explanation
            db.commit()
    except Exception as e:
        print(f"Background explanation failed: {e}")
    finally:
        db.close()

@router.post("/predict", response_model=TransactionResponse)
def predict(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    transaction_id = str(uuid.uuid4())
    features       = transaction.model_dump()
    amount         = features.get("Amount", 0.0)

    model_output  = get_fraud_probability(features)
    probability   = model_output["probability"]
    shap_features = model_output["shap_top_features"]
    model_version = model_output["model_version"]

    decision_output = make_decision(probability)
    decision        = decision_output["decision"]
    risk_level      = decision_output["risk_level"]

    # Pass amount so cost analysis reflects actual transaction value
    cost_output = evaluate_cost(decision, probability, amount)

    db_transaction = Transaction(
        id            = transaction_id,
        features      = json.dumps(features),
        probability   = probability,
        decision      = decision,
        risk_level    = risk_level,
        explanation   = "Generating...",
        status        = "PENDING" if decision == "REVIEW" else "CLOSED",
        model_version = model_version,
        timestamp     = datetime.utcnow()
    )
    db.add(db_transaction)

    if decision == "REVIEW":
        review = ReviewQueue(
            transaction_id = transaction_id,
            review_status  = "PENDING",
            reason         = "Generating...",
            queued_at      = datetime.utcnow()
        )
        db.add(review)

    db.commit()

    background_tasks.add_task(
        update_explanation_background,
        transaction_id,
        probability,
        shap_features
    )

    return TransactionResponse(
        transaction_id = transaction_id,
        decision       = decision,
        risk_level     = risk_level,
        probability    = probability,
        explanation    = None,
        cost_analysis  = cost_output,
        model_version  = model_version,
        shap_features  = shap_features
    )

@router.get("/explanation/{transaction_id}")
def get_explanation(transaction_id: str, db: Session = Depends(get_db)):
    transaction = db.query(Transaction).filter(
        Transaction.id == transaction_id
    ).first()

    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    is_ready = (
        transaction.explanation is not None and
        transaction.explanation != "Generating..."
    )

    return {
        "transaction_id":    transaction_id,
        "decision":          transaction.decision,
        "risk_level":        transaction.risk_level,
        "probability":       transaction.probability,
        "explanation":       transaction.explanation,
        "explanation_ready": is_ready,
        "model_version":     transaction.model_version,
        "timestamp":         transaction.timestamp
    }
