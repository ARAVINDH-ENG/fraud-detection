from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from database.db import get_db
from database.models import Transaction
from drift_detection.detector import detect_drift, get_recent_features

router = APIRouter()

@router.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    total    = db.query(Transaction).count()
    approved = db.query(Transaction).filter(Transaction.decision == "APPROVE").count()
    rejected = db.query(Transaction).filter(Transaction.decision == "REJECT").count()
    review   = db.query(Transaction).filter(Transaction.decision == "REVIEW").count()

    avg_probability = db.query(func.avg(Transaction.probability)).scalar() or 0.0

    fraud_rate    = round(rejected / total * 100, 2) if total > 0 else 0
    review_rate   = round(review   / total * 100, 2) if total > 0 else 0
    approval_rate = round(approved / total * 100, 2) if total > 0 else 0

    return {
        "total_transactions":      total,
        "approved":                approved,
        "rejected":                rejected,
        "under_review":            review,
        "fraud_rate_percent":      fraud_rate,
        "review_rate_percent":     review_rate,
        "approval_rate_percent":   approval_rate,
        "avg_fraud_probability":   round(float(avg_probability), 4)
    }

@router.get("/explanations")
def get_all_explanations(
    decision_filter: str = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    query = db.query(Transaction)
    if decision_filter:
        query = query.filter(Transaction.decision == decision_filter.upper())

    transactions = query.order_by(Transaction.timestamp.desc()).limit(limit).all()

    return {
        "total": len(transactions),
        "transactions": [
            {
                "transaction_id":    t.id,
                "decision":          t.decision,
                "risk_level":        t.risk_level,
                "probability":       t.probability,
                "explanation":       t.explanation,
                "explanation_ready": (
                    t.explanation is not None and
                    t.explanation != "Generating..."
                ),
                "timestamp":         t.timestamp
            }
            for t in transactions
        ]
    }

@router.get("/drift")
def check_drift():
    recent = get_recent_features(limit=500)
    if len(recent) < 30:
        return {
            "message": "Insufficient data for drift analysis",
            "minimum_required": 30
        }
    return detect_drift(recent)
