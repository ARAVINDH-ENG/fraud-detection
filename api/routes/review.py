from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.db import get_db
from database.models import ReviewQueue, Transaction
from datetime import datetime

router = APIRouter()

@router.get("/review")
def get_pending_reviews(db: Session = Depends(get_db)):
    pending = db.query(ReviewQueue).filter(
        ReviewQueue.review_status == "PENDING"
    ).all()
    return {
        "pending_reviews": [
            {
                "transaction_id": r.transaction_id,
                "review_status":  r.review_status,
                "reason":         r.reason,
                "queued_at":      r.queued_at
            }
            for r in pending
        ],
        "count": len(pending)
    }

@router.post("/review/{transaction_id}")
def submit_review(
    transaction_id: str,
    human_decision: str,
    db: Session = Depends(get_db)
):
    valid = {"APPROVED", "REJECTED"}
    if human_decision.upper() not in valid:
        raise HTTPException(status_code=400, detail=f"Must be one of: {valid}")

    review = db.query(ReviewQueue).filter(
        ReviewQueue.transaction_id == transaction_id
    ).first()

    if not review:
        raise HTTPException(status_code=404, detail="Not found in review queue")

    review.review_status = human_decision.upper()
    review.reviewed_at   = datetime.utcnow()

    transaction = db.query(Transaction).filter(
        Transaction.id == transaction_id
    ).first()
    if transaction:
        transaction.status = human_decision.upper()

    db.commit()

    return {
        "transaction_id": transaction_id,
        "updated_status": human_decision.upper(),
        "reviewed_at":    datetime.utcnow().isoformat()
    }
