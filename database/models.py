from sqlalchemy import Column, String, Float, DateTime, Text, Index
from database.db import Base
from datetime import datetime

class Transaction(Base):
    __tablename__ = "transactions"

    id            = Column(String, primary_key=True)
    features      = Column(Text)
    probability   = Column(Float)
    decision      = Column(String)
    risk_level    = Column(String)
    explanation   = Column(Text)
    status        = Column(String)
    model_version = Column(String)
    timestamp     = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_decision",  "decision"),
        Index("idx_status",    "status"),
        Index("idx_timestamp", "timestamp"),
    )

class ReviewQueue(Base):
    __tablename__ = "review_queue"

    transaction_id = Column(String, primary_key=True)
    review_status  = Column(String, default="PENDING")
    reason         = Column(Text)
    queued_at      = Column(DateTime, default=datetime.utcnow)
    reviewed_at    = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_review_status", "review_status"),
    )
