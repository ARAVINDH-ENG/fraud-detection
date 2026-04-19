from pydantic import BaseModel, field_validator
from typing import Optional

class TransactionRequest(BaseModel):
    Time:   float
    Amount: float
    V1:  float; V2:  float; V3:  float; V4:  float
    V5:  float; V6:  float; V7:  float; V8:  float
    V9:  float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

    @field_validator("Amount")
    @classmethod
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v

class TransactionResponse(BaseModel):
    transaction_id: str
    decision:       str
    risk_level:     str
    probability:    float
    explanation:    Optional[str] = None
    cost_analysis:  dict
    model_version:  str
    shap_features:  dict
