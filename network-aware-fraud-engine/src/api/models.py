from pydantic import BaseModel
from typing import List

class TransactionInput(BaseModel):
    TransactionAmt: float
    card1: int
    merchant_id: str
    # Add other feats...

class FraudResponse(BaseModel):
    score: float
    explanation: str