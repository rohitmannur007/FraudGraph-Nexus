from fastapi import FastAPI
from src.api.models import TransactionInput, FraudResponse
from src.models.lightgbm import train_lightgbm  # Load model
import lightgbm as lgb
import pandas as pd
import numpy as np

app = FastAPI()
model = lgb.Booster(model_file='data/processed/lightgbm_model.txt')

@app.post("/predict", response_model=FraudResponse)
def predict_fraud(input: TransactionInput):
    # Pad to full feats (dummy for demo)
    feats = np.zeros(len(model.feature_name()))
    feats[0] = np.log1p(input.TransactionAmt)  # amount_log
    # Map others...
    score = model.predict(feats.reshape(1, -1))[0]
    return FraudResponse(score=score, explanation="High amount flagged.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)