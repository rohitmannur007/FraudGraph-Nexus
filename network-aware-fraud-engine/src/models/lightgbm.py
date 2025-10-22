import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

def custom_loss(y_true, y_pred, fn_cost=1000, fp_cost=10):
    """Cost-sensitive eval: expected dollar loss."""
    y_pred = y_pred.reshape(y_true.shape)
    tn, fp, fn, tp = pd.crosstab(y_true, y_pred > 0.5, rownames=['actual'], colnames=['pred']).values.ravel()
    return -(tp * 0 + tn * 0 + fp * fp_cost + fn * fn_cost)  # Negative for maximization

def train_lightgbm(features_path='data/processed/features.parquet'):
    df = pd.read_parquet(features_path)
    df = df.dropna()

    # Features (exclude ID, label, time)
    feat_cols = [c for c in df.columns if c not in ['TransactionID', 'TransactionDT', 'isFraud']]
    X = df[feat_cols]
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'is_unbalance': True,  # Handles imbalance
        'device': 'cpu'  # M2 CPU fine
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        feval=lambda y_true, y_pred: ('custom_loss', custom_loss(y_true, y_pred), True),
        callbacks=[lgb.early_stopping(10)]
    )

    # Save
    model.save_model('data/processed/lightgbm_model.txt')
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    print(f"AUC: {auc:.4f}, Custom Loss: {custom_loss(y_test.values, preds):.2f}")

    return model

if __name__ == "__main__":
    train_lightgbm()