import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.models.lightgbm import train_lightgbm, custom_loss
from sklearn.metrics import precision_recall_curve, roc_auc_score

def validate_model(features_path='data/processed/features.parquet'):
    df = pd.read_parquet(features_path).sort_values('TransactionDT')
    feat_cols = [c for c in df.columns if c not in ['TransactionID', 'TransactionDT', 'isFraud']]
    X, y, dt = df[feat_cols], df['isFraud'], df['TransactionDT']

    tscv = TimeSeriesSplit(n_splits=3)
    aucs, losses = [], []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = train_lightgbm(pd.concat([X_tr, y_tr], axis=1).to_parquet('temp.parquet'))  # Temp for train
        preds = model.predict(X_te)

        auc = roc_auc_score(y_te, preds)
        aucs.append(auc)
        losses.append(custom_loss(y_te.values, preds))

        # Precision@K (top 1% flagged)
        k = int(0.01 * len(preds))
        top_k = np.argsort(preds)[-k:]
        prec_k = y_te.iloc[top_k].mean()
        print(f"Fold Prec@1%: {prec_k:.4f}")

        # FPR @ 1% friction (e.g., alert rate)
        fpr, tpr, _ = precision_recall_curve(y_te, preds)
        fpr_1pct = 0.01
        idx = np.argmin(np.abs(fpr - fpr_1pct))
        print(f"FPR @1% friction: {fpr[idx]:.4f}")

    print(f"Mean AUC: {np.mean(aucs):.4f}, Mean Loss Saved: {-np.mean(losses):.2f}$")

    # Reliability diagram (plot with matplotlib)
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_te, preds, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
    plt.savefig('notebooks/reliability.png')
    plt.close()

if __name__ == "__main__":
    validate_model()