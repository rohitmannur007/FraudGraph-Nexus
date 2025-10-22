import shap
import pandas as pd
from src.models.lightgbm import train_lightgbm
import torch
from src.models.gnn import TemporalGNN

def explain_models():
    # SHAP for LightGBM
    df = pd.read_parquet('data/processed/features.parquet')
    feat_cols = [c for c in df.columns if c not in ['TransactionID', 'TransactionDT', 'isFraud']]
    X = df[feat_cols].sample(1000)  # Subset for speed
    model = train_lightgbm(pd.concat([X, df.loc[X.index, 'isFraud']], axis=1).to_parquet('temp.parquet'))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('notebooks/shap_summary.png')

    # GNN Saliency (gradient-based)
    model_gnn = TemporalGNN()
    model_gnn.load_state_dict(torch.load('data/processed/gnn_model.pth'))
    model_gnn.eval()
    # Dummy input for saliency (adapt to your graph)
    x_dummy = torch.randn(1, 1, requires_grad=True)
    emb = model_gnn(x_dummy, torch.tensor([[0,1]]))  # Dummy edge
    emb[0, 0].backward()  # Saliency on first dim
    saliency = x_dummy.grad.abs().item()
    print(f"GNN Saliency Score: {saliency:.4f} (highlights temporal edges)")

if __name__ == "__main__":
    explain_models()