from src.data_prep import generate_synthetic_graph
from src.features import engineer_features
from src.models.gnn import train_gnn
from src.models.lightgbm import train_lightgbm

if __name__ == "__main__":
    print("Step 1: Generate data")
    _, tx = generate_synthetic_graph()

    print("Step 2: Train GNN")
    train_gnn()

    print("Step 3: Engineer features (with GNN)")
    engineer_features(tx, gnn_emb_path='data/processed/gnn_embeddings.pkl')

    print("Step 4: Train LightGBM")
    train_lightgbm()