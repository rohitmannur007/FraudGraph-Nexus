import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx

class TemporalGNN(nn.Module):
    def __init__(self, in_dim=1, hid_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr=None):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  # Node embeddings

def train_gnn(graph_edges_path='data/synthetic/graph_edges.csv', epochs=50):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load graph
    edge_df = pd.read_csv(graph_edges_path)
    G_nx = nx.from_pandas_edgelist(edge_df, 'source', 'target', edge_attr=['weight', 'time'])
    data = from_networkx(G_nx)
    data.x = torch.tensor([[1.0]] * len(data.nodes), dtype=torch.float).to(device)  # Dummy node feats
    data.edge_attr = torch.tensor(data.edge_attr.values if 'edge_attr' in data else [], dtype=torch.float).to(device)
    data = data.to(device)

    model = TemporalGNN(in_dim=1, out_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()  # Supervised on fraud_risk (placeholder)

    # Fake labels for demo (use real fraud propagation)
    labels = torch.tensor([G_nx.nodes[n].get('fraud_risk', 0.0) for n in G_nx.nodes], dtype=torch.float).to(device)
    data.y = labels

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        emb = model(data.x, data.edge_index)
        loss = criterion(emb.mean(dim=1), labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Extract txn embeddings
    with torch.no_grad():
        emb = model(data.x, data.edge_index).cpu().numpy()
    emb_df = pd.DataFrame(emb, index=[n for n in G_nx.nodes if 'txn_' in n])
    emb_df['TransactionID'] = emb_df.index.str.extract('(\d+)').astype(int)
    emb_df.to_pickle('data/processed/gnn_embeddings.pkl')

    torch.save(model.state_dict(), 'data/processed/gnn_model.pth')
    return model

if __name__ == "__main__":
    train_gnn()