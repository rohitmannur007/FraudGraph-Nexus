import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import networkx as nx
from src.models.gnn import TemporalGNN  # Forward ref; implement later

def engineer_features(tx_df, graph_edges_path='data/synthetic/graph_edges.csv', gnn_emb_path=None):
    """Full pipeline: tx features + aggregates + graph + GNN."""
    df = tx_df.copy()

    # 1. Transaction-level
    df['amount_log'] = np.log1p(df['TransactionAmt'])
    df['hour'] = df['TransactionDT'].dt.hour
    df['is_night'] = (df['hour'] >= 22) | (df['hour'] <= 4)
    df['device_risk'] = np.random.uniform(0, 1, len(df))  # Placeholder; use real if avail

    # 2. Aggregates (rolling)
    df = df.sort_values('TransactionDT')
    df['amt_1h'] = df.groupby('card1')['amount_log'].rolling('1h', on='TransactionDT').mean().reset_index(0, drop=True)
    df['tx_count_24h'] = df.groupby('card1')['TransactionID'].rolling('24h', on='TransactionDT').count().reset_index(0, drop=True)
    df['amt_30d'] = df.groupby('card1')['amount_log'].rolling('30D', on='TransactionDT').mean().reset_index(0, drop=True)

    # 3. Graph features (deterministic fallback)
    G = nx.from_pandas_edgelist(pd.read_csv(graph_edges_path), 'source', 'target', ['weight', 'time'])
    df['pagerank'] = df['TransactionID'].map({tid: nx.pagerank(G, alpha=0.85).get(f'txn_{tid}', 0) for tid in df['TransactionID']})
    df['degree'] = df['TransactionID'].map({tid: G.degree(f'txn_{tid}') for tid in df['TransactionID']})
    communities = nx.community.greedy_modularity_communities(G)
    comm_id = {node: i for i, comm in enumerate(communities) for node in comm}
    df['comm_id'] = df['TransactionID'].map(lambda tid: comm_id.get(f'txn_{tid}', -1))

    # 4. GNN embeddings (if path provided)
    if gnn_emb_path:
        emb_df = pd.read_pickle(gnn_emb_path)
        df = df.merge(emb_df, on='TransactionID')

    # Scale numerics
    scaler = StandardScaler()
    num_cols = ['amount_log', 'hour', 'amt_1h', 'tx_count_24h', 'amt_30d', 'pagerank', 'degree']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df.to_parquet('data/processed/features.parquet')
    return df, scaler

if __name__ == "__main__":
    from src.data_prep import generate_synthetic_graph
    _, tx = generate_synthetic_graph()
    engineer_features(tx)