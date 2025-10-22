import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

def generate_synthetic_graph(n_transactions=100000, fraud_rate=0.035):
    """Generate synthetic graph edges + fraud injection."""
    np.random.seed(42)

    # Sample base transactions from Kaggle
    tx = pd.read_csv('data/raw/train_transaction.csv', nrows=n_transactions)
    tx = tx[['TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud', 'card1', 'card2', 'merchant_id']].dropna(subset=['card1', 'merchant_id'])
    tx['TransactionDT'] = pd.to_datetime(tx['TransactionDT'], unit='s', origin='2017-12-01')  # Temporal

    # Synthetic nodes: merchants (M), cards (C), devices (D), IPs (I)
    merchants = tx['merchant_id'].unique()[:500]  # Subset for speed
    cards = tx['card1'].unique()[:1000]
    devices = np.random.choice(['dev_' + str(i) for i in range(2000)], size=len(tx), replace=True)
    ips = np.random.choice(['ip_' + str(i) for i in range(5000)], size=len(tx), replace=True)
    tx['device'] = devices
    tx['ip_address'] = ips

    # Build bipartite graph: edges between txn nodes and entities
    G = nx.Graph()
    G.add_nodes_from([f'txn_{tid}' for tid in tx['TransactionID']], bipartite=0)  # Txns
    G.add_nodes_from(merchants, bipartite=1, type='merchant')
    G.add_nodes_from(cards, bipartite=1, type='card')
    G.add_nodes_from(set(devices), bipartite=1, type='device')
    G.add_nodes_from(set(ips), bipartite=1, type='ip')

    edges = []
    for _, row in tx.iterrows():
        tid = f'txn_{row["TransactionID"]}'
        edges.extend([(tid, row['merchant_id'], {'weight': row['TransactionAmt'], 'time': row['TransactionDT']})])
        edges.extend([(tid, row['card1'], {'weight': row['TransactionAmt'], 'time': row['TransactionDT']})])
        edges.extend([(tid, row['device'], {'weight': row['TransactionAmt'], 'time': row['TransactionDT']})])
        edges.extend([(tid, row['ip_address'], {'weight': row['TransactionAmt'], 'time': row['TransactionDT']})])

    G.add_edges_from(edges[:100000])  # Sample 100k edges for M2 speed

    # Fraud injection: Boost fraud on high-degree nodes (long-tail)
    fraud_txns = tx[tx['isFraud'] == 1]
    for _, row in fraud_txns.iterrows():
        neighbors = list(G.neighbors(f'txn_{row["TransactionID"]}'))
        for neigh in neighbors[:3]:  # Propagate to 3 neighbors
            if np.random.rand() < 0.1:  # 10% propagation rate
                G.nodes[neigh]['fraud_risk'] = G.nodes.get(neigh, {}).get('fraud_risk', 0) + 0.5

    # Export
    edge_df = pd.DataFrame(G.edges(data=True))
    edge_df.to_csv('data/synthetic/graph_edges.csv', index=False)

    # Inject back to txns
    tx_with_graph = tx.merge(pd.DataFrame(G.degree()), left_on='TransactionID', right_index=True, how='left', suffixes=('', '_deg'))
    tx_with_graph.to_csv('data/synthetic/fraud_injected_transactions.csv', index=False)

    print(f"Generated {len(G.edges)} edges, {sum(nx.get_node_attributes(G, 'fraud_risk').values())} fraud-risky nodes.")
    return G, tx_with_graph

if __name__ == "__main__":
    G, tx = generate_synthetic_graph()