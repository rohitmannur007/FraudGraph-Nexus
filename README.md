# FraudGraph-Nexus — Network‑Aware Fraud Detection (Suggested 10/10 README)

**One‑line:** A reproducible, production‑grade graph‑ML system for detecting fraud rings and coordinated attacks using temporal graph neural networks, explainability, and streaming inference.

---

## Table of contents

* About
* Highlights (what makes this 10/10)
* Architecture overview
* Installation & Quickstart
* Dataset(s) & preprocessing
* Model zoo (implemented & recommended)
* Training / Evaluation
* Inference & Deployment (Docker, Streamlit/Gradio demo)
* Metrics & Baselines
* Explainability & Forensics
* Experiments & Results (example table)
* How to reproduce (commands)
* Roadmap / Future work
* Contributing
* License

---

## About

FraudGraph‑Nexus is an end‑to‑end framework that constructs transaction/person/entity graphs and trains temporal graph neural networks (TGN/GAT/GraphSAGE+RNN) to detect anomalous users, accounts, and coordinated fraud rings. It focuses on real‑world operational concerns: class imbalance, streaming inference, explainability, and adversarial robustness.

---

## Highlights (what makes this 10/10)

* **Temporal graph modeling** (not just static snapshots) to catch time‑coordinated attacks.
* **Network‑level detection:** node & subgraph (ring) scoring + link prediction for suspicious ties.
* **Explainability:** integrated GNNExplainer and counterfactual rule extraction for investigator‑friendly explanations.
* **Streaming pipeline:** Kafka → transform → incremental GNN inference for low latency.
* **Reproducible experiments:** seedable notebooks, `requirements.txt`, Dockerfile, and CI tests.
* **Benchmarks & baselines:** logistic regression / XGBoost / LightGBM on tabular features + static GNN baseline.
* **Adversarial testing:** attacker simulator to validate robustness.

---

## Architecture overview

1. **Data ingestion:** csv/parquet → standardize → feature engineering.
2. **Graph construction:** multi‑relation heterogeneous graph (accounts, cards, devices, IPs) with temporal edges.
3. **Modeling:** TGN / GAT / GraphSAGE variants + attention pooling for subgraph scoring.
4. **Evaluation & explainability:** per‑node and subgraph metrics, explainers, and investigator dashboard.
5. **Deployment:** REST API + stream consumer with Docker Compose and optional Kubernetes/Helm.

(Include a small ASCII/mermaid diagram placeholder.)

---

## Installation & Quickstart

**Prereqs:** Python 3.10+, Docker, Git.

```bash
# clone
git clone https://github.com/rohitmannur007/FraudGraph-Nexus.git
cd FraudGraph-Nexus/network-aware-fraud-engine

# python venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run unit tests
pytest -q

# start demo (streamlit)
streamlit run demo/app.py
```

---

## Datasets & preprocessing

* Provide links (or scripts) to synthetic dataset generator + instructions to transform public transaction datasets into a graph. Document schema and privacy notes.

---

## Model zoo

* `models/graphsage_runner.py` — baseline
* `models/gat_tgn.py` — recommended production model
* `models/subgraph_pooler.py` — ring scorer

Include hyperparameters and how to reproduce exact runs.

---

## Training & Evaluation

* Explain the train/val/test split strategy (time‑based split), handling class imbalance (Focal Loss, oversampling, negative sampling strategies), and metrics (AUC-PR, recall@k, precision@k, ring detection F1).

---

## Inference & Deployment

* Dockerfile + docker-compose for local deployment.
* Stream consumer and REST API (`/predict`, `/explain`) endpoints.
* Minimal `monitoring/` showing model drift detection using population stats.

---

## Explainability & Forensics

* Use GNNExplainer and SHAP‑like local attributions; export investigator reports (PDF/HTML) with key evidence edges and suggested next actions.

---

## Experiments & Results (example)

| Model                 |   AUC-PR | Recall@100 | Precision@100 |  Ring F1 |
| --------------------- | -------: | ---------: | ------------: | -------: |
| Tabular XGBoost       |     0.12 |       0.32 |          0.28 |     0.21 |
| GraphSAGE static      |     0.28 |       0.56 |          0.44 |     0.43 |
| TGN + subgraph pooler | **0.52** |   **0.78** |      **0.69** | **0.66** |

(Replace with real numbers once experiments finish.)

---

## Roadmap / High‑impact further work

* Real‑time streaming + CI/CD for model updates.
* Multi‑modal signals (text + transaction + device telemetry).
* Formal privacy & compliance notes for production.
* Model card + data card for transparency.

---

## How to reproduce (commands)

Short, copy‑paste commands to reproduce the main results and run the demo.

---

## Contributing

Add an accessible CONTRIBUTING.md and CODE_OF_CONDUCT.

---

## License

Add an OSI license (MIT/Apache‑2.0) and `LICENSE` file.

---

*Generated suggested README to make the repo presentation and reproducibility 10/10.*

