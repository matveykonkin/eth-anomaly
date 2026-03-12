# Anomaly Detection in Ethereum On-Chain Data

> Academic research project — Saint Petersburg State University, 2025  
> Faculty of Mathematics and Mechanics · AI & Data Science

---

## Overview

This project explores machine learning approaches for detecting anomalous transactions in the Ethereum network. The core hypothesis: enriching transaction features with **graph-based metrics** derived from the wallet interaction network improves anomaly detection precision — especially for coordinated fraud patterns.

Two unsupervised algorithms are compared across two feature sets (baseline vs. baseline + graph):

| Model | Features | Anomalies Found | % of Dataset | Notes |
|---|---|---|---|---|
| Isolation Forest | Baseline | 553 | 10% | Matches contamination target; high false positive rate |
| One-Class SVM | Baseline | 279 | 5% | More conservative; high anomaly scores |
| Isolation Forest | Baseline + Graph | 553 | 10% | No change — insensitive to graph features |
| **One-Class SVM** | **Baseline + Graph** | **111** | **2%** | **Significant precision gain; fewer false positives** |

**Key finding:** One-Class SVM with graph features reduced flagged anomalies from 279 → 111, filtering out statistically noisy transactions (e.g., arbitrage bots) while preserving high-confidence anomalies. Isolation Forest showed no sensitivity to graph enrichment.

Model agreement analysis (5,531 transactions total):
- Both flagged as anomaly: **205**
- Both flagged as normal: **4,904**  
- Only IF flagged: **348** — likely statistical noise
- Only SVM flagged: **74** — likely complex network anomalies
- Overall agreement: **92.4%**

---

## Data

- **Source:** Etherscan API
- **Wallets:** Binance hot wallet (`0x28C6c06298d514Db089934071355E5743bf21d60`) + wallets flagged for suspicious activity
- **Period:** November 2022 – November 2025
- **Volume:** 5,531 transactions after cleaning

---

## Features

**Baseline features:**
- `gas_used_ratio`, `gas_price_deviation`
- `input_length`
- `wallet_age_days`, `tx_count_7d`
- `is_new_wallet`, `is_high_frequency`
- `eth_value`, `eth_gas_cost` (converted from Wei)
- `day_of_week`, `hour` (extracted from timestamp)

**Graph features** (computed via NetworkX on directed wallet-transaction graph):
- `in_degree`, `out_degree` — incoming/outgoing transaction count
- `pagerank` — wallet importance in the network
- `betweenness` — how often a wallet lies on shortest paths between others
- `clustering` — clustering coefficient
- `second_order_neighbors` — number of wallets at distance 2

---

## Project Structure

```
eth-anomaly/
├── data/
│   ├── raw/                  # Raw data from Etherscan API
│   └── processed/            # Cleaned feature datasets
├── notebooks/                # EDA and model comparison notebooks
├── src/
│   ├── data/
│   │   ├── data_loader.py    # Etherscan API data collection
│   │   └── data_processor.py # Cleaning, feature engineering pipeline
│   ├── visual/
│   │   └── plots.py          # EDA visualizations
│   └── models/
│       ├── isolation_forest.py
│       ├── one_class_svm.py
│       ├── graph_features.py         # Graph construction & centrality metrics
│       └── data_models_comparison.py # Side-by-side model evaluation
└── README.md
```

---

## Methodology

**1. Data Collection**  
Transactions fetched via Etherscan API, sorted ascending by timestamp, limited to 3,000 per wallet per request. Saved as `etherscan_data.csv`.

**2. Preprocessing**  
- Cast numeric columns (`value`, `gas`, `gasPrice`, `gasUsed`, `nonce`, `blockNumber`) to `float64`
- Converted Unix timestamps to `datetime`
- Removed duplicate and failed transactions
- Engineered `eth_value` and `eth_gas_cost` (Wei → ETH)

**3. Models**  
Both models trained with `contamination=0.1` and features normalized via `StandardScaler`.

- **Isolation Forest** — isolates anomalies by random recursive partitioning; effective for point outliers
- **One-Class SVM** — learns a boundary around "normal" behavior; more conservative but sensitive to network context

**4. Graph Enrichment**  
A directed graph was constructed where nodes = wallets, edges = transactions. Five centrality metrics computed per node and merged into the feature matrix. Notably, wallets assumed to be independent were found to be connected in the transaction graph.

---

## Tech Stack

- Python 3.x
- `scikit-learn` — Isolation Forest, One-Class SVM, StandardScaler
- `pandas`, `numpy` — data processing
- `networkx` — graph construction and centrality metrics
- `matplotlib` — visualization
- `Jupyter Notebook` — exploratory analysis

---

## Conclusions

- Graph features **significantly improved One-Class SVM** precision by filtering out transactions that are statistical outliers but part of normal network activity (e.g., high-frequency arbitrage bots)
- **Isolation Forest** is better suited for fast screening of point anomalies (e.g., extreme gas fees); graph enrichment had no effect on its output
- The two models capture **fundamentally different anomaly types**: point deviations (IF) vs. semantic/behavioral deviations (OCSVM)
- Proposed two-tier architecture: IF for fast first-pass screening → OCSVM + graph features for deep analysis of suspicious candidates

**Directions for future work:**
- Graph Neural Networks (GNN) for subgraph-level anomaly detection
- Ensemble/stacking of IF and OCSVM
- Validation on labeled benchmark datasets
- Adaptation to other chains (Bitcoin, BSC, Solana)

---

## Author

**Matvey Konkin** · [github.com/matveykonkin](https://github.com/matveykonkin) · matvejkonkin4@gmail.com  
2nd-year student, AI & Data Science, SPbSU
