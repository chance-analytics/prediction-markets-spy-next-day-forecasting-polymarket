# Prediction Markets for Next-Day S&P 500 Forecasting (Polymarket)
*Polymarket + Feature Engineering + Walk-Forward ML*

## Overview
This project evaluates whether **prediction-market probabilities** can be engineered into useful features for forecasting **next-day S&P 500 direction** (proxied by SPY). Prediction markets aggregate “crowd-implied probabilities,” which may reflect real-time belief and sentiment. The project treats outputs as **research signals**, not trading advice.

A key practical challenge is that strict daily “S&P 500 up/down tomorrow” markets are **sparse and short-lived**, so the approach pivots to a broader **fusion universe** of finance/economics markets and aggregates them into **topic-level features** to reduce missingness.

---

## Data
- **Prediction markets:** Polymarket market metadata (Gamma API) + YES-token price histories (CLOB prices-history)
- **Market targets/controls:** SPY (target label) and VIX (volatility regime control) via Yahoo Finance (`yfinance`)
- **Modeling table:** <500 trading days after alignment and filtering (sample-size constraint)

---

## What I built
### 1) A reproducible Polymarket feature pipeline
- Pull market metadata via **paginated Gamma API** requests
- For binary markets, extract **YES token IDs** and download histories from **CLOB prices-history**
- Cache token histories locally (reduces repeated requests and improves reproducibility)
- Normalize timestamps to **UTC daily keys**, then align to U.S. trading days

### 2) Feature engineering with topic aggregation
Per-market histories are extremely sparse because markets start/end at different times.
To create stable daily inputs, I:
- mapped markets into interpretable **topics** via keyword rules on question text (e.g., **crypto, rates, growth/recession, policy risk**)
- built daily topic features such as probability levels, short-horizon changes, rolling means, and an **active-market count**

### 3) Walk-forward modeling (leakage-resistant)
- **Model:** elastic-net logistic regression (interpretable + regularized)
- **Validation:** walk-forward splits (time-ordered; avoids look-ahead bias)
- **Metrics:** Accuracy (hit rate) and **AUC** (ranking quality)

---

## Key results (high level)
- A simple **“always up”** baseline achieved ~0.59 accuracy (reflecting class imbalance in the sample).
- **Topic-only** features were near baseline accuracy and had weak discrimination (AUC below 0.50 in aggregate).
- Adding **VIX** produced the clearest improvement:
  - **Accuracy:** small lift (mean ~0.60)
  - **AUC:** meaningful lift (mean ~0.63), suggesting volatility regime helps the model rank next-day outcomes even when accuracy gains are limited.
- Adding minimal SPY technicals did not materially improve performance beyond Topics+VIX in this setup.

**Takeaway:** Prediction-market probabilities are feasible to transform into daily features, but in this sample they were not a strong standalone signal for next-day SPY direction. They appear more useful as a **context/risk overlay**, especially when combined with a volatility-regime control like VIX.

---

## Limitations
- **Effective sample size** is small (<500 trading days), limiting statistical confidence.
- Topic coverage is imbalanced; many markets are short-lived or thinly traded.
- Results are **regime-dependent**; performance varies across walk-forward folds.
- This is not a transaction-cost-aware trading backtest.

---

## Repository contents
- White paper and slides (full narrative, figures, and Q&A)
- Data ingestion + feature engineering + modeling artifacts (notebook/scripts)
- Cached market histories (if included) to support reproducibility

---

## How to run (typical)
1) Clone the repo:
```bash
git clone https://github.com/chance-analytics/prediction-markets-spy-next-day-forecasting-polymarket.git
cd prediction-markets-spy-next-day-forecasting-polymarket
```

2) Create an environment and install dependencies (example):
```bash
conda create -n polymarket-spy python=3.11 -y
conda activate polymarket-spy
pip install pandas numpy scikit-learn requests yfinance tqdm pyarrow
```

3) Run the main notebook/script to:
- fetch/update Polymarket market universe + token histories (with caching)
- build daily topic features
- merge SPY/VIX and create next-day labels
- run walk-forward training + evaluation

---

## Next improvements
- Expand and rebalance the topic universe (more stable histories, better coverage)
- Replace keyword topic mapping with embeddings + human spot-check (reduce misclassification)
- Explore alternative targets (multi-day horizon, volatility regime classification, large-move prediction)
- Add transaction-cost-aware backtesting + calibration to connect metrics to decisions

---

## Author
**Chance Xu**  
GitHub: https://github.com/chance-analytics
