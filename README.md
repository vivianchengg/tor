# 📈 Berkshire Hathaway-Inspired Stock Ranking Model

This project uses machine learning to replicate the institutional investment behavior of Berkshire Hathaway by analyzing its 13F filings. It extracts and engineers features from fundamental stock data to train a ranking model that scores and ranks stocks based on their similarity to Berkshire's portfolio.

Check out 'tor notes.pdf' for a brief literature review.

---

## 🔍 Objective

To build a data-driven model that learns from Berkshire Hathaway's past investment decisions and ranks stocks based on how "Berkshire-like" they are, using both financial fundamentals and time-series patterns.

---

## 📦 Dataset

- Source: SEC 13F filings (scraped)
- Columns include:
  - `ticker`, `value`, `shrs_or_prn_amt`, `value_change_pct`, `shares_change_pct`
  - Financial ratios: `ROE`, `D/E Ratio`, `profit_margin`, `P/B ratio`, `P/E ratio`, `beta`
  - Custom feature: `dividend_score`
  - Metadata: `cusip`, `industry`, `periodOfReport`

---

## ⚙️ Features

- Feature engineering on financial metrics (e.g. dividend yield, payout ratio)
- Custom composite score (`dividend_score`)
- Outlier handling via IQR
- Normalization using MinMaxScaler
- Label encoding of industry
- Target variable for ranking: `ranking_target`, ranked `value` per `periodOfReport`

---

## 🤖 Model

**Model Type:** Learning-to-Rank  
**Algorithm Used:** LightGBM `lambdarank` objective

### Training Flow:

1. Data Cleaning & Feature Engineering
2. Encode & Scale Features
3. Compute Ranking Target (`ranking_target`)
4. Train LightGBM model grouped by `periodOfReport`
5. Evaluate using NDCG@5

---

## 📈 Evaluation Metric

- **NDCG@5** (Normalized Discounted Cumulative Gain at rank 5)
- Reflects how well the model ranks stocks relative to actual Berkshire picks per quarter
