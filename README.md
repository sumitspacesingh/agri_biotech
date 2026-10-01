# 🌾 Agricultural Biotech Supply Chain Risk Model

An interactive risk analytics dashboard built with Streamlit and Monte Carlo simulation. This tool models multi-factor biological infection risk across distributed farm portfolios, accounting for regional contagion correlation, proactive bio-shield mitigations, indemnity insurance, and climate stress shocks.

---

## 📌 Overview & Methodology

The dashboard employs a **Gaussian single-factor copula model** structured by growing regions:
- **Spatial Contagion:** Farms within the same region share a systemic regional shock ($Z_{\text{region}}$), while having independent idiosyncratic shocks ($\epsilon_i$).
- **Regional Diversification:** Independent geographic regions provide non-correlated disease/weather diversification benefits.
- **Financial Risk Metrics:** Computes Expected Loss (EL), Value at Risk (VaR), Conditional Value at Risk (CVaR / Expected Shortfall), and Unexpected Loss (UL).
- **Mitigation Decision Engine:** Evaluates P&L viability and solvency protection by contrasting bio-shield capital expenditure and insurance premiums against net loss reductions.

---

## 🚀 Quickstart

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

## ⚠️ Disclaimer

This dashboard is built for educational, instructional, and exploratory purposes only. The risk weights, multipliers, and threshold calibrations are synthetic heuristics and do not constitute a statistically validated internal ratings-based (IRB) or IFRS 9 impairment model. Real-world underwriting requires models calibrated against historical default datasets, validated for discriminatory power (e.g., Gini/AUC), and audited for regulatory and fair-lending compliance.

### Note
This repository is made by using AI.
