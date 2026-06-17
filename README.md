# stock-price-analysis
Real-world Finance data project | End-to-end analysis of AAPL, MSFT, GOOGL stock prices | Data Science Assignment
# 📈 Real-World Stock Price Analysis — Finance Domain

> **Assignment 4** · Real-world Data Project · Due: 29 Jun 2026

---

## Overview

An end-to-end data analysis project on stock price data for three major tech companies — **AAPL**, **MSFT**, and **GOOGL** — covering 120 trading days (Jan–Jun 2025). The project applies core data science techniques including statistical analysis, risk metrics, correlation analysis, and data visualisation.

---

## Dataset

| Field | Details |
|---|---|
| Stocks | Apple (AAPL), Microsoft (MSFT), Google (GOOGL) |
| Period | January 2025 – June 2025 |
| Trading Days | 120 |
| Simulation Method | Geometric Brownian Motion (GBM) |

> Prices are simulated using GBM with realistic parameters (µ, σ) derived from historical averages.

---

## Key Findings

| Metric | AAPL | MSFT | GOOGL |
|---|---|---|---|
| Total Return (6M) | -8.76% | +19.45% | +6.82% |
| Annualised Volatility | 20.41% | 20.29% | 22.79% |
| Sharpe Ratio (rf=4%) | -1.046 | 1.760 | 0.551 |
| Max Drawdown | -19.09% | -6.34% | -10.48% |

**Conclusions:**
- **MSFT** was the best performer with a +19.45% return and the highest Sharpe ratio (1.76)
- **GOOGL** carried the most risk with 22.8% annualised volatility
- **AAPL** underperformed with a negative Sharpe ratio and the deepest drawdown (-19%)
- Low inter-stock correlations (~0.1) suggest good portfolio diversification potential

---

## Project Structure

```
stock-price-analysis/
├── stock_analysis.py          # Main analysis script
├── stock_analysis_charts.png  # Output visualisations
└── README.md                  # Project documentation
```

---

## Visualisations

The script generates 5 charts saved as `stock_analysis_charts.png`:

1. **Closing Price History** — daily price trends for all 3 stocks
2. **Cumulative Return (%)** — comparative performance over 6 months
3. **20-Day Rolling Volatility** — risk evolution over time
4. **Correlation Heatmap** — pairwise return correlation matrix
5. **Monthly Return Breakdown** — grouped bar chart per month

---

## How to Run

**1. Install dependencies**
```bash
pip install pandas numpy matplotlib
```

**2. Run the script**
```bash
python stock_analysis.py
```

**3. Output**
- Console: full statistical report with all metrics
- File: `stock_analysis_charts.png` saved in the same folder

---

## Skills Applied

- Data simulation (Geometric Brownian Motion)
- Descriptive statistics with `pandas`
- Financial metrics: Sharpe ratio, max drawdown, rolling volatility
- Correlation analysis
- Data visualisation with `matplotlib`

---

## Author

**Harsha** · Data Science Assignment · Finance Domain
