# RiskRadar

Small, readable risk engine. Simulates portfolio returns (Monte Carlo), supports historical correlation from Yahoo Finance, and reports VaR/CVaR. Optional Numba acceleration.

## Install (macOS)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
```

# Quick start

Fast (uncorrelated, Numba):

```bash
python3 src/risksim.py --portfolio data/sample_portfolio.csv \
  --paths 500000 --horizon 1 --alpha 0.99 --method mc \
  --engine numba --seed 123 \
  --out out/fast --save-json --save-csv --save-plot --save-returns 20000
```

Real data + correlation (free yfinance):

```bash
python3 src/risksim.py --portfolio data/sample_portfolio.csv \
  --paths 200000 --horizon 1 --alpha 0.95 --method mc \
  --use-history --start 2022-01-01 --end 2025-08-01 --seed 7 \
  --out out/hist --save-json --save-csv --save-plot
```

# Notes

- Portfolio CSV: ticker,weight[,mu,sigma]. If --use-history is set, mu/sigma from CSV are ignored.

- VaR/CVaR are computed on the loss distribution; in plots, the vertical lines show the corresponding returns (negative of losses).

- If history rows < 100, widen the date range.
