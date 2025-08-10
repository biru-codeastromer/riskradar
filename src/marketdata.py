"""Free historical prices via yfinance; returns one column per ticker."""

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_history(tickers, start: str, end: str) -> pd.DataFrame:
    """Download Adj Close (preferred) or Close for the tickers."""
    data = yf.download(
        tickers, start=start, end=end, progress=False, auto_adjust=False, group_by="column"
    )

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        if "Adj Close" in set(level0):
            prices = data["Adj Close"].copy()
        elif "Close" in set(level0):
            prices = data["Close"].copy()
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
    else:
        if "Adj Close" in data.columns:
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns:
            prices = data["Close"].copy()
        else:
            prices = data.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices.columns = [str(c) for c in prices.columns]
    prices = prices.loc[:, ~prices.columns.duplicated()].copy()
    return prices.dropna(how="all")


def daily_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Day-over-day simple returns."""
    return prices.pct_change().dropna(how="any")


def annualized_mu_sigma(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annualize daily mean/std using 252 trading days."""
    mu_d = returns.mean().to_numpy(dtype=float)
    sigma_d = returns.std(ddof=1).to_numpy(dtype=float)
    return mu_d * 252.0, sigma_d * np.sqrt(252.0)


def daily_correlation(returns: pd.DataFrame) -> np.ndarray:
    """Correlation of daily returns."""
    return returns.corr().to_numpy(dtype=float)
