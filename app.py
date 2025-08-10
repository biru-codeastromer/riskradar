# RiskRadar — Streamlit UI
# Minimal, fast, and readable. Pulls your core code from src/.

from __future__ import annotations
import io
import sys
import time
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Make 'src' importable when running streamlit from repo root
ROOT = pathlib.Path(__file__).parent.resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

# Import your engine bits
from risksim import (  # type: ignore
    Portfolio,
    load_portfolio,                      # used for file-path case
    simulate_mc_uncorrelated_numpy,
    simulate_mc_uncorrelated_numba,
    simulate_mc_correlated_numpy,
    simulate_hist_uncorrelated_numpy,
    apply_uniform_shock,
    var_cvar,
)
from marketdata import (  # type: ignore
    fetch_history,
    daily_returns_from_prices,
    annualized_mu_sigma,
    daily_correlation,
)

APP_TITLE = "RiskRadar"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# --------------------------
# Helpers
# --------------------------

def _parse_portfolio_df(df: pd.DataFrame) -> Portfolio:
    req = {"ticker", "weight"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {sorted(req)}")
    tickers = df["ticker"].astype(str).tolist()
    weights = df["weight"].astype(float).to_numpy()
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Portfolio weights invalid.")
    weights = weights / weights.sum()
    if {"mu", "sigma"}.issubset(df.columns):
        mu = df["mu"].astype(float).to_numpy()
        sigma = df["sigma"].astype(float).to_numpy()
    else:
        mu = np.zeros(len(tickers), dtype=float)
        sigma = np.zeros(len(tickers), dtype=float)
    return Portfolio(tickers, weights, mu, sigma, corr_daily=None)


def _hist_fig(returns: np.ndarray, var_cut: float, cvar: float, alpha: float):
    # VaR/CVaR defined on losses; convert to returns for markers
    var_ret = -var_cut
    cvar_ret = -cvar
    fig, ax = plt.subplots()
    ax.hist(returns, bins=80, alpha=0.8)
    ax.axvline(var_ret, linestyle="--", linewidth=2, label=f"VaR@{int(alpha*100)} ({var_ret:.2%})")
    ax.axvline(cvar_ret, linestyle="-.", linewidth=2, label=f"CVaR@{int(alpha*100)} ({cvar_ret:.2%})")
    ax.set_title("Simulated Portfolio Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig


def _download_df_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")


# --------------------------
# UI
# --------------------------

st.title("RiskRadar")
st.caption("Monte Carlo VaR/CVaR with optional historical correlation (free Yahoo data).")

left, right = st.columns([0.62, 0.38], gap="large")

with st.sidebar:
    st.header("Inputs")

    uploaded = st.file_uploader("Portfolio CSV", type=["csv"], help="Columns: ticker,weight[,mu,sigma]")
    use_sample = st.checkbox("Use sample portfolio (data/sample_portfolio.csv)", value=(uploaded is None))

    use_history = st.checkbox("Use historical data (Yahoo Finance)", value=True)
    start_date = st.date_input("History start", value=pd.to_datetime("2022-01-01").date())
    end_date = st.date_input("History end", value=pd.Timestamp.today().date())

    method = st.selectbox("Method", options=["mc", "hist"], index=0)
    engine = st.selectbox("Engine (MC uncorrelated)", options=["numpy", "numba"], index=1)

    paths = st.number_input("Paths", value=200_000, min_value=10_000, step=50_000)
    horizon = st.number_input("Horizon (days)", value=1, min_value=1, max_value=252, step=1)
    alpha = st.select_slider("Confidence (alpha)", options=[0.90, 0.95, 0.975, 0.99], value=0.95)
    shock = st.number_input("Uniform shock (e.g., -0.05 for -5%)", value=0.0, step=0.01, format="%.4f")
    seed = st.number_input("Random seed (optional)", value=7, step=1)

    run = st.button("Run simulation", type="primary", use_container_width=True)

# Portfolio preview
with right:
    st.subheader("Portfolio")
    try:
        if uploaded and not use_sample:
            df_port = pd.read_csv(uploaded)
            port = _parse_portfolio_df(df_port)
            st.dataframe(df_port, use_container_width=True, hide_index=True)
        else:
            sample_path = ROOT / "data" / "sample_portfolio.csv"
            df_port = pd.read_csv(sample_path)
            port = _parse_portfolio_df(df_port)
            st.dataframe(df_port, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Portfolio error: {e}")
        st.stop()

# Simulation
with left:
    st.subheader("Results")
    if run:
        t0 = time.time()
        # History-based params + correlation
        if use_history:
            try:
                prices = fetch_history(port.tickers, start=str(start_date), end=str(end_date))
                prices = prices.loc[:, ~prices.columns.duplicated()].copy()
                cols = [c for c in port.tickers if c in prices.columns]
                missing = [c for c in port.tickers if c not in prices.columns]
                if missing:
                    st.error(f"Missing symbols from Yahoo: {missing}. Got {list(prices.columns)}")
                    st.stop()
                prices = prices[cols]
                rets = daily_returns_from_prices(prices)
                if len(rets) < 100:
                    st.error(f"Too little history ({len(rets)} rows). Widen the date range.")
                    st.stop()
                mu_ann, sigma_ann = annualized_mu_sigma(rets)
                corr = daily_correlation(rets)
                port.mu_annual, port.sigma_annual, port.corr_daily = mu_ann, sigma_ann, corr
            except Exception as e:
                st.error(f"History fetch failed: {e}")
                st.stop()
        else:
            port.corr_daily = None

        # Pick simulator
        try:
            if method == "mc":
                if port.corr_daily is not None:
                    returns = simulate_mc_correlated_numpy(port, int(paths), int(horizon), int(seed))
                    engine_used = "numpy (correlated)"
                else:
                    if engine == "numba":
                        returns = simulate_mc_uncorrelated_numba(port, int(paths), int(horizon), int(seed))
                        engine_used = "numba (uncorrelated)"
                    else:
                        returns = simulate_mc_uncorrelated_numpy(port, int(paths), int(horizon), int(seed))
                        engine_used = "numpy (uncorrelated)"
            else:
                returns = simulate_hist_uncorrelated_numpy(port, int(paths), int(horizon), int(seed))
                engine_used = "numpy (hist)"
        except Exception as e:
            st.error(f"Simulation error: {e}")
            st.stop()

        # Shock + risk metrics
        returns = apply_uniform_shock(returns, float(shock))
        var_val, cvar_val = var_cvar(returns, float(alpha))
        dt = time.time() - t0
        thr = (int(paths) / dt) if dt > 0 else float("inf")

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("VaR (loss)", f"{var_val:.2%}")
        k2.metric("CVaR (loss)", f"{cvar_val:.2%}")
        k3.metric("Runtime", f"{dt:.3f}s")
        k4.metric("Throughput", f"{thr:,.0f} paths/s")

        # Histogram
        st.pyplot(_hist_fig(returns, var_val, cvar_val, float(alpha)), clear_figure=True)

        # Summary + downloads
        summary = pd.DataFrame({
            "metric": ["alpha", "paths", "horizon_days", "shock", "var_loss", "cvar_loss", "runtime_sec", "throughput", "engine", "correlated", "used_history"],
            "value": [alpha, int(paths), int(horizon), float(shock), var_val, cvar_val, dt, thr, engine_used, bool(port.corr_daily is not None), bool(use_history)]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

        _download_df_button(summary, "summary.csv", "Download summary.csv")
        returns_df = pd.DataFrame({"portfolio_return": returns})
        _download_df_button(returns_df.head(50_000), "returns_sample.csv", "Download returns_sample.csv")

        st.caption(
            f"Resume line → engine={engine_used}, corr={'yes' if port.corr_daily is not None else 'no'}, "
            f"paths={int(paths):,}, horizon={int(horizon)}d, VaR@{int(alpha*100)}={-var_val:.2%} return, "
            f"CVaR@{int(alpha*100)}={-cvar_val:.2%} return, runtime={dt:.3f}s, thrput={thr:,.0f}/s"
        )
