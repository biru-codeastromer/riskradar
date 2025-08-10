"""RiskRadar: VaR/CVaR stress testing with optional correlation and Numba acceleration."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Local modules (works both as package and script)
try:
    from .marketdata import (
        fetch_history,
        daily_returns_from_prices,
        annualized_mu_sigma,
        daily_correlation,
    )
    from .reporting import (
        save_summary_csv,
        save_summary_json,
        save_returns_sample_csv,
        plot_histogram,
    )
except Exception:
    from marketdata import (
        fetch_history,
        daily_returns_from_prices,
        annualized_mu_sigma,
        daily_correlation,
    )
    from reporting import (
        save_summary_csv,
        save_summary_json,
        save_returns_sample_csv,
        plot_histogram,
    )

# Numba is optional; fall back to NumPy if unavailable
try:
    from numba import njit, prange
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False


@dataclass
class Portfolio:
    tickers: List[str]
    weights: np.ndarray           # shape [n], sums to 1
    mu_annual: np.ndarray         # shape [n]
    sigma_annual: np.ndarray      # shape [n]
    corr_daily: Optional[np.ndarray] = None  # shape [n, n]


class _SeedScope:
    """Temporarily set RNG seed without affecting global state."""
    def __init__(self, seed: int | None):
        self._seed = seed
        self._state = None
    def __enter__(self):
        if self._seed is None:
            return
        self._state = np.random.get_state()
        np.random.seed(self._seed)
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._seed is None or self._state is None:
            return
        np.random.set_state(self._state)


def load_portfolio(csv_path: str) -> Portfolio:
    df = pd.read_csv(csv_path)
    required = {"ticker", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio is missing columns: {sorted(missing)}")

    tickers = df["ticker"].astype(str).tolist()
    weights = df["weight"].to_numpy(dtype=float)

    if not np.isfinite(weights).all():
        raise ValueError("Portfolio weights contain non-finite values.")
    if weights.sum() <= 0:
        raise ValueError("Portfolio weights must sum to something positive.")
    weights = weights / weights.sum()

    if {"mu", "sigma"}.issubset(df.columns):
        mu_annual = df["mu"].to_numpy(dtype=float)
        sigma_annual = df["sigma"].to_numpy(dtype=float)
    else:
        mu_annual = np.zeros(len(tickers), dtype=float)
        sigma_annual = np.zeros(len(tickers), dtype=float)

    return Portfolio(tickers, weights, mu_annual, sigma_annual, corr_daily=None)


# ---------- Simulators ----------

def simulate_mc_uncorrelated_numpy(port: Portfolio, paths: int, horizon_days: int, seed: int | None) -> np.ndarray:
    mu_d = port.mu_annual / 252.0
    sigma_d = port.sigma_annual / np.sqrt(252.0)
    with _SeedScope(seed):
        Z = np.random.normal(size=(paths, len(mu_d), horizon_days))
    drift = (mu_d - 0.5 * sigma_d**2)[None, :, None]
    shock = sigma_d[None, :, None] * Z
    log_r_h = (drift + shock).sum(axis=2)             # [paths, assets]
    asset_simple = np.exp(log_r_h) - 1.0              # [paths, assets]
    return asset_simple @ port.weights                # [paths]


if NUMBA_OK:
    @njit(parallel=True, fastmath=True)
    def _simulate_mc_uncorr_numba_inner(mu_d, sigma_d, weights, paths, assets, horizon, rng_normals):
        out = np.empty(paths, dtype=np.float64)
        for i in prange(paths):
            pr = 0.0
            for a in range(assets):
                lr = 0.0
                drift = mu_d[a] - 0.5 * sigma_d[a]*sigma_d[a]
                for h in range(horizon):
                    lr += drift + sigma_d[a] * rng_normals[i, a, h]
                pr += (np.exp(lr) - 1.0) * weights[a]
            out[i] = pr
        return out

def simulate_mc_uncorrelated_numba(port: Portfolio, paths: int, horizon_days: int, seed: int | None) -> np.ndarray:
    if not NUMBA_OK:
        return simulate_mc_uncorrelated_numpy(port, paths, horizon_days, seed)
    mu_d = port.mu_annual / 252.0
    sigma_d = port.sigma_annual / np.sqrt(252.0)
    with _SeedScope(seed):
        rng_normals = np.random.normal(size=(paths, len(mu_d), horizon_days))
    return _simulate_mc_uncorr_numba_inner(mu_d, sigma_d, port.weights, paths, len(mu_d), horizon_days, rng_normals)


def _nearest_psd_correlation(corr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    A = 0.5 * (corr + corr.T)
    vals, vecs = np.linalg.eigh(A)
    vals[vals < eps] = eps
    A_psd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(A_psd))
    A_corr = A_psd / np.outer(d, d)
    np.fill_diagonal(A_corr, 1.0)
    return A_corr


def simulate_mc_correlated_numpy(port: Portfolio, paths: int, horizon_days: int, seed: int | None) -> np.ndarray:
    if port.corr_daily is None:
        raise ValueError("Correlated simulation requires corr_daily.")
    mu_d = port.mu_annual / 252.0
    sigma_d = port.sigma_annual / np.sqrt(252.0)
    corr = _nearest_psd_correlation(port.corr_daily)
    L = np.linalg.cholesky(corr)
    with _SeedScope(seed):
        Z = np.random.normal(size=(paths, horizon_days, len(mu_d)))  # [paths, days, assets]
        Z_corr = Z @ L.T
    drift = (mu_d - 0.5 * sigma_d**2)[None, None, :]
    shock = sigma_d[None, None, :] * Z_corr
    log_r_h = (drift + shock).sum(axis=1)     # [paths, assets]
    asset_simple = np.exp(log_r_h) - 1.0
    return asset_simple @ port.weights


def simulate_hist_uncorrelated_numpy(port: Portfolio, paths: int, horizon_days: int, seed: int | None) -> np.ndarray:
    mu_d = port.mu_annual / 252.0
    sigma_d = port.sigma_annual / np.sqrt(252.0)
    with _SeedScope(seed):
        draws = np.random.normal(loc=mu_d, scale=sigma_d, size=(paths, len(mu_d), horizon_days))
    asset_h = (1.0 + draws).prod(axis=2) - 1.0
    return asset_h @ port.weights


# ---------- Risk math ----------

def apply_uniform_shock(portfolio_returns: np.ndarray, shock: float) -> np.ndarray:
    return portfolio_returns if shock == 0.0 else (portfolio_returns + shock)


def var_cvar(portfolio_returns: np.ndarray, alpha: float) -> Tuple[float, float]:
    losses = -portfolio_returns
    var_cut = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_cut]
    cvar = float(tail.mean()) if tail.size else var_cut
    return var_cut, cvar


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RiskRadar — VaR/CVaR with history and optional Numba.")
    p.add_argument("--portfolio", required=True, help="CSV: ticker,weight[,mu,sigma]")
    p.add_argument("--paths", type=int, default=100_000, help="Simulation paths")
    p.add_argument("--horizon", type=int, default=1, help="Horizon in trading days")
    p.add_argument("--alpha", type=float, default=0.95, help="Confidence (0.95 or 0.99)")
    p.add_argument("--method", choices=["mc", "hist"], default="mc", help="Simulation method")
    p.add_argument("--engine", choices=["numpy", "numba"], default="numpy",
                   help="Computation engine (Numba accelerates uncorrelated MC).")
    p.add_argument("--shock", type=float, default=0.0, help="Uniform shock to apply (e.g., -0.05)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--use-history", action="store_true",
                   help="Estimate mu/sigma + correlation from Yahoo via yfinance.")
    p.add_argument("--start", type=str, default="2022-01-01", help="History start (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="History end (YYYY-MM-DD)")
    p.add_argument("--out", type=str, default=None, help="Directory to write outputs into.")
    p.add_argument("--save-json", action="store_true", help="Write summary.json")
    p.add_argument("--save-csv", action="store_true", help="Write summary.csv")
    p.add_argument("--save-plot", action="store_true", help="Write histogram.png")
    p.add_argument("--save-returns", type=int, default=0, help="If >0, write returns_sample.csv with N rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    port = load_portfolio(args.portfolio)

    # Align history to portfolio tickers (order + subset)
    if args.use_history:
        prices = fetch_history(port.tickers, start=args.start, end=args.end)
        prices = prices.loc[:, ~prices.columns.duplicated()].copy()
        cols = [c for c in port.tickers if c in prices.columns]
        missing = [c for c in port.tickers if c not in prices.columns]
        if missing:
            raise ValueError(f"Missing symbols in downloaded data: {missing}. Got: {list(prices.columns)}")
        prices = prices[cols]
        rets = daily_returns_from_prices(prices)
        if len(rets) < 100:
            raise ValueError(f"Too little history ({len(rets)} rows). Pick a wider date range.")
        mu_ann, sigma_ann = annualized_mu_sigma(rets)
        corr = daily_correlation(rets)
        if len(mu_ann) != len(port.weights):
            raise ValueError(
                f"Shape mismatch: mu={len(mu_ann)} vs weights={len(port.weights)}; cols={list(rets.columns)}"
            )
        port.mu_annual, port.sigma_annual, port.corr_daily = mu_ann, sigma_ann, corr
    else:
        port.corr_daily = None

    # Choose simulator
    if args.method == "mc":
        if port.corr_daily is not None:
            simulated = simulate_mc_correlated_numpy(port, args.paths, args.horizon, args.seed)
            engine_used = "numpy (correlated)"
        else:
            if args.engine == "numba":
                simulated = simulate_mc_uncorrelated_numba(port, args.paths, args.horizon, args.seed)
                engine_used = "numba (uncorrelated)"
            else:
                simulated = simulate_mc_uncorrelated_numpy(port, args.paths, args.horizon, args.seed)
                engine_used = "numpy (uncorrelated)"
    else:
        simulated = simulate_hist_uncorrelated_numpy(port, args.paths, args.horizon, args.seed)
        engine_used = "numpy (hist)"

    simulated = apply_uniform_shock(simulated, args.shock)
    var_val, cvar_val = var_cvar(simulated, args.alpha)
    dt = time.time() - t0
    throughput = (args.paths / dt) if dt > 0 else float("inf")

    # Console report
    print("=== RiskRadar Report ===")
    print(f"Assets: {port.tickers}")
    print(f"Weights: {np.round(port.weights, 4).tolist()} (sum={port.weights.sum():.4f})")
    print(f"Method: {args.method} | Engine: {engine_used} | Paths: {args.paths:,} | Horizon: {args.horizon}d | Alpha: {args.alpha}")
    print("Params:", "history" if args.use_history else "csv", "| Correlation:", "yes" if port.corr_daily is not None else "no")
    print(f"Shock applied: {args.shock:+.2%}")
    print(f"VaR@{int(args.alpha*100)}  : {var_val:.4%} loss")
    print(f"CVaR@{int(args.alpha*100)} : {cvar_val:.4%} loss")
    print(f"Runtime: {dt:.3f}s | Throughput: {throughput:,.0f} paths/s")
    print(f"RESUME-LINE: engine={engine_used}, corr={'yes' if port.corr_daily is not None else 'no'}, "
          f"paths={args.paths:,}, horizon={args.horizon}d, VaR@{int(args.alpha*100)}={-var_val:.2%} return, "
          f"CVaR@{int(args.alpha*100)}={-cvar_val:.2%} return, runtime={dt:.3f}s, thrput={throughput:,.0f}/s")

    # Optional outputs
    if args.out:
        summary = {
            "alpha": args.alpha,
            "paths": float(args.paths),
            "horizon_days": float(args.horizon),
            "shock": float(args.shock),
            "var": var_val,
            "cvar": cvar_val,
            "runtime_sec": dt,
            "throughput_paths_per_sec": throughput,
            "engine": engine_used,
            "correlated": bool(port.corr_daily is not None),
            "used_history": bool(args.use_history),
        }
        if args.save_csv:
            print(f"Saved CSV summary → {save_summary_csv(args.out, summary)}")
        if args.save_json:
            print(f"Saved JSON summary → {save_summary_json(args.out, summary)}")
        if args.save_plot:
            print(f"Saved histogram plot → {plot_histogram(args.out, simulated, var_val, cvar_val, args.alpha)}")
        if args.save_returns and args.save_returns > 0:
            print(f"Saved returns sample → {save_returns_sample_csv(args.out, simulated, n=int(args.save_returns))}")


if __name__ == "__main__":
    main()
