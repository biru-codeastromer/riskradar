"""Simple reporting helpers: CSV/JSON summaries and a histogram plot."""

from __future__ import annotations
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_summary_csv(out_dir: str, summary: Dict[str, float]) -> str:
    _ensure_dir(out_dir)
    df = pd.DataFrame({"metric": list(summary.keys()), "value": list(summary.values())})
    p = os.path.join(out_dir, "summary.csv")
    df.to_csv(p, index=False)
    return p


def save_summary_json(out_dir: str, summary: Dict[str, float]) -> str:
    _ensure_dir(out_dir)
    p = os.path.join(out_dir, "summary.json")
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    return p


def save_returns_sample_csv(out_dir: str, returns: np.ndarray, n: int = 10000) -> str:
    _ensure_dir(out_dir)
    n = min(n, returns.size)
    sample = returns[:n]
    df = pd.DataFrame({"portfolio_return": sample})
    p = os.path.join(out_dir, "returns_sample.csv")
    df.to_csv(p, index=False)
    return p


def plot_histogram(out_dir: str, returns: np.ndarray, var_cut: float, cvar: float, alpha: float) -> str:
    """Histogram of returns with VaR/CVaR markers. Saves to PNG."""
    _ensure_dir(out_dir)
    var_return = -var_cut
    cvar_return = -cvar
    plt.figure()
    plt.hist(returns, bins=80, alpha=0.8)
    plt.axvline(var_return, linestyle="--", linewidth=2, label=f"VaR@{int(alpha*100)} ({var_return:.2%})")
    plt.axvline(cvar_return, linestyle="-.", linewidth=2, label=f"CVaR@{int(alpha*100)} ({cvar_return:.2%})")
    plt.title("Portfolio Returns — Simulated Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()
    p = os.path.join(out_dir, "histogram.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p
