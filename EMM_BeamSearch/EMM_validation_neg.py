# file: validate_emm_results.py
from __future__ import annotations
import ast
import json
from pathlib import Path
from typing import Any, List, Dict, Tuple

import numpy as np
import pandas as pd

DATA_JSONL = Path("EMM_final_dataset/EMM_FINAL.jsonl")     # original dataset (needed for Y)
RESULTS_CSV = Path("EMM_results/EMM_neg_results.csv")    # your mined subgroups
OUT_CSV     = Path("EMM_results/EMM_neg_results_validated.csv")

# ---------------------- helpers ----------------------

def parse_mask(mask_str: str, n: int) -> np.ndarray:
    """
    Parse the 'mask' column from CSV into a boolean array of length n.
    Expected format: '[1, 0, 1, ...]'. Falls back to interpreting as index list.
    """
    try:
        arr = ast.literal_eval(mask_str)
    except Exception:
        # very defensive fallback
        arr = [int(x) for x in mask_str.replace("["," ").replace("]"," ").replace(","," ").split() if x.isdigit()]
    a = np.asarray(arr, dtype=int)
    if a.size != n:
        # If it's actually a list of indices, expand to a 0/1 mask
        b = np.zeros(n, dtype=int)
        if a.size > 0 and a.min() >= 0 and a.max() < n:
            b[a] = 1
            a = b
        else:
            raise ValueError(f"Mask length {a.size} != N={n} and not valid indices.")
    return (a.astype(bool))

def parse_description(desc_str: str) -> Dict[str, Any]:
    """description is a JSON string in your CSV."""
    try:
        return json.loads(desc_str)
    except Exception:
        return {"raw": desc_str}

def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return (qvals, significant_mask) for BH–FDR."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    min_coeff = 1.0
    for i in range(n-1, -1, -1):
        coeff = ranked[i] * n / (i+1)
        min_coeff = min(min_coeff, coeff)
        q[i] = min_coeff
    qvals = np.empty_like(q)
    qvals[order] = q
    return qvals, (qvals <= alpha)

def bootstrap_ci_mean(values: np.ndarray, B: int = 5000, conf: float = 0.95, seed: int = 123) -> Tuple[float,float,float]:
    """Nonparametric bootstrap CI for mean(values)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return np.nan, np.nan, np.nan
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = values[idx].mean()
    lo = float(np.quantile(boots, (1-conf)/2))
    hi = float(np.quantile(boots, 1-(1-conf)/2))
    return lo, hi, hi - lo

def bootstrap_ci_delta(y: np.ndarray, mask: np.ndarray, B: int = 5000, conf: float = 0.95, seed: int = 123) -> Tuple[float,float,float]:
    """Bootstrap CI for Δ = mean(y[mask]) - mean(y). Resample within S only (fix μ_global)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    S = y[mask]
    if S.size == 0:
        return np.nan, np.nan, np.nan
    mu = float(y.mean())
    boots = np.empty(B, dtype=float)
    nS = S.size
    for b in range(B):
        idx = rng.integers(0, nS, size=nS)
        boots[b] = S[idx].mean() - mu
    lo = float(np.quantile(boots, (1-conf)/2))
    hi = float(np.quantile(boots, 1-(1-conf)/2))
    return lo, hi, hi - lo

def permutation_pvalue_delta(y: np.ndarray, mask: np.ndarray, B: int = 10000, seed: int = 123) -> Tuple[float, float]:
    """Two-sided permutation test for Δ = mean(y_S) - mean(y). Permute Y, keep mask fixed."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    mu = float(y.mean())
    obs = float(y[mask].mean() - mu)
    greater = 0
    for _ in range(B):
        y_perm = rng.permutation(y)
        stat = float(y_perm[mask].mean() - y_perm.mean())
        if abs(stat) >= abs(obs):
            greater += 1
    pval = (greater + 1) / (B + 1)  # +1 correction
    return pval, obs

# ---------------------- main flow ----------------------

def main():
    # 1) Load Y from the JSONL
    raw = pd.read_json(DATA_JSONL, lines=True)
    if "features" in raw.columns:
        feats = pd.json_normalize(raw["features"])
        df = pd.concat([raw.drop(columns=["features"]), feats], axis=1)
    else:
        df = raw.copy()
    y = pd.to_numeric(df["Y"], errors="coerce").to_numpy()
    N = len(y)
    mu_global = float(np.nanmean(y))

    # 2) Load your mined results CSV
    res = pd.read_csv(RESULTS_CSV)

    # 3) Build validated records
    records: List[Dict[str, Any]] = []
    for _, row in res.iterrows():
        mask = parse_mask(row["mask"], N)
        desc = parse_description(row["description"])
        S = y[mask]
        sg_size = int(mask.sum())
        sg_frac = sg_size / N if N else np.nan
        mean_S = float(S.mean()) if sg_size > 0 else np.nan
        delta_obs = float(mean_S - mu_global)

        # Bootstrap CIs
        m_lo, m_hi, m_w = bootstrap_ci_mean(S, B=1000, conf=0.95, seed=2025)
        d_lo, d_hi, d_w = bootstrap_ci_delta(y, mask, B=1000, conf=0.95, seed=2025)

        # Permutation p-value
        pval, obs_delta_perm = permutation_pvalue_delta(y, mask, B=2000, seed=2025)

        # Collect
        records.append({
            "rank": int(row.get("rank", np.nan)),
            "description": json.dumps(desc, ensure_ascii=False),
            "wracc": float(row.get("wracc", np.nan)),
            "t_stat": float(row.get("t_stat", np.nan)),
            "size": sg_size,
            "coverage": sg_frac,
            "mean_S": mean_S,
            "delta_csv": float(row.get("delta", np.nan)),
            "delta_recomp": delta_obs,
            "ci_mean_lo": m_lo,
            "ci_mean_hi": m_hi,
            "ci_delta_lo": d_lo,
            "ci_delta_hi": d_hi,
            "pval_perm": pval,
        })

    out = pd.DataFrame.from_records(records)

    # 4) BH–FDR across all p-values
    qvals, sig = benjamini_hochberg(out["pval_perm"].to_numpy(), alpha=0.05)
    out["qval_bh"] = qvals
    out["significant_bh"] = sig

    # 5) Sort for presentation
    out = out.sort_values(
        by=["significant_bh", "wracc", "t_stat", "coverage"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    # 6) Print a compact view
    cols = ["rank","wracc","t_stat","size","coverage",
            "mean_S","delta_recomp","ci_delta_lo","ci_delta_hi","pval_perm","qval_bh","significant_bh","description"]
    with pd.option_context("display.max_colwidth", 140, "display.width", 160):
        print("\n=== Validation of discovered subgroups (bootstrap CIs + permutation p-values + BH–FDR) ===")
        print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 7) Save to disk
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

if __name__ == "__main__":
    main()
