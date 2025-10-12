"""
Run a Beaamsearch beam search on the EMM BBQ gender dataset using
bi-directional WRAcc as the primary quality measure and a t-statistic
as the secondary tie-breaker.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ---- Postprocessing knobs ----
JACCARD_THRESH = 0.99     # masks considered near-duplicates if J >= 0.95
DELTA_EPS = 1e-4           # max allowed |Δ| difference when collapsing
COV_EPS = 1e-3           # max allowed coverage difference when collapsing
CATEGORY_COL = "category"  # for per-category summary


REPO_ROOT = Path(__file__).resolve().parent
BEAMSEARCH_PATH = REPO_ROOT / "Beaamsearch"
if str(BEAMSEARCH_PATH) not in sys.path:
    sys.path.append(str(BEAMSEARCH_PATH))

import collect_qualities as qu  # type: ignore  # pylint: disable=wrong-import-position
import constraints as cs  # type: ignore  # pylint: disable=wrong-import-position
import beam_search as bs  # type: ignore  # pylint: disable=wrong-import-position


def calculate_general_parameters_bias(
    dataset: pd.DataFrame,
    attributes: Dict[str, str],
    model_params: Dict[str, Any],
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute global statistics for WRAcc and t-statistic."""
    target = attributes["target"]
    target_values = dataset[target].to_numpy(dtype=float)
    data_size = int(target_values.size)
    global_mean = float(np.mean(target_values)) if data_size else 0.0
    global_var = float(np.var(target_values, ddof=1)) if data_size > 1 else 0.0

    return {
        "data_size": data_size,
        "global_mean": global_mean,
        "global_var": global_var,
    }


def calculate_first_part_subgroup_parameters_bias(
    subgroup: pd.DataFrame,
    attributes: Dict[str, str],
    model_params: Dict[str, Any],
    general_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Store subgroup membership and target values for follow-up calculations."""
    target = attributes["target"]
    target_values = subgroup[target].to_numpy(dtype=float)
    sg_size = int(target_values.size)

    return {
        "sg_size": sg_size,
        "target_values": target_values,
    }


def calculate_second_part_subgroup_parameters_bias(
    subgroup_params: Dict[str, Any],
    subgroup: pd.DataFrame,
    attributes: Dict[str, str],
    model_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive subgroup mean and variance."""
    values = subgroup_params.pop("target_values", np.array([], dtype=float))
    sg_size = subgroup_params["sg_size"]

    if values.size == 0 and sg_size > 0 and not subgroup.empty:
        values = subgroup[attributes["target"]].to_numpy(dtype=float)

    if sg_size == 0 or values.size == 0:
        mean = float("nan")
        variance = float("nan")
    elif sg_size == 1:
        mean = float(values[0])
        variance = 0.0
    else:
        mean = float(np.mean(values))
        variance = float(np.var(values, ddof=1))

    subgroup_params["mean"] = mean
    subgroup_params["variance"] = variance
    return subgroup_params


def add_qm_bias(
    desc: Dict[str, Any],
    general_params: Dict[str, Any],
    subgroup_params: Dict[str, Any],
    model_params: Dict[str, Any],
    beam_search_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach WRAcc and t-stat quality measures to a subgroup description."""
    mu = general_params["global_mean"]
    sigma2 = general_params["global_var"]
    total_size = general_params["data_size"]

    sg_size = subgroup_params.get("sg_size", 0)
    sg_mean = subgroup_params.get("mean", float("nan"))

    if total_size == 0 or math.isnan(sg_mean):
        wracc = 0.0
    else:
        wracc = (sg_size / total_size) * abs(sg_mean - mu)

    if sigma2 <= 0.0 or sg_size == 0 or total_size == 0:
        t_stat = 0.0
    else:
        denom = math.sqrt(sigma2 * ((1.0 / sg_size) + (1.0 / total_size)))
        t_stat = 0.0 if denom == 0 else (sg_mean - mu) / denom

    subgroup_snapshot = {
        key: value
        for key, value in subgroup_params.items()
        if key != "target_values"
    }
    subgroup_snapshot["qm_value"] = wracc
    subgroup_snapshot["wracc"] = wracc
    subgroup_snapshot["t_stat"] = t_stat
    subgroup_snapshot["abs_t_stat"] = abs(t_stat)
    subgroup_snapshot["sg_fraction"] = 0.0 if total_size == 0 else sg_size / total_size

    desc_qm = {
        "description": dict(desc["description"]),
        "qualities": subgroup_snapshot,
    }

    return desc_qm


def constraint_subgroup_size_bias(
    general_params: Dict[str, Any],
    subgroup_params: Dict[str, Any],
    constraints: Dict[str, Any],
    model_params: Dict[str, Any],
) -> Tuple[bool, str | None, Dict[str, Any]]:
    """Ensure subgroups meet a minimum size requirement."""
    sg_size = subgroup_params.get("sg_size", 0)
    total_size = general_params.get("data_size", 0)

    min_frac = float(constraints.get("min_size", 0.0))
    min_abs = int(constraints.get("min_subgroup_size", 0))
    required_size = max(int(math.ceil(min_frac * total_size)), min_abs)

    if sg_size >= required_size:
        return True, None, subgroup_params

    return False, "small_subgroup", subgroup_params


def constraint_connected_occassions_bias(
    general_params: Dict[str, Any],
    subgroup_params: Dict[str, Any],
    constraints: Dict[str, Any],
    model_params: Dict[str, Any],
) -> bool:
    """Placeholder for temporal connectivity requirements (always passes)."""
    return True


def patch_quality_modules() -> None:
    """Replace Beaamsearch quality hooks with WRAcc/t-stat implementations."""
    qu.calculate_general_parameters = calculate_general_parameters_bias
    qu.calculate_first_part_subgroup_parameters = calculate_first_part_subgroup_parameters_bias
    qu.calculate_second_part_subgroup_parameters = calculate_second_part_subgroup_parameters_bias
    qu.add_qm = add_qm_bias

    cs.constraint_subgroup_size = constraint_subgroup_size_bias
    cs.constraint_connected_occassions = constraint_connected_occassions_bias


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load and flatten the JSONL gender dataset."""
    raw_df = pd.read_json(dataset_path, lines=True)
    feature_df = pd.json_normalize(raw_df["features"])

    df = pd.concat(
        [
            raw_df[["item_id", "Y"]].reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )
    return df


def run() -> None:
    dataset_path = Path("EMM_FINAL.jsonl")
    df = load_dataset(dataset_path)

    descriptives = {
        "bin_atts": [],
        "nom_atts": list(df.columns.difference(["item_id", "Y"])),
        "num_atts": [],
        "ord_atts": [],
    }

    attributes = {"target": "Y"}

    model_params = {
        "order": "max",
        "qm": "wracc",
    }

    beam_search_params = {
        "b": 50,
        "d": 3,
        "w": 50,
        "q": 20,
    }

    wcs_params = {
        "gamma": 0.5,
        "stop_desc_sel": 100,
    }

    constraints = {
        "min_size": 0.1,
        "min_subgroup_size": 30,
    }

    patch_quality_modules()

    result_emm, general_params, considered_subgroups = bs.beam_search(
        dataset=df,
        attributes=attributes,
        descriptives=descriptives,
        model_params=model_params,
        beam_search_params=beam_search_params,
        wcs_params=wcs_params,
        constraints=constraints,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("Global statistics:", general_params)
    if result_emm is None:
        print("Beam search returned no subgroups.")
        print("Diagnostics:", considered_subgroups)
        return

    formatted = extract_ranked_subgroups(result_emm, global_mean=general_params["global_mean"])
    if not formatted:
        print("Beam search returned results but formatting failed.")
        print(result_emm)
    else:
        # sort primary list by (WRAcc, |t|, shorter rule)
        formatted.sort(key=lambda x: (x.get("wracc", 0.0),
                                      x.get("abs_t_stat", 0.0),
                                      -len(x["raw_description"])), reverse=True)

        # 1) Collapse near-duplicates
        collapsed = collapse_near_duplicates(formatted, df)

        print("\n=== Overall top (post-collapse) ===")
        for e in collapsed[:10]:
            print(f"#{e['rank']:2d}", e['description'], {"wracc": e["wracc"], "delta": e["delta"], "sg_fraction": e["sg_fraction"], "|t|": e["abs_t_stat"]})

        # 2) Split by direction
        male_list, female_list = split_by_direction(collapsed, general_params["global_mean"])

        # 3) Per-category summary
        cat_table = per_category_summary(df, target="Y", category_col=CATEGORY_COL)

        # ---- Print nicely ----
        print("\n=== Male-favoring (Δ>0) ===")
        for e in male_list[:10]:
            print(f"#{e['rank']:2d}", e['description'], {"wracc": e["wracc"], "delta": e["delta"], "sg_fraction": e["sg_fraction"], "|t|": e["abs_t_stat"]})

        print("\n=== Female-favoring (Δ<0) ===")
        for e in female_list[:10]:
            print(f"#{e['rank']:2d}", e['description'], {"wracc": e["wracc"], "delta": e["delta"], "sg_fraction": e["sg_fraction"], "|t|": e["abs_t_stat"]})

        print("\n=== Per-category summary (top 10 by |delta|) ===")
        cat_top = cat_table.reindex(cat_table["delta"].abs().sort_values(ascending=False).index).head(10)
        print(cat_top.to_string(index=False))
    print("Diagnostics:", considered_subgroups)


def extract_ranked_subgroups(result_emm: pd.DataFrame, global_mean: float) -> List[Dict[str, Any]]:
    if "sg" not in result_emm.columns:
        return []

    out: List[Dict[str, Any]] = []
    for sg_id in sorted(result_emm["sg"].dropna().unique()):
        sg_slice = result_emm[result_emm["sg"] == sg_id]
        if sg_slice.empty or sg_slice.shape[0] < 2:
            continue
        desc_row = (
            sg_slice[sg_slice.index == "description"]
            .iloc[0]
            .drop(labels=["sg"], errors="ignore")
        )
        qual_row = (
            sg_slice[sg_slice.index == "qualities"]
            .iloc[0]
            .drop(labels=["sg"], errors="ignore")
        )

        raw_desc = {}
        for col, val in desc_row.items():
            if pd.isna(val) or col in {"abs_t_stat", "mean", "q"}:
                continue
            raw_desc[col] = val
        pretty = {col: _format_condition(col, val) for col, val in raw_desc.items()}
        metrics = {k: _normalise_value(v)
                   for k, v in qual_row.items()
                   if pd.notna(v) and k not in {"sg_idx", "temp_qm_value"}}
        mean = float(metrics.get("mean", float("nan")))
        delta = mean - float(global_mean) if not math.isnan(mean) else float("nan")
        metrics["delta"] = delta
        out.append({
            "rank": int(sg_id) + 1,
            "description": pretty,
            "raw_description": raw_desc,
            **metrics,
        })
    return out


def _normalise_value(value: Any) -> Any:
    """Convert numpy scalar types to native Python values for readability."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_condition(attribute: str, raw_value: Any) -> str:
    """Generate a readable string representation for a subgroup condition."""
    if isinstance(raw_value, tuple) and len(raw_value) == 2:
        comparator, literal = raw_value
        if comparator == 1.0:
            return f"{attribute} == {literal}"
        if comparator == 0.0:
            return f"{attribute} != {literal}"
    if isinstance(raw_value, list):
        return f"{attribute} in {raw_value}"
    return f"{attribute} == {raw_value}"


def build_mask_from_raw_desc(df: pd.DataFrame, raw_desc: Dict[str, Any]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for attr, raw in raw_desc.items():
        col = df[attr].astype(object)
        if isinstance(raw, tuple) and len(raw) == 2:
            comparator, literal = raw
            if comparator == 1.0:
                mask &= (col == literal).to_numpy()
            elif comparator == 0.0:
                mask &= (col != literal).to_numpy()
            else:
                raise ValueError(f"Unknown comparator {comparator} for {attr}")
        elif isinstance(raw, list):
            mask &= col.isin(raw).to_numpy()
        else:
            mask &= (col == raw).to_numpy()
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else inter / union


def collapse_near_duplicates(entries: List[Dict[str, Any]],
                             df: pd.DataFrame,
                             jaccard_thresh: float = JACCARD_THRESH,
                             delta_eps: float = DELTA_EPS,
                             cov_eps: float = COV_EPS) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    kept_masks: List[np.ndarray] = []
    masks = [build_mask_from_raw_desc(df, e["raw_description"]) for e in entries]
    for e, m in zip(entries, masks):
        is_dup = False
        for km, ke in zip(kept_masks, kept):
            J = jaccard(m, km)
            cov_diff = abs(e.get("sg_fraction", 0.0) - ke.get("sg_fraction", 0.0))
            delta_diff = abs(e.get("delta", float("nan")) - ke.get("delta", float("nan")))
            if J >= jaccard_thresh and (math.isnan(delta_diff) or delta_diff <= delta_eps) and cov_diff <= cov_eps:
                is_dup = True
                break
        if not is_dup:
            kept.append(e)
            kept_masks.append(m)
    kept.sort(key=lambda x: (x.get("wracc", 0.0),
                             x.get("abs_t_stat", 0.0),
                             -len(x["raw_description"])), reverse=True)
    for i, e in enumerate(kept, 1):
        e["rank"] = i
    return kept


def split_by_direction(entries: List[Dict[str, Any]], global_mean: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    _ = global_mean
    pos, neg = [], []
    for e in entries:
        delta = e.get("delta", float("nan"))
        if math.isnan(delta) or delta == 0.0:
            continue
        (pos if delta > 0 else neg).append(e)
    for L in (pos, neg):
        L.sort(key=lambda x: (x.get("wracc", 0.0),
                              x.get("abs_t_stat", 0.0),
                              -len(x["raw_description"])), reverse=True)
        for i, e in enumerate(L, 1):
            e["rank"] = i
    return pos, neg


def per_category_summary(df: pd.DataFrame, target: str = "Y", category_col: str = CATEGORY_COL) -> pd.DataFrame:
    if category_col not in df.columns:
        raise ValueError(f"'{category_col}' not found for per-category summary.")
    g = df.groupby(category_col, dropna=False)[target]
    stats = g.agg(n="size", mean="mean", var=lambda s: float(np.var(s, ddof=1)) if len(s) > 1 else 0.0).reset_index()
    N = len(df); global_mean = float(df[target].mean())
    stats["se"] = (stats["var"] / stats["n"]).pow(0.5).astype(float)
    stats["delta"] = stats["mean"] - global_mean
    stats["ci_lo"] = stats["mean"] - 1.96 * stats["se"]
    stats["ci_hi"] = stats["mean"] + 1.96 * stats["se"]
    stats["comp_mean"] = ((N * global_mean) - (stats["n"] * stats["mean"])) / (N - stats["n"]).replace({0: np.nan})
    return stats.sort_values(by="delta", ascending=False)


if __name__ == "__main__":
    run()
