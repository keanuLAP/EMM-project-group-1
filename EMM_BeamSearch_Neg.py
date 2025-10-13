from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pysubgroup as ps


# ---- Configuration ---------------------------------------------------------

DATA_PATH = Path("EMM_FINAL.jsonl")
TARGET_COL = "Y"

# Beam search knobs
BEAM_WIDTH = 250          # b
MAX_DEPTH = 3            # d
RESULT_SET_SIZE = 200     # w
MIN_WRACC = 0.005         # min quality (WRAcc)

# Coverage constraints
MIN_COVERAGE_FRAC = 0.05
MIN_ABS_SIZE = 40        # > 30 for stability on ~4.8k rows

# Pre/post reporting
TOP_PRE = 50

# Duplicate collapse tolerances
JACCARD_THRESH = 0.95
DELTA_EPS = 0.01
COV_EPS = 0.02

# Dominance pruning
INCLUSION_TOL = 0.98

# Negations: create boolean complements only for these columns
NEGATION_COLS = {"category"}   # add "context_condition" if you want

# ---- Data containers -------------------------------------------------------

@dataclass
class SubgroupRow:
    subgroup: ps.Subgroup
    mask: np.ndarray
    description: Dict[str, str]
    wracc: float
    mean: float
    var: float
    delta: float
    sg_size: int
    sg_fraction: float
    t_stat: float
    abs_t: float
    rule_len: int


# ---- Dataset helpers -------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    return repr(obj)


def _is_hashable(value: Any) -> bool:
    if pd.isna(value):
        return True
    try:
        hash(value)
        return True
    except TypeError:
        return False


def _normalise_cell(value: Any) -> Any:
    if pd.isna(value):
        return value
    if _is_hashable(value):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=_json_default)
    except TypeError:
        return json.dumps(repr(value))


def normalise_non_hashable_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if series.dropna().map(_is_hashable).all():
            continue
        df[col] = series.apply(_normalise_cell)
    return df


def load_dataset(path: Path) -> pd.DataFrame:
    raw = pd.read_json(path, lines=True)
    if "features" in raw.columns:
        feats = pd.json_normalize(raw["features"])
        df = pd.concat([raw.drop(columns=["features"]), feats], axis=1)
    else:
        df = raw.copy()
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    return normalise_non_hashable_columns(df)


def list_nominal_columns(df: pd.DataFrame, ignore: Sequence[str]) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if col in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if "__ne__" in col:
            cols.append(col)
            continue
        # Include object/string columns
        if pd.api.types.is_object_dtype(df[col]):
            cols.append(col)
            continue
        # Include boolean columns (pandas treats bool as numeric, so handle explicitly)
        if pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
            continue
        cols.append(col)
    return cols


HAS_NOMINAL_SET = hasattr(ps, "NominalSetSelector")
HAS_BINARY_TARGET = hasattr(ps, "BinaryTarget")


def build_search_space(df: pd.DataFrame,
                       include_negations: bool = True,
                       ignore: Sequence[str] = ("Y",)) -> List[ps.Selector]:
    search_space: List[ps.Selector] = []
    for col in list_nominal_columns(df, ignore):
        # All equality selectors (attr == v)
        values = list(pd.Series(df[col], dtype="object").dropna().unique())
        values.sort(key=lambda v: str(v))
        for v in values:
            search_space.append(ps.EqualitySelector(col, v))

        # Optional "not equals" selectors for chosen columns
        if include_negations and col in NEGATION_COLS:
            for v in values:
                search_space.append(NotEqualsSelector(col, v))
    return search_space



class NotEqualsSelector:
    """
    Duck-typed selector for nominal "attr != value".
    Compatible with pysubgroup's expectations:
      - .covers(df) -> boolean mask/Series
      - .attribute_name, .value
      - .selectors -> iterable of leaf selectors (self)
      - __eq__/__hash__ for containment checks in beam expansion
    """
    __slots__ = ("attribute_name", "value")

    def __init__(self, attribute_name: str, value: Any):
        self.attribute_name = attribute_name
        self.value = value

    # --- API expected by pysubgroup ---
    def covers(self, df: pd.DataFrame):
        return df[self.attribute_name] != self.value

    @property
    def selectors(self):
        # behave like a leaf Conjunction: flattening code will iterate this
        return (self,)

    # --- Equality & hashing (used in "sel in last_sg.selectors", set ops, etc.) ---
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NotEqualsSelector):
            return False
        return (self.attribute_name == other.attribute_name) and (self.value == other.value)

    def __hash__(self) -> int:
        return hash(("!=", self.attribute_name, self.value))

    # --- Pretty print ---
    def __str__(self):
        return f"{self.attribute_name} != {self.value}"


# ---- Quality function ------------------------------------------------------

class BiDirWRAcc:
    """Bi-directional WRAcc that works for both numeric and binary targets."""

    def __init__(self) -> None:
        self._dataset_stats: Dict[str, Any] | None = None

    def __str__(self) -> str:
        return "BiDirWRAcc(|S|/N * |mu_S - mu|)"

    # --- Required hooks for pysubgroup quality functions ---
    def calculate_constant_statistics(self, data: pd.DataFrame, target: Any) -> Dict[str, Any]:
        stats = self._compute_stats(
            mask=np.ones(len(data), dtype=bool),
            values=pd.to_numeric(data[TARGET_COL], errors="coerce").to_numpy(),
        )
        self._dataset_stats = stats
        return stats

    def calculate_statistics(self, subgroup: Any, target: Any, data: pd.DataFrame) -> Dict[str, Any]:
        values = pd.to_numeric(data[TARGET_COL], errors="coerce").to_numpy()
        if isinstance(subgroup, slice) and subgroup == slice(None):
            mask = np.ones(len(data), dtype=bool)
        else:
            mask = subgroup_mask(subgroup, data)
        return self._compute_stats(mask=mask, values=values)

    def evaluate(self, subgroup: Any, target: Any, data: pd.DataFrame, statistics: Dict[str, Any] | None = None) -> float:
        stats = statistics or self.calculate_statistics(subgroup, target, data)
        if self._dataset_stats is None:
            self.calculate_constant_statistics(data, target)
        dataset_stats = self._dataset_stats or {}

        mu_dataset = self._extract_mean(dataset_stats, dataset=True)
        mu_s = self._extract_mean(stats, dataset=False)
        coverage = self._extract_coverage(stats)

        if any(math.isnan(x) for x in (mu_dataset, mu_s, coverage)):
            return 0.0
        return abs(mu_s - mu_dataset) * coverage

    def optimistic_estimate(self, *args: Any, **kwargs: Any) -> float:
        return float("inf")

    # --- Helpers ---
    def _compute_stats(self, mask: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        mask = mask.astype(bool)
        sg_vals = values[mask]
        sg_size = int(mask.sum())
        n = int(len(values))
        mean_dataset = float(np.nanmean(values)) if n else float("nan")
        mean_sg = float(np.nanmean(sg_vals)) if sg_size else float("nan")
        return {
            "size_sg": sg_size,
            "size_dataset": n,
            "mean_sg": mean_sg,
            "mean_dataset": mean_dataset,
            "relative_size_sg": (sg_size / n) if n else float("nan"),
        }

    @staticmethod
    def _extract_mean(stats: Dict[str, Any], dataset: bool) -> float:
        # Numeric target stats
        key = "mean_dataset" if dataset else "mean_sg"
        if key in stats and stats[key] is not None:
            return float(stats[key])

        # Binary target stats (shares)
        key = "target_share_dataset" if dataset else "target_share_sg"
        if key in stats and stats[key] is not None:
            return float(stats[key])

        # Generic fallbacks based on sums/counts
        numerator_keys = [
            "positives_dataset" if dataset else "positives_sg",
            "sum" if dataset else "sumSG",
            "targetSum" if dataset else "targetSumSG",
        ]
        denominator_key = "size_dataset" if dataset else "size_sg"
        for num_key in numerator_keys:
            if num_key in stats and denominator_key in stats:
                denom = float(stats[denominator_key])
                if denom:
                    return float(stats[num_key]) / denom
        return float("nan")

    @staticmethod
    def _extract_coverage(stats: Dict[str, Any]) -> float:
        if "relative_size_sg" in stats:
            return float(stats["relative_size_sg"])
        if {"size_sg", "size_dataset"} <= stats.keys():
            denom = float(stats["size_dataset"])
            if denom:
                return float(stats["size_sg"]) / denom
        return float("nan")



# ---- Metrics ---------------------------------------------------------------

def subgroup_mask(sg: ps.Subgroup, df: pd.DataFrame) -> np.ndarray:
    cov = sg.covers(df)
    if isinstance(cov, (pd.Series, pd.Index)):
        cov = cov.to_numpy()
    return np.asarray(cov, dtype=bool)


def subgroup_description(sg: ps.Subgroup) -> Dict[str, str]:
    desc: Dict[str, str] = {}
    selectors = getattr(sg, "selectors", None)
    if selectors is None and hasattr(sg, "subgroup_description"):
        selectors = sg.subgroup_description.selectors
    if selectors is None:
        selectors = []

    for selector in selectors:
        # our custom negation
        if isinstance(selector, NotEqualsSelector):
            attr = selector.attribute_name
            val = selector.value
            desc[attr] = f"{attr} != {val}"
            continue

        # built-in equality
        if isinstance(selector, ps.EqualitySelector):
            attr = selector.attribute_name
            val = getattr(selector, "attribute_value", None)
            desc[attr] = f"{attr} == {val}"
            continue

        # (optional) keep your NominalSetSelector branch if you want
        if HAS_NOMINAL_SET and isinstance(selector, ps.NominalSetSelector):
            attr = selector.attribute_name
            vals = list(selector.values)
            desc[attr] = f"{attr} in {vals}"
            continue

        # fallback
        attr = getattr(selector, "attribute_name", "attr")
        desc[attr] = str(selector)

    return desc


def t_stat_from_global(delta: float, sg_size: int, n: int, global_var: float) -> float:
    if global_var <= 0 or sg_size <= 0 or n <= 0:
        return 0.0
    denom = math.sqrt(global_var * (1.0 / sg_size + 1.0 / n))
    return 0.0 if denom == 0 else delta / denom


# ---- Mining ----------------------------------------------------------------

def run_beam_search(df: pd.DataFrame) -> Tuple[List[SubgroupRow], Dict[str, float]]:
    n = len(df)
    mu = float(df["Y"].mean())
    global_var = float(np.var(df["Y"], ddof=1)) if n > 1 else 0.0

    search_space = build_search_space(df, include_negations=True)
    target = ps.NumericTarget(TARGET_COL)
    qf = BiDirWRAcc()

    task = ps.SubgroupDiscoveryTask(
        data=df,
        target=target,
        search_space=search_space,
        result_set_size=RESULT_SET_SIZE,
        depth=MAX_DEPTH,
        qf=qf,
        min_quality=MIN_WRACC,
    )
    beam = ps.BeamSearch(beam_width=BEAM_WIDTH)
    result = beam.execute(task)

    rows: List[SubgroupRow] = []
    for quality, sg, stats_dict in result.results:
        if sg is True:
            continue
        wracc = float(quality)
        mask = subgroup_mask(sg, df)
        sg_size = int(mask.sum())
        if sg_size == 0:
            continue
        sg_vals = df.loc[mask, TARGET_COL]
        sg_mean = float(sg_vals.mean())
        sg_var = float(np.var(sg_vals, ddof=1)) if sg_size > 1 else 0.0
        delta = sg_mean - mu
        t_val = t_stat_from_global(delta, sg_size, n, global_var)
        desc = subgroup_description(sg)
        selectors = getattr(sg, "selectors", None)
        if selectors is None and hasattr(sg, "subgroup_description"):
            selectors = sg.subgroup_description.selectors
        if selectors is None:
            selectors = []
        rows.append(
            SubgroupRow(
                subgroup=sg,
                mask=mask,
                description=desc,
                wracc=float(wracc),
                mean=sg_mean,
                var=sg_var,
                delta=delta,
                sg_size=sg_size,
                sg_fraction=sg_size / n,
                t_stat=t_val,
                abs_t=abs(t_val),
                rule_len=len(selectors),
            )
        )

    rows.sort(key=lambda r: (r.wracc, r.abs_t, -r.rule_len), reverse=True)
    return rows, {"n": n, "global_mean": mu, "global_var": global_var}


def apply_size_constraints(rows: List[SubgroupRow], n: int) -> List[SubgroupRow]:
    min_abs = max(MIN_ABS_SIZE, int(math.ceil(MIN_COVERAGE_FRAC * n)))
    return [
        r for r in rows
        if r.sg_size >= min_abs and r.sg_fraction >= MIN_COVERAGE_FRAC and r.wracc >= MIN_WRACC
    ]


# ---- Post-processing -------------------------------------------------------

def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else inter / union

def collapse_by_inclusion(rows: List[SubgroupRow], inclusion_tol: float = 0.995) -> List[SubgroupRow]:
    """
    Globally collapse by inclusion, independent of input order.

    If A is (approximately) included in B, i.e., |A ∩ B| / |A| ≥ inclusion_tol,
    then drop A and keep B. We iterate until no changes occur.
    Preference is given to the *bigger* subgroup (higher coverage/size).
    Ties break on (wracc, |t|, shorter rule).

    Complexity: O(k^2) per pass, usually fine for k<=100.
    """
    if not rows:
        return []

    # Sort by size (coverage) descending first so supersets tend to appear earlier.
    # Then tie-break for reproducibility.
    current = sorted(
        rows,
        key=lambda r: (r.sg_size, r.wracc, r.abs_t, -r.rule_len),
        reverse=True,
    )

    changed = True
    while changed:
        changed = False
        kept: List[SubgroupRow] = []

        for r in current:
            # 1) if r is a subset of any already-kept k, drop r
            drop_r = False
            for k in kept:
                if inclusion_ratio(r.mask, k.mask) >= inclusion_tol:
                    # r ⊆ k  -> drop r, keep the bigger k
                    drop_r = True
                    changed = True
                    break
            if drop_r:
                continue

            # 2) if any kept k is a subset of r, remove k (keep the bigger r)
            #    do it in a batch so multiple subsets get removed
            to_remove = []
            for i, k in enumerate(kept):
                if inclusion_ratio(k.mask, r.mask) >= inclusion_tol:
                    # k ⊆ r  -> remove k
                    to_remove.append(i)
            if to_remove:
                for i in reversed(to_remove):
                    kept.pop(i)
                changed = True

            kept.append(r)

        # Next pass compares only survivors
        current = kept

    # Final deterministic ordering for reporting
    return sorted(current, key=lambda r: (r.wracc, r.abs_t, -r.rule_len), reverse=True)


def collapse_near_duplicates(rows: Iterable[SubgroupRow]) -> List[SubgroupRow]:
    kept: List[SubgroupRow] = []
    for row in rows:
        duplicate = False
        for other in kept:
            J = jaccard(row.mask, other.mask)
            cov_diff = abs(row.sg_fraction - other.sg_fraction)
            delta_diff = abs(row.delta - other.delta)
            if J >= JACCARD_THRESH and cov_diff <= COV_EPS and delta_diff <= DELTA_EPS:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    kept.sort(key=lambda r: (r.wracc, r.abs_t, -r.rule_len), reverse=True)
    return kept


def dominates(a: SubgroupRow, b: SubgroupRow) -> bool:
    dims_a = (a.sg_fraction, abs(a.delta), a.abs_t)
    dims_b = (b.sg_fraction, abs(b.delta), b.abs_t)
    ge = all(da >= db for da, db in zip(dims_a, dims_b))
    gt = any(da > db for da, db in zip(dims_a, dims_b))
    return ge and gt


def pareto_front(rows: Iterable[SubgroupRow], eps_cov=0.01, eps_delta=0.005, eps_t=0.1) -> List[SubgroupRow]:
    def dims(r): return (r.sg_fraction, abs(r.delta), r.abs_t)
    out = []
    for r in rows:
        r_cov, r_d, r_t = dims(r)
        dominated = False
        for s in rows:
            if s is r: continue
            s_cov, s_d, s_t = dims(s)
            ge = (s_cov >= r_cov - eps_cov) and (s_d >= r_d - eps_delta) and (s_t >= r_t - eps_t)
            gt = ((s_cov > r_cov + eps_cov) or
                  (s_d   > r_d   + eps_delta) or
                  (s_t   > r_t   + eps_t))
            if ge and gt:
                dominated = True
                break
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: (r.wracc, r.abs_t, -r.rule_len), reverse=True)


def inclusion_ratio(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum()
    if denom == 0:
        return 0.0
    return np.logical_and(a, b).sum() / float(denom)


def dominance_prune(rows: Iterable[SubgroupRow]) -> List[SubgroupRow]:
    kept: List[SubgroupRow] = []
    for row in rows:
        dominated = False
        for other in kept:
            if inclusion_ratio(row.mask, other.mask) >= INCLUSION_TOL and dominates(other, row):
                dominated = True
                break
        if not dominated:
            kept.append(row)
    kept.sort(key=lambda r: (r.wracc, r.abs_t, -r.rule_len), reverse=True)
    return kept


# ---- Reporting -------------------------------------------------------------

def format_row(row: SubgroupRow, idx: int) -> str:
    desc = "; ".join(f"{k}: {v}" for k, v in row.description.items())
    return (
        f"#{idx:2d} "
        f"wracc={row.wracc:.4f} var={row.var:.4f} "
        f"delta={row.delta:+.4f} "
        f"coverage={row.sg_fraction:.3f} size={row.sg_size:4d} "
        f"|t|={row.abs_t:.3f} :: {desc}"
    )


def print_section(title: str, rows: List[SubgroupRow]) -> None:
    print(f"\n=== {title} (showing {len(rows)}) ===")
    for idx, row in enumerate(rows, 1):
        print(format_row(row, idx))


def split_by_direction(rows: Iterable[SubgroupRow]) -> Tuple[List[SubgroupRow], List[SubgroupRow]]:
    male: List[SubgroupRow] = []
    female: List[SubgroupRow] = []
    for row in rows:
        if row.delta > 0:
            male.append(row)
        elif row.delta < 0:
            female.append(row)
    male.sort(key=lambda r: (r.wracc, r.abs_t, -r.rule_len), reverse=True)
    female.sort(key=lambda r: (r.wracc, r.abs_t, -r.rule_len), reverse=True)
    return male, female


def per_category_summary(df: pd.DataFrame, target: str = "Y", category_col: str = "category") -> pd.DataFrame:
    if category_col not in df.columns:
        raise ValueError(f"{category_col!r} column not available for summary")
    group = df.groupby(category_col, dropna=False)[target]
    stats = group.agg(
        n="size",
        mean="mean",
        var=lambda s: float(np.var(s, ddof=1)) if len(s) > 1 else 0.0,
    ).reset_index()
    N = len(df)
    global_mean = float(df[target].mean())
    stats["se"] = (stats["var"] / stats["n"]).pow(0.5).astype(float)
    stats["delta"] = stats["mean"] - global_mean
    stats["ci_lo"] = stats["mean"] - 1.96 * stats["se"]
    stats["ci_hi"] = stats["mean"] + 1.96 * stats["se"]
    stats["comp_mean"] = ((N * global_mean) - (stats["n"] * stats["mean"])) / (N - stats["n"]).replace({0: np.nan})
    return stats


def print_category_summary(df: pd.DataFrame) -> None:
    summary = per_category_summary(df)
    summary["abs_delta"] = summary["delta"].abs()
    summary = summary.sort_values(by="abs_delta", ascending=False).drop(columns=["abs_delta"])
    print("\n=== Per-category summary ===")
    print(summary.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ---- Orchestration --------------------------------------------------------

def main() -> None:
    df = load_dataset(DATA_PATH)
    
    rows, stats = run_beam_search(df)

    constrained = apply_size_constraints(rows, stats["n"])
    pre_top = constrained[:TOP_PRE]
    print_section(f"Top {min(TOP_PRE, len(constrained))} pre-processing", pre_top)
    
    inc = collapse_by_inclusion(pre_top)
    print_section(f"Post-inclusion collapse ({len(inc)} retained of {len(pre_top)})", inc)
    
    dedup = collapse_near_duplicates(inc)
    print_section(f"Post-duplicate collapse ({len(dedup)} retained of {len(inc)})", dedup)
    
    pruned = dominance_prune(dedup)
    print_section(f"Dominance-pruned ({len(pruned)} retained of {len(dedup)})", pruned)

    # pareto = pareto_front(pruned)
    # print_section(f"Pareto front ({len(pareto)} retained of {len(pruned)})", pareto)

    male, female = split_by_direction(pruned)
    print_section("Male-favoring (delta > 0)", male)
    print_section("Female-favoring (delta < 0)", female)

    print_category_summary(df)


if __name__ == "__main__":
    main()
