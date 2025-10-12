"""Dataset transformation utility for EMM input formats.

This script reads the original `emm_bbq_gender_dataset.csv` file and produces a
simplified dataset tailored for the EMM algorithm. Both CSV and JSONL outputs
are generated, with features and metadata split according to the specification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

# File locations relative to this script
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "emm_bbq_gender_dataset.csv"
DEFAULT_CSV_OUTPUT = BASE_DIR / "EMM_FINAL.csv"
DEFAULT_JSONL_OUTPUT = BASE_DIR / "EMM_FINAL.jsonl"

# Labels for the context length bins ordered from shortest to longest
CONTEXT_BIN_LABELS = ["short", "medium", "long", "very_long"]


def _assign_context_bins(lengths: pd.Series) -> pd.Series:
    """Return categorical bins for the provided context lengths.

    The number of bins adapts to the diversity of lengths present in the
    dataset; fewer unique values result in fewer bins. Falls back to a single
    "unknown" bin if `qcut` cannot be computed.
    """
    unique_count = lengths.nunique(dropna=True)
    if unique_count == 0:
        return pd.Series(["unknown"] * len(lengths), index=lengths.index)

    # Use as many bins as we have unique values (up to the label list size)
    bin_count = min(unique_count, len(CONTEXT_BIN_LABELS))
    labels = CONTEXT_BIN_LABELS[:bin_count]

    try:
        bins = pd.qcut(lengths, q=bin_count, labels=labels, duplicates="drop")
    except ValueError:
        # Happens when all values are identical; emit a single label
        bins = pd.Series([labels[0]] * len(lengths), index=lengths.index)

    return bins.astype(str)


def _build_record(row: pd.Series, context_bin: str) -> Dict[str, object]:
    """Convert a dataframe row into the desired output structure."""
    meta = {
        "question_index": row.get("question_index"),
        "male_key": row.get("male_key"),
        "female_key": row.get("female_key"),
        "unknown_key": row.get("unknown_key"),
        "context": row.get("context"),
        "question": row.get("question"),
    }

    features = {
        "context_condition": row.get("context_condition"),
        "question_polarity": row.get("question_polarity"),
        "subcategory": _safe_extract_subcategory(row.get("additional_metadata")),
        "category": row.get("mapped_category"),
        "context_length": context_bin,
    }

    return {
        "item_id": row.get("example_id"),
        "meta": meta,
        "Y": row.get("Y"),
        "features": features,
    }


def _safe_extract_subcategory(raw_metadata: object) -> object:
    """Extract `subcategory` when metadata is a stringified dict."""
    if isinstance(raw_metadata, dict):
        return raw_metadata.get("subcategory")

    if isinstance(raw_metadata, str) and raw_metadata.strip():
        try:
            parsed = json.loads(raw_metadata.replace("'", '"'))
            if isinstance(parsed, dict):
                return parsed.get("subcategory")
        except json.JSONDecodeError:
            # Leave as the original string if parsing fails
            return raw_metadata
    return raw_metadata


def _load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    return pd.read_csv(input_path)


def _write_jsonl(records: Iterable[Dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(records: List[Dict[str, object]], output_path: Path) -> None:
    """Persist records to CSV with JSON-encoded map columns."""
    frame = pd.DataFrame(records)
    for column in ("meta", "features"):
        frame[column] = frame[column].apply(lambda value: json.dumps(value, ensure_ascii=False))
    frame.to_csv(output_path, index=False)


def main(
    input_path: Path = DEFAULT_INPUT,
    csv_output_path: Path = DEFAULT_CSV_OUTPUT,
    jsonl_output_path: Path = DEFAULT_JSONL_OUTPUT,
) -> None:
    df = _load_dataset(input_path)

    context_lengths = df["context"].fillna("").apply(lambda text: len(str(text).split()))
    context_bins = _assign_context_bins(context_lengths)

    records = [_build_record(row, bin_label) for (_, row), bin_label in zip(df.iterrows(), context_bins)]

    _write_csv(records, csv_output_path)
    _write_jsonl(records, jsonl_output_path)

    print(f"Created {csv_output_path.name} and {jsonl_output_path.name}")


if __name__ == "__main__":
    main()
