#!/usr/bin/env python3
"""Fix inconsistent mapped categories in Dataset_prelim.jsonl."""

from collections import Counter, defaultdict
import json
from pathlib import Path


def load_jsonl(path: Path):
    """Yield JSON objects for each line in a JSONL file."""
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records):
    """Write JSON objects to a JSONL file."""
    with path.open("w", encoding="utf-8") as outfile:
        for record in records:
            json.dump(record, outfile)
            outfile.write("\n")


def main():
    data_path = Path("Dataset_notrans.jsonl")

    records = list(load_jsonl(data_path))
    grouped = defaultdict(list)

    for record in records:
        grouped[record["question_index"]].append(record)

    ties = []

    for question_index, bucket in grouped.items():
        categories = [item.get("mapped_category") for item in bucket if item.get("mapped_category")]
        if not categories:
            continue

        counts = Counter(categories)
        most_common = counts.most_common()
        top_count = most_common[0][1]
        top_categories = [cat for cat, count in most_common if count == top_count]

        if len(top_categories) == 1:
            dominant_category = top_categories[0]
            for item in bucket:
                item["mapped_category"] = dominant_category
        else:
            ties.append(
                {
                    "question_index": question_index,
                    "tied_categories": top_categories,
                    "counts": counts,
                }
            )

    write_jsonl(data_path, records)

    if ties:
        print("Encountered ties for the following question_index values:")
        for item in ties:
            question_index = item["question_index"]
            tied_categories = ", ".join(item["tied_categories"])
            counts_str = ", ".join(f"{cat}: {item['counts'][cat]}" for cat in item["tied_categories"])
            print(f"- question_index {question_index}: {tied_categories} ({counts_str})")
    else:
        print("All question_index groups resolved to a single dominant mapped_category.")


if __name__ == "__main__":
    main()
