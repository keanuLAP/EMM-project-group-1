import json
from pathlib import Path

# Input and output file paths
repo_path = Path(__file__).resolve().parent
input_file = repo_path / "Dataset_prelim.jsonl"
output_file = repo_path / "Dataset_notrans.jsonl"

filtered_entries = []
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        entry = json.loads(line)
        # Keep entry only if no value contains "trans"
        if not any("trans" in str(value).lower() for value in entry.values()):
            filtered_entries.append(entry)

# Write filtered entries back to JSONL
with open(output_file, "w", encoding="utf-8") as outfile:
    for entry in filtered_entries:
        outfile.write(json.dumps(entry) + "\n")

print(f"Filtered file saved as: {output_file}")
