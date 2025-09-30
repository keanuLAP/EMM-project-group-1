import json, math
from pathlib import Path
import pandas as pd

# -------- Config --------
REPO_DIR  = Path(__file__).resolve().parent
IN_JSONL  = REPO_DIR / "Gender_identity_nli_probabilities.jsonl"
OUT_CSV   = REPO_DIR / "emm_bbq_gender_dataset.csv"
OUT_JSONL = REPO_DIR / "emm_bbq_gender_dataset.jsonl"

# -------- Helpers --------
MALE_TAGS   = {"m","male","man","boy"}
FEMALE_TAGS = {"f","female","woman","girl"}
UNK_TAGS    = {"u","unk","unknown","not enough information","n/a","na"}

def norm_tag(tag):
    if tag is None: return None
    t = str(tag).strip().lower()
    if t in MALE_TAGS: return "male"
    if t in FEMALE_TAGS: return "female"
    if t in UNK_TAGS: return "unknown"
    return None

def safe_log(p: float) -> float:
    eps = 1e-12
    p = max(min(float(p), 1.0 - eps), eps)
    return math.log(p)

# -------- Main --------
rows = []

with IN_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        ex = json.loads(line)

        # Copy everything except category, metadata.version, metadata.source
        ex_out = {k:v for k,v in ex.items() if k != "category"}
        if "additional_metadata" in ex_out:
            ex_out["additional_metadata"] = {
                kk: vv for kk,vv in ex_out["additional_metadata"].items()
                if kk not in {"version","source"}
            }

        ans_info = ex.get("answer_info", {})
        probs    = ex.get("answer_probabilities", {})

        # Map answer_info -> male/female/unknown
        tag_by_ans = {}
        for k, v in ans_info.items():
            tag = None
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                tag = norm_tag(v[1])
            elif isinstance(v, str):
                tag = norm_tag(v)
            if tag:
                tag_by_ans[k] = tag

        male_key   = next((k for k,t in tag_by_ans.items() if t=="male"), None)
        female_key = next((k for k,t in tag_by_ans.items() if t=="female"), None)
        unknown_key= next((k for k,t in tag_by_ans.items() if t=="unknown"), None)

        if male_key is None or female_key is None:
            continue
        if male_key not in probs or female_key not in probs:
            continue

        p_male   = float(probs[male_key])
        p_female = float(probs[female_key])
        Y        = safe_log(p_male) - safe_log(p_female)

        # Add our fields
        ex_out["male_key"]   = male_key
        ex_out["female_key"] = female_key
        ex_out["unknown_key"]= unknown_key
        ex_out["p_male"]     = p_male
        ex_out["p_female"]   = p_female
        ex_out["Y"]          = Y

        rows.append(ex_out)

# -------- Save --------
df = pd.DataFrame(rows)
print(f"Usable rows: {len(df)}")

df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV.resolve()}")

with OUT_JSONL.open("w", encoding="utf-8") as w:
    for r in rows:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {OUT_JSONL.resolve()}")
