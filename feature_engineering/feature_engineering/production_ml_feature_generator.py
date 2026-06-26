import json
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = BASE_DIR / "data" / "candidates.jsonl"
OUTPUT_PATH = BASE_DIR / "artifacts" / "production_features.parquet"

PRODUCTION_TERMS = [
    r"\bproduction\b", r"\bproductionized\b", r"\bdeployed\b",
    r"\bdeployment\b", r"\bgo-live\b", r"\bserving\b",
    r"\binference\b", r"\breal[- ]time\b", r"\bonline system\b",
    r"\bhigh traffic\b", r"\bscalable\b", r"\bscaling\b",
    r"\bdistributed\b", r"\bdeployed machine learning\b",
    r"\bdeployed models\b", r"\bmodel serving\b", r"\bfeature store\b",
    r"\bmodel monitoring\b", r"\bonline inference\b",
    r"\breal time inference\b", r"\bproduction pipeline\b",
    r"\bml pipeline\b", r"\bdata pipeline\b"
]

MONITORING_TERMS = [
    r"\bmonitoring\b", r"\bobservability\b", r"\balerting\b",
    r"\bmetrics\b", r"\blogging\b", r"\bprometheus\b", r"\bgrafana\b"
]

INFRA_TERMS = [
    r"\bkubernetes\b", r"\bk8s\b", r"\bdocker\b", r"\baws\b",
    r"\bgcp\b", r"\bazure\b", r"\bterraform\b", r"\bcicd\b", r"\bmlops\b"
]

MLOPS_TERMS = [
    r"\bmlops\b", r"\bairflow\b", r"\bkubeflow\b", r"\bfeature store\b",
    r"\bmodel registry\b", r"\bmodel versioning\b", r"\bmodel monitoring\b"
]

LLM_WRAPPER_TERMS = [
    r"\blangchain\b", r"\bopenai api\b", r"\bgpt[- ]4\b",
    r"\bchatgpt\b", r"\brag\b"
]

def compile_regex(terms):
    return re.compile("|".join(terms), re.IGNORECASE)

RE_PROD = compile_regex(PRODUCTION_TERMS)
RE_MONITOR = compile_regex(MONITORING_TERMS)
RE_INFRA = compile_regex(INFRA_TERMS)
RE_MLOPS = compile_regex(MLOPS_TERMS)
RE_WRAPPER = compile_regex(LLM_WRAPPER_TERMS)

def build_career_history_text(profile):
    career_history = profile.get("career_history", [])
    parts = []
    for job in career_history:
        desc = job.get("description", "").strip()
        resp = job.get("responsibilities", "").strip()
        summary = job.get("summary", "").strip()
        achievements = job.get("achievements", "").strip()
        
        if desc: parts.append(desc)
        if resp: parts.append(resp)
        if summary: parts.append(summary)
        if achievements: parts.append(achievements)
        
    return " ".join(parts)

def count_matches(pattern, text):
    return len(pattern.findall(text))

def main():
    candidate_ids = []
    prod_counts = []
    monitor_counts = []
    infra_counts = []
    mlops_counts = []
    wrapper_counts = []
    
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing candidates"):
            
            candidate = json.loads(line)
            c_id = candidate.get("candidate_id")
            text = build_career_history_text(candidate)
            
            candidate_ids.append(c_id)
            prod_counts.append(count_matches(RE_PROD, text))
            monitor_counts.append(count_matches(RE_MONITOR, text))
            infra_counts.append(count_matches(RE_INFRA, text))
            mlops_counts.append(count_matches(RE_MLOPS, text))
            wrapper_counts.append(count_matches(RE_WRAPPER, text))
            
    df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "prod_ml_count": prod_counts,
        "monitoring_count": monitor_counts,
        "infra_count": infra_counts,
        "mlops_count": mlops_counts,
        "llm_wrapper_count": wrapper_counts
    })
    
    df["prod_ml_present"] = (df["prod_ml_count"] > 0).astype(int)
    df["monitoring_present"] = (df["monitoring_count"] > 0).astype(int)
    df["infra_present"] = (df["infra_count"] > 0).astype(int)
    df["mlops_present"] = (df["mlops_count"] > 0).astype(int)
    
    df["production_ml_score"] = df["prod_ml_count"] + df["monitoring_count"] + df["infra_count"]
    df["production_depth_score"] = (df["prod_ml_count"] * 2) + df["monitoring_count"] + df["infra_count"] + df["mlops_count"]
    
    df["wrapper_to_production_ratio"] = df["llm_wrapper_count"] / (df["prod_ml_count"] + 1)
    
    print("\n--- Validation on Sample ---")
    print(df[["prod_ml_count", "monitoring_count", "infra_count", "mlops_count", "production_ml_score", "production_depth_score"]].describe())
    
    print("\n--- Top 20 Candidates by Depth Score ---")
    top_candidates = df.sort_values("production_depth_score", ascending=False).head(20)
    print(top_candidates[["candidate_id", "prod_ml_count", "monitoring_count", "infra_count", "mlops_count", "production_depth_score"]])

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved production features to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
