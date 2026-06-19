import os
import sys
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "JAIN", "artifacts")
PIYUSH_ARTIFACTS = os.path.join(BASE_DIR, "PIYUSH", "artifacts")
CHECKPOINT_FILE = os.path.join(ARTIFACTS_DIR, "llm_evaluations_checkpoint.jsonl")
OUTPUT_FILE = os.path.join(ARTIFACTS_DIR, "training_dataset.parquet")

def create_dataset():
    print("Parsing LLM evaluations...")
    evals = {}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    cid = record.get("candidate_id")
                    ev = record.get("evaluation", {})
                    # If duplicate, we just overwrite and keep the latest
                    evals[cid] = ev
                except json.JSONDecodeError:
                    pass

    data = []
    for cid, ev in evals.items():
        if not ev: continue
        
        # Some fields might be missing depending on prompt compliance, use defaults or fallback
        fit_score = ev.get("fit_score", 0)
        technical_fit = ev.get("technical_fit", 0)
        product_fit = ev.get("product_fit", 0)
        career_fit = ev.get("career_fit", 0)
        behavioral_fit = ev.get("behavioral_fit", 0)
        honeypot_label = ev.get("honeypot_label", False)
        
        # Ensure they are numeric
        try: fit_score = float(fit_score)
        except: fit_score = 0
        try: technical_fit = float(technical_fit)
        except: technical_fit = 0
        try: product_fit = float(product_fit)
        except: product_fit = 0
        try: career_fit = float(career_fit)
        except: career_fit = 0
        try: behavioral_fit = float(behavioral_fit)
        except: behavioral_fit = 0
        
        # Calculate composite score
        # recruiter_composite_score = 0.40 * fit_score + 0.20 * technical_fit + 0.15 * product_fit + 0.15 * career_fit + 0.10 * behavioral_fit
        composite_score = (
            0.40 * fit_score +
            0.20 * technical_fit +
            0.15 * product_fit +
            0.15 * career_fit +
            0.10 * behavioral_fit
        )
        
        # Normally each is out of 100, if so composite is out of 100.
        # But if fit_score is out of 100 and others are out of 10 (or vice versa), we'd need to normalize.
        # Usually they are all out of 100. Let's just clip to 0-100 just in case.
        composite_score = max(0, min(100, composite_score))
        
        data.append({
            "candidate_id": cid,
            "fit_score": fit_score,
            "technical_fit": technical_fit,
            "product_fit": product_fit,
            "career_fit": career_fit,
            "behavioral_fit": behavioral_fit,
            "recruiter_composite_score": composite_score,
            "honeypot_label": int(honeypot_label)
        })
        
    labels_df = pd.DataFrame(data)
    print(f"Loaded {len(labels_df)} labels from checkpoint.")

    print("Loading features...")
    # Piyush's features
    try:
        features_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "merged_features.parquet"))
        print(f"Loaded merged_features with shape {features_df.shape}")
    except FileNotFoundError:
        print("merged_features.parquet not found. Attempting to merge individual files...")
        c_feat = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "candidate_features.parquet"))
        b_feat = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "behavior_features.parquet"))
        s_feat = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "semantic_scores.parquet"))
        features_df = c_feat.merge(b_feat, on="candidate_id", how="left").merge(s_feat, on="candidate_id", how="left")
        print(f"Created merged features with shape {features_df.shape}")
        
    # Merge labels with features
    training_df = labels_df.merge(features_df, on="candidate_id", how="inner")
    
    # We might have extra columns that aren't useful for training but let's keep them for now,
    # except we need to make sure we don't have object columns besides candidate_id.
    
    print(f"Final training dataset shape: {training_df.shape}")
    
    # Save to parquet
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    training_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved training dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_dataset()
