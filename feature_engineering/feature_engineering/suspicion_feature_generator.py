import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = BASE_DIR / "data" / "candidates.jsonl"
MERGED_FEATURES_PATH = BASE_DIR / "artifacts" / "merged_features.parquet"
OUTPUT_PATH = BASE_DIR / "artifacts" / "suspicion_features.parquet"

def main():
    print("Loading merged features to get existing timeline features...")
    # Load existing features: total_experience_years, num_roles, years_exp
    df_merged = pd.read_parquet(MERGED_FEATURES_PATH, columns=["candidate_id", "total_experience_years", "num_roles", "years_exp", "timeline_gap_count"])
    
    # We need to get num_skills from JSONL
    candidate_skills = {}
    
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading num_skills from JSONL"):
            candidate = json.loads(line)
            c_id = candidate.get("candidate_id")
            skills = candidate.get("skills", [])
            candidate_skills[c_id] = len(skills)
            
    df_merged["num_skills"] = df_merged["candidate_id"].map(candidate_skills).fillna(0)
    
    # 1. Skill Density
    df_merged["skill_density"] = df_merged["num_skills"] / (df_merged["total_experience_years"] + 1)
    
    # 2. Role Density
    df_merged["role_density"] = df_merged["num_roles"] / (df_merged["total_experience_years"] + 1)
    
    # 3. Experience Consistency
    # claimed years vs timeline years
    df_merged["experience_gap_years"] = (df_merged["years_exp"] - df_merged["total_experience_years"]).abs()
    df_merged["experience_consistency_score"] = 1.0 / (1.0 + df_merged["experience_gap_years"])
    
    # 4. Skill Inflation
    df_merged["skills_per_role"] = df_merged["num_skills"] / df_merged["num_roles"].clip(lower=1)
    
    # 5. Timeline Consistency
    # Since formula wasn't specified, let's use gap count
    df_merged["timeline_consistency_score"] = 1.0 / (1.0 + df_merged["timeline_gap_count"])
    
    # 6. Final Suspicion Score
    # Combine the suspicion indicators (higher is more suspicious)
    # skill_density + role_density + experience_gap_years
    df_merged["suspicion_score"] = df_merged["skill_density"] + df_merged["role_density"] + df_merged["experience_gap_years"]
    
    # Select only the new columns + candidate_id
    out_cols = [
        "candidate_id", "num_skills", "skill_density", "role_density",
        "experience_gap_years", "experience_consistency_score",
        "skills_per_role", "timeline_consistency_score", "suspicion_score"
    ]
    df_out = df_merged[out_cols]
    
    print("\n--- Validation Stats ---")
    print(df_out[["skill_density", "role_density", "experience_gap_years", "suspicion_score"]].describe())
    
    print("\n--- Top 10 by Suspicion Score ---")
    print(df_out.sort_values("suspicion_score", ascending=False).head(10)[["candidate_id", "num_skills", "experience_gap_years", "suspicion_score"]])

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved suspicion features to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
