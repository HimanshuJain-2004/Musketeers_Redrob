import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_FEATURES_PATH = BASE_DIR / "artifacts" / "merged_features.parquet"
OUTPUT_PATH = BASE_DIR / "artifacts" / "interaction_features.parquet"

def main():
    print("Loading merged features to generate interactions...")
    # Load required columns
    cols = [
        "candidate_id", "production_depth_score", "leadership_depth_score",
        "jd_core_score", "career_growth_score", "ownership_score",
        "career_semantic_score", "career_stability_score"
    ]
    df = pd.read_parquet(MERGED_FEATURES_PATH, columns=cols)
    
    # 1. Technical Leadership
    df["technical_leadership_score"] = df["production_depth_score"] * df["leadership_depth_score"]
    
    # 2. Production Retrieval
    df["production_retrieval_score"] = df["jd_core_score"] * df["production_depth_score"]
    
    # 3. Senior Production
    df["senior_production_score"] = df["production_depth_score"] * df["career_growth_score"]
    
    # 4. Ownership Relevance
    df["ownership_relevance_score"] = df["ownership_score"] * df["career_semantic_score"]
    
    # 5. Stability Leadership
    df["stability_leadership_score"] = df["career_stability_score"] * df["leadership_depth_score"]
    
    # 6. Seniority Relevance (User requested addition)
    df["seniority_relevance_score"] = df["career_growth_score"] * df["career_semantic_score"]
    
    # Select final columns
    out_cols = [
        "candidate_id", "technical_leadership_score", "production_retrieval_score",
        "senior_production_score", "ownership_relevance_score",
        "stability_leadership_score", "seniority_relevance_score"
    ]
    df_out = df[out_cols]
    
    print("\n--- Validation Stats ---")
    print(df_out.describe())
    
    print("\n--- Top 10 by Technical Leadership ---")
    print(df_out.sort_values("technical_leadership_score", ascending=False).head(10)[["candidate_id", "technical_leadership_score", "production_retrieval_score"]])

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved interaction features to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
