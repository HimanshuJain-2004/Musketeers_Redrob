import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_FEATURES_PATH = BASE_DIR / "artifacts" / "merged_features.parquet"
OUTPUT_DIR = BASE_DIR / "artifacts"

def main():
    print("Loading merged features for audit...")
    df = pd.read_parquet(MERGED_FEATURES_PATH)
    
    # Exclude non-feature columns
    features_df = df.drop(columns=["candidate_id"])
    
    # ---------------------------------------------------------
    # Audit 2: Missing Value Audit
    # ---------------------------------------------------------
    print("Running Missing Value Audit...")
    missing_ratios = features_df.isnull().mean().round(4)
    missing_df = pd.DataFrame({
        "Feature Name": missing_ratios.index,
        "Missing %": missing_ratios.values * 100
    })
    
    # Flags
    conditions = [
        missing_df["Missing %"] > 50,
        missing_df["Missing %"] > 25,
        missing_df["Missing %"] > 10,
    ]
    choices = ["> 50%", "> 25%", "> 10%"]
    missing_df["Flag"] = np.select(conditions, choices, default="")
    
    missing_df.sort_values("Missing %", ascending=False, inplace=True)
    missing_df.to_csv(OUTPUT_DIR / "missing_value_report.csv", index=False)
    
    # ---------------------------------------------------------
    # Audit 3: Distribution Audit
    # ---------------------------------------------------------
    print("Running Distribution Audit...")
    
    dist_data = []
    for col in features_df.columns:
        nunique = features_df[col].nunique()
        mean_val = features_df[col].mean()
        std_val = features_df[col].std()
        min_val = features_df[col].min()
        max_val = features_df[col].max()
        
        # Flags
        flag = ""
        if nunique == 0:
            flag = "All Nulls"
        elif nunique == 1:
            if mean_val == 0:
                flag = "All Zeros"
            elif mean_val == 1:
                flag = "All Ones"
            else:
                flag = "Constant"
        else:
            # Check near constant (99.9% identical)
            top_val_freq = features_df[col].value_counts(normalize=True).iloc[0]
            if top_val_freq > 0.999:
                flag = "Near Constant (>99.9%)"
                
        dist_data.append({
            "Feature Name": col,
            "nunique": nunique,
            "mean": round(mean_val, 4) if pd.notnull(mean_val) else None,
            "std": round(std_val, 4) if pd.notnull(std_val) else None,
            "min": round(min_val, 4) if pd.notnull(min_val) else None,
            "max": round(max_val, 4) if pd.notnull(max_val) else None,
            "Flag": flag
        })
        
    dist_df = pd.DataFrame(dist_data)
    dist_df.to_csv(OUTPUT_DIR / "feature_distribution_report.csv", index=False)
    
    # ---------------------------------------------------------
    # Audit 4: Correlation Audit
    # ---------------------------------------------------------
    print("Running Correlation Audit...")
    corr_matrix = features_df.corr().abs()
    
    corr_data = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            c_val = corr_matrix.iloc[i, j]
            if c_val > 0.95:
                corr_data.append({
                    "Feature 1": cols[i],
                    "Feature 2": cols[j],
                    "Correlation": round(c_val, 4),
                    "Flag": "corr > 0.95"
                })
                
    corr_df = pd.DataFrame(corr_data)
    if not corr_df.empty:
        corr_df.sort_values("Correlation", ascending=False, inplace=True)
    corr_df.to_csv(OUTPUT_DIR / "correlation_report.csv", index=False)
    
    print("\nAudits Complete! Reports saved to artifacts/")

if __name__ == "__main__":
    main()
