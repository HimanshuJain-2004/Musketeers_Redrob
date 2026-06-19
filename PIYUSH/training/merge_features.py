from pathlib import Path

import pandas as pd

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_FEATURES = (
    BASE_DIR
    / "artifacts"
    / "candidate_features.parquet"
)

BEHAVIOR_FEATURES = (
    BASE_DIR
    / "artifacts"
    / "behavior_features.parquet"
)

SEMANTIC_FEATURES = (
    BASE_DIR
    / "artifacts"
    / "semantic_scores.parquet"
)

OUTPUT_PATH = (
    BASE_DIR
    / "artifacts"
    / "merged_features.parquet"
)

# =====================================================
# LOAD
# =====================================================

print("Loading feature files...")

candidate_df = pd.read_parquet(
    CANDIDATE_FEATURES
)

behavior_df = pd.read_parquet(
    BEHAVIOR_FEATURES
)

semantic_df = pd.read_parquet(
    SEMANTIC_FEATURES
)

# =====================================================
# MERGE
# =====================================================

print("Merging...")

merged = candidate_df.merge(
    behavior_df,
    on="candidate_id",
    how="inner"
)

merged = merged.merge(
    semantic_df,
    on="candidate_id",
    how="inner"
)

# =====================================================
# VALIDATION
# =====================================================

assert (
    merged["candidate_id"]
    .nunique()
    ==
    len(merged)
)

# =====================================================
# SAVE
# =====================================================

merged.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\nMerge Complete")

print(
    "Rows:",
    len(merged)
)

print(
    "Columns:",
    len(merged.columns)
)

print(
    "\nSaved ->",
    OUTPUT_PATH
)

print(
    "\nColumns:"
)

for col in merged.columns:
    print(col)