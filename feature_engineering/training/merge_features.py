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

CAREER_SEMANTIC_FEATURES = (
    BASE_DIR
    / "artifacts"
    / "career_semantic_features.parquet"
)

PRODUCTION_FEATURES = (
    BASE_DIR
    / "artifacts"
    / "production_features.parquet"
)

OWNERSHIP_FEATURES = (
    BASE_DIR
    / "artifacts"
    / "ownership_features.parquet"
)

CAREER_PROGRESSION_FEATURES = (
    BASE_DIR
    / "artifacts"
    / "career_progression_features.parquet"
)

COMPANY_BACKGROUND_FEATURES = (
    BASE_DIR
    / "artifacts"
    / "company_background_features.parquet"
)

SUSPICION_FEATURES = (
    BASE_DIR
    / "artifacts"
    / "suspicion_features.parquet"
)

INTERACTION_FEATURES = (
    BASE_DIR
    / "artifacts"
    / "interaction_features.parquet"
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

career_semantic_df = pd.read_parquet(
    CAREER_SEMANTIC_FEATURES
)

production_df = pd.read_parquet(
    PRODUCTION_FEATURES
)

ownership_df = pd.read_parquet(
    OWNERSHIP_FEATURES
)

progression_df = pd.read_parquet(
    CAREER_PROGRESSION_FEATURES
)

company_df = pd.read_parquet(
    COMPANY_BACKGROUND_FEATURES
)

suspicion_df = pd.read_parquet(
    SUSPICION_FEATURES
)

interaction_df = pd.read_parquet(
    INTERACTION_FEATURES
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

merged = merged.merge(
    career_semantic_df,
    on="candidate_id",
    how="inner"
)

merged = merged.merge(
    production_df,
    on="candidate_id",
    how="inner"
)

merged = merged.merge(
    ownership_df,
    on="candidate_id",
    how="inner"
)

merged = merged.merge(
    progression_df,
    on="candidate_id",
    how="inner"
)

merged = merged.merge(
    company_df,
    on="candidate_id",
    how="inner"
)

merged = merged.merge(
    suspicion_df,
    on="candidate_id",
    how="inner"
)

merged = merged.merge(
    interaction_df,
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