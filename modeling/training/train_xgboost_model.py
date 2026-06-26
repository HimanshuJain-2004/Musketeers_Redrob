"""
train_and_save_try2_model.py
----------------------------
Trains the canonical XGBoost Regressor on the try_2 target
(0.8 * strict_weighted_score + 0.2 * raw_fit_score) using the 70
selected features, saves it as modeling/artifacts/xgboost_ranking_model.pkl,
generates the feature importance PNG, and produces the final Top-100 CSV.
"""
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from modeling.data_modules.label_loader import LabelLoader
from modeling.data_modules.dataset_builder import DatasetBuilder

JSONL_PATH     = os.path.join(BASE_DIR, "modeling", "artifacts", "llm_evaluations_checkpoint.jsonl")
FEATURES_PATH  = os.path.join(BASE_DIR, "feature_engineering", "artifacts", "merged_features.parquet")
SEL_FEAT_PATH  = os.path.join(BASE_DIR, "modeling", "artifacts", "model_feature_schema.json")
MODEL_OUT      = os.path.join(BASE_DIR, "modeling", "artifacts", "xgboost_ranking_model.pkl")
PLOT_OUT       = os.path.join(BASE_DIR, "modeling", "artifacts", "feature_importance_XGBRegressor_try2.png")
SUBMISSION_CSV = os.path.join(BASE_DIR, "team_musketeers.csv")

# ── 1. Labels ──────────────────────────────────────────────────────────────
print("Loading labels...")
loader    = LabelLoader(JSONL_PATH)
labels_df = loader.load_labels()

strict = (
    labels_df["technical_fit"] * 0.50 +
    labels_df["product_fit"]   * 0.30 +
    labels_df["career_fit"]    * 0.10 +
    labels_df["behavioral_fit"]* 0.10
)
labels_df["score"] = (0.8 * strict + 0.2 * labels_df["raw_fit_score"]) / 100.0

# ── 2. Features ─────────────────────────────────────────────────────────────
print("Building dataset...")
with open(SEL_FEAT_PATH) as f:
    selected_features = json.load(f)["selected_features"]

builder = DatasetBuilder(FEATURES_PATH)
ds      = builder.build_dataset(labels_df)

for split in ("X_train", "X_val", "X_test"):
    ds[split] = ds[split][selected_features]

X_all = pd.concat([ds["X_train"], ds["X_val"], ds["X_test"]])
y_all = pd.concat([ds["y_train"], ds["y_val"], ds["y_test"]])
X_all = X_all.fillna(0)

# ── 3. Train ────────────────────────────────────────────────────────────────
print("Training XGBoost Regressor (try_2 target)...")
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_all, y_all)

# ── 4. Save model ───────────────────────────────────────────────────────────
print(f"Saving model -> {MODEL_OUT}")
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)

# ── 5. Feature Importance PNG ───────────────────────────────────────────────
print("Generating feature importance plot...")
importances = model.feature_importances_
feat_df = pd.DataFrame({"feature": selected_features, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=False).head(25)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(feat_df["feature"][::-1], feat_df["importance"][::-1], color="#4C9BE8")
ax.set_xlabel("Importance (gain)", fontsize=12)
ax.set_title("XGBoost Regressor (try_2) — Top 25 Feature Importances", fontsize=14, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {PLOT_OUT}")

# ── 6. Inference over full 100k ─────────────────────────────────────────────
print("Loading full candidate pool...")
full_df = pd.read_parquet(FEATURES_PATH)
X_infer = full_df[selected_features].fillna(0)

print("Predicting scores...")
full_df["predicted_score"] = model.predict(X_infer)

top_df = (
    full_df
    .sort_values(by=["predicted_score", "candidate_id"], ascending=[False, True])
    .head(100)
    .copy()
    .reset_index(drop=True)
)
top_df["rank"] = range(1, 101)

print("\nTop 10 candidates:")
print(top_df[["rank", "candidate_id", "predicted_score"]].head(10).to_string(index=False))

# We'll regenerate the full reasoning in Rank.py — for now save bare CSV
top_df[["candidate_id", "rank", "predicted_score"]].to_csv(
    SUBMISSION_CSV.replace(".csv", "_preview.csv"), index=False
)
print(f"\nPreview saved -> {SUBMISSION_CSV.replace('.csv','_preview.csv')}")
print("\nDone! Model is ready. Run Rank.py to produce the full team_musketeers.csv with reasoning.")
