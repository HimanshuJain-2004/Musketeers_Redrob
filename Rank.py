# -*- coding: utf-8 -*-
"""
Rank.py  -  End-to-End Ranking Engine (Final)
==============================================
Pipeline:
  1. Score all candidates with XGBoost try_2 Regressor (70 selected features)
  2. Take top candidates (default top-N = 100)
  3. Validate each top candidate across all 70 features vs. full pool median
  4. Auto-swap genuinely weak candidates (< 35% features above median)
     with stronger replacements from the next-best pool
  5. Generate professional reasoning via advanced reasoning_generator.py
  6. Save team_musketeers.csv

Usage:
  python Rank.py                         # ranks entire candidates.jsonl
  python Rank.py --input_file FILE.jsonl # ranks a specific subset
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
JAIN_ARTIFACTS   = os.path.join(BASE_DIR, "modeling", "artifacts")
FEAT_ARTIFACTS   = os.path.join(BASE_DIR, "feature_engineering", "artifacts")
MODEL_JSON       = os.path.join(JAIN_ARTIFACTS, "xgboost_ranking_model.json")
SEL_FEAT_PATH    = os.path.join(JAIN_ARTIFACTS, "model_feature_schema.json")
FEATURES_PARQUET = os.path.join(FEAT_ARTIFACTS, "merged_features.parquet")
CANDIDATES_JSONL = os.path.join(BASE_DIR, "candidates.jsonl")
OUTPUT_CSV       = os.path.join(BASE_DIR, "team_musketeers.csv")

sys.path.insert(0, os.path.join(BASE_DIR, "reasoning"))

# ---------------------------------------------------------------------------
# Pool context classifier (for adaptive reasoning)
# ---------------------------------------------------------------------------
def _compute_pool_context(percentiles: dict, scores) -> dict:
    p20 = percentiles["P20"]
    p50 = percentiles["P50"]
    p80 = percentiles["P80"]
    p95 = percentiles["P95"]
    p99 = percentiles["P99"]
    body_spread  = max((p80 - p20), 1e-6)
    top10_scores = np.sort(scores)[-10:] if len(scores) >= 10 else scores
    top10_spread = float(top10_scores[-1]) - float(top10_scores[0])
    if top10_spread < 0.05 * body_spread:
        quality = "COMPRESSED"
    elif (p99 - p50) < 0.5 * body_spread:
        quality = "WEAK"
    elif (p99 - p50) > 3 * max((p95 - p50), 1e-6):
        quality = "OUTLIER"
    else:
        quality = "NORMAL"
    return {"quality": quality, "p80": float(p80), "p95": float(p95),
            "p99": float(p99), "p50": float(p50)}


# ---------------------------------------------------------------------------
# Reasoning fallback (z-score based, used if advanced system fails)
# ---------------------------------------------------------------------------
_FEAT_MAP = {
    "product_years": "Product Company Tenure",
    "production_depth_score": "Production Depth",
    "avg_assessment_score": "Behavioral Assessment Score",
    "technical_leadership_score": "Technical Leadership",
    "career_growth_score": "Career Growth",
    "ownership_score": "Ownership and Accountability",
    "stability_leadership_score": "Stability and Leadership",
    "senior_production_score": "Senior Production Experience",
    "production_retrieval_score": "Production Retrieval",
    "avg_tenure_months": "Average Tenure",
    "promotion_velocity": "Promotion Velocity",
    "startup_years": "Startup Exposure",
}

def _zscore_reasoning(cid, feat_row, mean_vals, std_vals, sel, raw_data):
    z    = (feat_row[sel] - mean_vals[sel]) / std_vals[sel]
    top3 = z.sort_values(ascending=False).head(3).index.tolist()
    feats = [_FEAT_MAP.get(f, f.replace("_", " ").title()) for f in top3]
    c     = raw_data.get(cid, {})
    p     = c.get("profile", {})
    years = p.get("years_of_experience", "several")
    title = p.get("current_title", "Professional")
    loc   = p.get("location", "their location")
    skills = ", ".join(s["name"] for s in c.get("skills", [])[:3]) or "relevant tools"
    return (f"Standout applicant with {years} years of experience as a {title}, "
            f"strong technical match ({skills}) in {loc}. "
            f"Ranked highly due to outstanding {feats[0]}, {feats[1]}, and {feats[2]}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(input_file=None, top_n=100):
    print("=" * 65)
    print("  Ranking Engine  -  XGBoost try_2  +  Validated Top-100")
    print("=" * 65)

    # ── 1. Load model & selected features ──────────────────────────────
    print("\n[1/6] Loading model and features...")
    with open(SEL_FEAT_PATH) as f:
        sel = json.load(f)["selected_features"]

    model = xgb.XGBRegressor()
    model.load_model(MODEL_JSON)

    # ── 2. Load candidate features ──────────────────────────────────────
    print("[2/6] Loading feature matrix...")
    full_feat_df = pd.read_parquet(FEATURES_PARQUET)

    candidate_raw_data = {}
    if input_file:
        print(f"      Parsing input file: {input_file}")
        target_ids = set()
        with open(input_file, "r", encoding="utf-8") as fh:
            for line in fh:
                if '"candidate_id"' in line:
                    cand = json.loads(line)
                    cid  = cand["candidate_id"]
                    target_ids.add(cid)           # set() auto-deduplicates
                    candidate_raw_data[cid] = cand
        print(f"      Found {len(target_ids):,} unique candidates in input file.")
        score_df = full_feat_df[full_feat_df["candidate_id"].isin(target_ids)].copy()
        # Warn if some input candidates aren't in the feature matrix
        missing_feats = target_ids - set(score_df["candidate_id"])
        if missing_feats:
            print(f"      WARNING: {len(missing_feats)} candidate(s) from input not found "
                  f"in feature matrix — they will be skipped. "
                  f"(Run feature engineering first to include new candidates.)")
    else:
        score_df = full_feat_df.copy()

    if len(score_df) == 0:
        print("No candidates to process. "
              "If using --input_file, ensure candidates have been feature-engineered.")
        return

    # Cap top_n to however many candidates are actually available
    top_n = min(top_n, len(score_df))
    print(f"      Scoring {len(score_df):,} candidates (will rank top {top_n})...")


    # ── 3. XGBoost scoring ──────────────────────────────────────────────
    print("[3/6] Running inference...")
    X = score_df[sel].fillna(0)
    score_df = score_df.copy()
    score_df["predicted_score"] = np.clip(model.predict(X), 0.0, 1.0)
    score_df = score_df.sort_values(
        by=["predicted_score", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)

    # Pool-wide feature medians (computed on the entire scoring pool)
    pool_medians = score_df[sel].median()

    # ── 4. Validation + auto-swap ────────────────────────────────────────
    print("[4/6] Validating top candidates (70-feature check)...")

    def pct_above_median(row):
        vals = pd.to_numeric(row[sel], errors="coerce").fillna(0)
        return (vals > pool_medians).sum() / len(sel)

    # Pull enough candidates from the ranked list to always find 100 clean ones
    OVERSAMPLE = min(len(score_df), max(top_n * 3, top_n + 200))
    candidates_pool = score_df.head(OVERSAMPLE).copy()
    candidates_pool["pct_above"] = candidates_pool.apply(pct_above_median, axis=1)

    WEAK_THRESHOLD = 0.35  # < 35% features above median -> not strong enough

    # Greedily fill top_n with valid candidates first, then fill gaps with best weak ones
    strong = candidates_pool[candidates_pool["pct_above"] >= WEAK_THRESHOLD]
    weak   = candidates_pool[candidates_pool["pct_above"] <  WEAK_THRESHOLD]

    if len(strong) >= top_n:
        final_pool = strong.head(top_n).copy()
        swapped = 0
    else:
        # Not enough strong candidates — fill remainder with best-scoring weak ones
        needed = top_n - len(strong)
        fill   = weak.head(needed)
        final_pool = pd.concat([strong, fill]).sort_values(
            "predicted_score", ascending=False
        ).head(top_n).copy()
        swapped = needed
        print(f"      Warning: Only {len(strong)} strong candidates found. "
              f"Filled {needed} slots with best available.")

    n_weak_removed = len(candidates_pool.head(top_n)) - len(
        candidates_pool.head(top_n)[candidates_pool.head(top_n)["pct_above"] >= WEAK_THRESHOLD]
    )
    n_swapped = max(0, n_weak_removed - max(0, top_n - len(strong)))
    print(f"      {len(strong)} strong candidates found. "
          f"{n_weak_removed} weak candidates replaced with next-best.")

    final_pool = final_pool.reset_index(drop=True)
    final_pool["rank"] = range(1, len(final_pool) + 1)

    print(f"      Top score : {final_pool['predicted_score'].iloc[0]:.4f}")
    print(f"      100th score: {final_pool['predicted_score'].iloc[-1]:.4f}")

    # ── 5. Load raw JSONL for reasoning ─────────────────────────────────
    print("[5/6] Loading raw candidate data for reasoning...")
    top_ids     = set(final_pool["candidate_id"])
    missing_ids = top_ids - set(candidate_raw_data.keys())
    if missing_ids:
        with open(CANDIDATES_JSONL, "r", encoding="utf-8") as fh:
            for line in fh:
                if '"candidate_id"' in line:
                    cand = json.loads(line)
                    cid  = cand["candidate_id"]
                    if cid in missing_ids:
                        candidate_raw_data[cid] = cand
                        missing_ids.discard(cid)
                        if not missing_ids:
                            break

    # ── 6. Reasoning ─────────────────────────────────────────────────────
    print("[6/6] Generating reasoning...")
    reasoning_list = []

    try:
        from reasoning.data_loader        import DataLoader
        from reasoning.candidate_profiler import CandidateProfiler
        from reasoning.reasoning_generator import ReasoningGenerator

        scores_arr  = final_pool["predicted_score"].values
        percentiles = {
            "P99": float(np.percentile(scores_arr, 99)),
            "P95": float(np.percentile(scores_arr, 95)),
            "P80": float(np.percentile(scores_arr, 80)),
            "P50": float(np.percentile(scores_arr, 50)),
            "P20": float(np.percentile(scores_arr, 20)),
        }
        pool_context = _compute_pool_context(percentiles, scores_arr)

        tmp_path = os.path.join(JAIN_ARTIFACTS, "_tmp_ranked.csv")
        final_pool[["candidate_id", "rank", "predicted_score"]].rename(
            columns={"predicted_score": "score"}
        ).to_csv(tmp_path, index=False)

        loader   = DataLoader(tmp_path, FEATURES_PARQUET, CANDIDATES_JSONL)
        profiler = CandidateProfiler(loader.features_df)
        gen      = ReasoningGenerator(profiler, loader, percentiles, pool_context)

        for _, row in tqdm(final_pool.iterrows(), total=len(final_pool), ncols=65):
            r = gen.generate(
                row["candidate_id"],
                float(row["predicted_score"]),
                rank=int(row["rank"])
            )
            reasoning_list.append(r)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    except Exception as e:
        print(f"      Advanced reasoning unavailable ({e}). Using z-score fallback.")
        mean_vals = score_df[sel].mean()
        std_vals  = score_df[sel].std().fillna(1).replace(0, 1)
        feat_idx  = score_df.set_index("candidate_id")

        for _, row in final_pool.iterrows():
            cid = row["candidate_id"]
            if cid in feat_idx.index:
                feat_row = feat_idx.loc[cid]
            else:
                feat_row = pd.Series(0, index=sel)
            r = _zscore_reasoning(cid, feat_row, mean_vals, std_vals, sel, candidate_raw_data)
            reasoning_list.append(r)

    # ── 7. Save ───────────────────────────────────────────────────────────
    final_pool["reasoning"] = reasoning_list
    final_pool["score"]     = final_pool["predicted_score"].apply(lambda x: f"{x:.4f}")

    final_pool[["candidate_id", "rank", "score", "reasoning"]].to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8"
    )

    print(f"\nDone! Saved {len(final_pool)} candidates -> {OUTPUT_CSV}")
    print("\nTop 5:")
    for _, r in final_pool.head(5).iterrows():
        print(f"  [{int(r['rank'])}] {r['candidate_id']}  score={r['score']}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-End Ranking Engine with built-in validation"
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Path to input .jsonl with candidates to rank (default: all candidates)"
    )
    parser.add_argument(
        "--top_n", type=int, default=100,
        help="Number of candidates to output (default: 100)"
    )
    args = parser.parse_args()
    main(input_file=args.input_file, top_n=args.top_n)
