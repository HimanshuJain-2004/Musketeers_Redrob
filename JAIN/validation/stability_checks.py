import os
import pickle
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "JAIN", "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
DATASET_FILE = os.path.join(ARTIFACTS_DIR, "training_dataset.parquet")
MODEL_FILE = os.path.join(MODELS_DIR, "best_benchmark_model.pkl")

def check_feature_leakage(model, df):
    logging.info("Running Feature Leakage Check...")
    
    # Extract feature names from the model if available
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
    else:
        logging.warning("Model does not have 'feature_names_in_'. Using default dataset features.")
        exclude_cols = [
            "candidate_id", "fit_score", "technical_fit", 
            "product_fit", "career_fit", "behavioral_fit", 
            "recruiter_composite_score", "honeypot_label"
        ]
        features = [c for c in df.columns if c not in exclude_cols]
        
    forbidden_targets = {
        "fit_score", "technical_fit", "product_fit", "career_fit", 
        "behavioral_fit", "honeypot_label", "recruiter_composite_score",
        "fit_label", "candidate_id"
    }
    
    leakage = [f for f in features if f in forbidden_targets]
    
    if leakage:
        logging.error(f"FEATURE LEAKAGE DETECTED! The following target variables are in the feature set: {leakage}")
        raise ValueError("Feature leakage detected. Halting pipeline.")
        
    logging.info("PASSED: No target variables found in the model's feature set.")
    return features

def check_robustness(model, df, feature_cols):
    logging.info("Running Multi-seed Robustness & Top-100 Overlap Analysis...")
    
    X = df[feature_cols].fillna(0)
    
    # Baseline Predictions
    baseline_preds = model.predict(X)
    df['baseline_score'] = baseline_preds
    
    # Get Baseline Top 100
    baseline_top100 = set(df.nlargest(100, 'baseline_score')['candidate_id'])
    
    # Add Gaussian Noise to semantic scores
    semantic_cols = [c for c in feature_cols if 'semantic' in c.lower()]
    logging.info(f"Adding N(0, 0.05) noise to {len(semantic_cols)} semantic features...")
    
    X_noisy = X.copy()
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, size=(len(X_noisy), len(semantic_cols)))
    X_noisy[semantic_cols] += noise
    
    # Noisy Predictions
    noisy_preds = model.predict(X_noisy)
    df['noisy_score'] = noisy_preds
    
    # Get Noisy Top 100
    noisy_top100 = set(df.nlargest(100, 'noisy_score')['candidate_id'])
    
    # Overlap
    overlap = len(baseline_top100.intersection(noisy_top100))
    overlap_pct = overlap / 100.0 * 100
    
    logging.info(f"Top-100 Overlap between Baseline and Noisy: {overlap_pct:.2f}%")
    
    if overlap_pct < 80.0:
        logging.warning("Robustness check indicates potentially unstable ranking (< 80% overlap).")
    else:
        logging.info("PASSED: Ranking is robust to semantic score noise.")

def run_checks():
    logging.info("Loading model and dataset...")
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
        
    df = pd.read_parquet(DATASET_FILE)
    
    # 1. Leakage Check
    features = check_feature_leakage(model, df)
    
    # 2. Robustness Check
    check_robustness(model, df, features)
    
    logging.info("All Stability Checks Completed.")

if __name__ == "__main__":
    run_checks()
