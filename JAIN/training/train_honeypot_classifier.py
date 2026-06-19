import os
import pickle
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "JAIN", "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
DATASET_FILE = os.path.join(ARTIFACTS_DIR, "training_dataset.parquet")

os.makedirs(MODELS_DIR, exist_ok=True)

def train_honeypot():
    print("Loading data for honeypot classification...")
    df = pd.read_parquet(DATASET_FILE)
    
    exclude_cols = [
        "candidate_id", "fit_score", "technical_fit", 
        "product_fit", "career_fit", "behavioral_fit", 
        "recruiter_composite_score", "honeypot_label"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df["honeypot_label"].astype(int)
    
    print("Training ExtraTreesClassifier...")
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X, y)
    
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]
    
    print("\n--- Training Performance ---")
    print(f"ROC-AUC: {roc_auc_score(y, probs):.4f}")
    print(classification_report(y, preds))
    
    # Save Model
    out_path = os.path.join(MODELS_DIR, "best_honeypot_classifier.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved best_honeypot_classifier.pkl to {MODELS_DIR}")

if __name__ == "__main__":
    train_honeypot()
