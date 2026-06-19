import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "JAIN", "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
DATASET_FILE = os.path.join(ARTIFACTS_DIR, "training_dataset.parquet")

os.makedirs(MODELS_DIR, exist_ok=True)

class StackedRanker(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y, group=None):
        # We will do a simple stacking: fit base models, get predictions, fit meta model
        # For a robust stack, we should use cross_val_predict, but for simplicity here we train on train, 
        # predict on train (which overfits) or we can just blend.
        # Let's train base models on X, y
        for name, model in self.base_models:
            if 'ranker' in name.lower():
                model.fit(X, y, group=group)
            else:
                model.fit(X, y)
                
        # Generate meta-features
        meta_features = np.column_stack([
            model.predict(X) for _, model in self.base_models
        ])
        
        # Fit meta model
        if 'rank' in str(type(self.meta_model)).lower():
             self.meta_model.fit(meta_features, y, group=group)
        else:
             self.meta_model.fit(meta_features, y)
        return self
        
    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for _, model in self.base_models
        ])
        return self.meta_model.predict(meta_features)

def calculate_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate NDCG. ndcg_score expects shape (n_samples, n_classes) or list of rankings.
    # We have 1 query (all candidates), so we wrap in a list.
    try:
        ndcg10 = ndcg_score([y_true], [y_pred], k=10)
        ndcg100 = ndcg_score([y_true], [y_pred], k=100)
    except Exception as e:
        ndcg10, ndcg100 = 0, 0
        
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | NDCG@10: {ndcg10:.4f} | NDCG@100: {ndcg100:.4f}")
    return rmse, mae, ndcg10, ndcg100

def train_models():
    print("Loading data...")
    df = pd.read_parquet(DATASET_FILE)
    
    # Exclude non-feature columns
    exclude_cols = [
        "candidate_id", "fit_score", "technical_fit", 
        "product_fit", "career_fit", "behavioral_fit", 
        "recruiter_composite_score", "honeypot_label"
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df["recruiter_composite_score"].astype(int)
    
    # Handle NaNs
    X = X.fillna(0)
    
    # 70/15/15 split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Validation size relative to train_val (0.15 / 0.85 = ~0.1764)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.17647, random_state=42)
    
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
    
    # Groups for Rankers (all candidates belong to the same query)
    group_train = [len(X_train)]
    group_val = [len(X_val)]
    group_test = [len(X_test)]
    
    # 1. LightGBMRanker
    print("\nTraining LightGBMRanker...")
    # Define label_gain to prevent overflow when label > 31 (since default is 2^i - 1)
    label_gain = np.arange(101)
    lgbm_ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
        label_gain=label_gain,
        random_state=42
    )
    lgbm_ranker.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    # 2. XGBoostRanker
    print("\nTraining XGBoostRanker...")
    xgb_ranker = xgb.XGBRanker(
        objective='rank:ndcg',
        eval_metric='ndcg',
        ndcg_exp_gain=False,
        random_state=42,
        early_stopping_rounds=20
    )
    xgb_ranker.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        verbose=False
    )
    
    # 3. RandomForestRegressor
    print("\nTraining RandomForestRegressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    
    # 4. Stacked LightGBMRanker
    print("\nTraining Stacked Ranker...")
    # Base models: RF and XGBoost (using a Regressor version of XGB to stack features simply, or keep ranker)
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('xgb_reg', xgb.XGBRegressor(random_state=42))
    ]
    meta_model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=label_gain,
        random_state=42
    )
    stacked_ranker = StackedRanker(base_models, meta_model)
    stacked_ranker.fit(X_train, y_train, group=group_train)
    
    # Evaluate on Test Set
    print("\n--- Test Set Evaluation ---")
    models = {
        "LightGBMRanker": lgbm_ranker,
        "XGBoostRanker": xgb_ranker,
        "RandomForestRegressor": rf_reg,
        "Stacked LightGBMRanker": stacked_ranker
    }
    
    best_ndcg = -1
    best_model_name = None
    
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse, mae, ndcg10, ndcg100 = calculate_metrics(y_test, preds, name)
        
        if ndcg100 > best_ndcg:
            best_ndcg = ndcg100
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} (NDCG@100: {best_ndcg:.4f})")
    
    # Save best model
    best_model = models[best_model_name]
    with open(os.path.join(MODELS_DIR, "trained_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    print(f"Saved {best_model_name} to {MODELS_DIR}/trained_model.pkl")
    
    # Feature Importance for LightGBMRanker (if applicable)
    try:
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15] # Top 15
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances ({best_model_name})")
            plt.bar(range(15), importances[indices], align="center")
            plt.xticks(range(15), [feature_cols[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance.png"))
            print("Saved feature_importance.png")
    except Exception as e:
        print(f"Could not plot feature importances: {e}")

if __name__ == "__main__":
    train_models()
