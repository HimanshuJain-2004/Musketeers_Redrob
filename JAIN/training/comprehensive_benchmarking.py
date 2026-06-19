import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostRanker, Pool

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "JAIN", "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
DATASET_FILE = os.path.join(ARTIFACTS_DIR, "training_dataset.parquet")

os.makedirs(MODELS_DIR, exist_ok=True)

class CustomEnsemble:
    def __init__(self, models_dict, weights_dict):
        self.models_dict = models_dict
        self.weights_dict = weights_dict
        
    def fit(self, X, y, group_count=None, group_id=None):
        for name, model in self.models_dict.items():
            if 'ranker' in name.lower():
                if 'catboost' in name.lower():
                    train_pool = Pool(data=X, label=y, group_id=group_id)
                    model.fit(train_pool, verbose=False)
                elif 'xgb' in name.lower() or 'lgbm' in name.lower():
                    model.fit(X, y, group=group_count)
            else:
                if 'catboost' in name.lower():
                    model.fit(X, y, verbose=False)
                else:
                    model.fit(X, y)
        return self

    def predict(self, X):
        preds = np.zeros(len(X))
        for name, model in self.models_dict.items():
            preds += self.weights_dict[name] * model.predict(X)
        return preds

    @property
    def feature_importances_(self):
        importances = None
        for name, model in self.models_dict.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                imp = imp / np.sum(imp)
                if importances is None:
                    importances = self.weights_dict[name] * imp
                else:
                    importances += self.weights_dict[name] * imp
        return importances / np.sum(importances)

def calc_ranking_metrics(y_true, y_pred):
    true_order = np.argsort(y_true)[::-1]
    # Handle ties randomly or take top 100
    actual_top_100_idx = set(true_order[:100])
    
    pred_order = np.argsort(y_pred)[::-1]
    pred_top_100_idx = pred_order[:100]
    
    hits = sum([1 for idx in pred_top_100_idx if idx in actual_top_100_idx])
    p100 = hits / 100.0
    r100 = hits / 100.0
    
    ap = 0.0
    hits_so_far = 0
    for i, idx in enumerate(pred_top_100_idx):
        if idx in actual_top_100_idx:
            hits_so_far += 1
            ap += hits_so_far / (i + 1.0)
    map100 = ap / 100.0
    
    try:
        ndcg10 = ndcg_score([y_true], [y_pred], k=10)
        ndcg100 = ndcg_score([y_true], [y_pred], k=100)
    except:
        ndcg10, ndcg100 = 0.0, 0.0
        
    return ndcg10, ndcg100, map100, p100, r100

def get_models():
    models = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "LightGBMRegressor": lgb.LGBMRegressor(random_state=42, n_jobs=-1),
        "XGBoostRegressor": xgb.XGBRegressor(random_state=42, n_jobs=-1),
        "CatBoostRegressor": CatBoostRegressor(random_state=42, verbose=False, thread_count=-1),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
        "LightGBMRanker": lgb.LGBMRanker(objective="lambdarank", metric="ndcg", label_gain=np.arange(101), random_state=42, n_jobs=-1),
        "XGBoostRanker": xgb.XGBRanker(objective='rank:ndcg', eval_metric='ndcg', ndcg_exp_gain=False, random_state=42, n_jobs=-1),
        "CatBoostRanker": CatBoostRanker(loss_function='YetiRank', random_state=42, verbose=False, thread_count=-1)
    }
    
    # Custom Ensemble
    ensemble_base = {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "LGBM": lgb.LGBMRegressor(random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(random_state=42, n_jobs=-1)
    }
    ensemble_weights = {"ExtraTrees": 0.5, "LGBM": 0.25, "XGBoost": 0.25}
    models["CustomEnsemble"] = CustomEnsemble(ensemble_base, ensemble_weights)
    
    return models

def benchmark():
    print("Loading data...")
    df = pd.read_parquet(DATASET_FILE)
    
    exclude_cols = [
        "candidate_id", "fit_score", "technical_fit", 
        "product_fit", "career_fit", "behavioral_fit", 
        "recruiter_composite_score", "honeypot_label"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0).values
    y = df["recruiter_composite_score"].astype(int).values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {m: {
        'rmse': [], 'mae': [], 'r2': [], 'spearman': [],
        'ndcg10': [], 'ndcg100': [], 'map100': [], 'p100': [], 'r100': [],
        'fit_time': [], 'infer_time': []
    } for m in get_models().keys()}
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"--- Fold {fold+1}/5 ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        group_count_train = [len(X_train)]
        group_id_train = np.zeros(len(X_train), dtype=int)
        
        models = get_models()
        for name, model in models.items():
            # Fit
            start_fit = time.time()
            if 'CatBoostRanker' == name:
                train_pool = Pool(data=X_train, label=y_train, group_id=group_id_train)
                model.fit(train_pool, verbose=False)
            elif 'Ranker' in name and name != 'CustomEnsemble':
                model.fit(X_train, y_train, group=group_count_train)
            elif name == 'CustomEnsemble':
                model.fit(X_train, y_train, group_count=group_count_train, group_id=group_id_train)
            else:
                model.fit(X_train, y_train)
            fit_time = time.time() - start_fit
            
            # Predict
            start_infer = time.time()
            preds = model.predict(X_test)
            infer_time = time.time() - start_infer
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            spearman, _ = spearmanr(y_test, preds)
            if np.isnan(spearman): spearman = 0.0
            
            ndcg10, ndcg100, map100, p100, r100 = calc_ranking_metrics(y_test, preds)
            
            res = results[name]
            res['rmse'].append(rmse); res['mae'].append(mae); res['r2'].append(r2); res['spearman'].append(spearman)
            res['ndcg10'].append(ndcg10); res['ndcg100'].append(ndcg100); res['map100'].append(map100)
            res['p100'].append(p100); res['r100'].append(r100)
            res['fit_time'].append(fit_time); res['infer_time'].append(infer_time)
            
    # Aggregate results
    print("Aggregating results...")
    agg = []
    for name, r in results.items():
        agg.append({
            'Model': name,
            'NDCG@100': np.mean(r['ndcg100']), 'NDCG@100_std': np.std(r['ndcg100']),
            'Precision@100': np.mean(r['p100']), 'Precision@100_std': np.std(r['p100']),
            'Recall@100': np.mean(r['r100']), 'Recall@100_std': np.std(r['r100']),
            'R2': np.mean(r['r2']), 'R2_std': np.std(r['r2']),
            'RMSE': np.mean(r['rmse']), 'RMSE_std': np.std(r['rmse']),
            'MAE': np.mean(r['mae']), 'MAE_std': np.std(r['mae']),
            'Spearman': np.mean(r['spearman']), 'Spearman_std': np.std(r['spearman']),
            'NDCG@10': np.mean(r['ndcg10']), 'NDCG@10_std': np.std(r['ndcg10']),
            'MAP@100': np.mean(r['map100']), 'MAP@100_std': np.std(r['map100']),
            'Fit Time (s)': np.mean(r['fit_time']), 'Infer Time (s)': np.mean(r['infer_time'])
        })
        
    df_res = pd.DataFrame(agg)
    # Sort by priority: NDCG@100 > Precision@100 > Recall@100 > R2 > Lowest RMSE
    df_res = df_res.sort_values(by=['NDCG@100', 'Precision@100', 'Recall@100', 'R2', 'RMSE'], 
                                ascending=[False, False, False, False, True]).reset_index(drop=True)
                                
    # Save Leaderboard
    df_res.to_csv(os.path.join(ARTIFACTS_DIR, "benchmark_leaderboard.csv"), index=False)
    
    md_str = "# Comprehensive Model Benchmarking Leaderboard\n\n"
    md_str += "| Rank | Model | NDCG@100 | Precision@100 | R2 | RMSE | Spearman | Fit Time (s) |\n"
    md_str += "|---|---|---|---|---|---|---|---|\n"
    for i, row in df_res.iterrows():
        md_str += f"| {i+1} | **{row['Model']}** | {row['NDCG@100']:.4f} ± {row['NDCG@100_std']:.4f} | {row['Precision@100']:.4f} ± {row['Precision@100_std']:.4f} | {row['R2']:.4f} ± {row['R2_std']:.4f} | {row['RMSE']:.4f} ± {row['RMSE_std']:.4f} | {row['Spearman']:.4f} ± {row['Spearman_std']:.4f} | {row['Fit Time (s)']:.2f} |\n"
        
    with open(os.path.join(ARTIFACTS_DIR, "benchmark_leaderboard.md"), "w") as f:
        f.write(md_str)
        
    # Top 3 Models - SHAP and Full Training
    top_3_models = df_res['Model'].head(3).tolist()
    print(f"\nTop 3 Models: {top_3_models}")
    
    # Train Top 3 on Full Data for SHAP
    group_count_full = [len(X)]
    group_id_full = np.zeros(len(X), dtype=int)
    
    best_model_obj = None
    
    models = get_models()
    for rank, m_name in enumerate(top_3_models):
        print(f"Generating SHAP and feature importance for {m_name}...")
        model = models[m_name]
        
        # Fit on full
        if 'CatBoostRanker' == m_name:
            full_pool = Pool(data=X, label=y, group_id=group_id_full)
            model.fit(full_pool, verbose=False)
        elif 'Ranker' in m_name and m_name != 'CustomEnsemble':
            model.fit(X, y, group=group_count_full)
        elif m_name == 'CustomEnsemble':
            model.fit(X, y, group_count=group_count_full, group_id=group_id_full)
        else:
            model.fit(X, y)
            
        if rank == 0:
            best_model_obj = model
            # Save Best Model
            with open(os.path.join(MODELS_DIR, "best_benchmark_model.pkl"), "wb") as f:
                pickle.dump(best_model_obj, f)
                
        # Feature Importance (if supported)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances ({m_name})")
            plt.bar(range(15), importances[indices], align="center")
            plt.xticks(range(15), [feature_cols[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(ARTIFACTS_DIR, f"feature_importance_{m_name}.png"))
            plt.close()
            
        # SHAP Analysis
        if m_name != 'CustomEnsemble': # Ensemble SHAP is complex, skip for now
            try:
                # TreeExplainer is fast for tree models
                explainer = shap.TreeExplainer(model)
                # Sample for faster SHAP
                sample_idx = np.random.choice(X.shape[0], min(500, X.shape[0]), replace=False)
                X_sample = X[sample_idx]
                shap_values = explainer.shap_values(X_sample)
                
                # If ranker, shap might return list of arrays
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] # Or 0 depending on objective
                    
                plt.figure()
                shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(ARTIFACTS_DIR, f"shap_summary_{m_name}.png"))
                plt.close()
            except Exception as e:
                print(f"SHAP failed for {m_name}: {e}")

    print("Benchmarking Complete!")

if __name__ == "__main__":
    benchmark()
