import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ARTIFACTS_DIR = os.path.join(WORKSPACE_ROOT, 'JAIN', 'artifacts')
PIYUSH_ARTIFACTS = os.path.join(WORKSPACE_ROOT, 'PIYUSH', 'artifacts')

def bucket_score(score):
    if score < 15:
        return 0
    elif score < 30:
        return 1
    elif score < 50:
        return 2
    elif score < 75:
        return 3
    else:
        return 4

def train_and_predict():
    labels_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'gold_labels.parquet'))
    features_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, 'merged_features.parquet'))
    
    # Merge
    df = pd.merge(labels_df[['candidate_id', 'fit_score']], features_df, on='candidate_id', how='inner')
    
    print(f"Training on {len(df)} labeled candidates.")
    
    y_raw = df['fit_score']
    y_relevance = y_raw.apply(bucket_score)
    X = df.drop(columns=['candidate_id', 'fit_score'])
    
    # Filter only numerical columns
    X = X.select_dtypes(include=['number'])
    # Fill NAs
    X = X.fillna(0)
    
    # We will use pointwise regression (LGBMRegressor) which often performs better
    # or equal to LGBMRanker when there is only a single query (pointwise vs listwise).
    # We train on the continuous fit_score to retain maximum signal.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)
    
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds_test = model.predict(X_test)
    
    # Calculate NDCG
    # NDCG needs 2D arrays: [n_queries, n_items]
    # Since we have 1 query (the job description), it's 1 row.
    try:
        test_ndcg = ndcg_score([y_test.values], [preds_test])
    except Exception as e:
        test_ndcg = float('nan')
        
    print(f"Test NDCG: {test_ndcg:.4f}")
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 features:")
    print(importance.head(10))
    
    # Predict on ALL 100k candidates
    print(f"Predicting on all {len(features_df)} candidates in the dataset...")
    X_all = features_df.drop(columns=['candidate_id'], errors='ignore')
    X_all = X_all.select_dtypes(include=['number']).fillna(0)
    # Ensure column order matches
    X_all = X_all[X.columns]
    
    features_df['predicted_fit_score'] = model.predict(X_all)
    
    # Read the original raw candidates to get name/title for the final output
    try:
        raw_df = pd.read_json(os.path.join(WORKSPACE_ROOT, 'candidates.jsonl'), lines=True)
        # Extract fields from profile dict
        raw_df['name'] = raw_df['profile'].apply(lambda x: x.get('anonymized_name', ''))
        raw_df['headline'] = raw_df['profile'].apply(lambda x: x.get('headline', ''))
        raw_df['current_role'] = raw_df['profile'].apply(lambda x: x.get('current_title', ''))
        
        raw_summary = raw_df[['candidate_id', 'name', 'headline', 'current_role']]
        features_df = pd.merge(features_df, raw_summary, on='candidate_id', how='left')
    except Exception as e:
        print(f"Could not load raw candidates for names: {e}")
        features_df['name'] = ''
        features_df['headline'] = ''
        features_df['current_role'] = ''
    
    # Sort by predicted fit score
    final_ranked = features_df.sort_values('predicted_fit_score', ascending=False)
    
    top_1000_path = os.path.join(ARTIFACTS_DIR, 'top_1000_ranked_candidates.csv')
    cols_to_export = ['candidate_id', 'predicted_fit_score', 'name', 'headline', 'current_role']
    # Add top 5 features
    cols_to_export += importance.head(5)['feature'].tolist()
    
    final_ranked.head(1000)[cols_to_export].to_csv(top_1000_path, index=False)
    print(f"Exported Top 1000 candidates to {top_1000_path}")
    
    # Save a model report
    report = f"# Ranking Model Report\n\n"
    report += f"- **Training Samples**: {len(X_train)}\n"
    report += f"- **Test Samples**: {len(X_test)}\n"
    report += f"- **Test NDCG**: {test_ndcg:.4f}\n\n"
    report += "## Feature Importances\n"
    report += "| Feature | Importance |\n|---|---|\n"
    for _, row in importance.head(15).iterrows():
        report += f"| {row['feature']} | {row['importance']} |\n"
        
    report_path = os.path.join(ARTIFACTS_DIR, 'ranking_model_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_path}")

if __name__ == '__main__':
    train_and_predict()
