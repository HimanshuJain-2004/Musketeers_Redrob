import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ARTIFACTS_DIR = os.path.join(WORKSPACE_ROOT, 'JAIN', 'artifacts')
PIYUSH_ARTIFACTS = os.path.join(WORKSPACE_ROOT, 'PIYUSH', 'artifacts')

def train_baseline():
    labels_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'gold_labels.parquet'))
    features_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, 'merged_features.parquet'))
    
    # Merge
    df = pd.merge(labels_df[['candidate_id', 'fit_score']], features_df, on='candidate_id', how='inner')
    
    print(f"Training on {len(df)} samples.")
    
    y = df['fit_score']
    X = df.drop(columns=['candidate_id', 'fit_score'])
    
    # Filter only numerical columns
    X = X.select_dtypes(include=['number'])
    # Fill NAs
    X = X.fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # LGBM Regressor for baseline
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    report = f"# Baseline LightGBM Report\n\n"
    report += f"- **Training Samples**: {len(X_train)}\n"
    report += f"- **Test Samples**: {len(X_test)}\n"
    report += f"- **Test RMSE**: {rmse:.2f}\n\n"
    report += "## Top 10 Feature Importances\n"
    report += "| Feature | Importance |\n|---|---|\n"
    for _, row in importance.iterrows():
        report += f"| {row['feature']} | {row['importance']} |\n"
        
    out_path = os.path.join(ARTIFACTS_DIR, 'baseline_model_report.md')
    with open(out_path, 'w') as f:
        f.write(report)
        
    print(f"Baseline report saved to {out_path}")

if __name__ == '__main__':
    train_baseline()
