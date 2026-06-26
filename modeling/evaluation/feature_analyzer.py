import pandas as pd
import numpy as np
import json
import os

class FeatureAnalyzer:
    @staticmethod
    def remove_highly_correlated_features(X_train: pd.DataFrame, threshold: float = 0.95, artifacts_dir: str = "") -> list:
        print("Calculating feature correlations...")
        corr_matrix = X_train.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        selected_features = [col for col in X_train.columns if col not in to_drop]
        
        print(f"Removed {len(to_drop)} highly correlated features.")
        
        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
            with open(os.path.join(artifacts_dir, "dropped_correlated_features.json"), "w") as f:
                json.dump({"threshold": threshold, "dropped": to_drop}, f, indent=2)
                
        return selected_features
