import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DatasetBuilder:
    def __init__(self, features_path: str):
        self.features_path = features_path
        
    def build_dataset(self, labels_df: pd.DataFrame):
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
            
        features_df = pd.read_parquet(self.features_path)
        
        merged_df = pd.merge(labels_df, features_df, on="candidate_id", how="inner")
        
        feature_names = [c for c in features_df.columns if c != "candidate_id"]
        
        X = merged_df[feature_names].fillna(0)
        y = merged_df["score"]
        candidate_ids = merged_df["candidate_id"]
        
        # We need validation set as well for Rankers
        # Let's do 70/15/15
        X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
            X, y, candidate_ids, test_size=0.15, random_state=42
        )
        
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_train_val, y_train_val, ids_train_val, test_size=0.17647, random_state=42
        )
        
        return {
            "X_train": X_train, "y_train": y_train, "ids_train": ids_train,
            "X_val": X_val, "y_val": y_val, "ids_val": ids_val,
            "X_test": X_test, "y_test": y_test, "ids_test": ids_test,
            "feature_names": feature_names
        }
