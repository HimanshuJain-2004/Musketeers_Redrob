import pandas as pd
import numpy as np
import os
import pickle
from evaluation.metric_calculator import MetricCalculator

class Trainer:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)
        
    def train_models(self, models: dict, dataset: dict):
        X_train, y_train = dataset["X_train"], dataset["y_train"]
        X_val, y_val = dataset["X_val"], dataset["y_val"]
        X_test, y_test = dataset["X_test"], dataset["y_test"]
        
        group_train = [len(X_train)]
        group_val = [len(X_val)]
        
        results = []
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                # Rankers might need groups and eval sets
                if 'rank' in str(type(model)).lower() or 'ranker' in name.lower():
                    # Handle specific ranker implementations
                    if 'XGBRanker' in name:
                        model.fit(X_train, y_train, group=group_train, eval_set=[(X_val, y_val)], eval_group=[group_val], verbose=False)
                    elif 'LGBMRanker' in name:
                        # LightGBM requires early_stopping in callbacks for latest versions, or via fit params
                        model.fit(X_train, y_train, group=group_train, eval_set=[(X_val, y_val)], eval_group=[group_val])
                    elif 'StackedRanker' in name or 'WeightedEnsemble' in name:
                        model.fit(X_train, y_train, group=group_train)
                    else:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                    
                preds = model.predict(X_test)
                metrics = MetricCalculator.calculate_metrics(y_test, preds)
                metrics["Model"] = name
                results.append(metrics)
                
                trained_models[name] = model
                
            except Exception as e:
                print(f"Failed to train {name}: {e}")
                
        results_df = pd.DataFrame(results)
        return trained_models, results_df
