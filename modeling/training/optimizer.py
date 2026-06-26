from sklearn.model_selection import RandomizedSearchCV

class Optimizer:
    def __init__(self):
        self.search_spaces = {
            "LightGBMRegressor": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [-1, 5, 10],
                "subsample": [0.8, 0.9, 1.0]
            },
            "XGBRegressor": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            },
            "RandomForestRegressor": {
                "n_estimators": [200, 500, 800],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "ExtraTreesRegressor": {
                "n_estimators": [200, 500, 800],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 4]
            }
        }

    def tune_model(self, model_name, model, X_train, y_train):
        if model_name not in self.search_spaces:
            print(f"No search space defined for {model_name}. Skipping tuning.")
            return model
            
        print(f"Tuning {model_name}...")
        param_dist = self.search_spaces[model_name]
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=15,
            cv=3,
            scoring="r2",
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        print(f"Best params for {model_name}: {search.best_params_}")
        print(f"Best cross-validated R2: {search.best_score_:.4f}")
        
        return search.best_estimator_
