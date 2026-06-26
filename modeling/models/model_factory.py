import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np

class ModelFactory:
    @staticmethod
    def get_models():
        models = {
            "LightGBMRegressor": lgb.LGBMRegressor(random_state=42, verbose=-1),
            "XGBRegressor": xgb.XGBRegressor(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        return models
