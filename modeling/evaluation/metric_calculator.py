import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score, r2_score

class MetricCalculator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        try:
            ndcg10 = ndcg_score([y_true], [y_pred], k=10)
            ndcg100 = ndcg_score([y_true], [y_pred], k=100)
        except Exception:
            ndcg10, ndcg100 = 0.0, 0.0
            
        from scipy.stats import spearmanr
        
        try:
            spearman, _ = spearmanr(y_true, y_pred)
            if np.isnan(spearman):
                spearman = 0.0
        except Exception:
            spearman = 0.0
            
        return {
            "R2": float(r2),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "Spearman": float(spearman),
            "NDCG@10": float(ndcg10),
            "NDCG@100": float(ndcg100)
        }
