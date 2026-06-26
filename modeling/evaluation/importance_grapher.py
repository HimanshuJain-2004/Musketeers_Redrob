import matplotlib.pyplot as plt
import os
import numpy as np

class ImportanceGrapher:
    @staticmethod
    def plot_top_3_models(top_3_models, feature_names, artifacts_dir):
        os.makedirs(artifacts_dir, exist_ok=True)
        
        for name, model in top_3_models:
            importances = None
            
            # Extract importances depending on model type
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "meta_model") and hasattr(model.meta_model, "feature_importances_"):
                # Stacked model: we don't have direct base feature importance easily, 
                # but we can try to extract from the primary base model if needed, 
                # or just skip. We'll skip complex stacked extraction for now.
                print(f"Skipping graph for stacked ensemble {name}")
                continue
                
            if importances is not None:
                indices = np.argsort(importances)[::-1][:15] # Top 15
                
                plt.figure(figsize=(10, 6))
                plt.title(f"Feature Importances ({name})")
                plt.bar(range(15), importances[indices], align="center")
                plt.xticks(range(15), [feature_names[i] for i in indices], rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(artifacts_dir, f"feature_importance_{name}.png"))
                plt.close()
                print(f"Saved feature_importance_{name}.png")
