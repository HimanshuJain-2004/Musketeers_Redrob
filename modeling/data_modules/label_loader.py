import json
import pandas as pd
import os

class LabelLoader:
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        
    def load_labels(self) -> pd.DataFrame:
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"Label file not found: {self.jsonl_path}")
            
        data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    cand_id = record.get("candidate_id")
                    eval_data = record.get("evaluation", {})
                    tech = eval_data.get("technical_fit", 0)
                    prod = eval_data.get("product_fit", 0)
                    behav = eval_data.get("behavioral_fit", 0)
                    career = eval_data.get("career_fit", 0)
                    
                    # A robust mathematical formula that balances the strict, technical-focused
                    # weights with the LLM's overall holistic intuition (raw fit_score).
                    # Both sides are normalized to 0-100.
                    strict_weighted = (tech * 0.50) + (prod * 0.30) + (career * 0.10) + (behav * 0.10)
                    raw_fit = eval_data.get("fit_score", 0)
                    composite_score = (0.5 * strict_weighted) + (0.5 * raw_fit)
                    record_data = {
                        "candidate_id": cand_id,
                        "score": composite_score,
                        "raw_fit_score": eval_data.get("fit_score", 0),
                        "technical_fit": tech,
                        "product_fit": prod,
                        "behavioral_fit": behav,
                        "career_fit": career,
                        "fit_label": eval_data.get("fit_label", ""),
                        "honeypot_probability": eval_data.get("honeypot_probability", 0.0),
                        "honeypot_label": eval_data.get("honeypot_label", False),
                        "confidence": eval_data.get("confidence", 0.0),
                        "reasoning": eval_data.get("reasoning", "")
                    }
                    data.append(record_data)
                    
        df = pd.DataFrame(data)
        return df
