"""
DataLoader v3.0
---------------
Loads:
  - ranked_candidates.csv   -> candidate_id, rank, score
  - merged_features.parquet -> engineered features
  - candidates.jsonl        -> nested profile, career_history, skills, redrob_signals

Skill-phrase builder maps actual skill names to human-readable domain strings
so Sentence 1 uses real candidate data, never internal feature names.
"""

import json
import re
import pandas as pd


# ---------------------------------------------------------------------------
# Skill bucket mapping — maps known skill keywords to human-readable phrases.
# Order matters: first match wins.
# ---------------------------------------------------------------------------
_SKILL_BUCKETS = [
    ("search and recommendation systems",
     ["elasticsearch", "solr", "lucene", "faiss", "ann", "vector search",
      "retrieval", "reranking", "ranking", "recommendation", "collaborative filtering",
      "matrix factorization", "information retrieval", "semantic search"]),
    ("large language models and NLP",
     ["llm", "gpt", "bert", "transformers", "language model", "nlp",
      "text classification", "named entity", "sentiment", "embeddings",
      "fine-tuning", "prompt engineering", "rag", "langchain"]),
    ("production ML and MLOps",
     ["mlops", "kubeflow", "airflow", "mlflow", "feature store", "model serving",
      "deployment", "kubernetes", "docker", "cicd", "model monitoring",
      "triton", "seldon", "bentoml", "sagemaker"]),
    ("computer vision and deep learning",
     ["computer vision", "cnn", "object detection", "image classification",
      "deep learning", "neural network", "resnet", "yolo", "opencv", "pytorch",
      "tensorflow", "image segmentation"]),
    ("data engineering and infrastructure",
     ["spark", "kafka", "hadoop", "flink", "pipeline", "etl",
      "data warehouse", "bigquery", "redshift", "databricks", "airflow",
      "dbt", "snowflake", "data lake"]),
    ("applied machine learning and analytics",
     ["machine learning", "scikit-learn", "xgboost", "lightgbm", "random forest",
      "gradient boosting", "regression", "classification", "clustering",
      "feature engineering", "a/b testing", "experimentation", "python", "sql"]),
]

_DEFAULT_SKILL_PHRASE = "applied machine learning and data systems"


def _build_skill_phrase(skills: list) -> str:
    """
    Given a list of skill dicts (from JSONL), return the best human-readable
    domain phrase. Never exposes raw feature/column names.
    """
    if not skills:
        return _DEFAULT_SKILL_PHRASE

    # Flatten skill names to lowercase for matching
    names = set()
    for s in skills:
        if isinstance(s, dict):
            n = s.get("name", "")
        elif isinstance(s, str):
            n = s
        else:
            continue
        names.add(n.lower())

    # Score each bucket by how many of its keywords appear
    bucket_scores = []
    for phrase, keywords in _SKILL_BUCKETS:
        score = sum(1 for kw in keywords if any(kw in name for name in names))
        bucket_scores.append((phrase, score))

    best_phrase, best_score = max(bucket_scores, key=lambda x: x[1])
    return best_phrase if best_score > 0 else _DEFAULT_SKILL_PHRASE


class DataLoader:

    def __init__(
        self,
        ranked_csv_path: str,
        features_parquet_path: str,
        candidates_jsonl_path: str,
    ):
        # ------------------------------------------------------------------
        # 1. Ranked scores
        # ------------------------------------------------------------------
        self.ranked_df = pd.read_csv(ranked_csv_path)
        self.ranked_df.columns = [c.lower() for c in self.ranked_df.columns]

        # ------------------------------------------------------------------
        # 2. Engineered features
        # ------------------------------------------------------------------
        self.features_df = pd.read_parquet(features_parquet_path)
        if "candidate_id" in self.features_df.columns:
            self.features_df.set_index("candidate_id", inplace=True)

        # ------------------------------------------------------------------
        # 3. Raw candidate facts from JSONL (nested structure)
        # ------------------------------------------------------------------
        self.raw_candidates: dict = {}
        try:
            with open(candidates_jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    c_id = obj.get("candidate_id")
                    if not c_id:
                        continue

                    profile = obj.get("profile", {}) or {}
                    redrob  = obj.get("redrob_signals", {}) or {}
                    career  = obj.get("career_history", []) or []
                    skills  = obj.get("skills", []) or []

                    # Build a safe, clean title
                    raw_title = profile.get("current_title", "") or ""
                    title = self._clean_title(raw_title)

                    # Years of experience
                    yoe = float(profile.get("years_of_experience", 0) or 0)

                    # Companies (list of names from career history)
                    companies = [
                        c.get("company", "") for c in career
                        if isinstance(c, dict) and c.get("company")
                    ]

                    # Availability signals from redrob
                    avail_score = float(redrob.get("profile_completeness_score", 50) or 50) / 100.0
                    days_since_active = self._days_since_active(redrob.get("last_active_date", ""))
                    open_to_work = bool(redrob.get("open_to_work_flag", False))
                    notice_days  = float(redrob.get("notice_period_days", 90) or 90)

                    # Compute composite availability score [0,1]
                    composite_avail = self._compute_avail(
                        avail_score, days_since_active, open_to_work, notice_days
                    )

                    self.raw_candidates[c_id] = {
                        "title":        title,
                        "years_exp":    yoe,
                        "skills":       skills,
                        "skill_phrase": _build_skill_phrase(skills),
                        "companies":    companies,
                        "location":     profile.get("location", "") or "",
                        "avail_score":  composite_avail,
                        "open_to_work": open_to_work,
                        "notice_days":  notice_days,
                    }

        except FileNotFoundError:
            print(f"[DataLoader] Warning: {candidates_jsonl_path} not found. Using fallbacks.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_title(raw: str) -> str:
        """Sanitise a title: strip, truncate at 60 chars, remove internal codes."""
        if not raw:
            return "Machine Learning Engineer"
        cleaned = raw.strip()
        # Remove anything in parens/brackets that looks like an internal tag
        cleaned = re.sub(r"\s*[\(\[].{1,30}[\)\]]", "", cleaned)
        # Truncate
        if len(cleaned) > 60:
            cleaned = cleaned[:57].rstrip() + "..."
        return cleaned or "Machine Learning Engineer"

    @staticmethod
    def _days_since_active(date_str: str) -> float:
        """Return approximate days since last active; defaults to 180 if unparseable."""
        if not date_str:
            return 180.0
        try:
            from datetime import datetime
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            delta = (datetime(2026, 6, 22) - dt).days
            return max(float(delta), 0.0)
        except Exception:
            return 180.0

    @staticmethod
    def _compute_avail(
        completeness: float,
        days_since: float,
        open_to_work: bool,
        notice_days: float,
    ) -> float:
        """
        Composite availability score [0, 1].
        open_to_work is the strongest signal (+0.3 flat).
        """
        score = 0.0
        # Recency of activity
        if days_since < 30:
            score += 0.4
        elif days_since < 90:
            score += 0.25
        elif days_since < 180:
            score += 0.1

        # Open-to-work flag
        if open_to_work:
            score += 0.35

        # Short notice period
        if notice_days <= 15:
            score += 0.2
        elif notice_days <= 30:
            score += 0.1
        elif notice_days >= 90:
            score -= 0.05

        # Profile completeness as mild proxy
        score += completeness * 0.1

        return min(max(score, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_candidate_facts(self, candidate_id: str) -> dict:
        """
        Returns a merged dict of raw + feature-derived facts.

        Keys guaranteed: title, years_exp, skill_phrase, companies,
                         location, avail_score, open_to_work, notice_days
        """
        facts = self.raw_candidates.get(
            candidate_id,
            {
                "title":        "Machine Learning Engineer",
                "years_exp":    5.0,
                "skills":       [],
                "skill_phrase": _DEFAULT_SKILL_PHRASE,
                "companies":    [],
                "location":     "",
                "avail_score":  0.4,
                "open_to_work": False,
                "notice_days":  90.0,
            },
        ).copy()

        # Prefer feature-derived years_exp if available and non-zero
        if candidate_id in self.features_df.index:
            row = self.features_df.loc[candidate_id]
            for col in ("years_exp", "total_experience_years", "career_history_length"):
                if col in row.index and pd.notna(row[col]) and float(row[col]) > 0:
                    facts["years_exp"] = float(row[col])
                    break

        return facts
