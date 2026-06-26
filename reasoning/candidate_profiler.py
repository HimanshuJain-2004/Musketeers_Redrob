import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Archetype feature families
# Maps human-readable archetype names to groups of numeric feature column
# prefixes/names that represent that family.  We use presence-testing so that
# columns that don't exist in a given parquet are silently skipped.
# ---------------------------------------------------------------------------
ARCHETYPE_FEATURE_FAMILIES = {
    "Retrieval Architect": [
        "retrieval_count", "ranking_count", "retrieval_score_old",
        "search_count", "reranking_count", "semantic_search_count",
        "retrieval_depth_score", "ranking_experience_score",
    ],
    "Production ML Builder": [
        "production_ml_score", "production_count", "prod_ml_count",
        "mlops_score", "deployment_count", "pipeline_count",
        "serving_experience_score", "infra_ml_count",
    ],
    "Technical Leader": [
        "ownership_score", "technical_leadership_score",
        "stability_leadership_score", "management_count",
        "architecture_score", "design_count", "leadership_score",
    ],
    "Research Specialist": [
        "semantic_score", "education_score", "research_count",
        "publications_count", "academic_score", "nlp_count",
        "paper_count",
    ],
    "Experienced Generalist": [
        "career_history_length", "mobility_score", "breadth_score",
        "domain_diversity_score", "total_companies",
        "total_years_experience",
    ],
    "Hidden Gem": [
        "suspicion_score", "underrated_score", "hidden_gem_score",
        "low_visibility_high_signal",
    ],
}

# Availability-adjacent columns
AVAILABILITY_COLS = [
    "availability_score", "days_since_active", "recency_score",
    "activity_score", "platform_activity",
]


class CandidateProfiler:
    """
    Profiles a candidate into:
      - Primary Archetype + Secondary Trait
      - Confidence Tier (HIGH / MODERATE / EXPLORATORY)
      - Availability Signal (HIGH / MODERATE / LOW)

    All logic is based purely on numeric feature columns; internal column names
    are NEVER surfaced in any output string.
    """

    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df

        # ------------------------------------------------------------------
        # Pre-compute global Z-scores (candidate × feature matrix)
        # ddof=0 → population std; replace 0-std cols with 1 to avoid NaN.
        # ------------------------------------------------------------------
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        mu = features_df[numeric_cols].mean()
        sigma = features_df[numeric_cols].std(ddof=0).replace(0.0, 1.0)
        self.z_scores_df = (features_df[numeric_cols] - mu) / sigma
        self.numeric_cols = numeric_cols

        # ------------------------------------------------------------------
        # Pre-compute global pool statistics for relative comparisons
        # ------------------------------------------------------------------
        self._pool_percentiles = {
            "P75": float(np.nanpercentile(features_df[numeric_cols].values, 75)),
            "P50": float(np.nanpercentile(features_df[numeric_cols].values, 50)),
            "P25": float(np.nanpercentile(features_df[numeric_cols].values, 25)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _family_z_score(self, candidate_id: str, family_cols: list) -> float:
        """
        Compute the mean Z-score across a feature family for one candidate.
        Using Z-scores (not raw values) normalizes each feature to the same
        scale, so dominant archetypes reflect relative standing — not just
        which family happens to have the largest raw values.
        Returns 0.0 if no columns in the family are present.
        """
        if candidate_id not in self.z_scores_df.index:
            return 0.0
        z_row = self.z_scores_df.loc[candidate_id]
        scores = []
        for col in family_cols:
            if col in z_row.index and not np.isnan(z_row[col]):
                scores.append(float(z_row[col]))
        return float(np.mean(scores)) if scores else 0.0

    def _top3_z(self, candidate_id: str) -> list:
        if candidate_id not in self.z_scores_df.index:
            return [0.0, 0.0, 0.0]
        z_vals = self.z_scores_df.loc[candidate_id].values
        valid = [v for v in z_vals if not np.isnan(v)]
        sorted_z = sorted(valid, reverse=True)
        # Pad to at least 3
        while len(sorted_z) < 3:
            sorted_z.append(0.0)
        return sorted_z[:3]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_archetypes(self, candidate_id: str) -> tuple:
        """Return (primary_archetype, secondary_trait) based on Z-score means per family."""
        if candidate_id not in self.z_scores_df.index:
            return "Experienced Generalist", "Technical Contributor"

        scores = {
            arch: self._family_z_score(candidate_id, cols)
            for arch, cols in ARCHETYPE_FEATURE_FAMILIES.items()
        }

        sorted_archs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_archs[0][0]
        secondary = next(
            (name for name, _ in sorted_archs[1:] if name != primary),
            "Experienced Generalist",
        )
        return primary, secondary

    def get_confidence_tier(self, candidate_id: str) -> str:
        """
        Compute confidence from top-3 Z-scores using the formula:
            0.5 * z1 + 0.3 * z2 + 0.2 * z3
        Returns 'HIGH', 'MODERATE', or 'EXPLORATORY'.
        """
        top3 = self._top3_z(candidate_id)
        score = 0.5 * top3[0] + 0.3 * top3[1] + 0.2 * top3[2]

        if score > 1.5:
            return "HIGH"
        elif score > 0.5:
            return "MODERATE"
        else:
            return "EXPLORATORY"

    def get_availability_signal(self, candidate_id: str) -> str:
        """Returns 'HIGH', 'MODERATE', or 'LOW' based on availability features."""
        if candidate_id not in self.features_df.index:
            return "MODERATE"

        row = self.features_df.loc[candidate_id]

        # Aggregate available signals
        avail_vals = []
        for col in AVAILABILITY_COLS:
            if col in row.index and pd.notna(row[col]):
                avail_vals.append(float(row[col]))

        if not avail_vals:
            return "MODERATE"

        avg_avail = np.mean(avail_vals)
        if avg_avail > 0.65:
            return "HIGH"
        elif avg_avail > 0.35:
            return "MODERATE"
        else:
            return "LOW"

    def get_pool_percentile_rank(self, candidate_id: str) -> float:
        """
        Returns the candidate's percentile rank within the evaluation pool
        based on their mean Z-score. Returns a value in [0, 100].
        """
        if candidate_id not in self.z_scores_df.index:
            return 50.0

        candidate_mean_z = float(self.z_scores_df.loc[candidate_id].mean())
        all_mean_z = self.z_scores_df.mean(axis=1).values
        pct = float(np.mean(all_mean_z <= candidate_mean_z)) * 100.0
        return round(pct, 1)
