# Feature Family Summary

This document summarizes the 108 engineered features (109 columns including `candidate_id`) constructed for the machine learning ranking model. The features are modularized into 8 distinct families, representing a complete translation of recruiter heuristics into numerical signals.

## Family 01: JD Alignment & General Profile
*   **Total Features**: ~27
*   **Focus**: Direct alignment with JD keywords (Product company, consulting penalty, title match) and core redrob signals (availability, GitHub activity).
*   **Key Features**: `years_exp`, `skill_coverage`, `title_score`, `product_company_score`, `market_demand_score` (See Audit Note), `recruitability_score`.

## Family 02: Career Semantic
*   **Total Features**: 2
*   **Focus**: Transformer-based text embeddings of candidate profiles and career histories against the Job Description.
*   **Key Features**: `semantic_score`, `career_semantic_score`.

## Family 03: Production ML
*   **Total Features**: 12
*   **Focus**: Extracting deep engineering experience beyond generic "ML" tags. Focuses on productionizing, serving, monitoring, and MLOps.
*   **Key Features**: `prod_ml_count`, `mlops_count`, `production_ml_score`, `production_depth_score`, `wrapper_to_production_ratio`.

## Family 04: Ownership & Leadership
*   **Total Features**: 14
*   **Focus**: Differentiating between junior contributors and principal architects who owned systems and mentored engineers.
*   **Key Features**: `ownership_score`, `leadership_depth_score`, `architecture_count`, `senior_title_flag`.

## Family 05: Career Progression
*   **Total Features**: 17
*   **Focus**: Career stability, job-hopping detection, and promotion velocity measured chronologically.
*   **Key Features**: `total_experience_years`, `avg_tenure_months`, `job_hopping_score`, `promotion_velocity`, `career_growth_score`, `career_stability_score`.

## Family 06: Company Background
*   **Total Features**: 11
*   **Focus**: Measuring exposure to high-scale big tech, startup, and product environments vs IT consulting.
*   **Key Features**: `bigtech_ratio`, `startup_ratio`, `consulting_ratio`, `company_type_switches`, `company_diversity_score`.

## Family 07: Suspicion
*   **Total Features**: 8
*   **Focus**: Detecting resume honeypots, skill inflation, and timeline inconsistencies safely without destructive filtering.
*   **Key Features**: `skill_density`, `role_density`, `experience_gap_years`, `suspicion_score`.

## Family 08: Interactions
*   **Total Features**: 6
*   **Focus**: Modeling complex multivariate recruiter logic (e.g., "Highly relevant AND career growth").
*   **Key Features**: `technical_leadership_score`, `production_retrieval_score`, `senior_production_score`, `seniority_relevance_score`, `stability_leadership_score`.
