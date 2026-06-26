# Feature Families 06 to 08: Final Enhancements

This document summarizes the final three feature families added to the training matrix. These features transform raw extraction counts into powerful recruiter-centric signals.

## Feature Family 06: Company Background
*   **Purpose**: Distinguish candidates heavily exposed to big tech or product scaling environments from those working primarily in IT services/consulting.
*   **Features**: `startup_years`, `product_years`, `consulting_years`, `bigtech_years`, and their respective `ratio` features.
*   **Company Switches**: `company_type_switches` tracks a candidate's versatility in moving across different environment paradigms.
*   **Logic**: Emphasizes company name matching (`KNOWN_BIGTECH`, `CONSULTING_TERMS`) for high reliability, using `company_size` only as a secondary fallback.

## Feature Family 07: Suspicion & Consistency
*   **Purpose**: Detect keyword stuffing, bloated timelines, and contradictory experience claims often found in "honeypot" scenarios.
*   **Features**: `skill_density`, `role_density`, `experience_gap_years`, `experience_consistency_score`, `skills_per_role`, `timeline_consistency_score`, `suspicion_score`.
*   **Logic**: Non-destructive penalization. Evaluates stability by comparing the declared `years_exp` against the mathematically reconstructed `total_experience_years`.

## Feature Family 08: Interaction Features
*   **Purpose**: Emulate complex, multivariate recruiter reasoning by combining existing independent signals.
*   **Features**:
    *   `technical_leadership_score` = `production_depth_score` * `leadership_depth_score`
    *   `production_retrieval_score` = `jd_core_score` * `production_depth_score`
    *   `senior_production_score` = `production_depth_score` * `career_growth_score`
    *   `ownership_relevance_score` = `ownership_score` * `career_semantic_score`
    *   `stability_leadership_score` = `career_stability_score` * `leadership_depth_score`
    *   `seniority_relevance_score` = `career_growth_score` * `career_semantic_score`

## Final Output
All features have been efficiently mapped across the 100,000 candidates and stored within `merged_features.parquet`, elevating the complete training matrix to **109 columns**.
