# Feature Family 05: Career Progression & Stability Features

## Purpose
Capture long-term career trajectory, promotion evidence, employment stability, and consistency. This mimics recruiter reasoning regarding a candidate's loyalty, career growth velocity, and job hopping risks, providing powerful, keyword-agnostic signals.

## Features Delivered
*   **Core Stability**: `total_experience_years`, `num_companies`, `num_roles`, `avg_tenure_months`, `longest_tenure_months`, `shortest_tenure_months`
*   **Job Hopping Risk**: `job_hopping_score`, `short_tenure_count`, `very_short_tenure_count`
*   **Career Trajectory**: `promotion_count`, `promotion_velocity`, `career_growth_score`
*   **Company Loyalty**: `multi_year_company_count`, `long_term_employment_ratio`
*   **Career Consistency**: `timeline_gap_count`, `largest_gap_months`
*   **Aggregates & Quality**: `career_stability_score`, `timeline_quality_score`

## Extraction Method
1.  **Timeline Reconstruction**: Extracted start and end dates directly from the `career_history` objects. Validated missing dates (an incredible 0.00% missing rate was found across 300,000+ roles!).
2.  **Date Normalization**: Translated string dates into datetime objects to measure overlapping companies, gap durations between roles, and exact fractional years of experience.
3.  **Title Hierarchy**: Leveraged a predefined mapping of seniority levels (`intern` -> `head`) to detect chronological vertical promotions (`promotion_count`) across a candidate's timeline.

## Output
Stored independently as `career_progression_features.parquet` and cleanly merged into the final `merged_features.parquet` training matrix.
