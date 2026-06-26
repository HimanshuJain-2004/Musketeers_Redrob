# Feature Dictionary

This document serves as the central data dictionary for Jain to interpret the final `merged_features.parquet` matrix. 

## Family 01: Core & JD Alignment

### `years_exp`
*   **Family**: Core Profile
*   **Description**: Total declared years of experience.
*   **Source**: Candidate Profile JSON
*   **Range**: `0 - 50`

### `skill_match_count` / `skill_coverage`
*   **Family**: JD Alignment
*   **Description**: Number of JD-required skills matched in the candidate's skill array, and the percentage coverage.
*   **Source**: Profile Skills vs JD Requirements
*   **Range**: `0 - N` / `0.0 - 1.0`

### `product_company_score` & `consulting_penalty`
*   **Family**: General Background
*   **Description**: Boolean flag tracking presence of major product vs consulting companies in timeline.
*   **Source**: `career_history` text match
*   **Range**: `0.0 - 1.0`

### `jd_core_score`
*   **Family**: JD Alignment
*   **Description**: Core domain score aggregating hits for retrieval, ranking, recommendation, matching, and evaluation systems.
*   **Source**: `career_history` text match
*   **Range**: `0+`

---

## Family 02: Career Semantic

### `semantic_score`
*   **Family**: Semantic Matching
*   **Description**: SentenceTransformer cosine similarity between the Job Description and the candidate's profile summary.
*   **Source**: `candidate_feature_generator.py`
*   **Range**: `0.0 - 1.0`

### `career_semantic_score`
*   **Family**: Semantic Matching
*   **Description**: High-fidelity cosine similarity between the JD and the aggregated career history text, bypassing keyword limitations.
*   **Source**: `career_semantic_matcher.py`
*   **Range**: `0.0 - 1.0`

---

## Family 03: Production ML

### `production_depth_score`
*   **Family**: Production ML
*   **Description**: Measures the depth of engineering maturity. Aggregates productionizing, monitoring, infrastructure scaling, and MLOps signals.
*   **Source**: `production_ml_feature_generator.py`
*   **Range**: `0+`

### `wrapper_to_production_ratio`
*   **Family**: Production ML
*   **Description**: Ratio of "LLM wrapper" keywords vs actual production/infrastructure engineering keywords. Highlights prompt engineers vs backend ML engineers.
*   **Source**: `production_ml_feature_generator.py`
*   **Range**: `0.0 - 1.0`

---

## Family 04: Ownership & Leadership

### `ownership_score`
*   **Family**: Ownership & Leadership
*   **Description**: Frequency of phrases implying architectural ownership (e.g., "designed from scratch", "owned the pipeline").
*   **Source**: `ownership_leadership_feature_generator.py`
*   **Range**: `0+`

### `leadership_depth_score`
*   **Family**: Ownership & Leadership
*   **Description**: Aggregation of team leadership, mentorship, and engineering impact signals.
*   **Source**: `ownership_leadership_feature_generator.py`
*   **Range**: `0+`

---

## Family 05: Career Progression & Stability

### `career_growth_score`
*   **Family**: Career Progression
*   **Description**: Promotion and title growth signal. Tracks vertical jumps (Engineer -> Senior -> Lead) chronologically.
*   **Source**: `career_history` timeline
*   **Range**: `0+`

### `career_stability_score`
*   **Family**: Career Progression
*   **Description**: A positive integer scoring average tenure and multi-year loyalty, penalized by short-tenure stints.
*   **Source**: `career_history` timeline
*   **Range**: `0 - 100+`

### `job_hopping_score`
*   **Family**: Career Progression
*   **Description**: Ratio of unique companies worked for divided by total career timeline years.
*   **Source**: `career_history` timeline
*   **Range**: `0.0 - N`

---

## Family 06: Company Background

### `bigtech_ratio` & `startup_ratio`
*   **Family**: Company Background
*   **Description**: Percentage of a candidate's chronological career spent in known tech giants vs small startups.
*   **Source**: `career_history` metadata mapping
*   **Range**: `0.0 - 1.0`

### `company_type_switches`
*   **Family**: Company Background
*   **Description**: Integer counting how many times a candidate switched between completely different operating environments (e.g., Consulting -> Startup -> Big Tech).
*   **Source**: `career_history` metadata mapping
*   **Range**: `0 - N`

---

## Family 07: Suspicion (Honeypot Detection)

### `experience_gap_years`
*   **Family**: Suspicion
*   **Description**: Absolute difference between the explicitly claimed `years_exp` metric and the mathematically reconstructed timeline years.
*   **Source**: `profile` vs `career_history`
*   **Range**: `0.0 - N`

### `skill_density`
*   **Family**: Suspicion
*   **Description**: Ratio of claimed skills divided by the total reconstructed years of experience. Extremely high values correlate with keyword stuffing.
*   **Source**: `skills` array
*   **Range**: `0.0 - N`

### `suspicion_score`
*   **Family**: Suspicion
*   **Description**: Aggregate tracker for resume inconsistencies. A high score means Jain should downweight the profile.
*   **Source**: `suspicion_feature_generator.py`
*   **Range**: `0.0 - N`

---

## Family 08: Interactions

### `technical_leadership_score`
*   **Family**: Interactions
*   **Description**: Multiplicative interaction `production_depth_score * leadership_depth_score`. Extremely powerful signal for Principal Engineers.
*   **Source**: Parquet merge math
*   **Range**: `0+`

### `seniority_relevance_score`
*   **Family**: Interactions
*   **Description**: High relevance to JD combined with high career growth (`career_growth_score * career_semantic_score`).
*   **Source**: Parquet merge math
*   **Range**: `0+`
