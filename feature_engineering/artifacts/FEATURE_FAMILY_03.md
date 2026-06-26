# Feature Family 03: Production ML Features

## Purpose
Capture evidence of production ML ownership, deployment, infrastructure exposure, and operational maturity. This clearly distinguishes hands-on Production ML Engineers from candidates whose experience consists solely of experimentation or API wrapper integrations (e.g., Prompt Engineers).

## Features Delivered
*   **Atomic Counts**:
    *   `prod_ml_count` / `prod_ml_present`
    *   `monitoring_count` / `monitoring_present`
    *   `infra_count` / `infra_present`
    *   `mlops_count` / `mlops_present`
*   **Aggregate Scores**:
    *   `production_ml_score`: Uniform sum of production, monitoring, and infra exposure.
    *   `production_depth_score`: Weighted sum explicitly rewarding real production logic (`prod_ml_count` * 2).
*   **Negative Signals**:
    *   `llm_wrapper_count`: Tracks buzzword-heavy LLM integrations.
    *   `wrapper_to_production_ratio`: Identifies candidates heavily slanted towards generic API usage over operational machine learning infrastructure.

## Extraction Method
1.  **Text Source**: Extracts metrics strictly from the descriptions, responsibilities, summary, and achievements of the candidate's `career_history` array. Generic skill lists are ignored.
2.  **Logic**: Count-based and boolean pattern-matching powered by specialized regex dictionaries targeting real-world operations rather than generic theory.

## Output
Stored independently as `production_features.parquet` and successfully merged into the final `merged_features.parquet` training matrix.
