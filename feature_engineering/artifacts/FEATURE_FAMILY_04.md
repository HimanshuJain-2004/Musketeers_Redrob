# Feature Family 04: Ownership & Leadership Features

## Purpose
Recruiters distinguish between candidates who merely contributed to systems versus those who owned, led, or architected them. This feature family explicitly extracts signals of ownership, mentorship, measurable impact, and architecture design from the candidate's core descriptions to reward leadership.

## Features Delivered
*   **Atomic Counts and Booleans**:
    *   `ownership_count` / `ownership_present`
    *   `leadership_count` / `leadership_present`
    *   `architecture_count` / `architecture_present`
    *   `mentorship_count` / `mentorship_present`
    *   `impact_count` / `impact_present`
*   **Seniority Flags**:
    *   `senior_title_flag`: Checks if current title indicates seniority (e.g. Senior, Lead, Principal).
    *   `executive_title_flag`: Checks if current title indicates executive status (e.g. VP, Director, Head).
*   **Aggregate Scores**:
    *   `ownership_score`: Uniform sum of ownership, leadership, and architecture counts.
    *   `leadership_depth_score`: Weighted sum emphasizing explicit leadership (`leadership_count` * 2) combined with mentorship, architecture, and impact.

## Extraction Method
1.  **Text Source**: Feature counts are evaluated strictly on the text found in the `description`, `responsibilities`, `summary`, and `achievements` of the `career_history`.
2.  **Logic**: Count-based and boolean pattern-matching powered by specialized regex dictionaries targeting action verbs (e.g., `spearheaded`, `architected`, `mentored`). Seniority flags are evaluated via regex on the candidate's `current_title` (or the most recent role title).

## Output
Stored independently as `ownership_features.parquet` and successfully merged into the final `merged_features.parquet` training matrix.
