# Feature Family 02: Career Semantic Features

## Purpose
Capture evidence of actual work performed while reducing the influence of skill lists, headlines, and profile keyword stuffing. This ensures candidates who superficially list AI terminology without supporting job descriptions are appropriately penalized.

## Features Delivered
*   `career_semantic_score`: The cosine similarity between the candidate's career-history-only text embedding and the JD embedding.

## Extraction Method
1.  **Text Source**: Extracts only the `description`, `responsibilities`, `summary`, and `achievements` from the candidate's `career_history` array.
2.  **Exclusions**: Explicitly excludes generic profile fields such as `skills`, `headline`, `education`, and profile `summary`.
3.  **Embedding Model**: Uses the existing `sentence-transformers/all-MiniLM-L6-v2` model to keep the vector space compatible with the JD embedding.
4.  **Scoring**: Computes cosine similarity (dot product of normalized vectors) between the resulting career history embeddings and the JD embedding.

## Output
Stored independently as `career_semantic_features.parquet` and successfully integrated into the final training matrix via `merge_features.py`.
