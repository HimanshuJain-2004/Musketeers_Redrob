# Evidence Grounded Narrative Generation (EGNG)

This deterministic engine converts outputs from the ranking pipeline into highly explainable, recruiter-quality narrative justifications.

## Architecture

- `extractors/`: Extracts deterministic facts and driver features from candidate datasets.
- `planners/`: Decides candidate story patterns, identifies concerns, and classifies performance tiers.
- `generators/`: Mixes phrase templates to compose 1-2 sentence reasons deterministically.
- `validators/`: Safely validates output against hallucinations.
- `pipelines/`: Orchestrates the complete flow.

## Running the Pipeline

```bash
python pipelines/reasoning_pipeline.py
```

## Outputs

- `outputs/submission_with_reasoning.csv`: Final ranking submission file.
- `outputs/debug_reasoning.json`: Explainability metadata.
