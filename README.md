# India Runs Data and AI Challenge - Team Musketeers

This repository contains the end-to-end pipeline for the AI Challenge, encompassing data preprocessing, model training, and final ranking & reasoning generation for the candidate submissions.

The repository is modularized into three core phases: **feature_engineering**, **modeling**, and **reasoning**, all orchestrated by a single root execution script.

---

## 🏗️ Phase 1: feature_engineering (Data Preprocessing)
**Directory**: `feature_engineering/`

The feature_engineering module handles the initial raw data ingestion, candidate sampling, and complex feature extraction required to build a structured training dataset.

**Key Components:**
- Extracts semantic information and generates embeddings between candidate resumes and target job descriptions.
- Processes behavioral signals to calculate reliability metrics.
- Outputs the finalized, merged features (`feature_engineering/artifacts/merged_features.parquet`) ready for modeling and inference.

---

## 🤖 Phase 2: modeling (Model Training & Inference)
**Directory**: `modeling/`

The modeling module is responsible for the core Machine Learning architecture, specifically optimizing for highly generalizable, accurate candidate scoring using a regularized XGBoost Regressor.

**Key Components:**
- **Training Script (`modeling/training/train_xgboost_model.py`):** Trains the core XGBoost Regressor against 70 heavily curated features to predict candidate performance.
- **Methodology & Experiments:** The `labeling/`, `prompts/`, `sampling/`, and `evaluation/` directories contain the exact methodology used to generate ground truth labels via LLMs, sample the data, and rigorously benchmark different model architectures before arriving at the final XGBoost solution.
- **Model Artifacts:** The trained weights are stored as `xgboost_ranking_model.pkl` and the input schema as `model_feature_schema.json` within the `artifacts/` folder.

---

## 🏆 Phase 3: reasoning (Advanced Candidate Profiling)
**Directory**: `reasoning/`

The reasoning module generates dynamic, professional, recruiter-ready justification text for each ranked candidate.

**Key Components:**
- **Adaptive Generation:** Dynamically profiles candidates by mapping their absolute scores against pool percentiles to classify the pool quality (e.g., `NORMAL`, `OUTLIER`, `WEAK`).
- **Contextual Reasoning:** Crafts human-readable strings explaining exactly why a candidate was ranked highly based on their top technical traits and experience levels.

---

## 🚀 Execution Engine: `Rank.py`

The entire pipeline has been unified into a single, robust execution script: **`Rank.py`**.

`Rank.py` automatically handles:
1. Loading the XGBoost model and candidate features.
2. Scoring the candidates using the `xgboost_ranking_model.pkl`.
3. Validating the top predictions by verifying their performance across the entire 70-feature space (automatically swapping out mathematically "weak" candidates for well-rounded ones).
4. Generating dynamic reasoning via the `reasoning` module.
5. Outputting the completely compliant, strictly validated hackathon file: `team_musketeers.csv`.

### How to Run:
To generate the final submission CSV from the candidates list, simply run:
```bash
python Rank.py
```
*(You can also rank subsets by passing `--input_file <your_subset.jsonl>`)*

To validate the final submission against the competition rules:
```bash
python validate_submission.py team_musketeers.csv
```

---

## 📂 Project Structure Note
> All old experimental scripts, previous checkpoints, and raw unstructured files have been moved into the `archive/` directory to maintain a clean root environment for final auditing.
