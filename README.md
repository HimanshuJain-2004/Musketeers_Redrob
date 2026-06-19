# India Runs Data and AI Challenge - Team Musketeers

This repository contains the end-to-end pipeline for the AI Challenge, encompassing data preprocessing, model training, and final ranking & reasoning generation for the top 100 candidate submissions.

The repository is modularized into three core phases owned by different pipeline components: **PIYUSH**, **JAIN**, and **JAINWIN**.

---

## 🏗️ Phase 1: PIYUSH (Data Preprocessing & Feature Engineering)
**Directory**: `PIYUSH/`

The PIYUSH module handles the initial raw data ingestion, candidate sampling, and complex feature extraction required to build a structured training dataset.

**Key Components:**
- **Candidate Parsing:** Extracts semantic information from `candidates.jsonl`.
- **Semantic Matching:** Generates embeddings and similarity scores (`semantic_scores.parquet`) between candidate resumes and the target Job Description (`jd_requirements.json`).
- **Behavioral Scoring:** Processes `redrob_signals` to calculate behavioral and reliability metrics (`behavior_features.parquet`).
- **Feature Store:** Outputs the finalized, merged features (`candidate_features.parquet`) ready for modeling.

---

## 🤖 Phase 2: JAIN (Model Training & Inference)
**Directory**: `JAIN/`

The JAIN module is responsible for the core Machine Learning architecture, specifically optimizing for accurate candidate scoring and honeypot detection.

**Key Components:**
- **Honeypot Classifier:** Trains a robust model to filter out invalid or low-effort applications (`train_honeypot_classifier.py`).
- **Comprehensive Benchmarking:** Evaluates multiple regressors and rankers (RandomForest, LightGBM, CatBoost, HistGradientBoosting) alongside a deeply tuned `CustomEnsemble` composed of ExtraTrees, LGBM, and XGBoost (`comprehensive_benchmarking.py`).
- **Full Dataset Prediction:** Executes inference across the entire 100k candidate pool using the globally optimal model (`full_dataset_prediction.py`), outputting raw scores and honeypot probabilities into `predictions.parquet`.

---

## 🏆 Phase 3: JAINWIN (Final Ranking & Reasoning)
**Directory**: `JAINWIN/`

The JAINWIN module (formerly BHAVIT) handles the final business logic, enforcing the competition rules, filtering out honeypots, and constructing the official hackathon submission.

**Key Components:**
- **Ranking Engine:** Loads the full predictions, automatically drops candidate profiles with a high honeypot probability, and ranks the remaining candidates to extract the elite Top 100 (`ranking_engine.py`).
- **Reasoning Generation:** Dynamically interrogates the raw `candidates.jsonl` data to craft human-readable, highly professional recruiter-style reasoning strings based on the candidate's actual top 3 technical drivers and traits.
- **Submission Output:** Outputs the completely compliant, strictly validated hackathon file (`team_musketeers.csv`).

---

## 🚀 How to Run the Pipeline

1. **Extract Features (PIYUSH):** Run the extraction scripts inside the PIYUSH directory to generate the initial parquet artifacts.
2. **Train Models (JAIN):** 
   ```bash
   python JAIN/training/train_honeypot_classifier.py
   python JAIN/training/comprehensive_benchmarking.py
   ```
3. **Run Inference (JAIN):**
   ```bash
   python JAIN/inference/full_dataset_prediction.py
   ```
4. **Generate Final Submission (JAINWIN):**
   ```bash
   python JAINWIN/ranking_engine.py
   ```
   *This automatically creates `JAINWIN/artifacts/team_musketeers.csv` and executes the hackathon's `validate_submission.py` to guarantee compliance.*

---

## 📂 Project Structure Note
> All raw, unstructured instruction files, previous specs, and loose `.docx` / `.txt` files have been moved into the `archive/` directory to maintain a clean root environment for final auditing.
