import os
import pickle
import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIYUSH_ARTIFACTS = os.path.join(BASE_DIR, "PIYUSH", "artifacts")
JAIN_ARTIFACTS = os.path.join(BASE_DIR, "JAIN", "artifacts")
MODELS_DIR = os.path.join(JAIN_ARTIFACTS, "models")
OUTPUT_FILE = os.path.join(JAIN_ARTIFACTS, "predictions.parquet")

def load_and_merge_data():
    logging.info("Loading 100k dataset from PIYUSH artifacts...")
    
    # Load all 3 files
    try:
        cand_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "candidate_features.parquet"))
        behav_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "behavior_features.parquet"))
        sem_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "semantic_scores.parquet"))
    except FileNotFoundError as e:
        logging.error(f"Missing input feature files: {e}")
        raise
        
    logging.info("Merging datasets on candidate_id...")
    # Secure merge
    df = cand_df.merge(behav_df, on="candidate_id", how="inner").merge(sem_df, on="candidate_id", how="inner")
    
    logging.info(f"Merged dataset shape: {df.shape}")
    return df

def run_inference():
    df = load_and_merge_data()
    
    # Extract features matching the model training exactly
    # We will exclude the same columns as we did during training
    exclude_cols = [
        "candidate_id", "fit_score", "technical_fit", 
        "product_fit", "career_fit", "behavioral_fit", 
        "recruiter_composite_score", "honeypot_label"
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    
    # Load Models
    logging.info("Loading Models...")
    with open(os.path.join(MODELS_DIR, "best_benchmark_model.pkl"), "rb") as f:
        regressor = pickle.load(f)
        
    with open(os.path.join(MODELS_DIR, "best_honeypot_classifier.pkl"), "rb") as f:
        classifier = pickle.load(f)
        
    logging.info("Running Regression Inference (Predicted Score)...")
    df['predicted_score'] = regressor.predict(X.values if hasattr(regressor, 'feature_names_in_') == False else X)
    
    logging.info("Running Classification Inference (Honeypot Probability)...")
    # predict_proba returns [prob_0, prob_1]
    df['honeypot_probability'] = classifier.predict_proba(X.values if hasattr(classifier, 'feature_names_in_') == False else X)[:, 1]
    
    # Prepare Output
    output_df = df[['candidate_id', 'predicted_score', 'honeypot_probability']]
    
    # Validations
    logging.info("Running Strict Validations...")
    row_count = len(output_df)
    unique_candidates = output_df['candidate_id'].nunique()
    
    if row_count != 100000:
        logging.error(f"Validation Failed: Expected 100,000 rows, got {row_count}")
        raise ValueError(f"Strict Assertion Failed: len == {row_count}")
        
    if unique_candidates != 100000:
        logging.error(f"Validation Failed: Expected 100,000 unique candidates, got {unique_candidates}")
        raise ValueError(f"Strict Assertion Failed: nunique == {unique_candidates}")
        
    logging.info("PASSED: 100,000 exactly scored rows. Zero duplicates.")
    
    # Save Output
    output_df.to_parquet(OUTPUT_FILE, index=False)
    logging.info(f"Successfully saved predictions to {OUTPUT_FILE}")

    # Generate Top 100 with Reasoning
    logging.info("Extracting top 100 candidates and generating reasoning...")
    top_100 = df.sort_values(by='predicted_score', ascending=False).head(100).copy()
    
    feature_mapping = {
        'years_exp': 'years_exp',
        'experience_score': 'years_exp, skill_match_count, seniority',
        'skill_match_count': 'skills',
        'skill_coverage': 'skills',
        'title_score': 'job_title',
        'education_score': 'education',
        'location_score': 'location',
        'product_company_score': 'company_history',
        'consulting_penalty': 'company_history',
        'retrieval_score': 'resume_text',
        'availability_score': 'days_since_active',
        'recruitability_score': 'days_since_active, notice_period_days',
        'market_demand_score': 'skills, location',
        'trust_score': 'github_activity, linkedin',
        'technical_credibility_score': 'github_activity, avg_assessment_score',
        'mobility_score': 'location',
        'days_since_active': 'last_active_date',
        'notice_period_days': 'notice_period',
        'github_activity_score': 'github_activity',
        'avg_assessment_score': 'assessments',
        'semantic_score': 'resume_text, job_description'
    }
    
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        best_feat_idx = np.argmax(importances)
        most_imp_feature = feature_cols[best_feat_idx]
    else:
        most_imp_feature = 'experience_score'
        
    original_features = feature_mapping.get(most_imp_feature, 'original_features_unknown')
    template = "Candidate {cand_id} was ranked in the top 100 primarily due to their {imp_feature}, which is derived from the original dataset features: {orig}."
    
    top_100['reasoning'] = top_100['candidate_id'].apply(
        lambda x: template.format(cand_id=x, imp_feature=most_imp_feature, orig=original_features)
    )
    
    top_100_file = os.path.join(JAIN_ARTIFACTS, "top_100_candidates_with_reasoning.csv")
    top_100[['candidate_id', 'predicted_score', 'reasoning']].to_csv(top_100_file, index=False)
    logging.info(f"Successfully saved Top 100 candidates to {top_100_file}")

if __name__ == "__main__":
    run_inference()
