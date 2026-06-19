import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIYUSH_ARTIFACTS = os.path.join(BASE_DIR, "PIYUSH", "artifacts")
JAIN_ARTIFACTS = os.path.join(BASE_DIR, "JAIN", "artifacts")

TOP100_FILE = os.path.join(JAIN_ARTIFACTS, "top_100_candidates_with_reasoning.csv")
RAW_JSONL = os.path.join(BASE_DIR, "candidates.jsonl")
FINAL_OUT = os.path.join(JAIN_ARTIFACTS, "final_submission.csv")

FEATURE_NAME_MAP = {
    'years_exp': 'Years of Experience',
    'experience_score': 'Senior-level Experience',
    'skill_match_count': 'Skill Match',
    'skill_coverage': 'Skill Coverage',
    'title_score': 'Title Relevance',
    'education_score': 'Educational Background',
    'location_score': 'Location Alignment',
    'product_company_score': 'Product Company Experience',
    'consulting_penalty': 'Consulting Background',
    'retrieval_score': 'Resume Relevance',
    'availability_score': 'Availability',
    'recruitability_score': 'Recruitability',
    'market_demand_score': 'Market Demand',
    'trust_score': 'Trust Score',
    'technical_credibility_score': 'Technical Credibility',
    'mobility_score': 'Mobility',
    'days_since_active': 'Recent Platform Activity',
    'notice_period_days': 'Notice Period',
    'github_activity_score': 'GitHub Activity',
    'avg_assessment_score': 'Behavioral Assessment Score',
    'semantic_score': 'Semantic Job Description Match'
}

def load_merged_features():
    cand_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "candidate_features.parquet"))
    behav_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "behavior_features.parquet"))
    sem_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "semantic_scores.parquet"))
    df = cand_df.merge(behav_df, on="candidate_id", how="inner").merge(sem_df, on="candidate_id", how="inner")
    return df

def generate_reasonings():
    print("Loading Top 100 Candidates...")
    top100_df = pd.read_csv(TOP100_FILE)
    top_cands = set(top100_df['candidate_id'].tolist())
    
    print("Loading Merged Features for Z-Score Calculation...")
    df = load_merged_features()
    exclude_cols = [
        "candidate_id", "fit_score", "technical_fit", 
        "product_fit", "career_fit", "behavioral_fit", 
        "recruiter_composite_score", "honeypot_label"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print("Calculating Z-scores to identify Top 3 Features per Candidate...")
    mean_vals = df[feature_cols].mean()
    std_vals = df[feature_cols].std().replace(0, 1) # avoid div by zero
    
    df_top = df[df['candidate_id'].isin(top_cands)].copy()
    
    top_3_dict = {}
    for _, row in df_top.iterrows():
        cid = row['candidate_id']
        z_scores = (row[feature_cols] - mean_vals) / std_vals
        # sort descending to get highest relative strengths
        top_3 = z_scores.sort_values(ascending=False).head(3).index.tolist()
        top_3_dict[cid] = top_3
        
    print("Extracting Raw Values from candidates.jsonl...")
    raw_data_dict = {}
    with open(RAW_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split('"', 7)
            if len(parts) > 3 and parts[3] in top_cands:
                c = json.loads(line)
                cid = c['candidate_id']
                profile = c.get('profile', {})
                years = profile.get('years_of_experience', 'several')
                title = profile.get('current_title', 'Professional')
                loc = profile.get('location', 'their location')
                
                skills = c.get('skills', [])
                top_skills = [s['name'] for s in skills[:3]]
                skills_str = ", ".join(top_skills) if top_skills else "relevant tools"
                
                raw_data_dict[cid] = {
                    'years': years,
                    'title': title,
                    'location': loc,
                    'skills': skills_str
                }
                
                if len(raw_data_dict) == len(top_cands):
                    break
                    
    print("Generating Professional Reasoning Strings...")
    new_reasonings = []
    for _, row in top100_df.iterrows():
        cid = row['candidate_id']
        raw = raw_data_dict.get(cid, {'years': 'several', 'title': 'Professional', 'location': 'their location', 'skills': 'core skills'})
        t3 = top_3_dict.get(cid, ['experience_score', 'skill_match_count', 'availability_score'])
        
        feats = [FEATURE_NAME_MAP.get(f, f.replace('_', ' ').title()) for f in t3]
        
        r = f"Standout applicant driven by {raw['years']} years of experience as a {raw['title']}, strong technical match ({raw['skills']}) in {raw['location']}. Ranked highly due to outstanding {feats[0]}, {feats[1]}, and {feats[2]}."
        new_reasonings.append(r)
        
    top100_df['reasoning'] = new_reasonings
    
    # Sort by predicted_score descending just to be safe, then add rank
    top100_df = top100_df.sort_values(by='predicted_score', ascending=False).reset_index(drop=True)
    top100_df['rank'] = range(1, len(top100_df) + 1)
    top100_df.rename(columns={'predicted_score': 'score'}, inplace=True)
    
    # Reorder columns
    final_df = top100_df[['candidate_id', 'rank', 'score', 'reasoning']]
    final_df.to_csv(FINAL_OUT, index=False)
    print(f"Successfully generated {FINAL_OUT}")

if __name__ == "__main__":
    generate_reasonings()
