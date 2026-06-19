import os
import json
import pickle
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIYUSH_ARTIFACTS = os.path.join(BASE_DIR, "PIYUSH", "artifacts")
JAIN_ARTIFACTS = os.path.join(BASE_DIR, "JAIN", "artifacts")
MODELS_DIR = os.path.join(JAIN_ARTIFACTS, "models")
CANDIDATES_JSONL_MAIN = os.path.join(BASE_DIR, "candidates.jsonl")
OUTPUT_CSV = os.path.join(BASE_DIR, "final_ranked_candidates.csv")
#hello
AI_SKILLS_KEYWORDS = {
    "machine learning", "deep learning", "nlp", "computer vision", 
    "llm", "generative ai", "rag", "pytorch", "tensorflow", 
    "vector database", "pinecone", "weaviate", "qdrant", "milvus", 
    "opensearch", "faiss", "bge", "e5", "sentence-transformers", 
    "openai", "lora", "qlora", "peft", "xgboost", "lightgbm", "catboost"
}

def generate_reasoning(cand, rank):
    profile = cand.get("profile", {})
    title = profile.get("current_title", "Candidate")
    exp = profile.get("years_of_experience", 0)
    loc = profile.get("location", "Unknown Location")
    
    # Analyze Skills
    raw_skills = cand.get("skills", [])
    matched_ai_skills = []
    for skill in raw_skills:
        skill_name = skill.get("name", "").lower()
        if any(keyword in skill_name for keyword in AI_SKILLS_KEYWORDS):
            matched_ai_skills.append(skill.get("name"))
    
    # Analyze Signals
    signals = cand.get("redrob_signals", {})
    response_rate = signals.get("recruiter_response_rate", 0)
    notice = signals.get("notice_period_days", 0)
    
    skill_str = "no core AI skills listed"
    if len(matched_ai_skills) > 0:
        top_skills = matched_ai_skills[:2]
        skill_str = f"strong in {', '.join(top_skills)} (out of {len(matched_ai_skills)} AI skills)"
        
    engagement_str = f"response rate {response_rate:.2f}"
    if notice > 60:
        engagement_str += f", high notice period ({notice} days)"
    elif notice <= 30:
        engagement_str += ", available immediately/sub-30 days"
        
    if rank <= 10:
        return f"{title} with {exp:.1f} years experience; {skill_str}; {engagement_str} and {loc}-based."
    elif rank <= 90:
        return f"{exp:.1f} years applied experience; {skill_str}; matches profile requirements with {engagement_str}."
    else:
        return f"Adjacent fit as {title} — likely below top tier but included given {exp:.1f} years exp and {engagement_str}."

def main(input_file=None):
    print("Loading precomputed features from PIYUSH artifacts...")
    cand_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "candidate_features.parquet"))
    behav_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "behavior_features.parquet"))
    sem_df = pd.read_parquet(os.path.join(PIYUSH_ARTIFACTS, "semantic_scores.parquet"))
    
    print("Merging datasets on candidate_id...")
    df = cand_df.merge(behav_df, on="candidate_id", how="inner").merge(sem_df, on="candidate_id", how="inner")
    
    candidate_raw_data = {}
    target_ids = None

    if input_file:
        print(f"Parsing input file: {input_file}")
        target_ids = set()
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if '"candidate_id"' in line:
                    cand = json.loads(line)
                    cid = cand["candidate_id"]
                    target_ids.add(cid)
                    candidate_raw_data[cid] = cand
        print(f"Found {len(target_ids)} candidates in input file.")
        
        # Filter dataframe
        df = df[df["candidate_id"].isin(target_ids)].copy()
        print(f"Dataset filtered to {len(df)} candidates.")
    
    if len(df) == 0:
        print("No candidates to process.")
        return

    exclude_cols = [
        "candidate_id", "fit_score", "technical_fit", 
        "product_fit", "career_fit", "behavioral_fit", 
        "recruiter_composite_score", "honeypot_label"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].fillna(0)
    
    print("Loading ML Models...")
    with open(os.path.join(MODELS_DIR, "best_benchmark_model.pkl"), "rb") as f:
        regressor = pickle.load(f)
        
    with open(os.path.join(MODELS_DIR, "best_honeypot_classifier.pkl"), "rb") as f:
        classifier = pickle.load(f)
        
    print("Running Inference...")
    df['predicted_score'] = regressor.predict(X.values if hasattr(regressor, 'feature_names_in_') == False else X)
    df['honeypot_probability'] = classifier.predict_proba(X.values if hasattr(classifier, 'feature_names_in_') == False else X)[:, 1]
    
    if len(df) <= 100:
        print(f"Total candidates is {len(df)} (<= 100). Skipping honeypot filtering to rank all available candidates.")
        clean_df = df.copy()
    else:
        print("Filtering honeypots (Probability < 0.5)...")
        clean_df = df[df["honeypot_probability"] < 0.5].copy()
        if len(clean_df) < 100:
            needed = 100 - len(clean_df)
            print(f"Honeypot filtering left only {len(clean_df)} candidates. Backfilling with {needed} highest-scoring honeypots to reach 100.")
            honeypots = df[df["honeypot_probability"] >= 0.5].sort_values(by=["predicted_score", "candidate_id"], ascending=[False, True])
            clean_df = pd.concat([clean_df, honeypots.head(needed)])
    
    print("Sorting by predicted score...")
    clean_df = clean_df.sort_values(by=["predicted_score", "candidate_id"], ascending=[False, True])
    
    # Rank Top 100 or all if less than 100
    top_df = clean_df.head(100).copy()
    num_candidates = len(top_df)
    
    if num_candidates == 0:
        print("No valid candidates found after honeypot filtering!")
        return

    print(f"Assigning ranks to {num_candidates} candidates...")
    top_df["rank"] = range(1, num_candidates + 1)
    top_ids = set(top_df["candidate_id"].tolist())
    
    # If candidate_raw_data is missing some (e.g. no input file provided), load from main jsonl
    missing_ids = top_ids - set(candidate_raw_data.keys())
    if missing_ids:
        print("Extracting raw features from main candidates.jsonl for reasoning...")
        with open(CANDIDATES_JSONL_MAIN, "r", encoding="utf-8") as f:
            for line in f:
                if '"candidate_id"' in line:
                    cand = json.loads(line)
                    cid = cand["candidate_id"]
                    if cid in missing_ids:
                        candidate_raw_data[cid] = cand
                        if len(missing_ids.intersection(candidate_raw_data.keys())) == len(missing_ids):
                            break
                            
    print("Generating reasoning explanations...")
    reasoning_list = []
    for idx, row in top_df.iterrows():
        cid = row["candidate_id"]
        rank = row["rank"]
        cand = candidate_raw_data.get(cid, {})
        reasoning = generate_reasoning(cand, rank)
        reasoning_list.append(reasoning)
        
    top_df["reasoning"] = reasoning_list
    top_df["score"] = top_df["predicted_score"].apply(lambda x: f"{x:.4f}")
    
    final_csv = top_df[["candidate_id", "rank", "score", "reasoning"]]
    
    print(f"Exporting results to {OUTPUT_CSV}...")
    final_csv.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print("Done! End-to-end processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Ranking Engine")
    parser.add_argument("--input_file", type=str, help="Path to input .jsonl containing candidates", default=None)
    args = parser.parse_args()
    
    main(input_file=args.input_file)
