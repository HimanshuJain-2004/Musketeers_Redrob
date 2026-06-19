import os
import json
import pandas as pd
import subprocess
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAIN_ARTIFACTS = os.path.join(BASE_DIR, "JAIN", "artifacts")
JAINWIN_DIR = os.path.join(BASE_DIR, "JAINWIN")
JAINWIN_ARTIFACTS = os.path.join(JAINWIN_DIR, "artifacts")
CANDIDATES_JSONL = os.path.join(BASE_DIR, "candidates.jsonl")
VALIDATOR_SCRIPT = os.path.join(BASE_DIR, "validate_submission.py")
OUTPUT_CSV = os.path.join(JAINWIN_ARTIFACTS, "team_musketeers.csv")

os.makedirs(JAINWIN_ARTIFACTS, exist_ok=True)

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

def run_ranking(input_file=None):
    print("Loading predictions...")
    preds_df = pd.read_parquet(os.path.join(JAIN_ARTIFACTS, "predictions.parquet"))
    
    if input_file:
        print(f"Filtering based on provided file: {input_file}")
        input_ids = set()
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if '"candidate_id"' in line:
                    cand = json.loads(line)
                    input_ids.add(cand["candidate_id"])
        
        preds_df = preds_df[preds_df["candidate_id"].isin(input_ids)]

    print("Filtering honeypots and ranking...")
    # Filter honeypots (keep those with probability < 0.5)
    clean_df = preds_df[preds_df["honeypot_probability"] < 0.5].copy()
    
    # Sort
    clean_df = clean_df.sort_values(by=["predicted_score", "candidate_id"], ascending=[False, True])
    
    # Take Top 100 (or less if fewer candidates exist)
    top100_df = clean_df.head(100).copy()
    num_candidates = len(top100_df)
    
    if num_candidates == 0:
        print("No valid candidates found to rank!")
        return

    top100_df["rank"] = range(1, num_candidates + 1)
    
    top100_ids = set(top100_df["candidate_id"].tolist())
    candidate_data = {}
    
    print("Extracting raw features from candidates.jsonl (this may take a moment)...")
    with open(CANDIDATES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if '"candidate_id"' in line: # Quick string check to speed up parsing
                cand = json.loads(line)
                cid = cand["candidate_id"]
                if cid in top100_ids:
                    candidate_data[cid] = cand
                    if len(candidate_data) == num_candidates:
                        break # Found all we need!
                        
    print("Generating reasoning strings...")
    reasoning_list = []
    for idx, row in top100_df.iterrows():
        cid = row["candidate_id"]
        rank = row["rank"]
        cand = candidate_data[cid]
        reasoning = generate_reasoning(cand, rank)
        reasoning_list.append(reasoning)
        
    top100_df["reasoning"] = reasoning_list
    
    # Format according to spec: candidate_id,rank,score,reasoning
    # Score should be float formatting (let's use 4 decimal places)
    top100_df["score"] = top100_df["predicted_score"].apply(lambda x: f"{x:.4f}")
    
    final_csv = top100_df[["candidate_id", "rank", "score", "reasoning"]]
    
    print(f"Exporting to {OUTPUT_CSV}...")
    final_csv.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    print("Running Validator...")
    if os.path.exists(VALIDATOR_SCRIPT):
        result = subprocess.run(["python", VALIDATOR_SCRIPT, OUTPUT_CSV], capture_output=True, text=True)
        if result.returncode != 0:
            print("VALIDATION FAILED! (If testing a small subset, this might fail expectedly due to row count checks).")
            print(result.stdout)
            print(result.stderr)
            if not input_file:
                raise RuntimeError("Submission validation failed.")
        else:
            print("VALIDATION PASSED!")
            print(result.stdout)
    else:
        print("Validator script not found, skipping validation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ranking Engine")
    parser.add_argument("--input_file", type=str, help="Path to input jsonl containing candidate subset to rank", default=None)
    args = parser.parse_args()
    
    run_ranking(input_file=args.input_file)
