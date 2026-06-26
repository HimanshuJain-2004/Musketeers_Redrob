import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = BASE_DIR / "data" / "candidates.jsonl"

def build_career_history_text(profile):
    career_history = profile.get("career_history", [])
    
    parts = []
    for job in career_history:
        desc = job.get("description", "").strip()
        resp = job.get("responsibilities", "").strip()
        summary = job.get("summary", "").strip()
        achievements = job.get("achievements", "").strip()
        
        # We only collect evidence of work performed
        if desc: parts.append(desc)
        if resp: parts.append(resp)
        if summary: parts.append(summary)
        if achievements: parts.append(achievements)
        
    return " ".join(parts)

def main():
    lengths = []
    
    count = 0
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if count >= 500:
                break
                
            candidate = json.loads(line)
            # The candidate json has profile and career_history as top-level keys
            # The instruction said "build_career_history_text(profile)" but candidate.get("career_history") is at root
            # I will pass candidate to build_career_history_text
            text = build_career_history_text(candidate)
            lengths.append(len(text))
            
            count += 1
            
    lengths = np.array(lengths)
    
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    p25 = np.percentile(lengths, 25)
    p75 = np.percentile(lengths, 75)
    
    # Let's define "almost empty" as length < 50 characters
    empty_rate = np.mean(lengths < 50) * 100
    
    print(f"--- Career History Text Validation (N=500) ---")
    print(f"Mean Length:   {mean_len:.1f} chars")
    print(f"Median Length: {median_len:.1f} chars")
    print(f"P25 Length:    {p25:.1f} chars")
    print(f"P75 Length:    {p75:.1f} chars")
    print(f"Empty Rate (<50 chars): {empty_rate:.1f}%")

if __name__ == "__main__":
    main()
