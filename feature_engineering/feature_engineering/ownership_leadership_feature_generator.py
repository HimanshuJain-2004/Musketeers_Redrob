import json
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = BASE_DIR / "data" / "candidates.jsonl"
OUTPUT_PATH = BASE_DIR / "artifacts" / "ownership_features.parquet"

OWNERSHIP_TERMS = [
    r"\bowned\b", r"\bownership\b", r"\bresponsible for\b",
    r"\bdrove\b", r"\bdelivered\b", r"\blaunched\b",
    r"\bexecuted\b", r"\binitiated\b", r"\bchampioned\b",
    r"\bspearheaded\b"
]

LEADERSHIP_TERMS = [
    r"\bled\b", r"\bleading\b", r"\bmanaged\b",
    r"\bmanager\b", r"\bhead of\b", r"\bdirector\b",
    r"\bteam lead\b", r"\btech lead\b", r"\bengineering lead\b"
]

ARCHITECTURE_TERMS = [
    r"\barchitected\b", r"\bdesigned\b", r"\bsystem design\b",
    r"\bdesigned architecture\b", r"\bplatform architecture\b",
    r"\bsolution architecture\b", r"\btechnical design\b"
]

MENTORSHIP_TERMS = [
    r"\bmentored\b", r"\bcoached\b", r"\btrained\b",
    r"\bguided\b", r"\bonboarded\b"
]

IMPACT_TERMS = [
    r"\bincreased\b", r"\breduced\b", r"\bimproved\b",
    r"\boptimized\b", r"\bsaved\b", r"\bboosted\b",
    r"\bgrew\b", r"\bscaled\b"
]

def compile_regex(terms):
    return re.compile("|".join(terms), re.IGNORECASE)

RE_OWNERSHIP = compile_regex(OWNERSHIP_TERMS)
RE_LEADERSHIP = compile_regex(LEADERSHIP_TERMS)
RE_ARCHITECTURE = compile_regex(ARCHITECTURE_TERMS)
RE_MENTORSHIP = compile_regex(MENTORSHIP_TERMS)
RE_IMPACT = compile_regex(IMPACT_TERMS)

# Seniority terms check current title
RE_SENIOR = re.compile(r"\b(senior|staff|principal|lead|manager|director|head)\b", re.IGNORECASE)
RE_EXEC = re.compile(r"\b(director|vp|head|chief)\b", re.IGNORECASE)

def build_career_history_text(profile):
    career_history = profile.get("career_history", [])
    parts = []
    for job in career_history:
        desc = job.get("description", "").strip()
        resp = job.get("responsibilities", "").strip()
        summary = job.get("summary", "").strip()
        achievements = job.get("achievements", "").strip()
        
        if desc: parts.append(desc)
        if resp: parts.append(resp)
        if summary: parts.append(summary)
        if achievements: parts.append(achievements)
        
    return " ".join(parts)

def count_matches(pattern, text):
    return len(pattern.findall(text))

def check_title(pattern, candidate):
    profile = candidate.get("profile", {})
    # Get current_title or the title of the first job in career_history
    title = profile.get("current_title", "")
    if not title:
        career_history = profile.get("career_history", [])
        if career_history:
            title = career_history[0].get("title", "")
    
    return int(bool(pattern.search(title)))

def main():
    candidate_ids = []
    ownership_counts = []
    leadership_counts = []
    architecture_counts = []
    mentorship_counts = []
    impact_counts = []
    senior_flags = []
    exec_flags = []
    
    # limit = 1000 # Sample limit for validation
    
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Processing candidates")):
            # if i >= limit:
            #    break
            
            candidate = json.loads(line)
            c_id = candidate.get("candidate_id")
            text = build_career_history_text(candidate)
            
            candidate_ids.append(c_id)
            ownership_counts.append(count_matches(RE_OWNERSHIP, text))
            leadership_counts.append(count_matches(RE_LEADERSHIP, text))
            architecture_counts.append(count_matches(RE_ARCHITECTURE, text))
            mentorship_counts.append(count_matches(RE_MENTORSHIP, text))
            impact_counts.append(count_matches(RE_IMPACT, text))
            
            # Check seniority flags against title
            senior_flags.append(check_title(RE_SENIOR, candidate))
            exec_flags.append(check_title(RE_EXEC, candidate))
            
    df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "ownership_count": ownership_counts,
        "leadership_count": leadership_counts,
        "architecture_count": architecture_counts,
        "mentorship_count": mentorship_counts,
        "impact_count": impact_counts,
        "senior_title_flag": senior_flags,
        "executive_title_flag": exec_flags
    })
    
    # Presence booleans
    for col in ["ownership", "leadership", "architecture", "mentorship", "impact"]:
        df[f"{col}_present"] = (df[f"{col}_count"] > 0).astype(int)
    
    # Aggregates
    df["ownership_score"] = df["ownership_count"] + df["leadership_count"] + df["architecture_count"]
    
    df["leadership_depth_score"] = (
        df["ownership_count"] +
        (df["leadership_count"] * 2) +
        df["architecture_count"] +
        df["mentorship_count"] +
        df["impact_count"]
    )
    
    print("\n--- Validation on Sample ---")
    print(df[["ownership_score", "leadership_depth_score"]].describe())
    
    print("\n--- Top 20 Candidates by Leadership Depth Score ---")
    top_candidates = df.sort_values("leadership_depth_score", ascending=False).head(20)
    
    cols_to_print = [
        "candidate_id", "ownership_count", "leadership_count", 
        "architecture_count", "mentorship_count", "impact_count", 
        "leadership_depth_score"
    ]
    print(top_candidates[cols_to_print])

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved ownership features to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
