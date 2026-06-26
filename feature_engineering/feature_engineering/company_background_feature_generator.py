import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = BASE_DIR / "data" / "candidates.jsonl"
OUTPUT_PATH = BASE_DIR / "artifacts" / "company_background_features.parquet"

KNOWN_BIGTECH = {
    "google", "meta", "amazon", "microsoft", "apple", 
    "netflix", "uber", "linkedin", "airbnb", "stripe",
    "databricks", "snowflake", "openai", "anthropic"
}

CONSULTING_TERMS = {
    "tcs", "infosys", "wipro", "cognizant", "capgemini", 
    "accenture", "hcl", "mindtree", "consulting", "services", 
    "solutions", "technologies", "deloitte", "pwc", "ey", "kpmg"
}

# Type Encodings
UNKNOWN = 0
CONSULTING = 1
STARTUP = 2
PRODUCT = 3
BIGTECH = 4

def classify_company(company_name, company_size):
    company_name = company_name.lower().strip()
    
    # 1. Company Name Match (Most reliable)
    if any(bt in company_name for bt in KNOWN_BIGTECH):
        return BIGTECH
        
    if any(ct in company_name for ct in CONSULTING_TERMS):
        return CONSULTING
        
    # 2. Size Fallback
    if company_size:
        # e.g., "1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10001+"
        if company_size in ["1-10", "11-50", "51-200"]:
            return STARTUP
        elif company_size in ["10001+"]:
            return PRODUCT # Assume large product if not big tech or consulting
        else:
            return PRODUCT # Default to product for mid-size
            
    # Default fallback
    return UNKNOWN

def main():
    candidate_ids = []
    
    startup_years_list = []
    product_years_list = []
    consulting_years_list = []
    bigtech_years_list = []
    
    startup_ratios = []
    product_ratios = []
    consulting_ratios = []
    bigtech_ratios = []
    
    company_type_switches_list = []
    company_diversity_scores = []
    current_company_types = []
    
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing candidates"):
            candidate = json.loads(line)
            c_id = candidate.get("candidate_id")
            career_history = candidate.get("career_history", [])
            
            # Sort roles chronologically (assuming they might not be)
            # Actually, we can just process them in order provided if they represent chronological
            # But let's track transitions. We assume career_history is sorted newest first or oldest first.
            # Usually resumes are newest first. So reversed is chronological.
            
            total_months = 0
            
            type_months = {
                STARTUP: 0,
                PRODUCT: 0,
                CONSULTING: 0,
                BIGTECH: 0,
                UNKNOWN: 0
            }
            
            history_types = []
            
            # Process chronological (reverse of newest-first)
            for job in reversed(career_history):
                c_name = job.get("company", "")
                c_size = job.get("company_size", "")
                dur = job.get("duration_months") or 0
                
                c_type = classify_company(c_name, c_size)
                
                type_months[c_type] += dur
                total_months += dur
                
                # Only record if it's a new job (avoid double counting same company)
                # Wait, if they had 2 roles at same company, type doesn't switch.
                if not history_types or history_types[-1] != c_type:
                    history_types.append(c_type)
                    
            # Current employer type (last in chronological, so first in newest-first)
            curr_type = UNKNOWN
            if career_history:
                curr_job = career_history[0]
                curr_type = classify_company(curr_job.get("company", ""), curr_job.get("company_size", ""))
                
            # Switches
            switches = max(0, len(history_types) - 1)
            
            # Diversity
            unique_types = len(set(history_types) - {UNKNOWN})
            
            # Years
            s_years = type_months[STARTUP] / 12.0
            p_years = type_months[PRODUCT] / 12.0
            c_years = type_months[CONSULTING] / 12.0
            b_years = type_months[BIGTECH] / 12.0
            
            # Ratios
            total_y = total_months / 12.0
            if total_y > 0:
                s_ratio = s_years / total_y
                p_ratio = p_years / total_y
                c_ratio = c_years / total_y
                b_ratio = b_years / total_y
            else:
                s_ratio = p_ratio = c_ratio = b_ratio = 0.0
                
            candidate_ids.append(c_id)
            startup_years_list.append(s_years)
            product_years_list.append(p_years)
            consulting_years_list.append(c_years)
            bigtech_years_list.append(b_years)
            
            startup_ratios.append(s_ratio)
            product_ratios.append(p_ratio)
            consulting_ratios.append(c_ratio)
            bigtech_ratios.append(b_ratio)
            
            company_type_switches_list.append(switches)
            company_diversity_scores.append(unique_types)
            current_company_types.append(curr_type)

    df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "startup_years": startup_years_list,
        "product_years": product_years_list,
        "consulting_years": consulting_years_list,
        "bigtech_years": bigtech_years_list,
        "startup_ratio": startup_ratios,
        "product_ratio": product_ratios,
        "consulting_ratio": consulting_ratios,
        "bigtech_ratio": bigtech_ratios,
        "company_type_switches": company_type_switches_list,
        "company_diversity_score": company_diversity_scores,
        "current_company_type": current_company_types
    })

    print("\n--- Validation Stats ---")
    print(df[["bigtech_years", "startup_ratio", "company_type_switches", "company_diversity_score"]].describe())
    
    print("\n--- Top 10 by BigTech Years ---")
    print(df.sort_values("bigtech_years", ascending=False).head(10)[["candidate_id", "bigtech_years", "company_type_switches"]])

    print("\n--- Top 10 by Startup Ratio ---")
    print(df.sort_values("startup_ratio", ascending=False).head(10)[["candidate_id", "startup_ratio", "startup_years"]])

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved company background features to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
