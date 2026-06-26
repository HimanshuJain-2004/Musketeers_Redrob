import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

BASE_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = BASE_DIR / "data" / "candidates.jsonl"
OUTPUT_PATH = BASE_DIR / "artifacts" / "career_progression_features.parquet"

TITLE_LEVELS = {
    "intern": 0,
    "associate": 1,
    "engineer": 2,
    "senior": 3,
    "staff": 4,
    "lead": 5,
    "manager": 6,
    "director": 7,
    "vp": 8,
    "head": 9
}

# Regex to find title level
def get_title_level(title):
    title_lower = title.lower()
    max_level = -1
    for k, v in TITLE_LEVELS.items():
        if re.search(r'\b' + k + r'\b', title_lower):
            max_level = max(max_level, v)
    return max_level

RE_SENIOR = re.compile(r"\b(senior|staff|principal|lead|manager|director|head)\b", re.IGNORECASE)
RE_EXEC = re.compile(r"\b(director|vp|head|chief)\b", re.IGNORECASE)

def check_title(pattern, current_title):
    return int(bool(pattern.search(current_title)))

def parse_date(date_str):
    if not date_str:
        return None
    try:
        # Assuming YYYY-MM-DD
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except:
        return None

def main():
    candidate_ids = []
    
    total_experience_years = []
    num_companies = []
    num_roles = []
    avg_tenure_months = []
    longest_tenure_months = []
    shortest_tenure_months = []
    
    job_hopping_scores = []
    short_tenure_counts = []
    very_short_tenure_counts = []
    
    promotion_counts = []
    promotion_velocities = []
    career_growth_scores = []
    
    multi_year_company_counts = []
    long_term_employment_ratios = []
    
    timeline_gap_counts = []
    largest_gap_months = []
    career_stability_scores = []
    timeline_quality_scores = []
    
    # Validation stats
    total_roles_processed = 0
    missing_starts = 0
    missing_ends = 0

    # limit = 1000 # Validation limit, remove later
    
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Processing candidates")):
            # if i >= limit:
            #     break
                
            candidate = json.loads(line)
            c_id = candidate.get("candidate_id")
            profile = candidate.get("profile", {})
            career_history = candidate.get("career_history", [])
            
            # 1. Timeline Reconstruction & Missing Dates
            roles = []
            for job in career_history:
                total_roles_processed += 1
                start = job.get("start_date")
                end = job.get("end_date")
                
                if not start: missing_starts += 1
                if not end and not job.get("is_current"): missing_ends += 1
                
                s_date = parse_date(start)
                e_date = parse_date(end)
                if not e_date and job.get("is_current"):
                    e_date = datetime.now()
                    
                dur = job.get("duration_months")
                if dur is None:
                    if s_date and e_date:
                        diff = relativedelta(e_date, s_date)
                        dur = diff.years * 12 + diff.months
                    else:
                        dur = 0
                        
                roles.append({
                    "company": (job.get("company") or "").strip().lower(),
                    "title": (job.get("title") or "").strip(),
                    "start_date": s_date,
                    "end_date": e_date,
                    "duration_months": max(dur, 0)
                })
            
            # Quality score
            missing = sum([1 for r in roles if r["start_date"] is None])
            t_qual = 1.0 - (missing / len(roles)) if roles else 0.0
            
            # Sort roles chronologically
            valid_roles = [r for r in roles if r["start_date"] is not None]
            valid_roles.sort(key=lambda x: x["start_date"])
            
            # 2. Core Stability Features
            n_roles = len(roles)
            companies = {}
            total_months = 0
            
            for r in roles:
                c = r["company"]
                if c not in companies: companies[c] = 0
                companies[c] += r["duration_months"]
                total_months += r["duration_months"]
                
            n_companies = len(companies) if companies else 1
            t_exp_years = total_months / 12.0
            
            company_tenures = list(companies.values())
            avg_tenure = np.mean(company_tenures) if company_tenures else 0
            long_tenure = max(company_tenures) if company_tenures else 0
            short_tenure = min(company_tenures) if company_tenures else 0
            
            # 3. Job Hopping Features
            jh_score = n_companies / (t_exp_years + 1)
            short_tc = sum([1 for r in roles if r["duration_months"] < 12])
            v_short_tc = sum([1 for r in roles if r["duration_months"] < 6])
            
            # 4 & 5. Promotion Detection
            promo_count = 0
            max_level_seen = -1
            
            for r in valid_roles:
                lvl = get_title_level(r["title"])
                if lvl > max_level_seen and max_level_seen != -1:
                    promo_count += 1
                max_level_seen = max(max_level_seen, lvl)
                
            promo_vel = promo_count / (t_exp_years + 1)
            
            # Title flags
            curr_title = profile.get("current_title", "")
            if not curr_title and valid_roles:
                curr_title = valid_roles[-1]["title"]
                
            senior_flag = check_title(RE_SENIOR, curr_title)
            exec_flag = check_title(RE_EXEC, curr_title)
            
            # 6. Career Growth Score
            cg_score = promo_count + senior_flag + exec_flag
            
            # 7. Company Loyalty Features
            myc_count = sum([1 for t in company_tenures if t >= 36])
            lte_months = sum([t for t in company_tenures if t >= 36])
            lte_ratio = lte_months / total_months if total_months > 0 else 0
            
            # 8. Career Consistency (Gaps)
            tl_gap_count = 0
            lg_months = 0
            
            max_end_so_far = None
            for r in valid_roles:
                if max_end_so_far and r["start_date"]:
                    if r["start_date"] > max_end_so_far:
                        diff = relativedelta(r["start_date"], max_end_so_far)
                        gap_m = diff.years * 12 + diff.months
                        if gap_m > 6:
                            tl_gap_count += 1
                        lg_months = max(lg_months, gap_m)
                        
                if r["end_date"]:
                    if not max_end_so_far or r["end_date"] > max_end_so_far:
                        max_end_so_far = r["end_date"]
                        
            # 9. Aggregate Stability Score
            cs_score = avg_tenure + myc_count - short_tc
            
            # Append all
            candidate_ids.append(c_id)
            total_experience_years.append(t_exp_years)
            num_companies.append(n_companies)
            num_roles.append(n_roles)
            avg_tenure_months.append(avg_tenure)
            longest_tenure_months.append(long_tenure)
            shortest_tenure_months.append(short_tenure)
            job_hopping_scores.append(jh_score)
            short_tenure_counts.append(short_tc)
            very_short_tenure_counts.append(v_short_tc)
            promotion_counts.append(promo_count)
            promotion_velocities.append(promo_vel)
            career_growth_scores.append(cg_score)
            multi_year_company_counts.append(myc_count)
            long_term_employment_ratios.append(lte_ratio)
            timeline_gap_counts.append(tl_gap_count)
            largest_gap_months.append(lg_months)
            career_stability_scores.append(cs_score)
            timeline_quality_scores.append(t_qual)

    print("\n--- Validation Stats ---")
    print(f"Total roles processed: {total_roles_processed}")
    if total_roles_processed > 0:
        print(f"Missing start dates: {missing_starts} ({(missing_starts/total_roles_processed)*100:.2f}%)")
        print(f"Missing end dates (not current): {missing_ends} ({(missing_ends/total_roles_processed)*100:.2f}%)")
    
    df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "total_experience_years": total_experience_years,
        "num_companies": num_companies,
        "num_roles": num_roles,
        "avg_tenure_months": avg_tenure_months,
        "longest_tenure_months": longest_tenure_months,
        "shortest_tenure_months": shortest_tenure_months,
        "job_hopping_score": job_hopping_scores,
        "short_tenure_count": short_tenure_counts,
        "very_short_tenure_count": very_short_tenure_counts,
        "promotion_count": promotion_counts,
        "promotion_velocity": promotion_velocities,
        "career_growth_score": career_growth_scores,
        "multi_year_company_count": multi_year_company_counts,
        "long_term_employment_ratio": long_term_employment_ratios,
        "timeline_gap_count": timeline_gap_counts,
        "largest_gap_months": largest_gap_months,
        "career_stability_score": career_stability_scores,
        "timeline_quality_score": timeline_quality_scores
    })

    print("\n--- Validation on Sample ---")
    print(df[["total_experience_years", "avg_tenure_months", "promotion_count", "career_growth_score", "career_stability_score"]].describe())
    
    print("\n--- Top 20 Candidates by Career Growth Score ---")
    top_growth = df.sort_values("career_growth_score", ascending=False).head(20)
    print(top_growth[["candidate_id", "promotion_count", "career_growth_score", "num_roles", "total_experience_years"]])

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved career progression features to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
