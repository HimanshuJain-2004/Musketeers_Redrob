import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATES_PATH = BASE_DIR / "data" / "candidates.jsonl"
JD_PATH = BASE_DIR / "artifacts" / "jd_requirements.json"
OUTPUT_PATH = BASE_DIR / "artifacts" / "candidate_features.parquet"

# =====================================================
# LOAD JD
# =====================================================

with open(JD_PATH, "r", encoding="utf-8") as f:
    JD = json.load(f)

# =====================================================
# TITLE SCORE MAP
# =====================================================

TITLE_SCORES = {
    "machine learning engineer": 1.0,
    "ml engineer": 1.0,
    "ai engineer": 1.0,
    "search engineer": 1.0,
    "retrieval engineer": 1.0,
    "ranking engineer": 1.0,
    "recommendation engineer": 1.0,
    "applied scientist": 0.9,
    "data scientist": 0.8,
    "software engineer": 0.5,
    "backend engineer": 0.5,
    "marketing manager": 0.0,
    "operations manager": 0.0,
    "accountant": 0.0
}

# =====================================================
# CONSULTING COMPANIES
# =====================================================

CONSULTING = {
    "tcs",
    "infosys",
    "wipro",
    "cognizant",
    "capgemini",
    "accenture",
    "hcl",
    "mindtree"
}

# =====================================================
# EDUCATION SCORE
# =====================================================

TIER_SCORE = {
    "tier_1": 4,
    "tier_2": 3,
    "tier_3": 2,
    "tier_4": 1,
    "unknown": 0
}

# =====================================================
# SKILL ALIAS MAP
# =====================================================

SKILL_ALIAS = {

    "python": [
        "python"
    ],

    "embeddings": [
        "embeddings",
        "sentence transformers",
        "bge",
        "e5"
    ],

    "vector databases": [
        "milvus",
        "pinecone",
        "qdrant",
        "weaviate",
        "faiss"
    ],

    "retrieval systems": [
        "retrieval",
        "rag",
        "information retrieval"
    ],

    "ranking systems": [
        "ranking",
        "learning to rank",
        "recommendation systems"
    ],

    "llm fine tuning": [
        "lora",
        "qlora",
        "peft",
        "fine-tuning llms"
    ],

    "distributed systems": [
        "spark",
        "kafka",
        "beam"
    ]
}

# =====================================================
# JD FEATURE DICTIONARIES
# =====================================================

RETRIEVAL_TERMS = [
    re.compile(r"\bretrieval\b"),
    re.compile(r"\bsemantic search\b"),
    re.compile(r"\bvector search\b"),
    re.compile(r"\bdense retrieval\b"),
    re.compile(r"\bhybrid retrieval\b"),
    re.compile(r"\binformation retrieval\b"),
    re.compile(r"\bsearch engine\b"),
    re.compile(r"\bcandidate search\b"),
    re.compile(r"\bdocument retrieval\b"),
    re.compile(r"\bembedding retrieval\b")
]

RANKING_TERMS = [
    re.compile(r"\branking\b"),
    re.compile(r"\bre-ranking\b"),
    re.compile(r"\blearning[- ]to[- ]rank\b"),
    re.compile(r"\bltr\b"),
    re.compile(r"\brelevance ranking\b"),
    re.compile(r"\bfeed ranking\b"),
    re.compile(r"\bcandidate ranking\b"),
    re.compile(r"\bsearch ranking\b")
]

RECOMMENDATION_TERMS = [
    re.compile(r"\brecommendation\b"),
    re.compile(r"\brecommendation engine\b"),
    re.compile(r"\brecommender\b"),
    re.compile(r"\bpersonalization\b"),
    re.compile(r"\bcontent recommendation\b"),
    re.compile(r"\bproduct recommendation\b")
]

MATCHING_TERMS = [
    re.compile(r"\bmatching\b"),
    re.compile(r"\bcandidate matching\b"),
    re.compile(r"\bjob matching\b"),
    re.compile(r"\bentity matching\b"),
    re.compile(r"\bsemantic matching\b")
]

EVALUATION_TERMS = [
    re.compile(r"\bndcg\b"),
    re.compile(r"\bmrr\b"),
    re.compile(r"\bmap\b"),
    re.compile(r"\bab testing\b"),
    re.compile(r"\bonline experiment\b"),
    re.compile(r"\boffline evaluation\b"),
    re.compile(r"\brelevance metrics\b"),
    re.compile(r"\bevaluation framework\b")
]

PRODUCTION_TERMS = [
    re.compile(r"\bproduction\b"),
    re.compile(r"\bdeployed\b"),
    re.compile(r"\bdeployment\b"),
    re.compile(r"\bserving\b"),
    re.compile(r"\binference\b"),
    re.compile(r"\bmonitoring\b"),
    re.compile(r"\bscaling\b"),
    re.compile(r"\blatency\b"),
    re.compile(r"\bonline system\b"),
    re.compile(r"\bproductionized\b")
]

# =====================================================
# EXPERIENCE SCORE
# =====================================================

def experience_score(exp):

    if 5 <= exp <= 9:
        return 1.0

    elif 4 <= exp < 5:
        return 0.8

    elif 9 < exp <= 12:
        return 0.7

    else:
        return 0.3

# =====================================================
# TITLE SCORE
# =====================================================

def title_score(title):

    title = title.lower()

    for key, score in TITLE_SCORES.items():

        if key in title:
            return score

    return 0.2

# =====================================================
# LOCATION SCORE
# =====================================================

def location_score(location):

    location = location.lower()

    for loc in JD["preferred_locations"]:

        if loc in location:
            return 1

    return 0

# =====================================================
# EDUCATION SCORE
# =====================================================

def education_score(education):

    if not education:
        return 0

    best = 0

    for edu in education:

        tier = edu.get("tier", "unknown")

        best = max(
            best,
            TIER_SCORE.get(tier, 0)
        )

    return best

# =====================================================
# SKILL FEATURES
# =====================================================

def skill_features(skills):

    candidate_skills = {
        skill.get("name", "").lower()
        for skill in skills
    }

    matched = 0

    for req_skill in JD["must_have_skills"]:

        aliases = SKILL_ALIAS.get(
            req_skill,
            [req_skill]
        )

        found = False

        for alias in aliases:

            if alias.lower() in candidate_skills:
                found = True
                break

        if found:
            matched += 1

    coverage = matched / max(
        len(JD["must_have_skills"]),
        1
    )

    return matched, coverage

# =====================================================
# PRODUCT COMPANY SCORE
# =====================================================

def product_company_score(history):

    score = 0

    keywords = {
        "startup",
        "saas",
        "product",
        "marketplace"
    }

    for job in history:

        industry = job.get(
            "industry",
            ""
        ).lower()

        for keyword in keywords:

            if keyword in industry:
                score += 1

    return min(score / 3, 1.0)

# =====================================================
# CONSULTING PENALTY
# =====================================================

def consulting_penalty(history):

    if not history:
        return 0

    consulting_count = 0

    for job in history:

        company = job.get(
            "company",
            ""
        ).lower()

        if any(
            c in company
            for c in CONSULTING
        ):
            consulting_count += 1

    if consulting_count == len(history):
        return 1

    return 0

# =====================================================
# RETRIEVAL SCORE (OLD)
# =====================================================

def retrieval_score_old(history):
    keywords = {"retrieval", "search", "ranking", "recommendation", "matching", "vector", "embedding", "relevance"}
    score = 0
    for job in history:
        desc = job.get("description", "").lower()
        for keyword in keywords:
            if keyword in desc:
                score += 1
    return min(score / 5, 1.0)

# =====================================================
# YEARS AND COUNTS
# =====================================================

def product_company_years(history):
    years = 0
    keywords = {"startup", "saas", "product", "marketplace"}
    for job in history:
        industry = job.get("industry", "").lower()
        if any(keyword in industry for keyword in keywords):
            years += job.get("duration_months", 0) / 12.0
    return round(years, 2)

def consulting_company_years(history):
    years = 0
    for job in history:
        company = job.get("company", "").lower()
        if any(c in company for c in CONSULTING):
            years += job.get("duration_months", 0) / 12.0
    return round(years, 2)

# =====================================================
# GRANULAR JD FEATURES
# =====================================================

def extract_granular_features(history):

    retrieval_count = 0
    ranking_count = 0
    recommendation_count = 0
    matching_count = 0
    evaluation_count = 0
    production_count = 0

    for job in history:
        text = job.get("description", "").lower()

        for pattern in RETRIEVAL_TERMS:
            if pattern.search(text):
                retrieval_count += 1
                
        for pattern in RANKING_TERMS:
            if pattern.search(text):
                ranking_count += 1
                
        for pattern in RECOMMENDATION_TERMS:
            if pattern.search(text):
                recommendation_count += 1
                
        for pattern in MATCHING_TERMS:
            if pattern.search(text):
                matching_count += 1
                
        for pattern in EVALUATION_TERMS:
            if pattern.search(text):
                evaluation_count += 1
                
        for pattern in PRODUCTION_TERMS:
            if pattern.search(text):
                production_count += 1

    jd_core_score = (
        retrieval_count +
        ranking_count +
        recommendation_count +
        matching_count +
        evaluation_count +
        production_count
    )

    return {
        "retrieval_count": retrieval_count,
        "retrieval_present": 1 if retrieval_count > 0 else 0,
        "ranking_count": ranking_count,
        "ranking_present": 1 if ranking_count > 0 else 0,
        "recommendation_count": recommendation_count,
        "recommendation_present": 1 if recommendation_count > 0 else 0,
        "matching_count": matching_count,
        "matching_present": 1 if matching_count > 0 else 0,
        "evaluation_count": evaluation_count,
        "evaluation_present": 1 if evaluation_count > 0 else 0,
        "production_count": production_count,
        "production_present": 1 if production_count > 0 else 0,
        "jd_core_score": jd_core_score
    }

# =====================================================
# MAIN
# =====================================================

rows = []

with open(
    CANDIDATES_PATH,
    "r",
    encoding="utf-8"
) as f:

    count = 0
    for line in tqdm(
        f,
        desc="Processing Candidates"
    ):

        candidate = json.loads(line)

        profile = candidate.get("profile", {})

        years_exp = profile.get(
            "years_of_experience",
            0
        )

        skill_match_count, skill_coverage = (
            skill_features(
                candidate.get(
                    "skills",
                    []
                )
            )
        )

        row = {

            "candidate_id":
                candidate["candidate_id"],

            "years_exp":
                years_exp,

            "experience_score":
                experience_score(
                    years_exp
                ),

            "skill_match_count":
                skill_match_count,

            "skill_coverage":
                round(skill_coverage, 4),

            "title_score":
                title_score(
                    profile.get(
                        "current_title",
                        ""
                    )
                ),

            "education_score":
                education_score(
                    candidate.get(
                        "education",
                        []
                    )
                ),

            "location_score":
                location_score(
                    profile.get(
                        "location",
                        ""
                    )
                ),

            "product_company_score":
                product_company_score(
                    candidate.get("career_history", [])
                ),

            "product_company_years":
                product_company_years(
                    candidate.get("career_history", [])
                ),

            "consulting_penalty":
                consulting_penalty(
                    candidate.get("career_history", [])
                ),

            "consulting_company_years":
                consulting_company_years(
                    candidate.get("career_history", [])
                ),

            "career_history_length":
                len(candidate.get("career_history", [])),

            "current_role_matches_jd":
                1 if title_score(profile.get("current_title", "")) >= 0.9 else 0,

            "retrieval_score_old":
                retrieval_score_old(
                    candidate.get("career_history", [])
                )
        }

        granular_features = extract_granular_features(
            candidate.get("career_history", [])
        )
        row.update(granular_features)

        rows.append(row)

df = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\nFeature Generation Complete")
print(df.shape)
print(df.head())
print(f"\nSaved -> {OUTPUT_PATH}")

