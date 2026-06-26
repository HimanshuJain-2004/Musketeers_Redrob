import json
import re
from pathlib import Path
from docx import Document

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

JD_PATH = BASE_DIR / "data" / "job_description.docx"
OUTPUT_PATH = BASE_DIR / "artifacts" / "jd_requirements.json"

# =====================================================
# SKILL MAP
# =====================================================

JD_SKILL_MAP = {
    "python": [
        "python"
    ],

    "embeddings": [
        "embedding",
        "embeddings",
        "sentence-transformers",
        "openai embeddings",
        "bge",
        "e5"
    ],

    "vector databases": [
        "pinecone",
        "milvus",
        "qdrant",
        "weaviate",
        "faiss",
        "elasticsearch",
        "opensearch"
    ],

    "retrieval systems": [
        "retrieval",
        "information retrieval"
    ],

    "ranking systems": [
        "ranking",
        "learning-to-rank"
    ],

    "hybrid search": [
        "hybrid search",
        "bm25",
        "dense retrieval"
    ],

    "evaluation frameworks": [
        "evaluation",
        "ndcg",
        "mrr",
        "map",
        "a/b testing"
    ],

    "llm fine tuning": [
        "lora",
        "qlora",
        "peft",
        "fine-tuning"
    ],

    "distributed systems": [
        "distributed systems",
        "large-scale inference"
    ],

    "mlops": [
        "mlops"
    ]
}

# =====================================================
# LOAD JD FROM DOCX
# =====================================================

def load_jd():
    doc = Document(JD_PATH)

    text = []

    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text)

    return "\n".join(text)

# =====================================================
# EXPERIENCE EXTRACTION
# =====================================================

def extract_experience(text):

    match = re.search(
        r"Experience Required:\s*(\d+)\s*[-–]\s*(\d+)",
        text,
        re.IGNORECASE
    )

    if match:
        return {
            "min": int(match.group(1)),
            "max": int(match.group(2))
        }

    return {
        "min": None,
        "max": None
    }

# =====================================================
# SKILL EXTRACTION
# =====================================================

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill, keywords in JD_SKILL_MAP.items():

        for keyword in keywords:

            if keyword.lower() in text:
                found_skills.append(skill)
                break

    return sorted(list(set(found_skills)))

# =====================================================
# LOCATION EXTRACTION
# =====================================================

def extract_locations(text):

    text = text.lower()

    locations = []

    location_keywords = [
        "pune",
        "noida",
        "hyderabad",
        "mumbai",
        "delhi",
        "delhi ncr"
    ]

    for location in location_keywords:

        if location in text:
            locations.append(location)

    return sorted(list(set(locations)))

# =====================================================
# BUILD JD REQUIREMENTS
# =====================================================

def build_jd_requirements():

    print("Loading JD...")

    jd_text = load_jd()

    print(f"JD Loaded ({len(jd_text)} characters)")

    requirements = {

        "role": "Senior AI Engineer",

        "experience_range":
            extract_experience(jd_text),

        "must_have_skills":
            extract_skills(jd_text),

        "preferred_skills": [
            "lora",
            "qlora",
            "peft",
            "learning to rank",
            "distributed systems",
            "mlops",
            "hr tech",
            "marketplace systems"
        ],

        "must_have_titles": [
            "machine learning engineer",
            "ml engineer",
            "ai engineer",
            "retrieval engineer",
            "search engineer",
            "ranking engineer",
            "recommendation engineer",
            "applied scientist",
            "senior ai engineer"
        ],

        "must_have_domains": [
            "retrieval",
            "search",
            "ranking",
            "recommendation",
            "matching systems"
        ],

        "preferred_domains": [
            "hr tech",
            "recruiting",
            "marketplace"
        ],

        "preferred_locations":
            extract_locations(jd_text),

        "company_preferences": [
            "product company",
            "startup",
            "saas"
        ],

        "negative_signals": [
            "pure_research",
            "only_langchain",
            "no_production_ml",
            "only_consulting_background",
            "computer_vision_only",
            "speech_only",
            "robotics_only",
            "inactive_candidate",
            "low_recruiter_response_rate"
        ]
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            requirements,
            f,
            indent=4
        )

    print("\nJD Parsed Successfully")
    print(f"Output saved at: {OUTPUT_PATH}")

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    build_jd_requirements()