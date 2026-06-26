import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATES_PATH = (
    BASE_DIR
    / "data"
    / "candidates.jsonl"
)

OUTPUT_PATH = (
    BASE_DIR
    / "artifacts"
    / "candidate_embeddings.npy"
)

# =====================================================
# MODEL
# =====================================================

print("Loading MiniLM model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# =====================================================
# TEXT BUILDER
# =====================================================

def build_candidate_text(candidate):

    profile = candidate["profile"]

    headline = profile.get(
        "headline",
        ""
    )

    summary = profile.get(
        "summary",
        ""
    )

    current_title = profile.get(
        "current_title",
        ""
    )

    skills = ", ".join(
        skill["name"]
        for skill in candidate.get(
            "skills",
            []
        )
    )

    career_text = " ".join(

        job.get(
            "description",
            ""
        )

        for job in candidate.get(
            "career_history",
            []
        )
    )

    education_text = " ".join(

        edu.get(
            "field_of_study",
            ""
        )

        for edu in candidate.get(
            "education",
            []
        )
    )

    text = f"""
    Headline: {headline}

    Summary:
    {summary}

    Current Title:
    {current_title}

    Skills:
    {skills}

    Career History:
    {career_text}

    Education:
    {education_text}
    """

    return text


# =====================================================
# EMBEDDING GENERATION
# =====================================================

BATCH_SIZE = 1000

all_embeddings = []

texts = []

print("Reading candidates...")

with open(
    CANDIDATES_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in tqdm(
        f,
        desc="Building Text"
    ):

        candidate = json.loads(line)

        texts.append(
            build_candidate_text(
                candidate
            )
        )

        if len(texts) == BATCH_SIZE:

            batch_embeddings = model.encode(
                texts,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            all_embeddings.append(
                batch_embeddings
            )

            texts = []

# Last Batch

if texts:

    batch_embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    all_embeddings.append(
        batch_embeddings
    )

# =====================================================
# MERGE
# =====================================================

embeddings = np.vstack(
    all_embeddings
)

# =====================================================
# SAVE
# =====================================================

np.save(
    OUTPUT_PATH,
    embeddings
)

print("\nDone")
print("Shape:", embeddings.shape)
print("Saved:", OUTPUT_PATH)