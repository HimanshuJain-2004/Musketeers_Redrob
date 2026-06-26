import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_EMBEDDINGS_PATH = (
    BASE_DIR
    / "artifacts"
    / "candidate_embeddings.npy"
)

JD_EMBEDDING_PATH = (
    BASE_DIR
    / "artifacts"
    / "jd_embedding.npy"
)

CANDIDATES_PATH = (
    BASE_DIR
    / "data"
    / "candidates.jsonl"
)

OUTPUT_PATH = (
    BASE_DIR
    / "artifacts"
    / "semantic_scores.parquet"
)

# =====================================================
# LOAD EMBEDDINGS
# =====================================================

print("Loading embeddings...")

candidate_embeddings = np.load(
    CANDIDATE_EMBEDDINGS_PATH
)

jd_embedding = np.load(
    JD_EMBEDDING_PATH
)

print(
    "Candidate Embeddings Shape:",
    candidate_embeddings.shape
)

print(
    "JD Embedding Shape:",
    jd_embedding.shape
)

# =====================================================
# COSINE SIMILARITY
# =====================================================
#
# Since embeddings were generated with:
#
# normalize_embeddings=True
#
# cosine similarity = dot product
#
# =====================================================

print("Computing cosine similarity...")

semantic_scores = np.dot(
    candidate_embeddings,
    jd_embedding
)

# =====================================================
# LOAD CANDIDATE IDS
# =====================================================

print("Loading candidate IDs...")

candidate_ids = []

with open(
    CANDIDATES_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in tqdm(
        f,
        desc="Loading Candidate IDs"
    ):

        candidate = json.loads(
            line
        )

        candidate_ids.append(
            candidate[
                "candidate_id"
            ]
        )

# =====================================================
# VALIDATION
# =====================================================

assert len(candidate_ids) == len(
    semantic_scores
), (
    "Mismatch between candidate count "
    "and embedding count"
)

# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame({

    "candidate_id":
        candidate_ids,

    "semantic_score":
        semantic_scores
})

# =====================================================
# SAVE
# =====================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\nSemantic Matching Complete")

print(
    "\nDataFrame Shape:",
    df.shape
)

print(
    "\nTop 5 Rows:"
)

print(
    df.head()
)

print(
    "\nScore Statistics:"
)

print(
    df["semantic_score"]
    .describe()
)

print(
    f"\nSaved -> {OUTPUT_PATH}"
)