import json
from pathlib import Path

import numpy as np
import pandas as pd
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

JD_EMBEDDING_PATH = (
    BASE_DIR
    / "artifacts"
    / "jd_embedding.npy"
)

OUTPUT_PATH = (
    BASE_DIR
    / "artifacts"
    / "career_semantic_features.parquet"
)

# =====================================================
# TEXT BUILDER
# =====================================================

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

# =====================================================
# MAIN ROUTINE
# =====================================================

def main():
    print("Loading JD embedding...")
    jd_embedding = np.load(JD_EMBEDDING_PATH)
    
    print("Loading MiniLM model...")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    BATCH_SIZE = 1000

    candidate_ids = []
    career_semantic_scores = []
    
    texts_batch = []
    ids_batch = []

    print("Processing candidates and computing career semantic scores...")

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building Texts & Matching"):
            candidate = json.loads(line)
            
            c_id = candidate.get("candidate_id")
            text = build_career_history_text(candidate)
            
            texts_batch.append(text)
            ids_batch.append(c_id)
            
            if len(texts_batch) == BATCH_SIZE:
                # Generate embeddings for batch
                batch_embeddings = model.encode(
                    texts_batch,
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Compute scores
                # jd_embedding is shape (1, 384) or (384,)
                # If jd_embedding is (384,), dot product works directly:
                batch_scores = np.dot(batch_embeddings, jd_embedding.flatten())
                
                career_semantic_scores.extend(batch_scores.tolist())
                candidate_ids.extend(ids_batch)
                
                texts_batch = []
                ids_batch = []

        # Last Batch
        if texts_batch:
            batch_embeddings = model.encode(
                texts_batch,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            batch_scores = np.dot(batch_embeddings, jd_embedding.flatten())
            career_semantic_scores.extend(batch_scores.tolist())
            candidate_ids.extend(ids_batch)

    # =====================================================
    # CREATE DATAFRAME
    # =====================================================
    
    df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "career_semantic_score": career_semantic_scores
    })

    # =====================================================
    # SAVE
    # =====================================================
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("\nCareer Semantic Matching Complete")
    print(f"DataFrame Shape: {df.shape}")
    print("\nTop 5 Rows:")
    print(df.head())
    print("\nScore Statistics:")
    print(df["career_semantic_score"].describe())
    print(f"\nSaved -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
