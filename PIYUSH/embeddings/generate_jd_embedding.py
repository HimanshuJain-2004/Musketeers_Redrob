from pathlib import Path

import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

JD_PATH = BASE_DIR / "data" / "job_description.docx"

OUTPUT_PATH = (
    BASE_DIR
    / "artifacts"
    / "jd_embedding.npy"
)

# =====================================================
# LOAD JD
# =====================================================

doc = Document(JD_PATH)

jd_text = "\n".join(
    para.text
    for para in doc.paragraphs
    if para.text.strip()
)

print("Loading MiniLM model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("Generating JD embedding...")

embedding = model.encode(
    jd_text,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(
    OUTPUT_PATH,
    embedding
)

print("Saved:", OUTPUT_PATH)
print("Shape:", embedding.shape)