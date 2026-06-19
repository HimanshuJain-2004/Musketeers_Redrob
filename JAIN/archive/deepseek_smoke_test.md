# DeepSeek Smoke Test Results

## Candidate: CAND_0004599
**Gemini Original Label**: Reject (Honeypot: True)
**Latency**: 13.19s
**HTTP Status**: 200

### Raw Response
```json
[
  {
    "candidate_id": "CAND_0004599",
    "technical_fit": 0,
    "product_fit": 0,
    "behavioral_fit": 0,
    "career_fit": 0,
    "fit_score": 5,
    "fit_label": "Reject",
    "honeypot_probability": 0.95,
    "honeypot_label": true,
    "confidence": 0.98,
    "reasoning": "Candidate profile exhibits severe contradictions and fabrication signals. Headline states 'Accountant' but summary describes 'marketing manager roles.' Work history titles (Accountant, Sales Executive, Civil Engineer, Business Analyst) are completely mismatched with their descriptions (Operations management, Senior accounting, Customer support, Enterprise sales). No evidence of any engineering, software development, ML, AI, retrieval, ranking, or search experience. Entire career is non-technical (operations, accounting, support, sales). Skills list includes FastAPI, Docker, AWS, MongoDB, GCP but zero work history supports any technical engineering role. Education is M.Sc Chemical Engineering from tier_4 institution with no connection to AI/ML. GitHub activity score of 3.7 confirms no coding evidence. This is a clear honeypot profile with fabricated/contradictory job titles and descriptions, zero relevant experience, and impossible skill claims unsupported by career history. Hard reject per mandatory rule for entirely non-technical background with no engineering evidence."
  }
]
```
**JSON Parse Status**: SUCCESS
---
## Candidate: CAND_0095619
**Gemini Original Label**: Excellent Fit (Honeypot: False)
**Latency**: 9.65s
**HTTP Status**: 200

### Raw Response
```json
[
  {
    "technical_fit": 78,
    "product_fit": 82,
    "behavioral_fit": 80,
    "career_fit": 72,
    "fit_score": 78,
    "fit_label": "Strong Fit",
    "honeypot_probability": 0.05,
    "honeypot_label": false,
    "confidence": 0.82,
    "reasoning": "Candidate has 4.2 years of experience, slightly below the 5-9 year range but within acceptable bounds given strong signals. Directly relevant experience: owned ranking layer for e-commerce search at Nykaa (product company), evolved from hand-tuned scoring to learning-to-rank model, designed relevance labeling pipeline, feature pipeline, and training/eval workflow. Improved revenue-per-search by 12% — clear production impact. Skills include Learning to Rank, Information Retrieval, BM25, Sentence Transformers, Pinecone, Weaviate, Fine-tuning LLMs, and Hugging Face Transformers — strong alignment with retrieval, ranking, embeddings, and vector search requirements. Shipped RAG-based feature and owns eval framework. M.Sc in Machine Learning from Tier-1 institution. Active on platform (last active April 2026), high recruiter response rate (90%), 30-day notice period, hybrid work preference. Weaknesses: only 4.2 years experience (below 5-year minimum), no explicit mention of NDCG/MRR/MAP evaluation metrics in work history (though eval framework ownership is mentioned), GitHub activity score is low (29.7), not willing to relocate (but Pune/Noida is flexible per JD). No consulting-only background, no pure research, no CV/speech/robotics focus. Product-company experience at Nykaa is a strong positive. Overall a strong fit for the role with minor experience gap."
  }
]
```
**JSON Parse Status**: SUCCESS
---
