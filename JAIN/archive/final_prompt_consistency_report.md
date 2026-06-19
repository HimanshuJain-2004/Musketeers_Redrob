# Final Prompt Consistency Report

Generated: 2026-06-15 23:46 UTC

---

## Summary

| Metric                       | Value |
|------------------------------|-------|
| Candidates evaluated         | 18 |
| Candidates audited           | 18 |
| Avg recruiter fit_score      | 37.9 |
| Avg audit quality score      | 86.1 |
| Label agreements (audit)     | 17 / 18 |
| Honeypot agreements (audit)  | 18 / 18 |
| Expected label correct       | 12 / 18 (66%) |
| Expected honeypot correct    | 15 / 18 (83%) |

### Fit Label Distribution

| Label          | Count |
|----------------|-------|
| Excellent Fit  |     4 |
| Moderate Fit   |     2 |
| Weak Fit       |     1 |
| Reject         |    11 |

Honeypot candidates detected: **0** / 18

---

## Final Verdict

### ⚠️ PROMPT REQUIRES FURTHER CALIBRATION

---

## Category Breakdown

### Category A — Excellent Fit (should score 70–100, no honeypot)

| Candidate ID | Expected | Actual Score | Actual Label | ✓ Label | Expected HP | Actual HP | ✓ HP | Audit Score | Quality |
|-------------|----------|-------------|--------------|---------|-------------|-----------|------|-------------|---------|
| CAND_0093912 | Excellent Fit | 90 | Excellent Fit | ✅ | False | False | ✅ | 95 | Excellent |
| CAND_0098846 | Excellent Fit | 92 | Excellent Fit | ✅ | False | False | ✅ | 95 | Excellent |
| CAND_0061175 | Strong Fit | 65 | Moderate Fit | ❌ | False | False | ✅ | 80 | Good |

### Category B — Strong / Moderate Technical (should score 50–84, no honeypot)

| Candidate ID | Expected | Actual Score | Actual Label | ✓ Label | Expected HP | Actual HP | ✓ HP | Audit Score | Quality |
|-------------|----------|-------------|--------------|---------|-------------|-----------|------|-------------|---------|
| CAND_0011925 | Moderate Fit | 40 | Moderate Fit | ✅ | False | False | ✅ | 80 | Good |
| CAND_0014710 | Moderate Fit | 30 | Weak Fit | ❌ | False | False | ✅ | 80 | Good |
| CAND_0095619 | Strong Fit | 90 | Excellent Fit | ✅ | False | False | ✅ | 95 | Excellent |

### Category C — Weak but Legitimate Engineers (should score 25–49, no honeypot)

| Candidate ID | Expected | Actual Score | Actual Label | ✓ Label | Expected HP | Actual HP | ✓ HP | Audit Score | Quality |
|-------------|----------|-------------|--------------|---------|-------------|-----------|------|-------------|---------|
| CAND_0012375 | Weak Fit | 20 | Reject | ❌ | False | False | ✅ | 90 | Good |
| CAND_0088645 | Weak Fit | 30 | Reject | ❌ | False | False | ✅ | 80 | Good |
| CAND_0044222 | Strong Fit | 90 | Excellent Fit | ✅ | False | False | ✅ | 95 | Excellent |

### Category D — Career Transition (should score 0–49, no honeypot)

| Candidate ID | Expected | Actual Score | Actual Label | ✓ Label | Expected HP | Actual HP | ✓ HP | Audit Score | Quality |
|-------------|----------|-------------|--------------|---------|-------------|-----------|------|-------------|---------|
| CAND_0076006 | Reject | 10 | Reject | ✅ | False | False | ✅ | 90 | Good |
| CAND_0040259 | Reject | 15 | Reject | ✅ | False | False | ✅ | 80 | Good |
| CAND_0013264 | Weak Fit | 20 | Reject | ❌ | False | False | ✅ | 80 | Good |

### Category E — Honeypot Candidates (should score 0–24, honeypot=true)

| Candidate ID | Expected | Actual Score | Actual Label | ✓ Label | Expected HP | Actual HP | ✓ HP | Audit Score | Quality |
|-------------|----------|-------------|--------------|---------|-------------|-----------|------|-------------|---------|
| CAND_0031820 | Reject | 10 | Reject | ✅ | True | False | ❌ | 90 | Excellent |
| CAND_0026016 | Reject | 15 | Reject | ✅ | True | False | ❌ | 90 | Good |
| CAND_0058807 | Reject | 20 | Reject | ✅ | True | False | ❌ | 90 | Good |

### Category F — Borderline (mixed expectations)

| Candidate ID | Expected | Actual Score | Actual Label | ✓ Label | Expected HP | Actual HP | ✓ HP | Audit Score | Quality |
|-------------|----------|-------------|--------------|---------|-------------|-----------|------|-------------|---------|
| CAND_0021768 | Reject | 10 | Reject | ✅ | False | False | ✅ | 80 | Good |
| CAND_0053507 | Reject | 15 | Reject | ✅ | False | False | ✅ | 80 | Good |
| CAND_0048271 | Weak Fit | 20 | Reject | ❌ | False | False | ✅ | 80 | Good |

---

## Detailed Reasoning per Candidate

### CAND_0093912  [Excellent Fit]

- **Expected**: Excellent Fit / honeypot=False
- **Recruiter**: score=90 / label=Excellent Fit / honeypot=False ✅ ✅
- **Recruiter Reasoning**: The candidate has strong experience in building ML-powered features in production, with a focus on NLP, recommendation systems, and applied AI. They have worked on ranking, retrieval, and matching systems, and have experience with embeddings, vector search, and evaluation frameworks.
- **Audit Score**: 95 / quality=Excellent
- **Missed Strengths**: ['Experience with semantic search', 'Hands-on experience with sentence-transformers and FAISS', 'Strong background in CS/ML']
- **Missed Weaknesses**: ['Limited experience with LLM fine-tuning', 'No open-source contributions mentioned']
- **Recommended Score**: 90 / label=Excellent Fit
- **Audit Reasoning**: The recruiter's evaluation is thorough and well-justified. The candidate's experience in building ML-powered features, NLP, recommendation systems, and applied AI aligns well with the job requirements. The candidate's hands-on experience with embeddings, vector search, and evaluation frameworks is a

### CAND_0098846  [Excellent Fit]

- **Expected**: Excellent Fit / honeypot=False
- **Recruiter**: score=92 / label=Excellent Fit / honeypot=False ✅ ✅
- **Recruiter Reasoning**: The candidate has extensive experience in building ML-powered features in production, with a strong background in NLP, recommendation systems, and applied AI. They have worked on ranking, retrieval, and matching systems, and have experience with embeddings, vector search, and evaluation frameworks. 
- **Audit Score**: 95 / quality=Excellent
- **Missed Strengths**: []
- **Missed Weaknesses**: []
- **Recommended Score**: 92 / label=Excellent Fit
- **Audit Reasoning**: The recruiter's evaluation is thorough and well-justified. The candidate's experience in building ML-powered features in production, particularly in ranking, retrieval, and matching systems, aligns well with the job requirements. The candidate's background in NLP, recommendation systems, and applied

### CAND_0061175  [Excellent Fit]

- **Expected**: Strong Fit / honeypot=False
- **Recruiter**: score=65 / label=Moderate Fit / honeypot=False ❌ ✅
- **Recruiter Reasoning**: The candidate has some experience in building ML-powered features in production, but their focus has been more on the modeling and analysis side. They have some experience with recommendation systems and NLP, but lack experience with ranking, retrieval, and matching systems. They are looking to grow
- **Audit Score**: 80 / quality=Good
- **Missed Strengths**: ['Production experience with recommendation systems', 'Experience with vector search and Milvus', 'Strong Python skills']
- **Missed Weaknesses**: ['Lack of experience with ranking, retrieval, and matching systems', 'Limited experience with LLMs and fine-tuning']
- **Recommended Score**: 60 / label=Moderate Fit
- **Audit Reasoning**: The recruiter's evaluation is mostly correct, but misses some key strengths and weaknesses of the candidate. The candidate has relevant experience in building ML-powered features in production, but lacks experience in ranking, retrieval, and matching systems. The recommended fit score is slightly lo

### CAND_0011925  [Strong/Moderate Technical]

- **Expected**: Moderate Fit / honeypot=False
- **Recruiter**: score=40 / label=Moderate Fit / honeypot=False ✅ ✅
- **Recruiter Reasoning**: The candidate has a strong background in data engineering, but lacks direct experience in retrieval, ranking, and recommendation systems. They have some relevant skills, but their work history and education do not align closely with the job requirements.
- **Audit Score**: 80 / quality=Good
- **Missed Strengths**: ['Strong data engineering background', 'Experience with data pipelines and analytics infrastructure', 'Interest in transitioning to AI/ML-focused work']
- **Missed Weaknesses**: ['Lack of direct experience in retrieval, ranking, and recommendation systems', 'Limited experience with modern ML practices']
- **Recommended Score**: 30 / label=Weak Fit
- **Audit Reasoning**: The recruiter's evaluation is mostly correct, but the fit label 'Moderate Fit' does not align with the fit score of 40. The candidate's background in data engineering is a strength, but the lack of direct experience in retrieval, ranking, and recommendation systems is a significant weakness. The hon

### CAND_0014710  [Strong/Moderate Technical]

- **Expected**: Moderate Fit / honeypot=False
- **Recruiter**: score=30 / label=Weak Fit / honeypot=False ❌ ✅
- **Recruiter Reasoning**: The candidate has a background in software engineering, but lacks experience in AI, ML, and data science. They have some relevant skills, but their work history and education do not align closely with the job requirements.
- **Audit Score**: 80 / quality=Good
- **Missed Strengths**: ["candidate's experience with Pinecone", "candidate's self-learning efforts in AI/ML"]
- **Missed Weaknesses**: ["candidate's lack of production experience in AI/ML", "candidate's limited experience with retrieval and ranking systems"]
- **Recommended Score**: 30 / label=Weak Fit
- **Audit Reasoning**: The recruiter's evaluation is reasonable, but could benefit from more consideration of the candidate's transferable skills and self-learning efforts. The candidate's lack of direct experience in AI/ML is a significant weakness, but their background in software engineering and willingness to learn ar

### CAND_0095619  [Strong/Moderate Technical]

- **Expected**: Strong Fit / honeypot=False
- **Recruiter**: score=90 / label=Excellent Fit / honeypot=False ✅ ✅
- **Recruiter Reasoning**: The candidate has a strong background in NLP, recommendation systems, and applied AI. They have direct experience in building and deploying ML-powered features, and their work history and education align closely with the job requirements.
- **Audit Score**: 95 / quality=Excellent
- **Missed Strengths**: ['Production experience with vector databases or hybrid search infrastructure', 'Hands-on experience designing evaluation frameworks for ranking systems']
- **Missed Weaknesses**: []
- **Recommended Score**: 95 / label=Excellent Fit
- **Audit Reasoning**: The recruiter's evaluation is well-justified by the candidate's strong background in NLP, recommendation systems, and applied AI, as well as their direct experience in building and deploying ML-powered features. The candidate's work history and education align closely with the job requirements, and 

### CAND_0012375  [Weak Legitimate Engineer]

- **Expected**: Weak Fit / honeypot=False
- **Recruiter**: score=20 / label=Reject / honeypot=False ❌ ✅
- **Recruiter Reasoning**: The candidate lacks direct experience with retrieval, ranking, and recommendation systems. Their background is in QA and backend development, which is not directly relevant to the role.
- **Audit Score**: 90 / quality=Good
- **Missed Strengths**: []
- **Missed Weaknesses**: ['Lack of production experience with embeddings-based retrieval systems', 'No experience with vector databases or hybrid search infrastructure']
- **Recommended Score**: 20 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is justified by the evidence in the candidate profile. The candidate lacks direct experience with retrieval, ranking, and recommendation systems, and their background is in QA and backend development, which is not directly relevant to the role.

### CAND_0088645  [Weak Legitimate Engineer]

- **Expected**: Weak Fit / honeypot=False
- **Recruiter**: score=30 / label=Reject / honeypot=False ❌ ✅
- **Recruiter Reasoning**: The candidate has some experience with relevant technologies like Qdrant, but their overall experience and skills are not strongly aligned with the role. They lack direct experience with retrieval, ranking, and recommendation systems.
- **Audit Score**: 80 / quality=Good
- **Missed Strengths**: ['Qdrant experience']
- **Missed Weaknesses**: ['Lack of production experience with embeddings-based retrieval systems', 'No hands-on experience designing evaluation frameworks for ranking systems']
- **Recommended Score**: 30 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is mostly correct, but missed some strengths and weaknesses. The candidate's experience with Qdrant is a positive signal, but their lack of production experience with embeddings-based retrieval systems and evaluation frameworks is a significant weakness.

### CAND_0044222  [Weak Legitimate Engineer]

- **Expected**: Strong Fit / honeypot=False
- **Recruiter**: score=90 / label=Excellent Fit / honeypot=False ✅ ✅
- **Recruiter Reasoning**: The candidate has strong experience with retrieval, ranking, and recommendation systems, including the development of a semantic search feature and the training and shipping of multiple ranking models. Their skills and experience are highly aligned with the role.
- **Audit Score**: 95 / quality=Excellent
- **Missed Strengths**: ['Strong background in NLP', 'Experience with evaluation rigor', 'Leadership experience in migrating a product to embedding-based retrieval']
- **Missed Weaknesses**: ['Limited experience with LLM fine-tuning', 'No explicit mention of experience with distributed systems or large-scale inference optimization']
- **Recommended Score**: 90 / label=Excellent Fit
- **Audit Reasoning**: The recruiter's evaluation is well-justified by the candidate's strong experience in retrieval, ranking, and recommendation systems, as well as their skills and education. The candidate's background in NLP and experience with evaluation rigor are additional strengths that support the excellent fit l

### CAND_0076006  [Career Transition]

- **Expected**: Reject / honeypot=False
- **Recruiter**: score=10 / label=Reject / honeypot=False ✅ ✅
- **Recruiter Reasoning**: The candidate has no production experience with embeddings-based retrieval systems, vector databases, or hybrid search infrastructure. Their work history is primarily in HR and business analysis, with no evidence of AI engineering experience.
- **Audit Score**: 90 / quality=Good
- **Missed Strengths**: []
- **Missed Weaknesses**: ["No mention of the candidate's recent interest in AI and GenAI, and their efforts to learn about RAG and vector databases", "No consideration of the candidate's potential to grow into the role, given their strong functional expertise and willingness to learn"]
- **Recommended Score**: 10 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is justified by the evidence in the candidate profile, which shows a lack of production experience with embeddings-based retrieval systems, vector databases, or hybrid search infrastructure. However, the recruiter could have considered the candidate's potential for growth 

### CAND_0040259  [Career Transition]

- **Expected**: Reject / honeypot=False
- **Recruiter**: score=15 / label=Reject / honeypot=False ✅ ✅
- **Recruiter Reasoning**: The candidate has some relevant skills, but their work history is primarily in non-technical roles, and they lack production experience with embeddings-based retrieval systems, vector databases, or hybrid search infrastructure.
- **Audit Score**: 80 / quality=Good
- **Missed Strengths**: []
- **Missed Weaknesses**: ['Lack of hands-on experience with evaluation frameworks for ranking systems', 'No evidence of prior experience with LLM fine-tuning or learning-to-rank models']
- **Recommended Score**: 15 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is justified by the candidate's lack of production experience with embeddings-based retrieval systems, vector databases, or hybrid search infrastructure, and their work history being primarily in non-technical roles.

### CAND_0013264  [Career Transition]

- **Expected**: Weak Fit / honeypot=False
- **Recruiter**: score=20 / label=Reject / honeypot=False ❌ ✅
- **Recruiter Reasoning**: The candidate has some relevant skills, but their work history is primarily in civil engineering, and they lack production experience with embeddings-based retrieval systems, vector databases, or hybrid search infrastructure. They are open to work and have a high recruiter response rate, but their f
- **Audit Score**: 80 / quality=Good
- **Missed Strengths**: ["candidate's enthusiasm for AI and GenAI applications", 'completion of online courses on RAG and vector databases']
- **Missed Weaknesses**: ['lack of direct experience in AI/ML roles', 'no production deployments mentioned in the work history']
- **Recommended Score**: 20 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is mostly correct, but it could benefit from a more detailed analysis of the candidate's skills and work history. The candidate's lack of direct experience in AI/ML roles and no production deployments are significant weaknesses. However, their enthusiasm for AI and GenAI a

### CAND_0031820  [Honeypot]

- **Expected**: Reject / honeypot=True
- **Recruiter**: score=10 / label=Reject / honeypot=False ✅ ❌
- **Recruiter Reasoning**: The candidate has no relevant experience in AI engineering, retrieval systems, or search systems. Their background is in business analysis, and their skills are not aligned with the job requirements.
- **Audit Score**: 90 / quality=Excellent
- **Missed Strengths**: []
- **Missed Weaknesses**: []
- **Recommended Score**: 10 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is justified by the evidence in the candidate profile. The candidate lacks relevant experience in AI engineering, retrieval systems, or search systems, and their background is in business analysis. The skills listed are not aligned with the job requirements, supporting the

### CAND_0026016  [Honeypot]

- **Expected**: Reject / honeypot=True
- **Recruiter**: score=15 / label=Reject / honeypot=False ✅ ❌
- **Recruiter Reasoning**: The candidate has no relevant experience in AI engineering, retrieval systems, or search systems. Their background is in mechanical engineering design, and their skills are not aligned with the job requirements.
- **Audit Score**: 90 / quality=Good
- **Missed Strengths**: []
- **Missed Weaknesses**: ['No production experience with embeddings-based retrieval systems', 'No experience with vector databases or hybrid search infrastructure', 'Limited hands-on experience with designing evaluation frameworks for ranking systems']
- **Recommended Score**: 15 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is justified by the evidence in the candidate profile. The candidate lacks relevant experience in AI engineering, retrieval systems, and search systems, and their skills are not aligned with the job requirements.

### CAND_0058807  [Honeypot]

- **Expected**: Reject / honeypot=True
- **Recruiter**: score=20 / label=Reject / honeypot=False ✅ ❌
- **Recruiter Reasoning**: The candidate has no relevant experience in AI engineering, retrieval systems, or search systems. Their background is in marketing management, and their skills are not aligned with the job requirements.
- **Audit Score**: 90 / quality=Good
- **Missed Strengths**: []
- **Missed Weaknesses**: ['No production experience with embeddings-based retrieval systems', 'No experience with vector databases or hybrid search infrastructure', 'Background in marketing management with no relevant AI engineering experience']
- **Recommended Score**: 20 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is justified by the evidence in the candidate profile. The candidate lacks relevant experience in AI engineering, retrieval systems, and search systems, and their skills are not aligned with the job requirements.

### CAND_0021768  [Borderline]

- **Expected**: Reject / honeypot=False
- **Recruiter**: score=10 / label=Reject / honeypot=False ✅ ✅
- **Recruiter Reasoning**: The candidate has no relevant experience in AI, ML, or related fields. Their work history is primarily in non-technical roles such as project management, marketing, and accounting. They have taken online courses on AI and experimented with LangChain and OpenAI API, but this is not sufficient to demo
- **Audit Score**: 80 / quality=Good
- **Missed Strengths**: ['Candidate has taken online courses on RAG and vector databases', 'Experimented with LangChain and OpenAI API for side projects', 'Has skills in Semantic Search, Information Retrieval, and Embeddings']
- **Missed Weaknesses**: ['Lack of production experience in AI, ML, or related fields', 'No experience with vector databases or hybrid search infrastructure', 'Primarily non-technical work history']
- **Recommended Score**: 10 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is mostly correct, but missed some strengths and weaknesses. The candidate's lack of relevant experience and production deployments in AI, ML, or related fields justifies the reject label.

### CAND_0053507  [Borderline]

- **Expected**: Reject / honeypot=False
- **Recruiter**: score=15 / label=Reject / honeypot=False ✅ ✅
- **Recruiter Reasoning**: The candidate has a similar profile to the first candidate, with no relevant experience in AI, ML, or related fields. Their work history is primarily in non-technical roles, and their education and skills do not demonstrate the required technical expertise.
- **Audit Score**: 80 / quality=Good
- **Missed Strengths**: []
- **Missed Weaknesses**: ['Lack of production experience with embeddings-based retrieval systems', 'No experience with vector databases or hybrid search infrastructure', 'Insufficient hands-on experience designing evaluation frameworks for ranking systems']
- **Recommended Score**: 15 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is justified by the evidence present in the candidate profile. The candidate lacks relevant experience in AI, ML, or related fields, and their work history is primarily in non-technical roles. The candidate's education and skills do not demonstrate the required technical e

### CAND_0048271  [Borderline]

- **Expected**: Weak Fit / honeypot=False
- **Recruiter**: score=20 / label=Reject / honeypot=False ❌ ✅
- **Recruiter Reasoning**: The candidate has some relevant skills such as Kafka, JavaScript, and React, but their work history is primarily in non-technical roles such as project management and accounting. They have taken online courses on AI and experimented with LangChain and OpenAI API, but this is not sufficient to demons
- **Audit Score**: 80 / quality=Good
- **Missed Strengths**: []
- **Missed Weaknesses**: ['No production experience with embeddings-based retrieval systems', 'No experience with vector databases or hybrid search infrastructure', 'Primarily non-technical work history']
- **Recommended Score**: 20 / label=Reject
- **Audit Reasoning**: The recruiter's evaluation is justified by the evidence in the candidate profile. The candidate lacks production experience with embeddings-based retrieval systems and vector databases, and their work history is primarily in non-technical roles. While they have taken online courses on AI and experim

---

## Conclusions

This report validates the labeling pipeline across all six candidate archetypes.
A score of ≥75% label accuracy and ≥80% honeypot accuracy is required for approval.
