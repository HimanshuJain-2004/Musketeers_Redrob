import os
import json
import pandas as pd

WORKSPACE_ROOT = r"c:\Users\himan\Downloads\India_runs_data_and_ai_challenge"
ARTIFACTS_DIR = os.path.join(WORKSPACE_ROOT, 'JAIN', 'artifacts')

df_sampled = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'sampled_candidates.parquet'))
df_suspicious = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'suspicious_candidates.parquet'))

suspicious_ids = set(df_suspicious['candidate_id'].tolist())
valid_sampled = df_sampled[~df_sampled['candidate_id'].isin(suspicious_ids)]
valid_ids = set(valid_sampled['candidate_id'].tolist())

valid_candidates = []
with open(os.path.join(WORKSPACE_ROOT, 'candidates.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        if record['candidate_id'] in valid_ids:
            valid_candidates.append(record)

recsys_search_keywords = [
    'retrieval', 'vector search', 'semantic search', 'embeddings', 
    'pinecone', 'milvus', 'qdrant', 'faiss', 'weaviate', 'haystack', 
    'learning-to-rank', 'learning to rank', 'ranking layer', 
    'recommendation system', 'recommendation engine', 'search engine',
    'search system', 'ranking system', 'information retrieval'
]

results = []
for cand in valid_candidates:
    profile = cand.get('profile', {})
    title = str(profile.get('current_title', '')).lower()
    
    career = cand.get('career_history', [])
    titles = [title] + [str(job.get('title', '')).lower() for job in career]
    
    # Check titles
    allowed_titles = [
        'software engineer', 'ml engineer', 'machine learning engineer', 
        'data scientist', 'nlp engineer', 'search engineer', 
        'recommendation engineer', 'backend engineer', 'ai research engineer',
        'ai specialist', 'ai engineer', 'applied ml engineer'
    ]
    has_valid_title = False
    for t in titles[:3]:
        if any(at in t for at in allowed_titles):
            has_valid_title = True
            break
            
    if not has_valid_title:
        continue
        
    matched_kws = []
    desc_combined = " ".join([str(job.get('description', '')).lower() for job in career])
    summary_combined = (str(profile.get('summary', '')) + " " + str(profile.get('headline', ''))).lower()
    skills_combined = " ".join([s.get('name', '').lower() for s in cand.get('skills', [])])
    
    full_text = desc_combined + " " + summary_combined + " " + skills_combined
    for kw in recsys_search_keywords:
        if kw in full_text:
            matched_kws.append(kw)
            
    if len(matched_kws) >= 2:
        results.append({
            'candidate_id': cand['candidate_id'],
            'current_title': profile.get('current_title'),
            'headline': profile.get('headline'),
            'matched_keywords': list(set(matched_kws)),
            'career_summary': [f"{job.get('title')} at {job.get('company')}" for job in career[:3]]
        })

print(f"Found {len(results)} candidate profiles with at least 2 matching search/recsys terms.")
results.sort(key=lambda x: len(x['matched_keywords']), reverse=True)
for i, r in enumerate(results[:40]):
    print(f"{i+1}. ID: {r['candidate_id']} | Title: {r['current_title']} | KWs: {r['matched_keywords']}")
    print(f"   Jobs: {r['career_summary']}")
