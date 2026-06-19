import os
import json
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
JAIN_DIR = os.path.join(WORKSPACE_ROOT, 'JAIN')
ARTIFACTS_DIR = os.path.join(JAIN_DIR, 'artifacts')

# Ensure output directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("Loading PIYUSH parquet files...")
cand_features = pd.read_parquet(os.path.join(WORKSPACE_ROOT, 'PIYUSH/artifacts/candidate_features.parquet'))
behav_features = pd.read_parquet(os.path.join(WORKSPACE_ROOT, 'PIYUSH/artifacts/behavior_features.parquet'))
sem_scores = pd.read_parquet(os.path.join(WORKSPACE_ROOT, 'PIYUSH/artifacts/semantic_scores.parquet'))

print("Merging Piyush's features...")
df_piyush = cand_features.merge(behav_features, on='candidate_id', how='inner')
df_piyush = df_piyush.merge(sem_scores, on='candidate_id', how='inner')

# AI Keywords for honeypot check
ai_keywords = {'retrieval', 'ranking', 'embeddings', 'vector database', 'pinecone', 'weaviate', 'qdrant', 'rag', 'llm', 'ndcg', 'mrr', 'map', 'machine learning', 'artificial intelligence', 'nlp', 'cv', 'deep learning'}
jd_keywords = {'retrieval', 'ranking', 'embeddings', 'vector database', 'pinecone', 'weaviate', 'qdrant', 'rag', 'llm', 'ndcg', 'mrr', 'map'}

print("Processing raw JSONL data for honeypot and diversity features...")
raw_data = []
jsonl_path = os.path.join(WORKSPACE_ROOT, 'candidates.jsonl')
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        c_id = record.get('candidate_id')
        
        prof = record.get('profile', {})
        industry = prof.get('current_industry', 'Unknown')
        current_title = prof.get('current_title', 'Unknown')
        
        headline = str(prof.get('headline', '')).lower()
        summary = str(prof.get('summary', '')).lower()
        
        skills = record.get('skills', [])
        num_ai_skills = sum(1 for s in skills if str(s.get('name', '')).lower() in ai_keywords)
        
        career_text = ""
        career = record.get('career_history', [])
        for job in career:
            career_text += str(job.get('description', '')).lower() + " "
        
        full_text = headline + " " + summary + " " + career_text
        keyword_stuffing_count = sum(full_text.count(kw) for kw in jd_keywords)
        
        # Career Anomaly
        # Simple heuristic: "intern" -> "senior/lead/architect" in < 3 years (36 months)
        career_anomaly = 0
        is_intern = False
        is_senior = False
        total_months = 0
        for job in career:
            jt = str(job.get('title', '')).lower()
            if 'intern' in jt:
                is_intern = True
            if any(lvl in jt for lvl in ['senior', 'lead', 'architect', 'head', 'director', 'vp']):
                is_senior = True
            total_months += job.get('duration_months', 0)
            
        if is_intern and is_senior and total_months < 36:
            career_anomaly = 1.0
        
        redrob = record.get('redrob_signals', {})
        recruiter_response_rate = redrob.get('recruiter_response_rate', 1.0)
        notice_period_days = redrob.get('notice_period_days', 30)
        open_to_work = redrob.get('open_to_work_flag', True)
        
        # Inactivity (use Piyush's days_since_active if available, else derive)
        # Let's trust Piyush's days_since_active when merged.
        
        raw_data.append({
            'candidate_id': c_id,
            'industry': industry,
            'current_role': current_title,
            'num_ai_skills': num_ai_skills,
            'keyword_stuffing_count': keyword_stuffing_count,
            'career_anomaly_raw': career_anomaly,
            'raw_recruiter_response_rate': recruiter_response_rate,
            'raw_notice_period': notice_period_days,
            'raw_open_to_work': open_to_work
        })

df_raw = pd.DataFrame(raw_data)
df = df_piyush.merge(df_raw, on='candidate_id', how='inner')

print("Calculating Honeypot Risk...")
# Skill Inflation Score = num_ai_skills / max(years_exp, 1)
df['years_exp_clamped'] = df['years_exp'].clip(lower=1.0)
df['skill_inflation'] = df['num_ai_skills'] / df['years_exp_clamped']
# Normalize
df['skill_inflation_norm'] = df['skill_inflation'] / df['skill_inflation'].max()

# Keyword Stuffing Score
# High count AND weak relevant experience
df['keyword_stuffing_norm'] = df['keyword_stuffing_count'] / df['keyword_stuffing_count'].max()
df['weak_experience_flag'] = (df['years_exp'] < 2).astype(float)
df['keyword_stuffing_risk'] = df['keyword_stuffing_norm'] * df['weak_experience_flag']

# Behavioral Risk
# Low response rate (<0.2), Long notice (>60), Not open to work, inactive
df['low_response'] = (df['raw_recruiter_response_rate'] < 0.2).astype(float)
df['long_notice'] = (df['raw_notice_period'] > 60).astype(float)
df['not_open'] = (~df['raw_open_to_work']).astype(float)
df['inactive'] = (df.get('days_since_active', 0) > 90).astype(float)
df['behavioral_risk'] = (df['low_response'] + df['long_notice'] + df['not_open'] + df['inactive']) / 4.0

# Overall Honeypot Risk
df['honeypot_risk'] = (0.35 * df['skill_inflation_norm'] + 
                       0.25 * df['keyword_stuffing_risk'] + 
                       0.20 * df['career_anomaly_raw'] + 
                       0.20 * df['behavioral_risk'])

# 1. Elite Candidate Pool (Top 500 Semantic)
print("Extracting Elite Candidate Pool...")
elite_pool = df.nlargest(500, 'semantic_score').copy()
elite_pool['sample_source'] = 'elite'

# 2. Suspicious Candidate Pool (Top 700 Honeypot)
print("Extracting Suspicious Candidate Pool...")
suspicious_candidates_mask = df['candidate_id'].isin(elite_pool['candidate_id']) == False
remaining_for_suspicious = df[suspicious_candidates_mask]
suspicious_pool = remaining_for_suspicious.nlargest(700, 'honeypot_risk').copy()
suspicious_pool['sample_source'] = 'suspicious'

# 3. Stratified Sampling (3800)
print("Performing Stratified Sampling...")
used_ids = set(elite_pool['candidate_id']).union(set(suspicious_pool['candidate_id']))
strat_pool_source = df[~df['candidate_id'].isin(used_ids)].copy()

# Bucketing
# Semantic Buckets
q75 = strat_pool_source['semantic_score'].quantile(0.75)
q25 = strat_pool_source['semantic_score'].quantile(0.25)
def get_semantic_bucket(x):
    if x >= q75: return 'Top 25%'
    elif x >= q25: return 'Middle 50%'
    else: return 'Bottom 25%'
strat_pool_source['semantic_bucket'] = strat_pool_source['semantic_score'].apply(get_semantic_bucket)

# Experience Buckets: 0-3, 3-5, 5-8, 8+
def get_exp_bucket(x):
    if x < 3: return '0-3 years'
    elif x < 5: return '3-5 years'
    elif x < 8: return '5-8 years'
    else: return '8+ years'
strat_pool_source['exp_bucket'] = strat_pool_source['years_exp'].apply(get_exp_bucket)

strat_pool_source['stratum'] = strat_pool_source['semantic_bucket'] + ' | ' + strat_pool_source['exp_bucket']

TOTAL_STRATIFIED_TARGET = 3800
stratum_counts = strat_pool_source['stratum'].value_counts()
n_strata = len(stratum_counts)

# Hybrid Allocation (70% proportional, 30% equalized)
prop_target = int(TOTAL_STRATIFIED_TARGET * 0.70)
equal_target = TOTAL_STRATIFIED_TARGET - prop_target

equal_per_stratum = equal_target // n_strata
proportions = stratum_counts / len(strat_pool_source)

allocations = {}
for stratum in stratum_counts.index:
    alloc = equal_per_stratum + int(proportions[stratum] * prop_target)
    # Ensure we don't ask for more than available
    allocations[stratum] = min(alloc, stratum_counts[stratum])

# Adjust allocations to hit exactly 3800 if possible (due to rounding / min limits)
current_total = sum(allocations.values())
diff = TOTAL_STRATIFIED_TARGET - current_total
if diff > 0:
    # distribute remainder to largest buckets
    for stratum in stratum_counts.index:
        if allocations[stratum] < stratum_counts[stratum]:
            add = min(diff, stratum_counts[stratum] - allocations[stratum])
            allocations[stratum] += add
            diff -= add
        if diff == 0: break

sampled_dfs = []
for stratum, n_samples in allocations.items():
    sub_df = strat_pool_source[strat_pool_source['stratum'] == stratum]
    
    # Diversity Sampling: Group by generic available attributes
    # We will use industry and exp_bucket. (current_role has too many unique values for clean grouping)
    # To sample proportionally within groups:
    if n_samples > 0:
        # compute weights inversely proportional to group size? No, proportional to group size is just random sampling.
        # "Diversity Sampling... preserve diversity across available career attributes"
        # We can just sample, but let's ensure representation by stratifying further if we want,
        # or simply `groupby` and sample fractionally.
        
        # A simple trick for diverse proportional sampling is just `sample(n, weights=...)` or simple uniform sample 
        # since uniform sample inherently gives proportional representation. 
        # But to ensure *coverage* of small minorities, we can take at least 1 from each group if possible.
        # For simplicity and correctness without failing on small groups:
        sample = sub_df.sample(n=n_samples, random_state=42)
        sampled_dfs.append(sample)

stratified_sample = pd.concat(sampled_dfs)
stratified_sample['sample_source'] = 'stratified'

# Combine and deduplicate
final_sample = pd.concat([elite_pool, suspicious_pool, stratified_sample]).drop_duplicates(subset=['candidate_id'])

# Save Data
print("Saving outputs...")
final_sample.to_parquet(os.path.join(ARTIFACTS_DIR, 'sampled_candidates.parquet'), index=False)
final_sample.to_csv(os.path.join(ARTIFACTS_DIR, 'sampled_candidates.csv'), index=False)

suspicious_cols = ['candidate_id', 'honeypot_risk', 'skill_inflation_norm', 'keyword_stuffing_risk', 'career_anomaly_raw', 'behavioral_risk']
suspicious_pool[suspicious_cols].to_parquet(os.path.join(ARTIFACTS_DIR, 'suspicious_candidates.parquet'), index=False)
suspicious_pool[suspicious_cols].to_csv(os.path.join(ARTIFACTS_DIR, 'suspicious_candidates.csv'), index=False)

# Generate Report
report_path = os.path.join(ARTIFACTS_DIR, 'sampling_report.md')
with open(report_path, 'w') as f:
    f.write("# Candidate Sampling Report\n\n")
    f.write(f"**Total Sampled Candidates:** {len(final_sample)}\n\n")
    
    f.write("### Sample Sources\n")
    f.write(final_sample['sample_source'].value_counts().to_string() + "\n\n")
    
    f.write("### Semantic Bucket Distribution (Stratified only)\n")
    f.write(stratified_sample['semantic_bucket'].value_counts().to_string() + "\n\n")
    
    f.write("### Experience Bucket Distribution (Stratified only)\n")
    f.write(stratified_sample['exp_bucket'].value_counts().to_string() + "\n\n")
    
    f.write("### Industry Distribution (Top 15)\n")
    f.write(final_sample['industry'].value_counts().head(15).to_string() + "\n\n")
    
    f.write("### Honeypot Risk Distribution (Suspicious Pool)\n")
    f.write(suspicious_pool['honeypot_risk'].describe().to_string() + "\n\n")

print("Done! Report saved to sampling_report.md")
