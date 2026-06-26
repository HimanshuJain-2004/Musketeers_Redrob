import json
import pandas as pd
import random
import os
from collections import defaultdict

# Fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def derive_role_family(candidate_json, df_row):
    """
    Derives role family based on Skills -> Career descriptions -> Current role -> Industry
    """
    text_sources = []
    
    # 1. Skills
    skills = candidate_json.get('skills', [])
    if isinstance(skills, list):
        text_sources.extend([s.get('name', '') for s in skills if isinstance(s, dict)])
    elif isinstance(skills, str):
        text_sources.append(skills)
        
    # 2. Career descriptions
    career_history = candidate_json.get('career_history', [])
    if isinstance(career_history, list):
        for role in career_history:
            if isinstance(role, dict):
                text_sources.append(role.get('title', ''))
                text_sources.append(role.get('description', ''))
                
    # 3. Current role
    text_sources.append(str(df_row.get('current_role', '')))
    
    # 4. Industry
    text_sources.append(str(df_row.get('industry', '')))
    
    full_text = " ".join([str(t) for t in text_sources if t]).lower()
    
    # Rule based mapping
    if any(k in full_text for k in ['search', 'information retrieval', 'lucene', 'elasticsearch', 'solr', ' BM25']):
        return 'Search_IR'
    elif any(k in full_text for k in ['recommendation', 'recommender', 'personalization']):
        return 'Recommendation'
    elif any(k in full_text for k in ['nlp', 'llm', 'natural language', 'generative ai', 'gpt', 'bert', 'transformer']):
        return 'NLP_LLM'
    elif any(k in full_text for k in ['ml engineer', 'machine learning engineer', 'deep learning']):
        return 'ML_Engineer'
    elif any(k in full_text for k in ['mlops', 'platform engineer', 'data platform', 'infrastructure', 'cloud architect']):
        return 'Platform_MLOps'
    elif any(k in full_text for k in ['data scientist', 'data science']):
        return 'Data_Science'
    elif any(k in full_text for k in ['backend', 'back-end', 'server', 'java developer', 'python developer', 'c++ developer', 'golang', 'software engineer']):
        return 'Backend'
    elif any(k in full_text for k in ['analytics', 'data analyst', 'business analyst']):
        return 'Analytics'
    elif any(k in full_text for k in ['career break', 'transition', 'bootcamp']):
        return 'Career_Transition'
    elif any(k in full_text for k in ['hr', 'recruiter', 'sales', 'marketing', 'manager']):
        return 'Non_Technical'
    
    return 'Other'

def main():
    print("Loading sampled candidates...")
    input_parquet = 'modeling/artifacts/sampled_candidates.parquet'
    df = pd.read_parquet(input_parquet)
    
    print(f"Total sampled candidates available: {len(df)}")
    
    # Sample from each group
    # 400 Elite, 400 Suspicious, 2200 Stratified
    df_elite = df[df['sample_source'] == 'elite'].sample(n=400, random_state=RANDOM_SEED)
    df_suspicious = df[df['sample_source'] == 'suspicious'].sample(n=400, random_state=RANDOM_SEED)
    df_stratified = df[df['sample_source'] == 'stratified'].sample(n=2200, random_state=RANDOM_SEED)
    
    pool_df = pd.concat([df_elite, df_suspicious, df_stratified]).reset_index(drop=True)
    
    # Shuffle
    pool_df = pool_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Target IDs
    target_ids = set(pool_df['candidate_id'].tolist())
    
    print("Extracting original candidate JSONs from candidates.jsonl...")
    original_jsons = {}
    with open('candidates.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand_obj = json.loads(line)
            cid = cand_obj.get('candidate_id')
            if cid in target_ids:
                original_jsons[cid] = cand_obj
                if len(original_jsons) == len(target_ids):
                    break
                    
    print(f"Extracted {len(original_jsons)} JSON objects.")
    
    # Create the sets and batches
    # We need to ensure mutually exclusive and representative sets.
    # To keep it representative, we split each group (elite, suspicious, stratified) exactly in half.
    
    set1_dfs = []
    set2_dfs = []
    
    # Group by sample_source, and for stratified, by exp_bucket and semantic_bucket to ensure representativeness
    # But since it's 3000, we can just randomly split the subgroups.
    for src in ['elite', 'suspicious']:
        sub_df = pool_df[pool_df['sample_source'] == src]
        shuffled = sub_df.sample(frac=1, random_state=RANDOM_SEED)
        half = len(shuffled) // 2
        set1_dfs.append(shuffled.iloc[:half])
        set2_dfs.append(shuffled.iloc[half:])
        
    strat_df = pool_df[pool_df['sample_source'] == 'stratified']
    strat_shuffled = strat_df.sample(frac=1, random_state=RANDOM_SEED)
    half = len(strat_shuffled) // 2
    set1_dfs.append(strat_shuffled.iloc[:half])
    set2_dfs.append(strat_shuffled.iloc[half:])
    
    set1_df = pd.concat(set1_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    set2_df = pd.concat(set2_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Assign labeling set
    set1_df['labeling_set'] = 'set_1'
    set2_df['labeling_set'] = 'set_2'
    
    # Assign batches for set 1
    # Batch 1: 100 (20 Elite, 20 Suspicious, 60 Stratified)
    batch1_elite = set1_df[set1_df['sample_source'] == 'elite'].iloc[:20]
    batch1_susp = set1_df[set1_df['sample_source'] == 'suspicious'].iloc[:20]
    batch1_strat = set1_df[set1_df['sample_source'] == 'stratified'].iloc[:60]
    batch1_df = pd.concat([batch1_elite, batch1_susp, batch1_strat])
    batch1_ids = set(batch1_df['candidate_id'])
    
    set1_remaining = set1_df[~set1_df['candidate_id'].isin(batch1_ids)]
    batch2_df = set1_remaining.iloc[:400]
    batch2_ids = set(batch2_df['candidate_id'])
    
    batch3_df = set1_remaining.iloc[400:]
    batch3_ids = set(batch3_df['candidate_id'])
    
    def get_batch(cid, l_set):
        if l_set == 'set_2':
            return 'reserve'
        if cid in batch1_ids: return 'batch_1'
        if cid in batch2_ids: return 'batch_2'
        if cid in batch3_ids: return 'batch_3'
        return 'unknown'

    final_pool = pd.concat([set1_df, set2_df])
    final_pool['labeling_batch'] = final_pool.apply(lambda row: get_batch(row['candidate_id'], row['labeling_set']), axis=1)
    
    # Derive role family
    print("Deriving role families...")
    final_pool['role_family'] = final_pool.apply(lambda row: derive_role_family(original_jsons[row['candidate_id']], row), axis=1)
    
    # Add metadata columns
    final_pool['label_status'] = 'unlabeled'
    final_pool['llm_provider'] = None
    
    required_cols = [
        'candidate_id', 'sample_source', 'semantic_bucket', 'exp_bucket', 
        'role_family', 'labeling_set', 'labeling_batch', 'label_status', 'llm_provider'
    ]
    
    # Include industry and current_role for report
    export_cols = required_cols + ['industry', 'current_role']
    # If there are columns missing, fill with None
    for col in export_cols:
        if col not in final_pool.columns:
            final_pool[col] = None
            
    final_export = final_pool[export_cols]
    
    # Write look up JSONL
    print("Writing original_candidate_lookup.jsonl...")
    lookup_path = 'modeling/artifacts/original_candidate_lookup.jsonl'
    with open(lookup_path, 'w', encoding='utf-8') as f:
        for cid in final_export['candidate_id']:
            record = {
                'candidate_id': cid,
                'original_json': original_jsons[cid]
            }
            f.write(json.dumps(record) + '\n')
            
    # Export
    print("Exporting parquet and csv...")
    final_export.to_parquet('modeling/artifacts/labeling_pool_3000.parquet', index=False)
    final_export.to_csv('modeling/artifacts/labeling_pool_3000.csv', index=False)
    
    # Report
    print("Generating report...")
    report_lines = [
        "# Labeling Pool Validation Report",
        "",
        f"**Total Candidates:** {len(final_export)}",
        f"**Set 1 Size:** {len(final_export[final_export['labeling_set'] == 'set_1'])}",
        f"**Set 2 Size:** {len(final_export[final_export['labeling_set'] == 'set_2'])}",
        "",
        "## Distributions",
        "### Sample Source",
        final_export['sample_source'].value_counts().to_string(),
        "",
        "### Semantic Bucket",
        final_export['semantic_bucket'].value_counts().to_string(),
        "",
        "### Experience Bucket",
        final_export['exp_bucket'].value_counts().to_string(),
        "",
        "### Role Family",
        final_export['role_family'].value_counts().to_string(),
        "",
        "### Top 15 Industries",
        final_export['industry'].value_counts().head(15).to_string(),
        "",
        "## Batch Compositions (Set 1)",
    ]
    
    for b in ['batch_1', 'batch_2', 'batch_3']:
        b_df = final_export[final_export['labeling_batch'] == b]
        report_lines.append(f"### {b} (Total: {len(b_df)})")
        report_lines.append(b_df['sample_source'].value_counts().to_string())
        report_lines.append("")
        
    overlap = len(set(set1_df['candidate_id']).intersection(set(set2_df['candidate_id'])))
    report_lines.append(f"**Candidate Overlap Check:** {overlap} (Must be 0)")
    
    missing_json = sum(1 for cid in final_export['candidate_id'] if cid not in original_jsons)
    report_lines.append(f"**Missing JSON Profiles:** {missing_json} (Must be 0)")
    
    with open('modeling/artifacts/labeling_pool_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
        
    print("Done! Artifacts created.")

if __name__ == "__main__":
    main()
