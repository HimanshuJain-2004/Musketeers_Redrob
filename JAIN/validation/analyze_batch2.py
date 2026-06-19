import json
import pandas as pd

evals = []
with open('JAIN/artifacts/llm_evaluations_checkpoint.jsonl') as f:
    for line in f:
        if line.strip():
            evals.append(json.loads(line))

df_cands = pd.read_parquet('JAIN/artifacts/labeling_pool_3000.parquet')
batch_map = dict(zip(df_cands['candidate_id'], df_cands['labeling_batch']))

records = []
for e in evals:
    cid = e['candidate_id']
    score = e['evaluation']['fit_score']
    label = e['evaluation']['fit_label']
    honeypot = e['evaluation']['honeypot_label']
    provider = e['provider']
    batch = batch_map.get(cid, 'unknown')
    records.append({'cid': cid, 'batch': batch, 'score': score, 'label': label, 'honeypot': honeypot, 'provider': provider})

df = pd.DataFrame(records)

print('--- Batch 2 Analysis ---')
b2 = df[df['batch'] == 'batch_2']
print(f'Total candidates evaluated: {len(b2)}')
print(f'Honeypots caught: {b2["honeypot"].sum()}')
print('\nScore Distribution:')
print(b2['score'].describe().to_string())
print('\nLabel Distribution:')
print(b2['label'].value_counts().to_string())
print('\nProvider Breakdown:')
print(b2['provider'].value_counts().to_string())

print('\n--- Combined Pass Rates (Batch 1 & 2) ---')
passed = df[df['score'] >= 65]
print(f'Candidates scoring >= 65: {len(passed)} out of {len(df)} ({(len(passed)/len(df))*100:.1f}%)')
