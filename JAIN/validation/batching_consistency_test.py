import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import math

ARTIFACTS_DIR = 'JAIN/artifacts'
os.makedirs('JAIN/validation', exist_ok=True)

# -----------------------------------------
# Phase 1: Existing Overlap Analysis
# -----------------------------------------
def load_historical_evaluations():
    files = [
        os.path.join(ARTIFACTS_DIR, 'llm_evaluations_checkpoint.jsonl'),
        os.path.join(ARTIFACTS_DIR, 'llm_evaluations_checkpoint_backup.jsonl')
    ]
    records = []
    cids = set()
    for f in files:
        if not os.path.exists(f): continue
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip(): continue
                rec = json.loads(line)
                key = rec['candidate_id'] + rec['timestamp']
                if key not in cids:
                    cids.add(key)
                    records.append(rec)
    records.sort(key=lambda x: x['timestamp'])

    gemini_1 = {}
    gemini_3 = {}
    gemini_5 = {}
    groq_5 = {}

    current_batch = []
    for r in records:
        if not current_batch:
            current_batch.append(r)
        else:
            t1 = pd.to_datetime(current_batch[-1]['timestamp'])
            t2 = pd.to_datetime(r['timestamp'])
            if (t2 - t1).total_seconds() < 2.0 and current_batch[-1]['provider'] == r['provider'] and current_batch[-1]['key_id'] == r['key_id']:
                current_batch.append(r)
            else:
                store_batch(current_batch, gemini_1, gemini_3, gemini_5, groq_5)
                current_batch = [r]
    if current_batch:
        store_batch(current_batch, gemini_1, gemini_3, gemini_5, groq_5)

    return gemini_1, gemini_3, gemini_5, groq_5

def store_batch(batch, gemini_1, gemini_3, gemini_5, groq_5):
    provider = batch[0]['provider']
    size = len(batch)
    if provider == 'gemini':
        if size in (1, 2):
            for r in batch: gemini_1[r['candidate_id']] = r['evaluation']
        elif size in (3, 4):
            for r in batch: gemini_3[r['candidate_id']] = r['evaluation']
        else:
            for r in batch: gemini_5[r['candidate_id']] = r['evaluation']
    elif provider == 'groq':
        for r in batch: groq_5[r['candidate_id']] = r['evaluation']

gemini_1_hist, gemini_3_hist, gemini_5_hist, groq_5_hist = load_historical_evaluations()

overlap_3 = set(gemini_1_hist.keys()) & set(gemini_3_hist.keys())
overlap_5 = set(gemini_1_hist.keys()) & set(gemini_5_hist.keys())
overlap_groq = set(gemini_1_hist.keys()) & set(groq_5_hist.keys())

num_overlap_3 = len(overlap_3)
num_overlap_5 = len(overlap_5)
num_overlap_groq = len(overlap_groq)

print(f"Gem1 intersect Gem3 = {num_overlap_3}")
print(f"Gem1 intersect Gem5 = {num_overlap_5}")
print(f"Gem1 intersect Groq = {num_overlap_groq}")

need_additional = (num_overlap_3 < 20) or (num_overlap_5 < 20) or (num_overlap_groq < 20)
print(f"Need additional evaluations? {'Yes' if need_additional else 'No'}")
if need_additional:
    print("Reason: Insufficient overlap with the Gemini-1 baseline to perform a reliable comparison (minimum 20 candidates required).")
    sys.exit(0)

# -----------------------------------------
# Proceed with Dataset Building (If Overlap Sufficient)
# -----------------------------------------
all_overlapping_cids = overlap_3 | overlap_5 | overlap_groq
dataset = []
for cid in all_overlapping_cids:
    g1 = gemini_1_hist.get(cid, {})
    g3 = gemini_3_hist.get(cid, {})
    g5 = gemini_5_hist.get(cid, {})
    gr = groq_5_hist.get(cid, {})
    
    row = {
        'candidate_id': cid,
        'gemini_1_score': g1.get('fit_score', np.nan),
        'gemini_1_label': g1.get('fit_label', ''),
        'gemini_1_hp': g1.get('honeypot_label', False),
        'gemini_1_hp_prob': g1.get('honeypot_probability', np.nan),
        'gemini_1_reasoning': g1.get('reasoning', ''),
        
        'gemini_3_score': g3.get('fit_score', np.nan),
        'gemini_3_label': g3.get('fit_label', ''),
        'gemini_3_hp': g3.get('honeypot_label', False),
        'gemini_3_hp_prob': g3.get('honeypot_probability', np.nan),
        'gemini_3_reasoning': g3.get('reasoning', ''),
        
        'gemini_5_score': g5.get('fit_score', np.nan),
        'gemini_5_label': g5.get('fit_label', ''),
        'gemini_5_hp': g5.get('honeypot_label', False),
        'gemini_5_hp_prob': g5.get('honeypot_probability', np.nan),
        'gemini_5_reasoning': g5.get('reasoning', ''),
        
        'groq_score': gr.get('fit_score', np.nan),
        'groq_label': gr.get('fit_label', ''),
        'groq_hp': gr.get('honeypot_label', False),
        'groq_hp_prob': gr.get('honeypot_probability', np.nan),
        'groq_reasoning': gr.get('reasoning', '')
    }
    dataset.append(row)

df = pd.DataFrame(dataset)
df.to_parquet(os.path.join(ARTIFACTS_DIR, 'label_consistency_dataset.parquet'), index=False)
print("Dataset saved to JAIN/artifacts/label_consistency_dataset.parquet.")

# -----------------------------------------
# Phase 3: Metrics & Reporting
# -----------------------------------------
def calculate_entropy(scores):
    counts = pd.Series(scores).value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in counts if p > 0)

def boundary_violations(scores, labels):
    violations = 0
    for s, l in zip(scores, labels):
        if pd.isna(s) or not l: continue
        valid = False
        if l == 'Excellent Fit' and 90 <= s <= 100: valid = True
        elif l == 'Strong Fit' and 75 <= s <= 89: valid = True
        elif l == 'Moderate Fit' and 55 <= s <= 74: valid = True
        elif l == 'Weak Fit' and 30 <= s <= 54: valid = True
        elif l == 'Reject' and 0 <= s <= 29: valid = True
        if not valid: violations += 1
    return violations, (violations / len(scores) * 100) if len(scores) > 0 else 0

report = ["# Provider & Batching Consistency Report\n"]

def evaluate_provider(name, prefix, df):
    # Base masks
    valid = df[prefix+'_score'].notna() & df['gemini_1_score'].notna()
    n_valid = valid.sum()
    if n_valid == 0: 
        return {
            "name": name, "lbl_agree": 0, "hp_agree": 0, "avg_diff": 0, "hp_prob_diff": 0,
            "p_corr": 0, "s_corr": 0, "mean_s": 0, "std_s": 0, "u_scores": 0, "entropy": 0,
            "bound_v": 0, "bound_p": 0, "avg_len": 0, "med_len": 0, "uniqueness": 0,
            "drift_0_5": 0, "drift_6_10": 0, "drift_11_20": 0, "drift_20_plus": 0
        }
    
    lbl_agree = (df[valid][prefix+'_label'] == df[valid]['gemini_1_label']).mean() * 100
    hp_agree = (df[valid][prefix+'_hp'] == df[valid]['gemini_1_hp']).mean() * 100
    avg_diff = (df[valid][prefix+'_score'] - df[valid]['gemini_1_score']).abs().mean()
    hp_prob_diff = (df[valid][prefix+'_hp_prob'] - df[valid]['gemini_1_hp_prob']).abs().mean()
    
    try:
        if n_valid > 1:
            p_corr, _ = pearsonr(df[valid]['gemini_1_score'], df[valid][prefix+'_score'])
            s_corr, _ = spearmanr(df[valid]['gemini_1_score'], df[valid][prefix+'_score'])
        else:
            p_corr, s_corr = 0, 0
        if np.isnan(p_corr): p_corr = 0
        if np.isnan(s_corr): s_corr = 0
    except:
        p_corr, s_corr = 0, 0
    
    scores = df[valid][prefix+'_score']
    mean_s = scores.mean()
    std_s = scores.std()
    if np.isnan(std_s): std_s = 0
    u_scores = scores.nunique()
    entropy = calculate_entropy(scores)
    
    bound_v, bound_p = boundary_violations(df[valid][prefix+'_score'], df[valid][prefix+'_label'])
    
    reasonings = df[valid][prefix+'_reasoning']
    word_counts = reasonings.apply(lambda x: len(str(x).split()))
    avg_len = word_counts.mean()
    med_len = word_counts.median()
    uniqueness = reasonings.nunique() / n_valid if n_valid > 0 else 0
    
    diffs = (df[valid][prefix+'_score'] - df[valid]['gemini_1_score']).abs()
    drift_0_5 = (diffs <= 5).sum()
    drift_6_10 = ((diffs > 5) & (diffs <= 10)).sum()
    drift_11_20 = ((diffs > 10) & (diffs <= 20)).sum()
    drift_20_plus = (diffs > 20).sum()
    
    return {
        "name": name,
        "lbl_agree": lbl_agree,
        "hp_agree": hp_agree,
        "avg_diff": avg_diff,
        "hp_prob_diff": hp_prob_diff,
        "p_corr": p_corr,
        "s_corr": s_corr,
        "mean_s": mean_s,
        "std_s": std_s,
        "u_scores": u_scores,
        "entropy": entropy,
        "bound_v": bound_v,
        "bound_p": bound_p,
        "avg_len": avg_len,
        "med_len": med_len,
        "uniqueness": uniqueness,
        "drift_0_5": drift_0_5,
        "drift_6_10": drift_6_10,
        "drift_11_20": drift_11_20,
        "drift_20_plus": drift_20_plus
    }

metrics = []
for name, prefix in [("Gemini-3", "gemini_3"), ("Gemini-5", "gemini_5"), ("Groq-5", "groq")]:
    res = evaluate_provider(name, prefix, df)
    metrics.append(res)
    
    report.append(f"## {name}")
    report.append(f"- **Label Agreement:** {res['lbl_agree']:.1f}%")
    report.append(f"- **Honeypot Agreement:** {res['hp_agree']:.1f}%")
    report.append(f"- **Avg Score Diff:** {res['avg_diff']:.1f}")
    report.append(f"- **Pearson Correlation:** {res['p_corr']:.3f}")
    report.append(f"- **Spearman Correlation:** {res['s_corr']:.3f}")
    report.append(f"- **Boundary Violations:** {res['bound_v']} ({res['bound_p']:.1f}%)")
    report.append(f"- **Score Compression:**")
    report.append(f"  - Mean: {res['mean_s']:.1f}")
    report.append(f"  - Std Dev: {res['std_s']:.1f} {'(FLAG: <12)' if res['std_s'] < 12 else ''}")
    report.append(f"  - Unique Scores: {res['u_scores']} {'(FLAG: <15)' if res['u_scores'] < 15 else ''}")
    report.append(f"  - Entropy: {res['entropy']:.2f}")
    report.append(f"- **Reasoning:**")
    report.append(f"  - Avg Length: {res['avg_len']:.1f} words {'(FLAG: <25)' if res['avg_len'] < 25 else ''}")
    report.append(f"  - Median Length: {res['med_len']:.1f} words")
    report.append(f"  - Uniqueness: {res['uniqueness']:.2f} {'(FLAG: <0.70)' if res['uniqueness'] < 0.7 else ''}")
    report.append(f"- **Score Drift:** 0-5: {res['drift_0_5']}, 6-10: {res['drift_6_10']}, 11-20: {res['drift_11_20']}, 20+: {res['drift_20_plus']}")
    report.append("")

g3 = metrics[0]
g5 = metrics[1]
gq = metrics[2]

g3_approve = (g3['lbl_agree'] >= 90 and g3['hp_agree'] >= 95 and g3['avg_diff'] <= 10 and g3['p_corr'] >= 0.90 and g3['s_corr'] >= 0.90)
g5_approve = (g5['lbl_agree'] >= 85 and g5['hp_agree'] >= 90 and g5['avg_diff'] <= 15)
gq_approve = (gq['lbl_agree'] >= 85 and gq['hp_agree'] >= 90 and gq['p_corr'] >= 0.85 and gq['s_corr'] >= 0.85 and gq['uniqueness'] >= 0.80 and gq['std_s'] >= 12)

report.append("## Final Recommendation")
if g3_approve:
    report.append("**Recommendation:** Switch to Gemini 3 candidates/request.")
    report.append("Gemini-3 meets all strict quality thresholds and preserves the baseline quality while significantly increasing throughput.")
elif g5_approve:
    report.append("**Recommendation:** Switch to Gemini 5 candidates/request.")
    report.append("Gemini-5 meets acceptable quality thresholds, though it may exhibit slightly more drift than Gemini-3.")
elif gq_approve:
    report.append("**Recommendation:** Use Groq only as emergency overflow.")
    report.append("Groq meets the minimum quality thresholds, but Gemini is preferred due to higher baseline alignment.")
else:
    report.append("**Recommendation:** Continue using Gemini 1 candidate/request.")
    report.append("No batching configuration met the required quality thresholds.")
    if not gq_approve:
        report.append("**Note:** Do not use Groq for labeling, as it failed quality checks.")

with open(os.path.join(ARTIFACTS_DIR, 'provider_quality_summary.md'), 'w') as f:
    f.write('\n'.join(report))

print("Report generated at JAIN/artifacts/provider_quality_summary.md")
