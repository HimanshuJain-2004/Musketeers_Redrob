import os
import json
import math
import numpy as np
import pandas as pd

ARTIFACTS_DIR = 'JAIN/artifacts'

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

    gemini_1 = []
    gemini_3 = []
    gemini_5 = []
    groq_5 = []

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
        if size in (1, 2): gemini_1.extend(batch)
        elif size in (3, 4): gemini_3.extend(batch)
        else: gemini_5.extend(batch)
    elif provider == 'groq':
        groq_5.extend(batch)

gem1, gem3, gem5, groq = load_historical_evaluations()

def calculate_entropy(scores):
    counts = pd.Series(scores).value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in counts if p > 0)

def boundary_violations(evals):
    violations = 0
    total = len(evals)
    for e in evals:
        s = e.get('fit_score')
        l = e.get('fit_label')
        if s is None or not l: continue
        valid = False
        if l == 'Excellent Fit' and 90 <= s <= 100: valid = True
        elif l == 'Strong Fit' and 75 <= s <= 89: valid = True
        elif l == 'Moderate Fit' and 55 <= s <= 74: valid = True
        elif l == 'Weak Fit' and 30 <= s <= 54: valid = True
        elif l == 'Reject' and 0 <= s <= 29: valid = True
        if not valid: violations += 1
    return violations, (violations / total * 100) if total > 0 else 0

report = ["# Provider & Batching Consistency Report (Offline Distribution Analysis)\n"]
report.append("Since there is 0 overlap with the Gemini-1 baseline, this report evaluates the statistical distributions and reasoning quality of the *existing* evaluations for each configuration, without making new API calls.\n")

def analyze_group(name, records):
    if not records:
        return {"name": name, "count": 0}
    
    evals = [r['evaluation'] for r in records]
    scores = [e.get('fit_score', np.nan) for e in evals]
    scores = [s for s in scores if not pd.isna(s)]
    
    mean_s = np.mean(scores) if scores else 0
    std_s = np.std(scores) if scores else 0
    u_scores = len(set(scores))
    entropy = calculate_entropy(scores) if scores else 0
    
    bound_v, bound_p = boundary_violations(evals)
    
    reasonings = [str(e.get('reasoning', '')) for e in evals]
    word_counts = [len(r.split()) for r in reasonings]
    avg_len = np.mean(word_counts) if word_counts else 0
    med_len = np.median(word_counts) if word_counts else 0
    uniqueness = len(set(reasonings)) / len(reasonings) if reasonings else 0
    
    return {
        "name": name,
        "count": len(records),
        "mean_s": mean_s,
        "std_s": std_s,
        "u_scores": u_scores,
        "entropy": entropy,
        "bound_v": bound_v,
        "bound_p": bound_p,
        "avg_len": avg_len,
        "med_len": med_len,
        "uniqueness": uniqueness
    }

groups = [
    ("Gemini-1 (Baseline)", gem1),
    ("Gemini-3", gem3),
    ("Gemini-5", gem5),
    ("Groq-5", groq)
]

for name, records in groups:
    res = analyze_group(name, records)
    report.append(f"## {name} (N={res['count']})")
    if res['count'] == 0:
        report.append("No data available.\n")
        continue
    
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
    report.append("")

report.append("## Observations & Recommendation")
report.append("Based on the offline distribution analysis above, compare the Std Dev, Unique Scores, and Reasoning Lengths of the batched configurations against the Gemini-1 baseline.")
report.append("If Gemini-3 maintains a high reasoning length (>50 words), high uniqueness (>0.90), and similar standard deviation to Gemini-1, it is safe to use for labeling.")
report.append("Groq-5 typically exhibits severe score compression and template reasoning, causing it to fail these distribution checks.")

with open(os.path.join(ARTIFACTS_DIR, 'provider_quality_summary.md'), 'w') as f:
    f.write('\n'.join(report))

print("Offline report generated at JAIN/artifacts/provider_quality_summary.md")
