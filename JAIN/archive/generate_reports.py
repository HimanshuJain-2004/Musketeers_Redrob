import os
import json
import pandas as pd
import random

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ARTIFACTS_DIR = os.path.join(WORKSPACE_ROOT, 'JAIN', 'artifacts')

def generate_reports():
    # Load Stage A labels
    df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'gold_labels.parquet'))
    
    # 1. Label Distribution
    dist = df['fit_label'].value_counts()
    total = len(df)
    
    report_md = f"# Stage A Label Distribution Report\n\nTotal Candidates: {total}\n\n"
    report_md += "| Fit Label | Count | Percentage |\n|---|---|---|\n"
    
    for label in ['Excellent Fit', 'Strong Fit', 'Moderate Fit', 'Weak Fit', 'Reject']:
        count = dist.get(label, 0)
        pct = (count / total) * 100 if total > 0 else 0
        report_md += f"| {label} | {count} | {pct:.1f}% |\n"
        
    hp_count = df['honeypot_label'].sum()
    hp_pct = (hp_count / total) * 100 if total > 0 else 0
    report_md += f"\n## Honeypot Detection\n- Detected as Honeypots: {hp_count} ({hp_pct:.1f}%)\n"
    report_md += f"- Average Honeypot Probability: {df['honeypot_probability'].mean():.2f}\n"
    
    with open(os.path.join(ARTIFACTS_DIR, 'stage_A_distribution_report.md'), 'w') as f:
        f.write(report_md)
        
    # 2. Audit Sample
    # Sample up to 50 candidates
    sample_size = min(50, len(df))
    sampled_df = df.sample(n=sample_size, random_state=42)
    
    audit_md = "# Manual Audit Sample\n\n"
    audit_md += "Here are 50 random candidates from Stage A for your manual review.\n\n"
    
    for _, row in sampled_df.iterrows():
        audit_md += f"### Candidate ID: `{row['candidate_id']}`\n"
        audit_md += f"- **Fit Label**: {row['fit_label']}\n"
        audit_md += f"- **Fit Score**: {row['fit_score']}/10\n"
        audit_md += f"- **Honeypot**: {row['honeypot_label']} (Prob: {row['honeypot_probability']})\n"
        audit_md += f"- **Reasoning**: {row['reasoning']}\n\n"
        audit_md += "---\n\n"
        
    with open(os.path.join(ARTIFACTS_DIR, 'stage_A_audit_sample.md'), 'w') as f:
        f.write(audit_md)
        
    print("Reports generated successfully!")

if __name__ == '__main__':
    generate_reports()
