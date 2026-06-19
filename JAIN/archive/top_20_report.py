import os
import pandas as pd

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ARTIFACTS_DIR = os.path.join(WORKSPACE_ROOT, 'JAIN', 'artifacts')

def generate_top_20_report():
    df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'gold_labels.parquet'))
    
    # Sort by fit_score descending
    df_sorted = df.sort_values(by='fit_score', ascending=False).head(20)
    
    report_md = "# Top 20 Candidates by Fit Score\n\n"
    report_md += "These are the highest-scoring candidates from the current 245-label sample.\n\n"
    
    for i, row in enumerate(df_sorted.iterrows(), 1):
        _, row = row
        report_md += f"### {i}. Candidate ID: `{row['candidate_id']}`\n"
        report_md += f"- **Fit Label**: {row['fit_label']}\n"
        report_md += f"- **Fit Score**: {row['fit_score']}/100\n"
        report_md += f"- **Technical Fit**: {row['technical_fit']} | **Product Fit**: {row['product_fit']} | **Behavioral Fit**: {row['behavioral_fit']} | **Career Fit**: {row['career_fit']}\n"
        report_md += f"- **Honeypot Probability**: {row['honeypot_probability']}\n"
        report_md += f"- **Reasoning**: {row['reasoning']}\n\n"
        report_md += "---\n\n"
        
    out_path = os.path.join(ARTIFACTS_DIR, 'top_20_candidates.md')
    with open(out_path, 'w') as f:
        f.write(report_md)
        
    print(f"Top 20 report saved to {out_path}")

if __name__ == '__main__':
    generate_top_20_report()
