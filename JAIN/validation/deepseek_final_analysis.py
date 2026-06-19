import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "JAIN", "artifacts")

def main():
    print("Loading data...")
    sample_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "deepseek_sample_combined.parquet"))
    
    # Load DS labels
    ds_labels = {}
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_labels.jsonl"), "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                ds_labels[obj['candidate_id']] = obj
                
    # Load DS audits
    ds_audits = {}
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_audits.jsonl"), "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                ds_audits[obj['candidate_id']] = obj
                
    # Load Execution Stats
    stats_list = []
    stats_file = os.path.join(ARTIFACTS_DIR, "deepseek_execution_stats.jsonl")
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            for line in f:
                if line.strip():
                    stats_list.append(json.loads(line))
                    
    stats_df = pd.DataFrame(stats_list) if len(stats_list) > 0 else pd.DataFrame()

    results = []
    high_disagreements = []
    
    for _, row in sample_df.iterrows():
        cid = row['candidate_id']
        gemini_score = row['fit_score']
        gemini_label = row['fit_label']
        gemini_hp = row['honeypot_label']
        gemini_reasoning = row['reasoning']
        
        if cid not in ds_labels: continue
        
        ds_eval = ds_labels[cid]['evaluation']
        ds_score = ds_eval.get('fit_score', 0)
        ds_label = ds_eval.get('fit_label', 'Unknown')
        ds_hp = ds_eval.get('honeypot_label', False)
        ds_reasoning = ds_eval.get('reasoning', '')
        
        delta = ds_score - gemini_score
        label_match = (gemini_label == ds_label)
        hp_match = (gemini_hp == ds_hp)
        
        # Determine upgrade/downgrade
        # Ranks: Reject(0), Weak(1), Moderate(2), Strong(3), Excellent(4)
        ranks = {'Reject':0, 'Weak Fit':1, 'Moderate Fit':2, 'Strong Fit':3, 'Excellent Fit':4}
        ds_rank = ranks.get(ds_label, -1)
        gem_rank = ranks.get(gemini_label, -1)
        
        direction = "Match"
        if ds_rank > gem_rank: direction = "Upgrade"
        elif ds_rank < gem_rank: direction = "Downgrade"
        
        cat = 'Borderline'
        if gemini_label in ['Excellent Fit', 'Strong Fit']: cat = 'Strong/Excellent'
        elif gemini_label == 'Moderate Fit': cat = 'Moderate'
        elif gemini_label in ['Weak Fit', 'Reject'] and not gemini_hp: cat = 'Weak/Reject'
        elif gemini_hp: cat = 'Honeypots'
        
        results.append({
            'candidate_id': cid,
            'gemini_score': gemini_score,
            'ds_score': ds_score,
            'delta': delta,
            'label_match': label_match,
            'hp_match': hp_match,
            'direction': direction,
            'category': cat,
            'gemini_label': gemini_label,
            'ds_label': ds_label,
            'gemini_hp': gemini_hp,
            'ds_hp': ds_hp
        })
        
        audit_obj = ds_audits.get(cid, {}).get('audit', {})
        audit_score = audit_obj.get('audit_score', 'N/A')
        
        # High Disagreement Rule
        is_hd = False
        if abs(delta) > 15 or not label_match or not hp_match:
            is_hd = True
            
        if is_hd:
            high_disagreements.append({
                'candidate_id': cid,
                'gemini_score': gemini_score, 'ds_score': ds_score, 'delta': delta,
                'gemini_label': gemini_label, 'ds_label': ds_label,
                'gemini_hp': gemini_hp, 'ds_hp': ds_hp,
                'gemini_reasoning': gemini_reasoning,
                'ds_reasoning': ds_reasoning,
                'audit_score': audit_score
            })
            
    df_res = pd.DataFrame(results)
    
    bias_md = "# DeepSeek Final Validation Analysis\n\n"
    
    if not stats_df.empty:
        bias_md += "## Execution Statistics\n"
        bias_md += f"- **Total candidates attempted**: 40\n"
        bias_md += f"- **Total candidates completed**: {len(df_res)}\n"
        bias_md += f"- **Number of 429 events**: {stats_df['429_events'].sum()}\n"
        bias_md += f"- **Total retries**: {stats_df['retries'].sum()}\n"
        success_rate = stats_df['success'].mean()
        bias_md += f"- **Success rate**: {success_rate:.1%}\n"
        bias_md += f"- **Avg latency/request**: {stats_df['latency'].mean():.2f}s\n"
        # throughput = (candidates / total time in hours)
        total_time_h = stats_df['latency'].sum() / 3600
        throughput = len(df_res) / total_time_h if total_time_h > 0 else 0
        bias_md += f"- **Effective throughput (cands/hour)**: {throughput:.1f}\n\n"
        
        bias_md += "## Throttling Metrics\n"
        bias_md += f"- **Avg input tokens/request**: {stats_df['in_tokens'].mean():.0f}\n"
        bias_md += f"- **Avg output tokens/request**: {stats_df['out_tokens'].mean():.0f}\n"
        corr = stats_df['in_tokens'].corr(stats_df['429_events'])
        bias_md += f"- **Correlation (Payload Size vs 429s)**: {corr:.3f}\n\n"
        
    bias_md += "## Score Bias\n"
    bias_md += f"- **Mean score delta**: {df_res['delta'].mean():.2f}\n"
    bias_md += f"- **Median score delta**: {df_res['delta'].median():.2f}\n"
    bias_md += f"- **Standard deviation**: {df_res['delta'].std():.2f}\n"
    
    # Text histogram
    bias_md += "\n### Delta Distribution Histogram\n```text\n"
    bins = [-100, -20, -10, -5, -1, 1, 5, 10, 20, 100]
    hist, edges = np.histogram(df_res['delta'], bins=bins)
    for i in range(len(hist)):
        bias_md += f"{edges[i]:>4} to {edges[i+1]:>4} : {'#' * hist[i]} ({hist[i]})\n"
    bias_md += "```\n\n"
    
    bias_md += "## Label Bias\n"
    upgrades = sum(df_res['direction'] == 'Upgrade')
    downgrades = sum(df_res['direction'] == 'Downgrade')
    matches = sum(df_res['direction'] == 'Match')
    bias_md += f"- **Upgraded relative to Gemini**: {upgrades}\n"
    bias_md += f"- **Downgraded relative to Gemini**: {downgrades}\n"
    bias_md += f"- **Matched Gemini exactly**: {matches}\n\n"
    
    bias_md += "## Honeypot Bias\n"
    tp = sum(df_res['ds_hp'] & df_res['gemini_hp'])
    tn = sum(~df_res['ds_hp'] & ~df_res['gemini_hp'])
    fp = sum(df_res['ds_hp'] & ~df_res['gemini_hp']) # DS says True, Gemini says False
    fn = sum(~df_res['ds_hp'] & df_res['gemini_hp']) # DS says False, Gemini says True
    bias_md += f"- **True Positives (Both HP)**: {tp}\n"
    bias_md += f"- **True Negatives (Both Normal)**: {tn}\n"
    bias_md += f"- **False Positives (DS called HP, Gemini didn't)**: {fp}\n"
    bias_md += f"- **False Negatives (DS missed Gemini's HP)**: {fn}\n\n"
    
    bias_md += "## Final Recommendation Matrix\n\n"
    label_agree = matches / len(df_res) if len(df_res) > 0 else 0
    hp_agree = (tp + tn) / len(df_res) if len(df_res) > 0 else 0
    avg_delta_abs = df_res['delta'].abs().mean()
    
    aud_appr = "APPROVE" if label_agree >= 0.8 else "REJECT"
    sec_appr = "APPROVE" if label_agree >= 0.9 and hp_agree >= 0.95 else "REJECT"
    full_appr = "APPROVE" if label_agree >= 0.95 and hp_agree >= 0.98 else "REJECT"
    
    bias_md += "| Use Case | Recommendation |\n"
    bias_md += "|---|---|\n"
    bias_md += f"| Independent auditing | **{aud_appr}** |\n"
    bias_md += f"| Secondary labeling | **{sec_appr}** |\n"
    bias_md += f"| Full replacement of Gemini | **{full_appr}** |\n\n"
    
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_bias_analysis.md"), "w") as f:
        f.write(bias_md)
        
    hd_md = "# DeepSeek High Disagreement Cases\n\n"
    for hd in high_disagreements:
        hd_md += f"## {hd['candidate_id']}\n"
        hd_md += f"- **Gemini**: {hd['gemini_label']} ({hd['gemini_score']}) | HP: {hd['gemini_hp']}\n"
        hd_md += f"- **DeepSeek**: {hd['ds_label']} ({hd['ds_score']}) | HP: {hd['ds_hp']}\n"
        hd_md += f"- **Score Delta**: {hd['delta']}\n"
        hd_md += f"- **DeepSeek Audit Score**: {hd['audit_score']}\n\n"
        hd_md += f"### Gemini Reasoning Snapshot\n> {hd['gemini_reasoning'][:400]}...\n\n"
        hd_md += f"### DeepSeek Reasoning Snapshot\n> {hd['ds_reasoning'][:400]}...\n\n"
        
        # Simple heuristic for judgment
        if hd['audit_score'] != 'N/A' and isinstance(hd['audit_score'], int) and hd['audit_score'] >= 80:
            judg = "Gemini likely correct (DeepSeek audited Gemini's original evaluation highly)"
        elif abs(hd['delta']) < 10 and hd['ds_hp'] == hd['gemini_hp']:
            judg = "Ambiguous (Scores are close)"
        else:
            judg = "DeepSeek likely correct (or requires manual human review)"
            
        hd_md += f"**Preliminary Judgment**: {judg}\n\n---\n"
        
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_high_disagreement_cases.md"), "w") as f:
        f.write(hd_md)
        
    print("Final Analysis Complete.")

if __name__ == "__main__":
    main()
