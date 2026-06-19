import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "JAIN", "artifacts")

def run_experiment():
    sample_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "deepseek_sample_combined.parquet"))
    out_labels_file = os.path.join(ARTIFACTS_DIR, "deepseek_labels.jsonl")
    
    ds_labels = {}
    if os.path.exists(out_labels_file):
        with open(out_labels_file, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    ds_labels[obj['candidate_id']] = obj['evaluation']
                    
    results = []
    for _, row in sample_df.iterrows():
        cid = row['candidate_id']
        gem_score = row['fit_score']
        gem_label = row['fit_label']
        gem_hp = row['honeypot_label']
        
        if cid not in ds_labels: continue
        
        ds_score = ds_labels[cid].get('fit_score', 0)
        ds_label = ds_labels[cid].get('fit_label', 'Unknown')
        ds_hp = ds_labels[cid].get('honeypot_label', False)
        
        ranks = {'Reject':0, 'Weak Fit':1, 'Moderate Fit':2, 'Strong Fit':3, 'Excellent Fit':4}
        ds_rank = ranks.get(ds_label, -1)
        gem_rank = ranks.get(gem_label, -1)
        
        direction = "Match"
        if ds_rank > gem_rank: direction = "Upgrade"
        elif ds_rank < gem_rank: direction = "Downgrade"
        
        eval_group = 'Gemini-1' if row.name < 20 else 'Gemini-3'
        
        results.append({
            'group': eval_group,
            'delta': ds_score - gem_score,
            'label_match': gem_label == ds_label,
            'hp_match': gem_hp == ds_hp,
            'direction': direction
        })
        
    df = pd.DataFrame(results)
    
    out_md = f"# DeepSeek Validation Summary (Partial: {len(df)}/40 candidates)\n\n"
    
    if len(df) == 0:
        out_md += "No data collected yet.\n"
    else:
        for grp in ['Gemini-1', 'Gemini-3']:
            grp_df = df[df['group'] == grp]
            if len(grp_df) == 0: continue
            out_md += f"## {grp} vs DeepSeek (N={len(grp_df)})\n"
            out_md += f"- Fit label agreement %: {grp_df['label_match'].mean():.1%}\n"
            out_md += f"- Honeypot agreement %: {grp_df['hp_match'].mean():.1%}\n"
            out_md += f"- Avg score difference: {grp_df['delta'].mean():.2f}\n"
            out_md += f"- Upgrade count: {sum(grp_df['direction'] == 'Upgrade')}\n"
            out_md += f"- Downgrade count: {sum(grp_df['direction'] == 'Downgrade')}\n\n"
            
        overall_label = df['label_match'].mean()
        overall_hp = df['hp_match'].mean()
        
        auditing = "Yes" if overall_label >= 0.8 else "No"
        secondary = "Yes" if overall_label >= 0.9 and overall_hp >= 0.95 else "No"
        replacement = "Yes" if overall_label >= 0.95 and overall_hp >= 0.98 else "No"
        
        out_md += "## Recommendation\n"
        out_md += "DeepSeek suitable for:\n"
        out_md += f"{'✓' if auditing == 'Yes' else '✗'} auditing\n"
        out_md += f"{'✓' if secondary == 'Yes' else '✗'} secondary validation\n"
        out_md += f"{'✓' if replacement == 'Yes' else '✗'} replacing Gemini\n"
        
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_final_recommendation.md"), "w", encoding="utf-8") as f:
        f.write(out_md)

if __name__ == "__main__":
    run_experiment()
