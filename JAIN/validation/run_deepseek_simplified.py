import os
import json
import time
import requests
import pandas as pd
import sys

NVIDIA_API_KEY = "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"
MODEL_ID = "deepseek-ai/deepseek-v4-pro"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "JAIN", "artifacts")
PIYUSH_ARTIFACTS = os.path.join(BASE_DIR, "PIYUSH", "artifacts")
PROMPTS_DIR = os.path.join(BASE_DIR, "JAIN", "prompts")
CANDIDATES_FILE = os.path.join(BASE_DIR, "candidates.jsonl")

from JAIN.labeling.build_prompt import build_batch_prompt, compress_candidate

def call_deepseek(system_prompt, user_prompt, max_retries=7):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 4000
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data['choices'][0]['message']['content']
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            parsed = json.loads(content)
            
            return parsed, True
        except Exception as e:
            time.sleep(2 ** attempt)
            
    return None, False

def load_docx(path):
    import docx
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs])

def run_experiment():
    cands_data = {}
    with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                cands_data[obj['candidate_id']] = obj
                
    sample_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "deepseek_sample_combined.parquet"))
    jd_doc = load_docx(os.path.join(BASE_DIR, "job_description.docx"))
    redrob_doc = load_docx(os.path.join(BASE_DIR, "redrob_signals_doc.docx"))
    with open(os.path.join(PIYUSH_ARTIFACTS, "jd_requirements.json"), "r") as f:
        jd_json = json.load(f)
        
    out_labels_file = os.path.join(ARTIFACTS_DIR, "deepseek_labels.jsonl")
    out_audits_file = os.path.join(ARTIFACTS_DIR, "deepseek_audits.jsonl")
    
    existing_labels = set()
    if os.path.exists(out_labels_file):
        with open(out_labels_file, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    existing_labels.add(obj.get("candidate_id"))
                    
    existing_audits = set()
    if os.path.exists(out_audits_file):
        with open(out_audits_file, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    existing_audits.add(obj.get("candidate_id"))
                    
    with open(os.path.join(PROMPTS_DIR, "system_prompt.txt"), "r") as f:
        lbl_sys_prompt = f.read().replace("{{JOB_DESCRIPTION_FULL}}", jd_doc).replace("{{REDROB_SIGNALS_DOC}}", redrob_doc)
    with open(os.path.join(PROMPTS_DIR, "candidate_prompt.txt"), "r") as f:
        lbl_user_prompt = f.read()
    with open(os.path.join(PROMPTS_DIR, "audit_prompt.txt"), "r") as f:
        aud_sys_prompt = f.read()
        
    all_cids = sample_df['candidate_id'].tolist()
    
    for cid in all_cids:
        if cid in existing_labels: continue
        cand_raw = cands_data[cid]
        user_prompt = build_batch_prompt([cand_raw], jd_doc, jd_json, lbl_user_prompt)
        parsed, success = call_deepseek(lbl_sys_prompt, user_prompt)
        if success:
            with open(out_labels_file, "a") as f:
                eval_obj = parsed[0] if isinstance(parsed, list) else parsed
                f.write(json.dumps({"candidate_id": cid, "evaluation": eval_obj}) + "\n")
                
    for _, row in sample_df.iterrows():
        cid = row['candidate_id']
        if cid in existing_audits: continue
        cand_compressed = compress_candidate(cands_data[cid])
        gemini_eval = row.to_dict()
        user_prompt = f"=== JOB DESCRIPTION ===\n{jd_doc}\n\n=== CANDIDATE PROFILE ===\n{json.dumps(cand_compressed, indent=2)}\n\n=== RECRUITER EVALUATION ===\n{json.dumps(gemini_eval, indent=2)}\n"
        parsed, success = call_deepseek(aud_sys_prompt, user_prompt)
        if success:
            with open(out_audits_file, "a") as f:
                f.write(json.dumps({"candidate_id": cid, "audit": parsed}) + "\n")
                
    # Analysis
    ds_labels = {}
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
        
        # Check if Gemini-1 or Gemini-3
        eval_group = 'Gemini-1' if row.name < 20 else 'Gemini-3' # The first 20 are Gemini-1
        
        results.append({
            'group': eval_group,
            'delta': ds_score - gem_score,
            'label_match': gem_label == ds_label,
            'hp_match': gem_hp == ds_hp,
            'direction': direction
        })
        
    df = pd.DataFrame(results)
    
    out_md = "# DeepSeek Validation Summary\n\n"
    
    for grp in ['Gemini-1', 'Gemini-3']:
        grp_df = df[df['group'] == grp]
        if len(grp_df) == 0: continue
        out_md += f"## {grp} vs DeepSeek\n"
        out_md += f"- Fit label agreement %: {grp_df['label_match'].mean():.1%}\n"
        out_md += f"- Honeypot agreement %: {grp_df['hp_match'].mean():.1%}\n"
        out_md += f"- Avg score difference: {grp_df['delta'].mean():.2f}\n"
        out_md += f"- Upgrade count: {sum(grp_df['direction'] == 'Upgrade')}\n"
        out_md += f"- Downgrade count: {sum(grp_df['direction'] == 'Downgrade')}\n\n"
        
    overall_label = df['label_match'].mean() if len(df) > 0 else 0
    overall_hp = df['hp_match'].mean() if len(df) > 0 else 0
    
    auditing = "Yes" if overall_label >= 0.8 else "No"
    secondary = "Yes" if overall_label >= 0.9 and overall_hp >= 0.95 else "No"
    replacement = "Yes" if overall_label >= 0.95 and overall_hp >= 0.98 else "No"
    
    out_md += "## Recommendation\n"
    out_md += "DeepSeek suitable for:\n"
    out_md += f"{'✓' if auditing == 'Yes' else '✗'} auditing\n"
    out_md += f"{'✓' if secondary == 'Yes' else '✗'} secondary validation\n"
    out_md += f"{'✓' if replacement == 'Yes' else '✗'} replacing Gemini\n"
    
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_final_recommendation.md"), "w") as f:
        f.write(out_md)

if __name__ == "__main__":
    run_experiment()
