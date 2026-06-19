import os
import json
import time
import requests
import pandas as pd
import sys
import numpy as np

# Configurations
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

from JAIN.labeling.build_prompt import build_batch_prompt

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
    
    start = time.time()
    for attempt in range(max_retries):
        try:
            resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                sleep_time = 2 ** attempt
                print(f"Rate limited (429). Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data['choices'][0]['message']['content']
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            parsed = json.loads(content)
            return parsed, time.time() - start
        except Exception as e:
            print(f"Error calling DeepSeek (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
            
    return None, time.time() - start

def load_docx(path):
    import docx
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs])

def phase_3_labeling(cands_data, sample_df, jd_doc, redrob_doc, jd_json):
    print("--- Phase 3: DeepSeek Independent Labeling ---")
    out_file = os.path.join(ARTIFACTS_DIR, "deepseek_labels.jsonl")
    
    existing = set()
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if isinstance(obj, list) and len(obj) > 0:
                        existing.add(obj[0].get("candidate_id"))
                    elif isinstance(obj, dict):
                        existing.add(obj.get("candidate_id"))

    with open(os.path.join(PROMPTS_DIR, "system_prompt.txt"), "r") as f:
        sys_prompt = f.read().replace("{{JOB_DESCRIPTION_FULL}}", jd_doc).replace("{{REDROB_SIGNALS_DOC}}", redrob_doc)
    
    with open(os.path.join(PROMPTS_DIR, "candidate_prompt.txt"), "r") as f:
        user_prompt_template = f.read()
        
    all_cids = sample_df['candidate_id'].tolist()
    
    for cid in all_cids:
        if cid in existing:
            continue
            
        print(f"Labeling {cid}...")
        cand_raw = cands_data[cid]
        user_prompt = build_batch_prompt([cand_raw], jd_doc, jd_json, user_prompt_template)
        
        parsed, lat = call_deepseek(sys_prompt, user_prompt)
        if parsed:
            with open(out_file, "a") as f:
                f.write(json.dumps({"candidate_id": cid, "evaluation": parsed[0] if isinstance(parsed, list) else parsed, "latency": lat}) + "\n")
        else:
            print(f"Failed to label {cid}")
            
def phase_4_auditing(cands_data, sample_df, jd_doc):
    print("--- Phase 4: DeepSeek Auditing Gemini Labels ---")
    out_file = os.path.join(ARTIFACTS_DIR, "deepseek_audits.jsonl")
    
    existing = set()
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    existing.add(obj.get("candidate_id"))
                    
    with open(os.path.join(PROMPTS_DIR, "audit_prompt.txt"), "r") as f:
        sys_prompt = f.read()
        
    from JAIN.labeling.build_prompt import compress_candidate
        
    for _, row in sample_df.iterrows():
        cid = row['candidate_id']
        if cid in existing:
            continue
            
        print(f"Auditing {cid}...")
        cand_compressed = compress_candidate(cands_data[cid])
        gemini_eval = row.to_dict() # the evaluation dict
        
        user_prompt = f"=== JOB DESCRIPTION ===\n{jd_doc}\n\n"
        user_prompt += f"=== CANDIDATE PROFILE ===\n{json.dumps(cand_compressed, indent=2)}\n\n"
        user_prompt += f"=== RECRUITER EVALUATION ===\n{json.dumps(gemini_eval, indent=2)}\n"
        
        parsed, lat = call_deepseek(sys_prompt, user_prompt)
        if parsed:
            with open(out_file, "a") as f:
                f.write(json.dumps({"candidate_id": cid, "audit": parsed, "latency": lat}) + "\n")
        else:
            print(f"Failed to audit {cid}")

def phase_5_and_6(sample_df):
    print("--- Phase 5 & 6: Analysis & Bias Report ---")
    
    # Load DS labels
    ds_labels = {}
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_labels.jsonl"), "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                ds_labels[obj['candidate_id']] = obj['evaluation']
                
    # Load DS audits
    ds_audits = {}
    audit_file = os.path.join(ARTIFACTS_DIR, "deepseek_audits.jsonl")
    if os.path.exists(audit_file):
        with open(audit_file, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    ds_audits[obj['candidate_id']] = obj['audit']
                
    results = []
    high_disagreements = []
    
    for _, row in sample_df.iterrows():
        cid = row['candidate_id']
        gemini_score = row['fit_score']
        gemini_label = row['fit_label']
        gemini_hp = row['honeypot_label']
        
        if cid not in ds_labels: continue
        
        ds_score = ds_labels[cid].get('fit_score', 0)
        ds_label = ds_labels[cid].get('fit_label', 'Unknown')
        ds_hp = ds_labels[cid].get('honeypot_label', False)
        
        delta = ds_score - gemini_score
        label_match = (gemini_label == ds_label)
        hp_match = (gemini_hp == ds_hp)
        
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
            'category': cat,
            'gemini_label': gemini_label,
            'ds_label': ds_label
        })
        
        if abs(delta) > 15 or not label_match or not hp_match:
            high_disagreements.append({
                'candidate_id': cid,
                'gemini_score': gemini_score, 'ds_score': ds_score, 'delta': delta,
                'gemini_label': gemini_label, 'ds_label': ds_label,
                'gemini_hp': gemini_hp, 'ds_hp': ds_hp,
                'ds_reasoning': ds_labels[cid].get('reasoning', ''),
                'audit_score': ds_audits.get(cid, {}).get('audit_score', 'N/A')
            })
            
    df_res = pd.DataFrame(results)
    
    bias_md = "# DeepSeek Bias Analysis\n\n"
    bias_md += f"Total Validated: {len(df_res)}\n"
    bias_md += f"Overall Label Agreement: {df_res['label_match'].mean():.1%}\n"
    bias_md += f"Overall Honeypot Agreement: {df_res['hp_match'].mean():.1%}\n"
    bias_md += f"Mean Score Delta (DS - Gemini): {df_res['delta'].mean():.2f}\n\n"
    
    bias_md += "## Category Breakdown\n"
    for cat in df_res['category'].unique():
        cat_df = df_res[df_res['category'] == cat]
        bias_md += f"### {cat} (n={len(cat_df)})\n"
        bias_md += f"- Label Match: {cat_df['label_match'].mean():.1%}\n"
        bias_md += f"- Mean Delta: {cat_df['delta'].mean():.2f}\n\n"
        
    avg_delta = df_res['delta'].mean()
    bias = "No systematic bias"
    if avg_delta < -3: bias = "Conservative scoring bias"
    elif avg_delta > 3: bias = "Liberal scoring bias"
        
    bias_md += f"## Conclusion\n"
    bias_md += f"DeepSeek exhibits: **{bias}**.\n"
    
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_bias_analysis.md"), "w") as f:
        f.write(bias_md)
        
    hd_md = "# DeepSeek High Disagreement Cases\n\n"
    for hd in high_disagreements:
        hd_md += f"## {hd['candidate_id']}\n"
        hd_md += f"- Gemini: {hd['gemini_label']} ({hd['gemini_score']}) | HP: {hd['gemini_hp']}\n"
        hd_md += f"- DeepSeek: {hd['ds_label']} ({hd['ds_score']}) | HP: {hd['ds_hp']}\n"
        hd_md += f"- Score Delta: {hd['delta']}\n"
        hd_md += f"- DeepSeek Audit Score of Gemini: {hd['audit_score']}\n"
        hd_md += f"- DeepSeek Reasoning: {hd['ds_reasoning']}\n\n"
        
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_high_disagreement_cases.md"), "w") as f:
        f.write(hd_md)

def main():
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
        
    phase_3_labeling(cands_data, sample_df, jd_doc, redrob_doc, jd_json)
    phase_4_auditing(cands_data, sample_df, jd_doc)
    phase_5_and_6(sample_df)

if __name__ == "__main__":
    main()
