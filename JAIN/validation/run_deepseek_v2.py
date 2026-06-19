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

def call_deepseek(system_prompt, user_prompt, max_retries=10):
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
    events_429 = 0
    total_retries = 0
    for attempt in range(max_retries):
        try:
            resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                events_429 += 1
                total_retries += 1
                sleep_time = 2 ** attempt
                print(f"Rate limited (429). Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            in_toks = usage.get('prompt_tokens', len(user_prompt + system_prompt)//4)
            out_toks = usage.get('completion_tokens', len(content)//4)
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            parsed = json.loads(content)
            
            return parsed, time.time() - start, events_429, total_retries, in_toks, out_toks, True
        except Exception as e:
            print(f"Error calling DeepSeek (attempt {attempt+1}): {e}")
            total_retries += 1
            time.sleep(2 ** attempt)
            
    return None, time.time() - start, events_429, total_retries, 0, 0, False

def load_docx(path):
    import docx
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs])

def run_experiment():
    print("Loading data...")
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
    
    # Track existing
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
    
    # Collect execution stats
    stats = []
    
    # Phase 3
    print("--- Phase 3: Labeling ---")
    for cid in all_cids:
        if cid in existing_labels: continue
        print(f"Labeling {cid}...")
        cand_raw = cands_data[cid]
        user_prompt = build_batch_prompt([cand_raw], jd_doc, jd_json, lbl_user_prompt)
        
        parsed, lat, ev429, retries, intoks, outtoks, success = call_deepseek(lbl_sys_prompt, user_prompt)
        
        record = {
            "type": "labeling", "candidate_id": cid, "success": success, 
            "latency": lat, "429_events": ev429, "retries": retries, 
            "in_tokens": intoks, "out_tokens": outtoks
        }
        stats.append(record)
        
        if success:
            with open(out_labels_file, "a") as f:
                eval_obj = parsed[0] if isinstance(parsed, list) else parsed
                f.write(json.dumps({"candidate_id": cid, "evaluation": eval_obj, "stats": record}) + "\n")
                
    # Phase 4
    print("--- Phase 4: Auditing ---")
    for _, row in sample_df.iterrows():
        cid = row['candidate_id']
        if cid in existing_audits: continue
        print(f"Auditing {cid}...")
        cand_compressed = compress_candidate(cands_data[cid])
        gemini_eval = row.to_dict()
        
        user_prompt = f"=== JOB DESCRIPTION ===\n{jd_doc}\n\n"
        user_prompt += f"=== CANDIDATE PROFILE ===\n{json.dumps(cand_compressed, indent=2)}\n\n"
        user_prompt += f"=== RECRUITER EVALUATION ===\n{json.dumps(gemini_eval, indent=2)}\n"
        
        parsed, lat, ev429, retries, intoks, outtoks, success = call_deepseek(aud_sys_prompt, user_prompt)
        
        record = {
            "type": "auditing", "candidate_id": cid, "success": success, 
            "latency": lat, "429_events": ev429, "retries": retries, 
            "in_tokens": intoks, "out_tokens": outtoks
        }
        stats.append(record)
        
        if success:
            with open(out_audits_file, "a") as f:
                f.write(json.dumps({"candidate_id": cid, "audit": parsed, "stats": record}) + "\n")
                
    # Save stats
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_execution_stats.jsonl"), "a") as f:
        for s in stats:
            f.write(json.dumps(s) + "\n")
            
    print("Execution complete.")

if __name__ == "__main__":
    run_experiment()
