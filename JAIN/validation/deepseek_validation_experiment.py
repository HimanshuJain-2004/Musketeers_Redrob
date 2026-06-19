import os
import json
import random
import time
import requests
import pandas as pd

# Configurations
NVIDIA_API_KEY = "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"
MODEL_ID = "deepseek-ai/deepseek-v4-pro"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "JAIN", "artifacts")
PROMPTS_DIR = os.path.join(BASE_DIR, "JAIN", "prompts")
CANDIDATES_FILE = os.path.join(BASE_DIR, "candidates.jsonl")
CHECKPOINT_FILE = os.path.join(ARTIFACTS_DIR, "llm_evaluations_checkpoint.jsonl")

# Load candidate data
def load_candidates():
    cands = {}
    with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            cands[obj['candidate_id']] = obj
    return cands

def load_evaluations():
    evals = []
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            evals.append(json.loads(line))
    return evals

def stratify_sample(eval_list, num_samples=20):
    # Categorize
    strong = []
    moderate = []
    weak = []
    reject = []
    honeypot = []
    
    for e in eval_list:
        ev = e['evaluation']
        if ev['honeypot_label']:
            honeypot.append(e)
        elif ev['fit_label'] in ['Excellent Fit', 'Strong Fit']:
            strong.append(e)
        elif ev['fit_label'] == 'Moderate Fit':
            moderate.append(e)
        elif ev['fit_label'] == 'Weak Fit':
            weak.append(e)
        elif ev['fit_label'] == 'Reject':
            reject.append(e)
            
    samples = []
    
    # Try to grab targets
    targets = [
        (strong, 4),
        (moderate, 5),
        (weak, 5),
        (reject, 4),
        (honeypot, 2)
    ]
    
    for pool, target_count in targets:
        # If we don't have enough, take all
        take = min(len(pool), target_count)
        samples.extend(random.sample(pool, take))
        
    # Fill remaining with random to hit num_samples
    remaining = num_samples - len(samples)
    if remaining > 0:
        pool_remaining = [e for e in eval_list if e not in samples]
        samples.extend(random.sample(pool_remaining, min(remaining, len(pool_remaining))))
        
    return samples

def call_deepseek(system_prompt, user_prompt):
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
    resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload)
    latency = time.time() - start
    
    return resp, latency

def phase_1_and_2():
    print("Loading data...")
    cands_data = load_candidates()
    evals = load_evaluations()
    
    gem1_evals = evals[:325]
    gem3_evals = evals[325:]
    
    print(f"Total Gem1: {len(gem1_evals)}, Total Gem3: {len(gem3_evals)}")
    
    gem1_sample = stratify_sample(gem1_evals, 20)
    gem3_sample = stratify_sample(gem3_evals, 20)
    
    print(f"Sampled Gem1: {len(gem1_sample)}, Sampled Gem3: {len(gem3_sample)}")
    
    # Convert to dataframe and save
    df1 = pd.DataFrame([e['evaluation'] for e in gem1_sample])
    df1['candidate_id'] = [e['candidate_id'] for e in gem1_sample]
    
    df3 = pd.DataFrame([e['evaluation'] for e in gem3_sample])
    df3['candidate_id'] = [e['candidate_id'] for e in gem3_sample]
    
    df1.to_parquet(os.path.join(ARTIFACTS_DIR, "deepseek_sample_gem1.parquet"))
    df3.to_parquet(os.path.join(ARTIFACTS_DIR, "deepseek_sample_gem3.parquet"))
    
    df_comb = pd.concat([df1, df3], ignore_index=True)
    df_comb.to_parquet(os.path.join(ARTIFACTS_DIR, "deepseek_sample_combined.parquet"))
    
    print("Phase 1 complete. Saved parquets.")
    
    # Smoke Test
    print("Running Smoke Test...")
    smoke_targets = []
    
    # Find 1 honeypot
    hp = next((e for e in gem1_sample + gem3_sample if e['evaluation']['honeypot_label']), None)
    if hp: smoke_targets.append(hp)
        
    # Find 1 strong
    st = next((e for e in gem1_sample + gem3_sample if e['evaluation']['fit_label'] in ['Excellent Fit', 'Strong Fit']), None)
    if st: smoke_targets.append(st)
        
    if len(smoke_targets) < 2:
        print("Warning: Could not find distinct strong/honeypot targets for smoke test. Using first 2.")
        smoke_targets = (gem1_sample + gem3_sample)[:2]
        
    # Build prompt
    with open(os.path.join(PROMPTS_DIR, "system_prompt.txt"), "r") as f:
        sys_prompt = f.read()
    with open(os.path.join(PROMPTS_DIR, "candidate_prompt.txt"), "r") as f:
        user_prompt_template = f.read()
        
    import docx
    def read_docx(path):
        d = docx.Document(path)
        return "\n".join([p.text for p in d.paragraphs])
        
    redrob_doc = read_docx(os.path.join(BASE_DIR, "redrob_signals_doc.docx"))
    jd_doc = read_docx(os.path.join(BASE_DIR, "job_description.docx"))
        
    sys_prompt = sys_prompt.replace("{{JOB_DESCRIPTION_FULL}}", jd_doc)
    sys_prompt = sys_prompt.replace("{{REDROB_SIGNALS_DOC}}", redrob_doc)
    
    import sys
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)
    from JAIN.labeling.build_prompt import build_batch_prompt
    
    piyush_artifacts = os.path.join(BASE_DIR, "PIYUSH", "artifacts")
    with open(os.path.join(piyush_artifacts, "jd_requirements.json"), "r") as f:
        jd_json = json.load(f)
        
    with open(os.path.join(ARTIFACTS_DIR, "deepseek_smoke_test.md"), "w") as f_out:
        f_out.write("# DeepSeek Smoke Test Results\n\n")
        
        for tgt in smoke_targets:
            cid = tgt['candidate_id']
            cand_raw = cands_data[cid]
            
            user_prompt = build_batch_prompt([cand_raw], jd_doc, jd_json, user_prompt_template)
            
            print(f"Calling DeepSeek for {cid}...")
            resp, lat = call_deepseek(sys_prompt, user_prompt)
            
            f_out.write(f"## Candidate: {cid}\n")
            f_out.write(f"**Gemini Original Label**: {tgt['evaluation']['fit_label']} (Honeypot: {tgt['evaluation']['honeypot_label']})\n")
            f_out.write(f"**Latency**: {lat:.2f}s\n")
            
            if resp.status_code == 200:
                data = resp.json()
                content = data['choices'][0]['message']['content']
                f_out.write(f"**HTTP Status**: 200\n\n")
                f_out.write("### Raw Response\n```json\n")
                f_out.write(content)
                f_out.write("\n```\n")
                
                # Check json validity
                try:
                    # Strip markdown blocks if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    parsed = json.loads(content)
                    f_out.write("**JSON Parse Status**: SUCCESS\n")
                except Exception as e:
                    f_out.write(f"**JSON Parse Status**: FAILED ({str(e)})\n")
                    
            else:
                f_out.write(f"**HTTP Status**: {resp.status_code}\n")
                f_out.write(f"**Error**: {resp.text}\n")
            
            f_out.write("---\n")
            
    print("Phase 2 complete. Wrote artifacts/deepseek_smoke_test.md.")

if __name__ == "__main__":
    phase_1_and_2()
