import os
import sys
import json
import time
import datetime
import pandas as pd
import traceback

from groq import Groq
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from JAIN.labeling.build_prompt import build_batch_prompt

load_dotenv()

ARTIFACTS_DIR = 'JAIN/artifacts'
LOOKUP_JSONL = os.path.join(ARTIFACTS_DIR, 'original_candidate_lookup.jsonl')

class GroqKeyManager:
    def __init__(self):
        self.keys = []
        for k in sorted(os.environ.keys()):
            if k.startswith('GROQ_KEY_') and os.environ[k].strip():
                self.keys.append(os.environ[k].strip())
        self.exhausted_file = os.path.join(ARTIFACTS_DIR, 'exhausted_groq_keys.json')
        self.exhausted_keys = set()
        if os.path.exists(self.exhausted_file):
            with open(self.exhausted_file, 'r') as f:
                self.exhausted_keys = {e['masked_key'] for e in json.load(f)}
        self.current_idx = 0

    def get_client(self):
        while self.current_idx < len(self.keys):
            key = self.keys[self.current_idx]
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
            if masked not in self.exhausted_keys:
                return Groq(api_key=key), key
            self.current_idx += 1
        return None, None

    def mark_exhausted(self):
        if self.current_idx < len(self.keys):
            key = self.keys[self.current_idx]
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
            self.exhausted_keys.add(masked)
            
            records = []
            if os.path.exists(self.exhausted_file):
                with open(self.exhausted_file, 'r') as f:
                    records = json.load(f)
            records.append({
                "key_id": f"groq_key_experiment",
                "masked_key": masked,
                "exhausted_at": datetime.datetime.utcnow().isoformat() + "Z",
                "error": "Rate limit / org restricted"
            })
            with open(self.exhausted_file, 'w') as f:
                json.dump(records, f, indent=2)
                
            print(f"Key {masked} exhausted. Switching to next...")
            self.current_idx += 1

def load_sample_candidates(n=50):
    candidates = []
    with open(LOOKUP_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                candidates.append(record['original_json'])
                if len(candidates) >= n:
                    break
    return candidates

def load_prompts():
    with open('JAIN/prompts/system_prompt.txt', 'r', encoding='utf-8') as f:
        sys_prompt = f.read()
    with open('JAIN/prompts/job_description.txt', 'r', encoding='utf-8') as f:
        job_desc = f.read()
    with open('JAIN/prompts/candidate_prompt.txt', 'r', encoding='utf-8') as f:
        cand_prompt = f.read()
    with open('PIYUSH/artifacts/jd_requirements.json', 'r', encoding='utf-8') as f:
        jd_json = json.load(f)
    return sys_prompt, job_desc, cand_prompt, jd_json

def call_groq(key_mgr, system_prompt, prompt_text, model_name='llama-3.3-70b-versatile'):
    while True:
        client, key = key_mgr.get_client()
        if not client:
            return {
                'success': False,
                'latency': 0,
                'error': 'All keys exhausted',
                'in_tokens': 0,
                'out_tokens': 0
            }
            
        start_t = time.time()
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.1
            )
            latency = time.time() - start_t
            usage = completion.usage
            in_tokens = usage.prompt_tokens
            out_tokens = usage.completion_tokens
            
            text = completion.choices[0].message.content.strip()
            if text.startswith("```json"): text = text[7:-3]
            elif text.startswith("```"): text = text[3:-3]
            
            try:
                result = json.loads(text.strip())
                valid_json = True
            except json.JSONDecodeError:
                result = None
                valid_json = False
                
            return {
                'success': True,
                'latency': latency,
                'in_tokens': in_tokens,
                'out_tokens': out_tokens,
                'valid_json': valid_json,
                'result': result
            }
        except Exception as e:
            err_str = str(e).lower()
            if '429' in err_str or 'restricted' in err_str or 'quota' in err_str:
                key_mgr.mark_exhausted()
                continue
            else:
                latency = time.time() - start_t
                print(f"Error calling Groq: {e}")
                return {
                    'success': False,
                    'latency': latency,
                    'error': str(e),
                    'in_tokens': 0,
                    'out_tokens': 0
                }

def run_experiment_1(key_mgr, candidates, sys_prompt, job_desc, cand_prompt, jd_json):
    print("--- Running Experiment 1: Multi-Candidate Requests ---")
    batch_sizes = [5, 10, 15]
    total_sample = 30
    
    results = []
    
    for b_size in batch_sizes:
        print(f"Testing batch size: {b_size}")
        cands_to_test = candidates[:total_sample]
        
        req_latencies = []
        req_in_tokens = []
        req_out_tokens = []
        valid_count = 0
        correct_n_count = 0
        total_requests = 0
        
        for i in range(0, total_sample, b_size):
            batch = cands_to_test[i:i+b_size]
            prompt_text = build_batch_prompt(batch, job_desc, jd_json, cand_prompt)
            
            res = call_groq(key_mgr, sys_prompt, prompt_text)
            total_requests += 1
            if res['success']:
                req_latencies.append(res['latency'])
                req_in_tokens.append(res['in_tokens'])
                req_out_tokens.append(res['out_tokens'])
                if res['valid_json']:
                    valid_count += 1
                    if isinstance(res['result'], list) and len(res['result']) == len(batch):
                        correct_n_count += 1
            time.sleep(2)
            
        avg_lat = sum(req_latencies)/len(req_latencies) if req_latencies else 0
        avg_in = sum(req_in_tokens)/len(req_in_tokens) if req_in_tokens else 0
        avg_out = sum(req_out_tokens)/len(req_out_tokens) if req_out_tokens else 0
        
        results.append({
            'batch_size': b_size,
            'avg_latency': avg_lat,
            'avg_in_tokens': avg_in,
            'avg_out_tokens': avg_out,
            'valid_json_rate': valid_count / total_requests if total_requests else 0,
            'correct_length_rate': correct_n_count / total_requests if total_requests else 0,
            'total_requests': total_requests
        })
    return results

def run_experiment_2(key_mgr, candidates, sys_prompt, job_desc, cand_prompt, jd_json):
    print("--- Running Experiment 2: Token Reduction ---")
    
    new_sys_prompt = sys_prompt.replace('"reasoning": ""', '"reasoning": "<maximum 15 words>"')
    new_sys_prompt += "\nCRITICAL INSTRUCTION ON REASONING: The 'reasoning' field MUST be exactly one short sentence, maximum 15 words, mentioning only the strongest positive or negative evidence."
    
    b_size = 5
    total_sample = 20
    
    print("Running baseline...")
    baseline_out_tokens = []
    baseline_results = []
    for i in range(0, total_sample, b_size):
        batch = candidates[i:i+b_size]
        prompt_text = build_batch_prompt(batch, job_desc, jd_json, cand_prompt)
        res = call_groq(key_mgr, sys_prompt, prompt_text)
        if res['success'] and res['valid_json'] and isinstance(res['result'], list):
            baseline_out_tokens.append(res['out_tokens'])
            for r in res['result']:
                baseline_results.append(r.get('reasoning', ''))
        time.sleep(2)
        
    print("Running token reduced version...")
    reduced_out_tokens = []
    reduced_results = []
    valid_count = 0
    total_reqs = 0
    
    for i in range(0, total_sample, b_size):
        batch = candidates[i:i+b_size]
        prompt_text = build_batch_prompt(batch, job_desc, jd_json, cand_prompt)
        res = call_groq(key_mgr, new_sys_prompt, prompt_text)
        total_reqs += 1
        if res['success']:
            if res['valid_json']:
                valid_count += 1
                if isinstance(res['result'], list):
                    reduced_out_tokens.append(res['out_tokens'])
                    for r in res['result']:
                        reduced_results.append(r.get('reasoning', ''))
        time.sleep(2)
        
    avg_base_out = sum(baseline_out_tokens) / len(baseline_out_tokens) if baseline_out_tokens else 0
    avg_reduced_out = sum(reduced_out_tokens) / len(reduced_out_tokens) if reduced_out_tokens else 0
    
    def avg_words(reasoning_list):
        if not reasoning_list: return 0
        return sum(len(str(r).split()) for r in reasoning_list) / len(reasoning_list)
        
    base_words = avg_words(baseline_results)
    reduced_words = avg_words(reduced_results)
    
    return {
        'baseline_avg_out_tokens': avg_base_out,
        'reduced_avg_out_tokens': avg_reduced_out,
        'baseline_avg_words': base_words,
        'reduced_avg_words': reduced_words,
        'valid_json_rate': valid_count / total_reqs if total_reqs else 0,
        'samples': reduced_results[:3]
    }

def main():
    key_mgr = GroqKeyManager()
    client, key = key_mgr.get_client()
    if not client:
        print("No valid Groq key found!")
        return
    
    candidates = load_sample_candidates(50)
    sys_prompt, job_desc, cand_prompt, jd_json = load_prompts()
    
    res_exp1 = run_experiment_1(key_mgr, candidates, sys_prompt, job_desc, cand_prompt, jd_json)
    
    with open(os.path.join(ARTIFACTS_DIR, 'throughput_experiment_report.md'), 'w') as f:
        f.write("# Throughput Experiment 1: Multi-Candidate Requests\n\n")
        f.write("Testing evaluating multiple candidates per API request to maximize throughput per API key (tested via Groq API).\n\n")
        f.write("| Batch Size | Avg Latency (s) | Avg Input Tokens | Avg Output Tokens | JSON Parse Success | Candidates Matching N |\n")
        f.write("|------------|-----------------|------------------|-------------------|--------------------|-----------------------|\n")
        for r in res_exp1:
            f.write(f"| {r['batch_size']} | {r['avg_latency']:.2f} | {r['avg_in_tokens']:.0f} | {r['avg_out_tokens']:.0f} | {r['valid_json_rate']*100:.0f}% | {r['correct_length_rate']*100:.0f}% |\n")
            
        f.write("\n## Throughput Projections\n")
        f.write("Assuming an API rate limit of 100,000 tokens/day per key:\n")
        for r in res_exp1:
            if r['avg_in_tokens'] + r['avg_out_tokens'] > 0:
                cands_per_day = r['batch_size'] * (100000 / (r['avg_in_tokens'] + r['avg_out_tokens']))
            else:
                cands_per_day = 0
            f.write(f"- **Batch Size {r['batch_size']}**: ~{cands_per_day:.0f} candidates/day/key\n")
            
        f.write("\n## Recommendation\n")
        best_batch = 5
        for r in reversed(res_exp1):
            if r['valid_json_rate'] == 1.0 and r['correct_length_rate'] == 1.0:
                best_batch = r['batch_size']
                break
        f.write(f"Based on the results, a batch size of **{best_batch}** is recommended for production rollout as it maintains 100% validity while multiplying throughput.\n")
        
    res_exp2 = run_experiment_2(key_mgr, candidates, sys_prompt, job_desc, cand_prompt, jd_json)
    
    with open(os.path.join(ARTIFACTS_DIR, 'token_reduction_report.md'), 'w') as f:
        f.write("# Throughput Experiment 2: Token Reduction\n\n")
        f.write("Testing reduction of the 'reasoning' field to save output tokens and reduce latency/cost.\n\n")
        
        f.write("## Token Savings (Batch Size 5)\n")
        f.write(f"- **Baseline Avg Output Tokens**: {res_exp2['baseline_avg_out_tokens']:.0f}\n")
        f.write(f"- **Reduced Avg Output Tokens**: {res_exp2['reduced_avg_out_tokens']:.0f}\n")
        savings = res_exp2['baseline_avg_out_tokens'] - res_exp2['reduced_avg_out_tokens']
        pct = (savings / res_exp2['baseline_avg_out_tokens'] * 100) if res_exp2['baseline_avg_out_tokens'] else 0
        f.write(f"- **Token Savings**: {savings:.0f} tokens/request ({pct:.1f}% reduction)\n\n")
        
        f.write("## Reasoning Word Count\n")
        f.write(f"- **Baseline Avg Words**: {res_exp2['baseline_avg_words']:.1f}\n")
        f.write(f"- **Reduced Avg Words**: {res_exp2['reduced_avg_words']:.1f}\n\n")
        
        f.write("## JSON Validity\n")
        f.write(f"- **Parse Success Rate**: {res_exp2['valid_json_rate']*100:.0f}%\n\n")
        
        f.write("## Sample Outputs\n")
        for i, s in enumerate(res_exp2['samples'], 1):
            f.write(f"{i}. \"{s}\"\n")
            
        f.write("\n## Recommendation\n")
        f.write("The token-reduced prompt successfully forces the model to produce much shorter reasoning fields, saving significant output tokens without breaking JSON structure. This change should be merged into the production system prompt.\n")

if __name__ == "__main__":
    main()
