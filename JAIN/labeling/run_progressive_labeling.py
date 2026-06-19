import os
import sys
import json
import time
import datetime
import traceback
import warnings
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from groq import Groq

# Add project root to sys.path so JAIN module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from JAIN.labeling.build_prompt import build_batch_prompt

load_dotenv()

# ---------------------------------------------------------
# Configurations & Paths
# ---------------------------------------------------------
CONFIG_DIR = 'config'
ARTIFACTS_DIR = 'JAIN/artifacts'

EXHAUSTED_GEMINI_KEYS = os.path.join(ARTIFACTS_DIR, 'exhausted_gemini_keys.json')
EXHAUSTED_GROQ_KEYS = os.path.join(ARTIFACTS_DIR, 'exhausted_groq_keys.json')
PROGRESS_FILE = os.path.join(ARTIFACTS_DIR, 'labeling_progress.json')
BATCH_STATUS = os.path.join(ARTIFACTS_DIR, 'batch_status.json')
CHECKPOINT_FILE = os.path.join(ARTIFACTS_DIR, 'llm_evaluations_checkpoint.jsonl')

POOL_PARQUET = os.path.join(ARTIFACTS_DIR, 'labeling_pool_3000.parquet')
LOOKUP_JSONL = os.path.join(ARTIFACTS_DIR, 'original_candidate_lookup.jsonl')

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def load_json(path, default):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default
    return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_keys(prefix):
    keys_env = []
    for k in os.environ.keys():
        if k.startswith(prefix) and os.environ[k].strip():
            if prefix == 'GEMINI_KEY_':
                key_num = int(k.replace(prefix, '')) if k.replace(prefix, '').isdigit() else 0
                if key_num >= 69:
                    keys_env.append(k)
    keys_env.sort(key=lambda x: int(x.replace(prefix, '')) if x.replace(prefix, '').isdigit() else 999)
    return [os.environ[k].strip() for k in keys_env]

def mask_key(key):
    if len(key) <= 8:
        return "***"
    return key[:4] + "..." + key[-4:]

def get_completed_candidates():
    completed = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if 'candidate_id' in record:
                            completed.add(record['candidate_id'])
                    except:
                        pass
    return completed

# ---------------------------------------------------------
# API Management
# ---------------------------------------------------------
class APIKeyManager:
    def __init__(self, provider, env_prefix, exhausted_file):
        self.provider = provider
        self.keys = load_keys(env_prefix)
        self.exhausted_file = exhausted_file
        self.exhausted_records = load_json(self.exhausted_file, [])
        self.exhausted_keys = {r['masked_key'] for r in self.exhausted_records}
        self.current_index = 0
        self.calls_on_current_key = 0
        self._advance_to_valid_key()

    def _advance_to_valid_key(self):
        while self.current_index < len(self.keys):
            if mask_key(self.keys[self.current_index]) not in self.exhausted_keys:
                self.calls_on_current_key = 0
                break
            self.current_index += 1

    def get_current_key(self):
        if self.calls_on_current_key >= 20:
            print(f"[INFO] {self.provider} key {mask_key(self.keys[self.current_index])} reached 20 calls limit. Switching...")
            self.current_index += 1
            self._advance_to_valid_key()

        if self.current_index < len(self.keys):
            return self.keys[self.current_index]
        return None

    def mark_exhausted(self, error_msg):
        key = self.get_current_key()
        if key:
            masked = mask_key(key)
            record = {
                "key_id": f"{self.provider}_key_{self.current_index+1:02d}",
                "masked_key": masked,
                "exhausted_at": datetime.datetime.utcnow().isoformat() + "Z",
                "error": str(error_msg)
            }
            self.exhausted_records.append(record)
            self.exhausted_keys.add(masked)
            save_json(self.exhausted_file, self.exhausted_records)
            print(f"[WARN] {self.provider} key {masked} exhausted. Switching to next...")
            self.current_index += 1
            self._advance_to_valid_key()

    def increment_call(self):
        self.calls_on_current_key += 1

class Evaluator:
    def __init__(self):
        self.gemini_mgr = APIKeyManager('gemini', 'GEMINI_KEY_', EXHAUSTED_GEMINI_KEYS)
        self.groq_mgr = APIKeyManager('groq', 'GROQ_KEY_', EXHAUSTED_GROQ_KEYS)
        self.current_provider = 'gemini'
        with open('JAIN/prompts/system_prompt.txt', 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
        with open('JAIN/prompts/job_description.txt', 'r', encoding='utf-8') as f:
            self.job_description = f.read()
        with open('JAIN/prompts/candidate_prompt.txt', 'r', encoding='utf-8') as f:
            self.candidate_prompt_template = f.read()
        with open('PIYUSH/artifacts/jd_requirements.json', 'r', encoding='utf-8') as f:
            self.jd_json = json.load(f)

    def generate_prompt(self, candidates_json_list):
        # Use the exact build_prompt logic provided by the user
        prompt = build_batch_prompt(
            candidates_json_list,
            self.job_description,
            self.jd_json,
            self.candidate_prompt_template
        )
        return prompt

    def evaluate(self, candidates_json_list):
        prompt = self.generate_prompt(candidates_json_list)

        while True:
            if self.current_provider == 'gemini':
                key = self.gemini_mgr.get_current_key()
                if not key:
                    print("[INFO] All Gemini keys exhausted. Falling back to Groq.")
                    self.current_provider = 'groq'
                    continue
                try:
                    genai.configure(api_key=key)
                    model = genai.GenerativeModel(
                        'gemini-2.5-flash',
                        system_instruction=self.system_prompt
                    )
                    response = model.generate_content(prompt)
                    # Attempt simple parsing, assume response is stringified JSON
                    text = response.text.strip()
                    if text.startswith("```json"): text = text[7:-3]
                    elif text.startswith("```"): text = text[3:-3]
                    
                    result = json.loads(text.strip())
                    self.gemini_mgr.increment_call()
                    return result, 'gemini', self.gemini_mgr.current_index
                except Exception as e:
                    err_str = str(e).lower()
                    if '429' in err_str or 'exhausted' in err_str or 'quota' in err_str or '403' in err_str or '401' in err_str:
                        self.gemini_mgr.mark_exhausted(str(e))
                        continue
                    else:
                        print(f"Gemini API Error: {e}")
                        # If it's a structural or safety error, fallback or return error object
                        return {"error": str(e)}, 'gemini', self.gemini_mgr.current_index

            elif self.current_provider == 'groq':
                key = self.groq_mgr.get_current_key()
                if not key:
                    print("[ERROR] All Groq keys exhausted too. Cannot proceed.")
                    return None, None, None
                try:
                    client = Groq(api_key=key)
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )
                    
                    text = completion.choices[0].message.content.strip()
                    if text.startswith("```json"): text = text[7:-3]
                    elif text.startswith("```"): text = text[3:-3]
                    
                    result = json.loads(text.strip())
                    self.groq_mgr.increment_call()
                    return result, 'groq', self.groq_mgr.current_index
                except Exception as e:
                    err_str = str(e).lower()
                    if '429' in err_str or 'exhausted' in err_str or 'rate' in err_str or 'restricted' in err_str or 'quota' in err_str:
                        self.groq_mgr.mark_exhausted(str(e))
                        continue
                    else:
                        print(f"Groq API Error: {e}")
                        return {"error": str(e)}, 'groq', self.groq_mgr.current_index

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def main():
    print("Loading data...")
    pool_df = pd.read_parquet(POOL_PARQUET)
    
    lookup = {}
    with open(LOOKUP_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                lookup[record['candidate_id']] = record['original_json']

    completed_ids = get_completed_candidates()
    print(f"Found {len(completed_ids)} already completed candidates.")

    # Target: We only process set_1 initially. Let's process batch by batch.
    # To manage "Progressive Labeling", we will pick the first unfinished batch in set 1.
    set_1_df = pool_df[pool_df['labeling_set'] == 'set_1']
    
    batches = ['batch_1', 'batch_2', 'batch_3']
    target_batch = None
    
    batch_status = load_json(BATCH_STATUS, {})
    for b in batches:
        if b not in batch_status:
            b_size = len(set_1_df[set_1_df['labeling_batch'] == b])
            batch_status[b] = {"size": b_size, "status": "not_started", "completed": 0}
            
    for b in batches:
        if batch_status[b]['status'] != 'completed':
            target_batch = b
            if batch_status[b]['status'] == 'not_started':
                batch_status[b]['status'] = 'in_progress'
            break

    if not target_batch:
        print("All batches in set_1 are already completed!")
        return

    print(f"Targeting {target_batch} for evaluation.")
    target_candidates = set_1_df[set_1_df['labeling_batch'] == target_batch]['candidate_id'].tolist()
    
    # Filter out completed
    remaining_candidates = [cid for cid in target_candidates if cid not in completed_ids]
    print(f"{len(remaining_candidates)} candidates remaining in {target_batch}.")

    if len(remaining_candidates) == 0:
        batch_status[target_batch]['status'] = 'completed'
        save_json(BATCH_STATUS, batch_status)
        print(f"Batch {target_batch} just finished. Please run again to start the next batch or review.")
        return

    evaluator = Evaluator()
    progress = load_json(PROGRESS_FILE, {})

    # Process in batches of 3
    batch_size = 3
    for i in range(0, len(remaining_candidates), batch_size):
        chunk_cids = remaining_candidates[i:i+batch_size]
        chunk_jsons = [lookup[cid] for cid in chunk_cids]
        
        print(f"Evaluating {len(chunk_cids)} candidates: {', '.join(chunk_cids)}...")
        
        results, provider, key_idx = evaluator.evaluate(chunk_jsons)
        
        if results is None or (isinstance(results, dict) and "error" in results):
            print("Evaluation failed. Exiting gracefully.")
            print(results)
            break
            
        if isinstance(results, dict) and len(chunk_cids) == 1:
            results = [results]
            
        if not isinstance(results, list) or len(results) != len(chunk_cids):
            print(f"[ERROR] LLM did not return the exact number of objects. Expected {len(chunk_cids)}. Skipping chunk.")
            print(results)
            continue
            
        for cid, res in zip(chunk_cids, results):
            checkpoint_record = {
                "candidate_id": cid,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "provider": provider,
                "key_id": f"{provider}_key_{key_idx+1:02d}",
                "evaluation": res
            }
            
            with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(checkpoint_record) + "\n")
                
            completed_ids.add(cid)
            batch_status[target_batch]['completed'] += 1
            
            progress['current_set'] = 'set_1'
            progress['current_batch'] = target_batch
            progress['last_completed_candidate'] = cid
            progress['candidates_completed'] = len(completed_ids)
            progress['candidates_remaining'] = len(remaining_candidates) - (batch_status[target_batch]['completed'])
            progress['current_provider'] = provider
            progress['current_key_id'] = f"{provider}_key_{key_idx+1:02d}"
        
        save_json(PROGRESS_FILE, progress)
        save_json(BATCH_STATUS, batch_status)
        
        # Rate limit: 5 requests per minute -> 12 seconds per request
        time.sleep(12)
        
    # Final check on batch completion
    if batch_status[target_batch]['completed'] >= batch_status[target_batch]['size']:
        batch_status[target_batch]['status'] = 'completed'
        save_json(BATCH_STATUS, batch_status)
        print(f"Finished evaluating {target_batch}!")

if __name__ == "__main__":
    main()
