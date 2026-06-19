import os
import sys
import json
import asyncio
import pandas as pd
import argparse
from build_prompt import build_batch_prompt
from validate_response import validate_response
from keys import GROQ_KEYS, GEMINI_KEYS, next_groq_key, next_gemini_key



def get_stage_0_candidates(sampled_df, suspicious_df):
    elite = sampled_df[sampled_df['sample_source'] == 'elite'].head(5)
    suspicious = suspicious_df.head(5)
    
    stratified = sampled_df[sampled_df['sample_source'] == 'stratified']
    medium = stratified[stratified['semantic_bucket'] == 'Middle 50%'].head(5)
    weak = stratified[stratified['semantic_bucket'] == 'Bottom 25%'].head(5)
    
    # Suspicious candidates in sampled_df are marked as 'suspicious' or we use candidate_ids
    # Actually, let's just get the IDs
    elite_ids = elite['candidate_id'].tolist()
    susp_ids = suspicious['candidate_id'].tolist()
    med_ids = medium['candidate_id'].tolist()
    weak_ids = weak['candidate_id'].tolist()
    
    return elite_ids + susp_ids + med_ids + weak_ids


def get_next_groq_key():
    return next_groq_key()

def get_next_gemini_key():
    return next_gemini_key()

def setup_clients():
    gemini_key = os.environ.get("GEMINI_API_KEY") or next_gemini_key()
    groq_key   = os.environ.get("GROQ_API_KEY")   or next_groq_key()
    return gemini_key, groq_key

async def call_gemini(prompt, sys_prompt, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=sys_prompt
    )
    # Using response_mime_type to force JSON
    response = await model.generate_content_async(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "max_output_tokens": 8192
        }
    )
    return response.text

async def call_groq(prompt, sys_prompt, api_key):
    import os
    from groq import AsyncGroq
    
    # Workaround for async event loop in Windows
    client = AsyncGroq(api_key=api_key)
    
    response = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        response_format={"type": "json_object"} # We will wrap the prompt array in an object
    )
    return response.choices[0].message.content

async def evaluate_batch(batch_raw, jd_text, jd_json, sys_prompt, cand_prompt_template, api_key, provider):
    prompt = build_batch_prompt(batch_raw, jd_text, jd_json, cand_prompt_template)
    
    if provider == 'groq':
        # Groq's json_object format requires the output to be an object, not an array.
        # We will wrap it.
        prompt += "\n\nCRITICAL: You must return a JSON object with a single key 'evaluations' that contains the JSON ARRAY of your candidate evaluations."
        
    retries = 3
    for attempt in range(retries):
        try:
            if provider == 'gemini':
                res_text = await call_gemini(prompt, sys_prompt, api_key)
            elif provider == 'groq':
                res_text = await call_groq(prompt, sys_prompt, api_key)
                # Extract array from {'evaluations': [...]}
                try:
                    res_json = json.loads(res_text)
                    if 'evaluations' in res_json:
                        res_text = json.dumps(res_json['evaluations'])
                except Exception:
                    pass
                
            is_valid, msg, data = validate_response(res_text, len(batch_raw))
            if is_valid:
                for i, cand in enumerate(batch_raw):
                    data[i]['candidate_id'] = cand['candidate_id']
                return data
            else:
                print(f"Validation failed on attempt {attempt+1}: {msg}")
        except Exception as e:
            err_msg = str(e)
            print(f"API Error on attempt {attempt+1}: {err_msg}")
            if '429' in err_msg or 'rate_limit_exceeded' in err_msg:
                print("Rate limit hit! Switching API key...")
                if provider == 'gemini':
                    api_key = get_next_gemini_key()
                elif provider == 'groq':
                    api_key = get_next_groq_key()
            
        await asyncio.sleep(2 ** attempt)
        
    return None

def get_stage_A_candidates(sampled_df, suspicious_df):
    elite = sampled_df[sampled_df['sample_source'] == 'elite'].head(100)
    suspicious = suspicious_df.head(100)
    
    stratified = sampled_df[sampled_df['sample_source'] == 'stratified']
    medium = stratified[stratified['semantic_bucket'] == 'Middle 50%'].head(150)
    weak = stratified[stratified['semantic_bucket'] == 'Bottom 25%'].head(150)
    
    elite_ids = elite['candidate_id'].tolist()
    susp_ids = suspicious['candidate_id'].tolist()
    med_ids = medium['candidate_id'].tolist()
    weak_ids = weak['candidate_id'].tolist()
    
    return elite_ids + susp_ids + med_ids + weak_ids

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, choices=['0', 'A', 'B', 'C', 'D', 'TOP20', 'OVERCORRECTION_TEST', 'GENUINE_TEST', 'STRONG_FIT_VALIDATION'], required=True)
    args = parser.parse_args()
    
    gemini_key, groq_key = setup_clients()
    
    WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    ARTIFACTS_DIR = os.path.join(WORKSPACE_ROOT, 'JAIN', 'artifacts')
    
    print("Loading data...")
    sampled_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'sampled_candidates.parquet'))
    suspicious_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'suspicious_candidates.parquet'))
    
    with open(os.path.join(WORKSPACE_ROOT, 'JAIN', 'prompts', 'system_prompt.txt'), 'r') as f:
        sys_prompt = f.read()
    with open(os.path.join(WORKSPACE_ROOT, 'JAIN', 'prompts', 'candidate_prompt.txt'), 'r') as f:
        cand_prompt = f.read()
    with open(os.path.join(WORKSPACE_ROOT, 'JAIN', 'prompts', 'job_description.txt'), 'r', encoding='utf-8') as f:
        jd_text = f.read()
    with open(os.path.join(WORKSPACE_ROOT, 'PIYUSH', 'artifacts', 'jd_requirements.json'), 'r') as f:
        jd_json = json.load(f)
        
    if args.stage == '0':
        print("Stage 0: Calibration (20 candidates) using Gemini")
        if not gemini_key:
            print("ERROR: GEMINI_API_KEY required for Stage 0")
            sys.exit(1)
        target_ids = get_stage_0_candidates(sampled_df, suspicious_df)
        provider = 'gemini'
        api_key = gemini_key
        batch_size = 5
    elif args.stage == 'A':
        print("Stage A: 500 candidates using Groq")
        if not groq_key:
            print("ERROR: GROQ_API_KEY required for Stage A")
            sys.exit(1)
        target_ids = get_stage_A_candidates(sampled_df, suspicious_df)
        provider = 'groq'
        api_key = groq_key
        batch_size = 5
    elif args.stage == 'TOP20':
        print("Stage TOP20: Re-evaluating top 20 candidates using Gemini")
        df_gold = pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'gold_labels.parquet'))
        target_ids = df_gold.sort_values(by='fit_score', ascending=False).head(20)['candidate_id'].tolist()
        provider = 'gemini'
        api_key = gemini_key
        batch_size = 5
    elif args.stage == 'OVERCORRECTION_TEST':
        print("Stage OVERCORRECTION_TEST: Testing overcorrection on 60 candidates")
        
        # Group 1: 20 suspicious candidates
        group1 = suspicious_df.head(20)['candidate_id'].tolist()
        
        # Group 2: 20 random candidates from sampled_df
        group2 = sampled_df.sample(n=20, random_state=42)['candidate_id'].tolist()
        
        # Group 3: 20 genuine retrieval/search engineers
        tech_keywords = ['engineer', 'developer', 'scientist', 'architect']
        is_tech = sampled_df['current_role'].str.lower().apply(lambda x: any(k in str(x) for k in tech_keywords))
        is_retrieval = sampled_df['retrieval_score'] > 0
        group3_df = sampled_df[is_tech & is_retrieval]
        # Avoid overlaps
        group3_df = group3_df[~group3_df['candidate_id'].isin(group1 + group2)]
        group3 = group3_df.head(20)['candidate_id'].tolist()
        
        target_ids = group1 + group2 + group3
        provider = 'gemini'
        api_key = gemini_key
        batch_size = 5
    elif args.stage == 'GENUINE_TEST':
        print("Stage GENUINE_TEST: Testing overcorrection on 5 known genuine AI Engineers")
        
        target_ids = ['CAND_0060072', 'CAND_0093912', 'CAND_0030827', 'CAND_0093547', 'CAND_0098846']
        provider = 'groq'
        api_key = groq_key
        batch_size = 5
    elif args.stage == 'STRONG_FIT_VALIDATION':
        print("Stage STRONG_FIT_VALIDATION: Evaluating 20 selected strong candidates using Groq")
        target_ids = [
            'CAND_0093912', 'CAND_0093547', 'CAND_0030827', 'CAND_0095619', 'CAND_0098846',
            'CAND_0044222', 'CAND_0061175', 'CAND_0060257', 'CAND_0005311', 'CAND_0015057',
            'CAND_0017590', 'CAND_0023076', 'CAND_0039308', 'CAND_0040092', 'CAND_0052827',
            'CAND_0064904', 'CAND_0095567', 'CAND_0006354', 'CAND_0048690', 'CAND_0057742'
        ]
        provider = 'groq'
        api_key = groq_key
        batch_size = 5
    else:
        print(f"Stage {args.stage} not fully implemented yet.")
        sys.exit(0)
        
    # Load raw JSONL for target IDs only
    print("Loading raw candidates...")
    target_raw = {}
    jsonl_path = os.path.join(WORKSPACE_ROOT, 'candidates.jsonl')
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            cid = record.get('candidate_id')
            if cid in target_ids:
                target_raw[cid] = record
                
    raw_list = [target_raw[cid] for cid in target_ids if cid in target_raw]
    
    print(f"Evaluating {len(raw_list)} candidates in batches of {batch_size}...")
    
    results = []
    
    # Determine out_file
    if args.stage == '0':
        out_file = os.path.join(ARTIFACTS_DIR, 'stage_0_checkpoint.jsonl')
    elif args.stage == 'A':
        out_file = os.path.join(ARTIFACTS_DIR, 'stage_A_labels.jsonl')
    else:
        out_file = os.path.join(ARTIFACTS_DIR, f'stage_{args.stage}_checkpoint.jsonl')
        
    # Clear file
    with open(out_file, 'w') as f:
        pass
    
    for i in range(0, len(raw_list), batch_size):
        batch = raw_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} / {(len(raw_list) + batch_size - 1)//batch_size}...")
        br = await evaluate_batch(batch, jd_text, jd_json, sys_prompt, cand_prompt, api_key, provider)
        if br:
            results.extend(br)
            with open(out_file, 'a') as f:
                for r in br:
                    f.write(json.dumps(r) + '\n')
        if provider == 'gemini':
            await asyncio.sleep(4) # Rate limit safety
        elif provider == 'groq':
            await asyncio.sleep(6) # Groq rate limit safety (14k TPM)

            
    print(f"Successfully evaluated {len(results)} candidates.")
    
    if args.stage == 'A':
        # Also save to parquet
        if results:
            df = pd.DataFrame(results)
            pq_file = os.path.join(ARTIFACTS_DIR, 'gold_labels.parquet')
            df.to_parquet(pq_file)
            print(f"Saved to {out_file} and {pq_file}")

if __name__ == "__main__":
    asyncio.run(main())
