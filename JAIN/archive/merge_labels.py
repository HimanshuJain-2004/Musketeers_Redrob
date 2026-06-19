import os
import pandas as pd
import json

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ARTIFACTS_DIR = os.path.join(WORKSPACE_ROOT, 'JAIN', 'artifacts')

def merge_labels():
    all_records = []
    
    files_to_merge = [
        'stage_A_labels.jsonl',
        'stage_TOP20_checkpoint.jsonl',
        'stage_OVERCORRECTION_TEST_checkpoint.jsonl',
        'stage_0_checkpoint.jsonl'
    ]
    
    for filename in files_to_merge:
        filepath = os.path.join(ARTIFACTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filename}, not found.")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    # Some files might not have candidate_id inside the JSON, but they do!
                    if 'candidate_id' in record:
                        all_records.append(record)
                except json.JSONDecodeError:
                    pass
                    
    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df)} total labels.")
    
    # Drop duplicates in case candidates were evaluated in multiple stages
    df = df.drop_duplicates(subset=['candidate_id'], keep='last')
    print(f"After dropping duplicates, {len(df)} unique candidates labeled.")
    
    out_path = os.path.join(ARTIFACTS_DIR, 'gold_labels.parquet')
    df.to_parquet(out_path, index=False)
    print(f"Saved merged labels to {out_path}")

if __name__ == '__main__':
    merge_labels()
