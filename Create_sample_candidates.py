import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_JSONL = os.path.join(BASE_DIR, "candidates.jsonl")
SAMPLE_JSONL = os.path.join(BASE_DIR, "sample_candidates_100.jsonl")

def create_sample():
    count = 0
    with open(CANDIDATES_JSONL, "r", encoding="utf-8") as fin, \
         open(SAMPLE_JSONL, "w", encoding="utf-8") as fout:
        for line in fin:
            if '"candidate_id"' in line:
                fout.write(line)
                count += 1
                if count >= 100:
                    break
    print(f"Created {SAMPLE_JSONL} with {count} candidates.")

if __name__ == "__main__":
    create_sample()
