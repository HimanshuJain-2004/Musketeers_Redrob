import os
import sys
import json
import time
import random
import traceback
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import re

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(ROOT_DIR)

# File Paths
CHECKPOINT_FILE = os.path.join(ROOT_DIR, 'JAIN/artifacts/llm_evaluations_checkpoint.jsonl')
LOOKUP_FILE = os.path.join(ROOT_DIR, 'JAIN/artifacts/original_candidate_lookup.jsonl')
JD_PATH = os.path.join(ROOT_DIR, 'JAIN/prompts/job_description.txt')
AUDIT_PROMPT_PATH = os.path.join(ROOT_DIR, 'JAIN/prompts/audit_prompt.txt')
EXHAUSTED_KEYS_FILE = os.path.join(ROOT_DIR, 'JAIN/artifacts/exhausted_groq_keys.json')
AUDIT_RESULTS_FILE = os.path.join(ROOT_DIR, 'JAIN/artifacts/batch1_audit_results.jsonl')
AUDIT_SAMPLE_JSONL = os.path.join(ROOT_DIR, 'JAIN/artifacts/batch1_audit_sample.jsonl')
AUDIT_SAMPLE_CSV = os.path.join(ROOT_DIR, 'JAIN/artifacts/batch1_audit_sample.csv')
AUDIT_REPORT_FILE = os.path.join(ROOT_DIR, 'JAIN/artifacts/batch1_audit_report.md')

RANDOM_SEED = 42

def load_json(filepath, default):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default
    return default

def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def get_groq_keys():
    load_dotenv()
    keys = []
    for k, v in os.environ.items():
        if k.startswith('GROQ_KEY') or k.startswith('GROQ_API_KEY'):
            keys.append({'id': k, 'key': v})
    return keys

def extract_json(text):
    # Try to find json block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return None

def main():
    print("Starting Batch 1 Audit...")
    random.seed(RANDOM_SEED)

    # 1. Load data
    if not os.path.exists(CHECKPOINT_FILE):
        print("Error: llm_evaluations_checkpoint.jsonl not found.")
        return

    evaluations = []
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                evaluations.append(json.loads(line))
    
    print(f"Loaded {len(evaluations)} completed evaluations.")
    
    # Randomly select 30
    if len(evaluations) < 30:
        print(f"Only {len(evaluations)} candidates available. Auditing all.")
        audit_sample = evaluations
    else:
        audit_sample = random.sample(evaluations, 30)
    
    audit_sample_ids = {c['candidate_id'] for c in audit_sample}
    
    # Load original profiles
    original_profiles = {}
    with open(LOOKUP_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            cid = rec['candidate_id']
            if cid in audit_sample_ids:
                original_profiles[cid] = rec['original_json']

    # Export sample
    print("Exporting audit sample...")
    with open(AUDIT_SAMPLE_JSONL, 'w', encoding='utf-8') as f:
        for ev in audit_sample:
            cid = ev['candidate_id']
            record = {
                'candidate_id': cid,
                'recruiter_evaluation': ev,
                'original_candidate_json': original_profiles.get(cid, {})
            }
            f.write(json.dumps(record) + '\n')
            
    # Also save CSV for easy viewing
    csv_data = []
    for ev in audit_sample:
        cid = ev['candidate_id']
        csv_data.append({
            'candidate_id': cid,
            'fit_score': ev['evaluation'].get('fit_score'),
            'fit_label': ev['evaluation'].get('fit_label'),
            'honeypot': ev['evaluation'].get('honeypot_label'),
            'reasoning': ev['evaluation'].get('reasoning')
        })
    pd.DataFrame(csv_data).to_csv(AUDIT_SAMPLE_CSV, index=False)

    # 2. Setup Groq Audit
    with open(JD_PATH, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    with open(AUDIT_PROMPT_PATH, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    groq_keys = get_groq_keys()
    if not groq_keys:
        print("Error: No GROQ_API_KEY found in .env")
        return
        
    exhausted_keys = load_json(EXHAUSTED_KEYS_FILE, [])
    active_keys = [k for k in groq_keys if k['id'] not in exhausted_keys]
    if not active_keys:
        print("All Groq keys are exhausted.")
        return

    # Load existing audits
    completed_audits = set()
    if os.path.exists(AUDIT_RESULTS_FILE):
        with open(AUDIT_RESULTS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed_audits.add(rec['candidate_id'])
                    
    print(f"Found {len(completed_audits)} previously completed audits.")
    
    key_idx = 0
    client = Groq(api_key=active_keys[key_idx]['key'])
    
    # 3. Execution
    with open(AUDIT_RESULTS_FILE, 'a', encoding='utf-8') as results_file:
        for idx, ev in enumerate(audit_sample):
            cid = ev['candidate_id']
            if cid in completed_audits:
                continue
                
            print(f"Auditing {idx+1}/{len(audit_sample)}: {cid}")
            original_cand = original_profiles.get(cid, {})
            
            user_prompt = f"JOB DESCRIPTION:\n{jd_text}\n\n"
            user_prompt += f"CANDIDATE PROFILE:\n{json.dumps(original_cand, indent=2)}\n\n"
            user_prompt += f"RECRUITER EVALUATION:\n{json.dumps(ev['evaluation'], indent=2)}\n\n"
            user_prompt += "Please audit the evaluation following the system prompt and return ONLY the JSON."

            while True:
                try:
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1
                    )
                    
                    response_text = completion.choices[0].message.content
                    parsed_json = extract_json(response_text)
                    
                    if not parsed_json:
                        print("Failed to parse JSON, retrying...")
                        time.sleep(2)
                        continue
                        
                    # Save audit result
                    parsed_json['candidate_id'] = cid
                    parsed_json['original_fit_score'] = ev['evaluation'].get('fit_score')
                    parsed_json['original_fit_label'] = ev['evaluation'].get('fit_label')
                    parsed_json['original_honeypot'] = ev['evaluation'].get('honeypot_label')
                    
                    results_file.write(json.dumps(parsed_json) + '\n')
                    results_file.flush()
                    completed_audits.add(cid)
                    break
                    
                except Exception as e:
                    err_str = str(e)
                    print(f"Error: {err_str}")
                    if "400" in err_str or "401" in err_str or "403" in err_str or "rate limit" in err_str.lower() or "429" in err_str or "restricted" in err_str.lower():
                        exhausted_keys.append(active_keys[key_idx]['id'])
                        save_json(EXHAUSTED_KEYS_FILE, exhausted_keys)
                        key_idx += 1
                        if key_idx >= len(active_keys):
                            print("All Groq keys exhausted during run.")
                            return
                        print(f"Rotating to key {active_keys[key_idx]['id']}")
                        client = Groq(api_key=active_keys[key_idx]['key'])
                        time.sleep(2)
                    else:
                        traceback.print_exc()
                        time.sleep(5)
                        break
            
            time.sleep(1) # Base rate limiting

    print("Audit execution completed. Generating report...")
    generate_report()

def generate_report():
    if not os.path.exists(AUDIT_RESULTS_FILE):
        return
        
    audits = []
    with open(AUDIT_RESULTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): audits.append(json.loads(line))
            
    if not audits:
        return
        
    total = len(audits)
    avg_score = sum(a.get('audit_score', 0) for a in audits) / total
    score_agree = sum(1 for a in audits if a.get('score_agreement', False))
    fit_agree = sum(1 for a in audits if a.get('fit_label_agreement', False))
    hp_agree = sum(1 for a in audits if a.get('honeypot_agreement', False))
    
    # Distributions
    labels = {}
    hps = 0
    hp_prob_sum = 0
    qualities = {}
    scores = []
    confidences = []
    
    for a in audits:
        fl = a.get('original_fit_label', 'Unknown')
        labels[fl] = labels.get(fl, 0) + 1
        
        if a.get('recommended_honeypot_probability', 0) > 0.5:
            hps += 1
        hp_prob_sum += a.get('recommended_honeypot_probability', 0)
        
        q = a.get('evaluation_quality', 'Unknown')
        qualities[q] = qualities.get(q, 0) + 1
        
        scores.append(a.get('audit_score', 0))
        confidences.append(a.get('audit_confidence', 0))

    # Boundary Violations
    violations = []
    for a in audits:
        rec_score = a.get('recommended_fit_score', 0)
        rec_label = a.get('recommended_fit_label', '')
        
        expected = ''
        if 85 <= rec_score <= 100: expected = 'Excellent Fit'
        elif 70 <= rec_score <= 84: expected = 'Strong Fit'
        elif 50 <= rec_score <= 69: expected = 'Moderate Fit'
        elif 25 <= rec_score <= 49: expected = 'Weak Fit'
        else: expected = 'Reject'
        
        # We check original recruiter evaluation boundaries if needed, 
        # but here we'll check if the auditor's own recommendations align.
        if expected != rec_label and expected != '':
            violations.append(f"Candidate {a['candidate_id']}: recommended score {rec_score} mapped to '{rec_label}', expected '{expected}'")

    # Generate Markdown
    md = [
        "# Batch 1 Audit Report",
        "",
        "## Overall Metrics",
        f"- **Candidates audited**: {total}",
        f"- **Average audit score**: {avg_score:.1f}/100",
        f"- **Score agreement**: {(score_agree/total)*100:.1f}%",
        f"- **Fit label agreement**: {(fit_agree/total)*100:.1f}%",
        f"- **Honeypot agreement**: {(hp_agree/total)*100:.1f}%",
        "",
        "### Evaluation Quality",
    ]
    for q, c in qualities.items(): md.append(f"- {q}: {c}")
    
    md.extend([
        "",
        "## Label Distribution (Original)",
    ])
    for l, c in labels.items(): md.append(f"- {l}: {c}")
    
    md.extend([
        "",
        "## Honeypot Distribution (Audit Recommended)",
        f"- **Number of honeypots (>0.5 prob)**: {hps}",
        f"- **Average honeypot probability**: {hp_prob_sum/total:.2f}",
        "",
        "## Additional Validation Checks",
        f"- **Mean Audit Score**: {pd.Series(scores).mean():.2f}",
        f"- **Median Audit Score**: {pd.Series(scores).median():.2f}",
        f"- **Audit Score StdDev**: {pd.Series(scores).std():.2f}",
        f"- **Audit Score Min/Max**: {min(scores)} / {max(scores)}",
        f"- **Mean Confidence**: {pd.Series(confidences).mean():.2f}",
        "",
        "### Boundary Violations",
    ])
    
    if violations:
        for v in violations: md.append(f"- {v}")
    else:
        md.append("- None detected.")
        
    md.extend([
        "",
        "### Suspicious Patterns",
    ])
    reject_pct = labels.get('Reject', 0) / total
    weak_pct = labels.get('Weak Fit', 0) / total
    hp_pct = hps / total
    low_audit_pct = sum(1 for s in scores if s < 70) / total
    avg_conf = pd.Series(confidences).mean()
    
    if reject_pct > 0.4: md.append(f"- **FLAG:** >40% are Reject ({reject_pct*100:.1f}%)")
    if weak_pct > 0.4: md.append(f"- **FLAG:** >40% are Weak Fit ({weak_pct*100:.1f}%)")
    if hp_pct > 0.2: md.append(f"- **FLAG:** >20% are Honeypots ({hp_pct*100:.1f}%)")
    if low_audit_pct > 0.2: md.append(f"- **FLAG:** >20% of audit scores below 70 ({low_audit_pct*100:.1f}%)")
    if avg_conf > 0.9 and avg_score < 70: md.append("- **FLAG:** Average confidence >0.9 but average audit score <70")
    if len(md) == md.index("### Suspicious Patterns") + 1:
        md.append("- None detected based on thresholds (Wait, note: Some flags might be triggered as expected by dataset design).")
        
    md.extend([
        "",
        "## Disagreement Analysis",
    ])
    
    disagreements = [a for a in audits if not a.get('fit_label_agreement', False) or not a.get('honeypot_agreement', False) or not a.get('score_agreement', False)]
    if not disagreements:
        md.append("No disagreements found.")
    else:
        for d in disagreements:
            md.extend([
                f"### {d['candidate_id']}",
                f"- **Original Score vs Recommended**: {d.get('original_fit_score')} vs {d.get('recommended_fit_score')} (Delta: {d.get('score_difference')})",
                f"- **Original Label vs Recommended**: {d.get('original_fit_label')} vs {d.get('recommended_fit_label')}",
                f"- **Original Honeypot vs Recommended**: {d.get('original_honeypot')} vs {d.get('recommended_honeypot_probability', 0) > 0.5}",
                f"- **Reasoning**: {d.get('audit_reasoning')}",
                ""
            ])
            
    with open(AUDIT_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
        
    print(f"Report written to {AUDIT_REPORT_FILE}")

if __name__ == "__main__":
    main()
