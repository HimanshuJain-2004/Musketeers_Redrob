import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_ROOT = r'c:\Users\himan\Downloads\India_runs_data_and_ai_challenge'

inspect_ids = [
    # Cat E - honeypots
    'CAND_0031820', 'CAND_0026016', 'CAND_0058807',
    # Cat D - transition
    'CAND_0076006', 'CAND_0040259', 'CAND_0013264',
    # Cat C - weak legit engineers
    'CAND_0012375', 'CAND_0088645',
    # Cat B - moderate tech
    'CAND_0011925', 'CAND_0014710',
    # Cat F - borderline
    'CAND_0021768', 'CAND_0053507',
]

profiles = {}
with open(os.path.join(WORKSPACE_ROOT, 'candidates.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r['candidate_id'] in inspect_ids:
            profiles[r['candidate_id']] = r

out_path = os.path.join(WORKSPACE_ROOT, 'JAIN', 'sampling', 'archetype_inspection.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    for cid in inspect_ids:
        if cid not in profiles:
            f.write(f'{cid}: NOT FOUND\n')
            continue
        cand = profiles[cid]
        p = cand.get('profile', {})
        careers = cand.get('career_history', [])
        edu = cand.get('education', [])
        skills = [s['name'] for s in cand.get('skills', [])[:12]]
        f.write(f'\n{"="*60}\n{cid}\n{"="*60}\n')
        f.write(f'Title: {p.get("current_title")}\n')
        f.write(f'Headline: {p.get("headline")}\n')
        f.write(f'YoE: {p.get("years_of_experience")}\n')
        f.write(f'Skills: {skills}\n')
        f.write(f'Education: {[e.get("degree") + " " + e.get("field_of_study","") + " @ " + e.get("institution","") for e in edu]}\n')
        f.write('Career:\n')
        for job in careers[:3]:
            f.write(f'  [{job.get("duration_months")} mos] {job.get("title")} @ {job.get("company")}\n')
            desc = (job.get('description') or '')[:300]
            f.write(f'  {desc}\n')

print(f'Written to {out_path}')
