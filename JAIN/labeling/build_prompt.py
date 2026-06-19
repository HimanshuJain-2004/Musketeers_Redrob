import json

def compress_candidate(candidate_raw):
    """
    Compresses raw candidate JSON to save tokens.
    Keeps: Headline, Summary, Skills, Top 10 Work Exp, Projects, Education, Behavioral Signals.
    """
    profile = candidate_raw.get('profile', {})
    compressed = {
        'candidate_id': candidate_raw.get('candidate_id'),
        'headline': profile.get('headline'),
        'summary': profile.get('summary'),
        'skills': [s.get('name') for s in candidate_raw.get('skills', [])][:15], # Top 15 skills
        'work_history': [],
        'education': [],
        'redrob_signals': candidate_raw.get('redrob_signals', {})
    }
    
    # Top 10 work experiences
    for job in candidate_raw.get('career_history', [])[:10]:
        compressed['work_history'].append({
            'title': job.get('title'),
            'company': job.get('company'),
            'duration_months': job.get('duration_months'),
            'description': job.get('description')
        })
        
    for edu in candidate_raw.get('education', []):
        compressed['education'].append({
            'degree': edu.get('degree'),
            'field': edu.get('field_of_study'),
            'tier': edu.get('tier')
        })
        
    return compressed

def build_batch_prompt(candidates_raw, jd_text, jd_json, candidate_prompt_template):
    compressed_candidates = [compress_candidate(c) for c in candidates_raw]
    
    candidates_json_str = json.dumps(compressed_candidates, indent=2)
    
    prompt = candidate_prompt_template.replace('{job_description}', jd_text)
    prompt = prompt.replace('{jd_requirements}', json.dumps(jd_json, indent=2))
    prompt = prompt.replace('{num_candidates}', str(len(candidates_raw)))
    prompt = prompt.replace('{candidates_json}', candidates_json_str)
    
    return prompt
