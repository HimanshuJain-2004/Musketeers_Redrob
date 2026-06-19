# Retest 2 Delta Report — Three-Way Comparison

Generated: 2026-06-15 23:17 UTC
Prompt: system_prompt.txt v3 (contradiction-based honeypot trigger)

---

## Accuracy Summary (5 candidates)

| Metric              | Original | Retest 1 (skill-count trigger) | Retest 2 (contradiction trigger) |
|---------------------|----------|-------------------------------|----------------------------------|
| Label accuracy      | 2/5 (40%) | 4/5 (80%) | 5/5 (100%) |
| Honeypot accuracy   | 3/5 (60%) | 0/5 (0%) | 3/5 (60%) |
| Fully correct       | — | — | 3/5 |
| Avg audit score     | 76 | 86 | 85 |

## Verdict

### STILL REQUIRES CALIBRATION

Approval criteria: label accuracy >= 80% AND honeypot accuracy >= 80% AND avg audit >= 80
Result: label=100%, honeypot=60%, audit=85

---

## Three-Way Candidate Table

| Candidate | Cat | Exp Label | Exp HP | Orig Score | R1 Score | R2 Score | Orig Label | R1 Label | R2 Label | Orig HP | R1 HP | R2 HP | R2 HPProb | R2 LblOK | R2 HPOK | R2 Audit |
|-----------|-----|-----------|--------|------------|----------|----------|------------|----------|----------|---------|-------|-------|-----------|----------|---------|----------|
| CAND_0040259 | D | Reject | False | 25 | 10 | 10 | Weak Fit | Reject | Reject | False | True | True | 0.8 | OK | FAIL | 80 |
| CAND_0013264 | D | Weak Fit | False | 35 | 20 | 25 | Moderate Fit | Reject | Weak Fit | False | True | False | 0.2 | OK | OK | 80 |
| CAND_0026016 | E | Reject | True | 12 | 6 | 10 | Reject | Reject | Reject | False | False | True | 0.9 | OK | OK | 80 |
| CAND_0058807 | E | Reject | True | 8 | 10 | 10 | Reject | Reject | Reject | False | False | True | 0.9 | OK | OK | 95 |
| CAND_0053507 | F | Reject | False | 18 | 5 | 10 | Weak Fit | Reject | Reject | False | True | True | 0.9 | OK | FAIL | 90 |

---

## Detailed Candidate Reasoning

### Category D — Career Transition

**CAND_0040259** — Expected: `Reject` / honeypot=False

| Run | Score | Label | Honeypot | HP Prob |
|-----|-------|-------|----------|---------|
| Original | 25 | Weak Fit | False | — |
| Retest 1 | 10 | Reject | True | — |
| Retest 2 | 10 (vs R1: +0) | Reject | True | 0.8 |

- Label: OK | Honeypot: FAIL | Audit: 80/100 (Good)
- Audit reasoning: The recruiter's evaluation is justified by the evidence in the candidate profile. The candidate lacks production ML experience, relevant technical skills, and a non-technical background, which aligns with the job require

**CAND_0013264** — Expected: `Weak Fit` / honeypot=False

| Run | Score | Label | Honeypot | HP Prob |
|-----|-------|-------|----------|---------|
| Original | 35 | Moderate Fit | False | — |
| Retest 1 | 20 | Reject | True | — |
| Retest 2 | 25 (vs R1: +5) | Weak Fit | False | 0.2 |

- Label: OK | Honeypot: OK | Audit: 80/100 (Good)
- Audit reasoning: The recruiter's evaluation is mostly correct, but the candidate's lack of direct experience in AI engineering and limited experience with production deployments are significant weaknesses. The candidate's enthusiasm and 

### Category E — Honeypot

**CAND_0026016** — Expected: `Reject` / honeypot=True

| Run | Score | Label | Honeypot | HP Prob |
|-----|-------|-------|----------|---------|
| Original | 12 | Reject | False | — |
| Retest 1 | 6 | Reject | False | — |
| Retest 2 | 10 (vs R1: +4) | Reject | True | 0.9 |

- Label: OK | Honeypot: OK | Audit: 80/100 (Good)
- Audit reasoning: The recruiter's evaluation is justified by the evidence present in the candidate profile. The candidate lacks relevant technical experience, production ML experience, and skills required for the role.

**CAND_0058807** — Expected: `Reject` / honeypot=True

| Run | Score | Label | Honeypot | HP Prob |
|-----|-------|-------|----------|---------|
| Original | 8 | Reject | False | — |
| Retest 1 | 10 | Reject | False | — |
| Retest 2 | 10 (vs R1: +0) | Reject | True | 0.9 |

- Label: OK | Honeypot: OK | Audit: 95/100 (Excellent)
- Audit reasoning: The recruiter's evaluation is justified by the evidence in the candidate profile. The candidate lacks relevant experience in AI engineering, retrieval systems, or search infrastructure, and their background is in marketi

### Category F — Borderline

**CAND_0053507** — Expected: `Reject` / honeypot=False

| Run | Score | Label | Honeypot | HP Prob |
|-----|-------|-------|----------|---------|
| Original | 18 | Weak Fit | False | — |
| Retest 1 | 5 | Reject | True | — |
| Retest 2 | 10 (vs R1: +5) | Reject | True | 0.9 |

- Label: OK | Honeypot: FAIL | Audit: 90/100 (Excellent)
- Audit reasoning: The recruiter's evaluation is justified by the evidence in the candidate profile. The candidate lacks relevant experience in AI engineering, retrieval systems, and search infrastructure, with a background in project mana

---

## Prompt Version History

| Version | Honeypot Rule | Label Acc (5 cands) | HP Acc (5 cands) |
|---------|--------------|---------------------|-----------------|
| Original (final_validation) | Implicit contradiction check only | 2/5 (40%) | 3/5 (60%) |
| Retest 1 | Skill-count trigger (>= 5 AI skills) | 4/5 (80%) | 0/5 (0%) |
| Retest 2 | Contradiction-based trigger | 5/5 (100%) | 3/5 (60%) |
