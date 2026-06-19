# Retest Delta Report

Generated: 2026-06-15 23:08 UTC
Prompt version: system_prompt.txt (with HONEYPOT TRIGGER + HARD REJECT rules)

---

## Summary

| Metric                      | Before | After |
|-----------------------------|--------|-------|
| Label accuracy (7 cands)    | 3/7 (42%) | 6/7 (85%) |
| Honeypot accuracy (7 cands) | 4/7 (57%) | 2/7 (28%) |
| Cases fully corrected       | — | 2/7 |
| Avg audit quality score     | 77 | 86 |

## Final Verdict

### CALIBRATION NEEDED: STILL REQUIRES CALIBRATION

Success criteria: >= 5/7 corrected AND avg audit >= 80
Result: 2/7 corrected, avg audit = 86

---

## Candidate Delta Table

| Candidate | Cat | Old Score | New Score | Old Label | New Label | Old HP | New HP | HP Prob | Exp Label | Exp HP | Label OK | HP OK | Audit | Quality |
|-----------|-----|-----------|-----------|-----------|-----------|--------|--------|---------|-----------|--------|----------|-------|-------|---------|
| CAND_0040259 | D | 25 | 10 | Weak Fit | Reject | False | True | 0.9 | Reject | False | OK | FAIL | 80 | Good |
| CAND_0013264 | D | 35 | 20 | Moderate Fit | Reject | False | True | 0.8 | Weak Fit | False | FAIL | FAIL | 80 | Good |
| CAND_0031820 | E | 10 | 10 | Reject | Reject | False | True | 0.9 | Reject | True | OK | OK | 90 | Excellent |
| CAND_0026016 | E | 12 | 6 | Reject | Reject | False | False | 0.2 | Reject | True | OK | FAIL | 90 | Good |
| CAND_0058807 | E | 8 | 10 | Reject | Reject | False | False | 0.3 | Reject | True | OK | FAIL | 90 | Good |
| CAND_0021768 | F | 13 | 15 | Weak Fit | Reject | False | False | 0.1 | Reject | False | OK | OK | 80 | Good |
| CAND_0053507 | F | 18 | 5 | Weak Fit | Reject | False | True | 0.9 | Reject | False | OK | FAIL | 90 | Excellent |

---

## Detailed Changes

### Category D — Career Transition

**CAND_0040259**
- Score: 25 -> 10 (delta: -15)
- Label: `Weak Fit` -> `Reject` (expected: `Reject`)
- Honeypot: False -> True (prob: 0.9, expected: False)
- Audit: 80/100 (Good)
- Audit reasoning: The recruiter's evaluation is justified by the evidence in the candidate profile. The candidate lacks relevant experience in AI engineering and has a work history in non-technical fields, which aligns

**CAND_0013264**
- Score: 35 -> 20 (delta: -15)
- Label: `Moderate Fit` -> `Reject` (expected: `Weak Fit`)
- Honeypot: False -> True (prob: 0.8, expected: False)
- Audit: 80/100 (Good)
- Audit reasoning: The recruiter's evaluation is mostly correct, but the honeypot probability seems too high given the candidate's genuine interest in AI and GenAI applications. The candidate's lack of direct experience

### Category E — Honeypot

**CAND_0031820**
- Score: 10 -> 10 (delta: +0)
- Label: `Reject` -> `Reject` (expected: `Reject`)
- Honeypot: False -> True (prob: 0.9, expected: True)
- Audit: 90/100 (Excellent)
- Audit reasoning: The recruiter's evaluation is justified by the evidence in the candidate profile, which shows no relevant experience in AI engineering and a work history in business analysis, aligning with the job de

**CAND_0026016**
- Score: 12 -> 6 (delta: -6)
- Label: `Reject` -> `Reject` (expected: `Reject`)
- Honeypot: False -> False (prob: 0.2, expected: True)
- Audit: 90/100 (Good)
- Audit reasoning: The recruiter's evaluation is mostly correct, but the candidate's lack of experience in NLP/IR and building end-to-end ranking systems could be emphasized more.

**CAND_0058807**
- Score: 8 -> 10 (delta: +2)
- Label: `Reject` -> `Reject` (expected: `Reject`)
- Honeypot: False -> False (prob: 0.3, expected: True)
- Audit: 90/100 (Good)
- Audit reasoning: The recruiter's evaluation is reasonable, given the candidate's lack of production experience with embeddings-based retrieval systems, vector databases, or hybrid search infrastructure, and their work

### Category F — Borderline

**CAND_0021768**
- Score: 13 -> 15 (delta: +2)
- Label: `Weak Fit` -> `Reject` (expected: `Reject`)
- Honeypot: False -> False (prob: 0.1, expected: False)
- Audit: 80/100 (Good)
- Audit reasoning: The recruiter's evaluation is mostly correct, but it could be improved by considering the candidate's enthusiasm for AI and GenAI applications. However, the lack of relevant experience and skills in t

**CAND_0053507**
- Score: 18 -> 5 (delta: -13)
- Label: `Weak Fit` -> `Reject` (expected: `Reject`)
- Honeypot: False -> True (prob: 0.9, expected: False)
- Audit: 90/100 (Excellent)
- Audit reasoning: The recruiter's evaluation is justified by the evidence in the candidate profile. The candidate lacks production experience with embeddings-based retrieval systems, vector databases, or hybrid search 
