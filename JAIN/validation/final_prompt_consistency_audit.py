"""
JAIN / validation / final_prompt_consistency_audit.py
======================================================
End-to-end pipeline consistency audit.

Runs the recruiter evaluation prompt (system_prompt.txt) on ~15 representative
candidates, then audits each evaluation with audit_prompt.txt, and writes:
  - JAIN/artifacts/final_validation_labels.jsonl      (recruiter pass)
  - JAIN/artifacts/final_validation_audit.jsonl       (audit pass)
  - JAIN/artifacts/final_prompt_consistency_report.md (human report)

Usage:
  python JAIN/validation/final_prompt_consistency_audit.py --api-key <groq-key>

Or (single key, no flag):
  python JAIN/validation/final_prompt_consistency_audit.py
  (picks first key from hard-coded rotation below)

Author: JAIN pipeline
"""

import os
import sys
import json
import time

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import asyncio
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
WORKSPACE   = SCRIPT_DIR.parent.parent
JAIN_DIR    = WORKSPACE / "JAIN"
ARTIFACTS   = JAIN_DIR / "artifacts"
PROMPTS     = JAIN_DIR / "prompts"
LABELING    = JAIN_DIR / "labeling"

sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(LABELING))

from JAIN.labeling.build_prompt import build_batch_prompt, compress_candidate
from JAIN.labeling.validate_response import validate_response
from JAIN.labeling.keys import next_groq_key, GROQ_KEYS  # loads from .env automatically

# ---------------------------------------------------------------------------
# Candidate selection  ─ 3 per category  ×  6 categories = 18 candidates
# Chosen after manual inspection of archetypes_inspection.txt
# ---------------------------------------------------------------------------
AUDIT_SET = [
    # ── Category A: Excellent Fit ──────────────────────────────────────────
    # Senior Data Scientist with retrieval_score=1.0, trust=1.0, honeypot<0.03
    {"candidate_id": "CAND_0093912", "category": "A", "category_label": "Excellent Fit",
     "expected_label": "Excellent Fit", "expected_honeypot": False},
    # AI Engineer with retrieval_score=0.8, trust=1.0, honeypot<0.04
    {"candidate_id": "CAND_0098846", "category": "A", "category_label": "Excellent Fit",
     "expected_label": "Excellent Fit", "expected_honeypot": False},
    # AI Research Engineer retrieval_score=0.8, trust=0.69
    {"candidate_id": "CAND_0061175", "category": "A", "category_label": "Excellent Fit",
     "expected_label": "Strong Fit",   "expected_honeypot": False},

    # ── Category B: Strong / Moderate Technical ────────────────────────────
    # Senior Software Engineer – data engineering, real products (upGrad/Paytm)
    {"candidate_id": "CAND_0011925", "category": "B", "category_label": "Strong/Moderate Technical",
     "expected_label": "Moderate Fit", "expected_honeypot": False},
    # Software Engineer – Haystack/Pinecone skills but primarily backend work
    {"candidate_id": "CAND_0014710", "category": "B", "category_label": "Strong/Moderate Technical",
     "expected_label": "Moderate Fit", "expected_honeypot": False},
    # NLP Engineer  retrieval_score=0.6, trust=1.0 – solid NLP background
    {"candidate_id": "CAND_0095619", "category": "B", "category_label": "Strong/Moderate Technical",
     "expected_label": "Strong Fit",   "expected_honeypot": False},

    # ── Category C: Weak but Legitimate Engineers ──────────────────────────
    # QA Engineer @ TCS – no retrieval, DevOps skills, 1.1 YoE
    {"candidate_id": "CAND_0012375", "category": "C", "category_label": "Weak Legitimate Engineer",
     "expected_label": "Weak Fit",     "expected_honeypot": False},
    # Full Stack Developer – Java/Kafka backend, Qdrant in skills but 1.3 YoE
    {"candidate_id": "CAND_0088645", "category": "C", "category_label": "Weak Legitimate Engineer",
     "expected_label": "Weak Fit",     "expected_honeypot": False},
    # AI Engineer retrieval_score=0.8 but medium trust (borderline C/B)
    {"candidate_id": "CAND_0044222", "category": "C", "category_label": "Weak Legitimate Engineer",
     "expected_label": "Strong Fit",   "expected_honeypot": False},

    # ── Category D: Career Transition ─────────────────────────────────────
    # HR Manager 13.7 YoE – all non-engineering history despite AI keywords in skills
    {"candidate_id": "CAND_0076006", "category": "D", "category_label": "Career Transition",
     "expected_label": "Reject",       "expected_honeypot": False},
    # HR Manager 8.8 YoE – CS degree but mechanical + HR + sales career history
    {"candidate_id": "CAND_0040259", "category": "D", "category_label": "Career Transition",
     "expected_label": "Reject",       "expected_honeypot": False},
    # Civil Engineer 3.9 YoE – design engineering background, Sentence Transformers in skills
    {"candidate_id": "CAND_0013264", "category": "D", "category_label": "Career Transition",
     "expected_label": "Weak Fit",     "expected_honeypot": False},

    # ── Category E: Honeypot Candidates ───────────────────────────────────
    # Business Analyst / "GenAI explorer" – Ph.D + 1 YoE warehouse operations, skill stuffing
    {"candidate_id": "CAND_0031820", "category": "E", "category_label": "Honeypot",
     "expected_label": "Reject",       "expected_honeypot": True},
    # Business Analyst @ Wipro – mechanical design career, random AI skill mix
    {"candidate_id": "CAND_0026016", "category": "E", "category_label": "Honeypot",
     "expected_label": "Reject",       "expected_honeypot": True},
    # Marketing Manager / "AI enthusiast" – M.Tech ML but ops career, contradictory skills
    {"candidate_id": "CAND_0058807", "category": "E", "category_label": "Honeypot",
     "expected_label": "Reject",       "expected_honeypot": True},

    # ── Category F: Borderline ─────────────────────────────────────────────
    # Project Manager 11.1 YoE – MLOps/Semantic Search in skills, mechanical + ops history
    {"candidate_id": "CAND_0021768", "category": "F", "category_label": "Borderline",
     "expected_label": "Reject",       "expected_honeypot": False},
    # Project Manager 14.7 YoE – PhD IT, Pinecone/FAISS in skills, marketing+engineering history
    {"candidate_id": "CAND_0053507", "category": "F", "category_label": "Borderline",
     "expected_label": "Reject",       "expected_honeypot": False},
    # Project Manager – moderate semantic score, low retrieval
    {"candidate_id": "CAND_0048271", "category": "F", "category_label": "Borderline",
     "expected_label": "Weak Fit",     "expected_honeypot": False},
]

AUDIT_IDS = [c["candidate_id"] for c in AUDIT_SET]

# ---------------------------------------------------------------------------
# Groq API helpers
# ---------------------------------------------------------------------------
async def call_groq_single(user_prompt: str, system_prompt: str, api_key: str) -> str:
    """Single-candidate call (used for audit step to keep prompts manageable)."""
    from groq import AsyncGroq
    client = AsyncGroq(api_key=api_key)
    response = await client.chat.completions.create(
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


async def call_groq_batch(user_prompt: str, system_prompt: str, api_key: str) -> str:
    """Batch evaluation call; wraps output in an 'evaluations' object."""
    from groq import AsyncGroq
    client = AsyncGroq(api_key=api_key)
    response = await client.chat.completions.create(
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# STEP 1: Recruiter evaluation
# ---------------------------------------------------------------------------
async def run_recruiter_step(candidates_raw: list, sys_prompt: str,
                              cand_prompt_tmpl: str, jd_text: str, jd_json: dict,
                              api_key: str) -> list:
    """
    Evaluates candidates in batches of 3.
    Returns list of dicts: original recruiter response + candidate_id.
    """
    results = []
    batch_size = 3

    for batch_start in range(0, len(candidates_raw), batch_size):
        batch = candidates_raw[batch_start: batch_start + batch_size]
        batch_ids = [c["candidate_id"] for c in batch]
        print(f"  [Recruiter] Batch {batch_start // batch_size + 1} – {batch_ids}")

        prompt = build_batch_prompt(batch, jd_text, jd_json, cand_prompt_tmpl)
        prompt += (
            "\n\nCRITICAL: Return a JSON object with a single key 'evaluations' "
            "containing a JSON ARRAY of candidate evaluation objects."
        )

        for attempt in range(4):
            try:
                raw_text = await call_groq_batch(prompt, sys_prompt, api_key)
                # Unwrap {evaluations: [...]}
                parsed = json.loads(raw_text)
                if "evaluations" in parsed:
                    raw_text = json.dumps(parsed["evaluations"])

                is_valid, msg, data = validate_response(raw_text, len(batch))
                if is_valid:
                    for i, cand in enumerate(batch):
                        data[i]["candidate_id"] = cand["candidate_id"]
                    results.extend(data)
                    break
                else:
                    print(f"    Validation failed (attempt {attempt+1}): {msg}")

            except Exception as exc:
                err = str(exc)
                print(f"    API error (attempt {attempt+1}): {err[:120]}")
                if "429" in err or "rate_limit" in err:
                    api_key = next_groq_key()
                    print(f"    Rotating to next Groq key …")

            wait = 2 ** attempt
            print(f"    Waiting {wait}s …")
            await asyncio.sleep(wait)
        else:
            print(f"    FAILED after 4 attempts for batch {batch_ids}")

        await asyncio.sleep(6)   # Groq rate-limit safety (14 k TPM limit)

    return results


# ---------------------------------------------------------------------------
# STEP 2: Audit step – one candidate at a time
# ---------------------------------------------------------------------------
def build_audit_prompt(jd_text: str, candidate_raw: dict, recruiter_eval: dict) -> str:
    compressed = compress_candidate(candidate_raw)
    return (
        "=== JOB DESCRIPTION ===\n"
        f"{jd_text}\n\n"
        "=== CANDIDATE PROFILE ===\n"
        f"{json.dumps(compressed, indent=2)}\n\n"
        "=== RECRUITER EVALUATION ===\n"
        f"{json.dumps(recruiter_eval, indent=2)}\n\n"
        "Now perform your audit and return ONLY valid JSON matching the required schema."
    )


AUDIT_SCHEMA_KEYS = {
    "audit_score", "audit_confidence", "score_agreement", "fit_label_agreement",
    "honeypot_agreement", "score_difference", "missed_strengths", "missed_weaknesses",
    "recommended_fit_score", "recommended_fit_label", "recommended_honeypot_probability",
    "evaluation_quality", "audit_reasoning"
}

def validate_audit_response(text: str) -> tuple[bool, str, dict]:
    """Validate that the audit JSON contains all required keys."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}", {}

    missing = AUDIT_SCHEMA_KEYS - set(data.keys())
    if missing:
        return False, f"Missing keys: {missing}", data

    return True, "OK", data


async def run_audit_step(candidates_raw: list, recruiter_results: list,
                          audit_sys_prompt: str, jd_text: str,
                          api_key: str) -> list:
    """
    Audits each recruiter result individually.
    Returns list of audit dicts + candidate_id.
    """
    recruiter_by_id = {r["candidate_id"]: r for r in recruiter_results}
    raw_by_id      = {c["candidate_id"]: c for c in candidates_raw}

    audit_results = []

    for cid in AUDIT_IDS:
        if cid not in recruiter_by_id:
            print(f"  [Audit] Skipping {cid} – no recruiter result.")
            continue
        if cid not in raw_by_id:
            print(f"  [Audit] Skipping {cid} – raw profile not loaded.")
            continue

        print(f"  [Audit] {cid} …")
        rec_eval  = recruiter_by_id[cid]
        cand_raw  = raw_by_id[cid]

        user_prompt = build_audit_prompt(jd_text, cand_raw, rec_eval)

        for attempt in range(4):
            try:
                raw_text = await call_groq_single(user_prompt, audit_sys_prompt, api_key)
                ok, msg, data = validate_audit_response(raw_text)
                if ok:
                    data["candidate_id"] = cid
                    audit_results.append(data)
                    break
                else:
                    print(f"    Audit validation failed (attempt {attempt+1}): {msg}")

            except Exception as exc:
                err = str(exc)
                print(f"    Audit API error (attempt {attempt+1}): {err[:120]}")
                if "429" in err or "rate_limit" in err:
                    api_key = next_groq_key()

            wait = 2 ** attempt
            await asyncio.sleep(wait)
        else:
            print(f"    AUDIT FAILED for {cid}")

        await asyncio.sleep(5)   # one-at-a-time audit; be gentle on rate limits

    return audit_results


# ---------------------------------------------------------------------------
# STEP 3: Report generation
# ---------------------------------------------------------------------------
def fit_label_from_score(score: float) -> str:
    if score is None:
        return "Unknown"
    if score >= 85:
        return "Excellent Fit"
    if score >= 70:
        return "Strong Fit"
    if score >= 50:
        return "Moderate Fit"
    if score >= 25:
        return "Weak Fit"
    return "Reject"


def generate_report(audit_set_meta: list, recruiter_results: list,
                    audit_results: list, out_path: Path) -> None:
    rec_by_id   = {r["candidate_id"]: r for r in recruiter_results}
    audit_by_id = {r["candidate_id"]: r for r in audit_results}

    # --- aggregate stats ---
    n_rec   = len(recruiter_results)
    n_audit = len(audit_results)

    fit_label_counts: dict[str, int] = {}
    honeypot_true = 0
    avg_fit_score = 0.0
    avg_audit_score = 0.0

    label_agreements = 0
    honeypot_agreements = 0

    expected_vs_actual: list[dict] = []

    for meta in audit_set_meta:
        cid = meta["candidate_id"]
        rec = rec_by_id.get(cid, {})
        aud = audit_by_id.get(cid, {})

        fit_score  = rec.get("fit_score", 0)
        fit_label  = rec.get("fit_label", "Unknown")
        honeypot   = rec.get("honeypot_label", False)

        fit_label_counts[fit_label] = fit_label_counts.get(fit_label, 0) + 1
        if honeypot:
            honeypot_true += 1
        avg_fit_score += fit_score

        audit_score = aud.get("audit_score", 0)
        avg_audit_score += audit_score

        lbl_agree = aud.get("fit_label_agreement", None)
        hp_agree  = aud.get("honeypot_agreement", None)
        if lbl_agree is True:
            label_agreements += 1
        if hp_agree is True:
            honeypot_agreements += 1

        # Compare expected vs actual
        expected_correct = (fit_label == meta["expected_label"] or
                            (meta["expected_label"] == "Strong Fit" and fit_label in {"Strong Fit", "Excellent Fit"}) or
                            (meta["expected_label"] == "Moderate Fit" and fit_label in {"Moderate Fit", "Strong Fit"}))
        hp_correct = (honeypot == meta["expected_honeypot"])

        expected_vs_actual.append({
            "candidate_id":      cid,
            "category":          meta["category"],
            "category_label":    meta["category_label"],
            "expected_label":    meta["expected_label"],
            "actual_label":      fit_label,
            "actual_score":      fit_score,
            "label_correct":     expected_correct,
            "expected_honeypot": meta["expected_honeypot"],
            "actual_honeypot":   honeypot,
            "hp_correct":        hp_correct,
            "audit_score":       audit_score,
            "audit_quality":     aud.get("evaluation_quality", "N/A"),
            "audit_reasoning":   aud.get("audit_reasoning", "")[:200],
        })

    if n_rec > 0:
        avg_fit_score /= n_rec
    if n_audit > 0:
        avg_audit_score /= n_audit

    # --- group by category ---
    categories_order = ["A", "B", "C", "D", "E", "F"]
    category_rows: dict[str, list] = {c: [] for c in categories_order}
    for row in expected_vs_actual:
        category_rows[row["category"]].append(row)

    # --- compute overall pass rate ---
    n_correct_labels  = sum(1 for r in expected_vs_actual if r["label_correct"])
    n_correct_hp      = sum(1 for r in expected_vs_actual if r["hp_correct"])
    total             = len(expected_vs_actual)

    verdict = (
        "✅ APPROVED FOR LARGE-SCALE LABELING"
        if (n_correct_labels / max(total, 1) >= 0.75 and n_correct_hp / max(total, 1) >= 0.80)
        else "⚠️ PROMPT REQUIRES FURTHER CALIBRATION"
    )

    # --- write markdown ---
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Final Prompt Consistency Report",
        f"",
        f"Generated: {ts}",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Metric                       | Value |",
        f"|------------------------------|-------|",
        f"| Candidates evaluated         | {n_rec} |",
        f"| Candidates audited           | {n_audit} |",
        f"| Avg recruiter fit_score      | {avg_fit_score:.1f} |",
        f"| Avg audit quality score      | {avg_audit_score:.1f} |",
        f"| Label agreements (audit)     | {label_agreements} / {n_audit} |",
        f"| Honeypot agreements (audit)  | {honeypot_agreements} / {n_audit} |",
        f"| Expected label correct       | {n_correct_labels} / {total} ({100*n_correct_labels//max(total,1)}%) |",
        f"| Expected honeypot correct    | {n_correct_hp} / {total} ({100*n_correct_hp//max(total,1)}%) |",
        f"",
        f"### Fit Label Distribution",
        f"",
        f"| Label          | Count |",
        f"|----------------|-------|",
    ]
    for lbl in ["Excellent Fit", "Strong Fit", "Moderate Fit", "Weak Fit", "Reject", "Unknown"]:
        cnt = fit_label_counts.get(lbl, 0)
        if cnt:
            lines.append(f"| {lbl:14s} | {cnt:5d} |")

    lines += [
        f"",
        f"Honeypot candidates detected: **{honeypot_true}** / {n_rec}",
        f"",
        f"---",
        f"",
        f"## Final Verdict",
        f"",
        f"### {verdict}",
        f"",
        f"---",
        f"",
        f"## Category Breakdown",
        f"",
    ]

    cat_descriptions = {
        "A": "Category A — Excellent Fit (should score 70–100, no honeypot)",
        "B": "Category B — Strong / Moderate Technical (should score 50–84, no honeypot)",
        "C": "Category C — Weak but Legitimate Engineers (should score 25–49, no honeypot)",
        "D": "Category D — Career Transition (should score 0–49, no honeypot)",
        "E": "Category E — Honeypot Candidates (should score 0–24, honeypot=true)",
        "F": "Category F — Borderline (mixed expectations)",
    }

    for cat in categories_order:
        rows = category_rows[cat]
        if not rows:
            continue
        lines.append(f"### {cat_descriptions[cat]}")
        lines.append(f"")
        lines.append(f"| Candidate ID | Expected | Actual Score | Actual Label | ✓ Label | Expected HP | Actual HP | ✓ HP | Audit Score | Quality |")
        lines.append(f"|-------------|----------|-------------|--------------|---------|-------------|-----------|------|-------------|---------|")
        for row in rows:
            lc  = "✅" if row["label_correct"] else "❌"
            hpc = "✅" if row["hp_correct"]    else "❌"
            lines.append(
                f"| {row['candidate_id']} "
                f"| {row['expected_label']} "
                f"| {row['actual_score']} "
                f"| {row['actual_label']} "
                f"| {lc} "
                f"| {row['expected_honeypot']} "
                f"| {row['actual_honeypot']} "
                f"| {hpc} "
                f"| {row['audit_score']} "
                f"| {row['audit_quality']} |"
            )
        lines.append(f"")

    lines += [
        f"---",
        f"",
        f"## Detailed Reasoning per Candidate",
        f"",
    ]

    for meta in audit_set_meta:
        cid  = meta["candidate_id"]
        rec  = rec_by_id.get(cid, {})
        aud  = audit_by_id.get(cid, {})
        row  = next((r for r in expected_vs_actual if r["candidate_id"] == cid), {})

        lc  = "✅" if row.get("label_correct") else "❌"
        hpc = "✅" if row.get("hp_correct")    else "❌"

        lines += [
            f"### {cid}  [{meta['category_label']}]",
            f"",
            f"- **Expected**: {meta['expected_label']} / honeypot={meta['expected_honeypot']}",
            f"- **Recruiter**: score={rec.get('fit_score','?')} / label={rec.get('fit_label','?')} / honeypot={rec.get('honeypot_label','?')} {lc} {hpc}",
            f"- **Recruiter Reasoning**: {(rec.get('reasoning') or '')[:300]}",
            f"- **Audit Score**: {aud.get('audit_score','?')} / quality={aud.get('evaluation_quality','?')}",
            f"- **Missed Strengths**: {aud.get('missed_strengths', [])}",
            f"- **Missed Weaknesses**: {aud.get('missed_weaknesses', [])}",
            f"- **Recommended Score**: {aud.get('recommended_fit_score','?')} / label={aud.get('recommended_fit_label','?')}",
            f"- **Audit Reasoning**: {(aud.get('audit_reasoning') or '')[:300]}",
            f"",
        ]

    lines += [
        f"---",
        f"",
        f"## Conclusions",
        f"",
        f"This report validates the labeling pipeline across all six candidate archetypes.",
        f"A score of ≥75% label accuracy and ≥80% honeypot accuracy is required for approval.",
        f"",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Report] Written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Final prompt consistency audit")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Groq API key (optional; falls back to rotation)")
    args = parser.parse_args()

    if args.api_key:
        GROQ_KEYS.insert(0, args.api_key)

    api_key = next_groq_key()

    print("=" * 60)
    print("JAIN Final Prompt Consistency Audit")
    print(f"Candidates: {len(AUDIT_IDS)}")
    print(f"Using Groq / llama-3.3-70b-versatile")
    print("=" * 60)

    # ── Load prompts ──────────────────────────────────────────────────────
    print("\n[Setup] Loading prompts and job description …")
    sys_prompt   = (PROMPTS / "system_prompt.txt").read_text(encoding="utf-8")
    audit_prompt = (PROMPTS / "audit_prompt.txt").read_text(encoding="utf-8")
    cand_prompt  = (PROMPTS / "candidate_prompt.txt").read_text(encoding="utf-8")
    jd_text      = (PROMPTS / "job_description.txt").read_text(encoding="utf-8")

    jd_json_path = WORKSPACE / "PIYUSH" / "artifacts" / "jd_requirements.json"
    with open(jd_json_path, encoding="utf-8") as f:
        jd_json = json.load(f)

    # ── Load raw candidate profiles ───────────────────────────────────────
    print("[Setup] Loading raw candidate profiles …")
    candidates_raw: list[dict] = []
    jsonl_path = WORKSPACE / "candidates.jsonl"
    audit_id_set = set(AUDIT_IDS)

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("candidate_id") in audit_id_set:
                candidates_raw.append(record)

    found_ids = {c["candidate_id"] for c in candidates_raw}
    missing   = audit_id_set - found_ids
    if missing:
        print(f"  WARNING: {len(missing)} candidate(s) not found in JSONL: {missing}")
    print(f"  Loaded {len(candidates_raw)} profiles.")

    # Reorder to match AUDIT_IDS order
    id_to_raw = {c["candidate_id"]: c for c in candidates_raw}
    candidates_raw = [id_to_raw[cid] for cid in AUDIT_IDS if cid in id_to_raw]

    # ── STEP 1: Recruiter evaluation ──────────────────────────────────────
    print(f"\n[Step 1] Running recruiter evaluation on {len(candidates_raw)} candidates …")
    recruiter_results = await run_recruiter_step(
        candidates_raw, sys_prompt, cand_prompt, jd_text, jd_json, api_key
    )
    print(f"  Recruiter evaluations completed: {len(recruiter_results)}")

    labels_path = ARTIFACTS / "final_validation_labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as f:
        for r in recruiter_results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved -> {labels_path}")

    # ── STEP 2: Audit step ────────────────────────────────────────────────
    print(f"\n[Step 2] Running audit on {len(recruiter_results)} evaluations (1-by-1) …")
    audit_results = await run_audit_step(
        candidates_raw, recruiter_results, audit_prompt, jd_text, api_key
    )
    print(f"  Audit results completed: {len(audit_results)}")

    audit_path = ARTIFACTS / "final_validation_audit.jsonl"
    with open(audit_path, "w", encoding="utf-8") as f:
        for r in audit_results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved -> {audit_path}")

    # ── STEP 3: Report ────────────────────────────────────────────────────
    print(f"\n[Step 3] Generating final consistency report …")
    report_path = ARTIFACTS / "final_prompt_consistency_report.md"
    generate_report(AUDIT_SET, recruiter_results, audit_results, report_path)

    # ── Print quick summary to stdout ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    rec_by_id = {r["candidate_id"]: r for r in recruiter_results}
    aud_by_id = {r["candidate_id"]: r for r in audit_results}

    print(f"\n{'Candidate':<15} {'Cat':<4} {'Score':>5} {'Label':<15} {'HP':<6} {'Audit':>5} {'Expected':<15} {'✓L':<3} {'✓HP':<3}")
    print("-" * 80)
    for meta in AUDIT_SET:
        cid = meta["candidate_id"]
        rec = rec_by_id.get(cid, {})
        aud = aud_by_id.get(cid, {})

        score  = rec.get("fit_score",    "?")
        label  = rec.get("fit_label",    "?")
        hp     = rec.get("honeypot_label", "?")
        ascore = aud.get("audit_score",  "?")

        score_val = rec.get("fit_score", 0) or 0
        label_ok  = (label == meta["expected_label"] or
                     (meta["expected_label"] == "Strong Fit"   and label in {"Strong Fit", "Excellent Fit"}) or
                     (meta["expected_label"] == "Moderate Fit" and label in {"Moderate Fit", "Strong Fit"}))
        hp_ok     = (hp == meta["expected_honeypot"])

        print(f"{cid:<15} {meta['category']:<4} {str(score):>5} {str(label):<15} {str(hp):<6} {str(ascore):>5} "
              f"{meta['expected_label']:<15} {'OK' if label_ok else 'XX':<4} {'OK' if hp_ok else 'XX':<4}")

    print(f"\nReports written:")
    print(f"  Labels  -> {labels_path}")
    print(f"  Audits  -> {audit_path}")
    print(f"  Report  -> {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
