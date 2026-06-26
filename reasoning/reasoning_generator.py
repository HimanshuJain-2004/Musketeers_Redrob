"""
ReasoningGenerator v4.0
-----------------------
Generation flow:

  S1: Title + Experience + Verified Domain            (~12-15 words)
  S2: Relative ranking explanation using evidence     (~12-15 words)
  [Negative signal: forced only for BOTTOM20]         (~9-11 words)
  S3: "{global_assessment}; {recommendation}"        (~14-18 words, 1 sentence)
  S4 (optional): Availability                        (~5-8 words)

  Target: 45-70 words, max 4 sentences.

Key rules:
  - No archetype label names ever in output.
  - No raw feature column names (hallucination guard).
  - No unsupported claims: no team sizes, revenue, scale numbers.
  - Recommendation is RANK × CONFIDENCE (not rank alone).
  - BOTTOM20 candidates always get a negative signal sentence.
  - S4 only if avail=HIGH or rank <= 100.
"""

import hashlib
import re

from phrase_library import PhraseLibrary


# ---------------------------------------------------------------------------
# Hallucination guard — all internal feature column names + extra risk terms.
# ---------------------------------------------------------------------------
_BANNED_TERMS = frozenset([
    # Feature column names
    "experience_score", "skill_match_count", "skill_coverage", "title_score",
    "education_score", "location_score", "product_company_score",
    "product_company_years", "consulting_penalty", "consulting_company_years",
    "career_history_length", "current_role_matches_jd", "retrieval_score_old",
    "retrieval_count", "retrieval_present", "ranking_count", "ranking_present",
    "recommendation_count", "recommendation_present", "matching_count",
    "matching_present", "evaluation_count", "evaluation_present",
    "production_count", "production_present", "jd_core_score",
    "availability_score", "recruitability_score", "market_demand_score",
    "trust_score", "technical_credibility_score", "mobility_score",
    "days_since_active", "notice_period_days", "github_activity_score",
    "avg_assessment_score", "semantic_score", "career_semantic_score",
    "prod_ml_count", "monitoring_count", "infra_count", "mlops_count",
    "llm_wrapper_count", "prod_ml_present", "monitoring_present",
    "infra_present", "mlops_present", "production_ml_score",
    "production_depth_score", "wrapper_to_production_ratio",
    "ownership_count", "leadership_count", "architecture_count",
    "mentorship_count", "impact_count", "senior_title_flag",
    "executive_title_flag", "ownership_present", "leadership_present",
    "architecture_present", "mentorship_present", "impact_present",
    "ownership_score", "leadership_depth_score", "total_experience_years",
    "num_companies", "num_roles", "avg_tenure_months", "longest_tenure_months",
    "shortest_tenure_months", "job_hopping_score", "short_tenure_count",
    "very_short_tenure_count", "promotion_count", "promotion_velocity",
    "career_growth_score", "multi_year_company_count",
    "long_term_employment_ratio", "timeline_gap_count", "largest_gap_months",
    "career_stability_score", "timeline_quality_score", "startup_years",
    "product_years", "consulting_years", "bigtech_years", "startup_ratio",
    "product_ratio", "consulting_ratio", "bigtech_ratio",
    "company_type_switches", "company_diversity_score", "current_company_type",
    "num_skills", "skill_density", "role_density", "experience_gap_years",
    "experience_consistency_score", "skills_per_role",
    "timeline_consistency_score", "suspicion_score",
    "technical_leadership_score", "production_retrieval_score",
    "senior_production_score", "ownership_relevance_score",
    "stability_leadership_score", "seniority_relevance_score",
    "years_exp", "retrieval_score", "jd_score",
    # Archetype label names (must never appear in output)
    "retrieval architect", "production ml builder", "technical leader",
    "research specialist", "experienced generalist", "hidden gem",
])

_BANNED_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _BANNED_TERMS) + r")\b",
    re.IGNORECASE,
)


def _check_hallucination(text: str, candidate_id: str) -> str:
    """Scan output for banned internal terms. Redact and log if found."""
    match = _BANNED_PATTERN.search(text)
    if match:
        print(
            f"[HALLUCINATION GUARD] '{match.group()}' detected for {candidate_id}. "
            "Redacting."
        )
        text = _BANNED_PATTERN.sub("[REDACTED]", text)
    return text


def _word_count(text: str) -> int:
    return len(text.split())


class ReasoningGenerator:

    def __init__(self, profiler, data_loader, score_percentiles: dict,
                 pool_context: dict = None):
        self.profiler = profiler
        self.data = data_loader
        self.p = score_percentiles
        self.pool_context = pool_context or {"quality": "NORMAL",
                                              "p80": 30.0, "p95": 60.0,
                                              "p99": 80.0, "p50": 10.0}
        self.seen_fingerprints = set()

    # ------------------------------------------------------------------
    # Deterministic phrase picker (SHA-256 → index)
    # ------------------------------------------------------------------
    def _pick(self, candidate_id: str, phrase_list: list, salt: str = "") -> str:
        key = (candidate_id + salt).encode("utf-8")
        idx = int(hashlib.sha256(key).hexdigest(), 16) % len(phrase_list)
        return phrase_list[idx]

    # ------------------------------------------------------------------
    # Band helpers
    # ------------------------------------------------------------------
    def _score_band(self, score: float) -> str:
        """Return the raw score band — use _effective_score_band() in generate()."""
        if score >= self.p.get("P99", float("inf")):  return "TOP1"
        if score >= self.p.get("P95", float("inf")):  return "TOP5"
        if score >= self.p.get("P80", float("inf")):  return "TOP20"
        if score >= self.p.get("P50", float("inf")):  return "TOP50"
        if score >= self.p.get("P20", float("inf")):  return "BOTTOM50"
        return "BOTTOM20"

    def _effective_score_band(self, score: float, raw_band: str) -> str:
        """
        Adjust score band based on pool quality so that:
          - OUTLIER pool: candidates below P80/2 are downgraded to BOTTOM tier
          - WEAK/COMPRESSED pool: cap max band at TOP50 (nobody gets 'strong')
            except the genuine rank-1 leader who gets TOP20 at most.
        """
        quality = self.pool_context["quality"]
        p95  = self.pool_context.get("p95", 0)
        p99  = self.pool_context["p99"]

        if quality == "OUTLIER":
            # Only genuine outliers keep strong bands.
            is_genuine_outlier = score >= (p95 + 0.5 * (p99 - p95))
            if is_genuine_outlier:
                return raw_band  # they are the outlier
            # Everyone else in an outlier pool is comparatively weak.
            if raw_band in ("TOP1", "TOP5"):   return "TOP20"
            if raw_band == "TOP20":            return "TOP50"
            if raw_band == "TOP50":            return "BOTTOM50"
            return "BOTTOM20"

        elif quality in ("WEAK", "COMPRESSED"):
            # Entire pool is weak — cap bands to prevent over-promising.
            if raw_band in ("TOP1", "TOP5"):   return "TOP20"
            if raw_band == "TOP20":            return "TOP50"
            return raw_band  # BOTTOM50 / BOTTOM20 unchanged

        return raw_band  # NORMAL

    def _is_top_of_weak_pool(self, raw_band: str, eff_band: str) -> bool:
        """True when candidate leads a WEAK/COMPRESSED pool (was TOP1/5, now capped)."""
        quality = self.pool_context["quality"]
        return (quality in ("WEAK", "COMPRESSED")
                and raw_band in ("TOP1", "TOP5")
                and eff_band in ("TOP20", "TOP50"))

    def _is_lower_in_outlier_pool(self, score: float, eff_band: str) -> bool:
        """
        True when candidate is NOT the outlier in an OUTLIER pool.
        """
        quality = self.pool_context["quality"]
        p95     = self.pool_context.get("p95", 0)
        p99     = self.pool_context["p99"]
        
        is_genuine_outlier = score >= (p95 + 0.5 * (p99 - p95))
        return (quality == "OUTLIER" and not is_genuine_outlier)

    @staticmethod
    def _rank_band(rank: int) -> str:
        if rank <= 10:    return "TOP10"
        if rank <= 100:   return "TOP100"
        if rank <= 1000:  return "TOP1000"
        return "REST"

    # ------------------------------------------------------------------
    # Evidence phrase builder (primary + optional secondary blend)
    # No archetype label names are ever returned.
    # ------------------------------------------------------------------
    def _build_evidence(self, candidate_id: str, primary: str, secondary: str) -> str:
        primary_ev   = PhraseLibrary.ARCHETYPE_EVIDENCE.get(
            primary, "applied machine learning and technical delivery"
        )
        secondary_ev = PhraseLibrary.ARCHETYPE_EVIDENCE_SHORT.get(
            secondary, "cross-functional technical contribution"
        )

        key_val = int(hashlib.sha256((candidate_id + "ev").encode()).hexdigest(), 16)

        # Only blend if primary phrase is concise enough to avoid run-ons
        if len(primary_ev) <= 55:
            if key_val % 3 == 0:
                return f"{primary_ev} and {secondary_ev}"
            elif key_val % 3 == 1:
                return primary_ev
            else:
                return f"{secondary_ev} and {primary_ev}"
        else:
            # Primary is long — use secondary alone or primary alone
            return secondary_ev if key_val % 2 == 0 else primary_ev

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------
    def generate(self, candidate_id: str, score: float, rank: int = 9999) -> str:
        """
        Generate a 3-4 sentence recruiter reasoning string.
        Target: 45-70 words.
        """

        # ---- Raw facts -----------------------------------------------
        facts       = self.data.get_candidate_facts(candidate_id)
        title       = facts.get("title", "Machine Learning Engineer")
        years_exp   = round(float(facts.get("years_exp", 5.0)), 1)
        skill_phrase = facts.get("skill_phrase", "applied machine learning and data systems")
        avail_score  = facts.get("avail_score", 0.4)

        # ---- Archetype (internal only) -------------------------------
        primary, secondary = self.profiler.get_archetypes(candidate_id)
        evidence = self._build_evidence(candidate_id, primary, secondary)

        # ---- Signal bands --------------------------------------------
        conf_tier  = self.profiler.get_confidence_tier(candidate_id)
        raw_band   = self._score_band(score)
        score_band = self._effective_score_band(score, raw_band)
        rank_band  = self._rank_band(rank)

        top_of_weak   = self._is_top_of_weak_pool(raw_band, score_band)
        lower_outlier = self._is_lower_in_outlier_pool(score, score_band)

        # Availability tier
        if avail_score >= 0.60:
            avail_tier = "HIGH"
        elif avail_score >= 0.30:
            avail_tier = "MODERATE"
        else:
            avail_tier = "LOW"

        is_bottom20 = (score_band == "BOTTOM20")

        # ==============================================================
        # Generation Loop (Deterministic rotation to prevent duplicates)
        # ==============================================================
        for attempt in range(10):
            salt_s = f"_{attempt}" if attempt > 0 else ""

            # S1 — Who is this candidate?
            s1 = self._pick(candidate_id, PhraseLibrary.OPENINGS, "s1" + salt_s).format(
                title=title,
                years_exp=years_exp,
                skill_phrase=skill_phrase,
            )

            # S2 — Why ranked here? (relative signal)
            if score_band in ("TOP1", "TOP5"):
                rel_pool = PhraseLibrary.RELATIVE_STRONG
            elif score_band in ("TOP20", "TOP50"):
                rel_pool = PhraseLibrary.RELATIVE_MODERATE
            else:
                rel_pool = PhraseLibrary.RELATIVE_WEAK

            s2 = self._pick(candidate_id, rel_pool, "s2" + salt_s).format(evidence=evidence)

            # Negative signal — always for BOTTOM20
            neg = ""
            if is_bottom20:
                neg = self._pick(candidate_id, PhraseLibrary.NEGATIVE_SIGNALS, "neg" + salt_s)

            # S3 — Global assessment + recommendation (one combined sentence)
            if top_of_weak:
                if self.pool_context["quality"] == "COMPRESSED":
                    global_phrase = self._pick(candidate_id, PhraseLibrary.GLOBAL_COMPRESSED_POOL, "g" + salt_s)
                    rec_phrase    = self._pick(candidate_id, PhraseLibrary.RECOMMENDATIONS_COMPRESSED_POOL, "rec" + salt_s)
                else:
                    global_phrase = self._pick(candidate_id, PhraseLibrary.GLOBAL_BEST_IN_WEAK_POOL, "g" + salt_s)
                    rec_phrase    = self._pick(candidate_id, PhraseLibrary.RECOMMENDATIONS_WEAK_POOL_TOP, "rec" + salt_s)

            elif lower_outlier:
                global_phrase = self._pick(candidate_id, PhraseLibrary.GLOBAL_OUTLIER_LOWER, "g" + salt_s)
                rec_phrase    = self._pick(candidate_id, PhraseLibrary.RECOMMENDATIONS_OUTLIER_LOWER, "rec" + salt_s)

            else:
                if conf_tier == "HIGH":
                    global_phrase = self._pick(candidate_id, PhraseLibrary.GLOBAL_HIGH, "g" + salt_s)
                elif conf_tier == "MODERATE":
                    global_phrase = self._pick(candidate_id, PhraseLibrary.GLOBAL_MODERATE, "g" + salt_s)
                else:
                    global_phrase = self._pick(candidate_id, PhraseLibrary.GLOBAL_EXPLORATORY, "g" + salt_s)

                rec_key  = f"{rank_band}_{conf_tier}"
                rec_pool = PhraseLibrary.RECOMMENDATIONS.get(rec_key, PhraseLibrary.RECOMMENDATIONS["REST_EXPLORATORY"])
                rec_phrase = self._pick(candidate_id, rec_pool, "rec" + salt_s)

            if not rec_phrase.endswith("."):
                rec_phrase += "."
            s3 = f"{global_phrase}; {rec_phrase}"

            # S4 — Availability (optional)
            s4 = ""
            if not is_bottom20 and (avail_tier == "HIGH" or rank <= 100):
                if avail_tier == "HIGH":
                    s4 = self._pick(candidate_id, PhraseLibrary.AVAILABILITY_HIGH, "av" + salt_s)
                elif avail_tier == "MODERATE":
                    s4 = self._pick(candidate_id, PhraseLibrary.AVAILABILITY_MODERATE, "av" + salt_s)
                else:
                    s4 = self._pick(candidate_id, PhraseLibrary.AVAILABILITY_LOW, "av" + salt_s)

            # Check fingerprint uniqueness for top 10
            fingerprint = f"{s1}|{s2}|{neg}|{s3}"
            if fingerprint not in self.seen_fingerprints or rank > 10:
                self.seen_fingerprints.add(fingerprint)
                break

        # ==============================================================
        # Assemble
        # ==============================================================
        parts = [p for p in [s1, s2, neg, s3, s4] if p]
        reasoning = " ".join(parts)

        # Word count guard — if over 75, drop S4 then re-check
        if _word_count(reasoning) > 75 and s4:
            parts = [p for p in [s1, s2, neg, s3] if p]
            reasoning = " ".join(parts)

        # Hallucination guard
        reasoning = _check_hallucination(reasoning, candidate_id)
        return reasoning
