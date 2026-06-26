"""
PhraseLibrary v4.0
------------------
Structure:
  S1: {title} + {years_exp} + {skill_phrase}  (~12-15 words)
  S2: Relative signal using {evidence}         (~12-15 words)
  [Negative: forced for BOTTOM20]             (~9-11 words)
  S3: {global}; {recommendation}              (~14-18 words)
  S4 (optional): availability                 (~5-8 words)
  Target total: 45-70 words.

Rules:
  - ARCHETYPE_EVIDENCE values: descriptive phrases ONLY, no taxonomy labels.
  - No raw feature column names anywhere.
  - No unsupported claims: no team sizes, revenue, scale numbers, leadership
    claims without supporting evidence.
"""


class PhraseLibrary:

    # ===================================================================
    # SENTENCE 1 — Who is this candidate? (30 openings)
    # {title}, {years_exp}, {skill_phrase}
    # ===================================================================
    OPENINGS = [
        "{title} with {years_exp} years of experience in {skill_phrase}.",
        "{title} with {years_exp} years of applied work across {skill_phrase}.",
        "{title} with a {years_exp}-year track record in {skill_phrase}.",
        "{title} with {years_exp} years of technical depth in {skill_phrase}.",
        "{title} with {years_exp} years of documented delivery in {skill_phrase}.",
        "{title} bringing {years_exp} years of focused work in {skill_phrase}.",
        "{title} with {years_exp} years of progressive experience in {skill_phrase}.",
        "{title} with {years_exp} years of hands-on work in {skill_phrase}.",
        "{title} offering {years_exp} years of practical exposure to {skill_phrase}.",
        "{title} with {years_exp} years of applied industry experience in {skill_phrase}.",
        "{title} demonstrating {years_exp} years of consistent contribution to {skill_phrase}.",
        "{title} with {years_exp} years of technical work spanning {skill_phrase}.",
        "{title} with {years_exp} years of domain exposure in {skill_phrase}.",
        "{title} with {years_exp} years of functional and technical work in {skill_phrase}.",
        "Engineering professional with {years_exp} years specializing in {skill_phrase}.",
        "Technology practitioner with {years_exp} years of work in {skill_phrase}.",
        "Technical specialist with {years_exp} years of applied {skill_phrase} experience.",
        "Practitioner with {years_exp} years of progressive work in {skill_phrase}.",
        "Data and engineering professional with {years_exp} years focused on {skill_phrase}.",
        "Technology professional with {years_exp} years of industry work in {skill_phrase}.",
        "{title} with {years_exp} years of technical involvement in {skill_phrase}.",
        "{title} building {skill_phrase} over {years_exp} years of industry experience.",
        "{title} with {years_exp} years delivering {skill_phrase} capabilities across roles.",
        "{title} with {years_exp} years of accumulated {skill_phrase} experience.",
        "Applied ML professional with {years_exp} years focused on {skill_phrase}.",
        "{title} with {years_exp} years of practical {skill_phrase} background.",
        "Engineering contributor with {years_exp} years of applied {skill_phrase} experience.",
        "{title} with {years_exp} years of technical delivery spanning {skill_phrase}.",
        "{title} presenting {years_exp} years of industry exposure to {skill_phrase}.",
        "{title} with {years_exp} years of cumulative technical experience in {skill_phrase}.",
    ]

    # ===================================================================
    # SENTENCE 2 — Why ranked here? Relative signal vs pool
    # {evidence} = descriptive phrase from ARCHETYPE_EVIDENCE (no labels)
    # ===================================================================

    # STRONG — score band TOP1 / TOP5 (top 5% of pool)
    RELATIVE_STRONG = [
        "Compared with most candidates in this evaluation, demonstrates stronger {evidence}.",
        "Among the more technically mature profiles reviewed, stands out through {evidence}.",
        "Relative to evaluated peers, provides notably more compelling signals of {evidence}.",
        "Within this evaluation set, consistently outperforms peers on {evidence}.",
        "Against candidates reviewed, delivers more differentiated signals of {evidence}.",
        "The profile provides superior evidence of {evidence} relative to this candidate pool.",
        "Stronger and more consistent signals of {evidence} distinguish this profile from peers.",
        "Relative to the pool, demonstrates more developed and credible {evidence}.",
        "Among candidates evaluated, offers more substantive evidence of {evidence} than typical peers.",
        "This profile shows more credible {evidence} than most others reviewed in this set.",
        "Evidence of {evidence} is notably stronger here than across most evaluated profiles.",
        "The profile distinguishes itself within this evaluation through stronger evidence of {evidence}.",
        "Compared with the broader candidate pool reviewed, {evidence} is more consistently evidenced.",
        "Relative to peer profiles, the candidate demonstrates a more mature foundation in {evidence}.",
    ]

    # MODERATE — score band TOP20 / TOP50 (top 20%-50%)
    RELATIVE_MODERATE = [
        "Shows solid evidence of {evidence}, though not uniformly across all evaluated dimensions.",
        "Demonstrates competitive signals of {evidence} in key areas, with some dimensions stronger than others.",
        "Provides credible but not dominant evidence of {evidence} relative to this candidate pool.",
        "Within the evaluation set, offers adequate signals of {evidence} with areas of genuine strength.",
        "Compared with peers reviewed, the profile presents a reasonable technical baseline in {evidence}.",
        "Shows partial alignment through {evidence}, though overall depth is mixed.",
        "Evidence of {evidence} is present, placing the profile in the competitive mid-range of this set.",
        "Relative to this evaluation pool, signals of {evidence} are solid but not exceptional.",
        "Demonstrates a credible technical foundation in {evidence}, with some dimensions requiring further validation.",
        "Among evaluated candidates, presents moderate but viable signals of {evidence}.",
        "Shows above-average evidence of {evidence} in select areas, with uneven consistency overall.",
        "Relative to the candidate pool, {evidence} signals are solid but not among the most differentiated.",
        "The profile compares favorably with mid-tier candidates through evidence of {evidence}.",
    ]

    # WEAK — score band BOTTOM50 / BOTTOM20 (bottom 50%)
    RELATIVE_WEAK = [
        "Provides comparatively less evidence of {evidence} than higher-ranked profiles in this set.",
        "Relative to stronger candidates reviewed, demonstrates fewer differentiated signals of {evidence}.",
        "Within this evaluation pool, offers limited evidence of {evidence} compared with top-ranked profiles.",
        "Against higher-ranked candidates, delivers weaker signals of {evidence}.",
        "Compared with profiles ranked above, provides less consistent evidence of {evidence}.",
        "The profile ranks below the central cluster, offering limited signals of {evidence}.",
        "Against the stronger profiles reviewed, evidence of {evidence} is less developed.",
        "Relative to the evaluation set, signals of {evidence} are below the observed median.",
        "Compared with the top tier of this evaluation, the {evidence} foundation is narrower.",
        "Within this pool, provides baseline but undifferentiated signals of {evidence}.",
    ]

    # ===================================================================
    # NEGATIVE SIGNAL — forced for BOTTOM20, always without placeholders
    # ===================================================================
    NEGATIVE_SIGNALS = [
        "Evidence of large-scale system ownership and domain specialization remains limited.",
        "The profile demonstrates fewer differentiated technical signals than higher-ranked candidates.",
        "Provides comparatively less evidence of sustained work in the core areas prioritized for this role.",
        "While some transferable strengths are present, overall alignment with the role appears modest.",
        "Technical depth in the core areas evaluated is limited relative to the broader candidate set.",
        "The profile does not demonstrate the breadth or specialization observed in higher-ranked candidates.",
        "Differentiated technical evidence is sparse; the profile aligns more closely with generalist backgrounds.",
        "Limited signals of domain-specific specialization and end-to-end technical ownership are present.",
        "Core technical signals are below the threshold observed for higher-priority candidates.",
        "Evidence of sustained delivery and specialized technical ownership is limited.",
        "The profile lacks the consistent depth signals observed in candidates ranked significantly higher.",
        "Available signals reflect a more generalist background with limited specialized contribution.",
        "End-to-end technical ownership and sustained delivery evidence is weaker than higher-ranked peers.",
        "The profile shows partial overlap with role requirements but lacks consistent depth signals.",
        "Overall alignment with the prioritized technical areas is modest based on available evidence.",
    ]

    # ===================================================================
    # SENTENCE 3 (part A) — Global assessment (conf tier only, ~8-10 words)
    # ===================================================================
    GLOBAL_HIGH = [
        "Multiple signals confirm strong technical evidence across the broader talent pool",
        "Evidence from the broader 100k pool is consistently strong and differentiated",
        "Cross-referenced against the broader distribution, the profile demonstrates strong signals",
        "The profile compares favorably against the full candidate distribution on key dimensions",
        "Benchmarked against the broader pool, evidence is consistently above average",
    ]

    GLOBAL_MODERATE = [
        "Some signals compare favorably against the broader pool, though evidence is uneven",
        "Against the broader distribution, the profile demonstrates above-average signals in select areas",
        "Evidence from the full pool is moderate; some dimensions compare well, others less so",
        "The profile shows credible but uneven performance when benchmarked against the full distribution",
        "Against the broader pool, signal strength is solid in specific areas but not uniform",
    ]

    GLOBAL_EXPLORATORY = [
        "Benchmarked against the broader pool, differentiated technical signals are limited",
        "Signal density relative to the full candidate distribution is below average",
        "Evidence of specialized technical depth is limited relative to the broader pool",
        "Against the full distribution, the profile does not demonstrate consistently differentiated signals",
        "The broader pool benchmark reveals limited corroborating evidence of technical specialization",
    ]

    # ===================================================================
    # SENTENCE 3 (part B) — Recommendation (rank_band × conf_tier)
    # Key format: "{BAND}_{CONF}"
    # ===================================================================
    RECOMMENDATIONS = {
        # TOP10 = rank 1-10
        "TOP10_HIGH": [
            "immediate interview prioritization is recommended.",
            "priority outreach and a first-round technical evaluation are advised.",
            "fast-track for early technical interview without delay.",
            "this candidate warrants immediate scheduling for a structured technical assessment.",
            "first-priority scheduling for a technical interview is recommended.",
        ],
        "TOP10_MODERATE": [
            "early evaluation is recommended to validate several promising signals.",
            "a structured first-round technical screen is the appropriate next step.",
            "a focused technical screen is recommended to confirm depth before advancing.",
            "early-stage screening is advised to validate the signals observed.",
        ],
        "TOP10_EXPLORATORY": [
            "a focused technical screen is recommended before advancing to interview stage.",
            "additional screening would help validate the profile before formal prioritization.",
            "further evaluation is needed before committing to interview prioritization.",
        ],

        # TOP100 = rank 11-100
        "TOP100_HIGH": [
            "this profile merits serious consideration for a first-round technical evaluation.",
            "a structured technical screen is recommended without delay.",
            "inclusion in the shortlist for structured first-round screening is warranted.",
            "scheduling a technical evaluation is a well-supported next step.",
        ],
        "TOP100_MODERATE": [
            "a first-round technical screen is a reasonable next step.",
            "further review is warranted before advancing to interview stage.",
            "a focused screening call is the proportionate next step.",
            "a brief technical evaluation would help confirm overall fit.",
        ],
        "TOP100_EXPLORATORY": [
            "further review before committing screening resources is advised.",
            "a brief exploratory screen would help clarify overall fit.",
            "conditional consideration is appropriate pending a short screening call.",
        ],

        # TOP1000 = rank 101-1000
        "TOP1000_HIGH": [
            "further review may be worthwhile given the stronger signals observed.",
            "additional screening may surface technical depth not fully captured in the profile.",
            "a light-touch screen may help determine whether the profile warrants advancement.",
        ],
        "TOP1000_MODERATE": [
            "further review may be worthwhile before committing screening resources.",
            "additional screening is recommended to assess depth before prioritization.",
            "conditional consideration pending further evaluation is appropriate.",
        ],
        "TOP1000_EXPLORATORY": [
            "exploratory consideration may be appropriate if adjacent experience is of interest.",
            "further screening before prioritization is advised given limited differentiation.",
            "a preliminary conversation would clarify whether the profile meets minimum requirements.",
        ],

        # REST = rank > 1000
        "REST_HIGH": [
            "exploratory consideration may be appropriate given specific technical strengths observed.",
            "long-term pipelining may be more appropriate than immediate engagement.",
            "the profile may warrant future consideration as requirements evolve.",
        ],
        "REST_MODERATE": [
            "exploratory consideration may be appropriate if adjacent experience is valued.",
            "the profile is better suited for long-term pipelining than immediate engagement.",
            "active prioritization is not recommended; future consideration may be more appropriate.",
        ],
        "REST_EXPLORATORY": [
            "the profile may be better suited for adjacent opportunities or future consideration.",
            "active prioritization is not recommended at this stage.",
            "the profile is not recommended for immediate engagement; long-term pipelining is more appropriate.",
        ],
    }

    # ===================================================================
    # SENTENCE 4 — Availability (optional, only if HIGH avail or rank <= 100)
    # Short: 5-8 words
    # ===================================================================
    AVAILABILITY_HIGH = [
        "Recent activity suggests near-term availability.",
        "Profile signals indicate openness to new opportunities.",
        "Engagement patterns suggest favorable timing for outreach.",
        "Recent platform activity implies the candidate is currently reachable.",
        "Activity signals suggest the candidate may be available in the near term.",
        "Digital engagement suggests receptivity to initial contact.",
    ]

    AVAILABILITY_MODERATE = [
        "Availability timing warrants an initial contact to confirm openness.",
        "A light-touch outreach is advisable to gauge current interest.",
        "A brief exploratory outreach is recommended to assess availability.",
        "Candidate availability is uncertain; initial contact is advisable.",
    ]

    AVAILABILITY_LOW = [
        "Engagement signals are limited; long-term pipelining is more appropriate.",
        "Platform activity is sparse; sustained outreach may be required.",
        "Low engagement signals suggest this is not an optimal outreach window.",
        "The candidate may not be actively seeking new roles at this time.",
    ]

    # ===================================================================
    # INTERNAL MAPPINGS — never appear in output as archetype label names
    # ===================================================================

    # Evidence phrases (used in S2 via {evidence}) — descriptive only
    ARCHETYPE_EVIDENCE = {
        "Retrieval Architect":    "experience building search, ranking, and large-scale retrieval systems",
        "Production ML Builder":  "experience delivering production machine learning systems end-to-end",
        "Technical Leader":       "evidence of technical ownership and progressively complex project responsibilities",
        "Research Specialist":    "deep specialization in advanced machine learning and modeling techniques",
        "Experienced Generalist": "broad technical exposure across multiple machine learning domains",
        "Hidden Gem":             "strong practical technical signals despite limited external visibility",
    }

    # Short versions for secondary blending (~4-6 words)
    ARCHETYPE_EVIDENCE_SHORT = {
        "Retrieval Architect":    "production retrieval and ranking delivery",
        "Production ML Builder":  "end-to-end ML production ownership",
        "Technical Leader":       "technical ownership and project progression",
        "Research Specialist":    "advanced modeling specialization",
        "Experienced Generalist": "broad multi-domain ML delivery",
        "Hidden Gem":             "practical technical contribution",
    }

    # Gap phrases (used in caveats via {gap})
    ARCHETYPE_GAPS = {
        "Retrieval Architect":    "large-scale retrieval and ranking system",
        "Production ML Builder":  "end-to-end production ML",
        "Technical Leader":       "technical ownership and architectural",
        "Research Specialist":    "advanced research and modeling",
        "Experienced Generalist": "specialized machine learning",
        "Hidden Gem":             "technical depth",
    }

    # Skill domain phrases (used in S1 via {skill_phrase} fallback)
    ARCHETYPE_DOMAINS = {
        "Retrieval Architect":    "search, ranking, and retrieval",
        "Production ML Builder":  "production ML and MLOps",
        "Technical Leader":       "machine learning systems and architecture",
        "Research Specialist":    "machine learning research and modeling",
        "Experienced Generalist": "machine learning and data engineering",
        "Hidden Gem":             "applied machine learning",
    }

    # ===================================================================
    # POOL-ADAPTIVE PHRASE POOLS
    # Used when pool_quality != NORMAL
    # ===================================================================

    # For rank #1 / top candidate in a WEAK or COMPRESSED pool.
    # Acknowledges they lead the set but qualifies the absolute assessment.
    GLOBAL_BEST_IN_WEAK_POOL = [
        "While leading this evaluation set, overall pool quality is limited; "
        "the absolute strength of this profile requires further validation",
        "The profile ranks highest within this evaluation, though the candidate set "
        "shows limited technical depth overall; independent validation is recommended",
        "Against the evaluated pool, this candidate demonstrates the strongest signals, "
        "though pool-level quality is below typical benchmarks; further assessment is needed",
        "This profile leads the current evaluation set; however, the pool does not represent "
        "a high-competition baseline, and additional screening is recommended before prioritization",
    ]

    # Recommendations for rank #1 in a WEAK pool — capped at moderate
    RECOMMENDATIONS_WEAK_POOL_TOP = [
        "further evaluation before formal prioritization is recommended.",
        "a structured screening call is advised to validate observed signals independently.",
        "a focused technical screen is the appropriate next step before advancing.",
    ]

    # For top candidates in a COMPRESSED pool where top scores are nearly identical.
    GLOBAL_COMPRESSED_POOL = [
        "Several profiles in this evaluation set exhibit similar levels of evidence, making absolute differentiation challenging",
        "The top of this evaluation pool is highly compressed with multiple profiles showing comparable technical depth",
        "Signal density among the top candidates is nearly identical, making it difficult to establish a single clear leader",
    ]

    RECOMMENDATIONS_COMPRESSED_POOL = [
        "several profiles exhibit similar levels of evidence and may require additional screening.",
        "a comparative screening approach is recommended to differentiate among the top cluster.",
        "additional technical evaluation is advised to separate this candidate from peers with similar profiles.",
    ]

    # For non-outlier candidates in an OUTLIER pool.
    # They rank within the set but the set is dominated by a clear leader.
    GLOBAL_OUTLIER_LOWER = [
        "Against the broader distribution, the profile does not demonstrate "
        "differentiated signals comparable to the stronger candidates in this evaluation",
        "The evaluation set contains significantly stronger profiles; "
        "this candidate provides fewer differentiated signals relative to the pool leader",
        "Signal density is limited relative to the top-ranked candidate in this set; "
        "the profile does not compare favorably against the stronger profiles evaluated",
        "Within an unbalanced evaluation pool, this profile ranks below the clear top candidate "
        "and provides fewer distinguishing technical signals",
    ]

    # Recommendations for lower-ranked candidates in an OUTLIER pool
    RECOMMENDATIONS_OUTLIER_LOWER = [
        "exploratory consideration may be appropriate if the top-ranked candidates are unavailable.",
        "further review is only recommended if higher-ranked candidates do not proceed.",
        "the profile is a backup consideration; prioritization of stronger candidates is advised.",
    ]
