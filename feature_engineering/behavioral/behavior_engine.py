import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATES_PATH = BASE_DIR / "data" / "candidates.jsonl"
OUTPUT_PATH = BASE_DIR / "artifacts" / "behavior_features.parquet"

# =====================================================
# DATE HELPERS
# =====================================================

TODAY = datetime(2026, 6, 15)


def days_since(date_string):

    try:
        dt = datetime.strptime(
            date_string,
            "%Y-%m-%d"
        )

        return (
            TODAY - dt
        ).days

    except:
        return 365


# =====================================================
# AVAILABILITY SCORE
# =====================================================

def availability_score(signals):

    score = 0

    if signals["open_to_work_flag"]:
        score += 0.4

    last_active_days = days_since(
        signals["last_active_date"]
    )

    if last_active_days <= 30:
        score += 0.3

    elif last_active_days <= 90:
        score += 0.15

    notice = signals[
        "notice_period_days"
    ]

    if notice <= 30:
        score += 0.3

    elif notice <= 60:
        score += 0.15

    return min(score, 1.0)


# =====================================================
# RECRUITABILITY SCORE
# =====================================================

def recruitability_score(signals):

    response_rate = signals[
        "recruiter_response_rate"
    ]

    interview_rate = signals[
        "interview_completion_rate"
    ]

    offer_rate = max(
        0,
        signals[
            "offer_acceptance_rate"
        ]
    )

    return round(
        (
            response_rate * 0.4
            + interview_rate * 0.3
            + offer_rate * 0.3
        ),
        4
    )


# =====================================================
# MARKET DEMAND SCORE
# =====================================================

def market_demand_score(signals):

    views = min(
        signals[
            "profile_views_received_30d"
        ] / 100,
        1.0
    )

    saves = min(
        signals[
            "saved_by_recruiters_30d"
        ] / 20,
        1.0
    )

    searches = min(
        signals[
            "search_appearance_30d"
        ] / 500,
        1.0
    )

    return round(
        (
            views * 0.3
            + saves * 0.3
            + searches * 0.4
        ),
        4
    )


# =====================================================
# TRUST SCORE
# =====================================================

def trust_score(signals):

    score = (
        signals[
            "profile_completeness_score"
        ] / 100
    )

    if signals["verified_email"]:
        score += 0.05

    if signals["verified_phone"]:
        score += 0.05

    if signals["linkedin_connected"]:
        score += 0.05

    return round(
        min(score, 1.0),
        4
    )


# =====================================================
# TECHNICAL CREDIBILITY
# =====================================================

def technical_credibility_score(signals):

    github = max(
        0,
        signals[
            "github_activity_score"
        ]
    ) / 100

    endorsements = min(
        signals[
            "endorsements_received"
        ] / 100,
        1.0
    )

    assessments = signals[
        "skill_assessment_scores"
    ]

    if len(assessments) > 0:

        avg_assessment = (
            sum(
                assessments.values()
            )
            /
            len(assessments)
        ) / 100

    else:
        avg_assessment = 0

    return round(
        (
            github * 0.4
            + endorsements * 0.2
            + avg_assessment * 0.4
        ),
        4
    )


# =====================================================
# MOBILITY SCORE
# =====================================================

def mobility_score(signals):

    score = 0

    if signals[
        "willing_to_relocate"
    ]:
        score += 0.5

    mode = signals[
        "preferred_work_mode"
    ]

    if mode == "flexible":
        score += 0.5

    elif mode == "hybrid":
        score += 0.4

    elif mode == "onsite":
        score += 0.2

    elif mode == "remote":
        score += 0.2

    return min(score, 1.0)


# =====================================================
# MAIN
# =====================================================

rows = []

with open(
    CANDIDATES_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in tqdm(
        f,
        desc="Processing Behavior Signals"
    ):

        candidate = json.loads(line)

        signals = candidate[
            "redrob_signals"
        ]

        assessments = signals[
            "skill_assessment_scores"
        ]

        if len(assessments) > 0:

            avg_assessment = round(
                sum(
                    assessments.values()
                )
                /
                len(assessments),
                2
            )

        else:
            avg_assessment = 0

        rows.append({

            "candidate_id":
                candidate[
                    "candidate_id"
                ],

            "availability_score":
                availability_score(
                    signals
                ),

            "recruitability_score":
                recruitability_score(
                    signals
                ),

            "market_demand_score":
                market_demand_score(
                    signals
                ),

            "trust_score":
                trust_score(
                    signals
                ),

            "technical_credibility_score":
                technical_credibility_score(
                    signals
                ),

            "mobility_score":
                mobility_score(
                    signals
                ),

            "days_since_active":
                days_since(
                    signals[
                        "last_active_date"
                    ]
                ),

            "notice_period_days":
                signals[
                    "notice_period_days"
                ],

            "github_activity_score":
                signals[
                    "github_activity_score"
                ],

            "avg_assessment_score":
                avg_assessment
        })

# =====================================================
# SAVE
# =====================================================

df = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\nBehavior Feature Generation Complete")
print(df.shape)
print(df.head())

print(
    f"\nSaved -> {OUTPUT_PATH}"
)