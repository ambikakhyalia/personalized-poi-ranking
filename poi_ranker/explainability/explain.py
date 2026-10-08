"""Rule-based templated explanations for top recommendations.

Reads pre-computed cross/content feature columns off the candidate row (already
computed once during candidate generation) rather than recomputing anything.
"""
from __future__ import annotations

import pandas as pd

from poi_ranker.features.traveler_features import TravelerProfile

REASON_THRESHOLD = 0.3


def explain_recommendation(traveler: TravelerProfile, candidate: pd.Series, top_n: int = 4) -> list[str]:
    reasons: list[str] = []

    if candidate["interest_tag_overlap"] > REASON_THRESHOLD:
        reasons.append("Strong match with traveler's stated interests")
    if candidate["implicit_category_affinity_score"] > 0:
        reasons.append(f"Similar to POIs the traveler previously engaged with ({candidate['category']})")
    if candidate["similarity_to_visited"] > 0.2:
        reasons.append("Similar to previously visited/booked experiences")
    if candidate["budget_compatibility"] >= 0.8:
        reasons.append("Within budget")
    if candidate["mobility_compatibility"] >= 0.8:
        reasons.append(f"Suitable for {traveler.mobility.replace('_', ' ')}")
    if candidate["novelty_score"] > 0.6 and candidate["touristy_preference_fit"] > 0.6:
        reasons.append("Lower tourist concentration than comparable POIs")
    if candidate["geo_relevance"] > 0.7:
        reasons.append("Conveniently located relative to the trip destination")
    if candidate["availability"] < 1.0:
        reasons.append("Note: limited overlap between opening days and trip dates")

    return reasons[:top_n] if reasons else ["Meets baseline relevance and practical-fit criteria"]
