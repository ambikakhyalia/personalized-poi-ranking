"""Cross features: combine traveler signals with a specific POI's content features.

These are the ML features fed to the ranker. Practical-compatibility terms (budget/
mobility/availability/geo) are NOT recomputed here — they are imported from
scoring.context_scoring, the single source of truth, and reused as-is.
"""
from __future__ import annotations

import pandas as pd

from poi_ranker.features.traveler_features import TravelerProfile
from poi_ranker.scoring.context_scoring import (
    availability,
    budget_compatibility,
    geo_relevance,
    mobility_compatibility,
)

CROSS_FEATURE_COLUMNS = [
    "interest_tag_overlap",
    "implicit_tag_affinity_score",
    "implicit_category_affinity_score",
    "touristy_preference_fit",
    "similarity_to_visited",
    "budget_compatibility",
    "mobility_compatibility",
    "availability",
    "geo_relevance",
]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _touristy_preference_fit(traveler: TravelerProfile, poi: pd.Series) -> float:
    is_touristy = bool(poi["is_touristy"])
    if traveler.explicit_preference == "prefer_less_touristy":
        explicit_component = 0.3 if is_touristy else 1.0
    elif traveler.explicit_preference == "prefer_famous_landmarks":
        explicit_component = 1.0 if is_touristy else 0.6
    else:
        explicit_component = 0.8

    if traveler.avg_touristy_affinity is None:
        return explicit_component
    implicit_component = 1.0 - abs((1.0 if is_touristy else 0.0) - traveler.avg_touristy_affinity)
    return float((explicit_component + implicit_component) / 2.0)


def build_cross_features(traveler: TravelerProfile, poi: pd.Series) -> dict[str, float]:
    return {
        "interest_tag_overlap": _jaccard(traveler.interests, poi["tag_set"]),
        "implicit_tag_affinity_score": sum(traveler.implicit_tag_affinity.get(t, 0.0) for t in poi["tag_set"]),
        "implicit_category_affinity_score": traveler.implicit_category_affinity.get(poi["category"], 0.0),
        "touristy_preference_fit": _touristy_preference_fit(traveler, poi),
        "similarity_to_visited": _jaccard(traveler.visited_tag_union, poi["tag_set"]),
        "budget_compatibility": budget_compatibility(traveler, poi),
        "mobility_compatibility": mobility_compatibility(traveler, poi),
        "availability": availability(traveler, poi),
        "geo_relevance": geo_relevance(traveler, poi),
    }
