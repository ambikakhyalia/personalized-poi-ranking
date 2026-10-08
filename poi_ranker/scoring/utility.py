"""Final utility combiner: preference (ranker) + context compatibility + geo relevance,
plus a confidence heuristic. Weights are fixed and documented, not learned, per the
assignment's "simpler is preferable" guidance — this keeps the formula auditable.
"""
from __future__ import annotations

import pandas as pd

from poi_ranker.features.traveler_features import TravelerProfile
from poi_ranker.scoring.context_scoring import context_compatibility, geo_relevance

UTILITY_WEIGHTS = {"preference": 0.50, "context": 0.35, "geo": 0.15}
POI_REVIEW_SATURATION = 50  # review_count beyond this contributes no extra confidence


def calibrate_preference_scores(raw_scores: pd.Series) -> pd.Series:
    """Min-max normalize raw ranker scores to 0-1 WITHIN one traveler's candidate set."""
    value_range = raw_scores.max() - raw_scores.min()
    if value_range <= 1e-9:
        return pd.Series(0.5, index=raw_scores.index)
    return (raw_scores - raw_scores.min()) / value_range


def compute_confidence(traveler: TravelerProfile, poi: pd.Series) -> float:
    traveler_component = min(1.0, 0.5 + 0.1 * traveler.n_interactions)
    poi_component = min(1.0, poi["review_count"] / POI_REVIEW_SATURATION)
    return float(0.5 * traveler_component + 0.5 * poi_component)


def compute_utility(traveler: TravelerProfile, poi: pd.Series, preference_score: float) -> dict:
    ctx = context_compatibility(traveler, poi)
    geo = geo_relevance(traveler, poi)
    final_score = (
        UTILITY_WEIGHTS["preference"] * preference_score
        + UTILITY_WEIGHTS["context"] * ctx
        + UTILITY_WEIGHTS["geo"] * geo
    )
    return {
        "poi_id": poi["poi_id"],
        "score": round(float(final_score), 4),
        "preference_weight": round(float(preference_score), 4),
        "context_compatibility": round(float(ctx), 4),
        "confidence": round(compute_confidence(traveler, poi), 4),
    }
