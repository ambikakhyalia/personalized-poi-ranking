"""End-to-end orchestration: traveler -> ranked, scored, explained POI recommendations."""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from poi_ranker.candidates.candidate_generation import generate_candidates
from poi_ranker.explainability.explain import explain_recommendation
from poi_ranker.features.traveler_features import build_traveler_profile
from poi_ranker.ranking.dataset_builder import FEATURE_COLUMNS
from poi_ranker.ranking.model import predict_raw_scores
from poi_ranker.scoring.utility import calibrate_preference_scores, compute_utility


def recommend_for_traveler(
    traveler_row: pd.Series,
    poi_table: pd.DataFrame,
    interactions: pd.DataFrame,
    ranker: lgb.LGBMRanker,
    top_k: int = 10,
) -> list[dict]:
    profile = build_traveler_profile(traveler_row, interactions, poi_table)
    candidates = generate_candidates(profile, poi_table)
    if candidates.empty:
        return []

    raw_scores = predict_raw_scores(ranker, candidates[FEATURE_COLUMNS])
    preference_scores = calibrate_preference_scores(raw_scores)

    results = []
    for idx, poi in candidates.iterrows():
        preference_score = preference_scores.loc[idx]
        result = compute_utility(profile, poi, preference_score)
        result["name"] = poi["name"]
        result["category"] = poi["category"]
        result["reasons"] = explain_recommendation(profile, poi)
        results.append(result)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
