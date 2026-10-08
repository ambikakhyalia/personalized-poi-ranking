import pandas as pd

from poi_ranker.features.traveler_features import build_traveler_profile
from poi_ranker.scoring.context_scoring import (
    availability,
    budget_compatibility,
    context_compatibility,
    geo_relevance,
    mobility_compatibility,
)
from poi_ranker.scoring.utility import calibrate_preference_scores, compute_confidence, compute_utility


def test_context_scoring_functions_are_bounded(
    travelers: pd.DataFrame, interactions: pd.DataFrame, poi_table: pd.DataFrame
) -> None:
    traveler_row = travelers.iloc[0]
    profile = build_traveler_profile(traveler_row, interactions, poi_table)
    dest_pois = poi_table[poi_table["destination"] == traveler_row["destination"]]

    for _, poi in dest_pois.iterrows():
        assert 0.0 <= budget_compatibility(profile, poi) <= 1.0
        assert 0.0 <= mobility_compatibility(profile, poi) <= 1.0
        assert 0.0 <= availability(profile, poi) <= 1.0
        assert 0.0 <= geo_relevance(profile, poi) <= 1.0
        assert 0.0 <= context_compatibility(profile, poi) <= 1.0


def test_calibrate_preference_scores_is_min_max_0_to_1() -> None:
    raw = pd.Series([2.0, 5.0, -1.0, 5.0])
    calibrated = calibrate_preference_scores(raw)
    assert calibrated.min() == 0.0
    assert calibrated.max() == 1.0


def test_calibrate_preference_scores_handles_constant_input() -> None:
    raw = pd.Series([3.0, 3.0, 3.0])
    calibrated = calibrate_preference_scores(raw)
    assert (calibrated == 0.5).all()


def test_compute_utility_output_schema(
    travelers: pd.DataFrame, interactions: pd.DataFrame, poi_table: pd.DataFrame
) -> None:
    traveler_row = travelers.iloc[0]
    profile = build_traveler_profile(traveler_row, interactions, poi_table)
    poi = poi_table[poi_table["destination"] == traveler_row["destination"]].iloc[0]

    result = compute_utility(profile, poi, preference_score=0.7)
    for key in ("poi_id", "score", "preference_weight", "context_compatibility", "confidence"):
        assert key in result
    assert 0.0 <= result["confidence"] <= 1.0
