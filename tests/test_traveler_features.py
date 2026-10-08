import pandas as pd

from poi_ranker.features.traveler_features import build_traveler_profile


def test_cold_start_traveler_has_no_implicit_signal(travelers: pd.DataFrame, poi_table: pd.DataFrame) -> None:
    traveler_row = travelers.iloc[0]
    empty_interactions = pd.DataFrame(columns=["traveler_id", "poi_id", "interaction", "timestamp"])
    profile = build_traveler_profile(traveler_row, empty_interactions, poi_table)

    assert profile.is_cold_start
    assert profile.implicit_tag_affinity == {}
    assert profile.avg_touristy_affinity is None


def test_warm_traveler_has_implicit_signal(
    travelers: pd.DataFrame, interactions: pd.DataFrame, poi_table: pd.DataFrame
) -> None:
    traveler_row = travelers.iloc[0]
    profile = build_traveler_profile(traveler_row, interactions, poi_table)

    assert not profile.is_cold_start
    assert profile.n_interactions > 0
