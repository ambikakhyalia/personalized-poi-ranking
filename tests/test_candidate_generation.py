import pandas as pd

from poi_ranker.candidates.candidate_generation import generate_candidates
from poi_ranker.features.traveler_features import build_traveler_profile


def test_candidates_restricted_to_traveler_destination(
    travelers: pd.DataFrame, interactions: pd.DataFrame, poi_table: pd.DataFrame
) -> None:
    traveler_row = travelers.iloc[0]
    profile = build_traveler_profile(traveler_row, interactions, poi_table)
    candidates = generate_candidates(profile, poi_table)

    assert not candidates.empty
    assert (candidates["destination"] == traveler_row["destination"]).all()


def test_long_tail_guarantee_surfaces_below_median_popularity_pois(
    travelers: pd.DataFrame, interactions: pd.DataFrame, poi_table: pd.DataFrame
) -> None:
    traveler_row = travelers.iloc[0]
    profile = build_traveler_profile(traveler_row, interactions, poi_table)
    candidates = generate_candidates(profile, poi_table, max_candidates=15, long_tail_slots=5)

    dest_pois = poi_table[poi_table["destination"] == traveler_row["destination"]]
    median_popularity = dest_pois["popularity"].median()
    n_below_median = (candidates["popularity"] < median_popularity).sum()

    assert n_below_median > 0, "candidate generation eliminated all long-tail POIs"
