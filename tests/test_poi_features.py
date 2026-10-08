import pandas as pd

from poi_ranker.features.poi_features import POI_CONTENT_FEATURE_COLUMNS, build_poi_feature_table


def test_poi_feature_table_has_expected_columns(poi_table: pd.DataFrame) -> None:
    for col in POI_CONTENT_FEATURE_COLUMNS:
        assert col in poi_table.columns


def test_novelty_score_is_inverse_of_popularity(pois: pd.DataFrame) -> None:
    table = build_poi_feature_table(pois)
    most_popular = table.loc[table["popularity"].idxmax()]
    least_popular = table.loc[table["popularity"].idxmin()]
    assert most_popular["novelty_score"] < least_popular["novelty_score"]


def test_feature_values_are_bounded(poi_table: pd.DataFrame) -> None:
    assert poi_table["log_popularity_norm"].between(0, 1).all()
    assert poi_table["novelty_score"].between(0, 1).all()
