"""Candidate generation: shrink the destination's POI catalog to a manageable set.

Two pools are combined:
  1. Main pool: top-ranked by a lightweight relevance score (interest, budget, geo,
     behavioral affinity, touristy fit).
  2. Long-tail guarantee pool: reserved slots for below-median-popularity POIs that
     still clear a minimum relevance floor — this is the explicit mechanism that
     prevents popularity bias from erasing relevant local/long-tail POIs.

Cross features are computed once here and reused by ranking/scoring downstream
(passed through as columns) instead of being recomputed at each stage.
"""
from __future__ import annotations

import pandas as pd

from poi_ranker.features.cross_features import build_cross_features
from poi_ranker.features.traveler_features import TravelerProfile

RELEVANCE_WEIGHTS = {
    "interest_tag_overlap": 0.30,
    "budget_compatibility": 0.20,
    "category_affinity_norm": 0.15,
    "tag_affinity_norm": 0.10,
    "touristy_preference_fit": 0.10,
    "geo_relevance": 0.15,
}


def _normalize(series: pd.Series) -> pd.Series:
    value_range = series.max() - series.min()
    if value_range <= 1e-9:
        return series * 0.0
    return (series - series.min()) / value_range


def generate_candidates(
    traveler: TravelerProfile,
    poi_table: pd.DataFrame,
    max_candidates: int = 40,
    long_tail_slots: int = 8,
) -> pd.DataFrame:
    dest_pois = poi_table[poi_table["destination"] == traveler.destination].copy()
    if dest_pois.empty:
        return dest_pois

    cross = dest_pois.apply(lambda row: build_cross_features(traveler, row), axis=1, result_type="expand")
    dest_pois = pd.concat([dest_pois, cross], axis=1)
    dest_pois["category_affinity_norm"] = _normalize(dest_pois["implicit_category_affinity_score"])
    dest_pois["tag_affinity_norm"] = _normalize(dest_pois["implicit_tag_affinity_score"])

    dest_pois["relevance_score"] = sum(
        weight * dest_pois[col] for col, weight in RELEVANCE_WEIGHTS.items()
    )

    ranked = dest_pois.sort_values("relevance_score", ascending=False)
    main_n = max(1, max_candidates - long_tail_slots)
    main_pool = ranked.iloc[:main_n]
    remaining = ranked.iloc[main_n:]

    median_popularity = dest_pois["popularity"].median()
    relevance_floor = remaining["relevance_score"].quantile(0.4) if len(remaining) else 0.0
    long_tail_eligible = remaining[
        (remaining["popularity"] < median_popularity) & (remaining["relevance_score"] > relevance_floor)
    ].sort_values("relevance_score", ascending=False)
    long_tail_pool = long_tail_eligible.iloc[:long_tail_slots]

    candidates = pd.concat([main_pool, long_tail_pool]).drop_duplicates(subset="poi_id")
    return candidates
