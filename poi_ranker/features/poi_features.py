"""POI representation: turn raw catalog rows into a numeric feature table.

Design: mix of category one-hot, normalized numeric signals, and a "novelty" score
(inverse log-popularity) so downstream ranking can favor local/long-tail POIs
independently of raw popularity, rather than just reproducing a popularity ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from poi_ranker.data.taxonomy import CATEGORIES


def _parse_tags(tags: str) -> set[str]:
    return set(t for t in tags.split(",") if t)


def build_poi_feature_table(pois: pd.DataFrame) -> pd.DataFrame:
    """Return a poi_id-indexed table of content features plus a parsed tag set."""
    df = pois.copy()
    df["tag_set"] = df["tags"].apply(_parse_tags)
    df["is_touristy"] = df["tag_set"].apply(lambda t: "touristy" in t)

    max_review = max(df["review_count"].max(), 1)
    max_popularity = max(df["popularity"].max(), 1)
    df["log_review_count_norm"] = np.log1p(df["review_count"]) / np.log1p(max_review)
    log_popularity = np.log1p(df["popularity"])
    df["log_popularity_norm"] = log_popularity / np.log1p(max_popularity)
    # Inverse popularity: high for long-tail/local POIs, low for famous ones.
    df["novelty_score"] = 1.0 - df["log_popularity_norm"]

    df["rating_norm"] = (df["rating"] - 2.5) / 2.5
    df["price_level_norm"] = (df["price_level"] - 1) / 3.0
    df["duration_norm"] = df["expected_duration_min"] / 180.0

    for cat in CATEGORIES:
        df[f"cat_{cat}"] = (df["category"] == cat).astype(int)

    df["reservation_required"] = df["reservation_required"].astype(int)
    df["wheelchair_accessible"] = df["wheelchair_accessible"].astype(int)
    df["requires_car"] = df["requires_car"].astype(int)

    return df.set_index("poi_id", drop=False)


POI_CONTENT_FEATURE_COLUMNS = (
    ["log_review_count_norm", "log_popularity_norm", "novelty_score", "rating_norm",
     "price_level_norm", "duration_norm", "reservation_required", "wheelchair_accessible", "requires_car"]
    + [f"cat_{cat}" for cat in CATEGORIES]
)
