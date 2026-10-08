"""Convert raw interaction events into graded relevance labels (0-4) for LambdaMART.

Multiple interactions on the same (traveler, poi) pair are summed via
INTERACTION_WEIGHTS, then bucketed. Dismiss is an explicit negative signal, so a
POI dominated by dismissals settles at label 0 rather than a small positive label.
"""
from __future__ import annotations

import pandas as pd

from poi_ranker.data.taxonomy import INTERACTION_WEIGHTS


def _bucket(raw_score: float) -> int:
    if raw_score <= 0:
        return 0
    if raw_score < 0.3:
        return 1
    if raw_score < 0.6:
        return 2
    if raw_score < 1.0:
        return 3
    return 4


def compute_interaction_labels(interactions: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (traveler_id, poi_id) with raw_score and graded label."""
    weighted = interactions.copy()
    weighted["signal"] = weighted["interaction"].map(INTERACTION_WEIGHTS).fillna(0.0)
    grouped = weighted.groupby(["traveler_id", "poi_id"], as_index=False)["signal"].sum()
    grouped = grouped.rename(columns={"signal": "raw_score"})
    grouped["label"] = grouped["raw_score"].apply(_bucket)
    return grouped
