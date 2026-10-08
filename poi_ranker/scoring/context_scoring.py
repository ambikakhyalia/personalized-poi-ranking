"""Context/practical-compatibility scoring — the single source of truth for these functions.

Used in two places (by design, not duplicated): as inputs to the ranking model's
cross-features, and again standalone in the final utility combiner + explanations.
Each function returns a 0-1 score: 1 = fully compatible, 0 = incompatible.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from poi_ranker.data.taxonomy import DESTINATIONS
from poi_ranker.features.traveler_features import TravelerProfile
from poi_ranker.utils.geo import haversine_km

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
GEO_DECAY_SCALE_KM = 8.0


def budget_compatibility(traveler: TravelerProfile, poi: pd.Series) -> float:
    diff = abs(traveler.budget_price - poi["price_level"])
    return float(1.0 - min(1.0, diff / 3.0))


def mobility_compatibility(traveler: TravelerProfile, poi: pd.Series) -> float:
    if traveler.mobility != "car" and bool(poi["requires_car"]):
        return 0.3
    return 1.0


def _expand_day_range(open_days: str) -> set[str]:
    start, end = open_days.split("-")
    i0, i1 = WEEKDAYS.index(start), WEEKDAYS.index(end)
    if i0 <= i1:
        return set(WEEKDAYS[i0 : i1 + 1])
    return set(WEEKDAYS[i0:] + WEEKDAYS[: i1 + 1])


def _trip_weekdays(travel_start: str, travel_end: str) -> set[str]:
    start = dt.date.fromisoformat(travel_start)
    end = dt.date.fromisoformat(travel_end)
    n_days = min((end - start).days + 1, 7)  # a week fully covers the weekday cycle
    return {WEEKDAYS[(start.weekday() + i) % 7] for i in range(n_days)}


def availability(traveler: TravelerProfile, poi: pd.Series) -> float:
    trip_days = _trip_weekdays(traveler.travel_start, traveler.travel_end)
    open_days = _expand_day_range(poi["open_days"])
    return 1.0 if trip_days & open_days else 0.3


def geo_relevance(traveler: TravelerProfile, poi: pd.Series) -> float:
    centroid = DESTINATIONS[traveler.destination]
    distance_km = haversine_km(centroid["lat"], centroid["lon"], poi["latitude"], poi["longitude"])
    return float(np.exp(-distance_km / GEO_DECAY_SCALE_KM))


def context_compatibility(traveler: TravelerProfile, poi: pd.Series) -> float:
    """Average of the three practical-fit dimensions — used as the utility's context term."""
    return float(
        np.mean(
            [
                budget_compatibility(traveler, poi),
                mobility_compatibility(traveler, poi),
                availability(traveler, poi),
            ]
        )
    )
