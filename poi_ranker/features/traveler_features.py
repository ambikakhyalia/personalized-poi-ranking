"""Traveler representation: explicit stated preferences kept separate from implicit
behavioral signals derived from history, per the assignment's explicit/implicit distinction.

The two signal groups are NOT hand-blended here — they are exposed as separate fields
and combined later, either as separate ranking-model features (so the model learns how
to weigh them) or explicitly in the scoring/explainability stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from poi_ranker.data.taxonomy import BUDGET_TO_PRICE, INTERACTION_WEIGHTS


@dataclass
class TravelerProfile:
    traveler_id: str
    destination: str
    travel_start: str
    travel_end: str
    trip_duration: int
    budget: str
    party_type: str
    mobility: str
    explicit_preference: str
    interests: set[str]

    # Implicit / behavioral signals derived from history. Empty/None => cold start.
    implicit_tag_affinity: dict[str, float] = field(default_factory=dict)
    implicit_category_affinity: dict[str, float] = field(default_factory=dict)
    avg_price_level_interacted: float | None = None
    avg_touristy_affinity: float | None = None
    visited_poi_ids: set[str] = field(default_factory=set)
    visited_tag_union: set[str] = field(default_factory=set)
    n_interactions: int = 0

    @property
    def budget_price(self) -> int:
        return BUDGET_TO_PRICE[self.budget]

    @property
    def is_cold_start(self) -> bool:
        return self.n_interactions == 0


def build_traveler_profile(
    traveler_row: pd.Series, interactions: pd.DataFrame, poi_table: pd.DataFrame
) -> TravelerProfile:
    """Build explicit + implicit profile. `interactions` is the FULL interaction log;
    it is filtered here to this traveler so callers don't have to pre-filter."""
    traveler_interactions = interactions[interactions["traveler_id"] == traveler_row["traveler_id"]]

    tag_affinity: dict[str, float] = {}
    category_affinity: dict[str, float] = {}
    price_weighted_sum, touristy_weighted_sum, weight_total = 0.0, 0.0, 0.0
    visited_poi_ids: set[str] = set()
    visited_tag_union: set[str] = set()

    for _, interaction_row in traveler_interactions.iterrows():
        poi_id = interaction_row["poi_id"]
        if poi_id not in poi_table.index:
            continue
        poi = poi_table.loc[poi_id]
        signal = INTERACTION_WEIGHTS.get(interaction_row["interaction"], 0.0)

        for tag in poi["tag_set"]:
            tag_affinity[tag] = tag_affinity.get(tag, 0.0) + signal
        category_affinity[poi["category"]] = category_affinity.get(poi["category"], 0.0) + signal

        if signal > 0:
            price_weighted_sum += signal * poi["price_level"]
            touristy_weighted_sum += signal * (1.0 if poi["is_touristy"] else 0.0)
            weight_total += signal
        if interaction_row["interaction"] in ("visit", "booking"):
            visited_poi_ids.add(poi_id)
            visited_tag_union |= poi["tag_set"]

    avg_price = price_weighted_sum / weight_total if weight_total > 0 else None
    avg_touristy = touristy_weighted_sum / weight_total if weight_total > 0 else None

    return TravelerProfile(
        traveler_id=traveler_row["traveler_id"],
        destination=traveler_row["destination"],
        travel_start=traveler_row["travel_start"],
        travel_end=traveler_row["travel_end"],
        trip_duration=int(traveler_row["trip_duration"]),
        budget=traveler_row["budget"],
        party_type=traveler_row["party_type"],
        mobility=traveler_row["mobility"],
        explicit_preference=traveler_row["explicit_preference"],
        interests=set(traveler_row["interests"].split(",")),
        implicit_tag_affinity=tag_affinity,
        implicit_category_affinity=category_affinity,
        avg_price_level_interacted=avg_price,
        avg_touristy_affinity=avg_touristy,
        visited_poi_ids=visited_poi_ids,
        visited_tag_union=visited_tag_union,
        n_interactions=len(traveler_interactions),
    )
