"""Synthetic dataset generator for POIs, travelers/trips, and historical interactions.

Interactions are sampled proportional to an affinity score (interest/tag overlap,
budget fit, touristy-preference alignment) rather than pure popularity, so the
dataset actually contains a personalization signal for the ranker to learn from.
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from poi_ranker.data.taxonomy import (
    BUDGET_LEVELS,
    BUDGET_TO_PRICE,
    CATEGORY_TAXONOMY,
    DESTINATIONS,
    EXPLICIT_PREFERENCES,
    INTERACTION_TYPES,
    INTERESTS,
    MOBILITY_TYPES,
    PARTY_TYPES,
    PRICE_LEVELS,
)


def _sample_tags(rng: random.Random, category_tags: list[str], extra_pool: list[str], touristy_bias: float) -> list[str]:
    n_base = rng.randint(2, min(3, len(category_tags)))
    tags = set(rng.sample(category_tags, n_base))
    if rng.random() < touristy_bias:
        tags.add("touristy")
    elif rng.random() < 0.6:
        tags.add("local_experience")
    # occasionally sprinkle an unrelated interest tag so overlap isn't trivially perfect
    if rng.random() < 0.25:
        tags.add(rng.choice(extra_pool))
    return sorted(tags)


def generate_pois(n_per_destination: int = 65, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    rows = []
    poi_counter = 1
    for destination, geo in DESTINATIONS.items():
        for _ in range(n_per_destination):
            category = rng.choice(list(CATEGORY_TAXONOMY.keys()))
            meta = CATEGORY_TAXONOMY[category]
            subcategory = rng.choice(meta["subcategories"])
            tags = _sample_tags(rng, meta["tags"], INTERESTS, meta["touristy_bias"])

            lat = geo["lat"] + np_rng.normal(0, geo["spread"])
            lon = geo["lon"] + np_rng.normal(0, geo["spread"])

            # Power-law-ish popularity: most POIs are long-tail, a few are famous.
            popularity = float(np.clip(np_rng.pareto(2.0) * 10, 1, 500))
            is_famous = popularity > 150
            price_level = rng.choice(PRICE_LEVELS)
            rating = float(np.clip(np_rng.normal(4.2 if is_famous else 4.0, 0.35), 2.5, 5.0))
            review_count = int(np.clip(popularity * rng.uniform(2, 6), 5, 5000))

            poi_id = f"P{poi_counter:04d}"
            poi_counter += 1
            rows.append(
                {
                    "poi_id": poi_id,
                    "name": f"{subcategory} #{poi_id[1:]}",
                    "category": category,
                    "subcategory": subcategory,
                    "destination": destination,
                    "latitude": round(lat, 5),
                    "longitude": round(lon, 5),
                    "description": f"A {subcategory.lower()} in {destination} known for {', '.join(tags[:2])}.",
                    "tags": ",".join(tags),
                    "price_level": price_level,
                    "rating": round(rating, 2),
                    "review_count": review_count,
                    "popularity": round(popularity, 2),
                    "expected_duration_min": rng.choice([30, 45, 60, 90, 120, 180]),
                    "open_days": "Mon-Sun" if rng.random() < 0.7 else rng.choice(["Tue-Sun", "Wed-Mon", "Thu-Tue"]),
                    "open_hour": rng.choice([8, 9, 10, 11]),
                    "close_hour": rng.choice([17, 18, 20, 22, 23]),
                    "reservation_required": rng.random() < (0.4 if category == "Restaurant" else 0.1),
                    "wheelchair_accessible": rng.random() < 0.75,
                    "requires_car": category == "Nature/Outdoor" and rng.random() < 0.3,
                }
            )
    return pd.DataFrame(rows)


def generate_travelers(n_travelers: int = 25, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed + 1)
    destinations = list(DESTINATIONS.keys())
    rows = []
    base_date = dt.date(2026, 3, 1)
    for i in range(1, n_travelers + 1):
        traveler_id = f"U{i:03d}"
        destination = destinations[(i - 1) % len(destinations)]
        n_interests = rng.randint(2, 4)
        interests = rng.sample(INTERESTS, n_interests)
        trip_duration = rng.choice([2, 3, 4, 5, 7])
        start = base_date + dt.timedelta(days=rng.randint(0, 200))
        end = start + dt.timedelta(days=trip_duration)
        rows.append(
            {
                "traveler_id": traveler_id,
                "destination": destination,
                "travel_start": start.isoformat(),
                "travel_end": end.isoformat(),
                "trip_duration": trip_duration,
                "interests": ",".join(interests),
                "budget": rng.choice(BUDGET_LEVELS),
                "party_type": rng.choice(PARTY_TYPES),
                "mobility": rng.choice(MOBILITY_TYPES),
                "explicit_preference": rng.choice(EXPLICIT_PREFERENCES),
            }
        )
    return pd.DataFrame(rows)


def _affinity(traveler: pd.Series, poi: pd.Series) -> float:
    """How much a traveler should plausibly like a POI; drives interaction sampling."""
    interests = set(traveler["interests"].split(","))
    tags = set(poi["tags"].split(","))
    overlap = len(interests & tags) / max(1, len(interests))

    budget_price = BUDGET_TO_PRICE[traveler["budget"]]
    budget_fit = 1.0 - min(1.0, abs(budget_price - poi["price_level"]) / 3.0)

    touristy = "touristy" in tags
    if traveler["explicit_preference"] == "prefer_less_touristy":
        pref_fit = 0.3 if touristy else 1.0
    elif traveler["explicit_preference"] == "prefer_famous_landmarks":
        pref_fit = 1.0 if touristy else 0.6
    else:
        pref_fit = 0.8

    mobility_fit = 0.5 if (traveler["mobility"] != "car" and poi["requires_car"]) else 1.0

    score = 0.45 * overlap + 0.2 * budget_fit + 0.25 * pref_fit + 0.1 * mobility_fit
    return float(np.clip(score, 0.0, 1.0))


def generate_interactions(pois: pd.DataFrame, travelers: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed + 2)
    np_rng = np.random.default_rng(seed + 2)
    rows = []

    for _, traveler in travelers.iterrows():
        dest_pois = pois[pois["destination"] == traveler["destination"]]
        affinities = dest_pois.apply(lambda poi: _affinity(traveler, poi), axis=1)

        n_interactions = rng.randint(40, 90)
        weights = (affinities.values + 0.05)  # small floor so low-affinity POIs can still appear
        weights = weights / weights.sum()
        sampled_idx = np_rng.choice(dest_pois.index, size=n_interactions, replace=True, p=weights)

        start = dt.date.fromisoformat(traveler["travel_start"])
        lookback_start = start - dt.timedelta(days=365)

        for idx in sampled_idx:
            poi = dest_pois.loc[idx]
            affinity = affinities.loc[idx]
            # Higher affinity -> more likely a strong-positive interaction; low affinity -> dismiss more likely.
            if affinity > 0.7:
                interaction = rng.choices(
                    ["booking", "visit", "save", "share", "navigate", "click", "view"],
                    weights=[0.2, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05],
                )[0]
            elif affinity > 0.4:
                interaction = rng.choices(
                    ["visit", "save", "navigate", "click", "view", "dismiss"],
                    weights=[0.15, 0.2, 0.15, 0.2, 0.2, 0.1],
                )[0]
            else:
                interaction = rng.choices(
                    ["view", "click", "dismiss"],
                    weights=[0.4, 0.2, 0.4],
                )[0]

            days_offset = rng.randint(0, (start - lookback_start).days)
            timestamp = lookback_start + dt.timedelta(days=days_offset)

            rows.append(
                {
                    "traveler_id": traveler["traveler_id"],
                    "poi_id": poi["poi_id"],
                    "interaction": interaction,
                    "timestamp": timestamp.isoformat(),
                }
            )
    return pd.DataFrame(rows)
