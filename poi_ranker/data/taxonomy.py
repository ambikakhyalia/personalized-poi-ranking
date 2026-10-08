"""Shared vocabulary for POIs and travelers so data-gen and feature code agree on categories.

Kept in one place because candidate generation, feature engineering, and the synthetic
data generator all need the exact same category/tag/interest vocabulary to be consistent.
"""
from __future__ import annotations

# Each destination gets its own centroid so geo-distance features are meaningful.
DESTINATIONS = {
    "Lisbon": {"lat": 38.7223, "lon": -9.1393, "spread": 0.06},
    "Kyoto": {"lat": 35.0116, "lon": 135.7681, "spread": 0.05},
    "Austin": {"lat": 30.2672, "lon": -97.7431, "spread": 0.07},
}

# category -> (subcategories, plausible tag pool, baseline "touristy-ness" 0-1)
CATEGORY_TAXONOMY = {
    "Museum": {
        "subcategories": ["Art Museum", "History Museum", "Science Museum"],
        "tags": ["art", "history", "architecture", "indoor", "educational"],
        "touristy_bias": 0.6,
    },
    "Historical Site": {
        "subcategories": ["Monument", "Ruins", "Old Town"],
        "tags": ["history", "architecture", "photography", "outdoor"],
        "touristy_bias": 0.75,
    },
    "Landmark": {
        "subcategories": ["Famous Landmark", "Viewpoint"],
        "tags": ["photography", "architecture", "history", "crowded"],
        "touristy_bias": 0.9,
    },
    "Restaurant": {
        "subcategories": ["Fine Dining", "Casual Dining", "Street Food"],
        "tags": ["local_food", "dining", "local_experience"],
        "touristy_bias": 0.3,
    },
    "Local Market": {
        "subcategories": ["Food Market", "Craft Market"],
        "tags": ["local_food", "shopping", "local_experience", "neighborhood"],
        "touristy_bias": 0.2,
    },
    "Neighborhood Experience": {
        "subcategories": ["Walking Area", "Local Hangout"],
        "tags": ["local_experience", "neighborhood", "relaxation"],
        "touristy_bias": 0.15,
    },
    "Nature/Outdoor": {
        "subcategories": ["Park", "Garden", "Hiking Trail"],
        "tags": ["nature", "outdoor", "relaxation", "family_friendly"],
        "touristy_bias": 0.35,
    },
    "Shopping": {
        "subcategories": ["Mall", "Boutique Street"],
        "tags": ["shopping", "local_experience"],
        "touristy_bias": 0.4,
    },
    "Activity": {
        "subcategories": ["Interactive Experience", "Adventure Activity", "Workshop"],
        "tags": ["adventure", "family_friendly", "interactive"],
        "touristy_bias": 0.45,
    },
    "Event": {
        "subcategories": ["Seasonal Event", "Live Performance"],
        "tags": ["nightlife", "local_experience", "interactive"],
        "touristy_bias": 0.5,
    },
}

CATEGORIES = list(CATEGORY_TAXONOMY.keys())

# Interests travelers can hold; POI tags are drawn from the same universe so
# interest-tag overlap is a meaningful, directly comparable feature.
INTERESTS = [
    "history", "architecture", "art", "local_food", "local_experience",
    "nature", "shopping", "nightlife", "family_friendly", "adventure",
    "relaxation", "photography",
]

BUDGET_LEVELS = ["low", "medium", "high"]
PARTY_TYPES = ["solo", "couple", "family", "friends"]
MOBILITY_TYPES = ["walking", "public_transport", "car"]
EXPLICIT_PREFERENCES = ["prefer_less_touristy", "prefer_famous_landmarks", "neutral"]

PRICE_LEVELS = [1, 2, 3, 4]
# Ordinal mapping so budget and POI price_level are directly comparable.
BUDGET_TO_PRICE = {"low": 1, "medium": 2, "high": 3}

INTERACTION_TYPES = ["view", "click", "save", "share", "navigate", "visit", "booking", "dismiss"]

# Interaction -> learning signal weight. Dismiss is an explicit negative signal.
INTERACTION_WEIGHTS = {
    "view": 0.1,
    "click": 0.2,
    "save": 0.4,
    "share": 0.4,
    "navigate": 0.5,
    "visit": 0.8,
    "booking": 1.0,
    "dismiss": -0.5,
}
