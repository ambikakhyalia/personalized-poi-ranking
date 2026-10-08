"""Produce the 3 required demo scenarios (Local Experience / History / Family).

These are new, previously-unseen traveler IDs with no interaction history — this
doubles as the cold-start (new traveler) demonstration: personalization here comes
purely from explicit stated preferences via content-based scoring.

Usage: python scripts/run_scenarios.py  (requires models/ranker.joblib from run_pipeline.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from poi_ranker.features.poi_features import build_poi_feature_table
from poi_ranker.pipeline import recommend_for_traveler
from poi_ranker.ranking.model import load_ranker

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "ranker.joblib"
RESULTS_DIR = ROOT / "results"

SCENARIOS = [
    {
        "name": "local_experience",
        "traveler_id": "SCENARIO_LOCAL",
        "destination": "Lisbon",
        "travel_start": "2026-06-03",
        "travel_end": "2026-06-08",
        "trip_duration": 5,
        "interests": "local_food,local_experience",
        "budget": "medium",
        "party_type": "solo",
        "mobility": "public_transport",
        "explicit_preference": "prefer_less_touristy",
    },
    {
        "name": "history",
        "traveler_id": "SCENARIO_HISTORY",
        "destination": "Lisbon",
        "travel_start": "2026-06-03",
        "travel_end": "2026-06-08",
        "trip_duration": 5,
        "interests": "history,architecture,art",
        "budget": "high",
        "party_type": "couple",
        "mobility": "public_transport",
        "explicit_preference": "prefer_famous_landmarks",
    },
    {
        "name": "family",
        "traveler_id": "SCENARIO_FAMILY",
        "destination": "Lisbon",
        "travel_start": "2026-06-03",
        "travel_end": "2026-06-08",
        "trip_duration": 5,
        "interests": "family_friendly,nature,adventure",
        "budget": "medium",
        "party_type": "family",
        "mobility": "car",
        "explicit_preference": "neutral",
    },
]


def main() -> None:
    pois = pd.read_csv(DATA_DIR / "pois.csv")
    empty_interactions = pd.read_csv(DATA_DIR / "interactions.csv").iloc[0:0]
    poi_table = build_poi_feature_table(pois)
    ranker = load_ranker(MODEL_PATH)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        traveler_row = pd.Series(scenario)
        recs = recommend_for_traveler(traveler_row, poi_table, empty_interactions, ranker, top_k=10)

        out_path = RESULTS_DIR / f"scenario_{scenario['name']}.json"
        out_path.write_text(json.dumps(recs, indent=2))

        print(f"\n=== Scenario: {scenario['name']} ===")
        for rec in recs[:5]:
            print(f"  {rec['name']:<30} score={rec['score']:.3f}  reasons={rec['reasons'][:2]}")
        print(f"  Saved full top-10 to {out_path}")


if __name__ == "__main__":
    main()
