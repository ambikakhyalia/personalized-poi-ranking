"""Train the ranker on generated data and print the evaluation report.

Usage: python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from poi_ranker.evaluation.evaluate import evaluate
from poi_ranker.features.poi_features import build_poi_feature_table
from poi_ranker.ranking.dataset_builder import build_ltr_dataset
from poi_ranker.ranking.model import save_ranker
from poi_ranker.ranking.split import time_based_split
from poi_ranker.ranking.train import train_ranker
from poi_ranker.utils.seed import SEED, set_global_seed

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "ranker.joblib"
RESULTS_PATH = ROOT / "results" / "evaluation_report.json"


def main() -> None:
    set_global_seed(SEED)

    pois = pd.read_csv(DATA_DIR / "pois.csv")
    travelers = pd.read_csv(DATA_DIR / "travelers.csv")
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")

    poi_table = build_poi_feature_table(pois)
    train_interactions, test_interactions = time_based_split(interactions, test_fraction=0.2)

    print("Building LTR training set...")
    X, y, group_sizes, _meta = build_ltr_dataset(poi_table, travelers, train_interactions)
    print(f"  {len(X)} (traveler, POI) rows across {len(group_sizes)} traveler query groups")

    print("Training LightGBM LambdaMART ranker...")
    ranker = train_ranker(X, y, group_sizes)
    save_ranker(ranker, MODEL_PATH)
    print(f"  Saved model to {MODEL_PATH}")

    print("Evaluating on held-out interactions...")
    report = evaluate(poi_table, travelers, train_interactions, test_interactions, ranker)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Saved report to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
