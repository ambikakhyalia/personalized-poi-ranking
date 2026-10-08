"""End-to-end evaluation: held-out interactions as ground truth, no train/test leakage.

`train_interactions` is what profiles + the ranker were built from; `test_interactions`
supplies the held-out positive engagements used as ground truth for Precision/Recall/NDCG.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from poi_ranker.evaluation.metrics import (
    long_tail_discovery,
    ndcg_at_k,
    personalization,
    precision_at_k,
    recall_at_k,
)
from poi_ranker.pipeline import recommend_for_traveler

POSITIVE_INTERACTION_TYPES = {"visit", "booking", "save", "share"}


def evaluate(
    poi_table: pd.DataFrame,
    travelers: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    ranker: lgb.LGBMRanker,
    k_values: tuple[int, ...] = (5, 10),
) -> dict:
    test_positive = test_interactions[test_interactions["interaction"].isin(POSITIVE_INTERACTION_TYPES)]

    topk_by_traveler: dict[str, list[str]] = {}
    precision_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}
    n_travelers_with_ground_truth = 0

    for _, traveler_row in travelers.iterrows():
        recs = recommend_for_traveler(traveler_row, poi_table, train_interactions, ranker, top_k=max(k_values))
        ranked_ids = [r["poi_id"] for r in recs]
        topk_by_traveler[traveler_row["traveler_id"]] = ranked_ids

        relevant = set(test_positive[test_positive["traveler_id"] == traveler_row["traveler_id"]]["poi_id"])
        if not relevant:
            continue
        relevance_map = {poi_id: 1.0 for poi_id in relevant}
        n_travelers_with_ground_truth += 1
        for k in k_values:
            precision_scores[k].append(precision_at_k(ranked_ids, relevant, k))
            recall_scores[k].append(recall_at_k(ranked_ids, relevant, k))
            ndcg_scores[k].append(ndcg_at_k(ranked_ids, relevance_map, k))

    report: dict = {}
    for k in k_values:
        report[f"precision@{k}"] = float(np.mean(precision_scores[k])) if precision_scores[k] else 0.0
        report[f"recall@{k}"] = float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0
        report[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0

    report["personalization"] = personalization(topk_by_traveler)
    report["long_tail_discovery"] = long_tail_discovery(topk_by_traveler, poi_table)
    report["n_travelers_with_ground_truth"] = n_travelers_with_ground_truth
    return report
