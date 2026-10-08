"""Ranking + recommendation-quality metrics."""
from __future__ import annotations

import math

import pandas as pd


def precision_at_k(ranked_poi_ids: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_poi_ids[:k]
    return len(set(top_k) & relevant) / k


def recall_at_k(ranked_poi_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = ranked_poi_ids[:k]
    return len(set(top_k) & relevant) / len(relevant)


def _dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_poi_ids: list[str], relevance_map: dict[str, float], k: int) -> float:
    top_k = ranked_poi_ids[:k]
    gains = [relevance_map.get(poi_id, 0.0) for poi_id in top_k]
    ideal_gains = sorted(relevance_map.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_gains)
    if ideal_dcg <= 0:
        return 0.0
    return _dcg(gains) / ideal_dcg


def personalization(topk_by_traveler: dict[str, list[str]]) -> float:
    """1 - average pairwise Jaccard overlap of top-K lists across travelers."""
    traveler_ids = list(topk_by_traveler.keys())
    pairs, total_overlap = 0, 0.0
    for i in range(len(traveler_ids)):
        for j in range(i + 1, len(traveler_ids)):
            a, b = set(topk_by_traveler[traveler_ids[i]]), set(topk_by_traveler[traveler_ids[j]])
            if not a or not b:
                continue
            total_overlap += len(a & b) / len(a | b)
            pairs += 1
    if pairs == 0:
        return 0.0
    return 1.0 - (total_overlap / pairs)


def long_tail_discovery(topk_by_traveler: dict[str, list[str]], poi_table: pd.DataFrame) -> float:
    """Fraction of recommended POIs below their destination's median popularity."""
    median_by_destination = poi_table.groupby("destination")["popularity"].median().to_dict()
    below_median, total = 0, 0
    for poi_ids in topk_by_traveler.values():
        for poi_id in poi_ids:
            if poi_id not in poi_table.index:
                continue
            poi = poi_table.loc[poi_id]
            total += 1
            if poi["popularity"] < median_by_destination.get(poi["destination"], float("inf")):
                below_median += 1
    return below_median / total if total else 0.0
