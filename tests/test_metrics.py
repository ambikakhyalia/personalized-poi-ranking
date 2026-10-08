from poi_ranker.evaluation.metrics import (
    long_tail_discovery,
    ndcg_at_k,
    personalization,
    precision_at_k,
    recall_at_k,
)


def test_precision_and_recall_at_k_toy_case() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "z"}

    assert precision_at_k(ranked, relevant, k=4) == 2 / 4
    assert recall_at_k(ranked, relevant, k=4) == 2 / 3


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    ranked = ["a", "b", "c"]
    relevance_map = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(ranked, relevance_map, k=3) == 1.0


def test_ndcg_at_k_reversed_ranking_is_less_than_one() -> None:
    ranked = ["c", "b", "a"]
    relevance_map = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(ranked, relevance_map, k=3) < 1.0


def test_personalization_is_zero_for_identical_lists() -> None:
    topk = {"u1": ["a", "b", "c"], "u2": ["a", "b", "c"]}
    assert personalization(topk) == 0.0


def test_personalization_is_one_for_disjoint_lists() -> None:
    topk = {"u1": ["a", "b"], "u2": ["c", "d"]}
    assert personalization(topk) == 1.0


def test_long_tail_discovery_counts_below_median(poi_table) -> None:
    dest = poi_table["destination"].iloc[0]
    dest_pois = poi_table[poi_table["destination"] == dest].sort_values("popularity")
    least_popular_id = dest_pois.iloc[0]["poi_id"]
    most_popular_id = dest_pois.iloc[-1]["poi_id"]

    topk = {"u1": [least_popular_id, most_popular_id]}
    ratio = long_tail_discovery(topk, poi_table)
    assert 0.0 <= ratio <= 1.0
