"""Assemble a LightGBM LambdaMART training set: one query group per traveler.

Candidates are generated with the same function used at inference time, so the
training distribution matches what the model will actually see when serving.
"""
from __future__ import annotations

import pandas as pd

from poi_ranker.candidates.candidate_generation import generate_candidates
from poi_ranker.features.cross_features import CROSS_FEATURE_COLUMNS
from poi_ranker.features.poi_features import POI_CONTENT_FEATURE_COLUMNS
from poi_ranker.features.traveler_features import build_traveler_profile
from poi_ranker.ranking.labels import compute_interaction_labels

FEATURE_COLUMNS = POI_CONTENT_FEATURE_COLUMNS + CROSS_FEATURE_COLUMNS


def build_ltr_dataset(
    poi_table: pd.DataFrame,
    travelers: pd.DataFrame,
    train_interactions: pd.DataFrame,
    cold_start_augmentation: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[int], pd.DataFrame]:
    """cold_start_augmentation adds a second query group per traveler built with an
    EMPTY interaction history (same ground-truth labels, zeroed implicit features).
    Without this, the model never sees the all-implicit-features-zero regime during
    training and generalizes poorly to genuinely new travelers at inference time."""
    labels = compute_interaction_labels(train_interactions)
    empty_interactions = train_interactions.iloc[0:0]

    feature_rows, label_values, group_sizes, meta_rows = [], [], [], []
    for _, traveler_row in travelers.iterrows():
        traveler_labels = labels[labels["traveler_id"] == traveler_row["traveler_id"]].set_index("poi_id")["label"]

        interaction_variants = [train_interactions]
        if cold_start_augmentation:
            interaction_variants.append(empty_interactions)

        for interactions_variant in interaction_variants:
            profile = build_traveler_profile(traveler_row, interactions_variant, poi_table)
            candidates = generate_candidates(profile, poi_table)
            if candidates.empty:
                continue

            candidate_labels = candidates["poi_id"].map(traveler_labels).fillna(0).astype(int)
            feature_rows.append(candidates[FEATURE_COLUMNS])
            label_values.append(candidate_labels)
            group_sizes.append(len(candidates))
            meta_rows.append(pd.DataFrame({"traveler_id": traveler_row["traveler_id"], "poi_id": candidates["poi_id"]}))

    X = pd.concat(feature_rows, ignore_index=True)
    y = pd.concat(label_values, ignore_index=True)
    meta = pd.concat(meta_rows, ignore_index=True)
    return X, y, group_sizes, meta
