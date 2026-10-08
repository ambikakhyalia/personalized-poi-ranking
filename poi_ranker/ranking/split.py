"""Time-based train/test split of interactions, per traveler.

Single source of truth for splitting so training (implicit profile + labels) and
evaluation (held-out ground truth) never see the same interactions.
"""
from __future__ import annotations

import pandas as pd


def time_based_split(interactions: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    for _, group in interactions.groupby("traveler_id"):
        ordered = group.sort_values("timestamp")
        n_test = max(1, int(len(ordered) * test_fraction))
        train_parts.append(ordered.iloc[:-n_test])
        test_parts.append(ordered.iloc[-n_test:])
    train_df = pd.concat(train_parts).reset_index(drop=True) if train_parts else interactions.iloc[0:0]
    test_df = pd.concat(test_parts).reset_index(drop=True) if test_parts else interactions.iloc[0:0]
    return train_df, test_df
