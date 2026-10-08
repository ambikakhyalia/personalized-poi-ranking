"""Train the LightGBM LambdaMART (learning-to-rank) model on the assembled dataset."""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from poi_ranker.utils.seed import SEED


def train_ranker(X: pd.DataFrame, y: pd.Series, group_sizes: list[int], seed: int = SEED) -> lgb.LGBMRanker:
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        random_state=seed,
        verbosity=-1,
    )
    ranker.fit(X, y, group=group_sizes)
    return ranker
