"""Persist and load the trained ranker so training and inference stay decoupled."""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd


def save_ranker(ranker: lgb.LGBMRanker, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ranker, path)


def load_ranker(path: Path) -> lgb.LGBMRanker:
    return joblib.load(path)


def predict_raw_scores(ranker: lgb.LGBMRanker, X: pd.DataFrame) -> pd.Series:
    return pd.Series(ranker.predict(X), index=X.index)
