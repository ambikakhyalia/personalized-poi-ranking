"""Shared fixtures: a tiny synthetic dataset for fast, deterministic tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from poi_ranker.data.generate import generate_interactions, generate_pois, generate_travelers
from poi_ranker.features.poi_features import build_poi_feature_table


@pytest.fixture(scope="session")
def pois() -> pd.DataFrame:
    return generate_pois(n_per_destination=20, seed=1)


@pytest.fixture(scope="session")
def poi_table(pois: pd.DataFrame) -> pd.DataFrame:
    return build_poi_feature_table(pois)


@pytest.fixture(scope="session")
def travelers() -> pd.DataFrame:
    return generate_travelers(n_travelers=6, seed=1)


@pytest.fixture(scope="session")
def interactions(pois: pd.DataFrame, travelers: pd.DataFrame) -> pd.DataFrame:
    return generate_interactions(pois, travelers, seed=1)
