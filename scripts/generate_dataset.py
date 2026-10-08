"""Generate the synthetic POI / traveler / interaction dataset and write it to data/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_ranker.data.generate import generate_interactions, generate_pois, generate_travelers
from poi_ranker.utils.seed import SEED, set_global_seed

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    set_global_seed(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    pois = generate_pois(n_per_destination=65, seed=SEED)
    travelers = generate_travelers(n_travelers=25, seed=SEED)
    interactions = generate_interactions(pois, travelers, seed=SEED)

    pois.to_csv(DATA_DIR / "pois.csv", index=False)
    travelers.to_csv(DATA_DIR / "travelers.csv", index=False)
    interactions.to_csv(DATA_DIR / "interactions.csv", index=False)

    print(f"Wrote {len(pois)} POIs, {len(travelers)} travelers, {len(interactions)} interactions to {DATA_DIR}")


if __name__ == "__main__":
    main()
