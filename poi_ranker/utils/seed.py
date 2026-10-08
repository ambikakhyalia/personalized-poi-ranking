"""Central seed so every generation/training run is reproducible."""
SEED = 42

import random
import numpy as np


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
