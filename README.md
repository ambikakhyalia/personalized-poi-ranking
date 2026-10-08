# Personalized POI Intelligence & Ranking System

A machine-learning pipeline that ranks Points of Interest (POIs) for a specific
traveler and trip, using synthetic data. See [docs/TECHNICAL_DOC.md](docs/TECHNICAL_DOC.md)
for the full design rationale.

## Requirements

- Python 3.9+
- macOS/Linux (tested on macOS)
- No API keys, accounts, or external services are required.

## Environment Configuration

This project currently has **no `.env` file and requires no environment
variables**. It uses a seeded synthetic dataset and runs locally. `.env` is listed
in `.gitignore` so a future local-only configuration file will not be committed;
there are no current values to create or copy. `python-dotenv` is not required.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

### macOS note: LightGBM + libomp

LightGBM requires the OpenMP runtime (`libomp`). If you have Homebrew:

```bash
brew install libomp
```

If you don't have Homebrew, `scikit-learn`'s wheel already bundles a `libomp.dylib`
you can point LightGBM at without installing anything extra:

```bash
mkdir -p /tmp/libomp_poi_ranker/lib
cp .venv/lib/python3.*/site-packages/sklearn/.dylibs/libomp.dylib /tmp/libomp_poi_ranker/lib/
install_name_tool -add_rpath /tmp/libomp_poi_ranker/lib \
  .venv/lib/python3.*/site-packages/lightgbm/lib/lib_lightgbm.dylib
```

Verify with: `python -c "import lightgbm; print(lightgbm.__version__)"`

If you use the full-path Python commands below, verify with
`./.venv/bin/python -c "import lightgbm; print(lightgbm.__version__)"`.

## Run

```bash
# 1. Generate the synthetic dataset (data/pois.csv, travelers.csv, interactions.csv)
./.venv/bin/python scripts/generate_dataset.py

# 2. Train the ranker and print the evaluation report (also saved to results/evaluation_report.json)
./.venv/bin/python scripts/run_pipeline.py

# 3. Produce the 3 required demo scenarios (saved to results/scenario_*.json)
./.venv/bin/python scripts/run_scenarios.py

# 4. Run the test suite
./.venv/bin/python -m pytest tests/ -v
```

Run these commands from the repository root, in order. Dataset generation uses
the fixed seed in `poi_ranker/utils/seed.py`; it recreates the CSV inputs. The
pipeline trains and saves `models/ranker.joblib` and
`results/evaluation_report.json`. The scenario script then writes one JSON result
for each of the three profiles under `results/`. The test command checks the
feature builders, candidate generation, scoring, and evaluation metrics.

To reproduce from a fresh clone, clone the repository, `cd` into its root, follow
Setup (and the macOS `libomp` note if needed), then run the four commands above.
No pre-generated model, `.env`, or network service is needed. For this workspace's
path containing a colon, invoking `./.venv/bin/python` directly also avoids zsh
`PATH` issues that can occur after activating the environment.

## Project Structure

```
poi_ranker/
  data/            synthetic data generation + shared taxonomy/vocabulary
  features/        POI representation, traveler representation, cross features
  candidates/       candidate generation (rule-based + long-tail guarantee)
  ranking/          label construction, training-set assembly, LightGBM training/inference
  scoring/          context/practical compatibility + final utility combiner
  explainability/   rule-based recommendation explanations
  evaluation/       ranking + recommendation-quality metrics
  pipeline.py       end-to-end orchestration for one traveler
scripts/            CLI entry points (generate data, train+evaluate, run scenarios)
data/               generated synthetic dataset (CSV)
models/             trained ranker artifact
results/            evaluation report + scenario outputs (JSON)
docs/TECHNICAL_DOC.md  full technical write-up
tests/              pytest suite
```

## Output Schema

Each recommendation returned by `pipeline.recommend_for_traveler` looks like:

```json
{
  "poi_id": "P0053",
  "score": 0.897,
  "preference_weight": 0.91,
  "context_compatibility": 0.96,
  "confidence": 0.72,
  "name": "Famous Landmark #0053",
  "category": "Landmark",
  "reasons": ["Strong match with traveler's stated interests", "Within budget"]
}
```
