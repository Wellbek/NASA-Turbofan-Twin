"""Engine level train/validation/test splits.

CMAPSS rows are cycles, not independent samples: two rows from the same engine
share rolling windows, lags and a degradation trajectory. Splitting on rows
would put the same engine on both sides and leak almost perfectly. So the split
is always by engine.

The split is computed once and written to disk. Everything downstream reads the
artifact instead of recomputing it, because two notebooks recomputing "the same"
split with different code is exactly how the LSTM ended up being evaluated on
engines the tree models had trained on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_FILE = _BASE_DIR / 'data' / 'models' / 'cmapss' / 'splits.json'

DEFAULT_SEED = 42
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15


def make_splits(engine_ids, seed: int = DEFAULT_SEED,
                train_frac: float = DEFAULT_TRAIN_FRAC,
                val_frac: float = DEFAULT_VAL_FRAC) -> dict:
    """Partition engine IDs into train/val/test.

    Uses an explicit ``Generator`` rather than the legacy global ``np.random``
    seed so the result does not depend on whatever else touched the global
    random state earlier in a notebook.
    """
    ids = np.array(sorted(engine_ids))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)

    n = len(shuffled)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    return {
        'seed': seed,
        'train': sorted(int(i) for i in shuffled[:train_end]),
        'val': sorted(int(i) for i in shuffled[train_end:val_end]),
        'test': sorted(int(i) for i in shuffled[val_end:]),
    }


def save_splits(splits: dict, path: Path = SPLITS_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(splits, f, indent=2)
    return path


def load_splits(path: Path = SPLITS_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f'No split artifact at {path}. Run `python -m src.train` first.')
    with open(path) as f:
        return json.load(f)


def split_frame(df, splits: dict, unit_col: str = 'engine_id'):
    """Slice a cycle level frame into (train, val, test) by engine."""
    return tuple(df[df[unit_col].isin(splits[name])].copy()
                 for name in ('train', 'val', 'test'))
