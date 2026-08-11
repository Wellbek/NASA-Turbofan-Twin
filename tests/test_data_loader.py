"""Loader tests.

The RUL definition is the one thing in this project that has no model behind
it, it is either right or every downstream number is wrong. So it gets checked
against the truth file NASA ships rather than against a golden value we made up.
"""

import numpy as np
import pandas as pd
import pytest


def test_train_columns_are_named_and_complete(train_fd001):
    expected = (['engine_id', 'time_cycles']
                + [f'operational_setting_{i}' for i in range(1, 4)])
    assert expected == list(train_fd001.columns[:5])
    assert train_fd001.notna().all().all()


def test_train_rul_hits_zero_at_the_last_cycle(train_fd001):
    """Training engines run to failure, so every engine ends at RUL 0."""
    last_rul = train_fd001.groupby('engine_id')['RUL'].min()
    assert (last_rul == 0).all()


def test_train_rul_decreases_by_one_per_cycle(train_fd001):
    engine = train_fd001[train_fd001['engine_id'] == 1].sort_values('time_cycles')
    steps = engine['RUL'].diff().dropna().unique()
    assert steps.tolist() == [-1]


def test_test_split_loads(test_fd001):
    """Regression test: this used to raise KeyError('time_cycles')."""
    assert len(test_fd001) > 0
    assert test_fd001['engine_id'].nunique() == 100


def test_test_rul_matches_the_truth_file(test_fd001, bronze_dir):
    """RUL at each engine's last recorded cycle must equal the NASA truth value.

    The old formula subtracted from the fleet-wide max cycle instead of the
    engine's own last cycle, which inflated RUL for all but the longest engine.
    """
    truth = pd.read_csv(bronze_dir / 'RUL_FD001.txt', sep=r'\s+', header=None,
                        usecols=[0], names=['truth_rul'])['truth_rul']

    last_rows = test_fd001.loc[test_fd001.groupby('engine_id')['time_cycles'].idxmax()]
    last_rows = last_rows.sort_values('engine_id')

    np.testing.assert_array_equal(last_rows['RUL'].values, truth.values)


def test_test_rul_decreases_by_one_per_cycle(test_fd001):
    engine = test_fd001[test_fd001['engine_id'] == 1].sort_values('time_cycles')
    steps = engine['RUL'].diff().dropna().unique()
    assert steps.tolist() == [-1]


def test_test_rul_is_never_negative(test_fd001):
    assert (test_fd001['RUL'] >= 0).all()


def test_missing_dataset_raises(loader):
    with pytest.raises(FileNotFoundError):
        loader.load_dataset('FD999', split='train')
