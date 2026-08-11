"""Split tests.

The split is the thing that decides whether any reported number means anything,
and it silently disagreed between two notebooks for months. So it gets pinned.
"""

import pytest

from splits import load_splits, make_splits, save_splits, split_frame


ENGINES = list(range(1, 101))


def test_partitions_are_disjoint_and_complete():
    s = make_splits(ENGINES)
    assert set(s['train']) | set(s['val']) | set(s['test']) == set(ENGINES)
    assert not set(s['train']) & set(s['val'])
    assert not set(s['train']) & set(s['test'])
    assert not set(s['val']) & set(s['test'])


def test_proportions():
    s = make_splits(ENGINES)
    assert (len(s['train']), len(s['val']), len(s['test'])) == (70, 15, 15)


def test_same_seed_gives_the_same_engines():
    assert make_splits(ENGINES, seed=7) == make_splits(ENGINES, seed=7)


def test_different_seed_gives_different_engines():
    assert make_splits(ENGINES, seed=1)['test'] != make_splits(ENGINES, seed=2)['test']


def test_split_does_not_depend_on_input_order():
    forward = make_splits(ENGINES)
    backward = make_splits(list(reversed(ENGINES)))
    assert forward == backward


def test_split_does_not_depend_on_the_global_random_state():
    """The old code seeded np.random globally, so anything that drew a random
    number earlier in a notebook could shift the split."""
    import numpy as np
    np.random.seed(1)
    first = make_splits(ENGINES)
    np.random.seed(999)
    np.random.random(50)
    assert make_splits(ENGINES) == first


def test_roundtrip_through_disk(tmp_path):
    s = make_splits(ENGINES)
    path = save_splits(s, tmp_path / 'splits.json')
    assert load_splits(path) == s


def test_load_missing_splits_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='src.train'):
        load_splits(tmp_path / 'nope.json')


def test_split_frame_keeps_engines_whole(train_fd001):
    s = make_splits(train_fd001['engine_id'].unique())
    train, val, test = split_frame(train_fd001, s)

    assert set(train['engine_id'].unique()) == set(s['train'])
    assert set(val['engine_id'].unique()) == set(s['val'])
    assert set(test['engine_id'].unique()) == set(s['test'])
    assert len(train) + len(val) + len(test) == len(train_fd001)


def test_no_engine_appears_on_both_sides(train_fd001):
    """The reason we split by engine and not by row: rows from one engine share
    rolling windows and a degradation curve, so a row split leaks."""
    s = make_splits(train_fd001['engine_id'].unique())
    train, val, test = split_frame(train_fd001, s)
    assert not set(train['engine_id']) & set(test['engine_id'])
    assert not set(val['engine_id']) & set(test['engine_id'])
