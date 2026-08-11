"""Shared fixtures.

Tests run against the raw bronze data that ships with the repo, so they need
no trained models and no generated silver/gold layer.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

BRONZE_DIR = REPO_ROOT / 'data' / 'bronze' / 'cmapss'


@pytest.fixture(scope='session')
def bronze_dir():
    return BRONZE_DIR


@pytest.fixture(scope='session')
def loader(bronze_dir):
    from data_loader import CMAPSSLoader
    return CMAPSSLoader(bronze_dir)


@pytest.fixture(scope='session')
def train_fd001(loader):
    return loader.load_dataset('FD001', split='train')


@pytest.fixture(scope='session')
def test_fd001(loader):
    return loader.load_dataset('FD001', split='test')


@pytest.fixture
def small_slice(train_fd001):
    """First 10 engines, enough to exercise the feature pipeline quickly."""
    engines = sorted(train_fd001['engine_id'].unique())[:10]
    return train_fd001[train_fd001['engine_id'].isin(engines)].copy()
