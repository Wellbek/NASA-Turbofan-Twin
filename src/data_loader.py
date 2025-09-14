"""
CMAPSS Data Loader

Data Wrangling

Typical usage example:
    # TO-DO
"""

import pandas as pd
import numpy as np
from pathlib import Path


class CMAPSSLoader:
    """CMAPSS Data Loader"""
    def __init__(self, data_dir='../data/raw/cmapss'):
        self.data_dir = Path(data_dir)
        self.columns = (
            ['engine_id', 'time_cycles']
            + [f'operational_setting_{i}' for i in range(1,4)]
            + [f'sensor_{i}' for i in range(1, 27)]
        )

    def load_dataset(self, dataset_name):
        """
        Load CMAPSS training dataset and calcluate RUL (Remaining Useful Life)
        
        Args:
            dataset_name (str): FD001, FD002, FD003, FD001

        Returns:
            pd.DataFrame: Training data with RUL calcluated
        """
        file_path = self.data_dir / f'train_{dataset_name}.txt'

        if not file_path.exists():
            raise FileNotFoundError(f'Dataset not found: {file_path}')

        # Load raw data
        df = pd.read_csv(file_path, sep=' ', header=None, names=self.columns)
        df = df.dropna(axis=1, how='all')

        df = self._calcluate_rul(df)

        print(f'Loaded {dataset_name}: {df.shape[0]} records, {df["engine_id"].nunique()} engines')

        return df

    def _calcluate_rul(self, df):
        """Calculate Remaining Useful Life for each engine"""
        max_cycles = df.groupby('engine_id')['time_cycles'].max().reset_index()
        max_cycles.columns = ['engine_id', 'max_cycles']

        df = df.merge(max_cycles, on='engine_id', how='left')
        df['RUL'] = df['max_cycles'] - df['time_cycles']
        df = df.drop('max_cycles', axis=1)

        return df






