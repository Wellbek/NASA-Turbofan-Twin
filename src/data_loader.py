"""
CMAPSS Data Loader

Data Wrangling

Typical usage example:
    # TO-DO
"""

from pathlib import Path

import pandas as pd


class CMAPSSLoader:
    """CMAPSS Data Loader"""
    def __init__(self, data_dir='../data/bronze/cmapss'):
        self.data_dir = Path(data_dir)
        self.columns = (
            ['engine_id', 'time_cycles']
            + [f'operational_setting_{i}' for i in range(1,4)]
            + [f'sensor_{i}' for i in range(1, 27)]
        )

    def load_dataset(self, dataset_name, split = 'train'):
        """
        Load CMAPSS datasets and calcluate RUL (Remaining Useful Life)
        
        Args:
            dataset_name (str): FD001, FD002, FD003, FD004
            split (str): 'train' or 'test'

        Returns:
            pd.DataFrame: Training data with RUL calcluated
        """
        file_path = self.data_dir / f'{split}_{dataset_name}.txt'

        if not file_path.exists():
            raise FileNotFoundError(f'Dataset not found: {file_path}')

        df = pd.read_csv(file_path, sep=' ', header=None, names=self.columns)
        df = df.dropna(axis=1, how='all')

        if split == 'train':
            df = self._calculate_rul(df)
        else:
            df = self._calculate_test_rul(df, dataset_name)

        print(f'Loaded {dataset_name}: {df.shape[0]} records, {df["engine_id"].nunique()} engines')

        return df

    def _calculate_rul(self, df):
        """Calculate Remaining Useful Life for each engine"""
        max_cycles = df.groupby('engine_id')['time_cycles'].max().reset_index()
        max_cycles.columns = ['engine_id', 'max_cycles']

        df = df.merge(max_cycles, on='engine_id', how='left')
        df['RUL'] = df['max_cycles'] - df['time_cycles']
        df = df.drop('max_cycles', axis=1)

        return df

    def _calculate_test_rul(self, df, dataset_name):
        """Calculate RUL for test split using truth file.

        Test engines are truncated some time before failure. The truth file
        gives the RUL remaining at each engine's *last* recorded cycle, so the
        RUL at any earlier cycle is that value plus the cycles still to come
        for that same engine.
        """
        rul_truth = pd.read_csv(self.data_dir / f'RUL_{dataset_name}.txt',
                                sep=r'\s+', header=None, usecols=[0],
                                names=['truth_rul'])

        last_cycle = (df.groupby('engine_id')['time_cycles'].max()
                        .rename('last_cycle').reset_index())

        if len(rul_truth) != len(last_cycle):
            raise ValueError(
                f'RUL truth file has {len(rul_truth)} rows but {dataset_name} '
                f'test split has {len(last_cycle)} engines')

        # The truth file is ordered by engine_id, one row per engine.
        last_cycle = last_cycle.sort_values('engine_id').reset_index(drop=True)
        last_cycle['truth_rul'] = rul_truth['truth_rul'].values

        df = df.merge(last_cycle, on='engine_id', how='left')
        df['RUL'] = df['truth_rul'] + (df['last_cycle'] - df['time_cycles'])
        return df.drop(columns=['truth_rul', 'last_cycle'])




