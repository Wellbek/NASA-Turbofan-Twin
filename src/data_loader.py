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
        """Calculate RUL for test split using truth file"""
        rul_truth = pd.read_csv(self.data_dir / f'RUL_{dataset_name}.txt', header=None)
        rul_truth.columns = ['truth_rul']

        max_cycles = df.groupby('engine_id')['time_cycles'].max().reset_index()
        max_cycles['truth_rul'] = rul_truth['truth_rul']
        
        df = df.merge(max_cycles, on='engine_id', how='left')
        df['RUL'] = df['truth_rul'] + (df['time_cycles'].max() - df['time_cycles'])
        return df.drop('truth_rul', axis=1)

    def filter_sensors(self, df, corr_threshold=0.95, variance_threshold=1e-3):
        """
        Remove low-variance and highly correlated sensor columns from a DataFrame.

        Args:
            df (pandas.DataFrame): Input DataFrame with sensor columns.
            corr_threshold (float): Correlation threshold to drop highly correlated sensors.
            variance_threshold (float): Variance threshold to drop nearly constant sensors.

        Returns:
            filtered_df (pandas.DataFrame): DataFrame with filtered sensors.
            dropped_sensors (list): List of sensor names that were removed.
        """

        sensors = [c for c in df.columns if 'sensor' in c]

        # Drop low-variance sensors (always constant values)
        variances = df[sensors].var()
        low_var = variances[variances < variance_threshold].index.tolist()

        # Drop highly correlated sensors
        corr_matrix = df[sensors].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]

        drop_list = set(low_var + high_corr)
        print(f"Dropping {len(drop_list)} sensors: {drop_list}")
        return df.drop(columns=drop_list), list(drop_list)




