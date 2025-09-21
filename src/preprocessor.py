"""
CMAPPS Data Preprocessor

This module handles data cleaning, filtering and preprocesses based on EDA findings. It aims to remove redundant sensors and applies professional data science best pracices.
"""

import pandas as pd
import numpy as np
from typings import Tuple, List, Dict
from sklearn.preprocessing import StandardScalar

class CMAPSSPreprocessor:

    def __init__(self, corr_threshold=0.95, variance_threshold=1e-3, save_dir=None):
        """
        Preprocessor for CMAPSS datasets.

        Args:
            corr_threshold (float): threshold above which sensors are considered redundant.
            variance_threshold (float): Minimum variance a sensor must have to be considered informative and to be kept.
            save_dir (str, optional): Direectory to save the processed data.
        """
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        self.save_dir = save_dir

    def remove_constant_sensors(self, df):
        """Remove sensors with zero variance (completely constant)."""
        sensors = [c for c in df.columns if 'sensor' in c]
        variances = df[sensors].var()
        to_drop = variances[variances == 0].index.tolist()
        return df.drop(columns=to_drop), to_drop

    def remove_low_variance_sensors(self, df):
        """Remove sensors with variance below threshold."""
        sensors = [c for c in df.columns if 'sensor' in c]
        variances = df[sensors].var()
        to_drop = variances[variances < self.variance_threshold].index.tolist()
        return df.drop(columns=to_drop), to_drop

    def remove_correlated_sensors(self, df):
        """Remove on of each pair of highly correlated sensors."""
        sensors = [c for c in df.columns if 'sensor' in c]
        corr_matrix = df[sensors].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[column] > self.corr_threshold)]
        return df.drop(columns=to_drop), to_drop

    def preprocess(self, df, dataset_name=None):
        """Run full preprocessing pipeline."""
        dropped = {}

        df, dropped['constant'] = self.remove_constant_sensors(df)
        df, dropped['low_variance'] = self.remove_low_variance_sensors(df)
        df, dropped['correlated'] = self.remove_correlated_sensors(df)

        if self.save_dir and dataset_name:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'{dataset_name}_processed.csv')
            df.to_csv(save_path, index=False)
            print(f'processed data saved to {save_path}')

        return df, dropped
