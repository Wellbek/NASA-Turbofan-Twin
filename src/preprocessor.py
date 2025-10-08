"""
CMAPPS Data Preprocessor

This module handles data cleaning, filtering and preprocesses based on EDA findings. It aims to remove redundant sensors and applies professional data science best pracices.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CMAPSSPreprocessor:

    def __init__(self, df, corr_threshold=0.95, variance_threshold=1e-3, 
                 bronze_dir='data/bronze/cmapss', silver_dir='data/silver/cmapss', gold_dir='data/gold/cmapss'):
        """
        Preprocessor for CMAPSS datasets.

        Args:
            corr_threshold (float): threshold above which sensors are considered redundant.
            variance_threshold (float): Minimum variance a sensor must have.
            bronze_dir (str): Raw data directory.
            silver_dir (str): Cleaned data directory.
            gold_dir (str): Feature-engineered data directory.
        """
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir
        self.gold_dir = gold_dir
        self.scaler = MinMaxScaler()
        self.sensor_cols = [c for c in df.columns if 'sensor' in c.lower()]

    def identify_invalid(self, df):
        """Filter columns that contain NaNs or invalid values (None, inf, ...).

        These could then consequently be dropped or corrected."""
        invalid_cols = []
        for col in df.columns:
            if df[col].isna().any() or np.isinf(df[col]).any():
                invalid_cols.append(col)
        return df.drop(columns=invalid_cols), invalid_cols

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
        to_drop = [c for c in upper.columns if any(upper[c] > self.corr_threshold)]
        return df.drop(columns=to_drop), to_drop

    def clean_data(self, df, dataset_name=None):
        """Run cleaning pipeline and save to Silver layer."""
        dropped = {}

        df, dropped['invalid'] = self.identify_invalid(df)
        df, dropped['constant'] = self.remove_constant_sensors(df)
        df, dropped['low_variance'] = self.remove_low_variance_sensors(df)
        df, dropped['correlated'] = self.remove_correlated_sensors(df)

        self.sensor_cols = [c for c in df.columns if 'sensor' in c.lower()]

        if self.silver_dir and dataset_name:
            parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
            save_dir = os.path.join(parent_dir, self.silver_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{dataset_name}_cleaned.csv')
            df.to_csv(save_path, index=False)
            print(f'cleaned data saved to {save_path}')

        return df, dropped

    def clip_rul(self, df, rul_col='RUL', clip_value=125):
        """Clip RUL to cap at degradation start.

        Ensures model does not waste capacity learning to distinguish between 'very health' and 'heathy' states."""
        df[f'{rul_col}_clipped'] = df[rul_col].clip(upper=clip_value)
        return df

    def add_rolling_features(self, df, windows=[5, 10, 20], unit_col='engine_id'):
        """Add rolling window statistics for each sensor.

        Rolling Mean: Smoothes out noise, shows trending direction.
        Rolling Std: Captures increasing instsability
        Rolling Min/Max: Tracks extreme values that might indicate stress"""
        print(f"Generating rolling features for windows: {windows}")
        
        for window in windows:
            for sensor in self.sensor_cols:
                # Group by unit to avoid mixing different engines
                grouped = df.groupby(unit_col)[sensor]
                
                # Rolling statistics
                df[f'{sensor}_rolling_mean_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{sensor}_rolling_std_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                df[f'{sensor}_rolling_min_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                df[f'{sensor}_rolling_max_{window}'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                
        print(f"Added {len(windows) * len(self.sensor_cols) * 4} rolling features")
        return df

    def add_lag_features(self, df, lags=[1, 3, 5], unit_col='engine_id'):
        """Add lagged sensor values to relatively capture decline."""
        print(f"Generating lag features for lags: {lags}")
        
        for lag in lags:
            for sensor in self.sensor_cols:
                df[f'{sensor}_lag_{lag}'] = df.groupby(unit_col)[sensor].shift(lag)
        
        # Fill NaN values in lag features with forward fill (for early cycles)
        lag_cols = [c for c in df.columns if '_lag_' in c]
        df[lag_cols] = df.groupby(unit_col)[lag_cols].fillna(method='bfill')
        
        print(f"Added {len(lags) * len(self.sensor_cols)} lag features")
        return df

    def add_trend_features(self, df, window=10, unit_col='engine_id'):
        """Add trend features (rate of change, slope).

        Catches acceleration as degradation often accelerates as failure approaches.
        Detects phase changes.
        Early warning signs.
        """
        print(f"Generating trend features with window={window}")
        
        for sensor in self.sensor_cols:
            # Rate of change (first difference)
            df[f'{sensor}_diff'] = df.groupby(unit_col)[sensor].diff()
            
            # Rolling slope (linear regression coefficient)
            def rolling_slope(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                y = series.values
                # Handle constant series
                if np.std(y) == 0:
                    return 0
                slope = np.polyfit(x, y, 1)[0]
                return slope
            
            df[f'{sensor}_slope_{window}'] = df.groupby(unit_col)[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(rolling_slope, raw=False)
            )
        
        # Fill NaN values
        trend_cols = [c for c in df.columns if '_diff' in c or '_slope_' in c]
        df[trend_cols] = df.groupby(unit_col)[trend_cols].fillna(0)
        
        print(f"Added {len(self.sensor_cols) * 2} trend features")
        return df

    def add_ewma_features(self, df, spans=[5, 10, 20], unit_col='engine_id'):
        """Add Exponentially Weighted Moving Average features.

        Better than rolling mean, as it responds faster to recent changes. It smoothes noise but weights recent spikes more siginificant. E.g.:
        
        Readings: [50, 52, 54, 56, 58, 100]  # Sudden spike!

        Simple MA:  70  # (50+52+54+56+58+100)/6
        EWMA:       82  # Recent spike weighted heavily
        """
        print(f"Generating EWMA features for spans: {spans}")
        
        for span in spans:
            for sensor in self.sensor_cols:
                df[f'{sensor}_ewma_{span}'] = df.groupby(unit_col)[sensor].transform(
                    lambda x: x.ewm(span=span, adjust=False).mean()
                )
        
        print(f"Added {len(spans) * len(self.sensor_cols)} EWMA features")
        return df

    def normalize_features(self, df, fit=True):
        """Normalize sensor features to [0, 1] range."""
        feature_cols = self.sensor_cols + [c for c in df.columns if any(
            x in c for x in ['_rolling_', '_lag_', '_diff', '_slope_', '_ewma_']
        )]
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            print(f"Normalized {len(feature_cols)} features (fitted scaler)")
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
            print(f"Normalized {len(feature_cols)} features (using existing scaler)")
        
        return df

    def engineer_features(self, df, dataset_name=None, unit_col='engine_id', cycle_col='time_cycles',
                         windows=[5, 10, 20], lags=[1, 3, 5], ewma_spans=[5, 10, 20],
                         trend_window=10, rul_clip=125, normalize=True):
        """
        Full feature engineering pipeline for Gold layer.
        
        Args:
            df: Cleaned dataframe from Silver layer
            dataset_name: Name for saving
            unit_col: Column name for engine unit ID
            cycle_col: Column name for cycle/time
            windows: Window sizes for rolling features
            lags: Lag values for lag features
            ewma_spans: Span values for EWMA
            trend_window: Window for trend calculation
            rul_clip: Value to clip RUL
            normalize: Whether to normalize features
        """                
        # Generate features
        df = self.add_rolling_features(df, windows, unit_col)
        df = self.add_lag_features(df, lags, unit_col)
        df = self.add_trend_features(df, trend_window, unit_col)
        df = self.add_ewma_features(df, ewma_spans, unit_col)
        
        # Normalize features
        if normalize:
            df = self.normalize_features(df, fit=True)
        
        # Save to Gold layer
        if self.gold_dir and dataset_name:
            parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
            save_dir = os.path.join(parent_dir, self.gold_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{dataset_name}_featured.csv')
            df.to_csv(save_path, index=False)
            print(f"\nFeatured data saved to {save_path}")
        
        return df
