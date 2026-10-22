import pandas as pd
import numpy as np
import logging
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature Engineering class for Time Series Demand Forecasting.
    Creates lag, rolling, expanding, trend, calendar, and encoding features.
    """
    
    @staticmethod
    def create_time_series_features(
        df: pd.DataFrame, 
        value_col: str = "Sales", 
        group_cols: Optional[List[str]] = None,
        date_col: str = "OrderMonth"
    ) -> pd.DataFrame:
        """
        Engineers time-series features: lags, rolling stats, expanding mean,
        date parts, and trend indicators. Prevents cross-series contamination via grouping.
        """
        df = df.copy()
        
        # Ensure correct date datatype and sort
        df[date_col] = pd.to_datetime(df[date_col])
        if group_cols:
            df = df.sort_values(by=group_cols + [date_col]).reset_index(drop=True)
            grouped = df.groupby(group_cols)
        else:
            df = df.sort_values(by=[date_col]).reset_index(drop=True)
            grouped = df
            
        logger.info(f"Engineering features for value_col='{value_col}'...")

        # 1. Lags
        lags = [1, 2, 3, 6, 12]
        for lag in lags:
            df[f"Lag{lag}"] = grouped[value_col].shift(lag)
            
        # Previous Year Sales is Lag 12
        df["PrevYearSales"] = df["Lag12"]

        # 2. Rolling stats (using shift(1) to avoid data leakage / look-ahead bias)
        windows = [3, 6, 12]
        for w in windows:
            df[f"RollingMean_{w}"] = grouped[value_col].shift(1).rolling(window=w, min_periods=1).mean()
            df[f"RollingMedian_{w}"] = grouped[value_col].shift(1).rolling(window=w, min_periods=1).median()
            df[f"RollingStd_{w}"] = grouped[value_col].shift(1).rolling(window=w, min_periods=1).std()
            
            # Fill NaNs in Std with 0
            df[f"RollingStd_{w}"] = df[f"RollingStd_{w}"].fillna(0)

        # 3. Expanding Mean (shift(1) to avoid look-ahead bias)
        df["ExpandingMean"] = grouped[value_col].shift(1).expanding(min_periods=1).mean()
        
        # 4. Date components
        df["Month"] = df[date_col].dt.month
        df["Quarter"] = df[date_col].dt.quarter
        df["Year"] = df[date_col].dt.year
        
        # 5. Trend Feature: integer step counter
        min_date = df[date_col].min()
        df["TrendFeature"] = (df[date_col].dt.year - min_date.year) * 12 + (df[date_col].dt.month - min_date.month)
        
        # 6. Seasonal Feature: Sine and Cosine of Month for circular mapping
        df["SinMonth"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["CosMonth"] = np.cos(2 * np.pi * df["Month"] / 12)
        
        # Drop rows where lag features are missing if we are training, but we keep them for validation/inference
        logger.info(f"Feature engineering completed. Columns: {df.columns.tolist()}")
        return df

    @staticmethod
    def encode_categorical(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """
        Performs category encoding (factorization / target code mapping).
        """
        df = df.copy()
        for col in cat_cols:
            if col in df.columns:
                df[f"{col}_Encoded"] = pd.factorize(df[col])[0]
        return df
