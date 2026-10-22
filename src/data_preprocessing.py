import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Cleaner and Preprocessor for Amazon Transaction Data.
    Provides cleaning, datetime mapping, feature extraction, and monthly aggregation.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Loads dataset from CSV."""
        logger.info(f"Loading data from {self.file_path}...")
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded data. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise e

    def clean_data(self) -> pd.DataFrame:
        """
        Cleans data: handles missing values, removes duplicates, filters invalid records.
        """
        if self.df is None:
            self.load_data()
            
        # 1. Duplicates
        num_duplicates = self.df.duplicated().sum()
        if num_duplicates > 0:
            logger.info(f"Removing {num_duplicates} duplicate rows...")
            self.df.drop_duplicates(inplace=True)
            
        # 2. Missing values
        # Fill missing text fields with 'Unknown'
        text_cols = ['CustomerName', 'Category', 'Brand', 'ProductName', 'PaymentMethod', 'OrderStatus', 'City', 'State']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')
                
        # 3. Handle anomalies/invalid values
        # Quantity must be positive, UnitPrice must be positive, TotalAmount must be positive
        original_len = len(self.df)
        self.df = self.df[
            (self.df['Quantity'] > 0) & 
            (self.df['UnitPrice'] >= 0) & 
            (self.df['TotalAmount'] >= 0)
        ]
        removed_records = original_len - len(self.df)
        if removed_records > 0:
            logger.info(f"Removed {removed_records} invalid records (Quantity <= 0, UnitPrice < 0, or TotalAmount < 0).")
            
        # 4. Correct Datatypes
        self.df['OrderDate'] = pd.to_datetime(self.df['OrderDate'], errors='coerce')
        # Drop rows where OrderDate is null
        self.df = self.df.dropna(subset=['OrderDate'])
        
        logger.info(f"Data cleaning complete. Re-indexed row count: {len(self.df)}")
        return self.df

    def engineer_date_features(self) -> pd.DataFrame:
        """
        Generates: Year, Month, Month Number, Month Name, Quarter, Week, Day of Week,
        Weekend Flag, and Trend Index columns.
        """
        if self.df is None:
            self.clean_data()
            
        # Ensure OrderDate is datetime
        self.df['OrderDate'] = pd.to_datetime(self.df['OrderDate'])
        
        self.df['Year'] = self.df['OrderDate'].dt.year
        self.df['Month Number'] = self.df['OrderDate'].dt.month
        self.df['Month'] = self.df['OrderDate'].dt.month
        self.df['Month Name'] = self.df['OrderDate'].dt.strftime('%B')
        self.df['Quarter'] = self.df['OrderDate'].dt.quarter
        self.df['Week'] = self.df['OrderDate'].dt.isocalendar().week.astype(int)
        self.df['Day of Week'] = self.df['OrderDate'].dt.dayofweek
        self.df['Weekend Flag'] = (self.df['Day of Week'] >= 5).astype(int)
        
        # Trend Index: months since the minimum date
        min_date = self.df['OrderDate'].min()
        self.df['Trend Index'] = (self.df['Year'] - min_date.year) * 12 + (self.df['Month Number'] - min_date.month)
        
        logger.info("Date feature engineering complete.")
        return self.df

    def get_monthly_aggregates(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Aggregates clean data monthly:
        1. Business Level
        2. Category Level
        3. Top Product Level (Top N selling products)
        
        Ensures a continuous monthly timeline.
        """
        if self.df is None:
            self.engineer_date_features()
            
        # Add OrderMonth representation (e.g. '2020-01')
        self.df['OrderMonth'] = self.df['OrderDate'].dt.to_period('M')
        
        # Find time boundaries
        min_month = self.df['OrderMonth'].min()
        max_month = self.df['OrderMonth'].max()
        all_months = pd.period_range(start=min_month, end=max_month, freq='M')
        
        # --- 1. Business Level ---
        logger.info("Aggregating Business Level Monthly dataset...")
        biz_agg = self.df.groupby('OrderMonth').agg(
            Sales=('Quantity', 'sum'),
            Revenue=('TotalAmount', 'sum'),
            Orders=('OrderID', 'nunique')
        ).reindex(all_months, fill_value=0)
        
        biz_agg.index.name = 'OrderMonth'
        biz_agg = biz_agg.reset_index()
        # Convert Period back to timestamp/date string representation for serialization
        biz_agg['OrderMonth'] = biz_agg['OrderMonth'].dt.to_timestamp()
        
        # --- 2. Category Level ---
        logger.info("Aggregating Category Level Monthly dataset...")
        categories = self.df['Category'].unique()
        cat_index = pd.MultiIndex.from_product([categories, all_months], names=['Category', 'OrderMonth'])
        
        cat_agg = self.df.groupby(['Category', 'OrderMonth']).agg(
            Sales=('Quantity', 'sum'),
            Revenue=('TotalAmount', 'sum'),
            Orders=('OrderID', 'nunique')
        ).reindex(cat_index, fill_value=0).reset_index()
        cat_agg['OrderMonth'] = cat_agg['OrderMonth'].dt.to_timestamp()
        
        # --- 3. Top Product Level ---
        # Find top N products by overall Quantity sold
        top_n = 30
        logger.info(f"Identifying Top {top_n} products by overall sales quantity...")
        top_products = self.df.groupby('ProductID')['Quantity'].sum().nlargest(top_n).index.tolist()
        
        # Extract product details mapping
        product_meta = self.df[self.df['ProductID'].isin(top_products)].groupby('ProductID').agg(
            ProductName=('ProductName', 'first'),
            Category=('Category', 'first')
        ).reset_index()
        
        prod_index = pd.MultiIndex.from_product([top_products, all_months], names=['ProductID', 'OrderMonth'])
        
        prod_agg = self.df[self.df['ProductID'].isin(top_products)].groupby(['ProductID', 'OrderMonth']).agg(
            Sales=('Quantity', 'sum'),
            Revenue=('TotalAmount', 'sum'),
            Orders=('OrderID', 'nunique')
        ).reindex(prod_index, fill_value=0).reset_index()
        
        # Join product metadata
        prod_agg = prod_agg.merge(product_meta, on='ProductID', how='left')
        prod_agg['OrderMonth'] = prod_agg['OrderMonth'].dt.to_timestamp()
        
        logger.info("Monthly aggregations complete.")
        return biz_agg, cat_agg, prod_agg

if __name__ == "__main__":
    import os
    data_path = os.path.join("Data", "Amazon_transaction_data.csv")
    if os.path.exists(data_path):
        preprocessor = DataPreprocessor(data_path)
        biz, cat, prod = preprocessor.get_monthly_aggregates()
        print("Business level aggregated shape:", biz.shape)
        print("Category level aggregated shape:", cat.shape)
        print("Product level aggregated shape:", prod.shape)
    else:
        print("Amazon transaction data CSV not found.")
